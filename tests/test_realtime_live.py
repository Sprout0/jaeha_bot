"""라이브 루프가 **자기 목소리를 되듣지 않는지**, **소리를 깨뜨리지 않는지**.

이 도구가 재려는 건 지연이 아니다(그건 09-02 에 이미 쟀다). 라이브로 돌렸을 때만
드러나는 셋 — 에코 / 바지인 / 리샘플 — 이고, 그중 둘은 순수 로직에서 깨진다:

- 봇이 말하는 동안 마이크를 열어 두면 서버 VAD 가 **봇의 말에 반응**한다.
  봇이 자기한테 대답하기 시작하면 그 자리에서 끝난다.
- 웹소켓 오디오 델타는 pcm16 **샘플 중간에서 잘려** 온다(바이트 수가 홀수).
  나머지 바이트를 버리면 그 지점부터 좌우가 밀려 잡음이 된다.
"""
import numpy as np

from tools.realtime_live import (FileSink, MicGate, PcmAccumulator, blind_labels)


class TestMicGate:
    """반이중(half-duplex). 봇이 말하는 동안은 안 보낸다."""

    def test_재생_중에는_안_보낸다(self):
        # 🔴 급소. 여기가 열려 있으면 봇이 제 말을 듣고 자기한테 대답한다.
        g = MicGate(pad_s=0.15)
        g.playing_started(at=0.0)
        assert g.should_send(now=0.5) is False

    def test_재생이_끝나도_pad_동안은_안_보낸다(self):
        # 스피커 잔향과 장치 버퍼가 남아 있다. 08-24 에 정한 값이 0.150s.
        g = MicGate(pad_s=0.15)
        g.playing_started(at=0.0)
        g.playing_ended(at=1.0)
        assert g.should_send(now=1.1) is False

    def test_pad_가_지나면_보낸다(self):
        g = MicGate(pad_s=0.15)
        g.playing_started(at=0.0)
        g.playing_ended(at=1.0)
        assert g.should_send(now=1.2) is True

    def test_아무것도_재생한_적_없으면_보낸다(self):
        assert MicGate(pad_s=0.15).should_send(now=0.0) is True


class TestPcmAccumulator:
    """pcm16 바이트를 float32 로. 델타 경계에서 깨지면 안 된다."""

    def test_홀수_바이트가_와도_남겨서_이어_붙인다(self):
        # 🔴 급소. 남은 1 바이트를 버리면 그 뒤 전부가 한 바이트씩 밀려 잡음이 된다.
        a = PcmAccumulator()
        first = a.feed(b"\x00\x40\x00")        # 샘플 하나 + 반쪽
        assert first.size == 1
        second = a.feed(b"\x80")               # 반쪽의 나머지
        assert second.size == 1

    def test_pcm16_을_정규화한다(self):
        a = PcmAccumulator()
        out = a.feed(np.array([0, 16384, -16384], dtype="<i2").tobytes())
        assert np.allclose(out, [0.0, 0.5, -0.5], atol=1e-4)

    def test_빈_입력은_빈_출력(self):
        assert PcmAccumulator().feed(b"").size == 0

    def test_반쪽만_오면_아직_아무것도_안_내놓는다(self):
        assert PcmAccumulator().feed(b"\x00").size == 0


class TestBlindLabels:
    """1단계 목소리 고르기. 08-25 에 목소리 F2 를 블라인드로 골랐다 — 같은 규율."""

    def test_파일명에_목소리_이름이_안_들어간다(self):
        # 🔴 급소. marin.wav 라고 적히면 그 순간 블라인드가 아니다.
        labels = blind_labels(["marin", "cedar", "alloy"], seed=1)
        for voice, name in labels.items():
            assert voice not in name, f"{name} 에 목소리 이름이 보인다"

    def test_목소리마다_다른_이름을_준다(self):
        labels = blind_labels(["marin", "cedar", "alloy"], seed=1)
        assert len(set(labels.values())) == 3

    def test_씨앗이_같으면_같은_배정(self):
        # 나중에 정답표를 잃어버려도 씨앗만 알면 되살린다.
        assert blind_labels(["a", "b", "c"], seed=7) == blind_labels(["a", "b", "c"], seed=7)

    def test_씨앗이_다르면_배정이_바뀐다(self):
        many = [f"v{i}" for i in range(8)]
        assert blind_labels(many, seed=1) != blind_labels(many, seed=2)


class TestMicGateSync:
    """스피커 상태를 매 프레임 물어보는 쪽에서 붙는다 — 가장자리를 놓치면 안 연다."""

    def test_끝난_상태를_계속_알려도_pad_는_한_번만_센다(self):
        # 🔴 급소. 매번 '끝났다'로 시계를 새로 잡으면 pad 가 영원히 안 지나
        #    마이크가 영영 안 열린다 — 봇이 벙어리가 된 것처럼 보인다.
        g = MicGate(pad_s=0.15)
        g.sync(playing=True, now=0.0)
        for t in (1.0, 1.05, 1.1, 1.15, 1.2):
            g.sync(playing=False, now=t)
        assert g.should_send(now=1.2) is True

    def test_다시_말하기_시작하면_또_닫힌다(self):
        g = MicGate(pad_s=0.15)
        g.sync(playing=True, now=0.0)
        g.sync(playing=False, now=1.0)
        assert g.should_send(now=1.3) is True
        g.sync(playing=True, now=1.4)
        assert g.should_send(now=1.4) is False


class TestFileSink:
    """스피커가 없을 때. 봇 목소리를 재생 대신 파일로 받아 마이크 쪽만 시험한다."""

    def test_여러_번_밀어_넣으면_이어_붙는다(self, tmp_path):
        s = FileSink(tmp_path / "out.wav")
        s.push(np.array([0.1, 0.2], dtype=np.float32))
        s.push(np.array([0.3], dtype=np.float32))
        s.close()
        import soundfile as sf
        assert sf.read(tmp_path / "out.wav")[0].size == 3

    def test_재생이_없으니_바쁘지_않다(self, tmp_path):
        # 🔴 여기서 True 를 주면 마이크가 영영 안 열린다 — 스피커가 없으니
        #    막을 이유도 없다(에코가 물리적으로 불가능하다).
        s = FileSink(tmp_path / "out.wav")
        s.push(np.ones(1000, dtype=np.float32))
        assert s.busy is False

    def test_받은_게_없으면_빈_파일을_안_만든다(self, tmp_path):
        FileSink(tmp_path / "out.wav").close()
        assert not (tmp_path / "out.wav").exists()
