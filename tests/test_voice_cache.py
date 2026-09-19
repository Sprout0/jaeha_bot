import numpy as np

from app.voice_cache import VoiceCache


def test_목소리나_모델이_바뀌면_키가_바뀐다():
    a = VoiceCache("x", "marin", "m").key("안녕")
    assert a != VoiceCache("x", "cedar", "m").key("안녕")
    assert a != VoiceCache("x", "marin", "n").key("안녕")
    assert a == VoiceCache("y", "marin", "m").key("안녕")


def test_속도가_바뀌면_키가_바뀌고_기본_속도는_옛_키를_그대로_쓴다():
    a = VoiceCache("x", "marin", "m").key("안녕")
    assert a == VoiceCache("x", "marin", "m", speed=1.0).key("안녕")   # 이미 만든 wav 재사용
    assert a != VoiceCache("x", "marin", "m", speed=1.2).key("안녕")


def test_없는_것만_만들고_읽는다(tmp_path):
    made = []

    def synth(p):
        made.append(p)
        return np.full(2400, 0.1, dtype=np.float32)

    c = VoiceCache(tmp_path, "marin", "m")
    assert c.ensure(["안녕", "잘가"], synth) == 2
    assert c.ensure(["안녕", "잘가"], synth) == 0
    assert made == ["안녕", "잘가"]
    got = VoiceCache(tmp_path, "marin", "m").get("안녕")
    assert got.dtype == np.float32 and got.size == 2400


def test_만들다_실패하면_건너뛴다(tmp_path):
    def synth(p):
        if p == "나쁨":
            raise RuntimeError("네트워크")
        return np.zeros(10, dtype=np.float32)

    c = VoiceCache(tmp_path, "marin", "m")
    assert c.ensure(["나쁨", "좋음"], synth) == 1
    assert c.get("나쁨") is None and c.get("좋음") is not None
