"""오디오 에셋 재생: 동요·효과음 음원 파일을 관리하고 튼다.

왜 TTS 가 아니라 음원 파일인가:
- **노래는 TTS 로 못 부른다.** 아무리 좋은 TTS 도 '곰 세 마리'를 읽을 뿐 부르지 않는다.
  음악 생성 API 는 생성에 수십 초 걸리고 매번 멜로디가 달라지며 기존 동요는 저작권이
  있다. 아이는 같은 노래를 스무 번 반복해 듣는다 — 매번 생성할 이유가 없다.
- **효과음도 마찬가지.** "멍멍"을 TTS 로 읽는 것과 진짜 개 짖는 소리는 다르고,
  아이가 반응하는 건 후자다.

역할 분담: 이 파일은 '무엇을 낼 수 있는지'(AudioLibrary)와 '내는 것'(AudioPlayer)만
한다. 언제 틀지는 놀이 상태머신 / 에이전트가 정한다.

🔴 AudioLibrary 는 '현재활동 날조'의 구조적 해결책이다. playable()/titles() 를
LLM 툴 스키마의 enum 으로 그대로 쓰면 **없는 노래는 부를 방법 자체가 없어진다.**

⚠️ 재생 샘플레이트 결정 로직(_resolve_play_rate)이 tts_module.TTSModule 에도 있다.
같은 문제를 두 곳에서 풀고 있으므로 언젠가 하나로 합쳐야 한다. 지금 합치지 않은 건
TTS 가 젯슨 GPU 경로에서 검증된 상태라 마이크·스피커 없는 노트북에서 손대면
확인 없이 깨질 수 있어서다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .text_norm import jamo_ratio

log = logging.getLogger(__name__)

# 자모 편집거리 임계. education_modes.match_trigger 와 **같은 값**을 쓴다 —
# 같은 문제(2세 발음 + STT 오인식)를 다른 기준으로 풀면 놀이마다 인식률이 달라진다.
MATCH_THRESHOLD = 0.2

KINDS = ("song", "sound")

# 노래는 2~3분이라 블로킹하면 그동안 아이 말을 못 듣는다("그만"도 못 듣는다).
# 효과음은 1~2초라 끝까지 기다리는 편이 낫다(뒤이을 봇 대사와 겹치지 않음).
_BLOCKS_BY_KIND = {"sound": True, "song": False}


def _norm(s: str) -> str:
    """공백·문장부호 제거(한글/영숫자만). education_modes._norm 과 같은 전처리."""
    return re.sub(r"\W+", "", s or "")


@dataclass(frozen=True)
class AudioAsset:
    id: str
    kind: str          # "song" | "sound"
    title: str
    path: Path
    aliases: tuple[str, ...] = ()
    # 라이선스 판단 근거. 시연·논문에 들어가므로 빈 값이면 테스트가 막는다.
    license: str = ""

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def names(self) -> tuple[str, ...]:
        """제목 + 별칭. 아이는 정식 제목을 모른다(곰 세 마리 → '곰돌이')."""
        return (self.title, *self.aliases)


class AudioLibrary:
    """레지스트리. 장치도 파일도 없어도 동작한다(기동 점검·테스트용)."""

    def __init__(self, assets: list[AudioAsset]) -> None:
        self._assets = list(assets)

    @classmethod
    def load(cls, config_path, assets_dir) -> "AudioLibrary":
        config_path, assets_dir = Path(config_path), Path(assets_dir)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        assets: list[AudioAsset] = []
        for kind in KINDS:
            for entry in raw.get(f"{kind}s") or []:
                assets.append(AudioAsset(
                    id=entry["id"],
                    kind=kind,
                    title=entry["title"],
                    path=assets_dir / entry["file"],
                    aliases=tuple(entry.get("aliases") or ()),
                    license=entry.get("license", ""),
                ))
        return cls(assets)

    # ── 조회 ─────────────────────────────────────────────────────────────────
    def _of_kind(self, kind: str | None) -> list[AudioAsset]:
        return [a for a in self._assets if kind is None or a.kind == kind]

    @property
    def songs(self) -> list[AudioAsset]:
        return self._of_kind("song")

    @property
    def sounds(self) -> list[AudioAsset]:
        return self._of_kind("sound")

    def get(self, asset_id: str) -> AudioAsset | None:
        return next((a for a in self._assets if a.id == asset_id), None)

    def find(self, text: str, kind: str | None = None) -> AudioAsset | None:
        """아이 발화에서 요청한 음원을 찾는다. 없으면 None.

        None 을 돌려주는 게 중요하다 — 비슷한 걸 아무거나 틀면 '상어가족'을 달라 한
        아이에게 '곰 세 마리'가 나간다. 유아 대상에선 엉뚱한 재생이 곧 사고다.
        """
        norm = _norm(text)
        if not norm:
            return None
        candidates = self._of_kind(kind)
        # 1단: 포함(정확). 모든 후보를 먼저 훑는다 — 다른 항목의 흐릿한 일치가
        # 정확한 일치를 이기면 안 되기 때문.
        for asset in candidates:
            if any(_norm(n) in norm for n in asset.names):
                return asset
        # 2단: 자모 근접(2세 발음 + STT 오인식 보정)
        for asset in candidates:
            if any(jamo_ratio(norm, _norm(n)) <= MATCH_THRESHOLD for n in asset.names):
                return asset
        return None

    # ── 기동 점검 ────────────────────────────────────────────────────────────
    def missing(self) -> list[AudioAsset]:
        """레지스트리에만 있고 파일이 없는 항목.

        기동 때 경고해야 한다. 안 그러면 봇이 "고양이 소리 들려줄게!"라고 말한 뒤
        아무 소리도 안 난다 — 아이에게는 약속을 어긴 것이다.
        """
        return [a for a in self._assets if not a.exists]

    def playable(self, kind: str | None = None) -> list[AudioAsset]:
        """파일이 실재하는 것만. LLM 에 넘기는 목록은 반드시 이쪽이어야 한다."""
        return [a for a in self._of_kind(kind) if a.exists]

    def titles(self, kind: str | None = None) -> list[str]:
        return [a.title for a in self.playable(kind)]


def default_library() -> AudioLibrary:
    """이 프로젝트의 레지스트리(configs/audio_assets.yaml + assets/)를 읽는다."""
    from .config import BASE_DIR, CONFIG_DIR
    return AudioLibrary.load(CONFIG_DIR / "audio_assets.yaml", BASE_DIR / "assets")


class SoundDeviceSink:
    """실제 스피커 출력. sounddevice 는 여기서만 import 한다 —
    장치 없는 노트북/CI 에서도 AudioLibrary 는 쓸 수 있어야 하므로."""

    def __init__(self) -> None:
        self._rate = None      # 장치가 받는 레이트(첫 재생 때 1회 결정·캐시)
        self._playing = False

    def _resolve_rate(self, sd, want: int) -> int:
        """장치가 원본 레이트를 지원하면 그대로, 아니면 장치 기본 레이트.

        젯슨 USB ReSpeaker 는 16000 전용이라 44100 mp3 를 그대로 못 준다.
        미리 확인하므로 paInvalidSampleRate 실패 로그가 뜨지 않는다.
        """
        if self._rate is not None:
            return self._rate
        rate = want
        try:
            sd.check_output_settings(samplerate=want)
        except Exception:
            try:
                dev = sd.default.device
                outdev = dev[1] if isinstance(dev, (list, tuple)) else dev
                rate = int(sd.query_devices(outdev, "output")["default_samplerate"]) or want
            except Exception:
                pass
        self._rate = rate
        return rate

    def play(self, samples: np.ndarray, rate: int, block: bool) -> None:
        import sounddevice as sd

        target = self._resolve_rate(sd, rate)
        if target != rate:
            samples = _resample(samples, rate, target)
        sd.play(samples, target)
        self._playing = True
        if block:
            sd.wait()
            self._playing = False

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        if not self._playing:
            return False
        try:
            import sounddevice as sd
            stream = sd.get_stream()
            self._playing = bool(stream and stream.active)
        except Exception:
            self._playing = False
        return self._playing


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """soxr(고품질) 우선, 없으면 선형보간 폴백. tts_module._resample 과 같은 방식."""
    if src == dst or audio.size == 0:
        return audio
    try:
        import soxr
        return soxr.resample(audio, src, dst).astype(np.float32)
    except Exception:
        n = int(round(audio.shape[0] * dst / src))
        xp = np.linspace(0.0, 1.0, audio.shape[0], dtype=np.float32)
        xn = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return np.interp(xn, xp, audio).astype(np.float32)


class AudioPlayer:
    """음원 재생. 실패해도 예외를 던지지 않는다 — 봇을 죽이면 안 되기 때문."""

    def __init__(self, library: AudioLibrary, sink=None) -> None:
        self._lib = library
        self._sink = SoundDeviceSink() if sink is None else sink

    def play(self, asset_or_id, block: bool | None = None) -> bool:
        """재생 성공 여부를 bool 로 돌려준다.

        🔴 없는 파일·없는 id·디코딩 실패 어느 쪽이든 False 를 돌려줄 뿐 예외는 안 낸다.
        호출어 감지가 실패해도 봇이 안 죽는 것과 같은 원칙 — 놀이가 멈추면 안 된다.
        호출부는 False 를 받으면 "그 노래는 없어, 대신 ~ 어때?"로 이어가면 된다.
        """
        asset = asset_or_id if isinstance(asset_or_id, AudioAsset) else self._lib.get(asset_or_id)
        if asset is None:
            log.warning("모르는 음원 id: %r", asset_or_id)
            return False
        if not asset.exists:
            log.warning("음원 파일 없음: %s (%s)", asset.path, asset.id)
            return False
        try:
            import soundfile as sf
            samples, rate = sf.read(asset.path, dtype="float32", always_2d=False)
        except Exception as e:
            log.warning("음원 디코딩 실패 %s: %s", asset.path, e)
            return False
        if samples.ndim == 2:
            # 스피커가 하나(ReSpeaker)라 스테레오는 의미가 없다. 합쳐야 한쪽 채널만
            # 나가는 사고를 막는다.
            samples = samples.mean(axis=1)
        if self.is_playing:
            self.stop()   # 노래 도중 다른 요청 — 겹쳐 나오면 안 된다
        if block is None:
            block = _BLOCKS_BY_KIND[asset.kind]
        self._sink.play(samples, rate, block)
        return True

    def stop(self) -> None:
        self._sink.stop()

    @property
    def is_playing(self) -> bool:
        return self._sink.is_playing


def _repl() -> None:
    """에셋 점검(스피커 불필요): `python -m app.audio_player`

    인자를 주면 그 id 를 실제로 틀어 본다: `python -m app.audio_player dog`
    새 로직 없음 — missing()/playable()/play() 를 부르기만 한다(전부 테스트됨).
    """
    import sys

    lib = default_library()
    for kind in KINDS:
        ok = lib.playable(kind)
        print(f"\n[{kind}] 재생 가능 {len(ok)}개 / 등록 {len(lib._of_kind(kind))}개")
        for a in ok:
            print(f"  ✅ {a.id:16} {a.title}")
    gone = lib.missing()
    if gone:
        print(f"\n⚠️ 파일 없음 {len(gone)}개 — 이 항목은 봇이 '못 하는 것'으로 취급된다:")
        for a in gone:
            print(f"  ❌ {a.id:16} {a.title:12} -> {a.path}")
        print("  넣는 법: assets/README.md")

    if len(sys.argv) > 1:
        target = sys.argv[1]
        print(f"\n재생 시도: {target}")
        print("결과:", "OK" if AudioPlayer(lib).play(target) else "실패(위 경고 참조)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _repl()
