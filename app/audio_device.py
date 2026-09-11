"""오디오 장치 지정 — 봇과 점검 도구가 **같은 길**로 소리를 내게 한다.

🔴 왜 main.py 에서 여기로 옮겼는가 (2026-09-07).

젯슨에서 `python -m app.audio_player world_playground` 가 "결과: OK" 를 찍는데
소리가 안 났다. 재생 경로는 멀쩡했다 — 문제는 **진입점마다 장치 지정이 달랐던 것**이다.

NVIDIA 의 `/etc/asound.conf` 는 ALSA `default` 를 `hw:APE,0`(Tegra 내부 크로스바)로
보낸다. 물리 출력에 안 붙어 있어서, 장치를 안 정하고 재생하면 소리가 거기서 사라진다.
실측(젯슨, `/proc/asound/card0/pcm0p/sub0/status`):
  - `device=0`(ReSpeaker) 지정   -> card0 이 2초 내내 RUNNING
  - 장치 미지정                  -> card0 은 내내 closed

이 함수는 원래 `main.py` 안에 있었고, 그래서 **봇 본체만** 장치를 잡았다. 점검 REPL 은
안 불렀는데 `assets/README.md` 와 `run.sh check` 가 둘 다 그 REPL 을 가리킨다.
2026-09-01 의 `run.sh` 사고(시스템 python 호출)와 같은 종류다 — 아무도 안 써서
안 드러났을 뿐, 시연에서 그걸 누르면 그 자리서 실패한다.

여기는 무거운 것을 import 하지 않는다. 점검 도구가 모델 없이 가볍게 떠야 하기 때문이다.
"""
from __future__ import annotations

import logging
import time

from .config import settings

log = logging.getLogger("jaeha_bot.audio")


def setup_audio_device(retries: int = 3, wait_s: float = 0.4,
                       output_device: str | None = None) -> bool:
    """설정된 오디오 장치를 sounddevice 기본값으로 지정한다.

    Jetson 등 헤드리스 환경은 기본 장치가 'default'(pulse) 로 잡혀 HDMI/모니터 쪽으로
    라우팅돼 재생·녹음이 멈추는 일이 있다. configs 에 audio.device(이름 부분일치, 예:
    'ReSpeaker') 를 주면 USB 마이크/스피커를 콕 집어 이 문제를 피한다. null 이면 시스템 기본.

    output_device: 출력만 **이름이 정확히 같은** 장치로 바꾼다(유튜브 공유 출력 'respk').
      부분일치로 찾으면 'respk' 가 'respk_dmix'(16kHz 전용)에 먼저 걸린다 — 젯슨 장치
      목록에서 respk_dmix 가 respk 보다 앞 번호다(2026-09-11 실측 30 vs 31).
      못 찾으면 False 를 돌려준다. 호출부는 그걸 보고 노래 기능을 끈다 — 공유 출력 없이
      유튜브를 켜면 음악이 스피커를 쥐는 동안 **봇이 말을 못 한다**.
    """
    dev = (settings.models.get("audio") or {}).get("device")
    if not dev and not output_device:
        return True
    import sounddevice as sd
    if not dev:
        return _set_exact_output(sd, output_device)
    # 지정한 장치(예: 젯슨의 'ReSpeaker')가 이 기계엔 없을 수 있다(노트북엔 내장 마이크뿐,
    # 젯슨도 USB 를 안 꽂으면 없음). 주의: sd.default.device 에 '이름 문자열'을 대입하면
    # sounddevice 는 그 자리서 검증하지 않고 재생/녹음 시점에 해석한다. 그래서 장치가 없어도
    # 대입은 조용히 통과하고 나중에 sd.play() 에서 ValueError 로 죽는다(실기 확인 2026-07-24).
    # → 이름을 직접 조회해 입출력 각각 '인덱스'로 지정하고, 없으면 경고 후 기본 장치를 쓴다.
    def _find(devices, key: str):
        for i, d in enumerate(devices):
            if dev.lower() in d["name"].lower() and d[key] > 0:
                return i
        return None

    # 🔴 한 번 보고 포기하지 않는다 (2026-08-27). 같은 명령을 두 번 돌렸는데 1회차는
    #    "'ReSpeaker' 의 입력 장치를 못 찾아", 2회차는 "입력 0 / 출력 0" 이었다 —
    #    직전 프로세스가 장치를 놓기 전이라 입력 채널이 0 으로 보이던 찰나다.
    #    거기서 포기하면 죽은 default 로 폴백해 **그 세션 전체가 귀머거리**가 된다.
    din = dout = None
    for attempt in range(max(1, retries)):
        devices = sd.query_devices()
        din, dout = _find(devices, "max_input_channels"), _find(devices, "max_output_channels")
        if din is not None and dout is not None:
            break
        if attempt + 1 < max(1, retries):
            log.info("오디오 장치 '%s' 를 아직 못 찾았다(입력 %s/출력 %s) — %.1f초 뒤 다시 본다",
                     dev, din, dout, wait_s)
            time.sleep(wait_s)
    if din is None and dout is None:
        log.warning("오디오 장치 '%s' 를 찾을 수 없음 -> 시스템 기본 장치 사용. "
                    "(USB 마이크가 연결됐는지 확인하세요)", dev)
        return _set_exact_output(sd, output_device) if output_device else True
    cur_in, cur_out = sd.default.device
    sd.default.device = (din if din is not None else cur_in,
                         dout if dout is not None else cur_out)
    log.info("오디오 장치 지정: %s -> 입력 %s / 출력 %s", dev, din, dout)
    if din is None or dout is None:
        log.warning("'%s' 의 %s 장치를 못 찾아 그쪽은 기본 장치를 사용합니다",
                    dev, "입력" if din is None else "출력")
    if output_device:
        return _set_exact_output(sd, output_device)
    return True


def _set_exact_output(sd, name: str) -> bool:
    """출력만 이름이 정확히 같은 장치로. 없으면 건드리지 않고 False."""
    idx = next((i for i, d in enumerate(sd.query_devices())
                if d["name"] == name and d["max_output_channels"] > 0), None)
    if idx is None:
        log.warning("공유 출력 장치 '%s' 가 없다 — ~/.asoundrc 확인(./run.sh setup-youtube)", name)
        return False
    cur_in, _ = sd.default.device
    sd.default.device = (cur_in, idx)
    log.info("출력 장치를 공유 장치로: %s -> %s", name, idx)
    return True
