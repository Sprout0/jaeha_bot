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


def setup_audio_device(retries: int = 3, wait_s: float = 0.4) -> None:
    """설정된 오디오 장치를 sounddevice 기본값으로 지정한다.

    Jetson 등 헤드리스 환경은 기본 장치가 'default'(pulse) 로 잡혀 HDMI/모니터 쪽으로
    라우팅돼 재생·녹음이 멈추는 일이 있다. configs 에 audio.device(이름 부분일치, 예:
    'ReSpeaker') 를 주면 USB 마이크/스피커를 콕 집어 이 문제를 피한다. null 이면 시스템 기본.
    """
    dev = (settings.models.get("audio") or {}).get("device")
    if not dev:
        return
    import sounddevice as sd
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
        return
    cur_in, cur_out = sd.default.device
    sd.default.device = (din if din is not None else cur_in,
                         dout if dout is not None else cur_out)
    log.info("오디오 장치 지정: %s -> 입력 %s / 출력 %s", dev, din, dout)
    if din is None or dout is None:
        log.warning("'%s' 의 %s 장치를 못 찾아 그쪽은 기본 장치를 사용합니다",
                    dev, "입력" if din is None else "출력")
