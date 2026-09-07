"""출력 장치 지정 — 점검 도구도 봇과 같은 길로 소리를 내야 한다.

🔴 2026-09-07 젯슨에서 드러난 사고. `python -m app.audio_player world_playground` 가
   "결과: OK" 를 찍는데 **소리가 안 났다.**

   원인은 재생 경로가 아니라 **진입점**이었다. NVIDIA 의 `/etc/asound.conf` 는
   ALSA `default` 를 `hw:APE,0`(Tegra 내부 크로스바 — 물리 출력에 안 붙어 있다)로
   보낸다. 그래서 장치를 지정하지 않고 재생하면 소리가 그리로 들어가 사라진다.
   실측: `device=0`(ReSpeaker) 지정 시 card0 이 RUNNING, 미지정이면 내내 closed.

   `app/main.py` 에는 이미 장치를 잡아 주는 함수가 있었다. 그런데 `audio_player` 의
   점검 REPL 은 그걸 안 불렀다 — 그리고 `assets/README.md` 와 `run.sh check` 가
   둘 다 그 점검 도구를 가리킨다. 아무도 안 써서 안 드러났을 뿐이다
   (`run.sh` 가 시스템 python 을 부르던 2026-09-01 사고와 같은 종류다).
"""
import sys

import pytest

from app import audio_player


def test_repl_calls_device_setup(monkeypatch, capsys):
    """점검 도구가 장치를 안 잡으면 젯슨에서 허공에 재생한다."""
    called = []
    monkeypatch.setattr(audio_player, "setup_audio_device",
                        lambda *a, **k: called.append(True))
    monkeypatch.setattr(sys, "argv", ["audio_player"])   # 재생은 하지 않는다

    audio_player._repl()

    assert called, "REPL 이 setup_audio_device() 를 안 불렀다 — 젯슨에서 소리가 사라진다"


def test_no_device_config_changes_nothing(monkeypatch):
    """audio.device 가 null 인 기계(노트북)에서는 기본 장치를 건드리면 안 된다."""
    from app import audio_device

    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": None}})
    audio_device.setup_audio_device()   # 예외 없이 조용히 끝나야 한다


def test_name_substring_resolves_both_indices(monkeypatch):
    """'ReSpeaker' 한 낱말로 입력·출력 둘 다 잡힌다(젯슨 실측: 입력 0 / 출력 0)."""
    from app import audio_device

    fake_devices = [
        {"name": "ReSpeaker 4 Mic Array (UAC1.0)", "max_input_channels": 6, "max_output_channels": 2},
        {"name": "NVIDIA Jetson Orin Nano HDA: HDMI 0", "max_input_channels": 0, "max_output_channels": 8},
        {"name": "default", "max_input_channels": 128, "max_output_channels": 128},
    ]

    class FakeSD:
        default = type("d", (), {"device": [2, 2]})()

        @staticmethod
        def query_devices():
            return fake_devices

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": "ReSpeaker"}})

    audio_device.setup_audio_device()

    assert FakeSD.default.device == (0, 0), "ReSpeaker 를 입출력 모두 0번으로 잡아야 한다"
