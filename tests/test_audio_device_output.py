"""공유 출력 장치(respk) 지정 — 노래 틀기를 켜면 봇 출력이 dmix 로 가야 한다. (2026-09-11)

ReSpeaker 출력(hw)은 독점이다. 유튜브(PulseAudio)가 쥐고 있으면 봇이 hw 로 말하려다
실패한다(젯슨 실측 'Device unavailable'). 그래서 둘 다 dmix 를 거친다.

🔴 이름은 **정확히** 맞춰야 한다. 부분일치로 찾으면 'respk' 가 'respk_dmix' 에 먼저
   걸린다 — 젯슨 장치 목록에서 respk_dmix(30)가 respk(31)보다 앞이다. respk_dmix 는
   16kHz 만 받아서 TTS 레이트와 안 맞는다.
"""
import sys

from app import audio_device

DEVICES = [
    {"name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:0,0)",
     "max_input_channels": 6, "max_output_channels": 2},
    {"name": "respk_dmix", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "respk", "max_input_channels": 0, "max_output_channels": 128},
    {"name": "default", "max_input_channels": 128, "max_output_channels": 128},
]


def _fake_sd(monkeypatch, devices):
    class FakeSD:
        default = type("d", (), {"device": [3, 3]})()

        @staticmethod
        def query_devices():
            return devices

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    return FakeSD


def test_shared_output_is_matched_exactly(monkeypatch):
    sd = _fake_sd(monkeypatch, DEVICES)
    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": "ReSpeaker"}})

    assert audio_device.setup_audio_device(output_device="respk") is True
    assert sd.default.device == (0, 2), "입력은 ReSpeaker, 출력은 respk(2) — respk_dmix(1)가 아니다"


def test_missing_shared_output_keeps_respeaker_and_says_so(monkeypatch):
    """~/.asoundrc 가 없는 기계. False 를 받은 main 은 노래 기능만 끈다."""
    sd = _fake_sd(monkeypatch, [DEVICES[0], DEVICES[3]])
    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": "ReSpeaker"}})

    assert audio_device.setup_audio_device(output_device="respk") is False
    assert sd.default.device == (0, 0), "공유 장치가 없으면 지금처럼 ReSpeaker 로 말해야 한다"


def test_without_output_override_nothing_changes(monkeypatch):
    """노래 기능을 끈 기계는 지금과 완전히 같아야 한다."""
    sd = _fake_sd(monkeypatch, DEVICES)
    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": "ReSpeaker"}})

    assert audio_device.setup_audio_device() is True
    assert sd.default.device == (0, 0)


def test_override_works_without_audio_device_config(monkeypatch):
    sd = _fake_sd(monkeypatch, DEVICES)
    monkeypatch.setattr(audio_device.settings, "models", {"audio": {"device": None}})

    assert audio_device.setup_audio_device(output_device="respk") is True
    assert sd.default.device == (3, 2)
