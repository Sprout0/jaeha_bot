import base64

from app.realtime_protocol import (RealtimeConfig, append_audio, cached_tokens, cancel,
                                   cost_usd, event_kind, history_items, respond,
                                   say_exactly, session_update)


def test_설정은_빠진_값을_기본으로_채우고_모르는_키는_버린다():
    c = RealtimeConfig.from_dict({"voice": "cedar", "없는키": 1})
    assert c.voice == "cedar" and c.model == "gpt-realtime-mini"
    assert c.silence_ms == 1200 and c.response_timeout_s == 8.0
    assert RealtimeConfig.from_dict(None).voice == "marin"


def test_세션은_자동으로_답하지_않는다():
    # 🔴 서버가 먼저 답하면 "노래 틀어줘" 를 우리가 보기 전에 "틀어줄게" 가 나간다.
    s = session_update(RealtimeConfig(), "지시")["session"]
    td = s["audio"]["input"]["turn_detection"]
    assert td["create_response"] is False and td["interrupt_response"] is False
    assert td["silence_duration_ms"] == 1200
    assert s["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert s["audio"]["output"]["voice"] == "marin" and s["instructions"] == "지시"


def test_이력은_최근_턴만_아이_말과_봇_답으로_넣는다():
    items = history_items([(f"아이{i}", f"봇{i}") for i in range(10)], max_turns=2)
    assert [it["item"]["content"][0]["text"] for it in items] == ["아이8", "봇8", "아이9", "봇9"]
    assert [it["item"]["role"] for it in items] == ["user", "assistant", "user", "assistant"]
    assert items[1]["item"]["content"][0]["type"] == "output_text"


def test_오디오_추가는_base64():
    m = append_audio(b"\x01\x02")
    assert m["type"] == "input_audio_buffer.append" and base64.b64decode(m["audio"]) == b"\x01\x02"


def test_놀이_답은_그_답에만_지시를_붙인다():
    assert respond() == {"type": "response.create"}
    assert respond("상황")["response"]["instructions"] == "상황"


def test_그대로_말하기는_대화_기록_밖이다():
    r = say_exactly("상어가족 틀어 줄게!")["response"]
    assert r["conversation"] == "none"
    assert "상어가족 틀어 줄게!" in r["input"][0]["content"][0]["text"]


def test_취소():
    assert cancel() == {"type": "response.cancel"}


def test_이벤트_분류는_베타_이름도_받는다():
    assert event_kind({"type": "response.output_audio.delta"}) == "audio"
    assert event_kind({"type": "response.audio.delta"}) == "audio"
    assert event_kind({"type": "response.output_audio_transcript.delta"}) == "text"
    assert event_kind({"type": "conversation.item.input_audio_transcription.completed"}) == "transcript"
    assert event_kind({"type": "input_audio_buffer.speech_stopped"}) == "speech_stopped"
    assert event_kind({"type": "response.done"}) == "done"
    assert event_kind({"type": "error"}) == "error"
    assert event_kind({"type": "session.updated"}) == "other"


def test_비용과_캐시_토큰():
    usage = {"input_token_details": {"audio_tokens": 100, "text_tokens": 0, "cached_tokens": 80,
                                     "cached_tokens_details": {"audio_tokens": 80}},
             "output_token_details": {"audio_tokens": 50, "text_tokens": 0}}
    assert abs(cost_usd("gpt-realtime-mini", usage) - (20 * 10.0 + 80 * 0.30 + 50 * 20.0) / 1e6) < 1e-12
    assert cached_tokens(usage) == 80
    assert cost_usd("모르는모델", usage) is None
