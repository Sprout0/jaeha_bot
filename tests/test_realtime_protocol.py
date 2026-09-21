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
    assert s["audio"]["output"]["speed"] == 1.0


def test_말_속도를_세션에_싣는다():
    # API 는 0.25~1.5 만 받는다(09-19 실측: 2.0 은 decimal_above_max_value).
    s = session_update(RealtimeConfig.from_dict({"speed": 1.2}), "지시")["session"]
    assert s["audio"]["output"]["speed"] == 1.2


def test_이력은_최근_턴만_아이_말과_봇_답으로_넣는다():
    items = history_items([(f"아이{i}", f"봇{i}") for i in range(10)], max_turns=2)
    assert [it["item"]["content"][0]["text"] for it in items] == ["아이8", "봇8", "아이9", "봇9"]
    assert [it["item"]["role"] for it in items] == ["user", "assistant", "user", "assistant"]
    assert items[1]["item"]["content"][0]["type"] == "output_text"


def test_오디오_추가는_base64():
    m = append_audio(b"\x01\x02")
    assert m["type"] == "input_audio_buffer.append" and base64.b64decode(m["audio"]) == b"\x01\x02"


def test_놀이_답은_그_답에만_지시를_붙인다():
    assert "instructions" not in respond()["response"]
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


def test_받아쓰기_힌트가_있으면_세션에_싣고_없으면_안_싣는다():
    # 09-19 실기: '티니핑' → '비니닝'. 이름 목록을 받아쓰기 모델에 힌트로 준다.
    s = session_update(RealtimeConfig(transcribe_prompt="티니핑, 뽀로로"), "지시")["session"]
    assert s["audio"]["input"]["transcription"]["prompt"] == "티니핑, 뽀로로"
    s = session_update(RealtimeConfig(), "지시")["session"]
    assert "prompt" not in s["audio"]["input"]["transcription"]


def test_아이_말을_글자_메시지로_넣는다():
    from app.realtime_protocol import user_text
    m = user_text("칼 어딨어?")
    assert m["type"] == "conversation.item.create" and m["item"]["role"] == "user"
    assert m["item"]["content"][0] == {"type": "input_text", "text": "칼 어딨어?"}


def test_명령_도구는_노래가_꺼지면_play_song_을_뺀다():
    from app.realtime_protocol import command_tools
    names = lambda ts: {t["name"] for t in ts}
    assert names(command_tools(True)) == {"go_to_sleep", "play_song", "start_game"}
    assert names(command_tools(False)) == {"go_to_sleep", "start_game"}


def test_도구는_세션에_싣고_요청마다_쓸지_정한다():
    from app.realtime_protocol import command_tools
    s = session_update(RealtimeConfig(), "지시", tools=command_tools(False))["session"]
    assert s["tools"] and s["tool_choice"] == "auto"
    assert respond(tools=True) == {"type": "response.create", "response": {"tool_choice": "auto"}}
    assert respond()["response"]["tool_choice"] == "none"
    assert respond("지시")["response"]["tool_choice"] == "none"
    assert say_exactly("안녕")["response"]["tool_choice"] == "none"


def test_응답에서_도구_호출을_꺼낸다():
    from app.realtime_protocol import function_calls, tool_output
    resp = {"output": [{"type": "function_call", "name": "play_song",
                        "arguments": "{\"title\": \"상어가족\"}", "call_id": "c1"},
                       {"type": "message", "content": []}]}
    assert function_calls(resp) == [("play_song", {"title": "상어가족"}, "c1")]
    assert function_calls({"output": [{"type": "function_call", "name": "x", "arguments": "{깨짐",
                                       "call_id": "c2"}]}) == [("x", {}, "c2")]
    o = tool_output("c1")
    assert o["item"] == {"type": "function_call_output", "call_id": "c1", "output": "ok"}
