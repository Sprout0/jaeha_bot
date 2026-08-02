"""재하봇 1 엔트리포인트. 파이프라인: 마이크 -> STT -> Agent -> TTS.

독립 실행(가이드 5): python -m app.main  또는  run.sh
STEP 6: 말하면 -> 듣고 -> 답을 소리로. (비전=vision 은 STEP 9 라 아직 미연결)
"""
from __future__ import annotations
import logging
import sys
import time

from .config import settings
from .stt_module import STTModule
from .tts_module import TTSModule
from .vision_module import VisionDetector
from .agent import LLMAgent
from .education_modes import GameManager
from .metrics import MetricsLogger
from .wake import is_wake_word, is_sleep_command, best_wake_ratio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jaeha_bot")

SAFE_RECOVERY = "음, 다시 한 번 말해줄래?"
GREETING = "안녕. 나는 재하봇이야! 나랑 놀자!"
GOODBYE = "안녕! 또 보자~"
# 트리거(호출어) 모드 문구.
READY_ASLEEP = "재하봇 준비됐어. 부르면 나올게!"  # 시작 시 대기 모드 안내
WAKE_GREETING = "응! 왜 불렀어? 나랑 놀자!"      # 호출어에 깨어날 때
SLEEP_MSG = "그래, 또 부르면 올게! 안녕~"          # 다시 대기(잠듦)로 갈 때

# TTS 재생 직후 스피커 잔향(에코)이 마이크에 남아 로봇이 자기 말을 인식하는 것을
# 막기 위한 쿨다운(초). 방식=echo suppression(half-duplex): 말할 땐 안 듣는다.
# 권장 구간 200~500ms 의 중앙값인 0.3s 를 사용(자료 조사 기반). barge-in(끼어들기)이
# 정말 필요해지면 그때 AEC 로 확장. [[jaeha-bot-progress]]
ECHO_COOLDOWN = 0.3


def _setup_audio_device() -> None:
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
    devices = sd.query_devices()

    def _find(key: str):
        for i, d in enumerate(devices):
            if dev.lower() in d["name"].lower() and d[key] > 0:
                return i
        return None

    din, dout = _find("max_input_channels"), _find("max_output_channels")
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


def build_pipeline():
    stt = STTModule(**settings.models.get("stt", {}))
    tts = TTSModule(**settings.models.get("tts", {}))
    vision = VisionDetector(**settings.models.get("vision", {}))
    llm_cfg = dict(settings.models["llm"])
    model_path = llm_cfg.pop("model_path")
    agent = LLMAgent(
        model_path=model_path,
        system_prompt=settings.prompts["system"],
        **llm_cfg,
    )
    return stt, tts, vision, agent


def main() -> None:
    log.info("재하봇 1 시작 (모듈 순차 로딩으로 OOM 방지)")
    _setup_audio_device()
    stt, tts, vision, agent = build_pipeline()
    # 놀이 모드: 흐름은 코드(상태머신), 칭찬·질문 문구만 LLM(agent.render)이 렌더.
    games = GameManager(render=agent.render)
    # STEP 7 계측: 턴마다 단계별 지연·메모리를 파일에 기록(젯슨 이식 전 PC baseline).
    mcfg = settings.models.get("metrics", {})
    metrics = MetricsLogger(enabled=mcfg.get("enabled", True), tag=mcfg.get("tag", "pc"))

    # 첫 대화 지연을 없애기 위해 무거운 모델을 미리 로드(순차 로딩).
    log.info("모델 미리 로딩 중... (STT -> TTS -> LLM)")
    stt.load()
    tts.load()
    agent._ensure_loaded()
    metrics.after_load()  # 모델 3종 로드 후 메모리 baseline
    log.info("준비 완료. 말을 걸어보세요! (종료: Ctrl+C)")

    # 트리거(호출어) 설정: wake.enabled 면 '대기↔대화' 세션. 아니면 항상 대화(옛 동작).
    wcfg = settings.models.get("wake", {}) or {}
    wake_enabled = wcfg.get("enabled", True)
    wake_word = wcfg.get("word", "재하봇")
    wake_threshold = float(wcfg.get("threshold", 0.6))
    wake_aliases = wcfg.get("aliases", [])
    sleep_timeout = float(wcfg.get("sleep_timeout", 30))
    sleep_words = wcfg.get("sleep_words")

    # 시작: 트리거 모드면 '대기'(부를 때까지 조용), 아니면 바로 인사하고 '대화'.
    if wake_enabled:
        log.info("트리거 모드: '%s' 라고 부르면 깨어납니다 (임계값 %.2f)", wake_word, wake_threshold)
        tts.speak(READY_ASLEEP)
        awake = False
    else:
        tts.speak(GREETING)
        awake = True
    time.sleep(ECHO_COOLDOWN)
    last_active = time.time()

    try:
        while True:
            # 듣기(항상): listen 전체시간 - 인식연산(tr_dt) = 녹음대기.
            t_listen = time.perf_counter()
            text, tr_dt = stt.listen()
            stt_wait = max(0.0, (time.perf_counter() - t_listen) - tr_dt)

            # ── 대기 모드: 호출어만 기다린다(LLM·TTS·놀이 안 함 = 자원 절약). ──
            # STT 는 사람이 말할 때만 도니(VAD), 조용하면 비용 없음. 비싼 LLM·TTS 를 막는 게 핵심.
            if not awake:
                if text and is_wake_word(text, wake_word, wake_threshold, wake_aliases):
                    log.info("[호출] %s → 깨어남", text)
                    awake = True
                    last_active = time.time()
                    tts.speak(WAKE_GREETING)
                    time.sleep(ECHO_COOLDOWN)
                elif text:
                    # 안 깨움: 뭐라고 들렸고 얼마나 가까웠는지 남긴다(임계값·별칭 튜닝용).
                    # 여기 자주 뜨는 문자열을 config wake.aliases 에 넣으면 그 발음도 깨운다.
                    log.info("[대기] 안 깨움: %r (거리 %.2f / 임계 %.2f)",
                             text, best_wake_ratio(text, wake_word), wake_threshold)
                # 호출어가 아니면 조용히 무시하고 계속 듣는다.
                continue

            # ── 대화 모드 ──
            # 무음이 이어지면 sleep_timeout 초 뒤 다시 대기로 잠든다.
            if not text:
                if wake_enabled and (time.time() - last_active) >= sleep_timeout:
                    log.info("무응답 %.0f초 → 대기 모드로", sleep_timeout)
                    awake = False
                    tts.speak(SLEEP_MSG)
                    time.sleep(ECHO_COOLDOWN)
                continue
            log.info("[아이] %s", text)

            # '잘자/바이바이' 등은 대화를 끝내고 대기로(놀이의 '그만'과 겹치지 않는 별도 단어).
            if wake_enabled and is_sleep_command(text, sleep_words):
                log.info("잠들기 명령 → 대기 모드로")
                awake = False
                tts.speak(SLEEP_MSG)
                time.sleep(ECHO_COOLDOWN)
                continue

            # 생각: (a)놀이 진행중이면 상태머신 처리 (b)아니면 놀이 시작 트리거 (c)둘 다 아니면 LLM.
            t_think = time.perf_counter()
            reply, kind = games.handle(text), "game"
            if reply is None:
                reply, kind = games.maybe_start(text), "game_start"
            if reply is None:
                reply, kind = agent.respond(text)["text"], "llm"
            if not reply:
                reply, kind = SAFE_RECOVERY, "recovery"
            think_s = time.perf_counter() - t_think
            log.info("[재하봇] %s", reply)

            # 말하기 + 에코 쿨다운(재생 여운이 가라앉은 뒤 다시 듣기).
            t_tts = time.perf_counter()
            tts.speak(reply)
            tts_s = time.perf_counter() - t_tts
            last_active = time.time()  # 마지막 상호작용 시각(잠들기 타이머 기준)
            metrics.record_turn(stt_wait_s=stt_wait, stt_rec_s=tr_dt,
                                think_s=think_s, think_kind=kind,
                                tts_s=tts_s, reply=reply)
            time.sleep(ECHO_COOLDOWN)
    except KeyboardInterrupt:
        log.info("종료 신호(Ctrl+C) 수신")
    finally:
        metrics.summary()  # 세션 요약(중앙값/p90/최대메모리) 출력·기록

    # 마무리 인사 후 깔끔하게 종료.
    try:
        tts.speak(GOODBYE)
    except Exception:
        pass
    log.info("재하봇 1 종료")


def text_repl() -> None:
    """마이크·TTS 없이 '타이핑으로' 전체 대화+놀이 흐름을 테스트한다.

    main() 과 '같은 라우팅'(놀이 handle -> 시작 트리거 -> 평소 대화)을 쓰되 입력만 키보드.
    놀이는 여기서도 정상 진행된다(app.agent 는 LLM 뿐이라 놀이가 안 굴러감 — 그때 이걸 쓸 것).
    실행: python -m app.main --text     (종료: 빈 줄 / exit / Ctrl+C)
    """
    _stt, _tts, _vision, agent = build_pipeline()
    games = GameManager(render=agent.render)
    log.info("LLM 로딩 중...")
    agent._ensure_loaded()
    print("\n재하봇 텍스트 테스트(마이크·TTS 없음). 놀이도 됩니다.")
    print("예: '동물 소리 놀이 하자' -> '멍멍' -> ... -> '그만'   (종료: 빈 줄/exit)\n")
    while True:
        try:
            text = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"exit", "quit"}:
            break
        reply = games.handle(text)
        if reply is None:
            reply = games.maybe_start(text)
        if reply is None:
            reply = agent.respond(text)["text"]
        print(f"재하봇: {reply or SAFE_RECOVERY}")


if __name__ == "__main__":
    if "--text" in sys.argv:
        text_repl()
    else:
        main()
