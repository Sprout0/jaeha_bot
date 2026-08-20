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
from .wake import is_sleep_command, make_detector
from .audio_source import AudioSource

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jaeha_bot")


def _add_file_log() -> None:
    """콘솔에만 나가던 로그를 파일에도 남긴다.

    🔴 왜 필요한가 (2026-08-12): 2단계 검증을 켜고 실기를 돌렸는데 **헛깨움이 몇 번
       났는지 사후에 셀 수가 없었다.** [호출]·[검증] 줄이 터미널에만 있었기 때문이다.
       임계값·검증컷은 실기 로그로만 정할 수 있는 값이라, 안 남기면 영영 못 고친다.
    """
    from datetime import datetime
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "logs"
    d.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(d / f"jaeha_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(h)


_add_file_log()

# 되묻기 문구는 configs/prompt_templates.yaml 의 `recovery` 한 곳에서만 온다.
# 🔴 예전엔 여기 하드코딩과 yaml 이 각자 있어서, yaml 만 고친 변경이 운영에 하나도
#    반영되지 않았다(yaml 의 유일한 소비처가 app/agent.py 의 터미널 REPL 이었다).
#    설정을 고쳤는데 봇이 그대로 말하는 종류의 사고라 눈에도 안 띈다. → 단일 소스로 묶는다.
SAFE_RECOVERY = settings.prompts.get("recovery") or "어? 잘 못 들었어. 다시 말해줄래?"
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


def _empty_text_action(*, rejected: bool, wake_enabled: bool,
                       idle_s: float, sleep_timeout: float) -> str:
    """인식 결과가 비었을 때 무엇을 할지 결정한다 — 'reask' | 'sleep' | 'wait'.

    빈 결과에는 성격이 다른 둘이 섞여 있다: (a)아무 말도 없었다 (b)말은 했는데
    환각이라 버렸다. (b)에서 침묵하면 아이는 로봇이 고장 난 줄 안다. 또 (b)를
    '무응답'으로 세면 말할수록 잠드는 로봇이 되므로 되묻기가 잠들기보다 우선이다.
    """
    if rejected:
        return "reask"
    if wake_enabled and idle_s >= sleep_timeout:
        return "sleep"
    return "wait"


def _to_standby(detector, source) -> None:
    """대화 → 대기로 돌아갈 때의 뒷정리.

    감지기 링버퍼를 비우지 않으면 대화 중 쌓인 임베딩이 남아 대기 첫 순간에
    엉뚱한 점수가 나온다. 입력 버퍼도 비워 방금 한 인사말(SLEEP_MSG)을
    호출어로 오인하지 않게 한다.
    """
    if detector is not None:
        detector.reset()
    if source is not None:
        source.drain()


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
    # 기동 때 미리 데운다. 첫 턴 지연을 기동 쪽으로 옮기는 것뿐이라 손해가 없다.
    # - 로컬 GGUF: 폴백이 차가우면 네트워크가 끊긴 그 턴에 6초를 기다린다(2026-08-10 실측).
    # - API 커넥션: 최초 TLS 수립이 3초대다. 아이의 첫 질문이 늘 이걸 치른다.
    agent.warm()
    tts.warm()
    return stt, tts, vision, agent


def preload(stt, tts, agent) -> None:
    """첫 대화 지연을 없애기 위해 무거운 모델을 미리 올린다(순차 로딩으로 OOM 방지).

    🔴 **LLM 은 여기서 올리지 않는다.** `agent.warm()` 이 이미 '로컬 폴백을 올릴지'를
       정했다(원격이 살아 있으면 안 올려서 2.3GB 를 아낀다). 여기서 `_ensure_loaded()`
       를 부르면 그 결정을 덮어써서 아낀 것을 도로 문다.
       2026-08-20 실기 로그에서 실제로 그러고 있었다 — 같은 기동에서 5초 사이에
         "로컬 폴백은 올리지 않는다(... 약 2.3GB 절약)"
         "LLM 로딩 중: models/exaone-3.5-2.4b-q4.gguf"
       가 연달아 찍혔다. 커밋 ca260c7 이 warm() 만 고치고 이 호출부를 놓쳤다.
       **로그에 '절약했다'고 찍히기까지 해서 더 안 보였다.**
    """
    log.info("모델 미리 로딩 중... (STT -> TTS)")
    stt.load()
    tts.load()


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
    preload(stt, tts, agent)
    metrics.after_load()  # 모델 로드 후 메모리 baseline
    log.info("준비 완료. 말을 걸어보세요! (종료: Ctrl+C)")

    # 트리거(호출어) 설정: wake.enabled 면 '대기↔대화' 세션. 아니면 항상 대화(옛 동작).
    wcfg = settings.models.get("wake", {}) or {}
    wake_enabled = wcfg.get("enabled", True)
    wake_word = wcfg.get("word", "재하봇")
    sleep_timeout = float(wcfg.get("sleep_timeout", 30))
    sleep_words = wcfg.get("sleep_words")

    # 감지기 준비. ONNX 감지기는 마이크 스트림을 직접 읽으므로 공유 AudioSource 가 필요하다.
    # STT 감지기(폴백)는 stt.listen() 을 쓰므로 공유 스트림이 없어야 한다 — 둘 다 열면
    # 젯슨 ReSpeaker 처럼 장치가 하나뿐인 환경에서 InputStream 충돌이 난다.
    source = None
    detector = None
    if wake_enabled:
        # 검증창은 프리롤과 **별도 버퍼**다. 프리롤을 늘리면 호출 직전 TV·부모 말소리가
        # 섞여 whisper 가 환각하므로 0.5초를 유지하고, 2단계 검증만 긴 창을 쓴다.
        source = AudioSource(
            preroll=float(wcfg.get("preroll", 0.5)),
            verify_window=float(((wcfg.get("onnx") or {}).get("verify") or {})
                                .get("window_s", 2.0)),
        ).open()
        detector = make_detector(wcfg, stt, source)
        # ONNX 감지기만 .source 를 갖는다. 폴백(STT)이면 공유 스트림을 닫아 충돌을 막는다.
        if getattr(detector, "source", None) is None:
            source.close()
            source = None
            log.info("STT 감지기 사용 — 공유 스트림 해제")

    # 시작: 트리거 모드면 '대기'(부를 때까지 조용), 아니면 바로 인사하고 '대화'.
    if wake_enabled:
        log.info("트리거 모드: '%s' 라고 부르면 깨어납니다 (감지기 %s)",
                 wake_word, type(detector).__name__)
        tts.speak(READY_ASLEEP)
        awake = False
    else:
        tts.speak(GREETING)
        awake = True
    time.sleep(ECHO_COOLDOWN)
    if source is not None:
        source.drain()      # 인사말이 마이크에 남은 것 버리기
    last_active = time.time()
    pending_prefix = None   # 호출어 직후 이어진 발화(있으면 STT 에 그대로 넘긴다)

    try:
        while True:
            # ── 대기 모드: 호출어만 기다린다(LLM·TTS·놀이 안 함 = 자원 절약). ──
            # 감지기가 깨어날 때까지 블로킹한다. 인자 없는 호출만 쓴다(감지기 종류에
            # 따라 인자 단위가 달라서 — wake.py 의 인터페이스 주석 참고).
            if wake_enabled and not awake:
                result = detector.wait_for_wake()
                if result is None:          # 감지기가 스스로 끝냄(테스트/진단 경로)
                    continue
                awake = True
                last_active = time.time()
                if result.continued and result.preroll.size:
                    # '재하봇 이거 뭐야?' 처럼 부르고 바로 이어 말한 경우 —
                    # 인사말을 하면 뒷말을 놓치므로 생략하고 그 오디오를 STT 로 넘긴다.
                    log.info("호출 직후 발화 이어짐 → 인사말 생략")
                    pending_prefix = result.preroll
                else:
                    # 부르고 기다리는 경우 — 인사하고 평소처럼 듣는다.
                    tts.speak(WAKE_GREETING)
                    time.sleep(ECHO_COOLDOWN)
                    if source is not None:
                        source.drain()
                    pending_prefix = None

            # 듣기: listen 전체시간 - 인식연산(tr_dt) = 녹음대기.
            # source 가 있으면 감지기와 같은 스트림을 쓴다(장치 하나만 열기 위해).
            t_listen = time.perf_counter()
            text, tr_dt = stt.listen(source=source, prefix=pending_prefix)
            pending_prefix = None
            stt_wait = max(0.0, (time.perf_counter() - t_listen) - tr_dt)

            # ── 대화 모드 ──
            # 빈 결과: 환각이라 버렸으면 되묻고, 진짜 무음이면 시간을 보고 잠든다.
            if not text:
                action = _empty_text_action(
                    rejected=stt.last_rejected, wake_enabled=wake_enabled,
                    idle_s=time.time() - last_active, sleep_timeout=sleep_timeout)
                if action == "reask":
                    log.info("환각으로 버림 → 되묻기")
                    tts.speak(SAFE_RECOVERY)
                    last_active = time.time()
                    time.sleep(ECHO_COOLDOWN)
                    if source is not None:
                        source.drain()
                elif action == "sleep":
                    log.info("무응답 %.0f초 → 대기 모드로", sleep_timeout)
                    awake = False
                    tts.speak(SLEEP_MSG)
                    time.sleep(ECHO_COOLDOWN)
                    _to_standby(detector, source)
                continue
            log.info("[아이] %s", text)

            # '잘자/바이바이' 등은 대화를 끝내고 대기로(놀이의 '그만'과 겹치지 않는 별도 단어).
            if wake_enabled and is_sleep_command(text, sleep_words):
                log.info("잠들기 명령 → 대기 모드로")
                awake = False
                tts.speak(SLEEP_MSG)
                time.sleep(ECHO_COOLDOWN)
                _to_standby(detector, source)
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
            # speak() 은 재생이 끝날 때까지 막힌다. 그래서 '첫 소리까지'와 '말하는 시간'을
            # 스스로 나눠 돌려준다 — 바깥에서 재면 둘이 합쳐져 지연이 부풀려진다.
            tm = tts.speak(reply)
            last_active = time.time()  # 마지막 상호작용 시각(잠들기 타이머 기준)
            # vad_tail = 말끝부터 녹음이 끊길 때까지. stt_wait 안에 아이가 말한
            # 시간과 섞여 있어 따로 못 보던 구간이다 — 이게 지연 예산의 첫 조각이다.
            metrics.record_turn(stt_wait_s=stt_wait, stt_rec_s=tr_dt,
                                vad_tail_s=stt.last_vad_tail_s,
                                think_s=think_s, think_kind=kind,
                                tts_first_s=tm.first_audio_s,
                                tts_synth_s=tm.synth_s,
                                tts_play_s=tm.play_s, reply=reply,
                                child_text=text)
            time.sleep(ECHO_COOLDOWN)
            if source is not None:
                source.drain()   # 답하는 동안 쌓인 자기 목소리 버리기
    except KeyboardInterrupt:
        log.info("종료 신호(Ctrl+C) 수신")
    finally:
        metrics.summary()  # 세션 요약(중앙값/p90/최대메모리) 출력·기록
        if source is not None:
            source.close()

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
