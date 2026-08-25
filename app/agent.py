"""LLM Agent: 로컬 LLM(llama.cpp / GGUF)을 구동해 유아 대화 응답을 만든다.

가이드 5: 작은 모델 + 짧은 프롬프트 + 양자화(INT4/8)로 8GB 안 안정 구동.

역할 분담: 이 파일은 **자유 대화**(respond)와 **놀이 대사 렌더링**(render)만 한다.
놀이의 흐름·정답판정·트리거는 education_modes.py 의 상태머신이 갖는다 —
2.4B 에 상태를 맡기면 깨지므로 **function-calling 스키마는 쓰지 않는다**(설계 확정).
"""
from __future__ import annotations
import json
import logging
import os
import re
import threading
from pathlib import Path

log = logging.getLogger("jaeha_bot.agent")

# CLOVA Studio(하이퍼클로바X)는 OpenAI 호환 엔드포인트를 준다. 그래서 백엔드를 새로
# 쓰지 않고 base_url·키만 갈아끼운다.
# 🔴 갈래 판단 규칙은 **tools/eval_llm.py 와 똑같이** '모델 이름이 HCX- 로 시작하는가'다.
#    규칙이 둘이면 평가에서 잰 품질·지연이 운영에서 재현되지 않는다. 그래서 백엔드
#    교체는 `api_model: HCX-005` 한 줄이면 된다(backend 는 openai 그대로).
CLOVA_BASE_URL = "https://clovastudio.stream.ntruss.com/v1/openai"


class _Speculation:
    """다음 답을 **미리** 만들어 두는 한 칸짜리 슬롯.

    🔴 왜: 실기 15턴에서 체감 4.53s 중 '생각'이 1.71s(38%)로 가장 크다. 그런데 그
       앞의 VAD 꼬리 1.20s 는 정의상 무음이라 아무것도 안 하고 기다리는 시간이다.
       선행 인식(b3db02c)이 그 안에 STT 를 밀어넣어 -0.85s 를 벌었고, 이건 같은 수를
       한 단계 더 쓰는 것이다 — 인식이 꼬리 안에서 끝나면 곧바로 LLM 을 미리 친다.

    🔴 **키는 '아이가 한 말 그 자체'다.** 최종 인식이 다르면 키가 안 맞아 추측이 그냥
       버려진다 — 틀린 추측이 답으로 나가는 일이 **구조적으로** 불가능하다.
       (STT 쪽이 '마지막 말소리까지 모인 샘플 수'를 키로 쓰는 것과 같은 장치다.)
    ⚠️ 버린 추측의 손해는 호출 한 번이고 **지연은 0** 이다 — 어차피 기다리는 중이었다.
    ⚠️ take() 는 돌고 있는 추측을 **반드시 기다린다.** 안 기다리고 새로 치면 로컬
       llama.cpp 를 두 스레드가 동시에 부를 수 있다(스레드 안전하지 않다).
    """

    def __init__(self, work) -> None:
        self._work = work                 # callable(text) -> str
        self._lock = threading.Lock()
        self._key: str | None = None
        self._thread: threading.Thread | None = None
        self._result: str | None = None

    def start(self, text: str) -> None:
        with self._lock:
            if text == self._key and self._thread is not None:
                return                    # 같은 말을 두 번 추측하지 않는다
            self._key, self._result = text, None
            self._thread = threading.Thread(target=self._run, args=(text,),
                                            daemon=True, name="llm-speculation")
            self._thread.start()

    def _run(self, text: str) -> None:
        try:
            out = self._work(text)
        except Exception as e:            # 최적화가 본 기능을 죽이면 안 된다
            log.warning("선행 생각 실패(무시하고 평소 경로로): %s: %s",
                        type(e).__name__, str(e)[:120])
            out = None
        with self._lock:
            if self._key == text:
                self._result = out

    def take(self, text: str, timeout: float = 15.0) -> str | None:
        """그 말에 대한 추측을 (돌고 있으면 기다렸다) 돌려준다. 없으면 None."""
        with self._lock:
            if text != self._key:
                return None               # 다른 말이었다 = 추측은 버린다
            th = self._thread
        if th is not None:
            th.join(timeout)
        with self._lock:
            out = self._result
            self._key, self._thread, self._result = None, None, None
            return out

# 대화 이력은 최근 N턴(=사용자+로봇 2N개 메시지)만 유지한다.
# 컨텍스트를 짧게 유지해야 n_ctx 안에서 3초 목표를 지키기 쉽다.
MAX_HISTORY_TURNS = 6

# 놀이 상태머신이 넘긴 '상황 지시'를 문장으로 바꿀 때 쓰는 system 프롬프트(render()).
# 상황 설명을 그대로 읽지 말고, 아이에게 말하듯 밝은 반말 한두 문장으로만 렌더링한다.
_RENDER_SYSTEM = (
    "너는 2세 유아와 노는 다정한 로봇 '티드'야. "
    "아래 상황을 아이에게 말하듯 밝고 신나는 반말로 말해. "
    "한두 문장까지만, 아주 짧고 쉽게. 존댓말('~요/~습니다')·이모지·목록·번호·따옴표는 쓰지 않는다. "
    "지식·사실 설명이나 부연은 절대 하지 말고(예: '고양이는 눈빛으로 표현해' 같은 설명 금지), "
    "시키는 말(칭찬·질문)만 놀이처럼 그대로 한다. 상황 설명을 읽지 말고 자연스러운 대사로만 말한다."
)

# 응답은 TTS 로 '소리내어' 읽으므로 이모지/픽토그램이 섞이면 이상하게 읽힌다.
# 프롬프트로도 금지하지만, 모델이 안 지킬 수 있어 코드에서 확실히 제거한다.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # 이모지·픽토그램(이모티콘/사물/교통/보충)
    "\U00002600-\U000027BF"  # 기타 기호·딩뱃
    "\U00002B00-\U00002BFF"  # 기타 기호·화살표(⭐ 등)
    "\U00002300-\U000023FF"  # 기술 기호(⌚⏰ 등)
    "\U0001F1E6-\U0001F1FF"  # 국기(지역 표시자)
    "\U0000FE00-\U0000FE0F"  # 변형 선택자
    "\U0000200D"             # ZWJ(이모지 결합)
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    """이모지/특수 픽토그램을 제거하고, 그로 인해 생긴 이중 공백을 정리한다."""
    text = _EMOJI_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _flatten_markdown(text: str) -> str:
    """마크다운/목록 서식을 제거해 '소리로 읽을 수 있는 한 문단'으로 만든다.

    EXAONE 이 목록형 질문에 글머리표(-, *, 1.)·굵게(**)·여러 줄로 답하는 경우가 있는데,
    TTS 로 읽으면 '별표'·번호를 소리내 읽어 치명적이다. 프롬프트로도 막지만 코드로 확실히 제거한다.
    """
    text = re.sub(r"[*#`\"'‘’“”「」]+", "", text)  # 굵게/제목/코드/따옴표 제거
    # (모델이 답 전체를 "..."로 감싸면 문장경계·완결판정이 깨지고 TTS에도 불필요)
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-•]\s+|\d+[.)]\s*)", "", ln)  # 줄머리 목록 마커 제거
        ln = ln.strip()
        if ln:
            lines.append(ln)
    text = " ".join(lines)                          # 여러 줄 -> 한 문단
    return re.sub(r"\s{2,}", " ", text).strip()


# 모델이 가끔 응답 앞에 스스로 붙이는 '티드:' 이름표(대사 아님)를 제거한다.
# 옛 이름 '재하봇'도 같이 받는다 — 파인튜닝 데이터·캐시에 남아 있을 수 있다(2026-08-25 개명).
_SPEAKER_PREFIX_RE = re.compile(r"^\s*(?:티드|재하봇)\s*[:：]\s*")


def _strip_speaker_prefix(text: str) -> str:
    """'티드:' 같은 화자 이름표를 앞에서 제거한다(중복 붙어도 모두 제거)."""
    prev = None
    while prev != text:
        prev = text
        text = _SPEAKER_PREFIX_RE.sub("", text)
    return text.strip()


def _clamp_sentences(text: str, max_sentences: int = 2) -> str:
    """완결 문장을 최대 N개까지만 남긴다(문장 경계에서 자르므로 TTS 뚝끊김 없음).

    2.4B 는 프롬프트만으론 '최대 2문장'을 자주 어겨(감탄사로 연 뒤 사족을 붙임),
    종결부호(.!?~)로 끝나는 문장을 세어 N개를 넘으면 그 뒤(대개 사족)를 버린다.
    """
    sentences = re.findall(r".+?[.!?~]+(?:\s+|$)", text, flags=re.S)
    if len(sentences) <= max_sentences:
        return text.strip()
    return "".join(sentences[:max_sentences]).strip()


def load_fewshot(path) -> list[dict]:
    """finetune_seed.jsonl 에서 few_shot=true 예시만 뽑아 chat 메시지 리스트로 만든다.

    이 파일은 QLoRA 데이터셋 씨앗과 동일(단일 소스). '#' 로 시작하는 줄은 주석.
    파일이 없거나 비면 빈 리스트를 돌려주므로 few-shot 없이도 정상 동작한다.
    """
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        log.warning("few-shot 파일 없음(무시): %s", p)
        return []
    msgs: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("few_shot"):
            continue
        for m in row.get("messages", []):
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
    log.info("few-shot 예시 %d개 로드: %s", len(msgs) // 2, p.name)
    return msgs


def _augment_system(system_prompt: str, fewshot: list[dict]) -> str:
    """few-shot 메시지쌍을 system 프롬프트 끝에 '예시' 블록으로 붙인다.

    화자 이름표('티드:')를 가르치지 않도록 화살표 형식으로 렌더링한다.
    """
    lines = ["", "좋은 답변 예시(이 말투·길이를 따라해):"]
    for i in range(0, len(fewshot) - 1, 2):
        if fewshot[i]["role"] != "user" or fewshot[i + 1]["role"] != "assistant":
            continue
        lines.append(f'- 아이가 "{fewshot[i]["content"]}" 하면 → "{fewshot[i + 1]["content"]}"')
    return system_prompt + "\n" + "\n".join(lines)


class LLMAgent:
    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        *,
        n_ctx: int = 4096,   # 프롬프트 2356토큰 + 이력 + 답변. 근거는 configs/model_paths.yaml
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        seed: int | None = None,    # 샘플링 시드. None 이면 llama.cpp 가 매 실행 랜덤으로 잡는다
                                    # → 같은 질문에 매번 다른 답, 설정 A/B 도 노이즈에 묻힌다.
                                    # 고정하면 말투가 일정해지고 프롬프트 변경 효과를 비교할 수 있다
                                    # (TTS 가 seed 를 고정한 것과 같은 이유).
        max_tokens: int = 128,
        temperature: float = 0.7,
        repeat_penalty: float = 1.1,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        max_sentences: int = 2,
        fewshot_path: str | None = None,
        # ── 원격 백엔드(선택) ────────────────────────────────────────────────
        # "local"(기본, llama.cpp) | "openai".
        # 2026-08-10 젯슨 실측: LLM 중앙 1.769s -> 0.892s, 길이 지시도 API 가 더 잘 따른다.
        # 원격이어도 GGUF 경로는 남긴다 — 네트워크가 끊기면 그대로 폴백해야 하기 때문.
        backend: str = "local",
        api_model: str = "gpt-4o-mini",
        # 🔴 2026-08-18 3.0 -> 5.0. **3.0 은 타임아웃이 아니라 지연 증폭기였다.**
        #    3.0 은 08-10 의 p95 1.72s 를 보고 정한 값인데, 08-18 실측 p95 는 3.757s /
        #    최대 4.444s 다(회선 왕복이 2배가 됐다). 타임아웃이 정상 분산 안으로 들어와
        #    평범한 호출이 매번 문턱을 스쳤고, 스칠 때마다 아래 max_retries 로 3배가 됐다.
        #      재시도 켜짐 중앙 2.580s / p95 13.233s   ->   끔 중앙 1.610s / p95 3.757s
        #    5.0 은 관측 최대(4.444s) 바로 위다 — 정상 호출은 안 자르고 진짜 멈춤만 잡는다.
        # 🔴 이 값은 '네트워크가 죽었을 때 아이가 견디는 침묵'과 같다 —
        #    타임아웃 후에 로컬이 1.7초를 더 쓰므로 총 침묵 = api_timeout + 1.7s.
        #    (로컬 1.7s 는 2026-08-18 n_ctx 4096 에서 재확인: 중앙 1.701s / 최대 1.759s)
        api_timeout: float = 5.0,
        # 🔴 기동 때 폴백 GGUF 를 미리 올릴지. 기본은 **안 올린다**(젯슨 2,319MB 절약).
        #    원격 예열이 실패하면 이 값과 무관하게 올린다 — 자세한 규칙은 warm().
        #    메모리가 넉넉한 기계에서만 True 로 되돌릴 것.
        preload_local: bool = False,
    ) -> None:
        self.preload_local = preload_local
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        # 0=CPU 전용, -1=전체 레이어 GPU 오프로드. PC(llama-cpp CPU 빌드)에선 무시되어 무해,
        # Jetson(CUDA 빌드)에선 config 에서 -1 로 켜 GPU 가속. [[jaeha-bot-progress]]
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_sentences = max_sentences
        # few-shot 예시(=파인튜닝 씨앗과 동일 파일)를 system 프롬프트 안에 '예시'로 넣어 말투를 고정한다.
        # 가짜 대화 이력이 아니라 예시 텍스트로 넣어야 이전 예시의 '주제'가 실제 답에 새지 않는다.
        self.fewshot = load_fewshot(fewshot_path)
        if self.fewshot:
            self.system_prompt = _augment_system(system_prompt, self.fewshot)
        self.backend = backend
        self.api_model = api_model
        self.api_timeout = api_timeout
        # 🔴 '키 없음'은 네트워크 장애가 아니라 설정 오류다. _complete() 의 폴백에
        #    삼켜지면 "클로바를 켰다"고 믿는데 실제로는 EXAONE 이 2.3초씩 답하는
        #    상태가 되고, 경고 로그 한 줄 말고는 아무 신호가 없다. 그래서 여기서
        #    **기동 때** 터뜨린다. 2026-08-10 에 이 종류로 한 번 당했다.
        self._is_clova = backend == "openai" and api_model.startswith("HCX-")
        if self._is_clova and not os.environ.get("CLOVA_STUDIO_KEY"):
            raise ValueError(
                f"api_model={api_model} 인데 CLOVA_STUDIO_KEY 가 없습니다. "
                ".env 에 넣으세요(CLOVA Studio > 테스트 앱 > API 키. "
                "CLOVA Voice 키와 다릅니다).")
        self._oa = None   # OpenAI 클라이언트(첫 사용 때 1회 생성)
        # 선행 생각(VAD 꼬리 동안 미리 답을 만들어 두는 것). 자세한 이유는 _Speculation.
        self._spec = _Speculation(
            lambda t: self._complete(self._build_messages(t, "")))
        self.history: list[dict] = []
        self._llm = None  # 지연 로딩(첫 호출 때 1회만 로드)

    def _ensure_loaded(self):
        """GGUF 모델을 최초 1회 로드한다(모듈 순차 로딩으로 OOM 방지)."""
        if self._llm is None:
            from llama_cpp import Llama

            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"LLM 모델 파일이 없습니다: {path} "
                    f"(configs/model_paths.yaml 의 llm.model_path 확인)"
                )
            log.info("LLM 로딩 중: %s (n_ctx=%d, threads=%d, gpu_layers=%d)",
                     path, self.n_ctx, self.n_threads, self.n_gpu_layers)
            kw = {} if self.seed is None else {"seed": self.seed}
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                **kw,
            )
            log.info("LLM 로딩 완료")
        return self._llm

    def warm(self) -> None:
        """기동 예열. **로컬을 올릴지 말지는 원격이 살아 있는지로 정한다.**

        🔴 2026-08-19: 예전엔 무조건 로컬을 데웠다. 그런데 젯슨 실측에서 앱 RSS
           6,259MB 중 **EXAONE GGUF 가 2,319MB(37%)** 였고, `backend: openai` 라
           網이 죽지 않는 한 한 번도 안 쓰인다. 시스템 메모리가 7,348/7,607MB(96.6%)
           까지 차서 다음 최적화를 넣을 자리가 없었다.
             whisper medium 1,881MB / Supertonic 1,321MB / **EXAONE 2,319MB**

        그래서 규칙을 이렇게 둔다:
          - backend=local        → 본진이므로 언제나 데운다
          - 원격 예열 성공        → **로컬을 안 올린다**(2.3GB 절약). 운영 중 원격이
                                   실패하는 첫 턴에만 콜드 로드를 문다
          - 원격 예열 실패        → 網 없이 켜졌다는 뜻이다. **폴백이 곧 필요하므로
                                   지금 올린다.** 미루면 아이의 첫 질문이 7초를 문다
          - preload_local=True   → 메모리가 넉넉한 기계용 강제 옵션

        🔴 **로드만으로는 부족하다.** 2026-08-10 젯슨 실측:
             GGUF 로드 1.84s → 그런데 첫 폴백은 여전히 7.18s
           비용의 대부분은 시스템 프롬프트의 prompt eval 과 CUDA 커널 초기화라,
           실제 추론을 한 번 돌려야 사라진다(이후 1.7s 대). 그래서 데울 때는 반드시
           추론까지 한 번 돌린다.
        실패해도 예외를 올리지 않는다(모델 파일 없는 기계에서도 원격 백엔드는 돌아야 한다).
        이력은 건드리지 않는다 — 예열 대화가 남으면 첫 답변의 맥락이 오염된다.
        """
        if self.backend != "openai":
            self._warm_local()
            return
        api_ok = self._warm_api()
        if api_ok and not self.preload_local:
            log.info("로컬 폴백은 올리지 않는다(원격 정상 — 메모리 약 2.3GB 절약). "
                     "원격이 실패하는 첫 턴에 그때 올라온다.")
            return
        self._warm_local()

    def _warm_local(self) -> None:
        try:
            self._complete_local([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "안녕"},
            ])
        except Exception as e:
            log.warning("로컬 폴백 예열 실패(%s: %s) — 원격이 끊기면 응답이 늦어진다",
                        type(e).__name__, str(e)[:120])

    def _warm_api(self) -> bool:
        """API 커넥션(DNS·TCP·TLS)을 미리 맺어 둔다.

        🔴 젯슨 실측: 첫 호출 3.1~3.6초, 그 다음부터 0.5~0.6초. 차이는 순전히 최초 연결이다.
           호출어로 깨운 뒤 오는 **아이의 첫 질문**이 늘 이 값을 치르므로 기동 때 미리 다녀온다.
        토큰은 1개면 된다 — 연결만 맺으면 되고 답은 버린다.
        """
        try:
            self._api_client().chat.completions.create(
                model=self.api_model, max_completion_tokens=1,
                messages=[{"role": "user", "content": "안녕"}],
            )
            return True
        except Exception as e:   # 네트워크 없이 켜질 수도 있다. 기동을 막지 않는다.
            log.warning("API 커넥션 예열 실패(%s: %s) — 폴백을 지금 올린다",
                        type(e).__name__, str(e)[:120])
            return False

    # ------------------------------------------------------------ 완성(백엔드 선택)
    def _complete(self, messages: list[dict]) -> str:
        """messages -> 답변 텍스트. 원격이 실패하면 로컬로 내려간다.

        아이가 아무 답도 못 듣는 것이 최악이므로 예외를 밖으로 던지지 않는다.
        대신 반드시 경고를 남긴다 — 조용한 폴백은 '왜 느려졌는지'를 못 찾게 만든다.
        후처리(_strip_speaker_prefix 등)는 호출부가 공통으로 하므로 여기서는 하지 않는다.
        """
        if self.backend == "openai":
            try:
                return self._complete_api(messages)
            except Exception as e:  # 네트워크·인증·타임아웃 무엇이든 로컬로 간다
                log.warning("OpenAI LLM 실패(%s: %s) — 로컬 EXAONE 으로 폴백",
                            type(e).__name__, str(e)[:120])
        return self._complete_local(messages)

    def _complete_api(self, messages: list[dict]) -> str:
        # 인자를 model/max_completion_tokens/messages 로만 두는 것은 tools/eval_llm.py
        # make_openai 와 같게 맞추려는 것이다 — 평가에서 잰 품질·지연을 그대로 재현해야 한다.
        r = self._api_client().chat.completions.create(
            model=self.api_model,
            max_completion_tokens=self.max_tokens,
            messages=messages,
        )
        return (r.choices[0].message.content or "").strip()

    def _api_client(self):
        if self._oa is None:
            from openai import OpenAI

            # 🔴 max_retries=0 은 필수다. SDK 기본값(2)이면 타임아웃된 호출을 **처음부터
            #    두 번 더** 쳐서 p95 가 13.2s 까지 부푼다(2026-08-18 실측). 느린 호출은
            #    다시 치는 게 아니라 로컬로 내려가야 한다 — 그래야 위 '침묵 = 타임아웃 +
            #    1.7s' 계산이 성립한다. 재시도는 SDK 가 아니라 우리가 정한다.
            kw = {}
            if self._is_clova:
                # 클로바는 전용 키를 쓴다. OPENAI_API_KEY 관례에 얹으면 둘이 섞인다.
                kw = {"base_url": CLOVA_BASE_URL,
                      "api_key": os.environ["CLOVA_STUDIO_KEY"]}
            self._oa = OpenAI(timeout=self.api_timeout, max_retries=0, **kw)
        return self._oa

    def _complete_local(self, messages: list[dict]) -> str:
        out = self._ensure_loaded().create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )
        return (out["choices"][0]["message"].get("content") or "").strip()

    def _postprocess(self, reply: str) -> str:
        """소리로 읽을 수 있는 형태로 다듬는다. **백엔드와 무관하게 반드시 거친다.**

        이모지·마크다운·'티드:' 이름표는 TTS 로 읽으면 치명적이라, 원격 답변이라고
        건너뛰면 안 된다. respond()/render() 가 같은 것을 쓰도록 여기 모아 둔다.
        """
        reply = _strip_speaker_prefix(reply)  # '티드:' 이름표 제거(모델이 가끔 붙임)
        reply = _flatten_markdown(reply)      # 목록/굵게/여러줄 -> 한 문단(TTS로 기호 읽기 방지)
        reply = _strip_emoji(reply)           # 소리로 읽으므로 이모지 제거(프롬프트+코드 이중 차단)
        return _clamp_sentences(reply, self.max_sentences)  # 최대 N문장(사족 제거, 뚝끊김 방지)

    def _build_messages(self, user_text: str, vision_context: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        content = user_text
        if vision_context:
            content = f"[카메라 상태: {vision_context}]\n{user_text}"
        messages.append({"role": "user", "content": content})
        return messages

    def _remember(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        # 최근 MAX_HISTORY_TURNS 턴만 남기고 오래된 이력은 버린다.
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def respond(self, user_text: str, vision_context: str = "") -> dict:
        """유저 발화(+비전 상태) -> {"text": str, "function_call": None}.

        반환 text 가 비면 상위(main)에서 안전 복구 멘트로 대체한다.
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return {"text": "", "function_call": None}

        # 꼬리 동안 미리 만들어 둔 답이 있으면 쓴다. 키가 이 말과 같을 때만 쓰이므로
        # 인식이 바뀌었으면 자동으로 버려진다(_Speculation 주석 참조).
        # ⚠️ 비전이 붙는 턴은 재사용하지 않는다 — 추측은 카메라 상태 없이 만들어졌다.
        raw = None if vision_context else self._spec.take(user_text)
        if raw is None:
            raw = self._complete(self._build_messages(user_text, vision_context))
        reply = self._postprocess(raw)
        if reply:
            self._remember(user_text, reply)
        return {"text": reply, "function_call": None}

    def speculate(self, user_text: str) -> None:
        """아이 말이 (아직 확정 전이지만) 잡혔을 때 답을 **미리** 만들기 시작한다.

        VAD 꼬리 안에서 불린다. 아이가 말을 이어가 최종 인식이 달라지면 이 추측은
        키가 안 맞아 그냥 버려진다 — 호출 한 번을 버릴 뿐 지연은 0 이다.
        이력은 건드리지 않는다(추측은 '아직 일어나지 않은 턴'이다).
        """
        text = (user_text or "").strip()
        if text:
            self._spec.start(text)

    def render(self, instruction: str) -> str:
        """놀이 상태머신용 1회성 문구 생성(이력·few-shot 없이 stateless).

        상태머신이 넘긴 '상황 지시(instruction)'를 밝은 반말 한두 문장으로만 렌더링한다.
        대화용 respond() 와 분리한 이유: 놀이 렌더가 대화 이력을 오염시키면 안 되고,
        few-shot(자유대화용)도 끼면 안 되기 때문. 후처리는 respond() 와 동일하게 재사용한다.
        빈 문자열을 반환할 수 있으며(상위 GameManager 가 템플릿으로 폴백), 흐름은 절대 못 바꾼다.
        """
        instruction = (instruction or "").strip()
        if not instruction:
            return ""
        return self._postprocess(self._complete([
            {"role": "system", "content": _RENDER_SYSTEM},
            {"role": "user", "content": instruction},
        ]))


def _repl() -> None:
    """터미널 대화 테스트(STEP 3 완료 기준 확인용).

    실행: python -m app.agent   (종료: 빈 줄 또는 'exit'/'quit'/Ctrl+C)
    """
    from .config import settings

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    llm_cfg = dict(settings.models["llm"])
    model_path = llm_cfg.pop("model_path")
    agent = LLMAgent(
        model_path=model_path,
        system_prompt=settings.prompts["system"],
        **llm_cfg,
    )
    # 되묻기 문구의 단일 소스는 prompt_templates.yaml 의 recovery 다(main.py 와 같은 값).
    recovery = settings.prompts.get("recovery") or "어? 잘 못 들었어. 다시 말해줄래?"

    print("재하봇 1 — 글자 대화 테스트. (종료: 빈 줄 / exit / quit)")
    while True:
        try:
            user = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        result = agent.respond(user)
        print(f"티드: {result['text'] or recovery}")


if __name__ == "__main__":
    _repl()
