"""호출어(웨이크워드) 판정 — 세션 트리거의 '감지기' 부분.

**현재 기본 감지기는 ONNX KWS(`wake_onnx.py`)다.** 이 파일에 남은 자모 매칭 방식은
ONNX 로드 실패 시의 **폴백**이다(config `wake.detector: onnx | stt`). 폴백을 남기는
이유는 모델 파일이 없거나 깨져도 봇이 죽으면 안 되기 때문.

자모 매칭이 주력에서 내려온 이유(실측 확정, 되돌리지 말 것):
- 유아 웅얼거림에서 positive/negative 분포가 겹쳐 **임계값을 그을 수 없다**(수집 28개).
- whisper 가 받아쓰는 4~6초 동안 호출을 통째로 놓친다.
→ 오디오를 직접 보는 전용 KWS 로 교체했다. 학습 툴은 **livekit-wakeword**(Colab).
  openWakeWord 는 Piper 기반이라 **한국어가 안 돼서 기각**했다.

임계값: 이 함수의 기본값은 0.6 이지만 **운영값은 config 의 `wake.threshold: 0.68`** 이다.
실측(2026-07-27, wake_samples): 또박또박 재하봇=0.25~0.57, 무관한 말=0.71~1.00.
주의: STT transcribe 는 이미 correct_stt(keywords)로 가까운 변형(≤0.34)을 '재하봇'으로
스냅한다. 여기 컷은 그보다 관대해 0.34~0.68 변형(재하/제하모 등)까지 깨운다.
마이크 없이 문자열로 검증 가능(REPL: python -m app.wake).
"""
from __future__ import annotations

import logging

import numpy as _np

from .text_norm import jamo_ratio

log = logging.getLogger("jaeha_bot.wake")

_PUNCT = " .,!?~\"'…"


def is_wake_word(
    text: str,
    word: str = "하이티드",
    threshold: float = 0.6,
    aliases: list[str] | None = None,
) -> bool:
    """text 안에 호출어(word)로 볼 만한 토큰이 있으면 True.

    - aliases: 확정 호출 오인식(정확 일치 시 즉시 True). 예: ["개하복","제하모"].
    - threshold: 자모 유사도 컷(작을수록 엄격). 단일 토큰 + 인접 2토큰 결합을 모두 검사
      (STT가 '하이 티드'처럼 띄어 적어도 결합해서 '하이티드'로 잡히게).
      🔴 그래서 매칭어는 **공백을 뺀 '하이티드'** 를 쓴다(config `wake.word`).
    """
    if not text:
        return False
    aliases = set(aliases or [])
    toks = [t.strip(_PUNCT) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return False

    # 1) 단일 토큰: 별칭 정확일치 or 자모 유사도 ≤ threshold
    for t in toks:
        if t in aliases or t == word:
            return True
        if jamo_ratio(t, word) <= threshold:
            return True

    # 2) 인접 2토큰 결합('하이 티드' -> '하이티드')도 검사
    for a, b in zip(toks, toks[1:]):
        if jamo_ratio(a + b, word) <= threshold:
            return True

    return False


def best_wake_ratio(text: str, word: str = "하이티드") -> float:
    """text 안에서 호출어(word)에 가장 가까운 자모거리(0=동일, 1=완전다름). 진단·튜닝용.

    대기 모드에서 '안 깨움' 시 이 값을 로그로 남기면, 실제로 뭐라고 들렸고 얼마나
    가까웠는지 보여 임계값·별칭을 실측으로 조정할 수 있다.
    """
    if not text:
        return 1.0
    toks = [t.strip(_PUNCT) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return 1.0
    cands = list(toks) + [a + b for a, b in zip(toks, toks[1:])]  # 단일 + 인접결합
    # 🔴 2026-08-27 **덧붙은 전사**를 살린다. 검증창(2초)에는 호출어 뒤에 뒷말이 같이
    #    들어오고, whisper 가 그걸 띄어쓰기 없이 한 토큰으로 붙여 적을 때가 있다.
    #    실기 14:37:07 — '하이치테오름'. 안에 '하이치테'(0.38)가 멀쩡히 있는데
    #    6음절을 4음절과 통째로 재느라 0.54 가 나와 기각됐다.
    #    ➡️ 호출어보다 긴 후보는 **앞 4음절**도 같이 본다. 토큰 경계는 지키므로
    #       문장 한복판에서 우연히 맞는 자리를 찾지는 않는다.
    #
    # 실기 전수(전사 46종) 재채점에서 **바뀌는 건 둘뿐**이다:
    #    하이치테오름  0.54 -> 0.38  (진짜 호출, 이제 통과)
    #    아이스크림    0.64 -> 0.50  (소음, 여전히 기각)
    # 컷 0.45 바로 위 값은 양쪽 다 0.50 이라 **여유가 줄지 않는다.**
    # ⚠️ 다만 '아이스크림'(3회 관측)이 컷에 0.05 까지 다가온다. 컷을 0.50 으로
    #    올리는 길은 이걸로 확실히 닫힌다 — 원래도 '아이쿠도'(0.50)가 막고 있었다.
    # ⚠️ 창을 통째로 미끄러뜨리는 방식도 재 봤고 **기각했다.** 얻는 건 같은 1건인데
    #    '오늘도 시청해 주셔서…'(0.75->0.50) 같은 문장이 딸려 내려와 여유를 잃는다.
    cands += [c[:len(word)] for c in cands if len(c) > len(word)]
    return min(jamo_ratio(c, word) for c in cands)


def is_sleep_command(text: str, words: list[str] | None = None) -> bool:
    """대화를 끝내고 다시 대기(잠듦)로 갈 발화인지. 놀이의 '그만'과 겹치지 않게 별도 단어."""
    if not text:
        return False
    words = words or ["잘자", "잘 자", "코자자", "코 자자", "바이바이", "안녕히"]
    t = text.strip(_PUNCT)
    return any(w in t for w in words)


# ── 감지기 인터페이스 ─────────────────────────────────────────────────────────
# 두 감지기는 입력이 다르다(STT=텍스트, ONNX=오디오). 그래서 is_wake_word(text)
# 수준에선 교체가 안 되고, '깨어날 때까지 기다린다' 수준으로 경계를 올린다.

class SttWakeDetector:
    """기존 STT 자모 판정을 감지기 인터페이스에 맞춘 래퍼.

    오디오를 다루지 않으므로 프리롤은 항상 비어 있고 continued=False 다
    (= 늘 '부르고 기다리기' 경로 = 이 방식의 기존 동작과 같다).
    """

    def __init__(self, stt, word: str = "하이티드", threshold: float = 0.68,
                 aliases: list | None = None) -> None:
        self.stt = stt
        self.word = word
        self.threshold = threshold
        self.aliases = aliases or []

    def wait_for_wake(self, max_turns: int | None = None):
        """호출어가 들릴 때까지 STT 로 듣는다.

        감지기 교체 가능한 표면은 **인자 없는 호출**뿐이다(`wait_for_wake()`).
        `max_turns` 는 테스트·진단 전용이며, `OnnxWakeDetector.wait_for_wake`의
        `max_frames`와 단위가 다르다(턴 수 vs 프레임 수). 감지기 종류를 모르는
        호출부(main.py)는 이 인자를 절대 넘기면 안 된다.
        """
        from .wake_onnx import WakeResult

        if self.stt is None:
            raise RuntimeError("stt 가 없다 — STT 인스턴스를 넘겨야 한다")

        turns = 0
        while max_turns is None or turns < max_turns:
            turns += 1
            text, _ = self.stt.listen()
            if not text:
                continue
            if is_wake_word(text, self.word, self.threshold, self.aliases):
                log.info("[호출] %s → 깨어남", text)
                return WakeResult(preroll=_np.zeros(0, dtype=_np.float32),
                                  continued=False, score=1.0)
            log.info("[대기] 안 깨움: %r (거리 %.2f / 임계 %.2f)",
                     text, best_wake_ratio(text, self.word), self.threshold)
        return None

    def reset(self) -> None:
        """ONNX 감지기와 인터페이스를 맞추기 위한 no-op."""


def make_wake_verifier(vcfg: dict, stt, word: str):
    """2단계 검증기를 만든다. 꺼져 있으면 None(= 1단계만 쓰는 옛 동작).

    무엇을 하나: 후보 구간(2초)을 whisper 로 전사해 호출어와 자모거리를 잰다.
    왜 이게 되나: 1단계 ONNX 는 실음성에서 점수가 눌려 있을 뿐 신호는 살아 있다
      (실측 2026-08-12: 임계 0.25→30% 인데 0.03→82%). 그래서 1단계는 낮게 열어
      놓치지 않는 데만 쓰고, 헛깨움은 여기서 건다. 캐스케이드 실측 재현율 80%.

    ⚠️ initial_prompt 없이는 whisper 가 '재하봇'을 못 읽는다(41%). 문장형 힌트로 98%.
       이 힌트는 **검증에만** 쓴다 — 대화 STT 에 쓰면 모든 말이 '재하봇'으로 편향된다.
    """
    if not vcfg.get("enabled", False):
        return None
    prompt = vcfg.get("initial_prompt") or None
    max_ratio = float(vcfg.get("max_ratio", 0.45))
    plain_ratio = float(vcfg.get("plain_max_ratio", 0.65))
    # 🟢 2026-08-26: 2패스를 끌 수 있게 했다. 기본 True 는 옛 설정을 그대로 읽기 위한 것이고,
    #    지금 우리 설정은 False 다 — 아래 2)번 주석에 근거가 있다.
    two_pass = bool(vcfg.get("two_pass", True))
    # 🔴 2026-08-27 기각된 후보를 **소리로 남긴다**(null = 끔).
    #    왜: 로그의 전사만으로는 그게 (a)진짜 호출인데 whisper 가 지어낸 것인지
    #    (b)정말 다른 소리였는지 **구분할 방법이 없다.** 실기에서 점수 0.459 짜리가
    #    '안녕히계세요' 로 전사돼 기각됐는데, 그게 놓친 호출인지 TV 소리인지 모른다.
    #    이 프로젝트는 추측으로 임계를 정했다가 두 번 틀렸다(우회컷 0.2, 컷 0.35).
    #    ➡️ 들어 보면 1초에 끝난다. 파일명에 점수·자모거리·전사가 들어간다.
    #    ⚠️ 계속 켜 두면 디스크가 찬다(후보 하나당 2초 = 128KB). 진단할 때만 켤 것.
    reject_dir = vcfg.get("save_rejects_dir") or None

    def _save_reject(audio, text: str, ratio: float) -> None:
        try:
            from datetime import datetime
            from pathlib import Path

            import soundfile as sf
            d = Path(reject_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            safe = "".join(c for c in (text or "무음") if c.isalnum())[:20] or "무음"
            name = f"{datetime.now():%H%M%S}_r{ratio:.2f}_{safe}.wav"
            sf.write(str(d / name), audio, 16000)
        except Exception as e:      # noqa: BLE001 — 진단 기능이 봇을 죽이면 안 된다
            log.debug("기각 오디오 저장 실패(무시): %s: %s", type(e).__name__, e)

    def verify(audio) -> bool:
        # ── 1) 힌트 **없이** 먼저 듣는다 ──────────────────────────────
        # 🔴 여기가 환각을 죽이는 자리다. initial_prompt 는 whisper 에게 '재하봇이 있다'고
        #    미리 알려 주는 장치라, 없는 오디오에도 만들어 낸다(유튜브 실측: 후보의 47~100%가
        #    '재하봇, 재하봇…' 으로 전사됐다). 힌트를 빼면 실제 내용이 나온다:
        #      힌트있음 0.00 '재하봇, 재하…'  /  힌트없음 1.00 '등과'
        #      힌트있음 0.00 '재하봇 두 이름…' /  힌트없음 0.88 '두 이릉이 있다면'
        #    유튜브 후보 15곳 중 이 관문을 통과한 것 **0건**.
        #    먼저 도는 이유는 비용이다 — 가짜는 여기서 끝나 whisper 를 한 번만 쓴다.
        plain, _ = stt.transcribe(audio, initial_prompt=None)
        r2 = best_wake_ratio(plain or "", word)
        if r2 > plain_ratio:
            log.info("[검증] 기각 — 힌트없이 '%s' 자모거리 %.2f > %.2f",
                     (plain or "")[:26], r2, plain_ratio)
            if reject_dir:
                _save_reject(audio, plain or "", r2)
            return False

        # ── 2) 힌트를 주고 다시 듣는다 ───────────────────────────────
        # 🔴 **호출어를 '하이 티드'로 바꾼 뒤로 이 관문은 할 일이 없다(2026-08-26).**
        #   원래 이유: '재하봇'은 OOV 고유명사라 힌트 없이는 41% 밖에 못 읽었다.
        #   '하이티드'는 whisper 가 그대로 읽는다(합성 36개 중 35개 거리 0.000).
        #
        #   실기 로그 전수(logs/jaeha_*.log):
        #     1패스에서 기각(2패스 안 감)   58건
        #     두 패스 다 돈 경우              9건
        #     그중 **2패스가 뒤집은 것        0건**
        #   즉 정밀도는 전부 1)이 만들고 있고, 2)는 성공한 호출마다 whisper 를 한 번 더
        #   돌릴 뿐이다. 실측 비용 **1.22초**(2026-08-26 15:37:23~24) — 깨어나는 데
        #   2.53초가 걸렸고 그 절반이 여기였다.
        #
        #   ⚠️ 되돌릴 상황: 호출어를 다시 OOV 고유명사로 바꾸면 이 관문이 필요해진다.
        #      그때는 설정에서 two_pass 를 true 로.
        if not two_pass:
            log.info("[검증] 통과 — 힌트없이 '%s' 자모거리 %.2f <= %.2f (1패스)",
                     (plain or "")[:26], r2, plain_ratio)
            return True

        hinted, _ = stt.transcribe(audio, initial_prompt=prompt)
        r1 = best_wake_ratio(hinted or "", word)
        ok = r1 <= max_ratio
        log.info("[검증] %s — 힌트없이 '%s'(%.2f) / 힌트주고 '%s'(%.2f, 컷 %.2f)",
                 "통과" if ok else "기각", (plain or "")[:16], r2,
                 (hinted or "")[:16], r1, max_ratio)
        return ok

    return verify


def _make_embed_rescue(vcfg: dict):
    """설정에서 임베딩 대조 장치를 만든다. 꺼져 있거나 본보기가 없으면 None.

    기본이 **꺼짐**인 이유: 컷(min_similarity)이 아직 3분짜리 소음 표본으로만
    정해져 있다. 이 프로젝트는 같은 크기의 표본으로 두 번 틀린 적이 있다
    (08-26 우회컷: 3분 최고 0.173 -> 같은 날 실기 0.725). 긴 녹음으로 다시 잡기 전엔
    켜지 않는다. 자세한 근거는 app/wake_embed.py 머리말.
    """
    ecfg = vcfg.get("embed_rescue", {}) or {}
    if not ecfg.get("enabled", False):
        return None
    from .wake_embed import load_rescue
    return load_rescue(ecfg.get("templates"),
                       float(ecfg.get("min_similarity", 0.85)))


def make_detector(wcfg: dict, stt, source):
    """설정에 따라 감지기를 고른다. ONNX 로드 실패 시 STT 로 폴백한다.

    폴백하는 이유: 모델을 아직 안 넣었거나 파일이 깨져도 봇이 죽으면 안 된다.
    """
    word = wcfg.get("word", "하이티드")
    threshold = float(wcfg.get("threshold", 0.68))
    aliases = wcfg.get("aliases", [])
    fallback = SttWakeDetector(stt, word, threshold, aliases)

    if wcfg.get("detector", "stt") != "onnx":
        return fallback

    ocfg = wcfg.get("onnx", {}) or {}
    vcfg = ocfg.get("verify", {}) or {}
    # 🔴 2026-09-02 검증 방식 스위치. **기본은 whisper = 현행 동작 그대로다.**
    #   whisper : 후보를 전사해 자모거리로 판정(현행)
    #   embed   : 임베딩 본보기와의 코사인 유사도로만 판정
    #             — **로컬 STT 를 안 올려도 된다**(전면 API/S2S 전환의 선결 조건)
    #   both    : whisper 로 판정하고 기각분을 임베딩이 건진다(OR 보강)
    # 🔴 **이 검사는 아래 try 밖에 있어야 한다.** 안에 두면 오타가 폴백 except 에
    #    삼켜져 "ONNX 로드 실패 → STT 폴백" 으로 둔갑한다(실제로 그렇게 났다).
    #    설정 오타는 조용히 넘어갈 일이 아니라 기동을 멈출 일이다.
    mode = vcfg.get("mode", "whisper")
    if mode not in ("whisper", "embed", "both"):
        raise ValueError(
            f"wake.onnx.verify.mode 가 이상하다: {mode!r} "
            "— whisper | embed | both 중 하나여야 한다")

    try:
        from .wake_onnx import OnnxWakeDetector
        verifier = None if mode == "embed" else make_wake_verifier(vcfg, stt, word)
        return OnnxWakeDetector(
            model_dir=ocfg.get("model_dir", "models/wake"),
            classifier=ocfg.get("classifier", "jaehabot.onnx"),
            threshold=float(ocfg.get("threshold", 0.5)),
            trigger_frames=int(ocfg.get("trigger_frames", 2)),
            providers=ocfg.get("providers"),
            source=source,
            verifier=verifier,
            verify_cooldown_s=float(vcfg.get("cooldown_s", 1.0)),
            verify_min_rms=float(vcfg.get("min_rms", 0.005)),
            verify_rearm_delta=float(vcfg.get("rearm_delta", 0.05)),
            # 1.01 = 끔. 점수는 1.0 을 못 넘으므로 설정이 없으면 옛 동작 그대로다.
            verify_bypass=float(vcfg.get("bypass_score", 1.01)),
            # 후보가 뜬 순간의 창은 호출어를 자르고 있을 수 있다 — 이만큼 더 듣고 본다.
            verify_settle_s=float(vcfg.get("settle_s", 0.0)),
            # whisper 가 지어낸 글 때문에 죽은 진짜 호출을 소리로 건진다(꺼져 있으면 None).
            embed_rescue=_make_embed_rescue(vcfg),
        )
    except Exception as e:
        log.warning("ONNX 호출어 감지기 로드 실패(%s: %s) → STT 감지기로 폴백",
                     type(e).__name__, e, exc_info=True)
        return fallback


def _repl() -> None:
    """문자열로 판정 확인(마이크 불필요)."""
    aliases = ["하이티브", "하이치드", "하이티들"]
    pos = ["하이티드", "하이 티드", "하이티드야", "하이티브", "하이치드"]
    neg = ["엄마 어디 있어", "사과 먹고 싶어", "이게 뭐야", "강아지 멍멍 해봐", "제가 웃어"]
    print("[깨워야 함]")
    for s in pos:
        print(f"  {is_wake_word(s, aliases=aliases)!s:5} <- {s!r}")
    print("[안 깨야 함]")
    for s in neg:
        print(f"  {is_wake_word(s, aliases=aliases)!s:5} <- {s!r}")


if __name__ == "__main__":
    _repl()
