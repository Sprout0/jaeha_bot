"""STEP 7 계측: 턴마다 단계별 지연·메모리를 기록하고 세션 요약을 낸다.

목적: 젯슨 이식 '전' PC baseline 확보(이식 후 '같은' 로깅으로 나란히 비교) + '8GB·3초'
      제약을 어림값이 아니라 실측으로 검증.
동작: 말 걸 때마다 = 한 턴마다 JSON 한 줄을 logs/metrics_YYYYMMDD.jsonl 에 자동 추가.
      세션 종료(Ctrl+C) 시 중앙값/p90/최대메모리 요약을 출력·기록.
켜고끄기: configs/model_paths.yaml 의 metrics.enabled (기본 켬). tag 로 환경 구분(pc/jetson).

지연은 매 턴 출렁이므로(콜드/웜·답 길이) 한 번이 아니라 여러 턴을 모아 중앙값으로 본다.
비교의 핵심은 '연산 단계'(STT 인식+LLM+TTS) — GPU가 바꾸는 부분. 녹음대기(사람이 말하는
시간)와 고정대기(무음·에코쿨다운)는 하드웨어 무관이라 분리해서 본다.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("jaeha_bot.metrics")

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # psutil 없으면 메모리만 생략하고 지연은 계속 측정
    _PROC = None


def _rss_mb() -> float | None:
    if _PROC is None:
        return None
    try:
        return round(_PROC.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _p90(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.9 * (len(s) - 1) + 0.5))]


def _words(text: str) -> list[str]:
    """공백 기준 낱말(문장부호 제거). 확장 폭·재사용률 계산용 대충 나누기다."""
    return re.sub(r"[^\w\s]", " ", text or "").split()


def _reuse_ratio(child: str, reply: str) -> float | None:
    """아이 낱말 중 봇 답변에 다시 나온 비율. 모방·확장의 **대리지표**다.

    부분일치로 센다 — 어절 정확 비교면 조사가 붙은 '멍멍을' 을 못 잡는다.
    ⚠️ 정확한 값이 아니라 추세를 보는 숫자다. 아이가 실제로 더 말했는지는 재지 못한다.
    """
    cw = _words(child)
    if not cw:
        return None
    rep = reply or ""
    return round(sum(1 for w in cw if w in rep) / len(cw), 2)


class MetricsLogger:
    """턴별 계측 수집기. main 루프에서 단계 시간을 넘겨 record_turn() 만 부르면 된다."""

    def __init__(self, enabled: bool = True, tag: str = "pc",
                 log_dir: str = "logs") -> None:
        self.enabled = enabled
        self.tag = tag
        self.turn = 0
        self.samples: list[dict] = []
        base = Path(log_dir)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent.parent / base
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"metrics_{datetime.now():%Y%m%d}.jsonl"

    def _write(self, rec: dict) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("계측 기록 실패(무시): %s", e)

    def after_load(self) -> None:
        """모델 3종 로드 직후의 메모리 baseline 을 한 줄 남긴다."""
        if not self.enabled:
            return
        rss = _rss_mb()
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "tag": self.tag, "event": "loaded", "rss_mb": rss}
        self._write(rec)
        if rss is not None:
            log.info("[계측] 모델 로드 후 메모리 RSS=%.0fMB", rss)

    def record_turn(self, *, stt_wait_s: float, stt_rec_s: float,
                    think_s: float, think_kind: str, tts_first_s: float,
                    tts_play_s: float = 0.0, reply: str = "",
                    child_text: str = "") -> None:
        """한 턴의 단계 지연을 기록한다.

        stt_wait_s : 녹음대기(사람이 말한 시간 포함, 시스템 비용 아님) — 참고용.
        stt_rec_s  : STT 인식 '연산' 시간.
        think_s    : 답 생성 시간(LLM 또는 놀이 렌더). think_kind=llm/game/game_start 등.
        tts_first_s: 말끝부터 **첫 소리가 날 때까지**. 아이가 체감하는 대기의 마지막 조각.
        tts_play_s : 봇이 실제로 말하는 시간. **지연이 아니다** — 따로 기록만 한다.
        child_text : 아이가 한 말(STT 출력). 확장·재사용 대리지표 계산에만 쓴다.

        🔴 2026-08-09 정정: 예전에는 `tts_s` 하나에 합성+**재생 완료까지**를 담고
           그걸 resp_compute_s 에 더했다(speak() 이 sd.wait() 로 끝까지 기다린다).
           그래서 '응답 3초' 지표에 **봇이 말하는 시간이 통째로** 들어가 있었다
           — 젯슨 실측 tts_s 중앙 7.74s / resp 12.87s 가 그렇게 나온 값이다.
           지금은 resp_compute_s = stt_rec + think + tts_first 로, 말끝부터 첫 소리까지만
           센다. 이게 '몇 초 만에 반응하나'에 해당하는 숫자다.
           ⚠️ 그래서 이 필드는 2026-08-09 이전 기록과 직접 비교하면 안 된다.
        """
        if not self.enabled:
            return
        self.turn += 1
        resp = round(stt_rec_s + think_s + tts_first_s, 3)
        # 낱말을 파싱해서 개수를 봐야 한다 — 원본 문자열은 "..." 처럼 문장부호만
        # 있어도 truthy라서, 원본으로 판단하면 reuse(None)와 expansion_delta(숫자)가
        # 같은 턴에서 서로 다른 결론을 낸다. 반드시 _words() 결과로 판단할 것.
        child_words = _words(child_text)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": self.tag, "turn": self.turn,
            "cold": self.turn == 1,           # 첫 턴은 워밍업 영향 있을 수 있음
            "stt_wait_s": round(stt_wait_s, 3),
            "stt_rec_s": round(stt_rec_s, 3),
            "think_s": round(think_s, 3),
            "think_kind": think_kind,
            "tts_first_s": round(tts_first_s, 3),   # 첫 소리까지(체감)
            "tts_play_s": round(tts_play_s, 3),     # 봇이 말하는 시간(지연 아님)
            "resp_compute_s": resp,           # = 말끝부터 첫 소리까지
            "metric_ver": 2,                  # 1 = tts 재생시간이 섞여 있던 옛 기록
            "rss_mb": _rss_mb(),
            "reply_len": len(reply or ""),
            "expansion_delta": (len(_words(reply)) - len(child_words)
                                if child_words else None),
            "reuse": _reuse_ratio(child_text, reply),
        }
        self.samples.append(rec)
        self._write(rec)
        log.info("[계측] 턴%d 첫소리까지 %.2fs (STT인식 %.2f + 생각 %.2f + TTS %.2f)"
                 " | 발화 %.2fs | 메모리 %sMB",
                 self.turn, resp, stt_rec_s, think_s, tts_first_s, tts_play_s,
                 rec["rss_mb"])

    def summary(self) -> None:
        """세션 요약(중앙값/p90/최대 메모리)을 출력하고 파일에 남긴다."""
        if not self.enabled or not self.samples:
            return
        # 콜드 턴은 대표값에서 빼서 웜 성능을 본다(콜드는 따로 표시).
        warm = [s for s in self.samples if not s["cold"]] or self.samples
        resp = [s["resp_compute_s"] for s in warm]
        rec_ = [s["stt_rec_s"] for s in warm]
        think = [s["think_s"] for s in warm]
        tts = [s["tts_first_s"] for s in warm]
        play = [s["tts_play_s"] for s in warm]
        rss = [s["rss_mb"] for s in self.samples if s["rss_mb"] is not None]
        under3 = sum(1 for r in resp if r <= 3.0)
        summ = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": self.tag, "event": "summary", "metric_ver": 2,
            "turns": len(self.samples),
            "resp_median_s": round(statistics.median(resp), 3),
            "resp_p90_s": round(_p90(resp), 3),
            "stt_rec_median_s": round(statistics.median(rec_), 3),
            "think_median_s": round(statistics.median(think), 3),
            "tts_first_median_s": round(statistics.median(tts), 3),
            "tts_play_median_s": round(statistics.median(play), 3),
            "rss_peak_mb": max(rss) if rss else None,
            "under_3s_ratio": round(under3 / len(resp), 2) if resp else None,
        }
        self._write(summ)
        print("\n===== 세션 계측 요약 (warm 기준) =====")
        print(f"  턴 수: {summ['turns']}  (환경 tag={self.tag}, 파일={self.path.name})")
        print(f"  ▶ 말끝부터 첫 소리까지  중앙값 {summ['resp_median_s']:.2f}s"
              f" / p90 {summ['resp_p90_s']:.2f}s"
              f"  (목표 3s 이내 {int((summ['under_3s_ratio'] or 0)*100)}%)")
        print(f"    단계  STT인식 {summ['stt_rec_median_s']:.2f}s"
              f" | 생각 {summ['think_median_s']:.2f}s"
              f" | TTS합성 {summ['tts_first_median_s']:.2f}s")
        print(f"    (참고) 봇이 말하는 시간 {summ['tts_play_median_s']:.2f}s"
              " — 지연이 아니라 발화 길이다")
        if summ["rss_peak_mb"]:
            print(f"  최대 메모리 RSS {summ['rss_peak_mb']:.0f}MB (8GB 예산 대비)")
        print("====================================\n")
