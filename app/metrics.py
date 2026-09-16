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
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("jaeha_bot.metrics")

try:
    import psutil
    _PROC = psutil.Process()
    # cpu_percent(interval=None) 은 '직전 호출 이후'를 재므로 첫 호출이 항상 0.0 이다.
    # 여기서 한 번 불러 기준점을 잡아 두지 않으면 **첫 턴 CPU 가 늘 0** 으로 남는다.
    psutil.cpu_percent(interval=None)
except Exception:  # psutil 없으면 메모리만 생략하고 지연은 계속 측정
    _PROC = None


def _rss_mb() -> float | None:
    if _PROC is None:
        return None
    try:
        return round(_PROC.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


# 젯슨 GPU 부하는 sysfs 에 per-mille(0~1000) 로 있다. 보드/JetPack 마다 경로가
# 달라 후보를 순서대로 훑는다. 노트북엔 없으므로 전부 실패하면 None 이다.
_GPU_LOAD_PATHS = (
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/platform/17000000.ga10b/load",
)


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _gpu_pct() -> float | None:
    for path in _GPU_LOAD_PATHS:
        v = _read_int(path)
        if v is not None:
            return round(v / 10.0, 1)     # per-mille -> %
    return None


def _temp_c() -> float | None:
    """가장 뜨거운 thermal zone. 젯슨이 열로 클럭을 떨구는지 보려는 값이다."""
    try:
        import glob
        vals = [v for p in glob.glob("/sys/class/thermal/thermal_zone*/temp")
                if (v := _read_int(p)) is not None and 0 < v < 200000]
        return round(max(vals) / 1000.0, 1) if vals else None
    except Exception:
        return None


def _sys_stats_now() -> dict:
    """**지금 이 순간**의 RSS·CPU·시스템 메모리·GPU·온도. 없는 값은 None, 예외는 안 낸다.

    ⚠️ 이건 스냅샷 한 장이다. 턴 요약의 '최대'로 쓰면 안 된다 — 그러다 GPU 0% 오진이
       났다. 구간 최댓값은 _PeakSampler 가 만든다.

    🔴 프로세스 RSS(`_rss_mb`)와 시스템 메모리는 다른 숫자다. 8GB 예산은 보드 전체
       기준이라 다른 프로세스(브라우저·빌드)가 먹은 것도 봐야 한다. RSS 만 보다가
       "여유 3.3GB" 라고 판단하면 틀릴 수 있다.
    """
    out: dict = {"rss_mb": _rss_mb(), "cpu_pct": None, "sys_mem_used_mb": None,
                 "sys_mem_total_mb": None, "gpu_pct": _gpu_pct(),
                 "temp_c": _temp_c()}
    if _PROC is not None:
        try:
            import psutil
            out["cpu_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            out["sys_mem_used_mb"] = round((vm.total - vm.available) / 1048576, 1)
            out["sys_mem_total_mb"] = round(vm.total / 1048576, 1)
        except Exception:
            pass
    return out


class _PeakSampler:
    """자원을 주기적으로 훑어 **구간 최댓값**을 남긴다. 스냅샷 한 장이 아니라.

    🔴 왜 필요한가 (2026-08-19 젯슨에서 겪은 오진):
       예전엔 record_turn() 안에서 _sys_stats() 를 한 번 읽었다. 그런데 record_turn()
       은 **턴이 다 끝난 뒤** 호출된다(같은 기록에 tts_play_s 가 있다 = 재생까지 끝났다).
       즉 그 턴에서 **가장 조용한 순간**을 찍고 그걸 '최대'라고 불렀다.
       그 결과 요약에 `GPU 0%` 가 찍혔고 "GPU 를 안 쓰는구나" 라는 정반대 결론이 나왔다.
       같은 시각 sysfs 를 직접 훑으니 추론 중 **874~999 per-mille(87~100%)** 였다.
       CPU 도 같은 병이었다 — psutil 은 '직전 호출 이후 평균'이라, 한 턴에 한 번 부르면
       아이가 말하기를 기다린 시간까지 섞여 희석된다. 여기서 0.2초마다 부르면
       그 구간의 진짜 CPU 가 된다.

    ⚠️ 이 종류의 오류는 숫자가 그럴듯해서 안 보인다. 0% 가 아니라 40% 였다면
       아무도 의심하지 않았을 것이다. 자원 지표를 새로 붙일 때는 '언제 재는가'부터 정할 것.
    """

    def __init__(self, period_s: float = 0.2) -> None:
        self._period = period_s
        self._lock = threading.Lock()
        self._peak: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="metrics-peak")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            try:
                self._observe(_sys_stats_now())
            except Exception:   # 계측이 본 기능을 절대 죽이지 않는다
                pass

    def _observe(self, sample: dict) -> None:
        with self._lock:
            for k, v in sample.items():
                if v is None:           # 노트북엔 GPU sysfs 가 없다. None 이 최댓값을 지우면 안 된다
                    continue
                cur = self._peak.get(k)
                if cur is None or v > cur:
                    self._peak[k] = v

    def pop(self) -> dict:
        """구간 최댓값을 돌려주고 **비운다** — 다음 턴은 새 구간이다.

        안 비우면 첫 턴의 최고치가 세션 내내 따라다녀서 턴별 비교가 무의미해진다.
        """
        with self._lock:
            peak, self._peak = self._peak, {}
        return peak


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
        self._sampler = _PeakSampler()
        if enabled:
            self._sampler.start()

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
                    tts_synth_s: float = 0.0,
                    tts_play_s: float = 0.0, reply: str = "",
                    child_text: str = "", vad_tail_s: float = 0.0,
                    filler_wait_s: float = 0.0, spec: str = "") -> None:
        """한 턴의 단계 지연을 기록한다.

        vad_tail_s : **말끝 → 녹음 종료**. 아이가 순전히 기다리는 구간이다.
                     `stt.last_vad_tail_s` 를 그대로 넘기면 된다.
        stt_wait_s : 녹음대기(사람이 말한 시간 + 꼬리 포함) — 참고용.
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

        🔴 2026-08-26 추가: `filler_wait_s`. 필러가 아직 말하는 중이면 답이 그 말끝까지
           기다렸다 이어받는다(filler.await_quiet). 그 대기는 think 측정이 끝난 뒤
           speak 이 시작되기 **전**에 일어나서 꼬리·인식·생각·첫소리 어디에도 안 잡혔다
           — 아이는 기다렸는데 지표만 몰랐다. 아래 08-09·08-13 과 **같은 종류의 구멍**이라
           같은 방식으로 막는다: felt 에 더하고, 따로도 보이게 남긴다.

        🔴 2026-08-13 추가(metric_ver 3): 그 `resp_compute_s` 도 '말끝부터'가
           아니었다. **VAD 꼬리 대기가 빠져 있었다** — 말끝 뒤 조용해질 때까지
           기다리는 시간은 `stt_wait_s` 안에서 아이가 말한 시간과 섞여 있어
           아무도 볼 수 없었다. 이름과 내용이 어긋난 08-09 와 같은 종류의 실수다.
           그래서 `under_3s_ratio` 가 꼬리만큼 부풀려져 있었다.
           → `resp_felt_s` = 꼬리 + 인식 + 생각 + 첫소리 가 **아이가 겪는 시간**이다.
             `resp_compute_s` 는 옛 기록과 이어 보려고 남겨 둔다.
        """
        if not self.enabled:
            return
        self.turn += 1
        resp = round(stt_rec_s + think_s + tts_first_s, 3)
        felt = round(vad_tail_s + resp + filler_wait_s, 3)
        # 낱말을 파싱해서 개수를 봐야 한다 — 원본 문자열은 "..." 처럼 문장부호만
        # 있어도 truthy라서, 원본으로 판단하면 reuse(None)와 expansion_delta(숫자)가
        # 같은 턴에서 서로 다른 결론을 낸다. 반드시 _words() 결과로 판단할 것.
        child_words = _words(child_text)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": self.tag, "turn": self.turn,
            "cold": self.turn == 1,           # 첫 턴은 워밍업 영향 있을 수 있음
            "stt_wait_s": round(stt_wait_s, 3),
            "vad_tail_s": round(vad_tail_s, 3),   # 말끝→녹음종료(순수 대기)
            "stt_rec_s": round(stt_rec_s, 3),
            "think_s": round(think_s, 3),
            "think_kind": think_kind,
            "tts_first_s": round(tts_first_s, 3),   # 첫 소리까지(체감)
            # 🔴 첫 소리 = 합성 + 재생 시작. 나눠 두지 않으면 어디가 느린지 못 찾는다.
            #    08-19 에 실제로 막혔다 — 세션 1.12s 인데 격리 측정 0.63s 였고, 가설
            #    넷(GPU 경합·메모리·입력 스트림·측정 낙관)을 다 기각해도 출처를 못 찾았다.
            "tts_synth_s": round(tts_synth_s, 3),
            "tts_play_s": round(tts_play_s, 3),     # 봇이 말하는 시간(지연 아님)
            "resp_compute_s": resp,           # 연산만(꼬리 제외). 옛 기록과 이어보기용
            "filler_wait_s": round(filler_wait_s, 3),
            # 선행 생각 결과: hit(그대로 씀) / no_guess(추측 자체가 없었다) /
            # different(아이가 말을 이어가 키가 안 맞음) / failed / vision.
            # 🔴 08-19 이후 '발동 0회'인 걸 여태 몰랐다 — 남기는 데가 없었다.
            "spec": spec,
            "resp_felt_s": felt,              # ★ 아이가 겪는 시간 = 꼬리 + 연산 + 필러대기
            "metric_ver": 3,                  # 1 = tts 재생시간 혼입 / 2 = 꼬리 누락
            # 🔴 자원은 **구간 최댓값**이다. 여기서 한 번 읽으면 턴이 끝나 조용해진
            #    순간을 찍게 되고, 그게 2026-08-19 의 'GPU 0%' 오진이었다.
            #    표본이 하나도 없는 아주 짧은 턴만 지금 값으로 채운다.
            **_sys_stats_now(),
            **self._sampler.pop(),
            "reply_len": len(reply or ""),
            "expansion_delta": (len(_words(reply)) - len(child_words)
                                if child_words else None),
            "reuse": _reuse_ratio(child_text, reply),
        }
        self.samples.append(rec)
        self._write(rec)
        log.info("[계측] 턴%d 체감 %.2fs (꼬리 %.2f + 인식 %.2f + 생각 %.2f"
                 " + TTS %.2f) | 발화 %.2fs | RSS %sMB GPU %s%%",
                 self.turn, felt, vad_tail_s, stt_rec_s, think_s, tts_first_s,
                 tts_play_s, rec["rss_mb"], rec.get("gpu_pct"))

    def record_realtime_turn(self, *, kind: str, perceived_s: float | None,
                             transcribe_s: float | None, respond_first_s: float | None,
                             filler: bool, reply: str, child_text: str,
                             cost_usd: float | None, cached_tokens: int,
                             safety: list[str], game_missing: list[str]) -> None:
        """전면 API(Realtime) 한 턴. 로컬 필드와 이름이 달라 samples 에는 안 넣는다.

        perceived_s     : 서버가 말끝을 잡은 순간 → 첫 답 소리. 로컬 resp_felt_s 와 비교
                          (둘 다 무음 대기 1.2초를 포함한다).
        transcribe_s    : 말끝 → 받아 적기 완료. 턴을 우리가 쥐는 값(spec R1).
        respond_first_s : response.create → 첫 오디오 조각.
        """
        if not self.enabled:
            return
        self.turn += 1

        def r(x):
            return None if x is None else round(float(x), 3)

        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "tag": self.tag,
               "event": "rt_turn", "metric_ver": "rt1", "turn": self.turn, "kind": kind,
               "perceived_s": r(perceived_s), "transcribe_s": r(transcribe_s),
               "respond_first_s": r(respond_first_s), "filler": bool(filler),
               "cost_usd": None if cost_usd is None else round(cost_usd, 6),
               "cached_tokens": int(cached_tokens), "safety": list(safety),
               "game_missing": list(game_missing), "reply_len": len(reply or ""),
               "child_text": child_text, "reply": reply,
               **_sys_stats_now(), **self._sampler.pop()}
        self._write(rec)
        log.info("[계측] rt턴%d %s 체감 %s (받아적기 %s / 요청→첫소리 %s)%s | $%s | RSS %sMB",
                 self.turn, kind, rec["perceived_s"], rec["transcribe_s"],
                 rec["respond_first_s"], " +맞장구" if filler else "",
                 rec["cost_usd"], rec.get("rss_mb"))

    def summary(self) -> None:
        """세션 요약(중앙값/p90/최대 메모리)을 출력하고 파일에 남긴다."""
        if not self.enabled or not self.samples:
            return
        # 콜드 턴은 대표값에서 빼서 웜 성능을 본다(콜드는 따로 표시).
        warm = [s for s in self.samples if not s["cold"]] or self.samples
        resp = [s["resp_compute_s"] for s in warm]
        felt = [s["resp_felt_s"] for s in warm]
        tail = [s["vad_tail_s"] for s in warm]
        rec_ = [s["stt_rec_s"] for s in warm]
        think = [s["think_s"] for s in warm]
        tts = [s["tts_first_s"] for s in warm]
        play = [s["tts_play_s"] for s in warm]
        rss = [s["rss_mb"] for s in self.samples if s["rss_mb"] is not None]

        def _peak(key):
            xs = [s[key] for s in self.samples if s.get(key) is not None]
            return max(xs) if xs else None

        # 🔴 목표 판정은 **체감(felt)** 으로 한다. 연산만으로 재면 꼬리만큼
        #    후하게 나온다 — metric_ver 2 까지가 그 상태였다.
        summ = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tag": self.tag, "event": "summary", "metric_ver": 3,
            "turns": len(self.samples),
            "resp_felt_median_s": round(statistics.median(felt), 3),
            "resp_felt_p90_s": round(_p90(felt), 3),
            "resp_median_s": round(statistics.median(resp), 3),
            "resp_p90_s": round(_p90(resp), 3),
            "vad_tail_median_s": round(statistics.median(tail), 3),
            "stt_rec_median_s": round(statistics.median(rec_), 3),
            "think_median_s": round(statistics.median(think), 3),
            "tts_first_median_s": round(statistics.median(tts), 3),
            "tts_play_median_s": round(statistics.median(play), 3),
            "rss_peak_mb": max(rss) if rss else None,
            "sys_mem_peak_mb": _peak("sys_mem_used_mb"),
            "sys_mem_total_mb": _peak("sys_mem_total_mb"),
            "cpu_peak_pct": _peak("cpu_pct"),
            "gpu_peak_pct": _peak("gpu_pct"),
            "temp_peak_c": _peak("temp_c"),
            "under_1_5s_ratio": (round(sum(1 for r in felt if r <= 1.5) / len(felt), 2)
                                 if felt else None),
            "under_3s_ratio": (round(sum(1 for r in felt if r <= 3.0) / len(felt), 2)
                               if felt else None),
        }
        self._write(summ)
        print("\n===== 세션 계측 요약 (warm 기준) =====")
        print(f"  턴 수: {summ['turns']}  (환경 tag={self.tag}, 파일={self.path.name})")
        print(f"  ▶ 체감 지연(말끝→첫 소리)  중앙값 {summ['resp_felt_median_s']:.2f}s"
              f" / p90 {summ['resp_felt_p90_s']:.2f}s")
        print(f"      목표 1.5s 이내 {int((summ['under_1_5s_ratio'] or 0)*100)}%"
              f"  |  상한 3.0s 이내 {int((summ['under_3s_ratio'] or 0)*100)}%")
        print(f"    단계  VAD꼬리 {summ['vad_tail_median_s']:.2f}s"
              f" | STT인식 {summ['stt_rec_median_s']:.2f}s"
              f" | 생각 {summ['think_median_s']:.2f}s"
              f" | TTS {summ['tts_first_median_s']:.2f}s")
        print(f"    (참고) 봇이 말하는 시간 {summ['tts_play_median_s']:.2f}s"
              " — 지연이 아니라 발화 길이다")
        # 🔴 선행 생각은 터지면 '생각'을 통째로 0 으로 만든다(-1.75s). 몇 번 터졌는지
        #    안 세면 켜 뒀는지조차 모른다 — 실제로 08-19~08-27 내내 0회인 걸 몰랐다.
        specs = [s.get("spec") for s in self.samples if s.get("spec")]
        if specs:
            from collections import Counter
            c = Counter(specs)
            rest = " / ".join(f"{k} {v}" for k, v in c.most_common() if k != "hit")
            print(f"    선행생각 발동 {c['hit']}/{len(specs)}턴"
                  + (f"  (못 쓴 이유: {rest})" if rest else ""))
        res = []
        if summ["rss_peak_mb"]:
            res.append(f"RSS {summ['rss_peak_mb']:.0f}MB")
        if summ["sys_mem_peak_mb"]:
            tot = summ.get("sys_mem_total_mb")
            # 8GB 는 젯슨 예산이다. 노트북에도 그렇게 찍으면 거짓말이 되므로
            # 총량은 그 기계에서 읽은 값을 쓴다.
            res.append(f"시스템메모리 {summ['sys_mem_peak_mb']:.0f}"
                       + (f"/{tot:.0f}MB" if tot else "MB"))
        if summ["cpu_peak_pct"] is not None:
            res.append(f"CPU {summ['cpu_peak_pct']:.0f}%")
        if summ["gpu_peak_pct"] is not None:
            res.append(f"GPU {summ['gpu_peak_pct']:.0f}%")
        if summ["temp_peak_c"] is not None:
            res.append(f"{summ['temp_peak_c']:.0f}°C")
        if res:
            print("  최대 자원  " + " | ".join(res))
        print("====================================\n")
