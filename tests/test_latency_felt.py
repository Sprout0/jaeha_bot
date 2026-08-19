"""체감 지연(말끝 → 첫 소리)과 자원 계측. 모델·마이크·젯슨 불필요.

🔴 왜 만들었나 — `resp_compute_s` 는 '말끝부터 첫 소리까지'라고 적혀 있었지만
   실제로는 **VAD 꼬리 대기를 뺀** 값이었다. 꼬리는 `stt_wait_s` 안에서 아이가
   말한 시간과 섞여 있어 아무도 볼 수 없었다. 2026-08-09 의 'tts_s 에 발화
   시간이 섞여 있던' 실수와 같은 종류다 — 이름과 내용이 어긋났다.

   그 결과 `under_3s_ratio`(3초 목표 달성률)가 꼬리만큼 부풀려져 있었다.
"""
import json

from app.metrics import MetricsLogger, _sys_stats_now


def _rec(m, **kw):
    base = dict(vad_tail_s=1.2, stt_wait_s=3.0, stt_rec_s=1.6, think_s=0.9,
                think_kind="llm", tts_first_s=0.9, tts_play_s=2.0)
    base.update(kw)
    m.record_turn(**base)
    return m.samples[-1]


def test_felt_latency_includes_vad_tail(tmp_path):
    """체감 지연은 꼬리 대기를 포함해야 한다 — 아이는 그 시간에도 기다린다."""
    m = MetricsLogger(log_dir=str(tmp_path))
    r = _rec(m)
    assert r["resp_felt_s"] == 4.6, "1.2+1.6+0.9+0.9 이어야 한다"
    assert r["vad_tail_s"] == 1.2, "꼬리는 따로도 보여야 어디를 깎을지 안다"


def test_compute_only_metric_is_kept_for_continuity(tmp_path):
    """옛 지표는 남긴다 — 08-09~08-13 기록과 이어서 봐야 한다."""
    m = MetricsLogger(log_dir=str(tmp_path))
    r = _rec(m)
    assert r["resp_compute_s"] == 3.4, "1.6+0.9+0.9 (꼬리 제외) 는 그대로"
    assert r["resp_felt_s"] > r["resp_compute_s"]


def test_metric_version_bumped(tmp_path):
    """지표가 바뀌었으면 판을 올려 옛 수치와 직접 비교하지 않게 한다."""
    m = MetricsLogger(log_dir=str(tmp_path))
    assert _rec(m)["metric_ver"] == 3


def test_targets_are_judged_on_felt_not_compute(tmp_path, capsys):
    """목표 1.5s / 상한 3.0s 판정은 **체감** 기준이어야 한다.

    꼬리를 빼고 재면 3.4s 짜리 턴이 '3초 안'으로 보인다 — 그게 지금까지의 상태였다.
    """
    m = MetricsLogger(log_dir=str(tmp_path))
    for _ in range(2):
        _rec(m, vad_tail_s=1.2, stt_rec_s=1.6, think_s=0.9, tts_first_s=0.9)  # 체감 4.6s
    m.summary()
    summ = json.loads(tmp_path.joinpath(m.path.name).read_text(
        encoding="utf-8").strip().splitlines()[-1])
    assert summ["resp_felt_median_s"] == 4.6
    assert summ["under_3s_ratio"] == 0.0, "4.6s 는 상한 3초를 넘는다"
    assert summ["under_1_5s_ratio"] == 0.0
    assert summ["vad_tail_median_s"] == 1.2
    out = capsys.readouterr().out
    assert "1.5" in out and "3" in out, "두 목표가 다 보여야 한다"


def test_fast_turn_counts_toward_both_targets(tmp_path):
    m = MetricsLogger(log_dir=str(tmp_path))
    for _ in range(2):
        _rec(m, vad_tail_s=0.3, stt_rec_s=0.4, think_s=0.3, tts_first_s=0.4)  # 1.4s
    m.summary()
    summ = json.loads(tmp_path.joinpath(m.path.name).read_text(
        encoding="utf-8").strip().splitlines()[-1])
    assert summ["under_1_5s_ratio"] == 1.0 and summ["under_3s_ratio"] == 1.0


def test_sys_stats_never_raises_and_has_keys():
    """자원 계측은 젯슨에만 있는 값을 읽는다. 노트북에서는 None 이어도 죽지 않아야 한다."""
    s = _sys_stats_now()
    for k in ("cpu_pct", "sys_mem_used_mb", "gpu_pct", "temp_c"):
        assert k in s, f"{k} 가 없다"


def test_turn_record_carries_sys_stats(tmp_path):
    m = MetricsLogger(log_dir=str(tmp_path))
    r = _rec(m)
    assert "cpu_pct" in r and "gpu_pct" in r, "턴마다 자원도 남겨야 추세를 본다"
