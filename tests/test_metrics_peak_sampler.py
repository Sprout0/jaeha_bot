"""자원 지표는 **구간 최댓값**이어야 한다 — 스냅샷 한 장이 아니라.

🔴 2026-08-19 젯슨에서 실제로 겪은 오진:
     세션 요약  "최대 자원  RSS 5632MB | CPU 18% | GPU 0%"
   이걸 보고 "GPU 를 안 쓰는구나" 라고 판단했는데, 같은 시각 sysfs 를 직접 훑으니
   추론 중 GPU load 가 **874~999 per-mille(87~100%)** 였다. GPU 는 잘 돌고 있었다.

   원인은 재는 **시점**이었다. record_turn() 은 턴이 다 끝난 뒤 호출되고
   (같은 기록에 tts_play_s 가 들어 있다 = 재생까지 끝났다), 그 안에서 _sys_stats() 를
   딱 한 번 읽었다. 즉 **그 턴에서 가장 조용한 순간을 찍고 '최대'라고 불렀다.**
   CPU 18% 도 같은 병이다 — psutil 은 '직전 호출 이후 평균'이라 아이가 말하기를
   기다린 시간까지 섞여 희석됐다.

⚠️ 이 종류의 오류는 숫자가 그럴듯해서 안 보인다. 0% 가 아니라 40% 로 찍혔다면
   아무도 의심하지 않았을 것이다. 자원 지표를 새로 추가할 때는 반드시
   '언제 재는가'를 먼저 정할 것.
"""
from app.metrics import _PeakSampler


def test_keeps_the_maximum_not_the_last_reading():
    s = _PeakSampler()

    s._observe({"gpu_pct": 99.9, "cpu_pct": 80.0})
    s._observe({"gpu_pct": 0.0, "cpu_pct": 3.0})     # 턴이 끝나 조용해진 순간

    peak = s.pop()
    assert peak["gpu_pct"] == 99.9, "마지막 관측으로 덮어썼다 — 이게 GPU 0% 의 정체다"
    assert peak["cpu_pct"] == 80.0


def test_none_readings_never_erase_a_real_peak():
    """노트북엔 GPU sysfs 가 없어 None 이 온다. None 이 최댓값을 지우면 안 된다."""
    s = _PeakSampler()

    s._observe({"gpu_pct": 55.0})
    s._observe({"gpu_pct": None})

    assert s.pop()["gpu_pct"] == 55.0


def test_pop_starts_a_fresh_window_for_the_next_turn():
    """턴마다 새 구간이어야 한다 — 안 비우면 첫 턴 최고치가 영원히 따라다닌다."""
    s = _PeakSampler()

    s._observe({"gpu_pct": 99.0})
    s.pop()
    s._observe({"gpu_pct": 12.0})

    assert s.pop()["gpu_pct"] == 12.0


def test_pop_on_an_empty_window_is_empty_not_an_error():
    assert _PeakSampler().pop() == {}
