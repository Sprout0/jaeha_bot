"""configs/local.yaml 오버레이 — 노트북·젯슨이 같은 파일을 공유해도 안 깨지게.

배경: push_code.sh 가 configs/ 를 통째로 젯슨에 밀어넣으므로 model_paths.yaml 은
한 벌뿐이다. 그런데 stt.device 는 젯슨=cuda / 노트북=cpu 로 **정반대**여야 하고
(노트북 ctranslate2 는 CPU 전용 휠이라 cuda 로 두면 죽는다), metrics.tag 도
기계마다 달라야 로그가 구분된다. 그래서 기계별 local.yaml 을 덧씌운다.
"""
import textwrap

from app.config import deep_merge, load_models


def _write(d, name, text):
    (d / name).write_text(textwrap.dedent(text), encoding="utf-8")


# ── deep_merge: '덮어쓰기'가 아니라 '깊은 병합'이어야 한다 ──────────────────────
def test_nested_override_keeps_siblings():
    """stt.device 만 바꿔도 stt 의 나머지 20여 개 값이 살아 있어야 한다.

    얕은 병합(dict.update)이면 stt 통째로 교체돼 model_size·가드 임계까지 사라진다.
    """
    base = {"stt": {"device": "cuda", "model_size": "large-v3-turbo", "cpu_threads": 6}}
    deep_merge(base, {"stt": {"device": "cpu"}})
    assert base["stt"] == {"device": "cpu", "model_size": "large-v3-turbo",
                           "cpu_threads": 6}


def test_merges_multiple_sections_independently():
    base = {"stt": {"device": "cuda"}, "metrics": {"tag": "jetson", "enabled": True}}
    deep_merge(base, {"stt": {"device": "cpu"}, "metrics": {"tag": "pc"}})
    assert base["stt"]["device"] == "cpu"
    assert base["metrics"] == {"tag": "pc", "enabled": True}


def test_scalar_replaces_dict_and_list_replaces_list():
    """타입이 다르면 병합하지 않고 교체한다(리스트는 '합치기'가 아니라 '교체')."""
    base = {"tts": {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}}
    deep_merge(base, {"tts": {"providers": ["CPUExecutionProvider"]}})
    assert base["tts"]["providers"] == ["CPUExecutionProvider"]


def test_new_key_is_added():
    base = {"stt": {"device": "cuda"}}
    deep_merge(base, {"wake": {"enabled": False}})
    assert base["wake"] == {"enabled": False}


# ── load_models: 파일이 있을 때만 덧씌운다 ────────────────────────────────────
def test_without_local_yaml_base_is_untouched(tmp_path):
    """젯슨에는 local.yaml 이 없다 — 그때 운영 설정이 그대로 나와야 한다."""
    _write(tmp_path, "model_paths.yaml", """
        stt:
          device: cuda
          model_size: large-v3-turbo
        metrics:
          tag: jetson
    """)
    cfg = load_models(tmp_path)
    assert cfg["stt"]["device"] == "cuda"
    assert cfg["metrics"]["tag"] == "jetson"


def test_local_yaml_overrides_only_listed_keys(tmp_path):
    """노트북 시나리오 — device 와 tag 만 뒤집고 나머지는 운영값을 따른다."""
    _write(tmp_path, "model_paths.yaml", """
        stt:
          device: cuda
          model_size: large-v3-turbo
          cpu_threads: 6
        metrics:
          tag: jetson
          enabled: true
    """)
    _write(tmp_path, "local.yaml", """
        stt:
          device: cpu
        metrics:
          tag: pc
    """)
    cfg = load_models(tmp_path)
    assert cfg["stt"]["device"] == "cpu"
    assert cfg["stt"]["model_size"] == "large-v3-turbo"  # 안 건드린 값은 유지
    assert cfg["stt"]["cpu_threads"] == 6
    assert cfg["metrics"] == {"tag": "pc", "enabled": True}


def test_empty_local_yaml_does_not_crash(tmp_path):
    """주석만 남기고 비운 local.yaml 은 safe_load 가 None 을 준다 — 터지면 안 된다."""
    _write(tmp_path, "model_paths.yaml", "stt:\n  device: cuda\n")
    _write(tmp_path, "local.yaml", "# 아직 덮어쓸 게 없음\n")
    assert load_models(tmp_path)["stt"]["device"] == "cuda"
