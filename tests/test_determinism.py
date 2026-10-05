import json
from pathlib import Path

from lancepilot.benchmark import load_config, run_benchmark
from lancepilot.workload import build_workload

ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_same_trace() -> None:
    cfg = load_config(ROOT / "configs" / "mixed_size.yaml")
    a = build_workload(cfg["workload"], cfg["objects"])
    b = build_workload(cfg["workload"], cfg["objects"])
    assert a.objects == b.objects and a.trace == b.trace


def test_different_seed_different_trace() -> None:
    cfg = load_config(ROOT / "configs" / "phase_shift.yaml")
    a = build_workload(cfg["workload"], cfg["objects"])
    b = build_workload({**cfg["workload"], "seed": 999}, cfg["objects"])
    assert a.trace != b.trace


def test_benchmark_end_to_end(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "configs" / "phase_shift.yaml")
    cfg["workload"] = {**cfg["workload"], "num_queries": 2000, "shift_at": 1000}
    cfg["metrics"] = {**cfg["metrics"], "steady_window": 500}
    r1 = run_benchmark(cfg, tmp_path / "a")
    run_benchmark(cfg, tmp_path / "b")
    assert set(r1["policies"]) == {"LRU", "LancePilot"}
    for pol in r1["policies"].values():
        assert pol["queries"] == 2000
        assert pol["max_used_bytes"] <= r1["capacity_bytes"]
        assert "recovery_queries" in pol["phase_shift"]
    assert (tmp_path / "a" / "phase_shift_rolling_hit_ratio.png").exists()
    j1 = json.loads((tmp_path / "a" / "phase_shift.json").read_text())
    j2 = json.loads((tmp_path / "b" / "phase_shift.json").read_text())
    assert j1 == j2  # fully deterministic results
