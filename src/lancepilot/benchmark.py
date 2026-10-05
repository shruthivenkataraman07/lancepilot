"""Benchmark runner: runs LRU and LancePilot on the identical synthetic trace,
writes JSON results and charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lancepilot.metrics import phase_shift_metrics, rolling_hit_ratio, summarize  # noqa: E402
from lancepilot.policies import CachePolicy, LancePilotPolicy, LRUPolicy  # noqa: E402
from lancepilot.simulator import SimResult, simulate  # noqa: E402
from lancepilot.workload import Workload, build_workload  # noqa: E402

LABEL = "Measured locally on synthetic workload"
COLORS = {"LRU": "#2a78d6", "LancePilot": "#eb6834"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for key in ("name", "workload"):
        if key not in cfg:
            raise ValueError(f"config {path} missing required key '{key}'")
    return cfg


def capacity_for(workload: Workload, cache_cfg: dict[str, Any]) -> int:
    if "capacity_bytes" in cache_cfg:
        return int(cache_cfg["capacity_bytes"])
    total = sum(o.size_bytes for o in workload.objects)
    return int(total * float(cache_cfg.get("capacity_fraction", 0.05)))


def make_policies(workload: Workload, cfg: dict[str, Any]) -> list[CachePolicy]:
    capacity = capacity_for(workload, cfg.get("cache", {}))
    median = float(np.median([o.size_bytes for o in workload.objects]))
    lp = cfg.get("lancepilot", {})
    return [
        LRUPolicy(capacity),
        LancePilotPolicy(
            capacity,
            median_object_size=median,
            alpha=float(lp.get("alpha", 0.01)),
            epsilon=float(lp.get("epsilon", 0.05)),
        ),
    ]


def run_benchmark(cfg: dict[str, Any], out_dir: str | Path = "results") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workload = build_workload(cfg["workload"], cfg.get("objects", {}))
    hit_ms = float(cfg.get("cache", {}).get("hit_latency_ms", 0.1))
    mcfg = cfg.get("metrics", {})
    results: list[SimResult] = []
    report: dict[str, Any] = {
        "experiment": cfg["name"],
        "label": LABEL,
        "workload": workload.meta,
        "num_objects": len(workload.objects),
        "num_queries": len(workload.trace),
        "dataset_bytes": sum(o.size_bytes for o in workload.objects),
        "policies": {},
    }
    for policy in make_policies(workload, cfg):
        res = simulate(policy, workload.objects, workload.trace, hit_ms)
        results.append(res)
        summary = summarize(res)
        if isinstance(policy, LancePilotPolicy):
            summary["rejected_admissions"] = policy.rejected_admissions
        if workload.meta["type"] == "phase_shift":
            summary["phase_shift"] = phase_shift_metrics(
                res.hits,
                int(workload.meta["shift_at"]),
                int(mcfg.get("rolling_window", 200)),
                int(mcfg.get("steady_window", 2000)),
                float(mcfg.get("recovery_threshold", 0.95)),
            )
        report["policies"][res.policy] = summary
    report["capacity_bytes"] = results[0].capacity_bytes
    (out / f"{cfg['name']}.json").write_text(json.dumps(report, indent=2))
    if workload.meta["type"] == "phase_shift":
        plot_phase_shift(
            results,
            int(workload.meta["shift_at"]),
            int(mcfg.get("rolling_window", 200)),
            out / f"{cfg['name']}_rolling_hit_ratio.png",
        )
    return report


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_phase_shift(results: list[SimResult], shift_at: int, window: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for res in results:
        roll = rolling_hit_ratio(res.hits, window)
        x = np.arange(1, roll.size + 1)
        ax.plot(x, roll, color=COLORS[res.policy], linewidth=2, label=res.policy)
        ax.annotate(
            res.policy, (x[-1], roll[-1]), xytext=(4, 0), textcoords="offset points",
            color=TEXT, fontsize=9, va="center",
        )
    ax.axvline(shift_at, color=MUTED, linestyle="--", linewidth=1)
    ax.text(shift_at, 1.02, " hot set shifts", color=MUTED, fontsize=9, va="bottom")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Query number", color=TEXT)
    ax.set_ylabel(f"Rolling hit ratio ({window}-query window)", color=TEXT)
    ax.set_title(f"Phase shift: rolling hit ratio\n{LABEL}", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(reports: list[dict[str, Any]], path: Path) -> None:
    """Two panels (p99 latency, remote MiB read): one scale per panel, no dual axis."""
    names = [r["experiment"] for r in reports]
    policies = ["LRU", "LancePilot"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    panels = [
        ("p99_ms", "p99 latency (ms)", 1.0),
        ("remote_bytes", "Remote data read (MiB)", 1 / (1024 * 1024)),
    ]
    x = np.arange(len(names))
    width = 0.38
    for ax, (key, title, scale) in zip(axes, panels, strict=True):
        for j, pol in enumerate(policies):
            vals = [r["policies"][pol][key] * scale for r in reports]
            bars = ax.bar(
                x + (j - 0.5) * width, vals, width - 0.02, color=COLORS[pol], label=pol,
                edgecolor="white", linewidth=1,
            )
            ax.bar_label(bars, fmt="%.0f", fontsize=8, color=TEXT, padding=2)
        ax.set_xticks(x, names)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"LRU vs LancePilot (lower is better)\n{LABEL}", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_report(report: dict[str, Any]) -> str:
    rows = [
        ("p50 latency (ms)", "p50_ms", "{:.2f}"),
        ("p95 latency (ms)", "p95_ms", "{:.2f}"),
        ("p99 latency (ms)", "p99_ms", "{:.2f}"),
        ("mean latency (ms)", "mean_ms", "{:.2f}"),
        ("hit ratio", "hit_ratio", "{:.4f}"),
        ("remote requests", "remote_requests", "{:,}"),
        ("remote MiB read", "remote_bytes", None),
        ("evictions", "evictions", "{:,}"),
    ]
    lru, lp = report["policies"]["LRU"], report["policies"]["LancePilot"]
    lines = [
        f"== {report['experiment']} ({LABEL}) ==",
        f"objects={report['num_objects']} queries={report['num_queries']} "
        f"capacity={report['capacity_bytes'] / 2**20:.1f} MiB "
        f"dataset={report['dataset_bytes'] / 2**20:.1f} MiB seed={report['workload']['seed']}",
        f"{'metric':<22}{'LRU':>14}{'LancePilot':>14}",
    ]
    for label, key, fmt in rows:
        if fmt is None:
            a, b = f"{lru[key] / 2**20:,.1f}", f"{lp[key] / 2**20:,.1f}"
        else:
            a, b = fmt.format(lru[key]), fmt.format(lp[key])
        lines.append(f"{label:<22}{a:>14}{b:>14}")
    if "phase_shift" in lru:
        for label, key in (
            ("pre-shift hit ratio", "pre_shift_hit_ratio"),
            ("1st window post-shift", "first_window_after_shift_hit_ratio"),
            ("post-shift steady", "post_shift_steady_hit_ratio"),
            ("recovery (queries)", "recovery_queries"),
        ):
            a, b = lru["phase_shift"][key], lp["phase_shift"][key]
            fa = "never" if a is None else (f"{a:.4f}" if isinstance(a, float) else str(a))
            fb = "never" if b is None else (f"{b:.4f}" if isinstance(b, float) else str(b))
            lines.append(f"{label:<22}{fa:>14}{fb:>14}")
    return "\n".join(lines)
