"""Command line entry point.

    python -m lancepilot.cli benchmark --config configs/phase_shift.yaml
    python -m lancepilot.cli report --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lancepilot.benchmark import format_report, load_config, plot_comparison, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lancepilot")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="run LRU and LancePilot on one config")
    b.add_argument("--config", required=True, nargs="+")
    b.add_argument("--out", default="results")
    r = sub.add_parser("report", help="build the p99/remote-bytes chart from result JSONs")
    r.add_argument("--results", default="results")
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        for path in args.config:
            report = run_benchmark(load_config(path), args.out)
            print(format_report(report))
            print(f"wrote {Path(args.out) / (report['experiment'] + '.json')}\n")
    else:
        out = Path(args.results)
        reports = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
        if not reports:
            parser.error(f"no result JSON files in {out}")
        chart = out / "p99_and_remote_bytes.png"
        plot_comparison(reports, chart)
        print(f"wrote {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
