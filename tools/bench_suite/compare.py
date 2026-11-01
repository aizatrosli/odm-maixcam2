"""Merge MaixCAM2 benchmark-suite results into one table.

    python3 tools/bench_suite/compare.py results/*.json            # table on stdout
    python3 tools/bench_suite/compare.py results/*.json --csv all.csv

One row per (run, build). Runs with a different schema are skipped with a
warning, so old and new files can sit in the same folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

SCHEMA = 1
COLUMNS = [
    ("run", lambda run, r: run["board_start"].get("host", "?") + " " + run["started"]),
    ("build", lambda run, r: r["name"]),
    ("ms", lambda run, r: r.get("idle", {}).get("median_ms")),
    ("p99_ms", lambda run, r: r.get("idle", {}).get("p99_ms")),
    ("hz", lambda run, r: r.get("idle", {}).get("hz")),
    ("spread_%", lambda run, r: r.get("idle", {}).get("round_spread_pct")),
    ("cam_ms", lambda run, r: r.get("camera", {}).get("median_ms")),
    ("e2e_ms", lambda run, r: r.get("e2e", {}).get("total", {}).get("p50_ms")),
    ("cos", lambda run, r: r.get("fidelity", {}).get("cos_mean")),
    ("cos_min", lambda run, r: r.get("fidelity", {}).get("cos_min")),
    ("cmm_mb", lambda run, r: r.get("cmm_mb")),
    ("load_ms", lambda run, r: r.get("load_ms")),
    ("loop_overruns", lambda run, r: r.get("loop", {}).get("overruns")),
    ("loop_late_p99", lambda run, r: r.get("loop", {}).get("wake_late", {}).get("p99_ms")),
    ("stack_hz", lambda run, r: r.get("stack", {}).get("yolo_plus_encoder_hz")),
    ("temp_end", lambda run, r: r.get("temp_end_c")),
    ("error", lambda run, r: r.get("error", "")),
]


def rows(paths):
    for p in paths:
        run = json.load(open(p))
        if run.get("schema") != SCHEMA:
            print(f"skip {p}: schema {run.get('schema')} != {SCHEMA}", file=sys.stderr)
            continue
        for r in run["results"]:
            yield [f(run, r) for _, f in COLUMNS]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1.5 and v != 0 else f"{v:.1f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sort", default="ms", help="column to sort by")
    a = ap.parse_args()
    data = list(rows(a.results))
    names = [c for c, _ in COLUMNS]
    k = names.index(a.sort)
    data.sort(key=lambda r: (r[k] is None, r[k] if r[k] is not None else 0))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(data)
    show = [n for n in names if n != "run"] if len({r[0] for r in data}) <= 1 else names
    idx = [names.index(n) for n in show]
    table = [show] + [[fmt(r[i]) for i in idx] for r in data]
    widths = [max(len(row[j]) for row in table) for j in range(len(show))]
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
