#!/usr/bin/env python3
"""
Summarize multi-seed evaluation replicates for Tool-set Playground v3.

This is meant for paper-quality reporting: mean/std and 95% CI (normal approx)
over seeds for stochastic samplers (primarily MD).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Metrics:
    seed: int
    k1_f1: float
    passk: float
    oracle: float


def _read_metrics(path: str, block: Dict[str, Any], k: int) -> Metrics:
    meta = block.get("sampler", {})
    # seed lives in top-level meta
    with open(path, "r") as f:
        top = json.load(f)
    seed = int(top.get("meta", {}).get("seed", -1))
    overall = block["overall"]
    return Metrics(
        seed=seed,
        k1_f1=float(overall["k1_tool_f1"]),
        passk=float(overall["passk_recall"][str(k)]),
        oracle=float(overall["oracle_f1"][str(k)]),
    )


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _ci95(m: float, s: float, n: int) -> Tuple[float, float]:
    if n <= 1:
        return m, m
    half = 1.96 * (s / math.sqrt(n))
    return m - half, m + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pattern", default="*.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    # ignore non-result jsons if any
    paths = [p for p in paths if os.path.basename(p).startswith("rep_")]
    if not paths:
        raise SystemExit(f"No replicate jsons found in {args.in_dir} (pattern={args.pattern})")

    ar_ms: List[Metrics] = []
    md_ms: List[Metrics] = []
    for p in paths:
        with open(p, "r") as f:
            j = json.load(f)
        ar_ms.append(_read_metrics(p, j["ar"], args.k))
        md_ms.append(_read_metrics(p, j["md"], args.k))

    def pack(name: str, ms: List[Metrics]) -> Dict[str, Any]:
        k1s = [m.k1_f1 for m in ms]
        passks = [m.passk for m in ms]
        oracles = [m.oracle for m in ms]
        k1_m, k1_s = _mean_std(k1s)
        pass_m, pass_s = _mean_std(passks)
        ora_m, ora_s = _mean_std(oracles)
        k1_lo, k1_hi = _ci95(k1_m, k1_s, len(ms))
        pass_lo, pass_hi = _ci95(pass_m, pass_s, len(ms))
        ora_lo, ora_hi = _ci95(ora_m, ora_s, len(ms))
        return {
            "name": name,
            "n": len(ms),
            "k1": (k1_m, k1_s, k1_lo, k1_hi),
            "passk": (pass_m, pass_s, pass_lo, pass_hi),
            "oracle": (ora_m, ora_s, ora_lo, ora_hi),
        }

    ar = pack("AR", ar_ms)
    md = pack("MD", md_ms)

    lines: List[str] = []
    lines.append("# Tool-set Playground v3 (Full Test) Multi-seed Summary")
    lines.append("")
    lines.append(f"- Dir: `{args.in_dir}`")
    lines.append(f"- Replicates: {len(paths)} seeds")
    lines.append(f"- Metrics at K={args.k}: `Pass@K` (tool recall) and `Oracle@K` (best-of-K tool-set F1)")
    lines.append("")
    lines.append("## Aggregate (mean ± std; 95% CI)")
    lines.append("")
    lines.append("| Model | n | k1F1 | Pass@K | Oracle@K |")
    lines.append("|---|---:|---:|---:|---:|")
    for blk in [ar, md]:
        k1_m, k1_s, k1_lo, k1_hi = blk["k1"]
        pass_m, pass_s, pass_lo, pass_hi = blk["passk"]
        ora_m, ora_s, ora_lo, ora_hi = blk["oracle"]
        lines.append(
            f"| {blk['name']} | {blk['n']} | {k1_m:.3f} ± {k1_s:.3f} [{k1_lo:.3f},{k1_hi:.3f}]"
            f" | {pass_m:.3f} ± {pass_s:.3f} [{pass_lo:.3f},{pass_hi:.3f}]"
            f" | {ora_m:.3f} ± {ora_s:.3f} [{ora_lo:.3f},{ora_hi:.3f}] |"
        )
    lines.append("")
    lines.append("## Per-seed (for debugging)")
    lines.append("")
    lines.append("| seed | AR k1F1 | AR Pass@K | AR Oracle@K | MD k1F1 | MD Pass@K | MD Oracle@K |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for a, m in sorted(zip(ar_ms, md_ms), key=lambda t: t[0].seed):
        lines.append(
            f"| {a.seed} | {a.k1_f1:.3f} | {a.passk:.3f} | {a.oracle:.3f}"
            f" | {m.k1_f1:.3f} | {m.passk:.3f} | {m.oracle:.3f} |"
        )

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
