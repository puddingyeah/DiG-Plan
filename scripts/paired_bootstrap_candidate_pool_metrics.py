#!/usr/bin/env python3
"""
Paired bootstrap deltas for candidate-pool diagnostics.

This script compares two candidate pools on shared sample_ids and reports paired
bootstrap deltas for sample-level metrics:
  - has_exact_tool: whether any candidate has ToolF1==1
  - has_exact_and_full_edge: whether any candidate has ToolF1==1 and EdgeRec==1
  - best_edge: best edge recall among all candidates for the sample
  - best_edge_given_exact: best edge recall among candidates with ToolF1==1
                          (missing values are imputed as 0 for paired comparison)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = data.get("candidates", [])
    if not isinstance(cands, list):
        raise ValueError(f"Invalid candidates format: {path}")
    return cands


def _sample_metrics(cands: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_sid[str(c.get("sample_id"))].append(c)

    out: Dict[str, Dict[str, float]] = {}
    for sid, xs in by_sid.items():
        has_exact = any(float(x.get("tool_f1", 0.0)) == 1.0 for x in xs)
        has_exact_and_full = any(
            (float(x.get("tool_f1", 0.0)) == 1.0 and float(x.get("edge_recall", 0.0)) == 1.0)
            for x in xs
        )
        best_edge = max(float(x.get("edge_recall", 0.0)) for x in xs) if xs else 0.0
        edge_given_exact = [float(x.get("edge_recall", 0.0)) for x in xs if float(x.get("tool_f1", 0.0)) == 1.0]
        best_edge_given_exact = max(edge_given_exact) if edge_given_exact else 0.0

        out[sid] = {
            "has_exact_tool": 1.0 if has_exact else 0.0,
            "has_exact_and_full_edge": 1.0 if has_exact_and_full else 0.0,
            "best_edge": best_edge,
            "best_edge_given_exact": best_edge_given_exact,
        }
    return out


def _paired_bootstrap_delta(
    ids: List[str],
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    diffs = np.array([a[sid][metric] - b[sid][metric] for sid in ids], dtype=np.float64)
    mean_delta = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot = []
    n = len(ids)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(diffs[idx])))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return mean_delta, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap deltas for candidate-pool metrics")
    ap.add_argument("--a_name", required=True)
    ap.add_argument("--a_path", required=True)
    ap.add_argument("--b_name", required=True)
    ap.add_argument("--b_path", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    a = _sample_metrics(_load(Path(args.a_path)))
    b = _sample_metrics(_load(Path(args.b_path)))
    ids = sorted(set(a.keys()) & set(b.keys()))
    if not ids:
        raise ValueError("No overlapping sample_ids between A and B.")

    metrics = [
        "has_exact_tool",
        "has_exact_and_full_edge",
        "best_edge",
        "best_edge_given_exact",
    ]
    rows = []
    for m in metrics:
        d, lo, hi = _paired_bootstrap_delta(ids, a, b, m, args.bootstrap, args.seed)
        rows.append({"metric": m, "delta_mean": d, "ci95": [lo, hi]})

    out = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "n_overlap_samples": len(ids),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Paired Bootstrap: Candidate-Pool Diagnostic Deltas")
    lines.append("")
    lines.append(f"- A: `{args.a_name}`")
    lines.append(f"- B: `{args.b_name}`")
    lines.append(f"- n overlap: {len(ids)}")
    lines.append(f"- bootstrap: {args.bootstrap}, seed={args.seed}")
    lines.append("")
    lines.append("| Metric | Δ mean (A-B) | 95% CI |")
    lines.append("|---|---:|---|")
    for r in rows:
        lo, hi = r["ci95"]
        lines.append(f"| {r['metric']} | {r['delta_mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.append("")
    lines.append("Metric definitions:")
    lines.append("- `has_exact_tool`: per sample, whether pool contains any candidate with ToolF1=1.")
    lines.append("- `has_exact_and_full_edge`: per sample, whether pool contains any candidate with ToolF1=1 and EdgeRec=1.")
    lines.append("- `best_edge`: per sample max EdgeRec over all candidates.")
    lines.append("- `best_edge_given_exact`: per sample max EdgeRec over candidates with ToolF1=1 (0 if none).")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")


if __name__ == "__main__":
    main()
