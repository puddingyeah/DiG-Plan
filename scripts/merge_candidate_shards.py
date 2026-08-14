#!/usr/bin/env python3
"""
Merge sharded per-candidate JSONs produced by collect_candidate_data_v2.py.

Input shards:
  results/shards/candidates_*.json

Output merged:
  results/candidates_K5_merged.json

Assumptions:
  - shards are disjoint by sample_id (enforced by shard slicing)
  - each shard follows the same schema:
      {meta, summary, sample_summaries, candidates}
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def load(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Merge candidate data shards")
    ap.add_argument("--pattern", default="results/shards/candidates_K5_shard*.json")
    ap.add_argument("--output_path", default="results/candidates_K5_merged.json")
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(args.pattern))
    if not shard_paths:
        raise SystemExit(f"No shard files matched pattern: {args.pattern}")

    metas = []
    all_candidates: List[Dict] = []
    sample_summaries: List[Dict] = []
    seen_samples = set()

    for p in shard_paths:
        data = load(p)
        metas.append(data.get("meta", {}))
        for s in data.get("sample_summaries", []):
            sid = s.get("sample_id")
            if sid in seen_samples:
                raise SystemExit(f"Duplicate sample_id across shards: {sid}")
            seen_samples.add(sid)
            sample_summaries.append(s)
        all_candidates.extend(data.get("candidates", []))

    # recompute summaries from candidates
    by_sample = defaultdict(list)
    for c in all_candidates:
        by_sample[c["sample_id"]].append(c)

    recomputed_summaries = []
    for sid, cs in by_sample.items():
        f1s = [float(c.get("tool_f1", 0.0)) for c in cs]
        best_f1 = max(f1s) if f1s else 0.0
        k1 = next((c for c in cs if c.get("k") == 1), None)
        k1_f1 = float(k1.get("tool_f1", 0.0)) if k1 else float(cs[0].get("tool_f1", 0.0))
        k1_conf = k1.get("confidence") if k1 else cs[0].get("confidence")
        task_type = (k1.get("task_type") if k1 else cs[0].get("task_type")) or "unknown"
        recomputed_summaries.append(
            {
                "sample_id": sid,
                "task_type": task_type,
                "k1_f1": k1_f1,
                "best_f1": best_f1,
                "k1_confidence": k1_conf,
            }
        )

    avg_k1_f1 = float(np.mean([s["k1_f1"] for s in recomputed_summaries])) if recomputed_summaries else 0.0
    avg_best_f1 = float(np.mean([s["best_f1"] for s in recomputed_summaries])) if recomputed_summaries else 0.0
    f1_values = [float(c.get("tool_f1", 0.0)) for c in all_candidates]
    f1_range = [float(min(f1_values)), float(max(f1_values))] if f1_values else [0.0, 0.0]

    # merge meta (keep first, add shard list)
    meta0 = metas[0] if metas else {}
    merged_meta = dict(meta0)
    merged_meta.update(
        {
            "n_samples": len(by_sample),
            "n_candidates": len(all_candidates),
            "merged_from": shard_paths,
        }
    )

    out = {
        "meta": merged_meta,
        "summary": {
            "avg_k1_f1": avg_k1_f1,
            "avg_best_f1": avg_best_f1,
            "f1_range": f1_range,
        },
        "sample_summaries": recomputed_summaries,
        "candidates": all_candidates,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(shard_paths)} shards -> {args.output_path}")
    print(f"Samples: {len(by_sample)}, candidates: {len(all_candidates)}")
    print(f"avg_k1_f1={avg_k1_f1:.3f}, avg_best_f1={avg_best_f1:.3f}, f1_range={f1_range}")


if __name__ == "__main__":
    main()
