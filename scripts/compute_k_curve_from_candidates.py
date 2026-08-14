#!/usr/bin/env python3
"""
Compute best-of-K curves from an existing per-candidate dataset (no GPU required).

Given a dataset with candidates for k=1..K_max per sample, compute performance
for each K' in [1..K_max] under different selection strategies:
  - K=1 baseline (always choose k==1 candidate)
  - Heuristic selection (max heuristic_score among k<=K')
  - Learned scorer selection (max predicted score among k<=K')
  - Oracle selection (max tool_f1 among k<=K')

Outputs a JSON + markdown table, optionally with bootstrap CIs by sample_id.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def load_candidates(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("candidates", [])


def load_model(model_path: str) -> Dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _is_dag(nodes: List[str], edges: List[Tuple[str, str]]) -> float:
    if not nodes:
        return 0.0
    node_set = set(nodes)
    indeg = {n: 0 for n in node_set}
    adj = {n: [] for n in node_set}
    for s, t in edges:
        if s not in node_set or t not in node_set:
            continue
        adj[s].append(t)
        indeg[t] = indeg.get(t, 0) + 1
    q = [n for n in node_set if indeg.get(n, 0) == 0]
    visited = 0
    while q:
        u = q.pop()
        visited += 1
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return 1.0 if visited == len(node_set) else 0.0


def build_features(candidate: Dict, bundle: Dict) -> np.ndarray:
    feature_set = bundle.get("feature_set", "basic")
    tool_to_idx = bundle.get("tool_to_idx", {})

    conf = float(candidate.get("confidence", 0.5) or 0.5)
    heur = float(candidate.get("heuristic_score", 0.5) or 0.5)
    n_tools = float(candidate.get("n_pred_tools", 0) or 0)
    n_edges = float(candidate.get("n_pred_edges", 0) or 0)
    k = float(candidate.get("k", 1) or 1)

    tools = candidate.get("pred_tools", []) or []
    edges = candidate.get("pred_edges", []) or []
    edges = [(e[0], e[1]) for e in edges if isinstance(e, (list, tuple)) and len(e) == 2]
    is_dag_pred = _is_dag(list(dict.fromkeys(tools)), edges)

    base = np.array([conf, heur, n_tools, n_edges, is_dag_pred, k], dtype=np.float32)
    if feature_set == "toolset":
        vec = np.zeros((len(tool_to_idx),), dtype=np.float32)
        for t in tools:
            if t in tool_to_idx:
                vec[tool_to_idx[t]] = 1.0
        return np.concatenate([base, vec], axis=0)
    return base


def summarize(per_sample: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["k1", "heur", "scorer", "oracle"]
    out = {"n_samples": float(len(per_sample))}
    for k in keys:
        out[f"{k}_tool_f1"] = float(np.mean([v[f"{k}_tool_f1"] for v in per_sample.values()])) if per_sample else 0.0
        out[f"{k}_edge_recall"] = float(np.mean([v[f"{k}_edge_recall"] for v in per_sample.values()])) if per_sample else 0.0
    return out


def bootstrap_ci(per_sample: Dict[str, Dict[str, float]], n: int, seed: int = 0) -> Dict[str, Dict[str, Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    sids = list(per_sample.keys())
    if not sids:
        return {}
    keys = ["k1", "heur", "scorer", "oracle"]
    tool_samples = {k: [] for k in keys}
    edge_samples = {k: [] for k in keys}
    for _ in range(n):
        draw = rng.choice(sids, size=len(sids), replace=True)
        for k in keys:
            tool_samples[k].append(float(np.mean([per_sample[s][f"{k}_tool_f1"] for s in draw])))
            edge_samples[k].append(float(np.mean([per_sample[s][f"{k}_edge_recall"] for s in draw])))
    ci = {}
    for k in keys:
        t = np.array(tool_samples[k])
        e = np.array(edge_samples[k])
        ci[k] = {
            "tool_f1": (float(np.quantile(t, 0.025)), float(np.quantile(t, 0.975))),
            "edge_recall": (float(np.quantile(e, 0.025)), float(np.quantile(e, 0.975))),
        }
    return ci


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute best-of-K curves from candidate dataset (CPU-only)")
    ap.add_argument("--candidates_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bundle = load_model(args.model_path)
    model = bundle["model"]

    candidates = load_candidates(args.candidates_path)
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_sample[str(c.get("sample_id"))].append(c)

    # determine K_max present
    K_max = 1
    for cs in by_sample.values():
        for c in cs:
            K_max = max(K_max, int(c.get("k", 1)))

    results = []

    for K in range(1, K_max + 1):
        per_sample: Dict[str, Dict[str, float]] = {}
        for sid, cs in by_sample.items():
            cs_k = [c for c in cs if int(c.get("k", 1)) <= K]
            if not cs_k:
                continue
            cs_k = sorted(cs_k, key=lambda x: int(x.get("k", 1)))
            k1 = next((c for c in cs_k if int(c.get("k", 1)) == 1), cs_k[0])
            heur_best = max(cs_k, key=lambda x: float(x.get("heuristic_score", 0.0)))
            oracle_best = max(cs_k, key=lambda x: float(x.get("tool_f1", 0.0)))

            preds = []
            for c in cs_k:
                x = build_features(c, bundle)
                preds.append(float(model.predict(x.reshape(1, -1))[0]))
            scorer_best = cs_k[int(np.argmax(preds))]

            per_sample[sid] = {
                "k1_tool_f1": float(k1.get("tool_f1", 0.0)),
                "k1_edge_recall": float(k1.get("edge_recall", 0.0)),
                "heur_tool_f1": float(heur_best.get("tool_f1", 0.0)),
                "heur_edge_recall": float(heur_best.get("edge_recall", 0.0)),
                "scorer_tool_f1": float(scorer_best.get("tool_f1", 0.0)),
                "scorer_edge_recall": float(scorer_best.get("edge_recall", 0.0)),
                "oracle_tool_f1": float(oracle_best.get("tool_f1", 0.0)),
                "oracle_edge_recall": float(oracle_best.get("edge_recall", 0.0)),
            }

        s = summarize(per_sample)
        ci = bootstrap_ci(per_sample, n=args.bootstrap, seed=args.seed) if args.bootstrap and args.bootstrap > 0 else None
        results.append({"K": K, "summary": s, "ci": ci})

    out = {
        "candidates_path": args.candidates_path,
        "model_path": args.model_path,
        "model_label": bundle.get("label", "?"),
        "feature_set": bundle.get("feature_set", "?"),
        "K_max": K_max,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # md table
    lines = []
    lines.append("# Best-of-K Curve (from existing candidates)\n")
    lines.append(f"Candidates: `{args.candidates_path}`\n")
    lines.append(f"Scorer: `{args.model_path}` (label={bundle.get('label','?')})\n")
    lines.append("")
    lines.append("| K | K=1 ToolF1 | Heur ToolF1 | Scorer ToolF1 | Oracle ToolF1 | K=1 Edge | Heur Edge | Scorer Edge | Oracle Edge |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r['K']} | {s['k1_tool_f1']:.3f} | {s['heur_tool_f1']:.3f} | {s['scorer_tool_f1']:.3f} | {s['oracle_tool_f1']:.3f} | "
            f"{s['k1_edge_recall']:.3f} | {s['heur_edge_recall']:.3f} | {s['scorer_edge_recall']:.3f} | {s['oracle_edge_recall']:.3f} |"
        )
    lines.append("")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")


if __name__ == "__main__":
    main()
