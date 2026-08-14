#!/usr/bin/env python3
"""
Evaluate selection strategies on a per-candidate dataset.

Designed for *external* test evaluation:
  - scorer trained on train-only data
  - candidates collected on disjoint test IDs

Strategies:
  1) K=1 baseline (candidate with k==1)
  2) Heuristic selection (max heuristic_score)
  3) Learned scorer selection (max predicted value)
  4) Oracle selection (max tool_f1)

Optionally computes bootstrap 95% CI (by resampling sample_id).
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


def load_model(model_path: str):
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

    conf_raw = candidate.get("confidence", 0.5)
    # AR-proposer candidate pools store "confidence": None explicitly.
    if conf_raw is None:
        conf_raw = 0.5
    conf = float(conf_raw)
    if bundle.get("ablate_confidence", False):
        conf = 0.5
    heur = float(candidate.get("heuristic_score", 0.5))
    n_tools = float(candidate.get("n_pred_tools", 0))
    n_edges = float(candidate.get("n_pred_edges", 0))
    k = float(candidate.get("k", 1))

    tools = candidate.get("pred_tools", []) or []
    edges = candidate.get("pred_edges", []) or []
    edges = [(e[0], e[1]) for e in edges if isinstance(e, (list, tuple)) and len(e) == 2]
    is_dag_pred = _is_dag(list(dict.fromkeys(tools)), edges)

    base = np.array([conf, heur, n_tools, n_edges, is_dag_pred, k], dtype=np.float32)

    if feature_set in ("toolset", "graph"):
        vec = np.zeros((len(tool_to_idx),), dtype=np.float32)
        for t in tools:
            if t in tool_to_idx:
                vec[tool_to_idx[t]] = 1.0
        if feature_set == "toolset":
            return np.concatenate([base, vec], axis=0)
        # graph: append directed edge multi-hot
        n = len(tool_to_idx)
        evec = np.zeros((n * n,), dtype=np.float32)
        for s, t in edges:
            if s in tool_to_idx and t in tool_to_idx:
                evec[tool_to_idx[s] * n + tool_to_idx[t]] = 1.0
        return np.concatenate([base, vec, evec], axis=0)

    return base


def summarize(per_sample: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["k1", "heur", "scorer", "oracle"]
    out = {}
    for k in keys:
        vals = [v[f"{k}_tool_f1"] for v in per_sample.values()]
        out[f"{k}_tool_f1"] = float(np.mean(vals)) if vals else 0.0
        vals_e = [v[f"{k}_edge_recall"] for v in per_sample.values()]
        out[f"{k}_edge_recall"] = float(np.mean(vals_e)) if vals_e else 0.0
    out["n_samples"] = float(len(per_sample))
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
    ap = argparse.ArgumentParser(description="Evaluate selection strategies on candidate dataset (external test)")
    ap.add_argument("--candidates_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples (e.g., 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    bundle = load_model(args.model_path)
    model = bundle["model"]

    candidates = load_candidates(args.candidates_path)
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_sample[str(c.get("sample_id"))].append(c)

    per_sample: Dict[str, Dict[str, float]] = {}

    for sid, cs in by_sample.items():
        cs_sorted = sorted(cs, key=lambda x: int(x.get("k", 1)))
        k1 = None
        for c in cs_sorted:
            if int(c.get("k", 1)) == 1:
                k1 = c
                break
        if k1 is None:
            k1 = cs_sorted[0]

        heur_best = max(cs_sorted, key=lambda x: float(x.get("heuristic_score", 0.0)))
        oracle_best = max(cs_sorted, key=lambda x: float(x.get("tool_f1", 0.0)))

        preds = []
        for c in cs_sorted:
            x = build_features(c, bundle)
            preds.append(float(model.predict(x.reshape(1, -1))[0]))
        scorer_best = cs_sorted[int(np.argmax(preds))]

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
    print("\n=== External Test Selection Evaluation ===")
    print(f"Samples: {int(s['n_samples'])}")
    print("\nStrategy             Tool F1    Edge Recall")
    print("-------------------  --------  ----------")
    print(f"K=1 (baseline)       {s['k1_tool_f1']:.3f}     {s['k1_edge_recall']:.3f}")
    print(f"Heuristic            {s['heur_tool_f1']:.3f}     {s['heur_edge_recall']:.3f}")
    print(f"Learned Scorer       {s['scorer_tool_f1']:.3f}     {s['scorer_edge_recall']:.3f}")
    print(f"Oracle               {s['oracle_tool_f1']:.3f}     {s['oracle_edge_recall']:.3f}")

    if args.bootstrap and args.bootstrap > 0:
        ci = bootstrap_ci(per_sample, n=args.bootstrap, seed=args.seed)
        print(f"\nBootstrap 95% CI (n={args.bootstrap})")
        for k, d in ci.items():
            t0, t1 = d["tool_f1"]
            e0, e1 = d["edge_recall"]
            name = {"k1": "K=1", "heur": "Heuristic", "scorer": "Scorer", "oracle": "Oracle"}[k]
            print(f"- {name}: ToolF1[{t0:.3f},{t1:.3f}]  EdgeRec[{e0:.3f},{e1:.3f}]")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        out = {"summary": s, "per_sample": per_sample}
        if args.bootstrap and args.bootstrap > 0:
            out["bootstrap_ci"] = ci
            out["notes"] = f"bootstrap_by_sample_id n={int(args.bootstrap)} seed={int(args.seed)}"
        with open(args.out_json, "w") as f:
            json.dump(out, f)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()
