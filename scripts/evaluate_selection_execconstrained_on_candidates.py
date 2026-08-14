#!/usr/bin/env python3
"""
Evaluate selection strategies on a per-candidate dataset, with an explicit structural
executability constraint (deployable; no oracle/judge).

This is motivated by a common reviewer concern:
  - Offline structure metrics (ToolF1 / EdgeRecall) do not guarantee the predicted plan is
    even structurally executable (e.g., cycles, invalid endpoints, broken chains).

We add a simple, deterministic exec_ok(G) proxy and compare:
  - K=1 baseline
  - Heuristic selection (max heuristic_score)
  - Learned scorer selection (max predicted value)
  - Exec-constrained scorer selection:
      * Hard filter: argmax_{exec_ok==1} score, fallback to unconstrained argmax.
      * Soft penalty: argmax score - λ*(1-exec_ok)
  - Oracle selection (max tool_f1) [upper bound; not deployable]

Input schema: JSON with {"candidates": [ ... ]} where each candidate has:
  - sample_id, task_type
  - pred_tools: List[str]
  - pred_edges: List[[src,tgt], ...]
  - tool_f1, edge_recall, heuristic_score, confidence, k, ...

Output schema: {"summary":..., "bootstrap_ci":..., "per_sample":..., "notes":...}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def load_candidates(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("candidates", [])


def load_model_bundle(model_path: str) -> Dict:
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except ValueError as e:
        # Compatibility fix for numpy RNG pickles serialized across versions.
        # Some bundles contain bit-generator payloads where numpy passes a class
        # object (e.g., MT19937) instead of a string name during unpickling.
        msg = str(e)
        if "is not a known BitGenerator module" not in msg:
            raise
        import numpy.random._pickle as _np_pickle

        _orig_ctor = _np_pickle.__bit_generator_ctor

        def _compat_ctor(bit_generator_name="MT19937"):
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            return _orig_ctor(bit_generator_name)

        _np_pickle.__bit_generator_ctor = _compat_ctor
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        finally:
            _np_pickle.__bit_generator_ctor = _orig_ctor


def _normalize_edges(edges: Iterable) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for e in edges or []:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
    return out


def _is_dag(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    node_set = set(nodes)
    if not node_set:
        return False
    indeg = {n: 0 for n in node_set}
    adj = {n: [] for n in node_set}
    for u, v in edges:
        if u not in node_set or v not in node_set:
            continue
        adj[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
    q = [n for n in node_set if indeg.get(n, 0) == 0]
    visited = 0
    while q:
        u = q.pop()
        visited += 1
        for w in adj.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return visited == len(node_set)


def _weak_non_isolation(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    if len(nodes) <= 1:
        return True
    touched = set()
    for u, v in edges:
        touched.add(u)
        touched.add(v)
    return all(n in touched for n in nodes)


def _has_sink(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    if not nodes:
        return False
    outdeg = {n: 0 for n in nodes}
    for u, _v in edges:
        if u in outdeg:
            outdeg[u] += 1
    return any(outdeg[n] == 0 for n in nodes)


def _chain_ok(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """
    Chain proxy: edges form a single path that visits all nodes exactly once.
    Conditions:
      - |E| == |V|-1 for |V|>=2
      - indeg/outdeg <= 1 for all nodes
      - exactly one source (indeg==0) and one sink (outdeg==0) when |V|>=2
      - following edges from the source visits all nodes.
    """
    n = len(nodes)
    if n <= 1:
        return True
    if len(edges) != n - 1:
        return False
    node_set = set(nodes)
    indeg = {v: 0 for v in node_set}
    outdeg = {v: 0 for v in node_set}
    nxt: Dict[str, str] = {}
    for u, v in edges:
        if u not in node_set or v not in node_set:
            return False
        if u == v:
            return False
        outdeg[u] += 1
        indeg[v] += 1
        if outdeg[u] > 1 or indeg[v] > 1:
            return False
        nxt[u] = v
    sources = [v for v in node_set if indeg[v] == 0]
    sinks = [v for v in node_set if outdeg[v] == 0]
    if len(sources) != 1 or len(sinks) != 1:
        return False
    cur = sources[0]
    visited = set([cur])
    while cur in nxt:
        cur = nxt[cur]
        if cur in visited:
            return False
        visited.add(cur)
    return len(visited) == n


def _edge_endpoints_valid(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    node_set = set(nodes)
    return all(u in node_set and v in node_set for u, v in edges)


def _no_duplicate_edges(edges: List[Tuple[str, str]]) -> bool:
    return len(edges) == len(set(edges))


def _no_self_loops(edges: List[Tuple[str, str]]) -> bool:
    return all(u != v for u, v in edges)


def exec_ok(task_type: str, pred_tools: List[str], pred_edges: List[Tuple[str, str]]) -> bool:
    """
    Deterministic structural executability proxy. Mirrors the definition used in
    scripts/evaluate_executability_on_candidates.py to keep comparisons aligned.
    """
    nodes = [str(t) for t in (pred_tools or [])]
    nodes = list(dict.fromkeys(nodes))  # preserve order, dedupe
    edges = _normalize_edges(pred_edges)

    if not nodes:
        return False
    if not _edge_endpoints_valid(nodes, edges):
        return False
    if not _no_self_loops(edges):
        return False
    if not _no_duplicate_edges(edges):
        return False

    t = (task_type or "").lower()
    if t == "chain":
        return _chain_ok(nodes, edges)

    # Default (single/dag/unknown): require a DAG and at least one sink, and avoid isolation.
    # For |V|==1, these reduce to trivial checks.
    if not _weak_non_isolation(nodes, edges):
        return False
    if not _is_dag(nodes, edges):
        return False
    if not _has_sink(nodes, edges):
        return False
    return True


def build_features(candidate: Dict, bundle: Dict) -> np.ndarray:
    # Keep feature construction consistent with scripts/evaluate_selection_on_candidates.py
    feature_set = bundle.get("feature_set", "basic")
    tool_to_idx = bundle.get("tool_to_idx", {})

    conf_raw = candidate.get("confidence", 0.5)
    if conf_raw is None:
        conf_raw = 0.5
    conf = float(conf_raw)
    heur = float(candidate.get("heuristic_score", 0.5))
    n_tools = float(candidate.get("n_pred_tools", 0))
    n_edges = float(candidate.get("n_pred_edges", 0))
    k = float(candidate.get("k", 1))

    tools = candidate.get("pred_tools", []) or []
    edges = _normalize_edges(candidate.get("pred_edges", []) or [])
    is_dag_pred = 1.0 if _is_dag(list(dict.fromkeys([str(t) for t in tools])), edges) else 0.0

    base = np.array([conf, heur, n_tools, n_edges, is_dag_pred, k], dtype=np.float32)

    if feature_set in ("toolset", "graph"):
        vec = np.zeros((len(tool_to_idx),), dtype=np.float32)
        for t in tools:
            tt = str(t)
            if tt in tool_to_idx:
                vec[tool_to_idx[tt]] = 1.0
        if feature_set == "toolset":
            return np.concatenate([base, vec], axis=0)
        n = len(tool_to_idx)
        evec = np.zeros((n * n,), dtype=np.float32)
        for s, t in edges:
            if s in tool_to_idx and t in tool_to_idx:
                evec[tool_to_idx[s] * n + tool_to_idx[t]] = 1.0
        return np.concatenate([base, vec, evec], axis=0)

    return base


def _pick_k1(group: List[Dict]) -> Dict:
    for c in group:
        if int(c.get("k", 1)) == 1:
            return c
    return group[0]


def _pick_max(group: List[Dict], key: str) -> Dict:
    return max(group, key=lambda c: float(c.get(key, 0.0)))


def _pick_scorer(group: List[Dict], model, bundle: Dict) -> Dict:
    xs = np.stack([build_features(c, bundle) for c in group], axis=0)
    preds = model.predict(xs)
    return group[int(np.argmax(preds))]


def _pick_scorer_exec_filter(group: List[Dict], model, bundle: Dict) -> Tuple[Dict, bool]:
    ok = [c for c in group if exec_ok(str(c.get("task_type", "")), c.get("pred_tools") or [], c.get("pred_edges") or [])]
    if ok:
        return _pick_scorer(ok, model, bundle), False
    return _pick_scorer(group, model, bundle), True


def _pick_scorer_exec_penalty(group: List[Dict], model, bundle: Dict, lam: float) -> Dict:
    xs = np.stack([build_features(c, bundle) for c in group], axis=0)
    preds = model.predict(xs).astype(np.float32)
    pen = np.array(
        [0.0 if exec_ok(str(c.get("task_type", "")), c.get("pred_tools") or [], c.get("pred_edges") or []) else 1.0 for c in group],
        dtype=np.float32,
    )
    scores = preds - float(lam) * pen
    return group[int(np.argmax(scores))]


def summarize(per_sample: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["k1", "heur", "scorer", "scorer_execfilt", "scorer_execpen", "oracle"]
    out: Dict[str, float] = {}
    for k in keys:
        out[f"{k}_tool_f1"] = float(np.mean([v[f"{k}_tool_f1"] for v in per_sample.values()])) if per_sample else 0.0
        out[f"{k}_edge_recall"] = float(np.mean([v[f"{k}_edge_recall"] for v in per_sample.values()])) if per_sample else 0.0
        out[f"{k}_exec_ok"] = float(np.mean([v[f"{k}_exec_ok"] for v in per_sample.values()])) if per_sample else 0.0
    out["n_samples"] = float(len(per_sample))
    return out


def bootstrap_ci(per_sample: Dict[str, Dict[str, float]], n: int, seed: int = 0) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    rng = np.random.default_rng(seed)
    sids = list(per_sample.keys())
    if not sids:
        return {}
    keys = ["k1", "heur", "scorer", "scorer_execfilt", "scorer_execpen", "oracle"]
    metrics = ["tool_f1", "edge_recall", "exec_ok"]
    samples: Dict[str, Dict[str, List[float]]] = {k: {m: [] for m in metrics} for k in keys}
    for _ in range(n):
        draw = rng.choice(sids, size=len(sids), replace=True)
        for k in keys:
            for m in metrics:
                samples[k][m].append(float(np.mean([per_sample[s][f"{k}_{m}"] for s in draw])))
    ci: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for k in keys:
        ci[k] = {}
        for m in metrics:
            arr = np.array(samples[k][m], dtype=np.float32)
            ci[k][m] = (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))
    return ci


def main() -> None:
    ap = argparse.ArgumentParser(description="Selection eval with executability-constrained value guidance")
    ap.add_argument("--candidates_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--exec_penalty_lambda", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    candidates = load_candidates(args.candidates_path)
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_sample[str(c.get("sample_id"))].append(c)

    bundle = load_model_bundle(args.model_path)
    model = bundle["model"]

    per_sample: Dict[str, Dict[str, float]] = {}
    fallback_execfilt = 0
    mean_ok_rate = []

    for sid, group in by_sample.items():
        group = sorted(group, key=lambda x: int(x.get("k", 1)))
        k1 = _pick_k1(group)
        heur_best = _pick_max(group, "heuristic_score")
        scorer_best = _pick_scorer(group, model, bundle)
        scorer_execfilt_best, used_fallback = _pick_scorer_exec_filter(group, model, bundle)
        if used_fallback:
            fallback_execfilt += 1
        scorer_execpen_best = _pick_scorer_exec_penalty(group, model, bundle, lam=float(args.exec_penalty_lambda))
        oracle_best = _pick_max(group, "tool_f1")

        def _rec(c: Dict) -> Tuple[float, float, float]:
            t = float(c.get("tool_f1", 0.0))
            e = float(c.get("edge_recall", 0.0))
            ok = 1.0 if exec_ok(str(c.get("task_type", "")), c.get("pred_tools") or [], c.get("pred_edges") or []) else 0.0
            return t, e, ok

        # Per-sample: also track how many candidates are exec_ok for diagnostics.
        ok_cnt = sum(
            1
            for c in group
            if exec_ok(str(c.get("task_type", "")), c.get("pred_tools") or [], c.get("pred_edges") or [])
        )
        mean_ok_rate.append(ok_cnt / max(len(group), 1))

        k1_t, k1_e, k1_ok = _rec(k1)
        h_t, h_e, h_ok = _rec(heur_best)
        s_t, s_e, s_ok = _rec(scorer_best)
        sf_t, sf_e, sf_ok = _rec(scorer_execfilt_best)
        sp_t, sp_e, sp_ok = _rec(scorer_execpen_best)
        o_t, o_e, o_ok = _rec(oracle_best)

        per_sample[sid] = {
            "k1_tool_f1": k1_t,
            "k1_edge_recall": k1_e,
            "k1_exec_ok": k1_ok,
            "heur_tool_f1": h_t,
            "heur_edge_recall": h_e,
            "heur_exec_ok": h_ok,
            "scorer_tool_f1": s_t,
            "scorer_edge_recall": s_e,
            "scorer_exec_ok": s_ok,
            "scorer_execfilt_tool_f1": sf_t,
            "scorer_execfilt_edge_recall": sf_e,
            "scorer_execfilt_exec_ok": sf_ok,
            "scorer_execpen_tool_f1": sp_t,
            "scorer_execpen_edge_recall": sp_e,
            "scorer_execpen_exec_ok": sp_ok,
            "oracle_tool_f1": o_t,
            "oracle_edge_recall": o_e,
            "oracle_exec_ok": o_ok,
            "n_candidates": float(len(group)),
            "n_exec_ok_candidates": float(ok_cnt),
        }

    s = summarize(per_sample)
    s["execfilt_fallback_rate"] = float(fallback_execfilt) / float(max(len(per_sample), 1))
    s["mean_exec_ok_candidate_rate"] = float(np.mean(mean_ok_rate)) if mean_ok_rate else 0.0
    s["exec_penalty_lambda"] = float(args.exec_penalty_lambda)

    ci = None
    if args.bootstrap and args.bootstrap > 0:
        ci = bootstrap_ci(per_sample, n=args.bootstrap, seed=args.seed)

    out = {
        "summary": s,
        "bootstrap_ci": ci,
        "per_sample": per_sample,
        "notes": {
            "exec_ok_definition": "Deterministic structural proxy: valid endpoints, no self-loops/duplicates, non-isolation; chain requires a single Hamiltonian path; otherwise require DAG+sink.",
            "scorer_execfilt": "Hard filter: argmax score among exec_ok==1 candidates; fallback to unconstrained argmax if none are exec_ok.",
            "scorer_execpen": "Soft penalty: argmax score - lambda*(1-exec_ok).",
        },
    }

    print("\n=== Selection Eval (Exec-Constrained) ===")
    print(f"Samples: {int(s['n_samples'])}")
    print(f"Mean exec_ok candidate rate: {s['mean_exec_ok_candidate_rate']:.3f}")
    print(f"Exec-filter fallback rate: {s['execfilt_fallback_rate']:.3f}")
    print("")
    print("Strategy                Tool F1   EdgeRec  ExecOK")
    print("----------------------  -------   -------  ------")
    for k in ["k1", "heur", "scorer", "scorer_execfilt", "scorer_execpen", "oracle"]:
        print(f"{k:<22}  {s[f'{k}_tool_f1']:.3f}    {s[f'{k}_edge_recall']:.3f}    {s[f'{k}_exec_ok']:.3f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()
