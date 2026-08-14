#!/usr/bin/env python3
"""
Executability sanity check for Tool Graph candidates.

Motivation (IJCAI defense):
  ToolF1 / EdgeRecall are offline structure metrics. Reviewers may ask whether a predicted
  tool graph is even *structurally executable*. This script provides a deterministic,
  rule-based executability proxy without any external API/judge.

What it checks (purely on predicted tools + edges):
  - edge endpoints are valid (both endpoints in tool set)
  - no self-loops
  - no duplicate edges
  - DAG acyclicity (Kahn)
  - connectivity / non-isolation (weak: nodes participate in at least one edge when |V|>1)
  - chain-specific: edges form a single Hamiltonian path over nodes (deg constraints + connectivity)

It evaluates multiple selection strategies on the same candidate pool:
  - K=1 (k==1)
  - Heuristic (max heuristic_score)
  - Scorer (max learned score; optional model bundle)
  - Oracle (max tool_f1; upper bound)
  - Consensus(toolfreq) and Consensus(Jaccard medoid)

Output:
  JSON with:
    - summary: mean executability rates + mean ToolF1/EdgeRec for each strategy
    - per_sample: per-sample executability flags for each strategy
    - notes: definitions
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


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
    for u, v in edges:
        if u in outdeg:
            outdeg[u] += 1
    return any(outdeg[n] == 0 for n in nodes)


def _chain_ok(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """
    Chain executability proxy: edges define a single path that visits all nodes exactly once.
    Conditions:
      - |E| == |V|-1 for |V|>=2
      - indeg/outdeg constraints: indeg<=1,outdeg<=1 for all nodes
      - exactly one source (indeg==0) and one sink (outdeg==0) when |V|>=2
      - connected: starting from source, following edges visits all nodes.
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
    # Follow path
    cur = sources[0]
    visited = set([cur])
    while cur in nxt:
        cur = nxt[cur]
        if cur in visited:
            return False
        visited.add(cur)
    return len(visited) == n


def _normalize_edges(edges: Iterable) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for e in edges or []:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
    return out


def _edge_endpoints_valid(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    node_set = set(nodes)
    return all(u in node_set and v in node_set for u, v in edges)


def _no_duplicate_edges(edges: List[Tuple[str, str]]) -> bool:
    return len(edges) == len(set(edges))


def _no_self_loops(edges: List[Tuple[str, str]]) -> bool:
    return all(u != v for u, v in edges)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def _pick_consensus_toolfreq(group: List[Dict]) -> Dict:
    # Score each candidate by sum of tool frequencies across the pool.
    freq = Counter()
    for c in group:
        for t in (c.get("pred_tools") or []):
            freq[str(t)] += 1
    def score(c: Dict) -> float:
        tools = [str(t) for t in (c.get("pred_tools") or [])]
        return float(sum(freq[t] for t in tools))
    return max(group, key=score)


def _pick_consensus_jaccard(group: List[Dict]) -> Dict:
    # Pick medoid candidate minimizing average Jaccard distance to others.
    sets = [set(map(str, (c.get("pred_tools") or []))) for c in group]
    if not sets:
        return group[0]
    n = len(group)
    # Precompute matrix in O(n^2) where n=K (small, typically <=20).
    sims = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _jaccard(sets[i], sets[j])
            sims[i][j] = s
            sims[j][i] = s
    avg_sim = [sum(sims[i]) / n for i in range(n)]
    best_i = int(np.argmax(avg_sim))
    return group[best_i]


def load_candidates(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("candidates", [])


def load_model(model_path: str) -> Dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def build_features(candidate: Dict, bundle: Dict) -> np.ndarray:
    # Reuse the feature construction logic from evaluate_selection_on_candidates.py
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

    tools = [str(t) for t in (candidate.get("pred_tools") or [])]
    edges = _normalize_edges(candidate.get("pred_edges") or [])
    is_dag_pred = 1.0 if _is_dag(list(dict.fromkeys(tools)), edges) else 0.0

    base = np.array([conf, heur, n_tools, n_edges, is_dag_pred, k], dtype=np.float32)

    if feature_set in ("toolset", "graph"):
        vec = np.zeros((len(tool_to_idx),), dtype=np.float32)
        for t in tools:
            if t in tool_to_idx:
                vec[tool_to_idx[t]] = 1.0
        if feature_set == "toolset":
            return np.concatenate([base, vec], axis=0)
        n = len(tool_to_idx)
        evec = np.zeros((n * n,), dtype=np.float32)
        for s, t in edges:
            if s in tool_to_idx and t in tool_to_idx:
                evec[tool_to_idx[s] * n + tool_to_idx[t]] = 1.0
        return np.concatenate([base, vec, evec], axis=0)

    return base


@dataclass(frozen=True)
class ExecFlags:
    endpoints_ok: bool
    no_self_loops: bool
    no_dup_edges: bool
    is_dag: bool
    has_sink: bool
    non_isolation: bool
    chain_ok: bool

    @property
    def exec_ok_daglike(self) -> bool:
        return (
            self.endpoints_ok
            and self.no_self_loops
            and self.no_dup_edges
            and self.is_dag
            and self.has_sink
            and self.non_isolation
        )

    @property
    def exec_ok_chain(self) -> bool:
        return self.endpoints_ok and self.no_self_loops and self.no_dup_edges and self.chain_ok


def compute_exec_flags(task_type: str, tools: List[str], edges: List[Tuple[str, str]]) -> ExecFlags:
    nodes = list(dict.fromkeys([str(t) for t in tools]))
    e = [(str(u), str(v)) for (u, v) in edges]

    endpoints_ok = _edge_endpoints_valid(nodes, e)
    no_self = _no_self_loops(e)
    no_dup = _no_duplicate_edges(e)
    is_dag = _is_dag(nodes, e) if endpoints_ok else False
    has_sink = _has_sink(nodes, e) if endpoints_ok else False
    non_iso = _weak_non_isolation(nodes, e) if endpoints_ok else False
    chain_ok = _chain_ok(nodes, e) if endpoints_ok else False

    return ExecFlags(
        endpoints_ok=endpoints_ok,
        no_self_loops=no_self,
        no_dup_edges=no_dup,
        is_dag=is_dag,
        has_sink=has_sink,
        non_isolation=non_iso,
        chain_ok=chain_ok,
    )


def _mean_bool(xs: List[bool]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Executability sanity check on candidate pools")
    ap.add_argument("--candidates_path", required=True)
    ap.add_argument("--model_path", default=None, help="Optional scorer bundle for scorer selection.")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    candidates = load_candidates(args.candidates_path)
    if not candidates:
        raise SystemExit(f"No candidates found in {args.candidates_path}")

    bundle = load_model(args.model_path) if args.model_path else None
    model = bundle.get("model") if bundle else None

    by_sample = defaultdict(list)
    for c in candidates:
        by_sample[str(c.get("sample_id"))].append(c)

    per_sample: Dict[str, Dict[str, float]] = {}

    for sid, group in by_sample.items():
        group = sorted(group, key=lambda x: int(x.get("k", 1)))
        k1 = next((c for c in group if int(c.get("k", 1)) == 1), group[0])
        heur_best = max(group, key=lambda x: float(x.get("heuristic_score", 0.0)))
        oracle_best = max(group, key=lambda x: float(x.get("tool_f1", 0.0)))
        cons_tool = _pick_consensus_toolfreq(group)
        cons_jac = _pick_consensus_jaccard(group)

        scorer_best = None
        if model is not None:
            # Predict scores for all candidates; pick max.
            X = np.stack([build_features(c, bundle) for c in group], axis=0)
            y = model.predict(X)
            best_idx = int(np.argmax(y))
            scorer_best = group[best_idx]

        task_type = str(k1.get("task_type") or "unknown")

        def flags_for(c: Dict) -> ExecFlags:
            tools = c.get("pred_tools") or []
            edges = _normalize_edges(c.get("pred_edges") or [])
            return compute_exec_flags(task_type, [str(t) for t in tools], edges)

        f_k1 = flags_for(k1)
        f_heur = flags_for(heur_best)
        f_oracle = flags_for(oracle_best)
        f_cons_tool = flags_for(cons_tool)
        f_cons_jac = flags_for(cons_jac)
        f_scorer = flags_for(scorer_best) if scorer_best is not None else None

        def ok(flags: ExecFlags) -> bool:
            if task_type == "chain":
                return flags.exec_ok_chain
            # For single/dag/unknown: require dag-like structure, but allow |V|<=1.
            if (k1.get("n_pred_tools", 0) or len(k1.get("pred_tools") or [])) <= 1:
                return flags.endpoints_ok and flags.no_self_loops and flags.no_dup_edges
            return flags.exec_ok_daglike

        per_sample[sid] = {
            "task_type": task_type,
            "k1_exec_ok": float(ok(f_k1)),
            "heur_exec_ok": float(ok(f_heur)),
            "oracle_exec_ok": float(ok(f_oracle)),
            "cons_toolfreq_exec_ok": float(ok(f_cons_tool)),
            "cons_jaccard_exec_ok": float(ok(f_cons_jac)),
        }
        if f_scorer is not None:
            per_sample[sid]["scorer_exec_ok"] = float(ok(f_scorer))

        # Also store core metrics for correlation sanity.
        per_sample[sid].update(
            {
                "k1_tool_f1": float(k1.get("tool_f1", 0.0)),
                "k1_edge_recall": float(k1.get("edge_recall", 0.0)),
                "heur_tool_f1": float(heur_best.get("tool_f1", 0.0)),
                "heur_edge_recall": float(heur_best.get("edge_recall", 0.0)),
                "oracle_tool_f1": float(oracle_best.get("tool_f1", 0.0)),
                "oracle_edge_recall": float(oracle_best.get("edge_recall", 0.0)),
                "cons_toolfreq_tool_f1": float(cons_tool.get("tool_f1", 0.0)),
                "cons_toolfreq_edge_recall": float(cons_tool.get("edge_recall", 0.0)),
                "cons_jaccard_tool_f1": float(cons_jac.get("tool_f1", 0.0)),
                "cons_jaccard_edge_recall": float(cons_jac.get("edge_recall", 0.0)),
            }
        )
        if scorer_best is not None:
            per_sample[sid]["scorer_tool_f1"] = float(scorer_best.get("tool_f1", 0.0))
            per_sample[sid]["scorer_edge_recall"] = float(scorer_best.get("edge_recall", 0.0))

    # Summaries
    keys = ["k1", "heur", "oracle", "cons_toolfreq", "cons_jaccard"] + (["scorer"] if model is not None else [])
    summary: Dict[str, float] = {"n_samples": float(len(per_sample))}
    for k in keys:
        summary[f"{k}_exec_ok"] = _mean_bool([bool(v.get(f"{k}_exec_ok", 0.0)) for v in per_sample.values()])
        summary[f"{k}_tool_f1"] = float(np.mean([v.get(f"{k}_tool_f1", 0.0) for v in per_sample.values()]))
        summary[f"{k}_edge_recall"] = float(np.mean([v.get(f"{k}_edge_recall", 0.0) for v in per_sample.values()]))

    notes = {
        "exec_ok_definition": {
            "chain": "endpoints_ok & no_self_loops & no_dup_edges & chain_ok (single path over all nodes)",
            "daglike": "endpoints_ok & no_self_loops & no_dup_edges & is_dag & has_sink & non_isolation (when |V|>1)",
        },
        "limitations": [
            "This is a structural executability proxy; it does not validate tool arguments or API semantics.",
            "It is deterministic and judge-free, intended as an IJCAI sanity check rather than final success metric.",
        ],
    }

    out = {"summary": summary, "per_sample": per_sample, "notes": notes}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    print("Summary:")
    for k in keys:
        print(f"- {k}: exec_ok={summary[f'{k}_exec_ok']:.3f}  tool_f1={summary[f'{k}_tool_f1']:.3f}  edge_rec={summary[f'{k}_edge_recall']:.3f}")


if __name__ == "__main__":
    main()
