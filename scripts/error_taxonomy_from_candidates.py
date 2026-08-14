#!/usr/bin/env python3
"""
Generate an IJCAI-ready qualitative error taxonomy from a per-candidate dataset.

Inputs:
  - per-candidate pool (e.g., candidates_K5_test500_merged.json)
  - TaskBench jsonl file to recover instruction + GT tools/edges
  - a trained scorer model bundle (pickle produced by train_plan_scorer_v3.py)

Outputs:
  - Markdown report containing:
      1) aggregate error category stats over the whole evaluation set
      2) 20-30 representative cases (wins + regressions) with tool/edge diffs

CPU-only; safe to run on shared GPU servers.
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl_by_id(path: str, wanted_ids: set) -> Dict[str, Dict]:
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("id"))
            if sid in wanted_ids:
                out[sid] = r
    return out


def parse_gt(sample: Dict) -> Tuple[List[str], List[Tuple[str, str]]]:
    nodes = json.loads(sample.get("tool_nodes", "[]"))
    tools = [n.get("task", "") for n in nodes if isinstance(n, dict)]
    links = json.loads(sample.get("tool_links", "[]"))
    edges = [(l.get("source", ""), l.get("target", "")) for l in links if isinstance(l, dict)]
    tools = [t for t in tools if t]
    edges = [(s, t) for s, t in edges if s and t]
    return tools, edges


def normalize_edges(edges) -> List[Tuple[str, str]]:
    out = []
    for e in edges or []:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
    return out


def is_dag(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    if not nodes:
        return True
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
    return visited == len(node_set)


def tool_metrics(pred_tools: List[str], gt_tools: List[str]) -> Tuple[float, float, float, List[str], List[str]]:
    ps = set(pred_tools or [])
    gs = set(gt_tools or [])
    if not ps and not gs:
        return 1.0, 1.0, 1.0, [], []
    if not ps or not gs:
        missing = sorted(gs - ps)
        extra = sorted(ps - gs)
        return 0.0, 0.0, 0.0, missing, extra
    tp = len(ps & gs)
    p = tp / len(ps) if ps else 0.0
    r = tp / len(gs) if gs else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    missing = sorted(gs - ps)
    extra = sorted(ps - gs)
    return float(p), float(r), float(f1), missing, extra


def edge_recall(pred_edges: List[Tuple[str, str]], gt_edges: List[Tuple[str, str]]) -> Tuple[float, int]:
    gs = set(gt_edges or [])
    ps = set(pred_edges or [])
    if not gs:
        return (1.0 if not ps else 0.0), 0
    miss = len(gs - ps)
    return float(len(gs & ps) / len(gs)), miss


def build_features_for_scorer(candidate: Dict, bundle: Dict) -> np.ndarray:
    """
    Use the exact same deployable feature extractor as train_plan_scorer_v3.py,
    so the feature dimension matches the loaded model bundle (basic/toolset/graph).
    """
    from train_plan_scorer_v3 import extract_features  # local import to keep this script standalone

    feature_set = str(bundle.get("feature_set", "basic"))
    tool_to_idx = bundle.get("tool_to_idx", {}) or {}
    return extract_features(candidate, feature_set, tool_to_idx)


@dataclass
class Pick:
    tool_f1: float
    edge_recall: float
    pred_tools: List[str]
    pred_edges: List[Tuple[str, str]]
    confidence: float
    heuristic_score: float
    k: int


def select_strategies(cands: List[Dict], scorer_bundle: Dict) -> Dict[str, Pick]:
    cs = sorted(cands, key=lambda x: int(x.get("k", 1)))
    k1 = next((c for c in cs if int(c.get("k", 1)) == 1), cs[0])
    heur = max(cs, key=lambda x: float(x.get("heuristic_score", 0.0)))
    oracle = max(cs, key=lambda x: float(x.get("tool_f1", 0.0)))

    model = scorer_bundle["model"]
    preds = []
    for c in cs:
        x = build_features_for_scorer(c, scorer_bundle)
        preds.append(float(model.predict(x.reshape(1, -1))[0]))
    scorer = cs[int(np.argmax(preds))]

    def to_pick(c: Dict) -> Pick:
        return Pick(
            tool_f1=float(c.get("tool_f1", 0.0)),
            edge_recall=float(c.get("edge_recall", 0.0)),
            pred_tools=[str(t) for t in (c.get("pred_tools") or [])],
            pred_edges=normalize_edges(c.get("pred_edges") or []),
            confidence=float(c.get("confidence", 0.5) or 0.5),
            heuristic_score=float(c.get("heuristic_score", 0.0) or 0.0),
            k=int(c.get("k", 1) or 1),
        )

    return {"k1": to_pick(k1), "heur": to_pick(heur), "scorer": to_pick(scorer), "oracle": to_pick(oracle)}


def categorize(pred_tools: List[str], pred_edges: List[Tuple[str, str]], gt_tools: List[str], gt_edges: List[Tuple[str, str]]) -> List[str]:
    p, r, f1, missing, extra = tool_metrics(pred_tools, gt_tools)
    er, miss_e = edge_recall(pred_edges, gt_edges)
    tags = []
    if missing:
        tags.append("miss_tool")
    if extra:
        tags.append("extra_tool")
    if gt_edges and miss_e > 0:
        tags.append("miss_edge")
    if not is_dag(list(dict.fromkeys(pred_tools)), pred_edges):
        tags.append("cycle")
    if not tags:
        tags.append("ok")
    return tags


def short(text: str, n: int = 220) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Error taxonomy from per-candidate pool (CPU-only)")
    ap.add_argument("--candidates_path", required=True)
    ap.add_argument("--taskbench_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    ap.add_argument("--scorer_model", required=True, help="Pickle bundle from train_plan_scorer_v3.py")
    ap.add_argument("--n_cases", type=int, default=30, help="Total cases to include (split evenly chain/dag)")
    ap.add_argument("--out_md", default="results/ijcai/error_taxonomy.md")
    args = ap.parse_args()

    data = load_json(args.candidates_path)
    candidates = data.get("candidates", [])
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_sample[str(c.get("sample_id"))].append(c)

    with open(args.scorer_model, "rb") as f:
        scorer_bundle = pickle.load(f)

    wanted_ids = set(by_sample.keys())
    tb = load_jsonl_by_id(args.taskbench_path, wanted_ids)

    # aggregate stats over all samples
    tag_counts = {k: Counter() for k in ["k1", "heur", "scorer"]}
    deltas = []  # (sid, type, scorer_minus_heur_f1, scorer_minus_heur_edge)
    per_sample_info = {}

    for sid, cs in by_sample.items():
        sample = tb.get(sid)
        if not sample:
            continue
        task_type = str(sample.get("type", "unknown"))
        instruction = sample.get("instruction", "")
        gt_tools, gt_edges = parse_gt(sample)

        picks = select_strategies(cs, scorer_bundle)
        for name in ["k1", "heur", "scorer"]:
            tags = categorize(picks[name].pred_tools, picks[name].pred_edges, gt_tools, gt_edges)
            for t in tags:
                tag_counts[name][t] += 1

        deltas.append(
            (
                sid,
                task_type,
                picks["scorer"].tool_f1 - picks["heur"].tool_f1,
                picks["scorer"].edge_recall - picks["heur"].edge_recall,
            )
        )
        per_sample_info[sid] = {
            "task_type": task_type,
            "instruction": instruction,
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
            "picks": picks,
        }

    # case selection (wins/regressions), stratified by task type
    n_each = args.n_cases // 2
    wins = sorted(deltas, key=lambda x: x[2], reverse=True)
    losses = sorted(deltas, key=lambda x: x[2])

    def pick_cases(pool, want_type: str, k: int) -> List[str]:
        out = []
        for sid, t, df1, de in pool:
            if t != want_type:
                continue
            if sid in per_sample_info and sid not in out:
                out.append(sid)
            if len(out) >= k:
                break
        return out

    win_chain = pick_cases(wins, "chain", n_each // 2)
    win_dag = pick_cases(wins, "dag", n_each // 2)
    loss_chain = pick_cases(losses, "chain", n_each // 2)
    loss_dag = pick_cases(losses, "dag", n_each // 2)

    selected = []
    selected.extend(win_chain + win_dag + loss_chain + loss_dag)

    # if n_cases not divisible / not enough, fill from wins regardless of type
    for sid, _, _, _ in wins:
        if len(selected) >= args.n_cases:
            break
        if sid not in selected and sid in per_sample_info:
            selected.append(sid)

    def fmt_tools(ts: List[str]) -> str:
        uniq = list(dict.fromkeys(ts))
        return ", ".join(uniq) if uniq else "(none)"

    def fmt_edges(es: List[Tuple[str, str]]) -> str:
        if not es:
            return "(none)"
        uniq = list(dict.fromkeys(es))
        return "; ".join([f"{a}->{b}" for a, b in uniq[:12]]) + ("; ..." if len(uniq) > 12 else "")

    # write report
    lines = []
    lines.append("# Error Taxonomy (Paper Test)\n")
    lines.append(f"Candidates: `{args.candidates_path}`\n")
    lines.append(f"Scorer: `{args.scorer_model}` (feature_set={scorer_bundle.get('feature_set','?')}, label={scorer_bundle.get('label','?')})\n")
    lines.append(f"Samples: {len(per_sample_info)}\n")
    lines.append("")
    lines.append("## Aggregate Tags (count / %)\n")
    tags = sorted(set().union(*[set(c.keys()) for c in tag_counts.values()]))
    lines.append("| Strategy | " + " | ".join(tags) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(tags)) + "|")
    for name in ["k1", "heur", "scorer"]:
        total = sum(tag_counts[name].values())
        row = [name]
        for t in tags:
            v = tag_counts[name][t]
            pct = 100.0 * v / total if total else 0.0
            row.append(f"{v} ({pct:.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Representative Cases\n")
    lines.append(f"Selection policy: top wins/regressions by (scorer ToolF1 − heuristic ToolF1), balanced across chain/dag. n={len(selected)}.\n")

    for sid in selected:
        info = per_sample_info[sid]
        gt_tools = info["gt_tools"]
        gt_edges = info["gt_edges"]
        picks = info["picks"]
        instruction = info["instruction"]
        task_type = info["task_type"]

        lines.append(f"### {sid} ({task_type})\n")
        lines.append(f"- Instruction: {short(instruction)}\n")
        lines.append(f"- GT tools ({len(set(gt_tools))}): {fmt_tools(gt_tools)}\n")
        lines.append(f"- GT edges ({len(set(gt_edges))}): {fmt_edges(gt_edges)}\n")
        lines.append("")
        lines.append("| Strategy | k | conf | ToolF1 | EdgeRec | missing_tools | extra_tools |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for name in ["k1", "heur", "scorer", "oracle"]:
            p = picks[name]
            _, _, _, missing, extra = tool_metrics(p.pred_tools, gt_tools)
            lines.append(
                f"| {name} | {p.k} | {p.confidence:.3f} | {p.tool_f1:.3f} | {p.edge_recall:.3f} | "
                f"{', '.join(missing) if missing else '-'} | {', '.join(extra) if extra else '-'} |"
            )
        lines.append("")
        lines.append("- Pred tools (scorer): " + fmt_tools(picks["scorer"].pred_tools) + "\n")
        lines.append("- Pred edges (scorer): " + fmt_edges(picks["scorer"].pred_edges) + "\n")
        lines.append("")

    import os

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {args.out_md}")


if __name__ == "__main__":
    main()
