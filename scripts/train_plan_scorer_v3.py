#!/usr/bin/env python3
"""
Train a DiG-Plan value function using per-candidate data,
with a *deployable* feature set (no GT-derived features, no dataset-only labels).

Input:
  artifacts/candidate_pools/taskbench_dream_k5_train670.json
    - contains per-candidate fields: confidence, heuristic_score, pred_tools/pred_edges,
      n_pred_tools, n_pred_edges, tool_f1 (label), edge_recall (aux label), sample_id, k, ...

Output:
  results/plan_scorer_combo07_toolset.pkl
    - trained model + feature_names + feature_set + train/test sample_ids + evaluation metrics

Feature sets:
  - basic: confidence + heuristic + counts + is_dag_pred + k
  - toolset: basic + 23-dim multi-hot over canonical tool names

Note:
  - This is designed to be *deployable* at inference time (no GT-based ratios).
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor


from tool_catalog import get_tool_definitions


def _is_dag(nodes: List[str], edges: List[Tuple[str, str]]) -> float:
    """Return 1.0 if DAG else 0.0 (computed from predicted graph only)."""
    if not nodes:
        return 0.0
    node_set = set(nodes)
    for src, tgt in edges:
        node_set.add(src)
        node_set.add(tgt)
    if not node_set:
        return 0.0

    indeg = {n: 0 for n in node_set}
    adj: Dict[str, List[str]] = {n: [] for n in node_set}
    for src, tgt in edges:
        adj.setdefault(src, []).append(tgt)
        indeg[tgt] = indeg.get(tgt, 0) + 1

    queue = [n for n in node_set if indeg.get(n, 0) == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return 1.0 if visited == len(node_set) else 0.0


def load_candidate_data(data_path: str) -> Tuple[List[Dict], Dict]:
    with open(data_path, "r") as f:
        data = json.load(f)
    candidates = data.get("candidates", [])
    meta = data.get("meta", {})
    summary = data.get("summary", {})
    print(f"Loaded {len(candidates)} candidates from {data_path}")
    print(f"  meta: K={meta.get('K')}, n_samples={meta.get('n_samples')}, n_candidates={meta.get('n_candidates')}")
    print(f"  summary: avg_k1_f1={summary.get('avg_k1_f1', 0):.3f}, avg_best_f1={summary.get('avg_best_f1', 0):.3f}")
    return candidates, meta


def _tool_to_idx(tool_desc_path: str | None = None) -> Dict[str, int]:
    """
    Build a tool-name -> index mapping.

    - If tool_desc_path is provided, use that tool library (for non-TaskBench domains like API-Bank).
    - Else, fall back to TaskBench canonical 23 tools.
    """
    tool_desc_path = (tool_desc_path or "").strip()
    tools = get_tool_definitions(tool_desc_path or None)
    return {str(t["name"]): i for i, t in enumerate(tools)}


def _toolset_vector(pred_tools: List[str], tool_to_idx: Dict[str, int]) -> np.ndarray:
    v = np.zeros(len(tool_to_idx), dtype=np.float32)
    for t in pred_tools:
        idx = tool_to_idx.get(t)
        if idx is not None:
            v[idx] = 1.0
    return v


def _edge_vector(pred_edges: List[Tuple[str, str]], tool_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Directed adjacency multi-hot over canonical tools.
    Size: |T| * |T|. Entry (i,j)=1 if edge tool_i -> tool_j is present in predicted graph.
    Deployable: computed from predicted edges only.
    """
    n = len(tool_to_idx)
    v = np.zeros(n * n, dtype=np.float32)
    for s, t in pred_edges:
        si = tool_to_idx.get(s)
        ti = tool_to_idx.get(t)
        if si is None or ti is None:
            continue
        v[si * n + ti] = 1.0
    return v


def extract_features(candidate: Dict, feature_set: str, tool_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Deployable feature sets:
      basic:
        - confidence
        - heuristic_score
        - n_pred_tools
        - n_pred_edges
        - is_dag_pred (from predicted graph)
        - k (candidate index 1..K)
      toolset:
        - basic + tool multi-hot (canonical tool names, 23 dims)
      graph:
        - toolset + directed edge multi-hot (|T|^2 dims)
    """
    conf = candidate.get("confidence", 0.5)
    if conf is None:
        conf = 0.5
    heur = candidate.get("heuristic_score", 0.5)
    n_pred_tools = float(candidate.get("n_pred_tools", 0))
    n_pred_edges = float(candidate.get("n_pred_edges", 0))
    k = float(candidate.get("k", 1))

    pred_tools = candidate.get("pred_tools", []) or []
    pred_edges = candidate.get("pred_edges", []) or []
    # pred_edges is list of [src, tgt] or tuple; normalize
    edges = []
    for e in pred_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            edges.append((str(e[0]), str(e[1])))
    is_dag_pred = _is_dag([str(t) for t in pred_tools], edges)

    base = np.array([conf, heur, n_pred_tools, n_pred_edges, is_dag_pred, k], dtype=np.float32)
    if feature_set == "basic":
        return base
    if feature_set == "toolset":
        tool_vec = _toolset_vector([str(t) for t in pred_tools], tool_to_idx)
        return np.concatenate([base, tool_vec], axis=0)
    if feature_set == "graph":
        tool_vec = _toolset_vector([str(t) for t in pred_tools], tool_to_idx)
        edge_vec = _edge_vector(edges, tool_to_idx)
        return np.concatenate([base, tool_vec, edge_vec], axis=0)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _get_label(candidate: Dict, label: str, combo_alpha: float) -> float:
    tool_f1 = float(candidate.get("tool_f1", 0.0))
    edge_recall = float(candidate.get("edge_recall", 0.0))
    if label == "tool_f1":
        return tool_f1
    if label == "edge_recall":
        return edge_recall
    if label == "combo":
        a = float(combo_alpha)
        a = max(0.0, min(1.0, a))
        return a * tool_f1 + (1.0 - a) * edge_recall
    raise ValueError(f"Unknown label: {label}")


def prepare_xy(
    candidates: List[Dict],
    feature_set: str,
    tool_to_idx: Dict[str, int],
    label: str,
    combo_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    X = np.stack([extract_features(c, feature_set, tool_to_idx) for c in candidates])
    y = np.array([_get_label(c, label, combo_alpha) for c in candidates], dtype=np.float32)
    return X, y, candidates


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str) -> Any:
    if model_type == "xgb" and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        ).fit(X_train, y_train)
    # fallback: sklearn GBM
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    ).fit(X_train, y_train)


def _group_sort(idxs: List[int], meta: List[Dict]) -> Tuple[List[int], List[int]]:
    """
    Sort candidate indices by sample_id and return (sorted_indices, group_sizes).
    Required for learning-to-rank: all candidates for a query (sample_id) must be contiguous.
    """

    def key(i: int):
        m = meta[i]
        return (str(m.get("sample_id")), int(m.get("k", 1)))

    sidx = sorted(list(idxs), key=key)

    group_sizes: List[int] = []
    last_sid = None
    cur = 0
    for i in sidx:
        sid = str(meta[i].get("sample_id"))
        if last_sid is None:
            last_sid = sid
            cur = 1
        elif sid == last_sid:
            cur += 1
        else:
            group_sizes.append(cur)
            last_sid = sid
            cur = 1
    if last_sid is not None:
        group_sizes.append(cur)

    return sidx, group_sizes


def train_ranker(X_train: np.ndarray, y_train: np.ndarray, group_train: List[int], model_type: str) -> Any:
    """
    Train a pairwise ranker: directly optimizes within-sample candidate ordering.
    This matches our downstream selection objective (pick best candidate among K).
    """
    if model_type != "xgb" or not HAS_XGB:
        raise RuntimeError("Ranking requires xgboost (model_type=xgb).")
    model = xgb.XGBRanker(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="rank:pairwise",
        random_state=42,
    )
    model.fit(X_train, y_train, group=group_train)
    return model


class PairwiseLinearRanker:
    """
    Simple deployable learning-to-rank baseline that does NOT require xgboost.

    We learn a linear scoring function s(x)=w^T x by fitting a classifier on pairwise
    differences: given two candidates (i,j) from the same sample, predict whether i>j.
    """

    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray, meta: List[Dict]) -> "PairwiseLinearRanker":
        from collections import defaultdict

        groups = defaultdict(list)
        for i, m in enumerate(meta):
            groups[str(m.get("sample_id"))].append(i)

        diffs = []
        labels = []
        for _, idxs in groups.items():
            # all pairs within group
            for a_i in range(len(idxs)):
                for b_i in range(a_i + 1, len(idxs)):
                    ia = idxs[a_i]
                    ib = idxs[b_i]
                    ya = float(y[ia])
                    yb = float(y[ib])
                    if ya == yb:
                        continue
                    if ya > yb:
                        diffs.append(X[ia] - X[ib])
                        labels.append(1)
                    else:
                        diffs.append(X[ia] - X[ib])
                        labels.append(0)

        if not diffs:
            raise RuntimeError("No training pairs (all labels tied?).")
        D = np.stack(diffs, axis=0)
        L = np.array(labels, dtype=np.int64)
        self.clf.fit(D, L)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # decision_function is a better ranking score than predict_proba for linear models
        return self.clf.decision_function(X)


def evaluate_ranking(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    meta: List[Dict],
    label_name: str,
) -> Dict[str, float]:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    sp, _ = spearmanr(y, y_pred)
    pr, _ = pearsonr(y, y_pred)

    # group by sample_id: model-selected vs oracle
    from collections import defaultdict

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["sample_id"]].append(i)

    model_selected = []
    oracle = []
    top1_hit = 0
    total = 0
    for sid, idxs in groups.items():
        if not idxs:
            continue
        # pick max predicted
        best_pred_i = max(idxs, key=lambda i: float(y_pred[i]))
        best_gt_i = max(idxs, key=lambda i: float(y[i]))
        model_selected.append(float(y[best_pred_i]))
        oracle.append(float(y[best_gt_i]))
        if best_pred_i == best_gt_i:
            top1_hit += 1
        total += 1

    avg_model = float(np.mean(model_selected)) if model_selected else 0.0
    avg_oracle = float(np.mean(oracle)) if oracle else 0.0
    top1_acc = top1_hit / max(total, 1)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "spearman_rho": float(sp),
        "pearson_r": float(pr),
        "top1_accuracy": float(top1_acc),
        "avg_model_selected": avg_model,
        "avg_oracle": avg_oracle,
        "gap_to_oracle": float(avg_oracle - avg_model),
        "label": str(label_name),
        "n_groups": int(total),
        "n_candidates": int(len(y)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train plan scorer v3 (deployable features, group split by sample_id)")
    ap.add_argument("--data_path", default="artifacts/candidate_pools/taskbench_dream_k5_train670.json")
    ap.add_argument("--output_path", default="results/plan_scorer_combo07_toolset.pkl")
    ap.add_argument("--model_type", default="gbm", choices=["xgb", "gbm"])
    ap.add_argument(
        "--train_mode",
        default="reg",
        choices=["reg", "rank"],
        help="reg=regression; rank=pairwise learning-to-rank within each sample_id.",
    )
    ap.add_argument("--feature_set", default="toolset", choices=["basic", "toolset", "graph"])
    ap.add_argument("--label", default="combo", choices=["tool_f1", "edge_recall", "combo"])
    ap.add_argument("--combo_alpha", type=float, default=0.7, help="For label=combo: weight on tool_f1 (1-alpha on edge_recall)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tool_desc_path",
        default="",
        help="Optional TaskBench-style tool_desc.json to define the tool library (e.g., API-Bank).",
    )
    ap.add_argument(
        "--ablate_confidence",
        action="store_true",
        help="Ablation: ignore candidate confidence and use constant 0.5.",
    )
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("Training Value Function (Plan Scorer v3, deployable features)")
    print("=" * 60 + "\n")

    candidates, meta_file = load_candidate_data(args.data_path)
    if args.ablate_confidence:
        for c in candidates:
            c["confidence"] = 0.5
    tool_to_idx = _tool_to_idx(args.tool_desc_path or None)
    X, y, meta = prepare_xy(candidates, args.feature_set, tool_to_idx, args.label, args.combo_alpha)
    print(f"Feature dim: {X.shape[1]}, candidates={X.shape[0]}")
    print(f"Label mean: {float(y.mean()):.4f}, range=[{float(y.min()):.3f}, {float(y.max()):.3f}]")

    # group split by sample_id
    unique_ids = sorted({m["sample_id"] for m in meta})
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_ids)
    n_test = max(1, int(len(unique_ids) * args.test_size))
    test_ids = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    train_idx = [i for i, m in enumerate(meta) if m["sample_id"] in train_ids]
    test_idx = [i for i, m in enumerate(meta) if m["sample_id"] in test_ids]

    if args.train_mode == "rank":
        train_idx_sorted, group_train = _group_sort(train_idx, meta)
        test_idx_sorted, _group_test = _group_sort(test_idx, meta)
        X_train, y_train = X[train_idx_sorted], y[train_idx_sorted]
        X_test, y_test = X[test_idx_sorted], y[test_idx_sorted]
        meta_train = [meta[i] for i in train_idx_sorted]
        meta_test = [meta[i] for i in test_idx_sorted]
        if args.model_type == "xgb" and HAS_XGB:
            model = train_ranker(X_train, y_train, group_train, args.model_type)
        else:
            model = PairwiseLinearRanker().fit(X_train, y_train, meta_train)
    else:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        meta_train = [meta[i] for i in train_idx]
        meta_test = [meta[i] for i in test_idx]
        model = train_model(X_train, y_train, args.model_type)

    print(f"Unique samples: {len(unique_ids)} (train={len(train_ids)}, test={len(test_ids)})")
    print(f"Candidates: train={len(train_idx)}, test={len(test_idx)}")

    label_name = args.label if args.label != "combo" else f"combo_{args.combo_alpha:.2f}"
    train_metrics = evaluate_ranking(model, X_train, y_train, meta_train, label_name)
    test_metrics = evaluate_ranking(model, X_test, y_test, meta_test, label_name)

    print("\n=== Train metrics ===")
    for k, v in train_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Test metrics (held-out sample_ids) ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    feature_names = [
        "confidence",
        "heuristic_score",
        "n_pred_tools",
        "n_pred_edges",
        "is_dag_pred",
        "k",
    ]
    if args.feature_set in ("toolset", "graph"):
        # stable order
        inv = {i: t for t, i in tool_to_idx.items()}
        feature_names.extend([f"tool_{inv[i]}" for i in range(len(inv))])
    if args.feature_set == "graph":
        inv = {i: t for t, i in tool_to_idx.items()}
        n = len(inv)
        # directed adjacency (i->j)
        for i in range(n):
            for j in range(n):
                feature_names.append(f"edge_{inv[i]}__to__{inv[j]}")

    with open(args.output_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "feature_set": args.feature_set,
                "label": label_name,
                "tool_to_idx": tool_to_idx,
                "ablate_confidence": bool(args.ablate_confidence),
                "train_mode": str(args.train_mode),
                "train_ids": sorted(train_ids),
                "test_ids": sorted(test_ids),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "data_meta": meta_file,
            },
            f,
        )
    print(f"\nSaved model to {args.output_path}")


if __name__ == "__main__":
    main()
