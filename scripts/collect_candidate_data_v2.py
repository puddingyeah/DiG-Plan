#!/usr/bin/env python3
"""
Memory-efficient version of candidate data collection.
Processes in two phases:
1. DLM phase: Generate all proposals for all samples (with confidence for k=1)
2. AR phase: Refine all proposals

This reduces peak GPU memory from ~40GB (both models) to ~14GB (one model at a time).
"""

import argparse
import json
import os
import random
import gc
from typing import List, Dict, Tuple

import numpy as np
import torch

from tool_catalog import filter_samples_by_tools, get_tool_definitions
from utils import get_sample_instruction, get_sample_task_links, get_sample_task_nodes

from ar_refinement_eval import (
    DLMToolSelector,
    AREdgeBuilder,
    build_tool_selection_prompt,
    build_edge_construction_prompt,
    extract_tools_from_text,
    extract_edges_from_text,
    compute_metrics,
)

from dream_confidence_utils import compute_plan_confidence


def compute_heuristic_score(
    tools: List[str],
    edges: List[Tuple[str, str]],
    valid_tools: set,
) -> float:
    """计算 plan 的 heuristic score"""
    if not tools:
        return 0.0

    uniq_tools = list(sorted(set(tools)))

    # 结构合法性
    struct_score = 0.0
    if uniq_tools:
        node_set = set(uniq_tools)
        for src, tgt in edges:
            node_set.add(src)
            node_set.add(tgt)

        indeg = {n: 0 for n in node_set}
        adj = {n: [] for n in node_set}
        for src, tgt in edges:
            adj[src].append(tgt)
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

        if visited == len(node_set):
            struct_score += 0.4

    if uniq_tools:
        out_deg = {n: 0 for n in uniq_tools}
        for src, tgt in edges:
            if src in out_deg:
                out_deg[src] += 1
        if any(out_deg[n] == 0 for n in uniq_tools):
            struct_score += 0.3

    if len(uniq_tools) <= 1:
        struct_score += 0.3
    else:
        connected = set()
        for src, tgt in edges:
            connected.add(src)
            connected.add(tgt)
        if all(n in connected for n in uniq_tools):
            struct_score += 0.3

    n_tools = len(uniq_tools)
    if 1 <= n_tools <= 6:
        coverage_score = 1.0
    elif n_tools > 6:
        coverage_score = max(0.0, 1.0 - 0.1 * (n_tools - 6))
    else:
        coverage_score = 0.5

    valid_count = sum(1 for t in uniq_tools if t in valid_tools)
    coherence_score = valid_count / len(uniq_tools) if uniq_tools else 0.0

    final_score = 0.3 * struct_score + 0.3 * coverage_score + 0.4 * coherence_score
    return float(max(0.0, min(1.0, final_score)))


def select_subset(
    samples: List[Dict],
    max_single: int,
    max_chain: int,
    max_dag: int,
    seed: int = 42,
    shard_idx: int = None,
    num_shards: int = None,
) -> List[Dict]:
    # Default behavior: exclude single unless max_single > 0 (backward compatible).
    single = [s for s in samples if s.get("type") == "single"]
    chain = [s for s in samples if s.get("type") == "chain"]
    dag = [s for s in samples if s.get("type") == "dag"]
    random.seed(seed)
    random.shuffle(single)
    random.shuffle(chain)
    random.shuffle(dag)

    # max_* 表示“全局总量”（在 sharding 前先截断），这样多卡并行时互不重叠。
    if max_single is not None:
        single = single[:max_single]
    if max_chain is not None:
        chain = chain[:max_chain]
    if max_dag is not None:
        dag = dag[:max_dag]

    # IMPORTANT: shard over the combined list, not per-type.
    # Per-type sharding can produce empty shards even when (chain+dag) >> num_shards
    # (e.g., chain=200, dag=200, num_shards=32 => last few per-type shards become empty).
    selected = single + chain + dag
    random.shuffle(selected)

    if shard_idx is not None and num_shards is not None:
        if shard_idx < 0 or shard_idx >= num_shards:
            raise ValueError(f"Invalid shard_idx={shard_idx}, num_shards={num_shards}")
        n = len(selected)
        if n == 0:
            return []
        # Use floor-based partition boundaries to avoid empty shards when n >= num_shards.
        # (ceil-based shard_size can create an empty last shard when (num_shards-1)*ceil(n/num_shards) >= n)
        start = (shard_idx * n) // num_shards
        end = ((shard_idx + 1) * n) // num_shards
        return selected[start:end]

    return selected


def collect_candidate_data(
    dlm_path: str,
    ar_path: str,
    data_path: str,
    tool_desc_path: str = "",
    ids_file: str = "",
    output_path: str = "",
    device: str = "cuda:0",
    K: int = 5,
    max_single: int = 0,
    max_chain: int = 50,
    max_dag: int = 50,
    dlm_steps: int = 128,
    ar_max_tokens: int = 512,
    seed: int = 42,
    shard_idx: int = None,
    num_shards: int = None,
) -> None:
    print(f"\n{'=' * 60}")
    print("Collecting Per-Candidate Data (Memory-Efficient Version)")
    print(f"K = {K}, device = {device}")
    print(f"{'=' * 60}\n")

    # Load data
    with open(data_path, "r") as f:
        all_samples = [json.loads(line) for line in f]

    tool_list = get_tool_definitions(tool_desc_path or None)
    tool_names = [t["name"] for t in tool_list]
    valid_tools = set(tool_names)
    filtered_samples = filter_samples_by_tools(all_samples, valid_tools)

    if not filtered_samples:
        raise ValueError("No samples remain after filtering.")

    print(f"Total canonical tools: {len(valid_tools)}")
    print(f"Filtered samples: {len(filtered_samples)} / {len(all_samples)}")

    if ids_file and os.path.exists(ids_file):
        with open(ids_file, "r") as f:
            target_ids = set(line.strip() for line in f if line.strip())
        samples = [s for s in filtered_samples if s.get("id", "") in target_ids]
        print(f"Loaded {len(target_ids)} IDs from {ids_file}, matched {len(samples)} samples")
    else:
        samples = filtered_samples

    subset = select_subset(
        samples,
        max_single=max_single,
        max_chain=max_chain,
        max_dag=max_dag,
        seed=seed,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    n_single = sum(1 for s in subset if s.get("type") == "single")
    n_chain = sum(1 for s in subset if s.get("type") == "chain")
    n_dag = sum(1 for s in subset if s.get("type") == "dag")
    if shard_idx is not None and num_shards is not None:
        print(f"Shard: {shard_idx}/{num_shards}")
    print(f"Using {len(subset)} samples (Single:{n_single}, Chain:{n_chain}, DAG:{n_dag})\n")

    # If this shard has no samples (can happen when num_shards > n_samples),
    # write an empty-but-valid shard JSON and exit early. This prevents watchers
    # from hanging and avoids unnecessary model loading.
    if len(subset) == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = {
            "meta": {
                "K": K,
                "n_samples": 0,
                "n_candidates": 0,
                "max_single": max_single,
                "max_chain": max_chain,
                "max_dag": max_dag,
                "seed": seed,
                "shard_idx": shard_idx,
                "num_shards": num_shards,
                "device": device,
                "dlm_path": dlm_path,
                "ar_path": ar_path,
                "tool_desc_path": tool_desc_path,
                "ids_file": ids_file,
            },
            "summary": {
                "avg_k1_f1": 0.0,
                "avg_best_f1": 0.0,
                "f1_range": [0.0, 0.0],
            },
            "sample_summaries": [],
            "candidates": [],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote empty shard: {output_path}", flush=True)
        return

    # ==========================================================
    # Phase 1: DLM Proposals
    # ==========================================================
    print("=" * 60)
    print("Phase 1: Generating DLM Proposals")
    print("=" * 60)

    dlm = DLMToolSelector(dlm_path, device)

    # Store proposals: list of (sample_idx, k, selected_tools, confidence, sample_info)
    proposals = []

    for idx, s in enumerate(subset):
        sample_id = s.get("id", f"sample-{idx}")
        task_type = s.get("type", "unknown")
        instruction = get_sample_instruction(s)
        gt_nodes = get_sample_task_nodes(s)
        gt_tools = [n.get("task", "") for n in gt_nodes]
        gt_links = get_sample_task_links(s)
        gt_edges = [(l.get("source", ""), l.get("target", "")) for l in gt_links]

        tool_prompt = build_tool_selection_prompt(instruction, tool_list)

        sample_info = {
            "sample_id": sample_id,
            "task_type": task_type,
            "instruction": instruction,
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
        }

        k1_conf = None

        for k in range(1, K + 1):
            seed = 1000 * idx + k
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            try:
                if k == 1:
                    text, history = dlm.select_tools_with_confidence(
                        tool_prompt,
                        max_tokens=256,
                        steps=dlm_steps,
                        timeout=180,
                    )
                    conf = compute_plan_confidence(history.get("mask_entropies", []))
                    k1_conf = conf
                else:
                    text = dlm.select_tools(tool_prompt, max_tokens=256, steps=dlm_steps)
                    conf = k1_conf  # Use k=1 confidence for all k

                selected = extract_tools_from_text(text, valid_tools)
            except Exception as e:
                print(f"  [WARNING] Sample {idx+1}, k={k} proposer failed: {e}")
                selected = []
                conf = k1_conf

            proposals.append({
                "sample_idx": idx,
                "k": k,
                "selected_tools": selected,
                "confidence": conf,
                "sample_info": sample_info,
            })

        conf_str = f"{k1_conf:.3f}" if k1_conf is not None else "N/A"
        print(f"[{idx+1}/{len(subset)}] {task_type}: {K} proposals generated, conf={conf_str}")

    print(f"\nPhase 1 complete: {len(proposals)} proposals generated")

    # Free DLM memory
    print("\nFreeing DLM model memory...")
    del dlm
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================
    # Phase 2: AR Refinement
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 2: AR Refinement")
    print("=" * 60)

    ar = AREdgeBuilder(ar_path, device)

    all_candidates = []

    for i, prop in enumerate(proposals):
        selected = prop["selected_tools"]
        sample_info = prop["sample_info"]
        instruction = sample_info["instruction"]
        gt_tools = sample_info["gt_tools"]
        gt_edges = sample_info["gt_edges"]

        if selected:
            try:
                edge_prompt = build_edge_construction_prompt(instruction, selected, tool_list)
                plan_text = ar.build_plan(edge_prompt, max_tokens=ar_max_tokens)
                pred_tools = extract_tools_from_text(plan_text, valid_tools)
                pred_edges = extract_edges_from_text(plan_text)
            except Exception as e:
                print(f"  [WARNING] Proposal {i+1} refinement failed: {e}")
                pred_tools, pred_edges = selected, []
        else:
            pred_tools, pred_edges = [], []

        # Compute metrics
        metrics = compute_metrics(pred_tools, gt_tools, pred_edges, gt_edges)
        heur_score = compute_heuristic_score(pred_tools, pred_edges, valid_tools)

        pred_set = set(pred_tools)
        gt_set = set(gt_tools)
        if pred_set and gt_set:
            tp = len(pred_set & gt_set)
            precision = tp / len(pred_set)
            recall = tp / len(gt_set)
        elif not pred_set and not gt_set:
            precision, recall = 1.0, 1.0
        else:
            precision, recall = 0.0, 0.0

        candidate = {
            "sample_id": sample_info["sample_id"],
            "task_type": sample_info["task_type"],
            "k": prop["k"],
            "confidence": prop["confidence"],
            "pred_tools": pred_tools,
            "pred_edges": pred_edges,
            "n_pred_tools": len(pred_tools),
            "n_pred_edges": len(pred_edges),
            "n_gt_tools": len(gt_tools),
            "n_gt_edges": len(gt_edges),
            "heuristic_score": heur_score,
            "tool_f1": metrics["tool_f1"],
            "tool_precision": precision,
            "tool_recall": recall,
            "edge_recall": metrics["edge_recall"],
        }
        all_candidates.append(candidate)

        if (i + 1) % 50 == 0:
            print(f"  Refined {i+1}/{len(proposals)} proposals...")

    print(f"\nPhase 2 complete: {len(all_candidates)} candidates refined")

    # Free AR memory
    del ar
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================
    # Compute summaries and save
    # ==========================================================
    print("\n" + "=" * 60)
    print("Computing summaries...")
    print("=" * 60)

    # Group by sample
    sample_groups = {}
    for c in all_candidates:
        sid = c["sample_id"]
        if sid not in sample_groups:
            sample_groups[sid] = []
        sample_groups[sid].append(c)

    sample_summaries = []
    for sid, candidates in sample_groups.items():
        f1s = [c["tool_f1"] for c in candidates]
        best_f1 = max(f1s) if f1s else 0.0
        k1 = next((c for c in candidates if c.get("k") == 1), None)
        k1_f1 = k1["tool_f1"] if k1 else (candidates[0]["tool_f1"] if candidates else 0.0)
        k1_conf = k1["confidence"] if k1 else (candidates[0]["confidence"] if candidates else None)

        # Find GT info from first candidate
        gt_tools = []
        gt_edges = []
        task_type = "unknown"
        for c in candidates:
            if c["k"] == 1:
                task_type = c["task_type"]
                break

        sample_summaries.append({
            "sample_id": sid,
            "task_type": task_type,
            "k1_f1": k1_f1,
            "best_f1": best_f1,
            "k1_confidence": k1_conf,
        })

    total_candidates = len(all_candidates)
    avg_k1_f1 = np.mean([s["k1_f1"] for s in sample_summaries])
    avg_best_f1 = np.mean([s["best_f1"] for s in sample_summaries])
    f1_values = [c["tool_f1"] for c in all_candidates]

    print(f"\n{'=' * 60}")
    print("Data Collection Summary")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(subset)}")
    print(f"Total candidates: {total_candidates}")
    print(f"K per sample: {K}")
    print(f"Average K=1 Tool F1: {avg_k1_f1:.3f}")
    print(f"Average best-of-{K} Tool F1: {avg_best_f1:.3f}")
    print(f"Tool F1 range: [{min(f1_values):.3f}, {max(f1_values):.3f}]")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output_data = {
        "meta": {
            "K": K,
            "n_samples": len(subset),
            "n_candidates": total_candidates,
            "max_single": max_single,
            "max_chain": max_chain,
            "max_dag": max_dag,
            "seed": seed,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "device": device,
            "dlm_path": dlm_path,
            "ar_path": ar_path,
            "tool_desc_path": tool_desc_path,
            "ids_file": ids_file,
        },
        "summary": {
            "avg_k1_f1": avg_k1_f1,
            "avg_best_f1": avg_best_f1,
            "f1_range": [min(f1_values), max(f1_values)],
        },
        "sample_summaries": sample_summaries,
        "candidates": all_candidates,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nData saved to: {output_path}")
    print("Ready for Value Function training!")


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-candidate data (memory-efficient version)"
    )
    parser.add_argument("--dlm_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    parser.add_argument("--tool_desc_path", default="", help="Optional TaskBench tool_desc.json to define the tool library.")
    parser.add_argument("--ids_file", default="data/ids_500.txt")
    parser.add_argument("--output_path", default="results/taskbench_dream_k5.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--max_single", type=int, default=0)
    parser.add_argument("--max_chain", type=int, default=50)
    parser.add_argument("--max_dag", type=int, default=50)
    parser.add_argument("--dlm_steps", type=int, default=128)
    parser.add_argument("--ar_max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    collect_candidate_data(**vars(args))


if __name__ == "__main__":
    main()
