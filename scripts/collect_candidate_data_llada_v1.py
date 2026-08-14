#!/usr/bin/env python3
"""
Collect per-candidate data for an LLaDA proposer baseline (LLaDA proposer + AR refiner).

Goal (IJCAI generalization):
  - Provide a second diffusion-family proposer instance beyond Dream.
  - Build a K>1 candidate pool with the same AR refiner and the same evaluation protocol.

Output JSON format matches scripts/collect_candidate_data_v2.py expectations:
  { meta, summary, sample_summaries, candidates }

Notes:
  - LLaDA confidence is not extracted in this script (confidence=None).
    Selection scripts handle missing confidence (default 0.5).
  - Uses a memory-friendly two-phase pipeline:
      Phase 1: LLaDA proposes tool sets for all samples/candidates
      Phase 2: AR refines edges for all proposed tool sets
"""

import argparse
import gc
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from tool_catalog import filter_samples_by_tools, get_tool_definitions
from ar_refinement_eval import (
    AREdgeBuilder,
    build_tool_selection_prompt,
    build_edge_construction_prompt,
    extract_tools_from_text,
    extract_edges_from_text,
    compute_metrics,
)


_NP_SEED_MOD = 2**32 - 1


def _seed32(x: int) -> int:
    # NumPy requires 0 <= seed < 2**32
    return int(x % _NP_SEED_MOD)


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _load_ids(ids_file: str) -> set:
    if not ids_file:
        return set()
    with open(ids_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def compute_heuristic_score(
    tools: List[str],
    edges: List[Tuple[str, str]],
    valid_tools: set,
) -> float:
    if not tools:
        return 0.0

    uniq_tools = list(sorted(set(tools)))

    # structure validity (DAG check over nodes in nodes/edges)
    struct_score = 0.0
    node_set = set(uniq_tools)
    for src, tgt in edges:
        node_set.add(src)
        node_set.add(tgt)

    if node_set:
        indeg = {n: 0 for n in node_set}
        adj = {n: [] for n in node_set}
        for src, tgt in edges:
            adj.setdefault(src, []).append(tgt)
            indeg[tgt] = indeg.get(tgt, 0) + 1
        q = [n for n in node_set if indeg.get(n, 0) == 0]
        visited = 0
        while q:
            u = q.pop()
            visited += 1
            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if visited == len(node_set):
            struct_score += 0.4

    # sink check (at least one out-degree==0 among selected tools)
    if uniq_tools:
        out_deg = {n: 0 for n in uniq_tools}
        for src, _tgt in edges:
            if src in out_deg:
                out_deg[src] += 1
        if any(out_deg[n] == 0 for n in uniq_tools):
            struct_score += 0.3

    # orphan check
    if len(uniq_tools) <= 1:
        struct_score += 0.3
    else:
        connected = set()
        for src, tgt in edges:
            connected.add(src)
            connected.add(tgt)
        if all(n in connected for n in uniq_tools):
            struct_score += 0.3

    # coverage
    n_tools = len(uniq_tools)
    if 1 <= n_tools <= 6:
        coverage_score = 1.0
    elif n_tools > 6:
        coverage_score = max(0.0, 1.0 - 0.1 * (n_tools - 6))
    else:
        coverage_score = 0.5

    # coherence: tool names valid
    valid_count = sum(1 for t in uniq_tools if t in valid_tools)
    coherence_score = valid_count / len(uniq_tools) if uniq_tools else 0.0

    final_score = 0.3 * struct_score + 0.3 * coverage_score + 0.4 * coherence_score
    return float(max(0.0, min(1.0, final_score)))


def select_subset(
    samples: List[Dict],
    max_chain: int,
    max_dag: int,
    seed: int,
    shard_idx: int | None,
    num_shards: int | None,
) -> List[Dict]:
    chain = [s for s in samples if s.get("type") == "chain"]
    dag = [s for s in samples if s.get("type") == "dag"]
    rng = random.Random(seed)
    rng.shuffle(chain)
    rng.shuffle(dag)

    if max_chain is not None:
        chain = chain[:max_chain]
    if max_dag is not None:
        dag = dag[:max_dag]

    selected = chain + dag
    rng.shuffle(selected)

    if shard_idx is None or num_shards is None:
        return selected

    n = len(selected)
    if n == 0:
        return []
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"Invalid shard_idx={shard_idx}, num_shards={num_shards}")

    start = (shard_idx * n) // num_shards
    end = ((shard_idx + 1) * n) // num_shards
    return selected[start:end]


def _ensure_llada_generate_on_path() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    llada_repo_path = os.path.join(project_root, "llada_repo")
    if llada_repo_path not in sys.path:
        sys.path.append(llada_repo_path)


def collect_llada_candidates(
    llada_path: str,
    ar_path: str,
    data_path: str,
    tool_desc_path: str,
    ids_file: str,
    output_path: str,
    device: str,
    K: int,
    max_chain: int,
    max_dag: int,
    seed: int,
    llada_steps: int,
    llada_max_tokens: int,
    llada_temperature: float,
    ar_max_tokens: int,
    shard_idx: int | None,
    num_shards: int | None,
) -> None:
    print("\n=== Collect Candidates (LLaDA proposer baseline) ===", flush=True)
    print(f"device={device} K={K} seed={seed} shard={shard_idx}/{num_shards}", flush=True)
    print(
        f"llada: steps={llada_steps} max_tokens={llada_max_tokens} temp={llada_temperature} | "
        f"ar_max_tokens={ar_max_tokens}",
        flush=True,
    )

    all_samples = _load_jsonl(data_path)
    tool_list = get_tool_definitions(tool_desc_path or None)
    valid_tools = set(t["name"] for t in tool_list)
    samples = filter_samples_by_tools(all_samples, valid_tools)

    target_ids = _load_ids(ids_file)
    if target_ids:
        samples = [s for s in samples if str(s.get("id", "")) in target_ids]
        print(f"ids_file={ids_file} matched={len(samples)}", flush=True)

    subset = select_subset(
        samples=samples,
        max_chain=max_chain,
        max_dag=max_dag,
        seed=seed,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    n_chain = sum(1 for s in subset if s.get("type") == "chain")
    n_dag = sum(1 for s in subset if s.get("type") == "dag")
    print(f"subset={len(subset)} (chain={n_chain}, dag={n_dag})", flush=True)

    if len(subset) == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = {
            "meta": {
                "proposer": "llada",
                "K": K,
                "n_samples": 0,
                "n_candidates": 0,
                "max_chain": max_chain,
                "max_dag": max_dag,
                "seed": seed,
                "shard_idx": shard_idx,
                "num_shards": num_shards,
                "device": device,
                "llada_path": llada_path,
                "ar_path": ar_path,
                "tool_desc_path": tool_desc_path,
                "llada_steps": llada_steps,
                "llada_max_tokens": llada_max_tokens,
                "llada_temperature": llada_temperature,
                "ar_max_tokens": ar_max_tokens,
                "ids_file": ids_file,
            },
            "summary": {"avg_k1_f1": 0.0, "avg_best_f1": 0.0, "f1_range": [0.0, 0.0]},
            "sample_summaries": [],
            "candidates": [],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote (empty shard): {output_path}", flush=True)
        return

    # -------------------------
    # Phase 1: LLaDA proposals
    # -------------------------
    _ensure_llada_generate_on_path()
    from generate import generate as llada_generate  # type: ignore

    print(f"[LLaDA] Loading model from {llada_path} on {device} ...", flush=True)
    llada = AutoModel.from_pretrained(
        llada_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    tok = AutoTokenizer.from_pretrained(llada_path, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    # LLaDA-8B uses "[MASK]" (126336) but some newer repos (e.g., LLaDA-MoE) define `<|mask|>` as `mask_token`.
    mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        mask_id = tok.convert_tokens_to_ids("[MASK]")
    if mask_id is None:
        mask_id = 126336

    proposals: List[Dict] = []
    for idx, s in enumerate(subset):
        sid = str(s.get("id", f"sample-{idx}"))
        instruction = s.get("instruction", "")
        tool_prompt = build_tool_selection_prompt(instruction, tool_list)

        per_sample = []
        for k in range(1, K + 1):
            combined = seed * 100000 + idx * 1000 + k
            s32 = _seed32(combined)
            torch.manual_seed(s32)
            np.random.seed(s32)
            random.seed(s32)

            messages = [{"role": "user", "content": tool_prompt}]
            chat_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = tok(
                chat_prompt,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            out = llada_generate(
                llada,
                input_ids,
                attention_mask=attention_mask,
                steps=llada_steps,
                gen_length=llada_max_tokens,
                block_length=min(128, llada_max_tokens),
                temperature=llada_temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=mask_id,
            )
            gen_tokens = out[:, input_ids.shape[1] :]
            text = tok.batch_decode(gen_tokens, skip_special_tokens=True)[0]
            sel_tools = extract_tools_from_text(text, valid_tools)

            per_sample.append(
                {
                    "sample_id": sid,
                    "k": k,
                    "selected_tools": sel_tools,
                    "tool_prompt": tool_prompt,
                }
            )

        proposals.extend(per_sample)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(subset):
            print(f"[LLaDA] Proposed tools for {idx+1}/{len(subset)} samples", flush=True)

    del llada
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------
    # Phase 2: AR refinement
    # -------------------------
    ar = AREdgeBuilder(ar_path, device)

    # Build GT lookup per sample_id
    gt_lookup: Dict[str, Dict] = {}
    sid_to_idx: Dict[str, int] = {}
    for idx, s in enumerate(subset):
        sid = str(s.get("id", ""))
        sid_to_idx[sid] = idx
        gt_nodes = json.loads(s.get("tool_nodes", "[]"))
        gt_tools = [n.get("task", "") for n in gt_nodes]
        gt_links = json.loads(s.get("tool_links", "[]"))
        gt_edges = [(l.get("source", ""), l.get("target", "")) for l in gt_links]
        gt_lookup[sid] = {"gt_tools": gt_tools, "gt_edges": gt_edges, "task_type": s.get("type", "unknown"), "instruction": s.get("instruction", "")}

    candidates: List[Dict] = []
    sample_summaries: Dict[str, Dict] = {}

    for i, p in enumerate(proposals):
        sid = str(p["sample_id"])
        k = int(p["k"])
        idx = int(sid_to_idx.get(sid, 0))
        gt = gt_lookup[sid]
        task_type = gt["task_type"]
        instruction = gt["instruction"]
        gt_tools = gt["gt_tools"]
        gt_edges = gt["gt_edges"]

        sel_tools = p["selected_tools"]
        if sel_tools:
            combined = seed * 100000 + idx * 1000 + k + 999_999
            s32 = _seed32(combined)
            torch.manual_seed(s32)
            np.random.seed(s32)
            random.seed(s32)
            edge_prompt = build_edge_construction_prompt(instruction, sel_tools, tool_list)
            plan_text = ar.build_plan(edge_prompt, max_tokens=ar_max_tokens, temperature=0.1, timeout=90)
            pred_tools = extract_tools_from_text(plan_text, valid_tools)
            pred_edges = extract_edges_from_text(plan_text)
        else:
            pred_tools, pred_edges = [], []

        metrics = compute_metrics(pred_tools, gt_tools, pred_edges, gt_edges)
        heur = compute_heuristic_score(pred_tools, pred_edges, valid_tools)

        c = {
            "sample_id": sid,
            "task_type": task_type,
            "k": k,
            "confidence": None,
            "heuristic_score": heur,
            "pred_tools": pred_tools,
            "pred_edges": pred_edges,
            **metrics,
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
        }
        candidates.append(c)

        ssum = sample_summaries.setdefault(sid, {"sample_id": sid, "task_type": task_type, "k1_f1": None, "best_f1": 0.0})
        f1 = float(metrics.get("tool_f1", 0.0))
        if k == 1:
            ssum["k1_f1"] = f1
        if f1 > float(ssum["best_f1"]):
            ssum["best_f1"] = f1

        if (i + 1) % 50 == 0 or (i + 1) == len(proposals):
            print(f"[AR] Refined {i+1}/{len(proposals)} candidates", flush=True)

    del ar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sample_summaries_list = list(sample_summaries.values())
    avg_k1_f1 = float(np.mean([s["k1_f1"] for s in sample_summaries_list if s["k1_f1"] is not None])) if sample_summaries_list else 0.0
    avg_best_f1 = float(np.mean([float(s["best_f1"]) for s in sample_summaries_list])) if sample_summaries_list else 0.0
    f1_values = [float(c.get("tool_f1", 0.0)) for c in candidates] or [0.0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = {
        "meta": {
            "proposer": "llada",
            "K": K,
            "n_samples": len(subset),
            "n_candidates": len(candidates),
            "max_chain": max_chain,
            "max_dag": max_dag,
            "seed": seed,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "device": device,
            "llada_path": llada_path,
            "ar_path": ar_path,
            "tool_desc_path": tool_desc_path,
            "llada_steps": llada_steps,
            "llada_max_tokens": llada_max_tokens,
            "llada_temperature": llada_temperature,
            "ar_max_tokens": ar_max_tokens,
            "ids_file": ids_file,
        },
        "summary": {
            "avg_k1_f1": avg_k1_f1,
            "avg_best_f1": avg_best_f1,
            "f1_range": [float(min(f1_values)), float(max(f1_values))],
        },
        "sample_summaries": sample_summaries_list,
        "candidates": candidates,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {output_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-candidate data for LLaDA proposer baseline")
    ap.add_argument("--llada_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    ap.add_argument("--tool_desc_path", default="", help="Optional TaskBench tool_desc.json to define the tool library.")
    ap.add_argument("--ids_file", default="data/ids_500.txt")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--max_chain", type=int, default=50)
    ap.add_argument("--max_dag", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llada_steps", type=int, default=128)
    ap.add_argument("--llada_max_tokens", type=int, default=128)
    ap.add_argument("--llada_temperature", type=float, default=0.7)
    ap.add_argument("--ar_max_tokens", type=int, default=512)
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    args = ap.parse_args()

    collect_llada_candidates(
        llada_path=args.llada_path,
        ar_path=args.ar_path,
        data_path=args.data_path,
        tool_desc_path=args.tool_desc_path,
        ids_file=args.ids_file,
        output_path=args.output_path,
        device=args.device,
        K=args.K,
        max_chain=args.max_chain,
        max_dag=args.max_dag,
        seed=args.seed,
        llada_steps=args.llada_steps,
        llada_max_tokens=args.llada_max_tokens,
        llada_temperature=args.llada_temperature,
        ar_max_tokens=args.ar_max_tokens,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
