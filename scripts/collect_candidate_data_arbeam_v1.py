#!/usr/bin/env python3
"""
Collect per-candidate data for an AR-beam proposer baseline (AR beam proposer + AR refiner).

Goal:
  - Directly address reviewer requests for stronger AR search baselines (beam-style decoding).
  - Keep output schema fully compatible with existing scorer/eval scripts.
"""

import argparse
import gc
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_catalog import filter_samples_by_tools, get_tool_definitions
from utils import get_sample_instruction, get_sample_task_links, get_sample_task_nodes
from ar_refinement_eval import (
    build_tool_selection_prompt,
    build_edge_construction_prompt,
    extract_tools_from_text,
    extract_edges_from_text,
    compute_metrics,
    run_with_timeout,
)


def compute_heuristic_score(
    tools: List[str],
    edges: List[Tuple[str, str]],
    valid_tools: set,
) -> float:
    if not tools:
        return 0.0

    uniq_tools = list(sorted(set(tools)))

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

    if uniq_tools:
        out_deg = {n: 0 for n in uniq_tools}
        for src, _tgt in edges:
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


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_ids(ids_file: str) -> set:
    if not ids_file:
        return set()
    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _select_subset(
    samples: List[Dict],
    max_single: int,
    max_chain: int,
    max_dag: int,
    seed: int,
    shard_idx: int | None,
    num_shards: int | None,
) -> List[Dict]:
    single = [s for s in samples if s.get("type") == "single"]
    chain = [s for s in samples if s.get("type") == "chain"]
    dag = [s for s in samples if s.get("type") == "dag"]
    rng = random.Random(seed)
    rng.shuffle(single)
    rng.shuffle(chain)
    rng.shuffle(dag)
    subset = single[:max_single] + chain[:max_chain] + dag[:max_dag]
    rng.shuffle(subset)

    if shard_idx is None or num_shards is None:
        return subset

    total = len(subset)
    if total == 0:
        return []
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"Invalid shard_idx={shard_idx}, num_shards={num_shards}")

    start = (shard_idx * total) // num_shards
    end = ((shard_idx + 1) * total) // num_shards
    return subset[start:end]


class ARBeamPlanner:
    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Loading AR model from {model_path} on {device}...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("AR model loaded.", flush=True)

    def generate_tool_candidates_beam(
        self,
        prompt: str,
        k: int,
        max_tokens: int,
        num_beams: int,
        timeout: int,
    ) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n_beams = max(int(num_beams), int(k), 1)
        n_return = max(int(k), 1)

        def generate():
            with torch.no_grad():
                out = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=n_beams,
                    num_return_sequences=n_return,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            texts = []
            prompt_len = int(inputs.input_ids.shape[1])
            for i in range(out.shape[0]):
                texts.append(self.tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True))
            return texts

        result = run_with_timeout(generate, timeout_seconds=timeout)
        if not result:
            return []
        return result

    def refine_edges_deterministic(
        self,
        prompt: str,
        max_tokens: int,
        num_beams: int,
        timeout: int,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n_beams = max(int(num_beams), 1)

        def generate():
            with torch.no_grad():
                out = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=n_beams,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = int(inputs.input_ids.shape[1])
            return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

        text = run_with_timeout(generate, timeout_seconds=timeout)
        return text or ""


def collect_ar_beam_candidates(
    ar_path: str,
    data_path: str,
    tool_desc_path: str,
    ids_file: str,
    output_path: str,
    device: str,
    K: int,
    max_single: int,
    max_chain: int,
    max_dag: int,
    seed: int,
    ar_tool_max_tokens: int,
    ar_tool_num_beams: int,
    ar_tool_timeout: int,
    ar_edge_max_tokens: int,
    ar_edge_num_beams: int,
    ar_edge_timeout: int,
    shard_idx: int | None,
    num_shards: int | None,
) -> None:
    print("\n=== Collect Candidates (AR beam proposer baseline) ===", flush=True)
    print(
        f"device={device} K={K} seed={seed} shard={shard_idx}/{num_shards} "
        f"tool_beams={ar_tool_num_beams} edge_beams={ar_edge_num_beams}",
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

    subset = _select_subset(
        samples=samples,
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
    print(f"subset={len(subset)} (single={n_single}, chain={n_chain}, dag={n_dag})", flush=True)

    if len(subset) == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = {
            "meta": {
                "proposer": "ar_beam",
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
                "ar_path": ar_path,
                "tool_desc_path": tool_desc_path,
                "ar_tool_max_tokens": ar_tool_max_tokens,
                "ar_tool_num_beams": ar_tool_num_beams,
                "ar_tool_timeout": ar_tool_timeout,
                "ar_edge_max_tokens": ar_edge_max_tokens,
                "ar_edge_num_beams": ar_edge_num_beams,
                "ar_edge_timeout": ar_edge_timeout,
                "ids_file": ids_file,
            },
            "summary": {"avg_k1_f1": 0.0, "avg_best_f1": 0.0, "f1_range": [0.0, 0.0]},
            "sample_summaries": [],
            "candidates": [],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote (empty shard): {output_path}", flush=True)
        return

    planner = ARBeamPlanner(ar_path, device)

    candidates: List[Dict] = []
    sample_summaries: List[Dict] = []

    for idx, s in enumerate(subset):
        sid = str(s.get("id", f"sample-{idx}"))
        task_type = s.get("type", "unknown")
        instruction = get_sample_instruction(s)

        gt_nodes = get_sample_task_nodes(s)
        gt_tools = [n.get("task", "") for n in gt_nodes]
        gt_links = get_sample_task_links(s)
        gt_edges = [(l.get("source", ""), l.get("target", "")) for l in gt_links]

        tool_prompt = build_tool_selection_prompt(instruction, tool_list)
        beam_texts = planner.generate_tool_candidates_beam(
            prompt=tool_prompt,
            k=K,
            max_tokens=ar_tool_max_tokens,
            num_beams=ar_tool_num_beams,
            timeout=ar_tool_timeout,
        )
        if not beam_texts:
            beam_texts = [""] * K
        elif len(beam_texts) < K:
            beam_texts += [beam_texts[-1]] * (K - len(beam_texts))

        per_sample = []
        for k in range(1, K + 1):
            # Keep deterministic behavior explicit even though beam decode itself is deterministic.
            local_seed = seed * 100000 + idx * 1000 + k
            torch.manual_seed(local_seed)
            np.random.seed(local_seed)
            random.seed(local_seed)

            tool_text = beam_texts[k - 1]
            sel_tools = extract_tools_from_text(tool_text, valid_tools)

            if sel_tools:
                edge_prompt = build_edge_construction_prompt(instruction, sel_tools, tool_list)
                plan_text = planner.refine_edges_deterministic(
                    prompt=edge_prompt,
                    max_tokens=ar_edge_max_tokens,
                    num_beams=ar_edge_num_beams,
                    timeout=ar_edge_timeout,
                )
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
            per_sample.append(c)
            candidates.append(c)

        k1_f1 = float(per_sample[0].get("tool_f1", 0.0))
        best_f1 = float(max(per_sample, key=lambda x: float(x.get("tool_f1", 0.0))).get("tool_f1", 0.0))
        sample_summaries.append(
            {
                "sample_id": sid,
                "task_type": task_type,
                "k1_f1": k1_f1,
                "best_f1": best_f1,
            }
        )

        if (idx + 1) % 5 == 0 or (idx + 1) == len(subset):
            print(f"Processed {idx + 1}/{len(subset)} samples", flush=True)

    del planner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_k1_f1 = float(np.mean([s["k1_f1"] for s in sample_summaries])) if sample_summaries else 0.0
    avg_best_f1 = float(np.mean([s["best_f1"] for s in sample_summaries])) if sample_summaries else 0.0
    f1_values = [float(c.get("tool_f1", 0.0)) for c in candidates] or [0.0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = {
        "meta": {
            "proposer": "ar_beam",
            "K": K,
            "n_samples": len(subset),
            "n_candidates": len(candidates),
            "max_single": max_single,
            "max_chain": max_chain,
            "max_dag": max_dag,
            "seed": seed,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "device": device,
            "ar_path": ar_path,
            "tool_desc_path": tool_desc_path,
            "ar_tool_max_tokens": ar_tool_max_tokens,
            "ar_tool_num_beams": ar_tool_num_beams,
            "ar_tool_timeout": ar_tool_timeout,
            "ar_edge_max_tokens": ar_edge_max_tokens,
            "ar_edge_num_beams": ar_edge_num_beams,
            "ar_edge_timeout": ar_edge_timeout,
            "ids_file": ids_file,
        },
        "summary": {
            "avg_k1_f1": avg_k1_f1,
            "avg_best_f1": avg_best_f1,
            "f1_range": [float(min(f1_values)), float(max(f1_values))],
        },
        "sample_summaries": sample_summaries,
        "candidates": candidates,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {output_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-candidate data for AR-beam proposer baseline")
    ap.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    ap.add_argument("--tool_desc_path", default="", help="Optional TaskBench tool_desc.json to define tool library.")
    ap.add_argument("--ids_file", default="data/ids_500.txt")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_single", type=int, default=0)
    ap.add_argument("--max_chain", type=int, default=167)
    ap.add_argument("--max_dag", type=int, default=167)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ar_tool_max_tokens", type=int, default=128)
    ap.add_argument("--ar_tool_num_beams", type=int, default=10)
    ap.add_argument("--ar_tool_timeout", type=int, default=120)
    ap.add_argument("--ar_edge_max_tokens", type=int, default=512)
    ap.add_argument("--ar_edge_num_beams", type=int, default=1)
    ap.add_argument("--ar_edge_timeout", type=int, default=120)
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    args = ap.parse_args()

    collect_ar_beam_candidates(**vars(args))


if __name__ == "__main__":
    main()
