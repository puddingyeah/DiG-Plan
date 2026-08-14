#!/usr/bin/env python3
"""
Collect per-candidate data for an LLaDA2.0 proposer baseline (LLaDA2 proposer + AR refiner).

Why:
  - Expand the diffusion/denoising proposer family beyond Dream and LLaDA-8B.
  - Provide a second-generation diffusion LM proposer for the same tool-graph planning protocol.

Notes:
  - LLaDA2.0 "mini-preview" on HuggingFace uses custom generation arguments
    (e.g., steps/gen_length/block_length). We call `model.generate(...)` with those
    kwargs under `trust_remote_code=True`.
  - Confidence is optional: by default we keep `confidence=None` for backward-compatibility.
    With `--llada2_extract_confidence`, we extract an entropy-based confidence signal during
    masked denoising generation (similar in spirit to the Dream confidence in this repo).
  - Output JSON format matches scripts/collect_candidate_data_v2.py expectations:
      { meta, summary, sample_summaries, candidates }
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return int(x % _NP_SEED_MOD)


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_ids(ids_file: str) -> set:
    if not ids_file:
        return set()
    if not os.path.exists(ids_file):
        return set()
    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def compute_heuristic_score(
    tools: List[str],
    edges: List[Tuple[str, str]],
    valid_tools: set,
) -> float:
    if not tools:
        return 0.0

    uniq_tools = list(sorted(set(tools)))

    # structure validity: DAG check over nodes appearing in nodes/edges
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

    # coverage (TaskBench-style prior; keep for comparability)
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


def _llada2_generate_text(
    model,
    tok,
    prompt: str,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    eos_early_stop: bool,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Avoid padding/attention_mask to stay compatible with LLaDA2 custom generate() APIs,
    # which may not accept `attention_mask`.
    encoded = tok(chat_prompt, add_special_tokens=False, padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # LLaDA2.0 uses custom generation args (trust_remote_code=True). We pass them through.
    gen_kwargs = dict(
        eos_early_stop=eos_early_stop,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        steps=steps,
    )
    try:
        out = model.generate(inputs=input_ids, **gen_kwargs)
    except TypeError:
        try:
            out = model.generate(input_ids=input_ids, **gen_kwargs)
        except TypeError:
            out = model.generate(input_ids, **gen_kwargs)

    if hasattr(out, "sequences"):
        out_ids = out.sequences
    else:
        out_ids = out

    # Best-effort: strip prompt tokens if generate returns concatenated sequences.
    if isinstance(out_ids, torch.Tensor) and out_ids.ndim == 2 and out_ids.shape[1] >= input_ids.shape[1]:
        gen = out_ids[:, input_ids.shape[1] :]
    else:
        gen = out_ids
    return tok.batch_decode(gen, skip_special_tokens=True)[0]


def _llada2_entropy_from_logits(
    model,
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """
    Compute token entropy under the sampling distribution:
      temperature scaling + top-k/top-p filtering + softmax.

    Args:
      logits: [..., vocab]
    Returns:
      entropy: logits.shape[:-1]
    """
    orig_shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    x = logits.reshape(-1, vocab)

    if temperature is not None and float(temperature) > 0 and float(temperature) != 1.0:
        x = x / float(temperature)
    if top_k is not None and int(top_k) > 0:
        x = model._top_k_logits(x, int(top_k))
    if top_p is not None and float(top_p) < 1.0:
        x = model._top_p_logits(x, float(top_p))

    p = F.softmax(x, dim=-1)
    logp = torch.log(p.clamp_min(1e-12))
    ent = -(p * logp).sum(dim=-1)
    return ent.view(*orig_shape)


@torch.no_grad()
def _llada2_generate_text_with_confidence(
    model,
    tok,
    prompt: str,
    device: str,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    eos_early_stop: bool,
    top_p: float | None = None,
    top_k: int | None = None,
    minimal_topk: int = 1,
    threshold: float = 0.95,
    eos_id: int = 156892,
    mask_id: int = 156895,
    drop_first_steps: int = 3,
) -> tuple[str, float]:
    """
    LLaDA2 generate() re-implemented with an entropy-based confidence signal.

    Confidence:
      conf = 1 / (1 + mean_entropy)
    where mean_entropy averages token entropy on masked positions across denoising steps,
    after dropping the first few steps (too noisy).
    """
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tok(chat_prompt, add_special_tokens=False, padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    steps = min(int(steps), int(gen_length) // int(max(minimal_topk, 1)))

    prompt_length = int(input_ids.shape[1])
    num_blocks = (prompt_length + int(gen_length) + int(block_length) - 1) // int(block_length)
    total_length = int(num_blocks) * int(block_length)

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_attn = (
        block_mask.repeat_interleave(int(block_length), dim=0)
        .repeat_interleave(int(block_length), dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).bool()
    block_attn = torch.where(block_attn, 0.0, float("-inf")).to(torch.bfloat16)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), int(mask_id), dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone().to(model.device)

    prefill_blocks = prompt_length // int(block_length)
    schedule = model._get_num_transfer_tokens(int(block_length), int(steps))

    entropies: list[float] = []
    global_step = 0

    for num_block in range(int(prefill_blocks), int(num_blocks)):
        current_window_end = (int(num_block) + 1) * int(block_length)
        cur_x = x[:, :current_window_end]
        cur_attn = block_attn[:, :, :current_window_end, :current_window_end]
        cur_pos = position_ids[:, :current_window_end]

        for step in range(int(steps)):
            active_block_mask = cur_x[:, -int(block_length) :] == int(mask_id)
            if int(active_block_mask.sum().item()) == 0:
                break

            logits = model.forward(cur_x, attention_mask=cur_attn, position_ids=cur_pos).logits
            active_logits = logits[:, -int(block_length) :, :]

            ent = _llada2_entropy_from_logits(
                model,
                active_logits,
                temperature=float(temperature),
                top_k=top_k,
                top_p=top_p,
            )
            if global_step >= int(drop_first_steps):
                try:
                    entropies.append(float(ent[active_block_mask].mean().item()))
                except Exception:
                    pass
            global_step += 1

            x0, x0_p = model._sample_with_temperature_topk_topp(
                active_logits,
                temperature=float(temperature),
                top_k=top_k,
                top_p=top_p,
            )

            num_to_transfer = int(schedule[int(step)].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            conf_tok = torch.where(active_block_mask, x0_p, -torch.inf)
            high_conf_mask = conf_tok[0] > float(threshold)
            num_high = int(high_conf_mask.sum().item())

            if num_high >= num_to_transfer:
                transfer_index[0] = high_conf_mask
            else:
                _vals, idx = torch.topk(conf_tok[0], k=min(num_to_transfer, int(active_block_mask.sum().item())))
                transfer_index[0, idx] = True

            if bool(transfer_index.any()):
                cur_x[:, -int(block_length) :][transfer_index] = x0[transfer_index]

            if bool(eos_early_stop) and bool((x0[transfer_index] == int(eos_id)).any()):
                eos_pos_in_x = (cur_x[0] == int(eos_id)).nonzero(as_tuple=True)
                if len(eos_pos_in_x[0]) > 0:
                    eos_pos = int(eos_pos_in_x[0][0].item())
                    if (cur_x[0, prompt_length:eos_pos] != int(mask_id)).all():
                        final_x = x[:, :total_length][:, : eos_pos + 1]
                        gen = final_x[:, prompt_length:]
                        text = tok.batch_decode(gen, skip_special_tokens=True)[0]
                        mean_ent = float(np.mean(entropies)) if entropies else 0.0
                        return text, float(1.0 / (1.0 + mean_ent))

        x[:, :current_window_end] = cur_x
        if (x[0, prompt_length:current_window_end] == int(eos_id)).any():
            break

    generated_answer = x[:, : prompt_length + int(gen_length)]
    eos_positions = (generated_answer[0][prompt_length:] == int(eos_id)).nonzero(as_tuple=True)[0]
    first_eos = int(eos_positions[0].item()) if len(eos_positions) > 0 else int(gen_length)
    out_ids = generated_answer[:, : prompt_length + first_eos + 1]

    gen = out_ids[:, prompt_length:]
    text = tok.batch_decode(gen, skip_special_tokens=True)[0]
    mean_ent = float(np.mean(entropies)) if entropies else 0.0
    return text, float(1.0 / (1.0 + mean_ent))


def collect_llada2_candidates(
    llada2_path: str,
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
    llada2_steps: int,
    llada2_gen_length: int,
    llada2_block_length: int,
    llada2_temperature: float,
    llada2_eos_early_stop: bool,
    llada2_extract_confidence: bool,
    ar_max_tokens: int,
    shard_idx: int | None,
    num_shards: int | None,
) -> None:
    print("\n=== Collect Candidates (LLaDA2 proposer baseline) ===", flush=True)
    print(f"device={device} K={K} seed={seed} shard={shard_idx}/{num_shards}", flush=True)
    print(
        f"llada2: steps={llada2_steps} gen_length={llada2_gen_length} block_length={llada2_block_length} "
        f"temp={llada2_temperature} eos_early_stop={llada2_eos_early_stop} | ar_max_tokens={ar_max_tokens}",
        flush=True,
    )
    print(f"llada2_extract_confidence={llada2_extract_confidence}", flush=True)

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
                "proposer": "llada2",
                "K": K,
                "n_samples": 0,
                "n_candidates": 0,
                "max_chain": max_chain,
                "max_dag": max_dag,
                "seed": seed,
                "shard_idx": shard_idx,
                "num_shards": num_shards,
                "device": device,
                "llada2_path": llada2_path,
                "ar_path": ar_path,
                "tool_desc_path": tool_desc_path,
                "llada2_steps": llada2_steps,
                "llada2_gen_length": llada2_gen_length,
                "llada2_block_length": llada2_block_length,
                "llada2_temperature": llada2_temperature,
                "llada2_eos_early_stop": llada2_eos_early_stop,
                "llada2_extract_confidence": bool(llada2_extract_confidence),
                "ar_max_tokens": ar_max_tokens,
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

    # Phase 1: LLaDA2 tool-set proposals
    print(f"[LLaDA2] Loading model from {llada2_path} on {device} ...", flush=True)
    # NOTE: Do NOT pass device_map=device here.
    # Recent transformers versions run a CUDA caching-allocator warmup when device_map is set,
    # which can transiently allocate a large extra buffer and OOM when we pack multiple workers
    # per GPU. We load on CPU first (no warmup), then move to the target device.
    llada2 = AutoModelForCausalLM.from_pretrained(
        llada2_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    if str(device).startswith("cuda"):
        llada2 = llada2.to(device)
    tok = AutoTokenizer.from_pretrained(llada2_path, trust_remote_code=True, local_files_only=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"

    proposals: List[Dict] = []
    for idx, s in enumerate(subset):
        sid = str(s.get("id", f"sample-{idx}"))
        instruction = s.get("instruction", "")
        tool_prompt = build_tool_selection_prompt(instruction, tool_list)

        for k in range(1, K + 1):
            combined = seed * 100000 + idx * 1000 + k
            s32 = _seed32(combined)
            torch.manual_seed(s32)
            np.random.seed(s32)
            random.seed(s32)

            try:
                if llada2_extract_confidence:
                    text, conf = _llada2_generate_text_with_confidence(
                        llada2,
                        tok,
                        tool_prompt,
                        device=device,
                        steps=llada2_steps,
                        gen_length=llada2_gen_length,
                        block_length=llada2_block_length,
                        temperature=llada2_temperature,
                        eos_early_stop=llada2_eos_early_stop,
                    )
                else:
                    text = _llada2_generate_text(
                        llada2,
                        tok,
                        tool_prompt,
                        device=device,
                        steps=llada2_steps,
                        gen_length=llada2_gen_length,
                        block_length=llada2_block_length,
                        temperature=llada2_temperature,
                        eos_early_stop=llada2_eos_early_stop,
                    )
                    conf = None
            except TypeError as e:
                raise RuntimeError(
                    "LLaDA2 generation API mismatch. "
                    "This model must support custom `generate` kwargs (steps/gen_length/block_length/eos_early_stop)."
                ) from e

            sel_tools = extract_tools_from_text(text, valid_tools)
            proposals.append(
                {
                    "sample_id": sid,
                    "k": k,
                    "confidence": conf,
                    "selected_tools": sel_tools,
                    "tool_prompt": tool_prompt,
                }
            )

        if (idx + 1) % 5 == 0 or (idx + 1) == len(subset):
            print(f"[LLaDA2] Proposed tools for {idx+1}/{len(subset)} samples", flush=True)

    del llada2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: AR refinement
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
        gt_lookup[sid] = {
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
            "task_type": s.get("type", "unknown"),
            "instruction": s.get("instruction", ""),
        }

    candidates: List[Dict] = []
    sample_summaries: Dict[str, Dict] = {}

    for i, p in enumerate(proposals):
        sid = str(p["sample_id"])
        k = int(p["k"])
        gt = gt_lookup[sid]
        task_type = gt["task_type"]
        instruction = gt["instruction"]
        gt_tools = gt["gt_tools"]
        gt_edges = gt["gt_edges"]
        sel_tools = p["selected_tools"]

        if sel_tools:
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
            "confidence": p.get("confidence", None),
            "heuristic_score": heur,
            "pred_tools": pred_tools,
            "pred_edges": pred_edges,
            **metrics,
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
        }
        candidates.append(c)

        ssum = sample_summaries.setdefault(
            sid, {"sample_id": sid, "task_type": task_type, "k1_f1": None, "best_f1": 0.0}
        )
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
            "proposer": "llada2",
            "K": K,
            "n_samples": len(subset),
            "n_candidates": len(candidates),
            "max_chain": max_chain,
            "max_dag": max_dag,
            "seed": seed,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "device": device,
            "llada2_path": llada2_path,
            "ar_path": ar_path,
            "tool_desc_path": tool_desc_path,
            "llada2_steps": llada2_steps,
            "llada2_gen_length": llada2_gen_length,
            "llada2_block_length": llada2_block_length,
            "llada2_temperature": llada2_temperature,
            "llada2_eos_early_stop": llada2_eos_early_stop,
            "llada2_extract_confidence": bool(llada2_extract_confidence),
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {output_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-candidate data for LLaDA2 proposer baseline")
    ap.add_argument("--llada2_path", default="inclusionAI/LLaDA2.0-mini-preview")
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
    ap.add_argument("--llada2_steps", type=int, default=32)
    ap.add_argument("--llada2_gen_length", type=int, default=128)
    ap.add_argument("--llada2_block_length", type=int, default=32)
    ap.add_argument("--llada2_temperature", type=float, default=0.0)
    ap.add_argument("--llada2_eos_early_stop", action="store_true")
    ap.add_argument("--llada2_extract_confidence", action="store_true")
    ap.add_argument("--ar_max_tokens", type=int, default=512)
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    args = ap.parse_args()

    collect_llada2_candidates(
        llada2_path=args.llada2_path,
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
        llada2_steps=args.llada2_steps,
        llada2_gen_length=args.llada2_gen_length,
        llada2_block_length=args.llada2_block_length,
        llada2_temperature=args.llada2_temperature,
        llada2_eos_early_stop=bool(args.llada2_eos_early_stop),
        llada2_extract_confidence=bool(args.llada2_extract_confidence),
        ar_max_tokens=args.ar_max_tokens,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
