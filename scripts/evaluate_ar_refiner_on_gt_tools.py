#!/usr/bin/env python3
"""
Evaluate the AR refiner when the selected tool set is fixed to the ground truth.

Purpose:
  - Diagnose whether missing-edge errors are mainly due to weak proposal quality
    or due to the AR edge refiner itself.
  - Directly answer the reviewer criticism that edge prediction is the dominant
    residual failure mode.

This script uses the same prompt template, parser, and metric definitions as the
main DiG-Plan pipeline, but replaces the proposer with an oracle tool set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ar_refinement_eval import (
    AREdgeBuilder,
    build_edge_construction_prompt,
    compute_metrics,
    extract_edges_from_text,
    extract_tools_from_text,
)
from tool_catalog import filter_samples_by_tools, get_tool_definitions
from utils import get_sample_instruction, get_sample_task_links, get_sample_task_nodes


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_ids(ids_file: str) -> set[str]:
    if not ids_file:
        return set()
    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _type_allowed(task_type: str, include_types: set[str]) -> bool:
    if not include_types:
        return True
    return task_type in include_types


def _summarize(results: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    if not results:
        return {"n_samples": 0, "avg_tool_f1": 0.0, "avg_edge_recall": 0.0, "by_type": {}}

    out["n_samples"] = len(results)
    out["avg_tool_f1"] = float(np.mean([r["tool_f1"] for r in results]))
    out["avg_edge_recall"] = float(np.mean([r["edge_recall"] for r in results]))
    out["tool_exact_match_rate"] = float(np.mean([1.0 if r["tool_f1"] == 1.0 else 0.0 for r in results]))
    out["edge_full_recall_rate"] = float(np.mean([1.0 if r["edge_recall"] == 1.0 else 0.0 for r in results]))

    by_type = {}
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        grouped[str(r["task_type"])].append(r)
    for task_type, group in sorted(grouped.items()):
        by_type[task_type] = {
            "n": len(group),
            "tool_f1": float(np.mean([r["tool_f1"] for r in group])),
            "edge_recall": float(np.mean([r["edge_recall"] for r in group])),
            "tool_exact_match_rate": float(np.mean([1.0 if r["tool_f1"] == 1.0 else 0.0 for r in group])),
            "edge_full_recall_rate": float(np.mean([1.0 if r["edge_recall"] == 1.0 else 0.0 for r in group])),
        }
    out["by_type"] = by_type
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate AR refiner on oracle ground-truth tool sets")
    ap.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    ap.add_argument("--tool_desc_path", default="", help="Optional TaskBench tool_desc.json for non-canonical tool libraries")
    ap.add_argument("--ids_file", default="data/ids_500.txt")
    ap.add_argument("--include_types", default="chain,dag", help="Comma-separated task types to include")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=0, help="Optional cap after filtering; 0 means all")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    args = ap.parse_args()

    all_samples = _load_jsonl(args.data_path)
    tool_list = get_tool_definitions(args.tool_desc_path or None)
    valid_tools = {t["name"] for t in tool_list}
    samples = filter_samples_by_tools(all_samples, valid_tools)

    wanted_ids = _load_ids(args.ids_file)
    if wanted_ids:
        samples = [s for s in samples if str(s.get("id", "")) in wanted_ids]

    include_types = {x.strip() for x in args.include_types.split(",") if x.strip()}
    samples = [s for s in samples if _type_allowed(str(s.get("type", "unknown")), include_types)]

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    if args.shard_idx is not None and args.num_shards is not None:
        total = len(samples)
        start = (args.shard_idx * total) // args.num_shards
        end = ((args.shard_idx + 1) * total) // args.num_shards
        samples = samples[start:end]

    ar = AREdgeBuilder(args.ar_path, args.device)
    results: List[Dict] = []

    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", i))
        instruction = get_sample_instruction(sample)
        gt_nodes = get_sample_task_nodes(sample)
        gt_links = get_sample_task_links(sample)
        gt_tools = [n.get("task", "") for n in gt_nodes if isinstance(n, dict) and n.get("task")]
        gt_edges = [(l.get("source", ""), l.get("target", "")) for l in gt_links if isinstance(l, dict)]
        task_type = str(sample.get("type", "unknown"))

        prompt = build_edge_construction_prompt(instruction, gt_tools, tool_list)
        text = ar.build_plan(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        pred_tools = extract_tools_from_text(text, valid_tools)
        pred_edges = extract_edges_from_text(text)
        metrics = compute_metrics(pred_tools, gt_tools, pred_edges, gt_edges)

        row = {
            "sample_id": sample_id,
            "task_type": task_type,
            "instruction": instruction,
            "gt_tools": gt_tools,
            "gt_edges": gt_edges,
            "pred_tools": pred_tools,
            "pred_edges": pred_edges,
            **metrics,
        }
        results.append(row)
        print(
            f"[{i}/{len(samples)}] {task_type} sid={sample_id} "
            f"tool_f1={metrics['tool_f1']:.3f} edge_recall={metrics['edge_recall']:.3f}",
            flush=True,
        )

    out = {
        "meta": {
            "ar_path": args.ar_path,
            "data_path": args.data_path,
            "tool_desc_path": args.tool_desc_path,
            "ids_file": args.ids_file,
            "include_types": sorted(include_types),
            "device": args.device,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timeout": args.timeout,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
        },
        "summary": _summarize(results),
        "results": results,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
