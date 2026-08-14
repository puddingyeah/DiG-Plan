#!/usr/bin/env python3
"""
Reconstruct a paper-compatible TaskBench JSONL source from HF `improved.parquet`.

Why this script exists:
- The paper-era pipeline used flattened fields:
  `instruction`, `tool_steps`, `tool_nodes`, `tool_links`.
- Current local official raw JSONL uses:
  `user_request`, `task_steps`, `task_nodes`, `task_links`.
- For strict reproducibility, we materialize a stable JSONL from `improved.parquet`
  and emit audit stats (including TaskBench-23 filtering counts).
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd


HF_IMPROVED_PARQUET_URL = (
    "https://huggingface.co/datasets/microsoft/Taskbench/resolve/main/"
    "data_huggingface/improved.parquet"
)


def _parse_list_field(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for fn in (json.loads, ast.literal_eval):
            try:
                obj = fn(text)
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass
    return []


def _extract_tool_names(nodes: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for n in nodes:
        if isinstance(n, dict):
            if "task" in n and n["task"]:
                out.append(str(n["task"]))
            elif "id" in n and n["id"]:
                out.append(str(n["id"]))
        elif isinstance(n, str) and n:
            out.append(n)
    return out


def _compute_stats(rows: List[dict], canonical_tools: set[str]) -> dict:
    type_counts = Counter()
    unique_tools = set()
    bad_tool_rows = 0

    filtered_type_counts = Counter()
    filtered_total = 0

    for r in rows:
        t = str(r.get("type", "unknown"))
        type_counts[t] += 1

        nodes = _parse_list_field(r.get("tool_nodes", "[]"))
        if not nodes:
            bad_tool_rows += 1
            continue
        names = _extract_tool_names(nodes)
        unique_tools.update(names)

        if names and all(name in canonical_tools for name in names):
            filtered_total += 1
            filtered_type_counts[t] += 1

    return {
        "n_samples": len(rows),
        "type_counts": dict(type_counts),
        "unique_tools_from_tool_nodes": len(unique_tools),
        "bad_tool_rows": bad_tool_rows,
        "taskbench23_filtered_total": filtered_total,
        "taskbench23_filtered_type_counts": dict(filtered_type_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper-compatible flattened TaskBench JSONL from improved.parquet")
    ap.add_argument(
        "--input_parquet",
        default="data/raw/taskbench/data_huggingface/improved.parquet",
        help="Path to improved.parquet (flattened TaskBench parquet).",
    )
    ap.add_argument(
        "--output_jsonl",
        default="data/taskbench/taskbench_hf_improved_flattened.jsonl",
        help="Output JSONL path for flattened data.",
    )
    ap.add_argument(
        "--tool_desc_path",
        default="data/raw/taskbench/data_huggingface/tool_desc.json",
        help="Canonical 23-tool descriptor for filter-count auditing.",
    )
    ap.add_argument(
        "--stats_json",
        default="artifacts/results/taskbench_reconstruction.json",
        help="Output path for reconstruction stats JSON.",
    )
    ap.add_argument(
        "--stats_md",
        default="artifacts/results/taskbench_reconstruction.md",
        help="Output path for reconstruction stats markdown.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_parquet)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {input_path}. "
            f"Please download first (e.g., from {HF_IMPROVED_PARQUET_URL})."
        )

    df = pd.read_parquet(input_path)
    required_cols = {
        "id",
        "seed",
        "n_tools",
        "type",
        "sampled_nodes",
        "sampled_links",
        "instruction",
        "tool_steps",
        "tool_nodes",
        "tool_links",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    rows: List[dict] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "id": str(row["id"]),
                "seed": int(row["seed"]),
                "n_tools": int(row["n_tools"]),
                "type": str(row["type"]),
                "sampled_nodes": row["sampled_nodes"],
                "sampled_links": row["sampled_links"],
                "instruction": row["instruction"],
                "tool_steps": row["tool_steps"],
                "tool_nodes": row["tool_nodes"],
                "tool_links": row["tool_links"],
            }
        )

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tool_desc = json.loads(Path(args.tool_desc_path).read_text(encoding="utf-8"))
    nodes = tool_desc["nodes"] if isinstance(tool_desc, dict) else tool_desc
    canonical_tools = {str(n["id"]) for n in nodes if isinstance(n, dict) and n.get("id")}

    stats = _compute_stats(rows, canonical_tools)
    stats.update(
        {
            "input_parquet": str(input_path),
            "output_jsonl": str(out_jsonl),
            "tool_desc_path": args.tool_desc_path,
            "hf_improved_parquet_url": HF_IMPROVED_PARQUET_URL,
        }
    )

    stats_json = Path(args.stats_json)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# TaskBench Paper-Source Reconstruction",
        "",
        "Using `improved.parquet` to reconstruct a paper-compatible flattened JSONL source.",
        "",
        f"- input parquet: `{stats['input_parquet']}`",
        f"- output jsonl: `{stats['output_jsonl']}`",
        f"- rows: `{stats['n_samples']}`",
        f"- type counts: `{stats['type_counts']}`",
        f"- unique tools from `tool_nodes`: `{stats['unique_tools_from_tool_nodes']}`",
        f"- bad tool-node rows: `{stats['bad_tool_rows']}`",
        "",
        "TaskBench-23 audit (filtered by canonical 23 tools):",
        f"- filtered total: `{stats['taskbench23_filtered_total']}`",
        f"- filtered type counts: `{stats['taskbench23_filtered_type_counts']}`",
    ]
    Path(args.stats_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote JSONL: {out_jsonl}")
    print(f"Wrote stats: {stats_json}")
    print(f"Wrote stats: {args.stats_md}")


if __name__ == "__main__":
    main()
