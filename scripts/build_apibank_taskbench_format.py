#!/usr/bin/env python3
"""
Convert API-Bank (LV1/LV2 jsonl dialogues) into a TaskBench-like JSONL format so we can
reuse the DiG-Plan/TaskBench pipeline unchanged.

Output:
  - data.json   (JSONL): each line is one sample with fields compatible with TaskBench
  - tool_desc.json      : TaskBench-style tool library (nodes: [{id, desc, parameters?}])

Notes:
  - We extract the ground-truth tool *set* from role=="API" lines (api_name).
  - We build a simple chain edge list from first-occurrence order of APIs.
  - Tool descriptions are extracted via static AST parsing from api-bank/apis/*.py to
    avoid importing optional dependencies (some APIs import extra packages).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _first_occurrence_sequence(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    seq: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        seq.append(x)
    return seq


def _chain_edges(seq: List[str]) -> List[Tuple[str, str]]:
    if len(seq) <= 1:
        return []
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _format_instruction_from_user_turns(turns: List[str]) -> str:
    if not turns:
        return ""
    if len(turns) == 1:
        return turns[0].strip()
    lines = []
    for i, t in enumerate(turns, 1):
        t = (t or "").strip()
        if not t:
            continue
        lines.append(f"User[{i}]: {t}")
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class ApiDef:
    name: str
    description: str
    input_parameters: Dict[str, Any]
    output_parameters: Dict[str, Any]


def _safe_literal_eval(node: ast.AST) -> Optional[Any]:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_class_api_defs_from_file(py_path: Path) -> List[ApiDef]:
    """
    Static extraction of API class metadata from an API-Bank api module.
    We look for class-level assignments to:
      - description: str
      - input_parameters: dict
      - output_parameters: dict
    """
    src = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return []

    out: List[ApiDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls_name = node.name
        desc_val: Optional[str] = None
        in_params: Dict[str, Any] = {}
        out_params: Dict[str, Any] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            key = stmt.targets[0].id
            val = _safe_literal_eval(stmt.value)
            if key == "description" and isinstance(val, str):
                desc_val = val
            elif key == "input_parameters" and isinstance(val, dict):
                in_params = val
            elif key == "output_parameters" and isinstance(val, dict):
                out_params = val

        # Only keep classes that look like an API (have a description).
        if desc_val is None:
            continue
        out.append(
            ApiDef(
                name=cls_name,
                description=desc_val.strip(),
                input_parameters=in_params,
                output_parameters=out_params,
            )
        )
    return out


def load_apibank_tool_desc(
    apis_dir: Path,
    allowed_tools: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a TaskBench-style tool_desc list from API-Bank api source files.
    Output nodes look like: {"id": name, "desc": "...", "parameters": [{"name": ...}, ...]}
    """
    nodes: List[Dict[str, Any]] = []
    for py in sorted(apis_dir.glob("*.py")):
        if py.name in {"__init__.py", "api.py"}:
            continue
        for api_def in _extract_class_api_defs_from_file(py):
            if allowed_tools is not None and api_def.name not in allowed_tools:
                continue
            params = []
            if isinstance(api_def.input_parameters, dict):
                for pname, pinfo in api_def.input_parameters.items():
                    if not isinstance(pname, str):
                        continue
                    ptype = ""
                    if isinstance(pinfo, dict):
                        ptype = str(pinfo.get("type", ""))
                    params.append({"name": pname, "type": ptype})
            desc = api_def.description
            if params:
                ps = ", ".join(
                    f"{p['name']}{(':'+p['type']) if p.get('type') else ''}" for p in params
                )
                desc = f"{desc} Inputs: {ps}."
            nodes.append({"id": api_def.name, "desc": desc, "parameters": params})

    # If AST parsing missed some allowed tools, still include placeholders so prompts/eval remain consistent.
    if allowed_tools is not None:
        seen = {n["id"] for n in nodes if isinstance(n, dict) and n.get("id")}
        missing = sorted(t for t in allowed_tools if t not in seen)
        for t in missing:
            nodes.append({"id": t, "desc": "No description available.", "parameters": []})

    return nodes


def iter_apibank_dialogue_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.jsonl"))


def build_taskbench_samples_from_apibank(
    samples_root: Path,
    ids_allowlist: Optional[Set[str]] = None,
    restrict_tools_to_dataset: bool = True,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Convert API-Bank dialogues into TaskBench-like sample dicts.
    Returns (samples, tools_used).
    """
    samples: List[Dict[str, Any]] = []
    tools_used: Set[str] = set()

    for fp in iter_apibank_dialogue_files(samples_root):
        # Stable sample_id derived from relative path.
        rel = fp.relative_to(samples_root).as_posix()
        sample_id = rel.replace("/", "__").replace(".jsonl", "")
        if ids_allowlist is not None and sample_id not in ids_allowlist:
            continue

        msgs = _read_jsonl(fp)
        user_turns = [m.get("text", "") for m in msgs if m.get("role") == "User"]
        api_calls = [m.get("api_name", "") for m in msgs if m.get("role") == "API" and m.get("api_name")]

        seq = _first_occurrence_sequence([c for c in api_calls if isinstance(c, str) and c.strip()])
        tools_used.update(seq)
        edges = _chain_edges(seq)

        tool_nodes = [{"task": t} for t in seq]
        tool_links = [{"source": s, "target": t} for (s, t) in edges]

        ttype = "single" if len(seq) <= 1 else "chain"
        instruction = _format_instruction_from_user_turns(user_turns)
        if not instruction:
            # Skip degenerate samples.
            continue

        samples.append(
            {
                "id": sample_id,
                "type": ttype,
                "instruction": instruction,
                "tool_nodes": json.dumps(tool_nodes, ensure_ascii=False),
                "tool_links": json.dumps(tool_links, ensure_ascii=False),
                "n_tools": len(seq),
            }
        )

    if restrict_tools_to_dataset:
        # No change needed here; caller will use tools_used for tool_desc and filtering.
        pass

    return samples, tools_used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples_root",
        default="data/raw/api-bank/lv1-lv2-samples/level-1-given-desc",
        help="Directory containing API-Bank LV1/LV2 dialogue jsonl files.",
    )
    ap.add_argument(
        "--apis_dir",
        default="data/raw/api-bank/apis",
        help="Directory containing API-Bank API definitions (python files).",
    )
    ap.add_argument(
        "--out_dir",
        default="data/processed/api-bank/level-1-given-desc",
        help="Output directory to write TaskBench-like data.json and tool_desc.json.",
    )
    ap.add_argument("--ids_file", default="", help="Optional allowlist of sample IDs (one per line).")
    ap.add_argument("--restrict_tools_to_dataset", action="store_true", help="Restrict tool_desc to tools used in selected samples.")
    args = ap.parse_args()

    samples_root = Path(args.samples_root)
    apis_dir = Path(args.apis_dir)
    out_dir = Path(args.out_dir)

    ids_allowlist: Optional[Set[str]] = None
    if args.ids_file:
        p = Path(args.ids_file)
        if p.exists():
            ids_allowlist = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}

    samples, tools_used = build_taskbench_samples_from_apibank(
        samples_root=samples_root,
        ids_allowlist=ids_allowlist,
        restrict_tools_to_dataset=args.restrict_tools_to_dataset,
    )
    if not samples:
        raise SystemExit("No samples produced. Check --samples_root / --ids_file.")

    allowed = tools_used if args.restrict_tools_to_dataset else None
    tool_nodes = load_apibank_tool_desc(apis_dir=apis_dir, allowed_tools=allowed)
    if not tool_nodes:
        raise SystemExit("No tool descriptions extracted. Check --apis_dir.")

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.json"
    tool_desc_path = out_dir / "tool_desc.json"

    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(tool_desc_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": tool_nodes}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(samples)} samples -> {data_path}")
    print(f"Wrote {len(tool_nodes)} tools   -> {tool_desc_path}")
    print(f"Tool used union size: {len(tools_used)}")


if __name__ == "__main__":
    main()
