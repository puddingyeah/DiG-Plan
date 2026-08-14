#!/usr/bin/env python3
"""
Create leakage-free IJCAI paper splits (by sample_id).

Default policy (standard + conservative):
  - Test split: use existing `data/ids_500.txt` (balanced 167/167/167).
  - Train split: sample from remaining filtered TaskBench pool, excluding test IDs.
  - For value-function training, we focus on chain/dag (where search matters).

Outputs:
  - data/splits/ijcai_test_ids.txt
  - data/splits/ijcai_train_ids_chain{N}_dag{M}_seed{S}.txt
  - data/splits/ijcai_splits_meta.json
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Set

from tool_catalog import filter_samples_by_tools, get_tool_definitions


def load_jsonl(path: str) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def load_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def write_ids(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in ids:
            f.write(str(i) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create IJCAI splits (train/test IDs)")
    ap.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    ap.add_argument("--tool_desc_path", default="", help="Optional TaskBench tool_desc.json to define the tool library.")
    ap.add_argument("--test_ids", default="data/ids_500.txt")
    ap.add_argument("--train_chain", type=int, default=450)
    ap.add_argument("--train_dag", type=int, default=450)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/splits")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tool_defs = get_tool_definitions(args.tool_desc_path or None)
    valid_tools = set(t["name"] for t in tool_defs)

    all_samples = load_jsonl(args.data_path)
    filtered = filter_samples_by_tools(all_samples, valid_tools)

    test_ids = load_ids(args.test_ids)

    by_id = {str(s.get("id")): s for s in filtered}
    found_test = [by_id[i] for i in test_ids if i in by_id]

    filtered_types = Counter(s.get("type") for s in filtered)
    test_types = Counter(s.get("type") for s in found_test)

    remaining = [s for s in filtered if str(s.get("id")) not in test_ids]
    remaining_types = Counter(s.get("type") for s in remaining)

    remaining_chain = [s for s in remaining if s.get("type") == "chain"]
    remaining_dag = [s for s in remaining if s.get("type") == "dag"]

    rng.shuffle(remaining_chain)
    rng.shuffle(remaining_dag)

    train_chain = remaining_chain[: min(args.train_chain, len(remaining_chain))]
    train_dag = remaining_dag[: min(args.train_dag, len(remaining_dag))]

    train_ids = [str(s.get("id")) for s in (train_chain + train_dag)]
    rng.shuffle(train_ids)

    out_test = os.path.join(args.out_dir, "ijcai_test_ids.txt")
    out_train = os.path.join(
        args.out_dir,
        f"ijcai_train_ids_chain{len(train_chain)}_dag{len(train_dag)}_seed{args.seed}.txt",
    )
    out_meta = os.path.join(args.out_dir, "ijcai_splits_meta.json")

    write_ids(out_test, sorted(test_ids))
    write_ids(out_train, train_ids)

    meta = {
        "data_path": args.data_path,
        "filtered_total": len(filtered),
        "filtered_type_counts": dict(filtered_types),
        "test_ids_path": args.test_ids,
        "test_total": len(test_ids),
        "test_found_in_filtered": len(found_test),
        "test_type_counts": dict(test_types),
        "train_ids_path": out_train,
        "train_total": len(train_ids),
        "train_chain": len(train_chain),
        "train_dag": len(train_dag),
        "remaining_type_counts_after_test": dict(remaining_types),
        "seed": args.seed,
        "note": "Train IDs are sampled from filtered pool excluding test IDs; train focuses on chain/dag.",
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("=== IJCAI split creation ===")
    print(f"Filtered samples: {len(filtered)}  types={dict(filtered_types)}")
    print(f"Test IDs: {len(test_ids)}  found_in_filtered={len(found_test)}  types={dict(test_types)}")
    print(f"Train IDs: {len(train_ids)} (chain={len(train_chain)}, dag={len(train_dag)})")
    print(f"Wrote: {out_test}")
    print(f"Wrote: {out_train}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
