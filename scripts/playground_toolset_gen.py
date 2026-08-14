#!/usr/bin/env python3
"""
Synthetic Tool-Set Planning Playground (v2): fixed-length toolset representation.

Goal:
  Provide a controlled synthetic environment to study *tool-set search bias*
  between prefix-AR and denoising-style decoders, without:
    - natural language semantics
    - pretraining confounds
    - length-oracle artifacts in denoising inference

Key design:
  - Universe of tools: T = {t1..tN} (default N=23).
  - Ground-truth is a DAG plan G=(V*,E*) over a subset V* ⊂ T.
  - Model predicts only the tool-set V* as a fixed-length bit-vector (N tokens of '0'/'1').
  - Input is a symbolic spec: [SRC] ... [TGT] ... [DEPTH] d, with mild noise.

Outputs:
  data/playground_toolset_v2/{train,valid,test}.jsonl
  data/playground_toolset_v2/meta.json
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def generate_dag(nodes: List[str], edge_prob: float) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    topo = nodes[:]
    random.shuffle(topo)

    edges: List[Tuple[str, str]] = []
    for i, v in enumerate(topo):
        for j in range(i):
            u = topo[j]
            if random.random() < edge_prob:
                edges.append((u, v))

    if not edges and len(topo) >= 2:
        for i in range(len(topo) - 1):
            edges.append((topo[i], topo[i + 1]))

    parents: Dict[str, List[str]] = {n: [] for n in topo}
    for u, v in edges:
        parents[v].append(u)

    depth: Dict[str, int] = {}
    for v in topo:
        if not parents[v]:
            depth[v] = 1
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1

    return edges, depth


def depth_bucket(max_depth: int) -> str:
    if max_depth <= 3:
        return "short"
    if max_depth <= 5:
        return "medium"
    return "long"


def build_spec_tokens(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    depth: Dict[str, int],
    noise_depth: int,
    noisy_size: int,
    max_src_tgt: int,
    hint_min: int,
    hint_max: int,
    include_size: bool,
    include_hint: bool,
    decoy_prob: float,
    universe: List[str],
) -> List[str]:
    incoming = {n: 0 for n in nodes}
    outgoing = {n: 0 for n in nodes}
    for u, v in edges:
        outgoing[u] += 1
        incoming[v] += 1

    sources = [n for n in nodes if incoming[n] == 0]
    sinks = [n for n in nodes if outgoing[n] == 0]
    random.shuffle(sources)
    random.shuffle(sinks)
    sources = sources[: max(1, min(max_src_tgt, len(sources)))]
    sinks = sinks[: max(1, min(max_src_tgt, len(sinks)))]

    spec: List[str] = ["[SRC]"] + sources + ["[TGT]"] + sinks + ["[DEPTH]", str(noise_depth)]
    if include_size:
        spec += ["[SIZE]", str(noisy_size)]

    # Hints: reveal a few GT nodes (plus optional decoy via [DECOY] below)
    if include_hint:
        hint_min_eff = max(0, min(hint_min, len(nodes)))
        hint_max_eff = max(hint_min_eff, min(hint_max, len(nodes)))
        if hint_max_eff > 0:
            hints = nodes[:]
            random.shuffle(hints)
            k = random.randint(hint_min_eff, hint_max_eff)
            hints = hints[:k]
            spec += ["[HINT]"] + hints

    if decoy_prob > 0:
        decoys = [t for t in universe if t not in nodes]
        random.shuffle(decoys)
        if decoys and random.random() < decoy_prob:
            spec = spec + ["[DECOY]", decoys[0]]

    return spec


def generate_one(
    sample_id: str,
    universe: List[str],
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    max_src_tgt: int,
    size_noise: int,
    hint_min: int,
    hint_max: int,
    include_size: bool,
    include_hint: bool,
    decoy_prob: float,
    bucket: str,
    max_tries: int = 2000,
) -> Dict:
    for _ in range(max_tries):
        n = random.randint(min_nodes, max_nodes)
        nodes = random.sample(universe, n)
        edges, depth = generate_dag(nodes, edge_prob=edge_prob)
        max_depth = max(depth.values()) if depth else 1
        b = depth_bucket(max_depth)
        if b != bucket:
            continue

        noisy_depth = max(1, max_depth + random.choice([-1, 0, 1]))
        noisy_size = max(1, n + random.randint(-size_noise, size_noise))
        spec_tokens = build_spec_tokens(
            nodes=nodes,
            edges=edges,
            depth=depth,
            noise_depth=noisy_depth,
            noisy_size=noisy_size,
            max_src_tgt=max_src_tgt,
            hint_min=hint_min,
            hint_max=hint_max,
            include_size=include_size,
            include_hint=include_hint,
            decoy_prob=decoy_prob,
            universe=universe,
        )

        bits = ["1" if t in nodes else "0" for t in universe]

        return {
            "id": sample_id,
            "spec_tokens": spec_tokens,
            "universe": universe,
            "gt_nodes": sorted(nodes),
            "gt_bits": bits,
            "gt_edges": [[u, v] for (u, v) in edges],
            "depth": int(max_depth),
            "depth_bucket": b,
            "num_nodes": int(n),
        }

    raise RuntimeError(f"Failed to sample bucket={bucket} after {max_tries} tries.")


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic toolset playground v2 (fixed-length bitvector).")
    ap.add_argument("--output_dir", default="data/playground_toolset_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--universe_size", type=int, default=23)
    ap.add_argument("--min_nodes", type=int, default=3)
    ap.add_argument("--max_nodes", type=int, default=7)
    ap.add_argument("--edge_prob", type=float, default=0.4)
    ap.add_argument("--max_src_tgt", type=int, default=2)
    ap.add_argument("--size_noise", type=int, default=1, help="Noise added to |V*| in [SIZE] token.")
    ap.add_argument("--hint_min", type=int, default=2, help="Minimum number of GT tools revealed in [HINT].")
    ap.add_argument("--hint_max", type=int, default=4, help="Maximum number of GT tools revealed in [HINT].")
    ap.add_argument("--decoy_prob", type=float, default=0.3)
    ap.add_argument("--no_size", action="store_true", help="Do not include [SIZE] token in spec.")
    ap.add_argument("--no_hint", action="store_true", help="Do not include [HINT] token in spec.")
    ap.add_argument("--train_per_bucket", type=int, default=20000)
    ap.add_argument("--valid_per_bucket", type=int, default=2000)
    ap.add_argument("--test_per_bucket", type=int, default=2000)
    args = ap.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    universe = [f"t{i}" for i in range(1, args.universe_size + 1)]
    buckets = ["short", "medium", "long"]

    def gen_split(split: str, per_bucket: int) -> List[Dict]:
        rows: List[Dict] = []
        for b in buckets:
            for i in range(per_bucket):
                rows.append(
                    generate_one(
                        sample_id=f"{split}-{b}-{i}",
                        universe=universe,
                        min_nodes=args.min_nodes,
                        max_nodes=args.max_nodes,
                        edge_prob=args.edge_prob,
                        max_src_tgt=args.max_src_tgt,
                        size_noise=args.size_noise,
                        hint_min=args.hint_min,
                        hint_max=args.hint_max,
                        include_size=(not args.no_size),
                        include_hint=(not args.no_hint),
                        decoy_prob=args.decoy_prob,
                        bucket=b,
                    )
                )
        random.shuffle(rows)
        return rows

    train = gen_split("train", args.train_per_bucket)
    valid = gen_split("valid", args.valid_per_bucket)
    test = gen_split("test", args.test_per_bucket)

    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.output_dir, "valid.jsonl"), valid)
    write_jsonl(os.path.join(args.output_dir, "test.jsonl"), test)

    meta = {
        "seed": args.seed,
        "universe_size": args.universe_size,
        "universe": universe,
        "min_nodes": args.min_nodes,
        "max_nodes": args.max_nodes,
        "edge_prob": args.edge_prob,
        "max_src_tgt": args.max_src_tgt,
        "size_noise": args.size_noise,
        "hint_min": args.hint_min,
        "hint_max": args.hint_max,
        "decoy_prob": args.decoy_prob,
        "no_size": bool(args.no_size),
        "no_hint": bool(args.no_hint),
        "train_per_bucket": args.train_per_bucket,
        "valid_per_bucket": args.valid_per_bucket,
        "test_per_bucket": args.test_per_bucket,
        "buckets": buckets,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Generated toolset playground v2:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  universe_size: {args.universe_size}")
    print(f"  train/valid/test sizes: {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
