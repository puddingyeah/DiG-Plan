#!/usr/bin/env python3
"""
Evaluate toolset playground v2 for:
  - prefix-AR sampler
  - masked-denoising sampler

Metrics:
  - Tool Precision/Recall/F1 (k=1)
  - union-of-k Tool Recall (Pass@k, k<=Kmax)
  - Oracle best-of-k Tool F1 (analysis upper bound)
  - Diversity proxies: unique candidates@k, avg pairwise Jaccard@k

This is intended to be the *controlled* synthetic counterpart of TaskBench proposer analysis:
  - AR tends to have higher single-sample precision
  - denoising tends to have stronger coverage scaling with K
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from playground_toolset_models import TinyTransformer, decode_toolset_from_bits, toolset_f1


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stratified_subsample(rows: List[Dict], max_per_bucket: int, seed: int) -> List[Dict]:
    if max_per_bucket <= 0:
        return rows
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("depth_bucket", "short")].append(r)
    out = []
    for b, br in buckets.items():
        if len(br) <= max_per_bucket:
            out.extend(br)
        else:
            out.extend(rng.sample(br, k=max_per_bucket))
    rng.shuffle(out)
    return out


def build_id_maps(vocab: Dict[str, int]) -> Tuple[int, int, int, int, Dict[int, str]]:
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    mask_id = vocab["<mask>"]
    id2tok = {i: t for t, i in vocab.items()}
    return pad_id, bos_id, eos_id, mask_id, id2tok


def ar_sample_bits(
    model: TinyTransformer,
    vocab: Dict[str, int],
    id2tok: Dict[int, str],
    spec_tokens: List[str],
    universe_size: int,
    temperature: float,
    device: str,
) -> List[str]:
    prefix = ["<bos>"] + spec_tokens + ["[SEP]", "[BITS]"]
    ids = torch.tensor([[vocab[t] for t in prefix]], dtype=torch.long, device=device)
    bit_ids = torch.tensor([vocab["0"], vocab["1"]], dtype=torch.long, device=device)

    for _ in range(universe_size):
        pad_mask = ids.eq(vocab["<pad>"])
        logits = model(ids, pad_mask=pad_mask, causal=True)
        next_logits = logits[:, -1, :].index_select(dim=-1, index=bit_ids)  # (1,2)
        if temperature <= 0:
            sampled = next_logits.argmax(dim=-1, keepdim=True)  # (1,1) in {0,1}
        else:
            probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)  # (1,1) in {0,1}
        next_id = bit_ids[sampled.squeeze(-1)].unsqueeze(-1)  # (1,1) token id
        ids = torch.cat([ids, next_id], dim=1)

    # Append <eos> deterministically (no need to sample).
    ids = torch.cat([ids, torch.tensor([[vocab["<eos>"]]], device=device)], dim=1)

    full = ids.squeeze(0).tolist()
    # Prefix length: <bos> + spec + [SEP] + [BITS]
    bits_start = 1 + len(spec_tokens) + 2
    bits_end = bits_start + universe_size
    bit_tokens = [id2tok[i] for i in full[bits_start:bits_end]]
    bit_tokens = ["1" if t == "1" else "0" for t in bit_tokens]
    return bit_tokens


def md_iterative_sample_bits(
    model: TinyTransformer,
    vocab: Dict[str, int],
    id2tok: Dict[int, str],
    spec_tokens: List[str],
    universe_size: int,
    steps: int,
    temperature: float,
    remask_frac: float,
    device: str,
) -> List[str]:
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask>"]
    bit0 = vocab["0"]
    bit1 = vocab["1"]
    bit_ids = torch.tensor([bit0, bit1], dtype=torch.long, device=device)

    bits_start = 1 + len(spec_tokens) + 2
    bits_end = bits_start + universe_size

    # Parse noisy size if present: ... [SIZE] <n>
    noisy_size = None
    if "[SIZE]" in spec_tokens:
        try:
            idx = spec_tokens.index("[SIZE]")
            noisy_size = int(spec_tokens[idx + 1])
        except Exception:
            noisy_size = None

    # Gibbs-style stochastic denoising:
    #  - initialize bits from a noisy prior (optionally using noisy_size)
    #  - at each step, remask a random subset and resample from the model
    init_bits = []
    if noisy_size is not None:
        m = max(0, min(universe_size, noisy_size))
        ones = set(random.sample(range(universe_size), k=m))
        init_bits = [bit1 if i in ones else bit0 for i in range(universe_size)]
    else:
        init_bits = [bit1 if random.random() < 0.3 else bit0 for _ in range(universe_size)]

    seq = ["<bos>"] + spec_tokens + ["[SEP]", "[BITS]"] + ["0"] * universe_size + ["<eos>"]
    ids = torch.tensor([[vocab[t] for t in seq]], dtype=torch.long, device=device)
    ids[0, bits_start:bits_end] = torch.tensor(init_bits, device=device)

    remask_frac = max(0.05, min(0.95, float(remask_frac)))
    for _step in range(steps):
        # choose positions to remask (always at least 1)
        n_remask = max(1, int(math.ceil(universe_size * remask_frac)))
        pos = random.sample(range(universe_size), k=n_remask)
        for p in pos:
            ids[0, bits_start + p] = mask_id

        pad_mask = ids.eq(pad_id)
        logits = model(ids, pad_mask=pad_mask, causal=False)
        bit_logits = logits[:, bits_start:bits_end, :].index_select(dim=-1, index=bit_ids)  # (1,N,2)
        if temperature <= 0:
            sampled01 = bit_logits.argmax(dim=-1)  # (1,N) in {0,1}
        else:
            probs = torch.softmax(bit_logits / max(1e-6, temperature), dim=-1)
            sampled01 = torch.multinomial(probs.view(-1, 2), num_samples=1).view(1, universe_size)  # in {0,1}
        sampled_ids = bit_ids[sampled01.squeeze(0)].unsqueeze(0)  # (1,N)

        # Fill only masked positions.
        mask_pos = (ids[:, bits_start:bits_end] == mask_id).squeeze(0)
        for i, is_mask in enumerate(mask_pos.tolist()):
            if is_mask:
                ids[0, bits_start + i] = sampled_ids[0, i]

    bit_tokens = [id2tok[i] for i in ids.squeeze(0).tolist()[bits_start:bits_end]]
    bit_tokens = ["1" if t == "1" else "0" for t in bit_tokens]
    return bit_tokens


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def eval_model(
    rows: List[Dict],
    sampler,
    sampler_name: str,
    Kmax: int,
    seed: int,
    compute_diversity: bool,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    buckets = ["short", "medium", "long"]
    out = {
        "sampler": sampler_name,
        "Kmax": Kmax,
        "n": len(rows),
        "overall": {},
        "by_bucket": {b: {} for b in buckets},
    }

    bucket_rows = {b: [] for b in buckets}
    for r in rows:
        bucket_rows[r.get("depth_bucket", "short")].append(r)

    def summarize(group: List[Dict]) -> Dict:
        if not group:
            return {}

        k1_f1 = []
        passk_recall = {k: [] for k in range(1, Kmax + 1)}
        oracle_f1 = {k: [] for k in range(1, Kmax + 1)}
        uniq = {k: [] for k in range(1, Kmax + 1)} if compute_diversity else None
        pair_j = {k: [] for k in range(2, Kmax + 1)} if compute_diversity else None

        for idx, r in enumerate(group, start=1):
            universe = r["universe"]
            gt_nodes = r["gt_nodes"]
            candidates_bits = [sampler(r, seed=seed * 1000003 + i) for i in range(Kmax)]

            candidates_tools = [decode_toolset_from_bits(universe, bits) for bits in candidates_bits]

            _, _, f1_1 = toolset_f1(candidates_tools[0], gt_nodes)
            k1_f1.append(f1_1)

            union: set = set()
            for k in range(1, Kmax + 1):
                union |= set(candidates_tools[k - 1])
                passk_recall[k].append(len(union & set(gt_nodes)) / max(1, len(set(gt_nodes))))

                best = 0.0
                for i in range(k):
                    _, _, f1_i = toolset_f1(candidates_tools[i], gt_nodes)
                    best = max(best, f1_i)
                oracle_f1[k].append(best)

                if compute_diversity:
                    uniq[k].append(len({tuple(candidates_bits[i]) for i in range(k)}))
                    if k >= 2:
                        js = []
                        for i in range(k):
                            for j in range(i + 1, k):
                                js.append(jaccard(candidates_tools[i], candidates_tools[j]))
                        pair_j[k].append(float(np.mean(js)) if js else 1.0)

            if idx % 200 == 0:
                print(f"  [{sampler_name}] processed {idx}/{len(group)} samples")

        summary = {
            "k1_tool_f1": float(np.mean(k1_f1)),
            "passk_recall": {str(k): float(np.mean(v)) for k, v in passk_recall.items()},
            "oracle_f1": {str(k): float(np.mean(v)) for k, v in oracle_f1.items()},
        }
        if compute_diversity:
            summary["unique_candidates"] = {str(k): float(np.mean(v)) for k, v in uniq.items()}
            summary["avg_pairwise_jaccard"] = {str(k): float(np.mean(v)) for k, v in pair_j.items()}
        return summary

    out["overall"] = summarize(rows)
    for b in buckets:
        out["by_bucket"][b] = summarize(bucket_rows[b])

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate synthetic toolset playground v2 (AR vs MD).")
    ap.add_argument("--test_path", default="data/playground_toolset_v2/test.jsonl")
    ap.add_argument("--ar_ckpt", default="experiments/playground_toolset_v2_ar/toolset_ar_best.pt")
    ap.add_argument("--md_ckpt", default="experiments/playground_toolset_v2_md/toolset_md_best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--Kmax", type=int, default=10)
    ap.add_argument("--ar_temperature", type=float, default=0.0, help="AR sampling temperature (0=greedy).")
    ap.add_argument("--md_temperature", type=float, default=0.7, help="MD sampling temperature (0=greedy).")
    ap.add_argument("--md_steps", type=int, default=10)
    ap.add_argument("--md_remask_frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_bucket", type=int, default=200, help="Stratified subsample size per depth bucket (0=all).")
    ap.add_argument("--compute_diversity", action="store_true", help="Compute diversity proxies (slower).")
    ap.add_argument("--out_json", default="results/playground_toolset_v2_eval.json")
    args = ap.parse_args()

    rows = load_jsonl(args.test_path)
    rows = stratified_subsample(rows, max_per_bucket=args.max_per_bucket, seed=args.seed)

    ar_ckpt = torch.load(args.ar_ckpt, map_location="cpu")
    md_ckpt = torch.load(args.md_ckpt, map_location="cpu")
    vocab = ar_ckpt["vocab"]
    pad_id, bos_id, eos_id, mask_id, id2tok = build_id_maps(vocab)

    ar_cfg = ar_ckpt["config"]
    md_cfg = md_ckpt["config"]

    ar_model = TinyTransformer(vocab_size=len(vocab), **ar_cfg).to(args.device)
    ar_model.load_state_dict(ar_ckpt["model_state"])
    ar_model.eval()

    md_model = TinyTransformer(vocab_size=len(vocab), **md_cfg).to(args.device)
    md_model.load_state_dict(md_ckpt["model_state"])
    md_model.eval()

    universe_size = len(rows[0]["universe"])

    def ar_sampler(r: Dict, seed: int) -> List[str]:
        torch.manual_seed(seed)
        random.seed(seed)
        return ar_sample_bits(
            model=ar_model,
            vocab=vocab,
            id2tok=id2tok,
            spec_tokens=r["spec_tokens"],
            universe_size=universe_size,
            temperature=args.ar_temperature,
            device=args.device,
        )

    def md_sampler(r: Dict, seed: int) -> List[str]:
        torch.manual_seed(seed)
        random.seed(seed)
        return md_iterative_sample_bits(
            model=md_model,
            vocab=vocab,
            id2tok=id2tok,
            spec_tokens=r["spec_tokens"],
            universe_size=universe_size,
            steps=args.md_steps,
            temperature=args.md_temperature,
            remask_frac=args.md_remask_frac,
            device=args.device,
        )

    res = {
        "meta": {
            "test_path": args.test_path,
            "Kmax": args.Kmax,
            "ar_temperature": args.ar_temperature,
            "md_temperature": args.md_temperature,
            "md_steps": args.md_steps,
            "md_remask_frac": args.md_remask_frac,
            "seed": args.seed,
            "universe_size": universe_size,
            "max_per_bucket": args.max_per_bucket,
            "compute_diversity": bool(args.compute_diversity),
        },
        "ar": eval_model(rows, ar_sampler, "mini_ar_toolset_v2", args.Kmax, args.seed, compute_diversity=args.compute_diversity),
        "md": eval_model(rows, md_sampler, "mini_md_toolset_v2", args.Kmax, args.seed, compute_diversity=args.compute_diversity),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out_json}")
    print("AR overall:", res["ar"]["overall"]["k1_tool_f1"], f"Pass@{args.Kmax}:", res["ar"]["overall"]["passk_recall"][str(args.Kmax)], f"Oracle@{args.Kmax}:", res["ar"]["overall"]["oracle_f1"][str(args.Kmax)])
    print("MD overall:", res["md"]["overall"]["k1_tool_f1"], f"Pass@{args.Kmax}:", res["md"]["overall"]["passk_recall"][str(args.Kmax)], f"Oracle@{args.Kmax}:", res["md"]["overall"]["oracle_f1"][str(args.Kmax)])


if __name__ == "__main__":
    main()
