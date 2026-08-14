#!/usr/bin/env python3
"""
Train masked-denoising toolset predictor on synthetic toolset playground v2.

Objective:
  Masked token prediction on the fixed-length bitvector conditioned on spec tokens,
  using a bidirectional attention mask (no causal mask).

Sequence format:
  <bos> [SPEC ...] [SEP] [BITS] b1 b2 ... bN <eos>

Training:
  - Randomly mask a fraction of the bit tokens (b1..bN) into <mask>
  - Predict only the masked bit positions ('0'/'1') via cross-entropy
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from playground_toolset_models import TinyTransformer, Vocab, build_vocab_from_tokens


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class ToolsetMDDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab = None, build_vocab: bool = False):
        self.rows = load_jsonl(path)
        if build_vocab:
            tokens = set()
            for r in self.rows:
                tokens.update(r["spec_tokens"])
                tokens.update(r["gt_bits"])
            tokens.update(["[SEP]", "[BITS]", "0", "1"])
            self.vocab = build_vocab_from_tokens(sorted(tokens))
        else:
            assert vocab is not None
            self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.vocab.token_to_id[t] for t in tokens]

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        spec = r["spec_tokens"]
        bits = r["gt_bits"]
        seq = ["<bos>"] + spec + ["[SEP]", "[BITS]"] + bits + ["<eos>"]
        ids = self.encode(seq)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "spec_len": len(spec),
            "universe": r["universe"],
        }


def collate(batch: List[Dict], pad_id: int, mask_id: int, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target = torch.full((len(batch), max_len), -100, dtype=torch.long)
    pad_mask = torch.ones((len(batch), max_len), dtype=torch.bool)

    for i, b in enumerate(batch):
        ids = b["input_ids"].tolist()
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        pad_mask[i, :L] = False

        eos_pos = L - 1
        # Fixed structure:
        # <bos> spec... [SEP] [BITS] bits... <eos>
        # Sequence: <bos> spec... [SEP] [BITS] bits... <eos>
        # Positions: 0=<bos>, 1..spec_len=spec, spec_len+1=[SEP], spec_len+2=[BITS], spec_len+3=bit_1
        bits_start = b["spec_len"] + 3
        bits_end = eos_pos

        for j in range(bits_start, bits_end):
            if random.random() < mask_prob:
                target[i, j] = input_ids[i, j]
                input_ids[i, j] = mask_id

    return input_ids, target, pad_mask


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = ToolsetMDDataset(args.train_path, build_vocab=True)
    vocab = train_ds.vocab
    valid_ds = ToolsetMDDataset(args.valid_path, vocab=vocab, build_vocab=False)

    def train_collate(b):
        p = args.mask_prob
        if args.mask_prob_max is not None:
            p = random.uniform(args.mask_prob, args.mask_prob_max)
        return collate(b, vocab.pad_id, vocab.mask_id, p)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=train_collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, vocab.pad_id, vocab.mask_id, args.valid_mask_prob))

    model = TinyTransformer(vocab_size=len(vocab.token_to_id), d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers, dim_ff=args.dim_ff)
    model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_valid = 1e9
    steps = 0
    for epoch in range(1, args.epochs + 1):
        reached_max_steps = False
        model.train()
        tr_loss = 0.0
        n = 0
        for x, tgt, pad_mask in train_loader:
            x = x.to(args.device)
            tgt = tgt.to(args.device)
            pad_mask = pad_mask.to(args.device)
            logits = model(x, pad_mask=pad_mask, causal=False)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            n += 1
            steps += 1
            if args.max_steps > 0 and steps >= args.max_steps:
                reached_max_steps = True
                break

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for x, tgt, pad_mask in valid_loader:
                x = x.to(args.device)
                tgt = tgt.to(args.device)
                pad_mask = pad_mask.to(args.device)
                logits = model(x, pad_mask=pad_mask, causal=False)
                loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
                va_loss += loss.item()
                vn += 1

        tr = tr_loss / max(1, n)
        va = va_loss / max(1, vn)
        print(f"epoch {epoch} train_loss={tr:.4f} valid_loss={va:.4f}")

        if va < best_valid:
            best_valid = va
            ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab.token_to_id,
                "pad_id": vocab.pad_id,
                "bos_id": vocab.bos_id,
                "eos_id": vocab.eos_id,
                "mask_id": vocab.mask_id,
                "config": {
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "num_layers": args.num_layers,
                    "dim_ff": args.dim_ff,
                },
            }
            out = os.path.join(args.output_dir, "toolset_md_best.pt")
            torch.save(ckpt, out)
            print(f"saved {out}")

        if reached_max_steps:
            print(f"reached max_steps={args.max_steps} at epoch={epoch}", flush=True)
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Train masked-denoising toolset predictor (synthetic playground v2).")
    ap.add_argument("--train_path", default="data/playground_toolset_v2/train.jsonl")
    ap.add_argument("--valid_path", default="data/playground_toolset_v2/valid.jsonl")
    ap.add_argument("--output_dir", default="experiments/playground_toolset_v2_md")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0, help="If >0, stop after this many optimizer steps (compute-matching).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask_prob", type=float, default=0.25)
    ap.add_argument("--mask_prob_max", type=float, default=0.9, help="If set, sample mask_prob ~ Uniform(mask_prob, mask_prob_max) per batch.")
    ap.add_argument("--valid_mask_prob", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_ff", type=int, default=256)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
