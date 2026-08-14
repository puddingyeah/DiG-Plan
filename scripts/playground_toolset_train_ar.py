#!/usr/bin/env python3
"""
Train prefix-AR toolset predictor on synthetic toolset playground v2.

Objective:
  Next-token prediction on a fixed-length bitvector (N bits) conditioned on spec tokens.

Sequence format:
  <bos> [SPEC ...] [SEP] [BITS] b1 b2 ... bN <eos>

Loss is applied to tokens after [BITS] (including b1..bN and <eos>).
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


class ToolsetARDataset(Dataset):
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


def collate(batch: List[Dict], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_pad = torch.ones((len(batch), max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        ids = b["input_ids"]
        input_ids[i, : len(ids)] = ids
        attn_pad[i, : len(ids)] = False

    x = input_ids[:, :-1].contiguous()
    y = input_ids[:, 1:].contiguous()
    pad_mask = attn_pad[:, :-1].contiguous()
    return x, y, pad_mask


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = ToolsetARDataset(args.train_path, build_vocab=True)
    vocab = train_ds.vocab
    valid_ds = ToolsetARDataset(args.valid_path, vocab=vocab, build_vocab=False)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate(b, vocab.pad_id),
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, vocab.pad_id))

    model = TinyTransformer(vocab_size=len(vocab.token_to_id), d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers, dim_ff=args.dim_ff)
    model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    best_valid = 1e9
    steps = 0
    for epoch in range(1, args.epochs + 1):
        reached_max_steps = False
        model.train()
        tr_loss = 0.0
        n = 0
        for x, y, pad_mask in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            pad_mask = pad_mask.to(args.device)
            logits = model(x, pad_mask=pad_mask, causal=True)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
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
            for x, y, pad_mask in valid_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                pad_mask = pad_mask.to(args.device)
                logits = model(x, pad_mask=pad_mask, causal=True)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
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
            out = os.path.join(args.output_dir, "toolset_ar_best.pt")
            torch.save(ckpt, out)
            print(f"saved {out}")

        if reached_max_steps:
            print(f"reached max_steps={args.max_steps} at epoch={epoch}", flush=True)
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Train prefix-AR toolset predictor (synthetic playground v2).")
    ap.add_argument("--train_path", default="data/playground_toolset_v2/train.jsonl")
    ap.add_argument("--valid_path", default="data/playground_toolset_v2/valid.jsonl")
    ap.add_argument("--output_dir", default="experiments/playground_toolset_v2_ar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0, help="If >0, stop after this many optimizer steps (compute-matching).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_ff", type=int, default=256)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
