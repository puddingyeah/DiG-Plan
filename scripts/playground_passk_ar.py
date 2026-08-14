#!/usr/bin/env python3
"""
Playground AR Pass@k (union-of-k node recall)

目标：
  在 synthetic playground 上，用已经训练好的 mini-AR seq2seq 模型，
  测 Node-level Pass@k 行为：
    - 对每个样本采样 k 次 plan（带 temperature 的采样）
    - 只看节点集合（tool-set），计算 union-of-k node recall
    - 按图深度分桶（short / medium / long）

用于后续与 masked-diffusion 小模型的低层结构对比。
"""

import argparse
import json
import math
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PlaygroundDataset(Dataset):
    def __init__(self, path: str, vocab: Dict[str, int], pad_token: str = "<pad>", bos_token: str = "<bos>", eos_token: str = "<eos>"):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.vocab = vocab
        self.id2tok = {i: t for t, i in vocab.items()}
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_id = vocab[self.pad_token]
        self.bos_id = vocab[self.bos_token]
        self.eos_id = vocab[self.eos_token]

    def __len__(self):
        return len(self.samples)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.vocab[t] for t in tokens]

    def __getitem__(self, idx):
        ex = self.samples[idx]
        spec_ids = self.encode(ex["spec_tokens"])
        return {
            "id": ex["id"],
            "depth": ex["depth"],
            "nodes": ex["nodes"],
            "spec_ids": torch.tensor(spec_ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int):
    spec_seqs = [b["spec_ids"] for b in batch]
    spec_max = max(len(s) for s in spec_seqs)
    spec_batch = torch.full((len(batch), spec_max), pad_id, dtype=torch.long)

    ids = []
    depths = []
    gt_nodes = []
    for i, b in enumerate(batch):
        s = spec_seqs[i]
        spec_batch[i, : len(s)] = s
        ids.append(b["id"])
        depths.append(b["depth"])
        gt_nodes.append(b["nodes"])
    return ids, torch.tensor(depths), gt_nodes, spec_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class ARSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, dim_ff: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_pad_mask):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory

    def decode_step(self, tgt, memory, src_pad_mask, tgt_pad_mask):
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        T = tgt.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        logits = self.out_proj(dec_out)
        return logits


def parse_nodes_from_plan_tokens(tokens: List[str]) -> List[str]:
    nodes: List[str] = []
    mode = None
    for tok in tokens:
        if tok == "[NODES]":
            mode = "nodes"
            continue
        if tok == "[EDGES]":
            mode = "edges"
            continue
        if mode == "nodes":
            nodes.append(tok)
    return nodes


def run_passk_ar(
    ckpt_path: str,
    test_path: str,
    device: str = "cuda:0",
    k_max: int = 10,
    max_gen_len: int = 128,
    temperature: float = 0.7,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    pad_id = ckpt["pad_id"]
    bos_id = ckpt["bos_id"]
    eos_id = ckpt["eos_id"]

    ds = PlaygroundDataset(test_path, vocab=vocab)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    model = ARSeq2Seq(vocab_size=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    id2tok = ds.id2tok

    # buckets: short / medium / long
    buckets = ["short", "medium", "long"]
    recall_per_k = {b: [[] for _ in range(k_max)] for b in buckets}

    with torch.no_grad():
        for ids, depths, gt_nodes_list, src_batch in loader:
            src_batch = src_batch.to(device)
            src_pad_mask = src_batch.eq(pad_id)
            memory = model.encode(src_batch, src_pad_mask)

            B = src_batch.size(0)

            for i in range(B):
                depth = depths[i].item()
                if depth <= 3:
                    bucket = "short"
                elif depth <= 5:
                    bucket = "medium"
                else:
                    bucket = "long"

                gt_nodes = set(gt_nodes_list[i])
                if not gt_nodes:
                    continue

                union_nodes: set = set()

                for k in range(1, k_max + 1):
                    # sampling one plan
                    dec = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                    finished = torch.zeros(1, dtype=torch.bool, device=device)
                    for _ in range(max_gen_len):
                        tgt_pad_mask = dec.eq(pad_id)
                        logits = model.decode_step(dec, memory[i : i + 1], src_pad_mask[i : i + 1], tgt_pad_mask)
                        next_logits = logits[:, -1, :]  # (1, V)
                        probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        dec = torch.cat([dec, next_id.unsqueeze(1)], dim=1)
                        finished |= next_id.eq(eos_id)
                        if finished.all():
                            break

                    gen_ids = dec[:, 1:].squeeze(0).tolist()
                    if eos_id in gen_ids:
                        eos_pos = gen_ids.index(eos_id)
                        gen_ids = gen_ids[:eos_pos]
                    tokens = [id2tok[t] for t in gen_ids if t != pad_id]
                    pred_nodes = parse_nodes_from_plan_tokens(tokens)
                    union_nodes.update(pred_nodes)

                    recall = len(union_nodes & gt_nodes) / len(gt_nodes)
                    recall_per_k[bucket][k - 1].append(recall)

    print(f"\n=== Playground AR Pass@k (node recall, k<= {k_max}) ===")
    for bucket in ["short", "medium", "long"]:
        print(f"\nBucket: {bucket}")
        print(f"{'k':<5}{'avg_recall':<12}{'std':<12}")
        print("-" * 30)
        for k in range(1, k_max + 1):
            vals = recall_per_k[bucket][k - 1]
            if not vals:
                avg = std = 0.0
            else:
                avg = float(np.mean(vals))
                std = float(np.std(vals))
            print(f"{k:<5}{avg:<12.3f}{std:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description="Playground AR Pass@k (union-of-k node recall)")
    parser.add_argument("--ckpt_path", default="experiments/playground_ar/ar_playground_best.pt")
    parser.add_argument("--test_path", default="data/playground/test.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_passk_ar(
        ckpt_path=args.ckpt_path,
        test_path=args.test_path,
        device=args.device,
        k_max=args.k_max,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
