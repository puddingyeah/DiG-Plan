#!/usr/bin/env python3
"""
Playground Masked-Denoiser Pass@k (union-of-k node recall)

目标：
  在 synthetic playground 上，用已经训练好的 mini masked-denoiser 模型，
  测 Node-level Pass@k 行为：
    - 对每个样本采样 k 次 plan（不同的 mask/noise seed）
    - 只看节点集合（tool-set），计算 union-of-k node recall
    - 按图深度分桶（short / medium / long）

用于与 AR 小模型的低层结构对比，验证：
  "masked-diffusion 家族在工具子集空间的搜索更 multi-modal"
"""

import argparse
import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PlaygroundDataset(Dataset):
    def __init__(self, path: str, vocab: Dict[str, int], pad_token: str = "<pad>",
                 bos_token: str = "<bos>", eos_token: str = "<eos>", mask_token: str = "<mask>"):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.vocab = vocab
        self.id2tok = {i: t for t, i in vocab.items()}
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.pad_id = vocab[self.pad_token]
        self.bos_id = vocab[self.bos_token]
        self.eos_id = vocab[self.eos_token]
        self.mask_id = vocab[self.mask_token]

    def __len__(self):
        return len(self.samples)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.vocab[t] for t in tokens]


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


class MaskedDenoiser(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, dim_ff: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask):
        emb = self.pos_encoder(self.embedding(x) * math.sqrt(self.d_model))
        enc = self.encoder(emb, src_key_padding_mask=pad_mask)
        logits = self.out_proj(enc)
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


def iterative_denoise_with_sampling(
    model, seq: torch.Tensor, pad_id: int, mask_id: int, bos_id: int, eos_id: int,
    num_steps: int = 10, temperature: float = 0.7, device: str = "cuda"
):
    """
    带随机性的迭代去噪生成：
    1. 从全 mask（除 BOS/EOS）开始
    2. 每步预测所有 mask 位置，按温度采样并选择一部分 unmask
    3. 通过温度采样引入多样性
    """
    B, L = seq.size()

    # 初始化：除 BOS/EOS/PAD 外全部 mask
    masked_seq = seq.clone()
    for i in range(L):
        if seq[0, i].item() not in [pad_id, bos_id, eos_id]:
            masked_seq[0, i] = mask_id

    masked_seq = masked_seq.to(device)

    for step in range(num_steps):
        mask_positions = (masked_seq == mask_id).squeeze(0)
        n_masked = mask_positions.sum().item()

        if n_masked == 0:
            break

        pad_mask = masked_seq.eq(pad_id)
        with torch.no_grad():
            logits = model(masked_seq, pad_mask)

        # 带温度的采样
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            # 对每个位置采样
            B, L_seq, V = probs.shape
            sampled = torch.multinomial(probs.view(-1, V), 1).view(B, L_seq)
        else:
            sampled = logits.argmax(dim=-1)

        # 计算每个 mask 位置的置信度（用于决定 unmask 顺序）
        # 使用采样后的 token 的概率作为置信度
        if temperature > 0:
            sampled_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            sampled_probs = torch.softmax(logits, dim=-1).max(dim=-1).values

        confidences = sampled_probs.squeeze(0)

        # 每步 unmask 的数量（带随机扰动）
        base_n = max(1, n_masked // (num_steps - step))
        # 加入随机性：在 [base_n * 0.5, base_n * 1.5] 之间
        n_to_unmask = max(1, int(base_n * (0.5 + random.random())))
        n_to_unmask = min(n_to_unmask, n_masked)

        # 只考虑当前还是 mask 的位置
        conf_at_mask = confidences.clone()
        conf_at_mask[~mask_positions] = -float('inf')

        # 加入随机噪声到置信度，增加多样性
        noise = torch.rand_like(conf_at_mask) * 0.1
        conf_at_mask = conf_at_mask + noise

        _, top_indices = conf_at_mask.topk(min(n_to_unmask, n_masked))

        for idx in top_indices:
            masked_seq[0, idx] = sampled[0, idx]

    return masked_seq


def run_passk_md(
    ckpt_path: str,
    test_path: str,
    device: str = "cuda:0",
    k_max: int = 10,
    num_denoise_steps: int = 10,
    temperature: float = 0.7,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    pad_id = ckpt["pad_id"]
    bos_id = ckpt["bos_id"]
    eos_id = ckpt["eos_id"]
    mask_id = ckpt["mask_id"]

    ds = PlaygroundDataset(test_path, vocab=vocab)
    id2tok = ds.id2tok

    model = MaskedDenoiser(vocab_size=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # buckets: short / medium / long
    buckets = ["short", "medium", "long"]
    recall_per_k = {b: [[] for _ in range(k_max)] for b in buckets}

    for sample_idx, sample in enumerate(ds.samples):
        depth = sample["depth"]
        if depth <= 3:
            bucket = "short"
        elif depth <= 5:
            bucket = "medium"
        else:
            bucket = "long"

        gt_nodes = set(sample["nodes"])
        if not gt_nodes:
            continue

        # 准备输入序列
        plan_tokens = sample["plan_tokens"]
        plan_ids = [bos_id] + ds.encode(plan_tokens) + [eos_id]
        seq = torch.tensor(plan_ids, dtype=torch.long).unsqueeze(0)

        union_nodes: set = set()

        for k in range(1, k_max + 1):
            # 每次采样使用不同的随机种子
            random.seed(sample_idx * 1000 + k)
            torch.manual_seed(sample_idx * 1000 + k)

            pred_seq = iterative_denoise_with_sampling(
                model, seq, pad_id, mask_id, bos_id, eos_id,
                num_steps=num_denoise_steps, temperature=temperature, device=device
            )

            pred_ids = pred_seq.squeeze(0).tolist()
            if pred_ids and pred_ids[0] == bos_id:
                pred_ids = pred_ids[1:]
            if pred_ids and pred_ids[-1] == eos_id:
                pred_ids = pred_ids[:-1]

            pred_tokens = [id2tok.get(t, "<unk>") for t in pred_ids if t != pad_id and t != mask_id]
            pred_nodes = parse_nodes_from_plan_tokens(pred_tokens)
            union_nodes.update(pred_nodes)

            recall = len(union_nodes & gt_nodes) / len(gt_nodes)
            recall_per_k[bucket][k - 1].append(recall)

        if (sample_idx + 1) % 100 == 0:
            print(f"Processed {sample_idx + 1}/{len(ds.samples)} samples...", flush=True)

    print(f"\n=== Playground Masked-Denoiser Pass@k (node recall, k<= {k_max}) ===")
    print(f"Temperature: {temperature}, Denoise steps: {num_denoise_steps}")

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

    # 返回结果用于对比
    results = {}
    for bucket in buckets:
        results[bucket] = []
        for k in range(1, k_max + 1):
            vals = recall_per_k[bucket][k - 1]
            avg = float(np.mean(vals)) if vals else 0.0
            results[bucket].append(avg)

    return results


def main():
    parser = argparse.ArgumentParser(description="Playground Masked-Denoiser Pass@k (union-of-k node recall)")
    parser.add_argument("--ckpt_path", default="experiments/playground_md/md_playground_best.pt")
    parser.add_argument("--test_path", default="data/playground/test.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--num_denoise_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_passk_md(
        ckpt_path=args.ckpt_path,
        test_path=args.test_path,
        device=args.device,
        k_max=args.k_max,
        num_denoise_steps=args.num_denoise_steps,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
