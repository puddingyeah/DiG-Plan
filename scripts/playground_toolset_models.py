#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    pad_id: int
    bos_id: int
    eos_id: int
    mask_id: int


def build_vocab_from_tokens(tokens: List[str]) -> Vocab:
    specials = ["<pad>", "<bos>", "<eos>", "<mask>"]
    all_tokens = sorted(set(specials + tokens))
    token_to_id = {t: i for i, t in enumerate(all_tokens)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=token_to_id["<pad>"],
        bos_id=token_to_id["<bos>"],
        eos_id=token_to_id["<eos>"],
        mask_id=token_to_id["<mask>"],
    )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyTransformer(nn.Module):
    """
    A small Transformer backbone shared across experiments.
    We vary:
      - attention mask (causal vs bidirectional)
      - training objective (AR next-token vs masked denoising)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 2, dim_ff: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        attn_mask = None
        if causal:
            T = input_ids.size(1)
            attn_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.lm_head(h)


def decode_toolset_from_bits(universe: List[str], bit_tokens: List[str]) -> List[str]:
    tools = []
    for t, b in zip(universe, bit_tokens):
        if b == "1":
            tools.append(t)
    return tools


def toolset_f1(pred: List[str], gt: List[str]) -> Tuple[float, float, float]:
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gt_set)
    p = tp / len(pred_set)
    r = tp / max(1, len(gt_set))
    f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
    return p, r, f1
