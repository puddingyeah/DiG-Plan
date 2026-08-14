#!/usr/bin/env python3
"""
Utilities for working with Dream diffusion model confidence signals.

Currently exposes:
  - compute_plan_confidence(mask_entropies, drop_first_steps=3)
"""

from typing import List

import torch


def compute_plan_confidence(mask_entropies: List[torch.Tensor], drop_first_steps: int = 3) -> float:
    """
    根据 mask entropy 计算一个简单的 plan-level confidence 标量。

    思路:
      - 略过前几个 step（通常非常混沌），默认 drop_first_steps=3
      - 将剩余 step 的 entropy 拼接，取均值 mean_ent
      - 用一个单调递减映射将 entropy -> confidence:
          confidence = 1 / (1 + mean_ent)
      这样 entropy 越大, confidence 越小；entropy=0 时 confidence=1。
    """
    if not mask_entropies:
        return 0.0

    usable = mask_entropies[drop_first_steps:] if len(mask_entropies) > drop_first_steps else mask_entropies
    usable = [e for e in usable if e.numel() > 0]
    if not usable:
        return 0.0

    all_ent = torch.cat(usable, dim=0).float()
    mean_ent = float(all_ent.mean().item())
    confidence = 1.0 / (1.0 + mean_ent)
    return confidence
