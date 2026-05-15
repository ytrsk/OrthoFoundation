"""
Pretraining-only metrics/losses.

This repo is intentionally trimmed to only keep what DINO-style pretraining needs:
- `DINOLoss`
- a small `Metric` wrapper (accumulates forward() values and exposes get()).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class Metric(nn.Module):
    """
    Simple accumulator wrapper: call forward() multiple times, then get() returns average.
    """

    def __init__(self, model_ctor: Callable[..., nn.Module], **kwargs) -> None:
        super().__init__()
        self.model = model_ctor(**kwargs)
        self.S = 0.0
        self.n = 0

    def reset(self) -> None:
        self.S = 0.0
        self.n = 0

    def forward(self, *args, **kwargs) -> None:
        y = self.model(*args, **kwargs)
        self.S += y
        self.n += 1

    def get(self):
        return self.S / max(self.n, 1)


class DINOLoss(nn.Module):
    """
    Cross-entropy between softmax outputs of teacher and student (DINO).
    """

    def __init__(
        self,
        out_dim: int = 65536,
        ncrops: int = 10,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, teacher_output, student_output, epoch: int):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = float(self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)])
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)

        total_loss = 0.0
        n_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss = total_loss / max(n_terms, 1)
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / max(len(teacher_output), 1)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def get_metric(tag: str, **kwargs):
    if tag != "dino":
        raise ValueError(f"Only pretraining loss 'dino' is supported in this trimmed repo. Got: {tag}")
    return Metric(DINOLoss, **kwargs)


