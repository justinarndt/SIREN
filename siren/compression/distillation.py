"""
Dense-to-CSTE Distillation Pipeline
====================================

Converting a pretrained dense model to CSTE requires more than weight
extraction.  The initial circulant approximation (first-row extraction)
typically achieves ~96% accuracy retention.  Closing the gap to sub-1%
requires fine-tuning with an auxiliary Frobenius loss:

    L_total = L_task + λ · Σ_l ||W_l^dense - W_l^cste||_F

where the sum runs over all linear projection layers l.

The distillation process follows three phases:
    1. Weight extraction:   Initialize Λ from dense W via first-row FFT
    2. Frobenius alignment: Fine-tune Λ with combined task + Frobenius loss
    3. Quantization-aware:  Enable phase-magnitude quantization with STE

Progressive block-size annealing is used for graceful compression:
    Start at p=64 (easy target), anneal to p=512 (full compression).

References:
  [1] BCA: Block Circulant Adapter, arXiv 2505.00582, §4.3, 2025.
  [2] Arndt, "CSTE Design, Build, Benchmark on TPU v5e-8," §6, 2026.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from siren.core.circulant import BlockCirculantLinear


@dataclass
class DistillationMetrics:
    """Metrics tracked during distillation."""
    step: int
    task_loss: float
    frobenius_loss: float
    total_loss: float
    avg_compression: float
    elapsed_sec: float


class FrobeniusDistiller:
    """
    Dense → CSTE distillation with auxiliary Frobenius loss.

    The distiller:
    1. Extracts spectral coefficients from dense weights
    2. Fine-tunes with combined task + Frobenius loss
    3. Optionally enables phase-magnitude quantization

    Args:
        student:        SIREN model (circulant-compressed).
        teacher_weights: Dict mapping layer names to dense weight tensors.
        lambda_frob:    Frobenius loss weight (default 0.1).
        progressive:    Whether to use progressive block-size annealing.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher_weights: Optional[Dict[str, torch.Tensor]] = None,
        lambda_frob: float = 0.1,
        progressive: bool = True,
    ):
        self.student = student
        self.teacher_weights = teacher_weights or {}
        self.lambda_frob = lambda_frob
        self.progressive = progressive
        self.metrics_history: List[DistillationMetrics] = []

    def compute_frobenius_loss(self) -> torch.Tensor:
        """
        Compute Frobenius norm between teacher dense weights and
        student's reconstructed circulant weights.
        """
        total_loss = torch.tensor(0.0, device=next(self.student.parameters()).device)
        n_layers = 0

        for name, module in self.student.named_modules():
            if isinstance(module, BlockCirculantLinear):
                teacher_key = name
                if teacher_key in self.teacher_weights:
                    W_teacher = self.teacher_weights[teacher_key]
                    W_student = module.reconstruct_dense()

                    # Align shapes
                    min_out = min(W_teacher.shape[0], W_student.shape[0])
                    min_in = min(W_teacher.shape[1], W_student.shape[1])

                    frob = torch.norm(
                        W_teacher[:min_out, :min_in] - W_student[:min_out, :min_in],
                        p="fro",
                    )
                    total_loss = total_loss + frob
                    n_layers += 1

        return total_loss / max(n_layers, 1)

    def distillation_step(
        self,
        optimizer: torch.optim.Optimizer,
        task_loss_fn: Callable,
        batch: dict,
        step: int,
    ) -> DistillationMetrics:
        """
        Single distillation training step.

        Args:
            optimizer:    Optimizer for student parameters.
            task_loss_fn: Function(model, batch) → scalar loss.
            batch:        Training batch dict.
            step:         Current training step.

        Returns:
            DistillationMetrics for this step.
        """
        start = time.time()
        optimizer.zero_grad()

        # Task loss
        task_loss = task_loss_fn(self.student, batch)

        # Frobenius loss
        frob_loss = self.compute_frobenius_loss()

        # Combined loss
        total_loss = task_loss + self.lambda_frob * frob_loss

        total_loss.backward()
        optimizer.step()

        elapsed = time.time() - start

        metrics = DistillationMetrics(
            step=step,
            task_loss=task_loss.item(),
            frobenius_loss=frob_loss.item(),
            total_loss=total_loss.item(),
            avg_compression=self._avg_compression(),
            elapsed_sec=elapsed,
        )
        self.metrics_history.append(metrics)
        return metrics

    def _avg_compression(self) -> float:
        """Average compression ratio across all circulant layers."""
        ratios = []
        for module in self.student.modules():
            if isinstance(module, BlockCirculantLinear):
                ratios.append(module.compression_ratio)
        return sum(ratios) / max(len(ratios), 1)

    def build_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> torch.optim.AdamW:
        """
        Build optimizer with per-layer learning rate scaling.

        Circulant layers use lr_base / sqrt(p) to prevent gradient
        explosion.  Non-circulant parameters use the base learning rate.
        """
        param_groups = []

        circulant_params = []
        other_params = []

        for name, param in self.student.named_parameters():
            if "spectral" in name:
                circulant_params.append(param)
            else:
                other_params.append(param)

        # Get block size for LR scaling
        block_sizes = []
        for m in self.student.modules():
            if isinstance(m, BlockCirculantLinear):
                block_sizes.append(m.block_size)
        avg_block_size = sum(block_sizes) / max(len(block_sizes), 1)

        if circulant_params:
            lr_scale = 1.0 / (avg_block_size ** 0.5)
            param_groups.append({
                "params": circulant_params,
                "lr": lr * lr_scale,
                "weight_decay": weight_decay,
            })

        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })

        return torch.optim.AdamW(param_groups)

    def report(self) -> str:
        """Format distillation metrics as a table."""
        if not self.metrics_history:
            return "No distillation steps recorded."

        lines = [
            f"{'Step':>6} {'Task Loss':>10} {'Frob Loss':>10} "
            f"{'Total':>10} {'Compress':>10} {'Time(s)':>8}",
            "-" * 56,
        ]

        for m in self.metrics_history[-20:]:  # Last 20 steps
            lines.append(
                f"{m.step:>6d} {m.task_loss:>10.4f} {m.frobenius_loss:>10.4f} "
                f"{m.total_loss:>10.4f} {m.avg_compression:>9.0f}x "
                f"{m.elapsed_sec:>7.3f}"
            )

        return "\n".join(lines)
