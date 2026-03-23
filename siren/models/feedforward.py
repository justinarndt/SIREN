"""
Circulant Feed-Forward Network (SwiGLU)
=======================================

The feed-forward network in modern transformers uses the SwiGLU activation:
    FFN(x) = (SiLU(x W_gate) ⊙ x W_up) W_down

In a standard 7B model, each FFN layer has 3 dense matrices of size
d_model × d_ff (typically d_ff = 4 × d_model / 3 × 2 for SwiGLU):
    W_gate: d_model × d_ff     (e.g., 4096 × 11008)
    W_up:   d_model × d_ff
    W_down: d_ff    × d_model

Total: ~135M parameters per FFN layer.  With 32 layers: ~4.3B parameters
in feed-forward layers alone (61% of a 7B model).

CSTE compression at p=512: 135M / 512 = ~264K parameters per FFN layer.
With 32 layers: ~8.4M total FFN parameters.  From 4.3B to 8.4M = 512x.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from siren.core.circulant import BlockCirculantLinear


class CirculantFeedForward(nn.Module):
    """
    SwiGLU feed-forward network with circulant-compressed projections.

    Architecture:
        y = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    All three projections (gate, up, down) are BlockCirculantLinear layers.

    Args:
        d_model:     Model dimension.
        d_ff:        Feed-forward intermediate dimension.
                     If None, defaults to 4/3 × d_model × 2 (SwiGLU sizing).
        block_size:  Circulant block size.
        dropout:     Dropout rate after FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        block_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_ff is None:
            # Standard SwiGLU sizing: d_ff = round(4/3 * d_model * 2 / 64) * 64
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # Round to multiple of 64

        self.d_model = d_model
        self.d_ff = d_ff

        # SwiGLU projections — all circulant
        self.gate_proj = BlockCirculantLinear(d_model, d_ff, block_size, bias=False)
        self.up_proj = BlockCirculantLinear(d_model, d_ff, block_size, bias=False)
        self.down_proj = BlockCirculantLinear(d_ff, d_model, block_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def compression_ratio(self) -> float:
        """Average compression across gate/up/down projections."""
        ratios = [
            self.gate_proj.compression_ratio,
            self.up_proj.compression_ratio,
            self.down_proj.compression_ratio,
        ]
        return sum(ratios) / len(ratios)

    @property
    def dense_equivalent_params(self) -> int:
        return (
            self.gate_proj.dense_equivalent_params +
            self.up_proj.dense_equivalent_params +
            self.down_proj.dense_equivalent_params
        )

    @property
    def actual_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward:  down(SiLU(gate(x)) ⊙ up(x))

        Args:
            x: Input tensor of shape (..., d_model).
        Returns:
            Output tensor of shape (..., d_model).
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        out = self.down_proj(hidden)
        return self.dropout(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"compression={self.compression_ratio:.0f}x, "
            f"dense_equiv={self.dense_equivalent_params:,} → "
            f"actual={self.actual_params:,}"
        )
