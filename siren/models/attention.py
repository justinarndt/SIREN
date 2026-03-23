"""
Circulant Multi-Head Attention
==============================

Standard multi-head attention uses dense projections for Q, K, V, and output:
    Q = x W_Q,  K = x W_K,  V = x W_V,  out = attn(Q, K, V) W_O

Each projection is O(d²) parameters.  Replacing with BlockCirculantLinear
reduces to O(d²/p) per projection — 4 projections × 512x compression = 
2048x fewer parameters in the attention layer alone.

The attention *mechanism* (scaled dot-product) remains dense — only the
linear projections are circulant-compressed.  This preserves the O(n²d)
attention computation while eliminating the O(d²) parameter bottleneck.

Positional encoding: Rotary Position Embeddings (RoPE) following
Su et al. (2021), which are compatible with circulant projections since
RoPE applies in the head dimension, not the projection dimension.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from siren.core.circulant import BlockCirculantLinear


# ---------------------------------------------------------------------------
#  Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotation frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query/key tensors."""
    # x: (batch, seq, heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


# ---------------------------------------------------------------------------
#  Circulant Multi-Head Attention
# ---------------------------------------------------------------------------

class CirculantMultiHeadAttention(nn.Module):
    """
    Multi-head attention with circulant-compressed linear projections.

    Architecture:
        Q, K, V projections: BlockCirculantLinear (not dense nn.Linear)
        Attention:            Standard scaled dot-product (unchanged)
        Output projection:    BlockCirculantLinear

    Compression: 4 × (d_model²/p) parameters instead of 4 × d_model².

    Args:
        d_model:    Model dimension.
        num_heads:  Number of attention heads.
        block_size: Circulant block size for projections.
        max_seq_len: Maximum sequence length for RoPE.
        dropout:    Attention dropout rate.
        use_rope:   Whether to use rotary position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 512,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope

        # Circulant projections — the key compression
        self.W_q = BlockCirculantLinear(d_model, d_model, block_size, bias=False)
        self.W_k = BlockCirculantLinear(d_model, d_model, block_size, bias=False)
        self.W_v = BlockCirculantLinear(d_model, d_model, block_size, bias=False)
        self.W_o = BlockCirculantLinear(d_model, d_model, block_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # RoPE frequencies
        if use_rope:
            freqs = precompute_rope_freqs(self.head_dim, max_seq_len)
            self.register_buffer("rope_freqs", freqs)

    @property
    def compression_ratio(self) -> float:
        """Average compression across all projections."""
        ratios = [
            self.W_q.compression_ratio,
            self.W_k.compression_ratio,
            self.W_v.compression_ratio,
            self.W_o.compression_ratio,
        ]
        return sum(ratios) / len(ratios)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:    Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask (batch, 1, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, D = x.shape

        # Project Q, K, V through circulant layers
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE
        if self.use_rope:
            q = apply_rope(q, self.rope_freqs)
            k = apply_rope(k, self.rope_freqs)

        # Transpose for attention: (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)

        # Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)

        return out

    @classmethod
    def from_dense(
        cls,
        dense_attn: nn.Module,
        d_model: int,
        num_heads: int,
        block_size: int = 512,
    ) -> "CirculantMultiHeadAttention":
        """Initialize from pretrained dense attention layer weights."""
        layer = cls(d_model, num_heads, block_size)

        # Map dense projection weights to circulant spectral coefficients
        for name, dense_proj in [
            ("W_q", "q_proj"), ("W_k", "k_proj"),
            ("W_v", "v_proj"), ("W_o", "out_proj"),
        ]:
            if hasattr(dense_attn, dense_proj):
                dense_linear = getattr(dense_attn, dense_proj)
                circ = BlockCirculantLinear.from_dense(dense_linear, block_size)
                setattr(layer, name, circ)

        return layer

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"compression={self.compression_ratio:.0f}x, "
            f"params={self.total_params:,}"
        )
