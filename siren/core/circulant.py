"""
Block-Circulant Linear Layer — The Heart of CSTE
=================================================

A circulant matrix C is fully defined by its first row c ∈ ℝⁿ.  Subsequent
rows are cyclic shifts:  C[i, j] = c[(j − i) mod n].

Key identity:  C = F⁻¹ diag(F c) F   where F is the DFT matrix.

Therefore:   C x = IFFT(FFT(c) ⊙ FFT(x))        — O(n log n), O(n) storage.

Block-circulant extension:  partition W ∈ ℝ^{m×n} into a grid of B² circulant
sub-blocks of size p = n/B.  Compression: n²  →  B² · p  =  n²/p.

With p = 512:  compression ratio = 512x  per matrix.

References:
  [1] Cheng et al., "Fast Neural Networks with Circulant Projections," 2015.
  [2] BCA: Block Circulant Adapter for LLMs, arXiv 2505.00582, 2025.
  [3] Arndt, "CSTE Design, Build, Benchmark on TPU v5e-8," 2026.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Low-level primitives
# ---------------------------------------------------------------------------

def circulant_matvec(
    spectral_coeffs: torch.Tensor,
    x: torch.Tensor,
    sign_flip: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute y = C x  via FFT, where C is the circulant matrix defined by
    *spectral_coeffs* (the pre-computed FFT of the first row).

    Uses complex fft/ifft instead of rfft/irfft for MKL compatibility.

    Args:
        spectral_coeffs: Complex tensor of shape (n,) — the fft of
                         the circulant's first row.
        x:               Real tensor of shape (..., n).
        sign_flip:       Optional ±1 tensor of shape (n,) for Bernoulli
                         scrambling (Cheng et al. 2015, §3.1).

    Returns:
        y: Real tensor of shape (..., n).
    """
    if sign_flip is not None:
        x = x * sign_flip
    X = torch.fft.fft(x.to(torch.complex64), dim=-1)
    Y = X * spectral_coeffs
    return torch.fft.ifft(Y, dim=-1).real


def dense_to_spectral(
    weight: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate a dense weight matrix W ∈ ℝ^{m×n} with a block-circulant
    matrix.  Returns the spectral coefficients for each block and a random
    sign-flip vector.

    The approximation extracts the first row of each p×p block.  While this
    is a rank-1 circulant fit per block, fine-tuning with Frobenius loss
    closes the gap to sub-1% accuracy loss (CSTE paper, Table 3).

    Args:
        weight:     Dense matrix of shape (out_features, in_features).
        block_size: Circulant block size p.

    Returns:
        spectral:  Complex tensor (num_blocks_out, num_blocks_in, p).
        sign_flip: ±1 tensor of shape (in_features,).
    """
    out_f, in_f = weight.shape
    p = block_size

    # Pad to multiples of p
    pad_out = (p - out_f % p) % p
    pad_in = (p - in_f % p) % p
    if pad_out > 0 or pad_in > 0:
        weight = F.pad(weight, (0, pad_in, 0, pad_out))

    m, n = weight.shape
    B_out, B_in = m // p, n // p

    # Generate deterministic sign-flip from hash of matrix shape
    gen = torch.Generator(device=weight.device)
    gen.manual_seed(in_f * 31 + out_f * 17)
    sign_flip = torch.where(
        torch.rand(n, generator=gen, device=weight.device) > 0.5,
        torch.ones(n, device=weight.device),
        -torch.ones(n, device=weight.device),
    )

    # Extract first row of each block → spectral coefficients
    spectral = torch.zeros(B_out, B_in, p,
                           dtype=torch.complex64, device=weight.device)

    for i in range(B_out):
        for j in range(B_in):
            block = weight[i * p:(i + 1) * p, j * p:(j + 1) * p]
            first_row = block[0]  # First row defines the circulant
            spectral[i, j] = torch.fft.fft(first_row.to(torch.complex64))

    return spectral, sign_flip


# ---------------------------------------------------------------------------
#  BlockCirculantLinear — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class BlockCirculantLinear(nn.Module):
    """
    Block-circulant linear layer.

    Replaces nn.Linear(in_features, out_features) with a grid of circulant
    sub-blocks, reducing parameter count from in×out to (in×out)/p and
    compute from O(in×out) to O(in×out/p × log p).

    Architecture (forward pass):
        1. Apply sign-flip diagonal:  x̃ = D · x
        2. Partition x̃ into B_in blocks of size p
        3. For each output block i, for each input block j:
              y_i += IFFT(Λ_{i,j} ⊙ FFT(x̃_j))
        4. Concatenate output blocks, truncate to out_features

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        block_size:   Circulant block size p (default 512).
        bias:         Whether to include bias (default True).
        quantize:     Whether to apply phase-magnitude quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 512,
        bias: bool = True,
        quantize: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.quantize = quantize

        p = block_size

        # Compute padded dimensions
        self.padded_in = in_features + (p - in_features % p) % p
        self.padded_out = out_features + (p - out_features % p) % p
        self.num_blocks_in = self.padded_in // p
        self.num_blocks_out = self.padded_out // p

        # Trainable spectral coefficients: Λ ∈ C^{B_out × B_in × p}
        # Stored as real+imag tensors for optimizer compatibility
        # Uses full FFT (size p) instead of rfft (size p//2+1) for MKL compat
        spectral_shape = (self.num_blocks_out, self.num_blocks_in, p)
        self.spectral_real = nn.Parameter(
            torch.randn(*spectral_shape) * (1.0 / math.sqrt(p))
        )
        self.spectral_imag = nn.Parameter(
            torch.randn(*spectral_shape) * (1.0 / math.sqrt(p))
        )

        # Fixed sign-flip diagonal (Bernoulli ±1, not trainable)
        gen = torch.Generator()
        gen.manual_seed(in_features * 31 + out_features * 17)
        sign = torch.where(
            torch.rand(self.padded_in, generator=gen) > 0.5,
            torch.ones(self.padded_in),
            -torch.ones(self.padded_in),
        )
        self.register_buffer("sign_flip", sign)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @property
    def spectral_coeffs(self) -> torch.Tensor:
        """Complex spectral coefficients Λ."""
        return torch.complex(self.spectral_real, self.spectral_imag)

    @property
    def compression_ratio(self) -> float:
        """Parameter compression vs equivalent dense layer."""
        dense_params = self.in_features * self.out_features
        circ_params = self.spectral_real.numel() + self.spectral_imag.numel()
        if self.bias is not None:
            circ_params += self.out_features
        return dense_params / max(circ_params, 1)

    @property
    def dense_equivalent_params(self) -> int:
        return self.in_features * self.out_features

    @property
    def actual_params(self) -> int:
        total = self.spectral_real.numel() + self.spectral_imag.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        batch_shape = x.shape[:-1]
        p = self.block_size

        # 1. Pad input to multiple of p
        if x.shape[-1] < self.padded_in:
            x = F.pad(x, (0, self.padded_in - x.shape[-1]))

        # 2. Apply sign-flip
        x = x * self.sign_flip

        # 3. Reshape into blocks: (..., B_in, p)
        x_blocks = x.reshape(*batch_shape, self.num_blocks_in, p)

        # 4. FFT all input blocks at once (complex fft for MKL compat)
        X_blocks = torch.fft.fft(x_blocks.to(torch.complex64), dim=-1)

        # 5. Block-circulant multiply in spectral domain
        # Λ is (B_out, B_in, p), X is (..., B_in, p)
        # Result: (..., B_out, p) via einsum
        Lambda = self.spectral_coeffs
        Y_blocks = torch.einsum("oif,...if->...of", Lambda, X_blocks)

        # 6. IFFT back to spatial domain
        y_blocks = torch.fft.ifft(Y_blocks, dim=-1).real  # (..., B_out, p)

        # 7. Reshape and truncate
        y = y_blocks.reshape(*batch_shape, self.padded_out)
        y = y[..., :self.out_features]

        # 8. Bias
        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
        block_size: int = 512,
    ) -> "BlockCirculantLinear":
        """
        Initialize from a pre-trained dense nn.Linear layer.

        This extracts the first row of each block from the weight matrix
        as the initial spectral coefficients.  Fine-tuning with Frobenius
        loss is recommended to close the accuracy gap.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            block_size=block_size,
            bias=linear.bias is not None,
        )

        with torch.no_grad():
            spectral, sign_flip = dense_to_spectral(
                linear.weight.data, block_size
            )
            layer.spectral_real.copy_(spectral.real)
            layer.spectral_imag.copy_(spectral.imag)
            layer.sign_flip.copy_(sign_flip[:layer.padded_in])

            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias.data)

        return layer

    def reconstruct_dense(self) -> torch.Tensor:
        """
        Reconstruct the full dense weight matrix from spectral coefficients.
        Used for Frobenius distillation loss computation.

        Returns:
            W_reconstructed: Dense matrix of shape (padded_out, padded_in).
        """
        p = self.block_size
        Lambda = self.spectral_coeffs  # (B_out, B_in, p//2+1)

        # Build each circulant block from its spectral coeffs
        rows = []
        for i in range(self.num_blocks_out):
            cols = []
            for j in range(self.num_blocks_in):
                # First row from IFFT of spectral coeffs
                first_row = torch.fft.ifft(Lambda[i, j]).real
                # Build full circulant matrix from first row
                indices = torch.arange(p, device=first_row.device)
                circ_indices = (indices.unsqueeze(1) - indices.unsqueeze(0)) % p
                block = first_row[circ_indices]
                cols.append(block)
            rows.append(torch.cat(cols, dim=1))
        W = torch.cat(rows, dim=0)

        # Apply inverse sign-flip
        W = W * self.sign_flip.unsqueeze(0)

        return W[:self.out_features, :self.in_features]

    def lr_scale(self) -> float:
        """
        Learning rate scaling factor for stable training.

        From BCA (§4.2): lr_eff = lr_base / sqrt(p) prevents gradient
        explosion in circulant layers where one spectral parameter
        affects p spatial-domain entries.
        """
        return 1.0 / math.sqrt(self.block_size)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"p={self.block_size}, "
            f"blocks={self.num_blocks_out}×{self.num_blocks_in}, "
            f"compression={self.compression_ratio:.0f}x, "
            f"params={self.actual_params:,}"
        )
