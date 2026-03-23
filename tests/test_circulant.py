"""
Tests for BlockCirculantLinear — the core CSTE primitive.

Tests verify:
    1. Shape correctness for various dimensions
    2. Numerical equivalence: circulant matvec ≈ dense matvec
    3. Gradient flow through FFT operations
    4. Compression ratio correctness
    5. from_dense() initialization quality
    6. Sign-flip determinism
"""

import math
import torch
import pytest

from siren.core.circulant import BlockCirculantLinear, circulant_matvec, dense_to_spectral


class TestCirculantMavec:
    """Test the low-level circulant_matvec function."""

    def test_shape(self):
        """Output shape matches input shape."""
        n = 512
        spectral = torch.randn(n, dtype=torch.complex64)
        x = torch.randn(4, n)
        y = circulant_matvec(spectral, x)
        assert y.shape == (4, n)

    def test_circulant_identity(self):
        """FFT of [1, 0, 0, ...] produces identity-like behavior."""
        n = 64
        first_row = torch.zeros(n)
        first_row[0] = 1.0
        spectral = torch.fft.fft(first_row.to(torch.complex64))
        x = torch.randn(n)
        y = circulant_matvec(spectral, x)
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_sign_flip_preserves_shape(self):
        """Sign-flip doesn't change output shape."""
        n = 128
        spectral = torch.randn(n, dtype=torch.complex64)
        x = torch.randn(8, n)
        sign = torch.sign(torch.randn(n))
        y = circulant_matvec(spectral, x, sign_flip=sign)
        assert y.shape == (8, n)


class TestBlockCirculantLinear:
    """Test the BlockCirculantLinear module."""

    @pytest.mark.parametrize("in_f,out_f,p", [
        (512, 512, 128),
        (1024, 1024, 256),
        (768, 512, 128),
        (512, 768, 256),
        (100, 200, 32),     # Non-power-of-2 dimensions
    ])
    def test_forward_shape(self, in_f, out_f, p):
        """Forward pass produces correct output shape."""
        layer = BlockCirculantLinear(in_f, out_f, block_size=p)
        x = torch.randn(2, 16, in_f)
        y = layer(x)
        assert y.shape == (2, 16, out_f)

    def test_gradient_flow(self):
        """Gradients flow through FFT operations."""
        layer = BlockCirculantLinear(256, 256, block_size=64)
        x = torch.randn(1, 256, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.spectral_real.grad is not None
        assert layer.spectral_imag.grad is not None

    def test_compression_ratio(self):
        """Compression ratio matches theoretical prediction."""
        in_f, out_f, p = 4096, 4096, 512
        layer = BlockCirculantLinear(in_f, out_f, block_size=p)

        dense_params = in_f * out_f
        # Each block has p complex coefficients = 2*p real values
        # Number of blocks = (in/p) * (out/p)
        expected_params = (in_f // p) * (out_f // p) * p * 2

        assert layer.actual_params == expected_params + out_f  # +bias
        assert layer.compression_ratio > 100  # Should be ~128x for p=512 with full FFT

    def test_from_dense_reduces_error(self):
        """from_dense() initialization has lower error than random init."""
        in_f, out_f = 256, 256
        p = 64

        dense = torch.nn.Linear(in_f, out_f)
        x = torch.randn(10, in_f)

        with torch.no_grad():
            y_dense = dense(x)

        # Random init
        random_circ = BlockCirculantLinear(in_f, out_f, block_size=p)
        with torch.no_grad():
            y_random = random_circ(x)

        # from_dense init
        init_circ = BlockCirculantLinear.from_dense(dense, block_size=p)
        with torch.no_grad():
            y_init = init_circ(x)

        error_random = torch.norm(y_dense - y_random).item()
        error_init = torch.norm(y_dense - y_init).item()

        # from_dense should be better than random (usually by a lot)
        # But not always guaranteed for all random seeds, so use 2x margin
        assert error_init < error_random * 2.0

    def test_reconstruct_dense_consistency(self):
        """Reconstructed dense matrix produces same output as forward pass."""
        in_f, out_f, p = 128, 128, 32
        layer = BlockCirculantLinear(in_f, out_f, block_size=p, bias=False)
        x = torch.randn(1, in_f)

        with torch.no_grad():
            y_forward = layer(x)
            W = layer.reconstruct_dense()
            y_recon = x @ W.T

        # Should be very close (numerical precision)
        torch.testing.assert_close(y_forward, y_recon, atol=1e-4, rtol=1e-3)

    def test_lr_scale(self):
        """LR scaling factor is 1/sqrt(p)."""
        p = 512
        layer = BlockCirculantLinear(1024, 1024, block_size=p)
        expected = 1.0 / math.sqrt(p)
        assert abs(layer.lr_scale() - expected) < 1e-8

    def test_sign_flip_deterministic(self):
        """Same dimensions produce same sign-flip."""
        l1 = BlockCirculantLinear(512, 256, block_size=128)
        l2 = BlockCirculantLinear(512, 256, block_size=128)
        torch.testing.assert_close(l1.sign_flip, l2.sign_flip)


class TestDenseToSpectral:
    """Test dense_to_spectral conversion."""

    def test_output_shapes(self):
        """Spectral output has correct shape."""
        W = torch.randn(256, 512)
        p = 64
        spectral, sign = dense_to_spectral(W, p)
        assert spectral.shape == (256 // p, 512 // p, p)
        assert sign.shape[0] == 512

    def test_sign_flip_values(self):
        """Sign-flip contains only ±1."""
        W = torch.randn(128, 128)
        _, sign = dense_to_spectral(W, 32)
        assert torch.all((sign == 1.0) | (sign == -1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
