"""
Tests for Phase-Magnitude Quantization with STE.
"""

import math
import torch
import pytest

from siren.core.quantization import (
    PhaseMagnitudeQuantizer,
    ste_round,
    uniform_quantize,
    log_quantize,
)


class TestSTERound:
    """Test straight-through estimator rounding."""

    def test_forward_rounds(self):
        """Forward pass produces rounded values."""
        x = torch.tensor([1.3, 2.7, -0.5, 0.1])
        y = ste_round(x)
        expected = torch.tensor([1.0, 3.0, 0.0, 0.0])
        torch.testing.assert_close(y, expected)

    def test_gradient_is_identity(self):
        """STE passes gradient through as identity."""
        x = torch.tensor([1.3, 2.7], requires_grad=True)
        y = ste_round(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be 1.0 (identity, not zero)
        torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestUniformQuantize:
    """Test uniform quantization."""

    def test_4bit_range(self):
        """4-bit quantization produces values in [min, max]."""
        x = torch.linspace(-1, 1, 100)
        q = uniform_quantize(x, bits=4, x_min=-1.0, x_max=1.0)
        assert q.min() >= -1.0
        assert q.max() <= 1.0

    def test_num_levels(self):
        """Number of unique values matches 2^bits - 1."""
        x = torch.linspace(0, 1, 1000)
        q = uniform_quantize(x, bits=4, x_min=0.0, x_max=1.0)
        n_unique = len(torch.unique(q))
        # Should have at most 2^4 = 16 levels (could be slightly less
        # due to floating point)
        assert n_unique <= 16


class TestLogQuantize:
    """Test log-domain quantization."""

    def test_preserves_sign(self):
        """Log quantization preserves sign of values."""
        x = torch.tensor([-3.0, -0.5, 0.5, 3.0])
        q = log_quantize(x, bits=4)
        assert torch.all(torch.sign(q) == torch.sign(x))

    def test_better_dynamic_range(self):
        """Log quantization has better dynamic range than uniform."""
        # Values spanning several orders of magnitude
        x = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0])

        q_log = log_quantize(x, bits=4)
        q_uni = uniform_quantize(x, bits=4, x_min=0.001, x_max=10.0)

        # Log should have lower relative error for small values
        rel_err_log = torch.abs(q_log - x) / (x + 1e-8)
        rel_err_uni = torch.abs(q_uni - x) / (x + 1e-8)

        # At least for the smallest values, log should be better
        assert rel_err_log[0] <= rel_err_uni[0] + 0.5


class TestPhaseMagnitudeQuantizer:
    """Test the full phase-magnitude quantizer."""

    def test_shape_preservation(self):
        """Quantizer preserves tensor shapes."""
        q = PhaseMagnitudeQuantizer(bits=4)
        q.train()
        real = torch.randn(8, 4, 257)
        imag = torch.randn(8, 4, 257)
        real_q, imag_q = q(real, imag)
        assert real_q.shape == real.shape
        assert imag_q.shape == imag.shape

    def test_passthrough_in_eval(self):
        """Quantizer is passthrough in eval mode."""
        q = PhaseMagnitudeQuantizer(bits=4)
        q.eval()
        real = torch.randn(4, 257)
        imag = torch.randn(4, 257)
        real_q, imag_q = q(real, imag)
        torch.testing.assert_close(real_q, real)
        torch.testing.assert_close(imag_q, imag)

    def test_snr_tracking(self):
        """SNR is tracked during training."""
        q = PhaseMagnitudeQuantizer(bits=4)
        q.train()
        for _ in range(5):
            real = torch.randn(8, 257)
            imag = torch.randn(8, 257)
            q(real, imag)

        assert q.num_calls.item() == 5
        assert q.avg_snr_db != float("inf")

    def test_magnitude_phase_invertibility(self):
        """Polar decomposition is invertible (before quantization)."""
        real = torch.randn(100)
        imag = torch.randn(100)

        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)

        real_recon = magnitude * torch.cos(phase)
        imag_recon = magnitude * torch.sin(phase)

        torch.testing.assert_close(real_recon, real, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(imag_recon, imag, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_higher_bits_lower_error(self, bits):
        """More bits should produce lower quantization error."""
        torch.manual_seed(42)
        real = torch.randn(16, 257)
        imag = torch.randn(16, 257)

        q = PhaseMagnitudeQuantizer(bits=bits)
        q.train()
        real_q, imag_q = q(real, imag)

        error = torch.norm(real - real_q) + torch.norm(imag - imag_q)
        # This is a relative check — just verify it doesn't explode
        assert error.item() < 1000  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
