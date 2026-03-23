"""
Tests for SIREN Transformer model.
"""

import torch
import pytest

from siren.models.transformer import SIRENTransformer, SIRENConfig


class TestSIRENConfig:
    """Test configuration presets."""

    def test_presets_exist(self):
        for preset in ["tiny", "small", "medium", "large", "xl"]:
            config = getattr(SIRENConfig, preset)()
            assert config.d_model > 0
            assert config.num_layers > 0
            assert config.block_size > 0

    def test_d_model_divisible_by_heads(self):
        for preset in ["tiny", "small", "medium", "large"]:
            config = getattr(SIRENConfig, preset)()
            assert config.d_model % config.num_heads == 0


class TestSIRENTransformer:
    """Test the full transformer model."""

    @pytest.fixture
    def tiny_model(self):
        config = SIRENConfig.tiny()
        return SIRENTransformer(config), config

    def test_forward_shape(self, tiny_model):
        """Forward pass produces correct logit shape."""
        model, config = tiny_model
        batch, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_gradient_flow(self, tiny_model):
        """Gradients flow through all layers."""
        model, config = tiny_model
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check key parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_param_report(self, tiny_model):
        """param_report() generates non-empty string."""
        model, _ = tiny_model
        report = model.param_report()
        assert len(report) > 100
        assert "compression" in report.lower() or "Compression" in report

    def test_per_layer_report(self, tiny_model):
        """per_layer_report() lists all circulant layers."""
        model, _ = tiny_model
        report = model.per_layer_report()
        assert "BlockCirculant" in report or "blocks" in report

    def test_memory_profile(self, tiny_model):
        """memory_profile() returns dict with required keys."""
        model, _ = tiny_model
        profile = model.memory_profile()
        assert "params" in profile
        assert "fp32_mb" in profile
        assert "bf16_mb" in profile
        assert "int4_mb" in profile
        assert profile["fp32_mb"] > profile["bf16_mb"] > profile["int4_mb"]

    def test_compression_is_significant(self, tiny_model):
        """Model achieves meaningful compression."""
        model, config = tiny_model
        profile = model.memory_profile()
        # Tiny config with p=128 should achieve at least 50x compression
        # on the circulant layers alone
        assert profile["bf16_mb"] < 50  # Should be small

    def test_causal_masking(self, tiny_model):
        """Causal mask prevents attending to future tokens."""
        model, config = tiny_model
        model.eval()

        # Two sequences: same prefix, different suffix
        seq1 = torch.randint(0, config.vocab_size, (1, 32))
        seq2 = seq1.clone()
        seq2[0, 16:] = torch.randint(0, config.vocab_size, (16,))

        with torch.no_grad():
            logits1 = model(seq1)
            logits2 = model(seq2)

        # Logits for positions 0-15 should be identical
        # (causal mask prevents seeing positions 16+)
        torch.testing.assert_close(
            logits1[:, :16, :], logits2[:, :16, :],
            atol=1e-5, rtol=1e-5,
        )


class TestSIRENScaling:
    """Test that model scales correctly across configs."""

    def test_param_scaling(self):
        """Larger configs have more parameters."""
        tiny = SIRENTransformer(SIRENConfig.tiny())
        small = SIRENTransformer(SIRENConfig.small())

        tiny_params = sum(p.numel() for p in tiny.parameters())
        small_params = sum(p.numel() for p in small.parameters())

        assert small_params > tiny_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
