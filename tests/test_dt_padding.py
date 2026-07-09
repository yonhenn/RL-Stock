"""Regression tests for DT padding mask correctness.

Verifies that the src_key_padding_mask is properly applied so that
changing values in the padded prefix does NOT affect the model's
predictions for active tokens.
"""

from __future__ import annotations

import torch
import pytest

from rl_stock.dt_model import DecisionTransformer


@pytest.fixture
def small_dt() -> DecisionTransformer:
    """Small DT for fast deterministic tests."""
    torch.manual_seed(42)
    model = DecisionTransformer(
        state_dim=10,
        action_dim=1,
        d_model=32,
        n_heads=2,
        n_layers=2,
        context_length=8,
        dropout=0.0,  # deterministic
    )
    model.eval()
    return model


@pytest.mark.unit
class TestPaddingMaskInvariance:
    """Active token outputs must be invariant to padded prefix values."""

    def test_padded_prefix_does_not_affect_active_output(self, small_dt):
        """Core regression test: change padded prefix states/actions/rtgs,
        active outputs should remain identical."""
        B, K, S = 2, 8, 10
        torch.manual_seed(0)

        # Base inputs
        states = torch.randn(B, K, S)
        actions = torch.randn(B, K, 1)
        rtgs = torch.randn(B, K, 1)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)

        # Mask: first 5 positions padded, last 3 active
        mask = torch.zeros(B, K)
        mask[:, -3:] = 1.0

        with torch.no_grad():
            out_base = small_dt(states, actions, rtgs, timesteps, attention_mask=mask)

        # Perturbed inputs: completely different values in padded prefix
        states_pert = states.clone()
        actions_pert = actions.clone()
        rtgs_pert = rtgs.clone()
        states_pert[:, :5] = torch.randn(B, 5, S) * 100  # wildly different
        actions_pert[:, :5] = torch.randn(B, 5, 1) * 100
        rtgs_pert[:, :5] = torch.randn(B, 5, 1) * 100

        with torch.no_grad():
            out_pert = small_dt(states_pert, actions_pert, rtgs_pert, timesteps, attention_mask=mask)

        # Active positions (last 3) must be identical
        active_base = out_base[:, -3:, :]
        active_pert = out_pert[:, -3:, :]

        torch.testing.assert_close(
            active_base, active_pert,
            atol=1e-5, rtol=1e-5,
            msg="Padding mask leak: active outputs changed when padded prefix was modified!",
        )

    def test_no_nan_with_heavy_padding(self, small_dt):
        """Ensure no NaN even when most of the sequence is padded."""
        B, K, S = 1, 8, 10
        states = torch.randn(B, K, S)
        actions = torch.randn(B, K, 1)
        rtgs = torch.randn(B, K, 1)
        timesteps = torch.arange(K).unsqueeze(0)

        # Only last 1 position is active — extreme padding
        mask = torch.zeros(B, K)
        mask[:, -1] = 1.0

        with torch.no_grad():
            out = small_dt(states, actions, rtgs, timesteps, attention_mask=mask)

        assert torch.isfinite(out).all(), f"NaN/Inf detected with heavy padding: {out}"

    def test_no_mask_still_works(self, small_dt):
        """Without attention_mask, model should still produce valid outputs."""
        B, K, S = 2, 8, 10
        states = torch.randn(B, K, S)
        actions = torch.randn(B, K, 1)
        rtgs = torch.randn(B, K, 1)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = small_dt(states, actions, rtgs, timesteps, attention_mask=None)

        assert out.shape == (B, K, 1)
        assert torch.isfinite(out).all()

    def test_full_active_mask_matches_no_mask(self, small_dt):
        """An all-ones mask should produce identical output to no mask."""
        B, K, S = 2, 8, 10
        torch.manual_seed(7)
        states = torch.randn(B, K, S)
        actions = torch.randn(B, K, 1)
        rtgs = torch.randn(B, K, 1)
        timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)

        all_active_mask = torch.ones(B, K)

        with torch.no_grad():
            out_no_mask = small_dt(states, actions, rtgs, timesteps, attention_mask=None)
            out_all_active = small_dt(states, actions, rtgs, timesteps, attention_mask=all_active_mask)

        torch.testing.assert_close(
            out_no_mask, out_all_active,
            atol=1e-5, rtol=1e-5,
            msg="All-active mask should behave identically to no mask.",
        )

    def test_different_padding_lengths_per_batch(self, small_dt):
        """Batch items with different padding lengths should each be invariant."""
        K, S = 8, 10
        torch.manual_seed(3)

        states = torch.randn(2, K, S)
        actions = torch.randn(2, K, 1)
        rtgs = torch.randn(2, K, 1)
        timesteps = torch.arange(K).unsqueeze(0).expand(2, -1)

        # Batch 0: 6 padded, 2 active. Batch 1: 3 padded, 5 active.
        mask = torch.tensor([
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ], dtype=torch.float32)

        with torch.no_grad():
            out_base = small_dt(states, actions, rtgs, timesteps, attention_mask=mask)

        # Perturb padded regions
        states_pert = states.clone()
        states_pert[0, :6] = torch.randn(6, S) * 50   # batch 0 pad region
        states_pert[1, :3] = torch.randn(3, S) * 50   # batch 1 pad region

        with torch.no_grad():
            out_pert = small_dt(states_pert, actions, rtgs, timesteps, attention_mask=mask)

        # Batch 0: last 2 active
        torch.testing.assert_close(
            out_base[0, -2:], out_pert[0, -2:], atol=1e-5, rtol=1e-5,
        )
        # Batch 1: last 5 active
        torch.testing.assert_close(
            out_base[1, -5:], out_pert[1, -5:], atol=1e-5, rtol=1e-5,
        )
