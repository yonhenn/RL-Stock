"""Tests for Decision Transformer model architecture and inference agent."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from rl_stock.dt_model import DecisionTransformer
from rl_stock.dt_inference import DTAgent, evaluate_dt_model
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig


@pytest.fixture
def dummy_dt() -> DecisionTransformer:
    """DecisionTransformer model instance."""
    return DecisionTransformer(
        state_dim=25,
        action_dim=1,
        d_model=64,
        n_heads=2,
        n_layers=1,
        context_length=10,
    )


@pytest.mark.unit
class TestDecisionTransformerModel:
    """Verify PyTorch DT forward pass, inputs, and predicted action range."""

    def test_forward_shape(self, dummy_dt):
        batch = 4
        K = 10
        states = torch.randn(batch, K, 25)
        actions = torch.randn(batch, K, 1)
        rtgs = torch.randn(batch, K, 1)
        timesteps = torch.randint(0, 100, (batch, K))

        pred = dummy_dt(states, actions, rtgs, timesteps)
        assert pred.shape == (batch, K, 1), f"Expected (4, 10, 1), got {pred.shape}"

    def test_attention_mask_masking(self, dummy_dt):
        batch = 2
        K = 10
        states = torch.randn(batch, K, 25)
        actions = torch.randn(batch, K, 1)
        rtgs = torch.randn(batch, K, 1)
        timesteps = torch.randint(0, 100, (batch, K))
        # Mask leaves only 3 items active
        mask = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ], dtype=torch.float32)

        pred = dummy_dt(states, actions, rtgs, timesteps, attention_mask=mask)
        assert pred.shape == (batch, K, 1)
        # Ensure outputs are finite and valid
        assert torch.isfinite(pred).all()


@pytest.mark.unit
class TestDTAgentWrapper:
    """Verify SB3-compatible DTAgent behavior."""

    def test_agent_initialization(self, dummy_dt):
        mean = np.zeros(25, dtype=np.float32)
        std = np.ones(25, dtype=np.float32)
        agent = DTAgent(dummy_dt, target_rtg=0.5, state_mean=mean, state_std=std)
        
        assert agent.step_idx == 0
        assert agent.states_buf.shape == (1, 10, 25)
        assert agent.actions_buf.shape == (1, 10, 1)
        assert agent.rtgs_buf.shape == (1, 10, 1)

    def test_agent_predict_returns_sb3_format(self, dummy_dt):
        mean = np.zeros(25, dtype=np.float32)
        std = np.ones(25, dtype=np.float32)
        agent = DTAgent(dummy_dt, target_rtg=0.5, state_mean=mean, state_std=std)

        obs = np.random.randn(25).astype(np.float32)
        action, state_out = agent.predict(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert state_out is None
        assert agent.step_idx == 1

    def test_agent_predict_rolls_context_correctly(self, dummy_dt):
        mean = np.zeros(25, dtype=np.float32)
        std = np.ones(25, dtype=np.float32)
        agent = DTAgent(dummy_dt, target_rtg=0.5, state_mean=mean, state_std=std)

        for _ in range(15):
            obs = np.random.randn(25).astype(np.float32)
            agent.predict(obs)

        # Timestep buffer should roll up to 15
        assert agent.step_idx == 15
        # The right-most timestep must equal step_idx
        assert agent.timesteps_buf[0, -1].item() == 15


@pytest.mark.unit
class TestDTOldEvaluationCompatibility:
    """Verify evaluate_dt_model runs without crashes on synthetic data."""

    def test_evaluate_dt_runs(self, synthetic_ohlc, dummy_dt):
        risk = RiskConfig(flat_obs=True)
        env = PositionMgmtEnv(
            df=synthetic_ohlc,
            window_size=30,
            frame_bound=(30, 80),
            risk_config=risk,
        )

        mean = np.zeros(19, dtype=np.float32)  # shape depends on env config
        std = np.ones(19, dtype=np.float32)

        # Get actual env dimension by inspecting space
        real_state_dim = env.observation_space.shape[0]
        model = DecisionTransformer(
            state_dim=real_state_dim,
            action_dim=1,
            d_model=32,
            n_heads=2,
            n_layers=1,
            context_length=5,
        )

        rewards, profits, infos = evaluate_dt_model(
            model=model,
            env=env,
            target_rtg=0.1,
            state_mean=np.zeros(real_state_dim, dtype=np.float32),
            state_std=np.ones(real_state_dim, dtype=np.float32),
            num_episodes=2,
        )

        assert len(rewards) == 2
        assert len(profits) == 2
        assert len(infos) == 2
        for info in infos:
            assert "returns" in info
            assert "position_history" in info
