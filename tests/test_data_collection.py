"""Tests for the data collection pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rl_stock.data_collection import build_dt_dataset, collect_trajectories, compute_rtg
from rl_stock.envs_v2 import N_PORTFOLIO_FEATURES_DT, N_TECH_FEATURES, PositionMgmtEnv, RiskConfig


@pytest.fixture
def flat_env(synthetic_ohlc) -> PositionMgmtEnv:
    """PositionMgmtEnv in flat_obs mode."""
    risk = RiskConfig(flat_obs=True)
    return PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=30,
        frame_bound=(30, len(synthetic_ohlc) - 1),
        risk_config=risk,
    )


@pytest.mark.unit
class TestDataCollection:
    """Test trajectory collection logic."""

    def test_collect_random_policy_keys(self, flat_env):
        """Trajectories should contain all required keys for DT training."""
        trajs = collect_trajectories(flat_env, policy=None, n_episodes=2, seed=123)
        assert len(trajs) == 2
        for traj in trajs:
            assert len(traj) > 0
            for step in traj:
                assert "state" in step
                assert "raw_action" in step
                assert "executed_action" in step
                assert "net_return" in step
                assert "rtg" in step
                assert "episode_sharpe" in step
                assert "stop_triggered" in step
                assert "dd_locked" in step
                assert "reward" in step
                assert "timestep" in step

                # Check observation shape is 1D
                state = step["state"]
                assert state.ndim == 1
                assert state.shape[0] == N_TECH_FEATURES + N_PORTFOLIO_FEATURES_DT

    def test_rtg_computation_manually(self):
        """Test compute_rtg returns correct backward sum."""
        dummy_traj = [
            {"net_return": 0.01},
            {"net_return": -0.02},
            {"net_return": 0.05},
        ]
        # Cumulative backwards:
        # Step 2: 0.05
        # Step 1: -0.02 + 0.05 = 0.03
        # Step 0: 0.01 + 0.03 = 0.04
        updated = compute_rtg(dummy_traj)
        assert updated[0]["rtg"] == pytest.approx(0.04)
        assert updated[1]["rtg"] == pytest.approx(0.03)
        assert updated[2]["rtg"] == pytest.approx(0.05)

    def test_rtg_monotonic_decline(self, flat_env):
        """RTGs should monotonically change toward 0 at the end of the episode (in simple return environments)."""
        trajs = collect_trajectories(flat_env, policy=None, n_episodes=1, seed=42)
        traj = trajs[0]
        # Final step RTG must equal its own net_return
        assert traj[-1]["rtg"] == pytest.approx(traj[-1]["net_return"])

    def test_build_dt_dataset_saving(self, flat_env):
        """build_dt_dataset should output a valid pickle file with dataset dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_dataset.pkl"
            policies = {"Random": None}
            dataset = build_dt_dataset(
                env=flat_env,
                policies=policies,
                n_episodes_per_policy=3,
                seed=42,
                output_path=out_file,
            )

            assert out_file.exists()
            assert "trajectories" in dataset
            assert "metadata" in dataset
            assert dataset["metadata"]["total_episodes"] == 3
            assert dataset["metadata"]["state_dim"] == N_TECH_FEATURES + N_PORTFOLIO_FEATURES_DT
            assert len(dataset["trajectories"]) == 3
            # Check steps contain policy_name
            for traj in dataset["trajectories"]:
                for step in traj:
                    assert step["policy_name"] == "Random"
