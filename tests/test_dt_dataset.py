"""Tests for the TrajectoryDataset loading and preprocessing logic."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_stock.dt_dataset import TrajectoryDataset


@pytest.fixture
def dummy_dataset_pkl() -> Path:
    """Generate a dummy trajectories dataset pickle file."""
    trajectories = []
    
    # 3 episodes
    for ep in range(3):
        traj = []
        n_steps = 15 + ep * 5  # lengths: 15, 20, 25
        for t in range(n_steps):
            traj.append({
                "state": np.random.randn(10).astype(np.float32) + float(ep),
                "raw_action": float(np.sin(t)),
                "executed_action": float(np.sin(t)),
                "net_return": float(t * 0.001 - 0.01),
                "rtg": float(1.0 - t * 0.05),
                "timestep": t,
                "stop_triggered": False,
                "dd_locked": False,
                "reward": 0.5,
                "policy_name": "Random",
                "episode_sharpe": 1.2,
            })
        trajectories.append(traj)

    dataset = {
        "trajectories": trajectories,
        "metadata": {
            "mean_sharpe": 1.2,
            "max_sharpe": 1.2,
            "min_sharpe": 1.2,
            "total_episodes": 3,
            "state_dim": 10,
            "action_dim": 1,
        }
    }

    # Write to a temp file
    temp_dir = tempfile.mkdtemp()
    pkl_path = Path(temp_dir) / "dummy_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)
    
    return pkl_path


@pytest.mark.unit
class TestTrajectoryDataset:
    """Verify slicing subsegments, padding, masking, and normalization."""

    def test_dataset_loading(self, dummy_dataset_pkl):
        dataset = TrajectoryDataset(dummy_dataset_pkl, context_length=10)
        # Total steps across episodes: 15 + 20 + 25 = 60
        # Since we sample K-context at every step t, len(dataset) should equal total timesteps
        assert len(dataset) == 60
        assert dataset.state_mean.shape == (10,)
        assert dataset.state_std.shape == (10,)

    def test_getitem_shapes(self, dummy_dataset_pkl):
        dataset = TrajectoryDataset(dummy_dataset_pkl, context_length=10)
        states, actions, rtgs, timesteps, mask = dataset[20]

        assert states.shape == (10, 10), f"Expected (10, 10), got {states.shape}"
        assert actions.shape == (10, 1), f"Expected (10, 1), got {actions.shape}"
        assert rtgs.shape == (10, 1)
        assert timesteps.shape == (10,)
        assert mask.shape == (10,)

    def test_left_padding_for_early_steps(self, dummy_dataset_pkl):
        """At step t=2, we have only 3 valid steps. Mask should show left-padding."""
        dataset = TrajectoryDataset(dummy_dataset_pkl, context_length=10)
        # Find index for first episode, step t=2
        # self.samples list is sorted by ep_idx then t.
        # So samples[2] is ep=0, t=2
        states, actions, rtgs, timesteps, mask = dataset[2]

        # First 7 elements should be padded (mask = 0)
        # Last 3 elements should be active (mask = 1)
        expected_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
        assert torch.equal(mask, expected_mask), f"Expected {expected_mask}, got {mask}"

        # Padded elements must be zeros
        assert torch.equal(states[:7], torch.zeros(7, 10))
        assert torch.equal(actions[:7], torch.zeros(7, 1))
        assert torch.equal(rtgs[:7], torch.zeros(7, 1))
        assert torch.equal(timesteps[:7], torch.zeros(7, dtype=torch.int64))

    def test_state_normalization(self, dummy_dataset_pkl):
        """States retrieved from __getitem__ should be normalized."""
        dataset = TrajectoryDataset(dummy_dataset_pkl, context_length=10)
        states, _, _, _, mask = dataset[30]
        
        # Get one valid (non-padded) state from states tensor
        active_indices = torch.where(mask == 1)[0]
        assert len(active_indices) > 0
        idx = active_indices[0].item()
        
        # Normalized state should not match raw state directly
        norm_state = states[idx].numpy()
        raw_state = dataset.trajectories[dataset.samples[30][0]][dataset.samples[30][1] - (9 - idx)]["state"]
        
        expected_norm = (raw_state - dataset.state_mean) / dataset.state_std
        assert norm_state == pytest.approx(expected_norm, abs=1e-5)
