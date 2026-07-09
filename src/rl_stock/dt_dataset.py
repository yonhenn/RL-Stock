"""PyTorch Dataset and sequence processing for Decision Transformer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset for sampling fixed-length (K) subsequences from offline trajectories.

    Applies state normalization (z-score) and RTG scaling.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        context_length: int = 20,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        rtg_scale: float = 1.0,
        step_stride: int = 1,
    ):
        self.context_length = context_length
        self.rtg_scale = rtg_scale

        dataset_path = Path(dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        self.trajectories = data["trajectories"]
        self.metadata = data["metadata"]

        # Filter out empty trajectories if any
        self.trajectories = [t for t in self.trajectories if len(t) > 0]

        # Compute state normalization stats if not provided
        all_states = []
        for traj in self.trajectories:
            for step in traj:
                all_states.append(step["state"])
        all_states = np.array(all_states, dtype=np.float32)

        if state_mean is None:
            self.state_mean = np.mean(all_states, axis=0)
        else:
            self.state_mean = np.asarray(state_mean, dtype=np.float32)

        if state_std is None:
            self.state_std = np.std(all_states, axis=0)
            # Avoid division by zero
            self.state_std[self.state_std < 1e-6] = 1.0
        else:
            self.state_std = np.asarray(state_std, dtype=np.float32)

        # Build list of valid sampling points (episode_idx, step_idx)
        # Each index corresponds to a subsegment of length context_length
        self.samples = []
        for ep_idx, traj in enumerate(self.trajectories):
            n_steps = len(traj)
            # We stride to reduce redundancy and speed up training
            for t in range(0, n_steps, step_stride):
                self.samples.append((ep_idx, t))
            if (n_steps - 1) % step_stride != 0:
                self.samples.append((ep_idx, n_steps - 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        traj = self.trajectories[ep_idx]

        # Determine slice indices. We look BACKWARDS from t to t - K + 1.
        # This matches the online generation paradigm where we predict a_t given a history of length K.
        start = max(0, t - self.context_length + 1)
        end = t + 1
        slice_steps = traj[start:end]
        actual_len = len(slice_steps)

        # Extract values
        states_list = [step["state"] for step in slice_steps]
        # In our specs, we train on raw_action
        actions_list = [[step["raw_action"]] for step in slice_steps]
        rtgs_list = [[step["rtg"] / self.rtg_scale] for step in slice_steps]
        timesteps_list = [step["timestep"] for step in slice_steps]

        # Convert to numpy arrays
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)
        rtgs = np.array(rtgs_list, dtype=np.float32)
        timesteps = np.array(timesteps_list, dtype=np.int64)
        attention_mask = np.ones(actual_len, dtype=np.float32)

        # Normalize states
        states = (states - self.state_mean) / self.state_std

        # Left-padding with zeros if actual_len < context_length
        if actual_len < self.context_length:
            pad_len = self.context_length - actual_len
            state_dim = states.shape[1]
            action_dim = actions.shape[1]

            states = np.concatenate([np.zeros((pad_len, state_dim), dtype=np.float32), states], axis=0)
            actions = np.concatenate([np.zeros((pad_len, action_dim), dtype=np.float32), actions], axis=0)
            rtgs = np.concatenate([np.zeros((pad_len, 1), dtype=np.float32), rtgs], axis=0)
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(pad_len, dtype=np.float32), attention_mask], axis=0)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rtgs),
            torch.from_numpy(timesteps),
            torch.from_numpy(attention_mask),
        )
