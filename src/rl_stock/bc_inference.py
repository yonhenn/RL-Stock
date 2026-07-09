"""Inference wrapper and evaluation helper for Behavior Cloning (BC) baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl_stock.envs_v2 import PositionMgmtEnv
from rl_stock.risk import compute_sharpe_ratio


def evaluate_bc_model(
    model: torch.nn.Module,
    env: PositionMgmtEnv,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    num_episodes: int = 20,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    """Evaluate Behavior Cloning model in the environment with rolling context.

    Returns rewards, profits, and infos.
    """
    model.eval()
    state_mean = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
    state_std = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)

    rewards: list[float] = []
    profits: list[float] = []
    infos: list[dict[str, Any]] = []

    context_len = model.context_length
    state_dim = model.state_dim

    for ep in range(num_episodes):
        episode_seed = seed + ep
        obs, _ = env.reset(seed=episode_seed)

        # Context buffers (no RTG buffer)
        states_buf = torch.zeros((1, context_len, state_dim), device=device, dtype=torch.float32)
        actions_buf = torch.zeros((1, context_len, 1), device=device, dtype=torch.float32)
        timesteps_buf = torch.zeros((1, context_len), device=device, dtype=torch.int64)

        # Initialize first step
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
        states_buf[0, -1] = (obs_t - state_mean) / state_std

        total_reward = 0.0
        done = False
        step_idx = 0
        info: dict[str, Any] = {}

        while not done:
            # Predict action from current context
            with torch.no_grad():
                attention_mask = torch.zeros((1, context_len), device=device, dtype=torch.float32)
                active_len = min(step_idx + 1, context_len)
                attention_mask[0, -active_len:] = 1.0

                pred_actions = model(
                    states=states_buf,
                    actions=actions_buf,
                    timesteps=timesteps_buf,
                    attention_mask=attention_mask,
                )
                action_val = pred_actions[0, -1, 0].item()

            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(np.array([action_val], dtype=np.float32))
            total_reward += reward
            done = terminated or truncated

            # Record action in buffer for the step that was just taken
            actions_buf[0, -1, 0] = action_val

            step_idx += 1

            if not done:
                # Roll buffers left
                states_buf = torch.roll(states_buf, shifts=-1, dims=1)
                actions_buf = torch.roll(actions_buf, shifts=-1, dims=1)
                timesteps_buf = torch.roll(timesteps_buf, shifts=-1, dims=1)

                # Set new step inputs on the right
                obs_t = torch.from_numpy(obs_next).to(device=device, dtype=torch.float32)
                states_buf[0, -1] = (obs_t - state_mean) / state_std
                actions_buf[0, -1, 0] = 0.0  # place-holder for next prediction
                timesteps_buf[0, -1] = step_idx

        rewards.append(total_reward)
        profits.append(info.get("total_profit", 1.0))
        infos.append(info)

    return rewards, profits, infos


class BCAgent:
    """Wrapper that makes Behavior Cloning look like a stable-baselines3 model.

    Maintains rolling context during evaluation/testing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model
        self.state_mean = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
        self.state_std = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)
        self.device = device

        self.context_len = model.context_length
        self.state_dim = model.state_dim
        self.reset()

    def reset(self):
        """Reset the context buffers."""
        self.states_buf = torch.zeros((1, self.context_len, self.state_dim), device=self.device, dtype=torch.float32)
        self.actions_buf = torch.zeros((1, self.context_len, 1), device=self.device, dtype=torch.float32)
        self.timesteps_buf = torch.zeros((1, self.context_len), device=self.device, dtype=torch.int64)
        self.step_idx = 0

    def predict(
        self,
        obs: np.ndarray,
        state: Any | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any | None]:
        """Predict the next action based on environment observation.

        If episode_start is True or step_idx is 0, we reset buffers.
        """
        if (episode_start is not None and bool(episode_start[0])) or self.step_idx == 0:
            self.reset()

        obs_t = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
        if obs_t.ndim == 2:
            obs_t = obs_t[0]

        # Put new state in buffer
        self.states_buf[0, -1] = (obs_t - self.state_mean) / self.state_std

        # Predict
        self.model.eval()
        with torch.no_grad():
            attention_mask = torch.zeros((1, self.context_len), device=self.device, dtype=torch.float32)
            active_len = min(self.step_idx + 1, self.context_len)
            attention_mask[0, -active_len:] = 1.0

            pred_actions = self.model(
                states=self.states_buf,
                actions=self.actions_buf,
                timesteps=self.timesteps_buf,
                attention_mask=attention_mask,
            )
            action_val = pred_actions[0, -1, 0].item()

        # Update action buffer for current prediction
        self.actions_buf[0, -1, 0] = action_val

        # Roll buffers for next step
        self.states_buf = torch.roll(self.states_buf, shifts=-1, dims=1)
        self.actions_buf = torch.roll(self.actions_buf, shifts=-1, dims=1)
        self.timesteps_buf = torch.roll(self.timesteps_buf, shifts=-1, dims=1)

        self.step_idx += 1
        self.timesteps_buf[0, -1] = self.step_idx
        self.actions_buf[0, -1, 0] = 0.0  # placeholder for next step

        return np.array([action_val], dtype=np.float32), None
