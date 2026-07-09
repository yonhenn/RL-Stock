"""Data collection pipeline for offline RL (Decision Transformer)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from rl_stock.envs_v2 import PositionMgmtEnv
from rl_stock.risk import compute_sharpe_ratio


def compute_rtg(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute Return-to-Go backwards for a given trajectory."""
    net_returns = np.array([step["net_return"] for step in trajectory], dtype=float)
    rtgs = np.cumsum(net_returns[::-1])[::-1]
    for i, step in enumerate(trajectory):
        step["rtg"] = float(rtgs[i])
    return trajectory


def collect_trajectories(
    env: PositionMgmtEnv,
    policy: Any | None,
    n_episodes: int = 100,
    seed: int = 42,
    deterministic: bool = False,
) -> list[list[dict[str, Any]]]:
    """Collect trajectories from a given policy in the flat_obs environment.

    Each step dict contains state, actions, net_return, flags.
    RTG and episode quality (Sharpe) are computed post-episode.
    """
    trajectories: list[list[dict[str, Any]]] = []

    for episode in range(n_episodes):
        episode_seed = seed + episode
        obs, reset_info = env.reset(seed=episode_seed)
        if policy is None:
            env.action_space.seed(episode_seed)

        current_flat_obs = reset_info["flat_obs"]
        trajectory: list[dict[str, Any]] = []
        done = False
        step_idx = 0

        while not done:
            if policy is not None:
                # Handle stable-baselines3 model predict output
                predict_result = policy.predict(obs, deterministic=deterministic)
                if isinstance(predict_result, tuple):
                    action = predict_result[0]
                else:
                    action = predict_result
            else:
                action = env.action_space.sample()

            obs_next, reward, terminated, truncated, info = env.step(action)

            # Ensure action is float or list of floats
            raw_action_val = float(info["raw_action"])
            executed_action_val = float(info["executed_action"])
            net_return_val = float(info["net_return"])

            trajectory.append({
                "state": current_flat_obs,  # flat 1D state
                "raw_action": raw_action_val,
                "executed_action": executed_action_val,
                "net_return": net_return_val,
                "stop_triggered": bool(info.get("stop_triggered", False)),
                "dd_locked": bool(info.get("dd_locked", False)),
                "reward": float(reward),
                "timestep": step_idx,
            })

            current_flat_obs = info["flat_obs"]
            obs = obs_next
            done = terminated or truncated
            step_idx += 1

        # Post-compute RTG (cumulative sum of net returns backwards)
        trajectory = compute_rtg(trajectory)

        # Post-compute Episode Sharpe
        net_returns = np.array([step["net_return"] for step in trajectory], dtype=float)
        episode_sharpe = compute_sharpe_ratio(net_returns)

        # Enforce metrics to every step in the trajectory
        for step in trajectory:
            step["episode_sharpe"] = episode_sharpe

        trajectories.append(trajectory)

    return trajectories


def build_dt_dataset(
    env: PositionMgmtEnv,
    policies: dict[str, Any | None],
    n_episodes_per_policy: int = 100,
    seed: int = 42,
    output_path: Path | str = "outputs/dt_dataset.pkl",
) -> dict[str, Any]:
    """Collect trajectories across multiple policies and save to a pickle file.

    Saves a dict containing raw trajectories and aggregate stats.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_trajectories: list[list[dict[str, Any]]] = []

    for name, policy in policies.items():
        print(f"Collecting {n_episodes_per_policy} episodes using policy: {name}...")
        trajs = collect_trajectories(
            env=env,
            policy=policy,
            n_episodes=n_episodes_per_policy,
            seed=seed,
            deterministic=False if name == "Random" else True,
        )
        # Tag each step with policy name
        for traj in trajs:
            for step in traj:
                step["policy_name"] = name
        all_trajectories.extend(trajs)

    # Compute dataset metadata / summary stats
    all_sharpes = [traj[0]["episode_sharpe"] for traj in all_trajectories if len(traj) > 0]
    mean_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0
    max_sharpe = float(np.max(all_sharpes)) if all_sharpes else 0.0
    min_sharpe = float(np.min(all_sharpes)) if all_sharpes else 0.0

    print(f"Dataset collection complete. Total trajectories: {len(all_trajectories)}")
    print(f"Sharpe Ratio stats - Mean: {mean_sharpe:.4f}, Max: {max_sharpe:.4f}, Min: {min_sharpe:.4f}")

    dataset = {
        "trajectories": all_trajectories,
        "metadata": {
            "mean_sharpe": mean_sharpe,
            "max_sharpe": max_sharpe,
            "min_sharpe": min_sharpe,
            "total_episodes": len(all_trajectories),
            "state_dim": all_trajectories[0][0]["state"].shape[0] if len(all_trajectories) > 0 else 0,
            "action_dim": 1,
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset to {output_path}")
    return dataset
