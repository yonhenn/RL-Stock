#!/usr/bin/env python3
"""Build the diverse long-only Dataset V2 used by the one-day MVP.

The original dataset repeated deterministic SAC and PPO trajectories 100 times.
This collector instead samples stochastic expert actions, adds fixed-exposure
demonstrations, removes exact action-path duplicates, and writes a compact QA
report next to the pickle artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.data_collection import collect_trajectories
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig


class FixedPositionPolicy:
    """SB3-compatible policy that always requests one target exposure."""

    def __init__(self, position: float):
        self.position = float(position)

    def predict(self, observation, deterministic: bool = True):
        del observation, deterministic
        return np.array([self.position], dtype=np.float32), None


def action_hash(trajectory: list[dict[str, Any]]) -> str:
    actions = np.asarray(
        [step["raw_action"] for step in trajectory], dtype=np.float32
    )
    return hashlib.sha256(actions.tobytes()).hexdigest()


def tag_trajectory(
    trajectory: list[dict[str, Any]],
    *,
    trajectory_id: str,
    policy_name: str,
    collection_seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    digest = action_hash(trajectory)
    net_returns = np.asarray(
        [step["net_return"] for step in trajectory], dtype=float
    )
    raw_actions = np.asarray(
        [step["raw_action"] for step in trajectory], dtype=float
    )
    executed = np.asarray(
        [step["executed_action"] for step in trajectory], dtype=float
    )
    for step in trajectory:
        step["trajectory_id"] = trajectory_id
        step["policy_name"] = policy_name
        step["collection_seed"] = collection_seed
        step["deterministic"] = deterministic
        step["action_hash"] = digest
    return {
        "trajectory_id": trajectory_id,
        "policy_name": policy_name,
        "collection_seed": collection_seed,
        "deterministic": deterministic,
        "action_hash": digest,
        "episode_length": len(trajectory),
        "starting_rtg": float(trajectory[0]["rtg"]),
        "compound_return": float(np.prod(1.0 + net_returns) - 1.0),
        "episode_sharpe": float(trajectory[0]["episode_sharpe"]),
        "avg_raw_action": float(np.mean(raw_actions)),
        "avg_executed_position": float(np.mean(executed)),
    }


def collect_one(
    env: PositionMgmtEnv,
    policy: Any | None,
    *,
    seed: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    if policy is not None and hasattr(policy, "set_random_seed"):
        policy.set_random_seed(seed)
    return collect_trajectories(
        env=env,
        policy=policy,
        n_episodes=1,
        seed=seed,
        deterministic=deterministic,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("nasdaq100.xlsx"))
    parser.add_argument("--sac-model", type=Path, default=Path("outputs/long_only/sac.zip"))
    parser.add_argument("--ppo-model", type=Path, default=Path("outputs/long_only/ppo.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/long_only_v2_mvp/dataset_v2.pkl"))
    parser.add_argument("--qa-output", type=Path, default=Path("outputs/long_only_v2_mvp/dataset_qa.json"))
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--episodes-per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--long-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use [0,1] actions; --no-long-only uses [-1,1].",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_market_data(args.data_path)
    train_bound, _ = split_frame_bounds(
        df, window_size=30, test_start_date=args.test_start_date
    )
    risk = RiskConfig(
        flat_obs=False,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=args.long_only,
    )
    env = PositionMgmtEnv(
        df=df, window_size=30, frame_bound=train_bound, risk_config=risk
    )

    if not args.sac_model.exists() or not args.ppo_model.exists():
        raise FileNotFoundError("Both SAC and PPO checkpoints are required")
    sac = SAC.load(args.sac_model, env=env)
    ppo = PPO.load(args.ppo_model, env=env)
    expected_low = 0.0 if args.long_only else -1.0
    for name, model in (("SAC", sac), ("PPO", ppo)):
        if not np.allclose(model.action_space.low, expected_low) or not np.allclose(
            model.action_space.high, 1.0
        ):
            raise ValueError(
                f"{name} checkpoint action space {model.action_space} is incompatible "
                f"with expected [{expected_low}, 1.0]"
            )

    requests: list[tuple[str, Any | None, int, bool]] = []
    for offset in range(args.episodes_per_source):
        requests.append(("Random", None, args.seed + offset, False))
        requests.append(("SAC-Stochastic", sac, args.seed + 100 + offset, False))
        requests.append(("PPO-Stochastic", ppo, args.seed + 200 + offset, False))
    fixed_targets = (
        (0.0, 0.25, 0.5, 0.75, 1.0)
        if args.long_only
        else (-1.0, -0.5, 0.0, 0.5, 1.0)
    )
    for index, position in enumerate(fixed_targets):
        label = f"Fixed-{position:+.2f}"
        requests.append(
            (label, FixedPositionPolicy(position), args.seed + 300 + index, True)
        )

    trajectories: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}

    for request_index, (name, policy, seed, deterministic) in enumerate(requests):
        print(f"Collecting {name} seed={seed} deterministic={deterministic}")
        trajectory = collect_one(
            env, policy, seed=seed, deterministic=deterministic
        )
        digest = action_hash(trajectory)
        trajectory_id = f"v2-{request_index:03d}-{name}"
        if digest in seen_hashes:
            duplicate_records.append(
                {
                    "policy_name": name,
                    "duplicate_of": seen_hashes[digest],
                    "action_hash": digest,
                }
            )
            continue
        seen_hashes[digest] = trajectory_id
        summary = tag_trajectory(
            trajectory,
            trajectory_id=trajectory_id,
            policy_name=name,
            collection_seed=seed,
            deterministic=deterministic,
        )
        trajectories.append(trajectory)
        summaries.append(summary)

    if not trajectories:
        raise RuntimeError("Dataset collection produced no unique trajectories")

    all_sharpes = np.asarray([item["episode_sharpe"] for item in summaries])
    starts = np.asarray([item["starting_rtg"] for item in summaries])
    all_raw = np.asarray(
        [step["raw_action"] for traj in trajectories for step in traj], dtype=float
    )
    all_executed = np.asarray(
        [step["executed_action"] for traj in trajectories for step in traj],
        dtype=float,
    )
    metadata = {
        "mean_sharpe": float(np.mean(all_sharpes)),
        "max_sharpe": float(np.max(all_sharpes)),
        "min_sharpe": float(np.min(all_sharpes)),
        "total_episodes": len(trajectories),
        "requested_episodes": len(requests),
        "unique_episodes": len(trajectories),
        "duplicate_episodes_removed": len(duplicate_records),
        "state_dim": int(trajectories[0][0]["state"].shape[0]),
        "action_dim": 1,
        "long_only": args.long_only,
        "test_start_date": args.test_start_date,
    }
    dataset = {"trajectories": trajectories, "metadata": metadata}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dataset, handle)

    qa = {
        **metadata,
        "policy_distribution": dict(
            sorted(Counter(item["policy_name"] for item in summaries).items())
        ),
        "starting_rtg_percentiles": {
            str(percentile): float(np.percentile(starts, percentile))
            for percentile in (0, 25, 50, 75, 90, 100)
        },
        "raw_action": {
            "min": float(np.min(all_raw)),
            "mean": float(np.mean(all_raw)),
            "max": float(np.max(all_raw)),
        },
        "executed_position": {
            "min": float(np.min(all_executed)),
            "mean": float(np.mean(all_executed)),
            "max": float(np.max(all_executed)),
        },
        "executed_position_step_proportions": {
            "short": float(np.mean(all_executed < -1e-8)),
            "flat": float(np.mean(np.abs(all_executed) <= 1e-8)),
            "long": float(np.mean(all_executed > 1e-8)),
        },
        "trajectories": summaries,
        "duplicates_removed": duplicate_records,
    }
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.qa_output.write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"Saved {len(trajectories)} unique trajectories to {args.output}; "
        f"removed {len(duplicate_records)} duplicates"
    )
    print(f"Saved QA report to {args.qa_output}")


if __name__ == "__main__":
    main()
