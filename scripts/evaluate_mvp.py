#!/usr/bin/env python3
"""Evaluate the long-only MVP once after training and write final artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.bc_inference import evaluate_bc_model
from rl_stock.bc_model import BehaviorCloning
from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import evaluate_dt_model
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio, compute_turnover


class FixedPositionPolicy:
    def __init__(self, position: float):
        self.position = float(position)

    def predict(self, observation, deterministic: bool = True):
        del observation, deterministic
        return np.array([self.position], dtype=np.float32), None


class DirectPositionEnv(PositionMgmtEnv):
    """Baseline-only env where the requested exposure is the executed exposure."""

    def _volatility_scaled_target(self, raw_target: float) -> float:
        low = 0.0 if self.risk.long_only else -self.risk.max_position
        return float(np.clip(raw_target, low, self.risk.max_position))


def evaluate_policy(policy, env: PositionMgmtEnv, seed: int = 42) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    if policy is None:
        env.action_space.seed(seed)
    done = False
    info: dict[str, Any] = {}
    while not done:
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def metrics_from_info(name: str, info: dict[str, Any]) -> dict[str, Any]:
    returns = np.asarray(info["returns"], dtype=float)
    equity = np.asarray(info["equity_curve"], dtype=float)
    positions = np.asarray(info["position_history"], dtype=float)
    raw = np.asarray(info.get("raw_action_history", []), dtype=float)
    executed = np.asarray(
        info.get("executed_action_history", positions[1:]), dtype=float
    )
    return {
        "strategy": name,
        "profit": float(info["total_profit"] - 1.0),
        "sharpe": compute_sharpe_ratio(returns),
        "max_drawdown": compute_max_drawdown(equity),
        "avg_raw_action": float(np.mean(raw)) if raw.size else float("nan"),
        "avg_executed_position": (
            float(np.mean(executed)) if executed.size else float("nan")
        ),
        "turnover": compute_turnover(positions),
        "stop_loss_events": int(info.get("stop_loss_events", 0)),
        "equity_curve": equity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("nasdaq100.xlsx"))
    parser.add_argument("--dataset", type=Path, default=Path("outputs/long_only_v2_mvp/dataset_v2.pkl"))
    parser.add_argument("--dt-checkpoint", type=Path, default=Path("outputs/long_only_v2_mvp/dt/dt_model.pt"))
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("outputs/long_only_v2_mvp/bc/bc_model.pt"))
    parser.add_argument("--sac-model", type=Path, default=Path("outputs/long_only/sac.zip"))
    parser.add_argument("--ppo-model", type=Path, default=Path("outputs/long_only/ppo.zip"))
    parser.add_argument("--v1-comparison", type=Path, default=Path("outputs/long_only/comparison_dt_bc.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/long_only_v2_mvp"))
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_market_data(args.data_path)
    _, test_bound = split_frame_bounds(
        df, window_size=30, test_start_date=args.test_start_date
    )
    common_risk = dict(
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=True,
    )
    env_window = PositionMgmtEnv(
        df, 30, test_bound, RiskConfig(flat_obs=False, **common_risk)
    )
    env_flat = PositionMgmtEnv(
        df, 30, test_bound, RiskConfig(flat_obs=True, **common_risk)
    )
    # Fixed-exposure benchmarks must not be silently rewritten by ATR scaling
    # or hard risk rails; otherwise a "100%" command is not Buy & Hold.
    env_direct = DirectPositionEnv(
        df,
        30,
        test_bound,
        RiskConfig(
            flat_obs=False,
            stop_loss_pct=0.0,
            max_drawdown_pct=0.0,
            use_trend_features=True,
            long_only=True,
        ),
    )

    rows: list[dict[str, Any]] = []
    rows.append(metrics_from_info("Random (seed 42)", evaluate_policy(None, env_window, args.seed)))
    for label, exposure in (("Fixed 25% (direct)", 0.25), ("Fixed 50% (direct)", 0.5), ("Fixed 75% (direct)", 0.75), ("Passive Buy & Hold", 1.0)):
        rows.append(
            metrics_from_info(
                label,
                evaluate_policy(FixedPositionPolicy(exposure), env_direct, args.seed),
            )
        )

    sac = SAC.load(args.sac_model, env=env_window)
    ppo = PPO.load(args.ppo_model, env=env_window)
    rows.append(metrics_from_info("SAC V1", evaluate_policy(sac, env_window, args.seed)))
    rows.append(metrics_from_info("PPO V1", evaluate_policy(ppo, env_window, args.seed)))

    bc_checkpoint = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
    bc_model = BehaviorCloning(
        state_dim=bc_checkpoint["state_dim"],
        action_dim=1,
        context_length=bc_checkpoint["context_length"],
        max_episode_len=bc_checkpoint["max_episode_len"],
        action_activation=bc_checkpoint.get("action_activation", "tanh"),
    )
    bc_model.load_state_dict(bc_checkpoint["state_dict"])
    _, _, bc_infos = evaluate_bc_model(
        bc_model,
        env_flat,
        bc_checkpoint["state_mean"],
        bc_checkpoint["state_std"],
        num_episodes=1,
        seed=args.seed,
        device="cpu",
    )
    rows.append(metrics_from_info("BC V2", bc_infos[0]))

    dt_checkpoint = torch.load(args.dt_checkpoint, map_location="cpu", weights_only=False)
    dt_model = DecisionTransformer(
        state_dim=dt_checkpoint["state_dim"],
        action_dim=1,
        context_length=dt_checkpoint["context_length"],
        max_episode_len=dt_checkpoint["max_episode_len"],
        action_activation=dt_checkpoint.get("action_activation", "tanh"),
    )
    dt_model.load_state_dict(dt_checkpoint["state_dict"])

    with args.dataset.open("rb") as handle:
        dataset = pickle.load(handle)
    starting_rtgs = np.asarray(
        [trajectory[0]["rtg"] for trajectory in dataset["trajectories"]],
        dtype=float,
    )
    target_map = {
        percentile: float(np.percentile(starting_rtgs, percentile))
        for percentile in (50, 75, 90)
    }
    for percentile, target in target_map.items():
        _, _, dt_infos = evaluate_dt_model(
            dt_model,
            env_flat,
            target,
            dt_checkpoint["state_mean"],
            dt_checkpoint["state_std"],
            dt_checkpoint["rtg_scale"],
            num_episodes=1,
            seed=args.seed,
            device="cpu",
        )
        rows.append(metrics_from_info(f"DT V2 RTG P{percentile}", dt_infos[0]))

    historical: dict[str, Any] = {}
    if args.v1_comparison.exists():
        with args.v1_comparison.open("rb") as handle:
            historical_raw = pickle.load(handle)
        for key, label in (("bc", "BC V1 historical"), ("dt08", "DT V1 historical RTG 2.4")):
            item = historical_raw.get(key)
            if item:
                historical[label] = {
                    "profit": float(item["profit"]),
                    "sharpe": float(item["sharpe"]),
                    "max_drawdown": float(item["max_dd"]),
                    "avg_raw_action": float(item.get("avg_pos", float("nan"))),
                    "note": "Exploratory historical result; checkpoint/RTG used test selection.",
                }

    csv_fields = [
        "strategy",
        "profit",
        "sharpe",
        "max_drawdown",
        "avg_raw_action",
        "avg_executed_position",
        "turnover",
        "stop_loss_events",
    ]
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in csv_fields})

    fig, axis = plt.subplots(figsize=(14, 8))
    for row in rows:
        if row["strategy"] in {
            "Passive Buy & Hold",
            "SAC V1",
            "BC V2",
            "DT V2 RTG P50",
            "DT V2 RTG P75",
            "DT V2 RTG P90",
        }:
            equity = row["equity_curve"]
            axis.plot((equity / equity[0] - 1.0) * 100.0, label=row["strategy"])
    axis.set_title("Long-Only MVP: Test Equity Curves (2024-2026)")
    axis.set_xlabel("Trading step")
    axis.set_ylabel("Cumulative return (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "equity_curves.png", dpi=160)
    plt.close(fig)

    run_config = {
        "seed": args.seed,
        "test_start_date": args.test_start_date,
        "checkpoint_selection": "final_epoch",
        "dt_targets": {f"P{k}": v for k, v in target_map.items()},
        "dynamic_rtg": True,
        "long_only": True,
        "historical_v1": historical,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for row in rows:
        print(
            f"{row['strategy']:22s} profit={row['profit'] * 100:7.2f}% "
            f"Sharpe={row['sharpe']:6.3f} avg_exec={row['avg_executed_position']:.3f}"
        )
    print(f"Saved final artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
