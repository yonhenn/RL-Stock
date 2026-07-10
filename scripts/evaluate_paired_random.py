#!/usr/bin/env python3
"""Evaluate paired random policies and rank models against each action space."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio, compute_turnover


def paired_actions(seed: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Map identical uniform draws to [0,1] and [-1,1] action paths."""
    uniform = np.random.default_rng(seed).random(length).astype(np.float32)
    return uniform, 2.0 * uniform - 1.0


def empirical_percentile(
    random_values: np.ndarray, model_value: float, *, lower_is_better: bool = False
) -> float:
    """Finite-sample empirical percentile with a +1 correction."""
    random_values = np.asarray(random_values, dtype=float)
    if lower_is_better:
        count = np.count_nonzero(random_values >= model_value)
    else:
        count = np.count_nonzero(random_values <= model_value)
    # Laplace smoothing keeps a finite sample away from exact 0th/100th ranks.
    return float(100.0 * (1 + count) / (len(random_values) + 2))


def evaluate_actions(
    env: PositionMgmtEnv, actions: np.ndarray, *, seed: int
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    del observation
    done = False
    index = 0
    info: dict[str, Any] = {}
    while not done:
        _, _, terminated, truncated, info = env.step(
            np.array([actions[index]], dtype=np.float32)
        )
        done = terminated or truncated
        index += 1
    returns = np.asarray(info["returns"], dtype=float)
    equity = np.asarray(info["equity_curve"], dtype=float)
    positions = np.asarray(info["position_history"], dtype=float)
    return {
        "profit": float(info["total_profit"] - 1.0),
        "sharpe": compute_sharpe_ratio(returns),
        "max_drawdown": compute_max_drawdown(equity),
        "avg_raw_action": float(np.mean(info["raw_action_history"])),
        "avg_executed_position": float(
            np.mean(info["executed_action_history"])
        ),
        "turnover": compute_turnover(positions),
        "stop_loss_events": int(info["stop_loss_events"]),
    }


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for action_space in ("long_only", "symmetric"):
        selected = [row for row in rows if row["action_space"] == action_space]
        for metric in (
            "profit",
            "sharpe",
            "max_drawdown",
            "avg_raw_action",
            "avg_executed_position",
            "turnover",
            "stop_loss_events",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary.append(
                {
                    "action_space": action_space,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p5": float(np.percentile(values, 5)),
                    "p25": float(np.percentile(values, 25)),
                    "p50": float(np.percentile(values, 50)),
                    "p75": float(np.percentile(values, 75)),
                    "p95": float(np.percentile(values, 95)),
                }
            )
    return summary


def read_models(path: Path, action_space: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    models: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["strategy"].startswith(("SAC", "PPO", "BC", "DT")):
                continue
            models.append(
                {
                    "strategy": row["strategy"],
                    "action_space": action_space,
                    "profit": float(row["profit"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                }
            )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("nasdaq100.xlsx"))
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument(
        "--long-comparison",
        type=Path,
        default=Path("outputs/long_only_v2_mvp/comparison.csv"),
    )
    parser.add_argument(
        "--symmetric-comparison",
        type=Path,
        default=Path("outputs/symmetric_v2_mvp/comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/action_space_matched_audit"),
    )
    args = parser.parse_args()

    df = load_market_data(args.data_path)
    _, test_bound = split_frame_bounds(
        df, window_size=30, test_start_date=args.test_start_date
    )
    common = dict(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
    )
    env_long = PositionMgmtEnv(
        df, 30, test_bound, RiskConfig(long_only=True, **common)
    )
    env_symmetric = PositionMgmtEnv(
        df, 30, test_bound, RiskConfig(long_only=False, **common)
    )
    action_length = max(env_long._end_tick - env_long._start_tick + 1, 1)

    rows: list[dict[str, Any]] = []
    for offset in range(args.num_seeds):
        seed = args.start_seed + offset
        long_actions, symmetric_actions = paired_actions(seed, action_length)
        long_metrics = evaluate_actions(env_long, long_actions, seed=seed)
        symmetric_metrics = evaluate_actions(
            env_symmetric, symmetric_actions, seed=seed
        )
        rows.append({"seed": seed, "action_space": "long_only", **long_metrics})
        rows.append(
            {"seed": seed, "action_space": "symmetric", **symmetric_metrics}
        )
        if (offset + 1) % 10 == 0 or offset == 0:
            print(f"Completed paired random paths: {offset + 1}/{args.num_seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_fields = [
        "seed",
        "action_space",
        "profit",
        "sharpe",
        "max_drawdown",
        "avg_raw_action",
        "avg_executed_position",
        "turnover",
        "stop_loss_events",
    ]
    with (args.output_dir / "random_100_paths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=path_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    with (args.output_dir / "random_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    model_rows = read_models(args.long_comparison, "long_only") + read_models(
        args.symmetric_comparison, "symmetric"
    )
    ranked: list[dict[str, Any]] = []
    for model in model_rows:
        random_space_rows = [
            row for row in rows if row["action_space"] == model["action_space"]
        ]
        profits = np.asarray([row["profit"] for row in random_space_rows])
        sharpes = np.asarray([row["sharpe"] for row in random_space_rows])
        max_dds = np.asarray([row["max_drawdown"] for row in random_space_rows])
        ranked.append(
            {
                **model,
                "profit_percentile": empirical_percentile(
                    profits, model["profit"]
                ),
                "sharpe_percentile": empirical_percentile(
                    sharpes, model["sharpe"]
                ),
                "max_drawdown_percentile": empirical_percentile(
                    max_dds, model["max_drawdown"], lower_is_better=True
                ),
            }
        )
    if ranked:
        with (args.output_dir / "model_percentiles.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)

    fig, axis = plt.subplots(figsize=(10, 6))
    for action_space, color in (("long_only", "#1f77b4"), ("symmetric", "#7f7f7f")):
        profits = np.sort(
            [row["profit"] * 100.0 for row in rows if row["action_space"] == action_space]
        )
        ecdf = np.arange(1, len(profits) + 1) / len(profits)
        axis.step(profits, ecdf, where="post", label=action_space, color=color)
    axis.set_title("Paired Random Policy Profit ECDF (Seeds 42-141)")
    axis.set_xlabel("Profit (%)")
    axis.set_ylabel("Empirical cumulative probability")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "random_profit_ecdf.png", dpi=160)
    plt.close(fig)

    for item in summary:
        if item["metric"] in {"profit", "sharpe", "max_drawdown"}:
            print(
                f"{item['action_space']:10s} {item['metric']:14s} "
                f"mean={item['mean']:+.4f} p5={item['p5']:+.4f} "
                f"p50={item['p50']:+.4f} p95={item['p95']:+.4f}"
            )
    print(f"Saved paired-random artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
