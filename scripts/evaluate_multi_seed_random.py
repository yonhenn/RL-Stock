#!/usr/bin/env python3
"""Script to evaluate Random trading baseline across 100+ seeds.

Computes mean, standard deviation, and robust percentiles (5th, 25th, 50th, 75th, 95th)
to account for non-Gaussian/skewed distributions of financial returns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio
from rl_stock.visualization import configure_matplotlib


def evaluate_random_multi_seed(
    df, test_bound, num_seeds: int, long_only: bool, start_seed: int = 42
) -> dict:
    risk_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=long_only,
    )
    env = PositionMgmtEnv(
        df=df, window_size=30, frame_bound=test_bound, risk_config=risk_config
    )

    profits = []
    sharpes = []
    max_dds = []
    stop_outs_list = []

    for episode in range(num_seeds):
        seed = start_seed + episode
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        done = False
        returns = []
        equity_curve = [env._equity]

        while not done:
            action = env.action_space.sample()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

            returns.append(info.get("net_return", 0.0))
            equity_curve.append(info.get("equity", 1.0))

        # Compute stats
        final_profit = info.get("total_profit", 1.0) - 1.0
        profits.append(final_profit * 100.0)  # in %

        sharpes.append(compute_sharpe_ratio(np.array(returns)))
        max_dds.append(compute_max_drawdown(np.array(equity_curve)) * 100.0)  # in %
        stop_outs_list.append(info.get("stop_loss_events", 0))

    return {
        "profits": np.array(profits),
        "sharpes": np.array(sharpes),
        "max_dds": np.array(max_dds),
        "stops": np.array(stop_outs_list),
    }


def print_stats_table(name: str, metrics: dict):
    print("=" * 60)
    print(f"Random Baseline Stats: {name} (N = {len(metrics['profits'])})")
    print("=" * 60)

    for key, label in [
        ("profits", "Profit (%)"),
        ("sharpes", "Sharpe Ratio"),
        ("max_dds", "Max Drawdown (%)"),
    ]:
        data = metrics[key]
        mean_val = np.mean(data)
        std_val = np.std(data)
        p5 = np.percentile(data, 5)
        p25 = np.percentile(data, 25)
        p50 = np.percentile(data, 50)
        p75 = np.percentile(data, 75)
        p95 = np.percentile(data, 95)

        print(f"{label:<18} | Mean: {mean_val:>+6.2f} ± {std_val:<5.2f} | Percentiles: [5th: {p5:>+6.2f}, 25th: {p25:>+6.2f}, 50th (Med): {p50:>+6.2f}, 75th: {p75:>+6.2f}, 95th: {p95:>+6.2f}]")

    # Stops distribution
    stops = metrics["stops"]
    unique_stops, counts = np.unique(stops, return_counts=True)
    stop_dist = ", ".join(
        [f"{k} stops: {v} times" for k, v in zip(unique_stops, counts)]
    )
    print(f"{'Stop-outs dist':<18} | {stop_dist}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Random baseline over multiple seeds."
    )
    parser.add_argument("--data-path", type=str, default="nasdaq100.xlsx")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--test-start-date", type=str, default="2024-01-01")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    configure_matplotlib()

    data_path = Path(args.data_path)
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(
        df, window_size=30, test_start_date=args.test_start_date
    )

    print(
        f"Evaluating Random baseline across {args.num_seeds} seeds on test set..."
    )

    # 1. Symmetric
    sym_metrics = evaluate_random_multi_seed(
        df, test_bound, args.num_seeds, long_only=False
    )
    print_stats_table("Symmetric [-1, 1] (Stops & Drawdown Active)", sym_metrics)

    # 2. Long-only
    lo_metrics = evaluate_random_multi_seed(
        df, test_bound, args.num_seeds, long_only=True
    )
    print_stats_table("Long-Only [0, 1] (Stops & Drawdown Active)", lo_metrics)

    # Generate profit distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (data, title, color) in enumerate(
        [
            (sym_metrics["profits"], "Symmetric [-1, 1]", "#a9a9a9"),
            (lo_metrics["profits"], "Long-Only [0, 1]", "#1f77b4"),
        ]
    ):
        ax = axes[idx]
        ax.hist(
            data,
            bins=15,
            edgecolor="black",
            alpha=0.7,
            color=color,
            density=False,
        )

        mean_val = np.mean(data)
        median_val = np.percentile(data, 50)
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_val:+.2f}%",
        )
        ax.axvline(
            median_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:+.2f}%",
        )
        ax.axvline(
            p5,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"5th %: {p5:+.2f}%",
        )
        ax.axvline(
            p95,
            color="green",
            linestyle=":",
            linewidth=1.5,
            label=f"95th %: {p95:+.2f}%",
        )

        ax.set_title(
            f"Random Profit Distribution: {title}", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Profit (%)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Frequency (Count)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "random_multi_seed_distribution.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved distribution plot to {plot_path}")


if __name__ == "__main__":
    main()
