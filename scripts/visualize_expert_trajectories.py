#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.visualization import configure_matplotlib

def main():
    parser = argparse.ArgumentParser(description="Visualize all expert policy trajectories from a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset_v2.pkl")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated image")
    parser.add_argument("--title-suffix", type=str, default="", help="Suffix for plot titles")
    args = parser.parse_args()

    configure_matplotlib()

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    trajectories = dataset["trajectories"]
    metadata = dataset["metadata"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color map for policies
    colors = {
        "Random": "#ff7f0e",          # Orange
        "SAC-Stochastic": "#1f77b4",  # Blue
        "PPO-Stochastic": "#2ca02c",  # Green
    }

    # Group trajectories by policy type
    for i, traj in enumerate(trajectories):
        policy_name = traj[0]["policy_name"]

        # Extract executed actions (positions)
        executed_pos = np.array([step["executed_action"] for step in traj])

        # Extract net returns and compute cumulative equity
        net_returns = np.array([step["net_return"] for step in traj])
        equity = np.cumprod(1.0 + net_returns)

        # Plot executed position on ax1
        if "Fixed" in policy_name:
            color = "#7f7f7f"
            alpha = 0.5
            linewidth = 1.0
            linestyle = ":"
        else:
            color = colors.get(policy_name, "#bcbd22")
            alpha = 0.4
            linewidth = 1.2
            linestyle = "-"

        # Draw position trajectory
        ax1.plot(executed_pos, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)

        # Draw equity trajectory
        ax2.plot(equity, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)

    # Ax1 formatting (Positions)
    ax1.set_title(f"Executed Position Trajectories {args.title_suffix}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Trading Day (Timestep)", fontsize=12)
    ax1.set_ylabel("Executed Position Ratio", fontsize=12)
    if metadata.get("long_only", True):
        ax1.set_ylim(-0.05, 1.05)
    else:
        ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Ax2 formatting (Equity)
    ax2.set_title(f"Cumulative Equity Curves (Training Period) {args.title_suffix}", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Trading Day (Timestep)", fontsize=12)
    ax2.set_ylabel("Portfolio Value (Starting from 1.0)", fontsize=12)
    ax2.set_yscale("log")  # Use log scale for long-term compounding returns
    ax2.grid(True, alpha=0.3)

    # Add custom legends
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="SAC-Stochastic (Expert)"),
        Line2D([0], [0], color="#2ca02c", lw=2, label="PPO-Stochastic (Expert)"),
        Line2D([0], [0], color="#ff7f0e", lw=2, label="Random"),
        Line2D([0], [0], color="#7f7f7f", lw=1, ls=":", label="Fixed Constant Target"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left")
    ax2.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory visualization to {output_path}")

if __name__ == "__main__":
    main()
