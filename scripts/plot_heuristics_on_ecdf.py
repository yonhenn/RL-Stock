#!/usr/bin/env python3
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.visualization import configure_matplotlib

def empirical_percentile(random_values, model_value):
    random_values = np.asarray(random_values, dtype=float)
    count = np.count_nonzero(random_values <= model_value)
    return float(100.0 * (1 + count) / (len(random_values) + 2))

def main():
    configure_matplotlib()

    output_dir = Path("outputs/action_space_matched_audit")
    random_csv = output_dir / "random_100_paths.csv"

    if not random_csv.exists():
        print(f"Error: Random 100 paths CSV not found at {random_csv}")
        return

    df_rand = pd.read_csv(random_csv)

    # Heuristics profits (in %)
    heuristics_data = [
        {"space": "long_only", "name": "SMA Crossover", "profit": 15.86},
        {"space": "long_only", "name": "Uptrend SMA20", "profit": 9.65},
        {"space": "long_only", "name": "RSI Mean Rev", "profit": 7.82},
        {"space": "long_only", "name": "DT V2 P90", "profit": 8.68},
        {"space": "long_only", "name": "BC V2", "profit": 0.86},

        {"space": "symmetric", "name": "SMA Crossover", "profit": -8.57},
        {"space": "symmetric", "name": "Uptrend SMA20", "profit": -11.38},
        {"space": "symmetric", "name": "RSI Mean Rev", "profit": -12.09},
        {"space": "symmetric", "name": "DT V2 P90", "profit": 7.67},
        {"space": "symmetric", "name": "BC V2", "profit": -6.83},
    ]

    results = []

    # Setup plot
    fig, axis = plt.subplots(figsize=(12, 8))

    # 100-Random distributions
    lo_rand_profits = df_rand[df_rand["action_space"] == "long_only"]["profit"].values * 100.0
    sym_rand_profits = df_rand[df_rand["action_space"] == "symmetric"]["profit"].values * 100.0

    # Plot ECDF step lines
    for space, profits, color, label in (
        ("long_only", lo_rand_profits, "#1f77b4", "Long-only [0,1] Random Distribution"),
        ("symmetric", sym_rand_profits, "#7f7f7f", "Symmetric [-1,1] Random Distribution")
    ):
        sorted_profits = np.sort(profits)
        ecdf = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
        axis.step(sorted_profits, ecdf, where="post", label=label, color=color, alpha=0.6, linewidth=1.5)

    # Calculate percentiles and plot markers
    markers = {"SMA Crossover": "o", "Uptrend SMA20": "s", "RSI Mean Rev": "^", "DT V2 P90": "★", "BC V2": "D"}
    colors = {
        "SMA Crossover": "#2ca02c", # Green
        "Uptrend SMA20": "#ff7f0e",   # Orange
        "RSI Mean Rev": "#d62728",    # Red
        "DT V2 P90": "#9467bd",       # Purple
        "BC V2": "#e377c2"            # Pink
    }

    print("=" * 80)
    print(f"{'Space':<15} | {'Strategy':<20} | {'Profit (%)':<12} | {'ECDF Percentile (%)':<20}")
    print("-" * 80)

    for h in heuristics_data:
        space = h["space"]
        name = h["name"]
        profit = h["profit"]

        rand_vals = lo_rand_profits if space == "long_only" else sym_rand_profits
        percentile = empirical_percentile(rand_vals, profit)

        h["percentile"] = percentile
        results.append(h)

        print(f"{space:<15} | {name:<20} | {profit:<+12.2f} | {percentile:<20.1f}")

        # Plot marker on the ECDF
        # ECDF y-coordinate is percentile / 100.0
        y_val = percentile / 100.0
        marker = markers.get(name, "x")
        color = colors.get(name, "black")

        # Draw dot on ECDF
        if name == "DT V2 P90":
            axis.scatter(profit, y_val, marker="*", s=250, color=color, edgecolor="black", zorder=10, label=f"{name} ({space})")
        else:
            axis.scatter(profit, y_val, marker=marker, s=80, color=color, zorder=8)

        # Draw horizontal/vertical lines to ECDF dot
        axis.plot([profit, profit], [0, y_val], color=color, linestyle=":", alpha=0.4, linewidth=1.0)

        # Text annotation (offset to avoid overlapping)
        xytext = (10, -5) if space == "long_only" else (-15, 10)
        ha = "left" if space == "long_only" else "right"
        axis.annotate(
            f"{name}\n({percentile:.1f}%)",
            xy=(profit, y_val),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if "DT" in name else "normal",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.3),
            horizontalalignment=ha
        )

    print("=" * 80)

    axis.set_title("Heuristics and Models Placed on Random Profit ECDF", fontsize=15, fontweight="bold")
    axis.set_xlabel("Profit (%)", fontsize=12)
    axis.set_ylabel("Empirical Cumulative Probability (Percentile)", fontsize=12)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)

    # Legend formatting
    handles, labels = axis.get_legend_handles_labels()
    # Unique labels only
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axis.legend(*zip(*unique), loc="lower right")

    plt.tight_layout()
    output_path = output_dir / "heuristics_profit_ecdf.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved ECDF plot with heuristics to {output_path}")

    # Save CSV results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "heuristics_and_models_percentiles.csv", index=False)

if __name__ == "__main__":
    main()
