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

    output_dir = Path("outputs/optimization_ablation")
    random_csv = Path("outputs/action_space_matched_audit/random_100_paths.csv")
    comparison_csv = output_dir / "comparison.csv"

    if not random_csv.exists():
        print(f"Error: Random 100 paths CSV not found at {random_csv}")
        return
    if not comparison_csv.exists():
        print(f"Error: Comparison CSV not found at {comparison_csv}")
        return

    df_rand = pd.read_csv(random_csv)
    df_comp = pd.read_csv(comparison_csv)

    # We want to extract final epoch metrics for each run
    # Finals are identified by the maximum epoch for each (action_space, model, run_id)
    finals = df_comp.sort_values("epoch").groupby(["action_space", "model", "run_id"]).tail(1).copy()

    lo_rand_profits = df_rand[df_rand["action_space"] == "long_only"]["profit"].values * 100.0
    sym_rand_profits = df_rand[df_rand["action_space"] == "symmetric"]["profit"].values * 100.0

    results = []

    print("=" * 90)
    print(f"{'Space':<12} | {'Model':<5} | {'Run ID':<8} | {'Profit (%)':<12} | {'ECDF Percentile (%)':<20}")
    print("-" * 90)

    for _, row in finals.iterrows():
        space = row["action_space"]
        model = row["model"]
        run_id = row["run_id"]
        profit = float(row["profit"]) * 100.0  # convert to %

        rand_vals = lo_rand_profits if space == "long_only" else sym_rand_profits
        percentile = empirical_percentile(rand_vals, profit)

        results.append({
            "action_space": space,
            "model": model,
            "run_id": run_id,
            "profit_pct": profit,
            "percentile": percentile,
            "sharpe": float(row["sharpe"]),
            "max_dd": float(row["max_drawdown"]) * 100.0
        })

        print(f"{space:<12} | {model.upper():<5} | {run_id:<8} | {profit:<+12.2f} | {percentile:<20.1f}")

    print("=" * 90)

    # Setup plot
    fig, axis = plt.subplots(figsize=(13, 9))

    # Plot ECDF step lines
    for space, profits, color, label in (
        ("long_only", lo_rand_profits, "#1f77b4", "Long-only [0,1] Random Distribution"),
        ("symmetric", sym_rand_profits, "#7f7f7f", "Symmetric [-1,1] Random Distribution")
    ):
        sorted_profits = np.sort(profits)
        ecdf = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
        axis.step(sorted_profits, ecdf, where="post", label=label, color=color, alpha=0.5, linewidth=1.5)

    # Let's plot trajectories for DT and BC
    # For each space and model, we have e20_s20 -> e40_s10 -> e60_s5
    res_df = pd.DataFrame(results)

    styles = {
        ("long_only", "dt"): ("#9467bd", "o-", "DT Long-Only"),
        ("long_only", "bc"): ("#e377c2", "s-", "BC Long-Only"),
        ("symmetric", "dt"): ("#2ca02c", "^-", "DT Symmetric"),
        ("symmetric", "bc"): ("#ff7f0e", "d-", "BC Symmetric")
    }

    for (space, model), (color, style, label) in styles.items():
        sub = res_df[(res_df.action_space == space) & (res_df.model == model)].copy()
        if sub.empty:
            continue

        # Sort by run ID in order of intensity: e20_s20 -> e40_s10 -> e60_s5
        sub["order"] = sub["run_id"].map({"e20_s20": 0, "e40_s10": 1, "e60_s5": 2})
        sub = sub.sort_values("order")

        x_vals = sub["profit_pct"].values
        y_vals = sub["percentile"].values / 100.0

        # Plot line connecting the steps
        axis.plot(x_vals, y_vals, style, color=color, linewidth=2.0, markersize=8, label=label, zorder=10)

        # Draw arrows between steps to show trajectory direction
        for i in range(len(x_vals) - 1):
            dx = x_vals[i+1] - x_vals[i]
            dy = y_vals[i+1] - y_vals[i]
            # Draw standard annotation arrow
            axis.annotate(
                "",
                xy=(x_vals[i+1], y_vals[i+1]),
                xytext=(x_vals[i], y_vals[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="-", alpha=0.8),
                zorder=9
            )

        # Label the endpoints/milestones
        for _, row in sub.iterrows():
            lbl = f"{row['run_id']}\n({row['profit_pct']:.1f}%)"
            xytext = (10, -5) if space == "long_only" else (-15, 10)
            ha = "left" if space == "long_only" else "right"
            axis.annotate(
                lbl,
                xy=(row["profit_pct"], row["percentile"]/100.0),
                xytext=xytext,
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha=ha,
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.3),
                zorder=12
            )

    axis.set_title("Ablation Optimization Trajectories Placed on Random Profit ECDF", fontsize=15, fontweight="bold")
    axis.set_xlabel("Profit (%)", fontsize=12)
    axis.set_ylabel("Empirical Cumulative Probability (Percentile)", fontsize=12)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")

    plt.tight_layout()
    output_path = output_dir / "ablation_profit_ecdf.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved ECDF ablation trajectory plot to {output_path}")

    # Save percentile results to csv
    res_df.to_csv(output_dir / "ablation_percentiles.csv", index=False)

if __name__ == "__main__":
    main()
