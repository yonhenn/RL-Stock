import sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/rickchen/Antigravity/RL wit Nasdaq")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.visualization import configure_matplotlib

def main():
    configure_matplotlib()
    
    dt_history_path = PROJECT_ROOT / "outputs" / "dt" / "history.pkl"
    bc_history_path = PROJECT_ROOT / "outputs" / "bc" / "history.pkl"
    output_dir = Path("/Users/rickchen/.gemini/antigravity/brain/4b083282-a711-4b21-823e-1218356772a9")
    
    with open(dt_history_path, "rb") as f:
        dt_hist = pickle.load(f)
    with open(bc_history_path, "rb") as f:
        bc_hist = pickle.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Train Loss Decay Comparison
    ax1 = axes[0]
    ax1.plot(range(1, len(dt_hist["train_loss"]) + 1), dt_hist["train_loss"], color="#1f77b4", marker="o", markersize=4, linewidth=2, label="Decision Transformer (DT)")
    ax1.plot(range(1, len(bc_hist["train_loss"]) + 1), bc_hist["train_loss"], color="#ff7f0e", marker="s", markersize=4, linewidth=2, label="Behavior Cloning (BC)")
    ax1.set_title("Training Loss Decay Comparison", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Masked MSE Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10.5)
    
    # Subplot 2: Online Evaluation Sharpe Ratio Progression
    ax2 = axes[1]
    dt_epochs = dt_hist["eval_epoch"]
    bc_epochs = bc_hist["eval_epoch"]
    
    ax2.plot(dt_epochs, dt_hist["eval_sharpe"], color="#1f77b4", marker="o", markersize=6, linewidth=2, label="DT (RTG = 0.05 Target)")
    ax2.plot(bc_epochs, bc_hist["eval_sharpe"], color="#ff7f0e", marker="s", markersize=6, linewidth=2, label="BC (No Target)")
    
    # Add baseline references
    ax2.axhline(y=0.455, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8, label="PPO Expert Baseline (0.455)")
    ax2.axhline(y=-1.046, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.8, label="SAC Baseline (-1.046)")
    
    ax2.set_title("Online Eval Sharpe Ratio Comparison", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10.5, loc="lower right")
    
    plt.tight_layout()
    plot_path = output_dir / "dt_bc_training_curves.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved training comparison curves to {plot_path}")

if __name__ == "__main__":
    main()
