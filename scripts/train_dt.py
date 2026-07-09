#!/usr/bin/env python3
"""CLI script to train Decision Transformer on offline trajectories."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_dataset import TrajectoryDataset
from rl_stock.dt_model import DecisionTransformer
from rl_stock.dt_training import train_dt
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig


def main():
    parser = argparse.ArgumentParser(description="Train a Decision Transformer.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/dt_dataset.pkl",
        help="Path to offline dataset pickle file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="nasdaq100.xlsx",
        help="Path to NASDAQ-100 Excel file.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=20,
        help="Context sequence length K (default: 20).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps (default: 500).",
    )
    parser.add_argument(
        "--target-rtg",
        type=float,
        default=0.05,
        help="Target RTG for evaluation (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dt",
        help="Directory to save checkpoints (default: outputs/dt).",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2024-01-01",
        help="Test start date (default: 2024-01-01).",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=10,
        help="Timestep stride for dataset sampling (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # For fully deterministic CUDA ops (may slow down training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {args.seed}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load market data
    print(f"Loading market data from {args.data_path}...")
    df = load_market_data(args.data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date=args.test_start_date)

    # Initialize environment in flat_obs mode for validation
    risk_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
    )
    eval_env = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=risk_config,
    )

    # Setup dataset to inspect maximum RTG for scaling
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        raw_data = pickle.load(f)

    # Find the maximum absolute RTG in dataset to scale inputs
    all_rtgs = []
    for traj in raw_data["trajectories"]:
        for step in traj:
            all_rtgs.append(abs(step["rtg"]))
    
    rtg_scale = max(all_rtgs) if all_rtgs else 1.0
    if rtg_scale < 1e-6:
        rtg_scale = 1.0
    print(f"Computed RTG scale factor: {rtg_scale:.6f} (max absolute RTG in dataset)")

    # Create dataset
    dataset = TrajectoryDataset(
        dataset_path=args.dataset,
        context_length=args.context_length,
        rtg_scale=rtg_scale,
        step_stride=args.step_stride,
    )

    state_dim = dataset.metadata["state_dim"]

    # Auto-compute max_episode_len from dataset (avoid embedding OOB)
    max_timestep = 0
    for traj in raw_data["trajectories"]:
        for step in traj:
            max_timestep = max(max_timestep, step["timestep"])
    max_episode_len = max_timestep + 128  # headroom for longer eval episodes
    print(f"Dataset stats - State dim: {state_dim} | Episodes: {len(dataset.trajectories)}")
    print(f"Max timestep in dataset: {max_timestep} → max_episode_len: {max_episode_len}")

    # Instantiate model
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=1,
        context_length=args.context_length,
        max_episode_len=max_episode_len,
    )

    print("Starting Decision Transformer training...")
    history = train_dt(
        model=model,
        dataset=dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_env=eval_env,
        eval_rtg=args.target_rtg,
        eval_freq=min(5, args.n_epochs),
        device=device,
    )

    # Ensure output dir exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save checkpoints
    # Save the best evaluation state dict if it was populated during training
    state_dict_to_save = getattr(model, "best_state_dict", model.state_dict())
    
    checkpoint = {
        "state_dict": state_dict_to_save,
        "state_mean": dataset.state_mean,
        "state_std": dataset.state_std,
        "rtg_scale": dataset.rtg_scale,
        "context_length": args.context_length,
        "state_dim": state_dim,
        "max_episode_len": max_episode_len,
        "seed": args.seed,
    }

    checkpoint_file = output_path / "dt_model.pt"
    torch.save(checkpoint, checkpoint_file)
    print(f"Saved best model checkpoint to {checkpoint_file}")

    # Save training history
    history_file = output_path / "history.pkl"
    with open(history_file, "wb") as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_file}")

    print("\nDecision Transformer training complete!")


if __name__ == "__main__":
    main()
