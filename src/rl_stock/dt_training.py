"""Training orchestration for Decision Transformer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rl_stock.dt_dataset import TrajectoryDataset
from rl_stock.dt_inference import evaluate_dt_model
from rl_stock.envs_v2 import PositionMgmtEnv
from rl_stock.risk import compute_sharpe_ratio


def train_dt(
    model: nn.Module,
    dataset: TrajectoryDataset,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1000,
    eval_env: PositionMgmtEnv | None = None,
    eval_rtg: float = 0.05,
    eval_freq: int = 5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the Decision Transformer model on the collected trajectories dataset.

    Performs online evaluations periodically in the provided environment.
    """
    model.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device != "cpu"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate schedule with linear warmup
    # We step the scheduler after each batch
    total_steps = len(dataloader) * n_epochs
    
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.MSELoss(reduction="none")

    history = {
        "train_loss": [],
        "eval_epoch": [],
        "eval_mean_reward": [],
        "eval_mean_profit": [],
        "eval_sharpe": [],
    }

    best_sharpe = -float("inf")
    current_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for states, actions, rtgs, timesteps, attention_mask in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)
            attention_mask = attention_mask.to(device)

            # Target action is a_t (actions[:, :, 0])
            # The model predicts action given state and preceding RTG/actions
            pred_actions = model(
                states=states,
                actions=actions,
                rtgs=rtgs,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

            # Masked MSE loss (compute loss only where mask = 1)
            raw_loss = loss_fn(pred_actions, actions)  # (batch, K, 1)
            masked_loss = (raw_loss * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()

            optimizer.zero_grad()
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(masked_loss.item())
            current_step += 1

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {mean_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Evaluation
        if eval_env is not None and (epoch == 1 or epoch % eval_freq == 0 or epoch == n_epochs):
            print(f"Running online evaluation for epoch {epoch}...")
            rewards, profits, infos = evaluate_dt_model(
                model=model,
                env=eval_env,
                target_rtg=eval_rtg,
                state_mean=dataset.state_mean,
                state_std=dataset.state_std,
                rtg_scale=dataset.rtg_scale,
                num_episodes=1,
                device=device,
            )

            mean_reward = float(np.mean(rewards))
            mean_profit = float(np.mean(profits))

            # Collect all returns to compute annualized Sharpe
            all_returns = []
            for info in infos:
                all_returns.extend(info.get("returns", []))
            eval_sharpe = compute_sharpe_ratio(np.array(all_returns))

            history["eval_epoch"].append(epoch)
            history["eval_mean_reward"].append(mean_reward)
            history["eval_mean_profit"].append(mean_profit)
            history["eval_sharpe"].append(eval_sharpe)

            print(f"Eval Stats - Reward: {mean_reward:.4f} | Profit: {mean_profit:.4f} | Sharpe: {eval_sharpe:.4f}")

            # Keep track of best model
            if eval_sharpe > best_sharpe:
                best_sharpe = eval_sharpe
                # Store best model state dict inside the model wrapper if needed,
                # or save it in scripts
                model.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return history
