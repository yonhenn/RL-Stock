"""Model evaluation and result reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """Simple tqdm progress bar for Stable-Baselines3 training."""

    def __init__(self, check_freq: int = 100):
        super().__init__()
        self.check_freq = check_freq
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.model._total_timesteps, desc="Training")

    def _on_step(self):
        if self.progress_bar is not None and self.n_calls % self.check_freq == 0:
            self.progress_bar.update(self.check_freq)
        return True

    def _on_training_end(self):
        if self.progress_bar is not None:
            remaining = self.model._total_timesteps - self.progress_bar.n
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()


def evaluate_model(model, env, num_episodes: int = 10, seed: int = 42, deterministic: bool = True):
    """Evaluate a model and return episode rewards and final profits."""
    rewards = []
    profits = []

    for episode in range(num_episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if model is None:
            env.action_space.seed(episode_seed)

        total_reward = 0.0
        done = False
        info = {}
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        profits.append(info["total_profit"])

    return rewards, profits


def print_stats(name: str, rewards, profits) -> None:
    """Print one strategy's aggregate statistics."""
    print(f"\n{'=' * 50}")
    print(f"Strategy: {name}")
    print(f"{'=' * 50}")
    print(f"Average Reward: {np.mean(rewards):.2f} (+/-{np.std(rewards):.2f})")
    print(f"Average Profit: {np.mean(profits):.4f}")
    print(f"Return Rate: {(np.mean(profits) - 1) * 100:.2f}%")
    print(f"Highest Profit: {np.max(profits):.4f} ({(np.max(profits) - 1) * 100:.2f}%)")
    print(f"Lowest Profit: {np.min(profits):.4f} ({(np.min(profits) - 1) * 100:.2f}%)")


def summarize_results(results: dict) -> pd.DataFrame:
    """Build a summary table for all evaluated strategies."""
    summary = []
    for name, metrics in results.items():
        profits = metrics["profits"]
        summary.append(
            {
                "Strategy": name,
                "Average Profit": f"{np.mean(profits):.4f}",
                "Return Rate": f"{(np.mean(profits) - 1) * 100:.2f}%",
                "Highest": f"{np.max(profits):.4f}",
                "Lowest": f"{np.min(profits):.4f}",
                "Standard Deviation": f"{np.std(profits):.4f}",
            }
        )
    return pd.DataFrame(summary)


def print_summary(results: dict, total_timesteps: int, test_period: str | None = None) -> pd.DataFrame:
    """Print and return the final summary table."""
    summary_df = summarize_results(results)
    print("\n" + "=" * 70)
    print("                    Test Results Summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    best = max(results.keys(), key=lambda key: np.mean(results[key]["profits"]))
    best_profit = np.mean(results[best]["profits"])
    print(f"\nBest Strategy: {best}")
    print(f"Average Return Rate: {(best_profit - 1) * 100:.2f}%")
    if test_period:
        print(f"Test Period: {test_period}")
    print(f"Training Steps: {total_timesteps}")
    return summary_df
