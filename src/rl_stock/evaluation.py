"""Model evaluation and result reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from rl_stock.risk import (
    EPS,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_turnover,
    compute_win_rate,
)


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


def evaluate_model(
    model,
    env,
    num_episodes: int = 10,
    seed: int = 42,
    deterministic: bool = True,
):
    """Evaluate a model and return rewards, profits, and per-episode info dicts."""
    rewards: list[float] = []
    profits: list[float] = []
    infos: list[dict] = []

    for episode in range(num_episodes):
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if model is None:
            env.action_space.seed(episode_seed)

        total_reward = 0.0
        done = False
        info: dict = {}
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        profits.append(info.get("total_profit", 1.0))
        infos.append(info)

    return rewards, profits, infos


def buy_and_hold_result(env, num_episodes: int = 1) -> dict:
    """Full-long buy-and-hold benchmark over the environment's test window.

    Computed straight from the price series (not from the env's reward / position
    dynamics) so it is a clean "just hold the index" reference: buy at the first
    tradable tick and hold to the end. Shaped like a model result dict so it
    flows through :func:`summarize_results` and the risk-metric helpers.
    """
    prices = np.asarray(env.prices, dtype=float)
    start = int(env.window_size)
    end = int(getattr(env, "_end_tick", len(prices) - 1))
    path = prices[start : end + 1]
    if path.size < 2 or path[0] <= 0:
        return {"profits": [1.0] * num_episodes, "rewards": [0.0] * num_episodes, "infos": [{}]}

    initial_cash = float(getattr(getattr(env, "risk", None), "initial_cash", 1.0))
    equity_curve = (initial_cash * path / path[0]).tolist()
    returns = (np.diff(path) / np.maximum(path[:-1], EPS)).tolist()
    total_profit = float(path[-1] / path[0])
    info = {
        "total_profit": total_profit,
        "equity_curve": equity_curve,
        "returns": returns,
        "position_history": [1.0] * path.size,
        "trade_returns": [total_profit - 1.0],
        "stop_loss_events": 0,
    }
    return {
        "profits": [total_profit] * num_episodes,
        "rewards": [0.0] * num_episodes,
        "infos": [info],
    }


def _risk_stats_from_infos(infos: list[dict]) -> dict[str, float]:
    """Aggregate per-episode risk metrics from PositionMgmtEnv info dicts."""
    sharpes: list[float] = []
    sortinos: list[float] = []
    max_dds: list[float] = []
    calmars: list[float] = []
    win_rates: list[float] = []
    turnovers: list[float] = []
    stop_events: list[float] = []

    for info in infos:
        equity = np.asarray(info.get("equity_curve", []), dtype=float)
        returns = np.asarray(info.get("returns", []), dtype=float)
        positions = np.asarray(info.get("position_history", []), dtype=float)
        trade_returns = np.asarray(info.get("trade_returns", []), dtype=float)

        if equity.size < 2 or returns.size == 0:
            continue

        sharpes.append(compute_sharpe_ratio(returns))
        sortinos.append(compute_sortino_ratio(returns))
        max_dds.append(compute_max_drawdown(equity))
        calmars.append(compute_calmar_ratio(returns, equity))
        if trade_returns.size > 0:
            win_rates.append(compute_win_rate(trade_returns))
        turnovers.append(compute_turnover(positions))
        stop_events.append(float(info.get("stop_loss_events", 0)))

    def mean_or_zero(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "sharpe": mean_or_zero(sharpes),
        "sortino": mean_or_zero(sortinos),
        "max_drawdown": mean_or_zero(max_dds),
        "calmar": mean_or_zero(calmars),
        "win_rate": mean_or_zero(win_rates),
        "turnover": mean_or_zero(turnovers),
        "stop_loss_events": mean_or_zero(stop_events),
    }


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


def print_risk_stats(name: str, infos: list[dict]) -> dict[str, float]:
    """Print Sharpe / drawdown / win-rate / turnover for the strategy."""
    stats = _risk_stats_from_infos(infos)
    if not stats or all(v == 0.0 for v in stats.values()):
        return stats
    print(f"  Sharpe:       {stats['sharpe']:.3f}")
    print(f"  Sortino:      {stats['sortino']:.3f}")
    print(f"  Max DD:       {stats['max_drawdown'] * 100:.2f}%")
    print(f"  Calmar:       {stats['calmar']:.3f}")
    print(f"  Win Rate:     {stats['win_rate'] * 100:.2f}%")
    print(f"  Turnover:     {stats['turnover']:.4f}")
    print(f"  Stop Outs:    {stats['stop_loss_events']:.2f}")
    return stats


def summarize_results(results: dict) -> pd.DataFrame:
    """Build a summary table for all evaluated strategies."""
    summary = []
    for name, metrics in results.items():
        profits = metrics["profits"]
        row: dict[str, object] = {
            "Strategy": name,
            "Average Profit": f"{np.mean(profits):.4f}",
            "Return Rate": f"{(np.mean(profits) - 1) * 100:.2f}%",
            "Highest": f"{np.max(profits):.4f}",
            "Lowest": f"{np.min(profits):.4f}",
            "Standard Deviation": f"{np.std(profits):.4f}",
        }
        infos = metrics.get("infos")
        if infos:
            risk = _risk_stats_from_infos(infos)
            row["Sharpe"] = f"{risk['sharpe']:.3f}"
            row["Max DD"] = f"{risk['max_drawdown'] * 100:.2f}%"
            row["Calmar"] = f"{risk['calmar']:.3f}"
        summary.append(row)
    return pd.DataFrame(summary)


def print_summary(
    results: dict,
    total_timesteps: int,
    test_period: str | None = None,
) -> pd.DataFrame:
    """Print and return the final summary table."""
    summary_df = summarize_results(results)
    print("\n" + "=" * 70)
    print("                    Test Results Summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    for name, metrics in results.items():
        infos = metrics.get("infos")
        if not infos:
            continue
        print(f"\nRisk metrics for {name}:")
        print_risk_stats(name, infos)

    best = max(results.keys(), key=lambda key: np.mean(results[key]["profits"]))
    best_profit = np.mean(results[best]["profits"])
    print(f"\nBest Strategy: {best}")
    print(f"Average Return Rate: {(best_profit - 1) * 100:.2f}%")
    if test_period:
        print(f"Test Period: {test_period}")
    print(f"Training Steps: {total_timesteps}")
    return summary_df
