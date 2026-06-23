"""Generate and embed figures for the feature-branch presentation."""

from __future__ import annotations

import argparse
import base64
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs import IndexTradingEnvT1
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_turnover,
    compute_win_rate,
)


ENV_LABELS = {
    "t1": "T+1 discrete",
    "position": "Position risk control",
}
ALGO_ORDER = ["Random", "A2C", "PPO", "SAC", "TD3"]
FIGURE_ALTS = {
    "returns": "Generated return comparison chart",
    "risk": "Generated risk metrics dashboard",
    "equity": "Generated equity curve comparison",
}


@dataclass
class StrategyEval:
    env_key: str
    strategy: str
    rewards: list[float]
    profits: list[float]
    infos: list[dict[str, Any]]
    metrics: dict[str, float]

    @property
    def label(self) -> str:
        return f"{ENV_LABELS[self.env_key]} / {self.strategy}"

    @property
    def avg_profit(self) -> float:
        return float(np.mean(self.profits)) if self.profits else 1.0

    @property
    def return_pct(self) -> float:
        return (self.avg_profit - 1.0) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature presentation charts from trained RL outputs."
    )
    parser.add_argument("--t1-dir", type=Path, default=ROOT / "outputs" / "t1")
    parser.add_argument("--position-dir", type=Path, default=ROOT / "outputs" / "position")
    parser.add_argument("--data-path", type=Path, default=ROOT / "nasdaq100.xlsx")
    parser.add_argument("--html", type=Path, default=ROOT / "feature_presentation.html")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "outputs" / "feature_presentation" / "figures",
    )
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed", action="store_true")

    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.15)
    parser.add_argument("--trade-fee-rate", type=float, default=0.001)
    parser.add_argument("--vol-target", type=float, default=0.02)
    parser.add_argument("--drawdown-coeff", type=float, default=0.5)
    parser.add_argument("--turnover-coeff", type=float, default=0.1)
    parser.add_argument("--sharpe-coeff", type=float, default=1.0)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#030303",
            "axes.facecolor": "#0b0b0b",
            "savefig.facecolor": "#030303",
            "axes.edgecolor": "#777777",
            "axes.labelcolor": "#e8e8e8",
            "xtick.color": "#b8b8b8",
            "ytick.color": "#b8b8b8",
            "text.color": "#f7f7f7",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the training workflow first."
        )


def load_summary(output_dir: Path, env_key: str) -> pd.DataFrame:
    summary_path = output_dir / "summary.csv"
    require_file(summary_path, f"{env_key} summary.csv")
    summary = pd.read_csv(summary_path)
    if "Strategy" not in summary.columns:
        raise ValueError(f"{summary_path} must contain a Strategy column.")
    return summary


def sorted_strategies(summary: pd.DataFrame) -> list[str]:
    names = [str(name) for name in summary["Strategy"].tolist()]
    return sorted(names, key=lambda name: ALGO_ORDER.index(name) if name in ALGO_ORDER else 99)


def build_test_env(env_key: str, df: pd.DataFrame, args: argparse.Namespace):
    _, test_bound = split_frame_bounds(
        df,
        window_size=args.window_size,
        test_start_date=args.test_start_date,
    )
    if env_key == "t1":
        return IndexTradingEnvT1(
            df=df,
            window_size=args.window_size,
            frame_bound=test_bound,
        )

    risk = RiskConfig(
        initial_cash=args.initial_cash,
        max_position=args.max_position,
        stop_loss_pct=args.stop_loss_pct,
        max_drawdown_pct=args.max_drawdown_pct,
        trade_fee_rate=args.trade_fee_rate,
        vol_target=args.vol_target,
        drawdown_coeff=args.drawdown_coeff,
        turnover_coeff=args.turnover_coeff,
        sharpe_coeff=args.sharpe_coeff,
        dsr_eta=args.dsr_eta,
    )
    return PositionMgmtEnv(
        df=df,
        window_size=args.window_size,
        frame_bound=test_bound,
        risk_config=risk,
    )


def load_model(strategy: str, output_dir: Path, env) -> Any:
    try:
        from stable_baselines3 import A2C, PPO, SAC, TD3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required to load trained models. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    loaders = {"A2C": A2C, "PPO": PPO, "SAC": SAC, "TD3": TD3}
    if strategy not in loaders:
        raise ValueError(f"Unknown trained strategy in summary.csv: {strategy!r}")

    model_path = output_dir / f"{strategy.lower()}.zip"
    require_file(model_path, f"{strategy} model")
    return loaders[strategy].load(model_path, env=env)


def run_episode(model: Any, env, seed: int) -> tuple[float, float, dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    if model is None:
        env.action_space.seed(seed)

    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return total_reward, float(info.get("total_profit", 1.0)), info


def evaluate_strategy(
    env_key: str,
    strategy: str,
    model: Any,
    env,
    args: argparse.Namespace,
) -> StrategyEval:
    rewards: list[float] = []
    profits: list[float] = []
    infos: list[dict[str, Any]] = []

    for episode in range(args.num_eval_episodes):
        reward, profit, info = run_episode(model, env, args.seed + episode)
        rewards.append(reward)
        profits.append(profit)
        infos.append(info)

    metrics = aggregate_metrics(env_key, env, infos)
    metrics["return"] = float(np.mean(profits) - 1.0) if profits else 0.0
    return StrategyEval(
        env_key=env_key,
        strategy=strategy,
        rewards=rewards,
        profits=profits,
        infos=infos,
        metrics=metrics,
    )


def evaluate_output_dir(
    env_key: str,
    output_dir: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[StrategyEval]:
    summary = load_summary(output_dir, env_key)
    env = build_test_env(env_key, df, args)
    results: list[StrategyEval] = []

    for strategy in sorted_strategies(summary):
        model = None if strategy == "Random" else load_model(strategy, output_dir, env)
        results.append(evaluate_strategy(env_key, strategy, model, env, args))

    return results


def position_to_exposure(position: Any) -> float:
    name = getattr(position, "name", None)
    if name is not None:
        return 1.0 if name.lower() == "long" else 0.0
    value = getattr(position, "value", position)
    try:
        return 1.0 if int(value) == 1 else 0.0
    except (TypeError, ValueError):
        return 1.0 if "long" in str(position).lower() else 0.0


def t1_trade_returns(env: IndexTradingEnvT1, info: dict[str, Any]) -> np.ndarray:
    prices = np.asarray(env.prices, dtype=float)
    buy_ticks = info.get("buy_ticks", []) or []
    sell_ticks = info.get("sell_ticks", []) or []
    returns: list[float] = []
    for buy_tick, sell_tick in zip(buy_ticks, sell_ticks):
        if 0 <= buy_tick < len(prices) and 0 <= sell_tick < len(prices):
            gross = prices[sell_tick] / max(prices[buy_tick], 1e-8) - 1.0
            net = gross - env.trade_fee_bid_percent - env.trade_fee_ask_percent
            returns.append(float(net))
    return np.asarray(returns, dtype=float)


def t1_equity_series(
    env: IndexTradingEnvT1,
    info: dict[str, Any],
    initial_equity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions_raw = info.get("position_history", []) or []
    exposures = np.asarray([position_to_exposure(pos) for pos in positions_raw], dtype=float)
    if exposures.size == 0:
        return np.asarray([initial_equity]), np.asarray([]), np.asarray([]), np.asarray([])

    start = int(getattr(env, "window_size", 0))
    end = min(start + exposures.size, len(env.prices))
    prices = np.asarray(env.prices[start:end], dtype=float)
    exposures = exposures[: prices.size]
    if prices.size < 2:
        return np.asarray([initial_equity]), np.asarray([]), exposures, t1_trade_returns(env, info)

    price_returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)
    previous_exposure = np.concatenate(([0.0], exposures[:-1]))
    exposure_change = exposures - previous_exposure
    fees = np.where(
        exposure_change > 0,
        exposure_change * env.trade_fee_bid_percent,
        np.abs(exposure_change) * env.trade_fee_ask_percent,
    )
    step_returns = exposures[:-1] * price_returns - fees[:-1]
    equity = initial_equity * np.concatenate(([1.0], np.cumprod(1.0 + step_returns)))
    return equity, step_returns, exposures, t1_trade_returns(env, info)


def position_equity_series(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    returns = np.asarray(info.get("returns", []), dtype=float)
    positions = np.asarray(info.get("position_history", []), dtype=float)
    trade_returns = np.asarray(info.get("trade_returns", []), dtype=float)
    if equity.size == 0:
        equity = np.asarray([1.0])
    if equity[0] != 0:
        equity = equity / equity[0]
    return equity, returns, positions, trade_returns


def episode_series(
    env_key: str,
    env,
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if env_key == "t1":
        return t1_equity_series(env, info)
    return position_equity_series(info)


def mean_or_zero(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def aggregate_metrics(env_key: str, env, infos: list[dict[str, Any]]) -> dict[str, float]:
    final_returns: list[float] = []
    sharpes: list[float] = []
    sortinos: list[float] = []
    max_dds: list[float] = []
    calmars: list[float] = []
    win_rates: list[float] = []
    turnovers: list[float] = []
    stop_events: list[float] = []

    for info in infos:
        equity, returns, positions, trade_returns = episode_series(env_key, env, info)
        if equity.size:
            final_returns.append(float(equity[-1] - 1.0))
        if returns.size > 1:
            sharpes.append(compute_sharpe_ratio(returns))
            sortinos.append(compute_sortino_ratio(returns))
        if equity.size > 1:
            max_dds.append(compute_max_drawdown(equity))
        if returns.size and equity.size:
            calmars.append(compute_calmar_ratio(returns, equity))
        if trade_returns.size:
            win_rates.append(compute_win_rate(trade_returns))
        if positions.size > 1:
            turnovers.append(compute_turnover(positions))
        stop_events.append(float(info.get("stop_loss_events", 0)))

    return {
        "return": mean_or_zero(final_returns),
        "sharpe": mean_or_zero(sharpes),
        "sortino": mean_or_zero(sortinos),
        "max_drawdown": mean_or_zero(max_dds),
        "calmar": mean_or_zero(calmars),
        "win_rate": mean_or_zero(win_rates),
        "turnover": mean_or_zero(turnovers),
        "stop_loss_events": mean_or_zero(stop_events),
    }


def choose_best(results: list[StrategyEval], env_key: str) -> StrategyEval:
    candidates = [
        result
        for result in results
        if result.env_key == env_key and result.strategy != "Random"
    ]
    if not candidates:
        raise ValueError(f"No trained strategies found for env_key={env_key!r}.")
    return max(candidates, key=lambda result: result.return_pct)


def get_result(results: list[StrategyEval], env_key: str, strategy: str) -> StrategyEval | None:
    for result in results:
        if result.env_key == env_key and result.strategy == strategy:
            return result
    return None


def representative_curve(result: StrategyEval, env) -> np.ndarray:
    if not result.infos:
        return np.asarray([1.0])
    equity, _, _, _ = episode_series(result.env_key, env, result.infos[0])
    if equity.size == 0:
        return np.asarray([1.0])
    return equity / equity[0] if equity[0] != 0 else equity


def build_envs_for_curves(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "t1": build_test_env("t1", df, args),
        "position": build_test_env("position", df, args),
    }


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#2b2b2b", linewidth=0.7, alpha=0.9)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#030303", pad_inches=0.12)
    plt.close(fig)


def generate_return_chart(results: list[StrategyEval], path: Path) -> None:
    colors = {
        "Random": "#555555",
        "A2C": "#f5f5f5",
        "PPO": "#bdbdbd",
        "SAC": "#8c8c8c",
        "TD3": "#303030",
    }
    hatches = {"Random": "///", "A2C": "", "PPO": "\\\\", "SAC": "..", "TD3": "xx"}
    envs = ["t1", "position"]
    strategies = ["Random", "A2C", "PPO", "SAC"]
    width = 0.18
    x = np.arange(len(envs))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for idx, strategy in enumerate(strategies):
        values = []
        labels = []
        for env_key in envs:
            result = get_result(results, env_key, strategy)
            values.append(result.return_pct if result else np.nan)
            labels.append(result.strategy if result else "")
        offset = (idx - 1.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=strategy,
            color=colors[strategy],
            edgecolor="#ffffff",
            linewidth=0.6,
            alpha=0.94,
            hatch=hatches[strategy],
        )
        for bar, value, label in zip(bars, values, labels):
            if not label or not np.isfinite(value):
                continue
            va = "bottom" if value >= 0 else "top"
            y = value + (0.45 if value >= 0 else -0.45)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:+.1f}%",
                ha="center",
                va=va,
                fontsize=8,
                color="#f7f7f7",
            )

    ax.axhline(0, color="#ffffff", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[key] for key in envs])
    ax.set_ylabel("Average return on test episodes")
    ax.set_title("Real training output: strategy return comparison", loc="left")
    ax.legend(ncols=4, frameon=False, loc="upper left")
    clean_axis(ax)
    save_figure(fig, path)


def generate_risk_chart(best_t1: StrategyEval, best_position: StrategyEval, path: Path) -> None:
    labels = [f"T+1 / {best_t1.strategy}", f"Position / {best_position.strategy}"]
    metrics = [
        ("Return %", "return", 100.0, True),
        ("Sharpe", "sharpe", 1.0, True),
        ("Sortino", "sortino", 1.0, True),
        ("Calmar", "calmar", 1.0, True),
        ("Max DD %", "max_drawdown", 100.0, False),
        ("Turnover", "turnover", 1.0, False),
    ]
    x = np.arange(len(metrics))
    width = 0.36

    fig = plt.figure(figsize=(10.6, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    first_values = [best_t1.metrics[key] * scale for _, key, scale, _ in metrics]
    second_values = [best_position.metrics[key] * scale for _, key, scale, _ in metrics]
    ax.bar(
        x - width / 2,
        first_values,
        width,
        label=labels[0],
        color="#f5f5f5",
        edgecolor="#ffffff",
        linewidth=0.5,
        hatch="",
    )
    ax.bar(
        x + width / 2,
        second_values,
        width,
        label=labels[1],
        color="#8c8c8c",
        edgecolor="#ffffff",
        linewidth=0.5,
        hatch="\\\\",
    )
    ax.axhline(0, color="#ffffff", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _, _ in metrics])
    ax.set_title("Risk-adjusted dashboard for best trained strategies", loc="left")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)

    ax_table.axis("off")
    table_rows = [
        ["Best trained strategy", best_t1.strategy, best_position.strategy],
        ["Win rate", fmt_pct(best_t1.metrics["win_rate"]), fmt_pct(best_position.metrics["win_rate"])],
        [
            "Stop-loss events",
            fmt_num(best_t1.metrics["stop_loss_events"]),
            fmt_num(best_position.metrics["stop_loss_events"]),
        ],
        [
            "Interpretation",
            "Post-hoc analytics only",
            "Native risk controls in env",
        ],
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", "T+1 discrete", "Position risk control"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.8)
        cell.set_facecolor("#0b0b0b")
        cell.get_text().set_color("#f7f7f7")

    save_figure(fig, path)


def generate_equity_chart(
    results: list[StrategyEval],
    df: pd.DataFrame,
    args: argparse.Namespace,
    path: Path,
) -> None:
    envs = build_envs_for_curves(df, args)
    best_t1 = choose_best(results, "t1")
    best_position = choose_best(results, "position")
    t1_random = get_result(results, "t1", "Random")
    position_random = get_result(results, "position", "Random")

    _, test_bound = split_frame_bounds(
        df,
        window_size=args.window_size,
        test_start_date=args.test_start_date,
    )
    close = df["Close"].iloc[test_bound[0] : test_bound[1]].to_numpy(dtype=float)
    buy_hold = close / close[0] if close.size else np.asarray([1.0])

    series: list[tuple[str, np.ndarray, str, float]] = [
        (f"T+1 best: {best_t1.strategy}", representative_curve(best_t1, envs["t1"]), "#f5f5f5", 2.2),
        (
            f"Position best: {best_position.strategy}",
            representative_curve(best_position, envs["position"]),
            "#9e9e9e",
            2.2,
        ),
        ("Buy & Hold", buy_hold, "#d6d6d6", 1.55),
    ]
    if t1_random:
        series.append(("T+1 random", representative_curve(t1_random, envs["t1"]), "#666666", 1.2))
    if position_random:
        series.append(
            (
                "Position random",
                representative_curve(position_random, envs["position"]),
                "#3f3f3f",
                1.2,
            )
        )

    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(10.8, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08},
    )
    linestyles = ["-", "--", ":", "-.", (0, (5, 1))]
    for idx, (label, curve, color, linewidth) in enumerate(series):
        if curve.size == 0:
            continue
        x = np.arange(curve.size)
        linestyle = linestyles[idx % len(linestyles)]
        ax_eq.plot(x, curve, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        ax_dd.plot(
            x,
            drawdown * 100.0,
            color=color,
            linewidth=max(linewidth - 0.4, 0.9),
            linestyle=linestyle,
            alpha=0.86,
        )

    ax_eq.axhline(1.0, color="#ffffff", linewidth=0.9, alpha=0.5)
    ax_eq.set_ylabel("Equity, start = 1.0")
    ax_eq.set_title("Representative test-window equity curves", loc="left")
    ax_eq.legend(frameon=False, ncols=2, loc="upper left")
    clean_axis(ax_eq)

    ax_dd.axhline(0, color="#ffffff", linewidth=0.9, alpha=0.45)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel("Test step")
    clean_axis(ax_dd)
    save_figure(fig, path)


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100.0:.{decimals}f}%"


def fmt_signed_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100.0:+.{decimals}f}%"


def fmt_num(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def html_metric_rows(best_t1: StrategyEval, best_position: StrategyEval) -> str:
    rows = [
        ("Best strategy", best_t1.strategy, best_position.strategy),
        ("Return", fmt_signed_pct(best_t1.metrics["return"]), fmt_signed_pct(best_position.metrics["return"])),
        ("Sharpe", fmt_num(best_t1.metrics["sharpe"]), fmt_num(best_position.metrics["sharpe"])),
        ("Sortino", fmt_num(best_t1.metrics["sortino"]), fmt_num(best_position.metrics["sortino"])),
        ("Max drawdown", fmt_pct(best_t1.metrics["max_drawdown"]), fmt_pct(best_position.metrics["max_drawdown"])),
        ("Calmar", fmt_num(best_t1.metrics["calmar"]), fmt_num(best_position.metrics["calmar"])),
        ("Win rate", fmt_pct(best_t1.metrics["win_rate"]), fmt_pct(best_position.metrics["win_rate"])),
        ("Turnover", fmt_num(best_t1.metrics["turnover"]), fmt_num(best_position.metrics["turnover"])),
        (
            "Stop-loss events",
            fmt_num(best_t1.metrics["stop_loss_events"]),
            fmt_num(best_position.metrics["stop_loss_events"]),
        ),
    ]
    return "\n".join(
        "              <tr>"
        f"<th>{html.escape(metric)}</th>"
        f"<td>{html.escape(t1_value)}</td>"
        f"<td>{html.escape(position_value)}</td>"
        "</tr>"
        for metric, t1_value, position_value in rows
    )


def html_summary(best_t1: StrategyEval, best_position: StrategyEval) -> str:
    delta = best_position.metrics["return"] - best_t1.metrics["return"]
    dd_delta = best_position.metrics["max_drawdown"] - best_t1.metrics["max_drawdown"]
    return "\n".join(
        [
            f'              <span class="data-pill">Best T+1: {html.escape(best_t1.strategy)} '
            f'{fmt_signed_pct(best_t1.metrics["return"])}</span>',
            f'              <span class="data-pill">Best Position: {html.escape(best_position.strategy)} '
            f'{fmt_signed_pct(best_position.metrics["return"])}</span>',
            f'              <span class="data-pill">Return delta: {fmt_signed_pct(delta)}</span>',
            f'              <span class="data-pill">MaxDD delta: {fmt_signed_pct(dd_delta)}</span>',
        ]
    )


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def replace_img_src(document: str, alt_text: str, uri: str) -> str:
    pattern = rf'(<img\b(?=[^>]*\balt="{re.escape(alt_text)}")[^>]*\bsrc=")[^"]*(")'
    new_document, count = re.subn(
        pattern,
        lambda match: f"{match.group(1)}{uri}{match.group(2)}",
        document,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Expected to replace one image with alt={alt_text!r}; replaced {count}.")
    return new_document


def replace_block(document: str, marker: str, replacement: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    pattern = rf"({re.escape(start)})(.*?)({re.escape(end)})"
    new_document, count = re.subn(
        pattern,
        lambda match: f"{match.group(1)}\n{replacement}\n{match.group(3)}",
        document,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"Expected one {marker} block; replaced {count}.")
    return new_document


def embed_into_html(
    html_path: Path,
    figures: dict[str, Path],
    best_t1: StrategyEval,
    best_position: StrategyEval,
) -> None:
    require_file(html_path, "feature_presentation.html")
    document = html_path.read_text(encoding="utf-8")
    for key, alt_text in FIGURE_ALTS.items():
        document = replace_img_src(document, alt_text, image_uri(figures[key]))
    document = replace_block(document, "FEATURE_METRIC_ROWS", html_metric_rows(best_t1, best_position))
    document = replace_block(document, "FEATURE_SUMMARY", html_summary(best_t1, best_position))
    html_path.write_text(document, encoding="utf-8")


def print_result_summary(results: list[StrategyEval]) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "Environment": ENV_LABELS[result.env_key],
                "Strategy": result.strategy,
                "Return %": f"{result.return_pct:+.2f}",
                "Sharpe": f"{result.metrics['sharpe']:.3f}",
                "Max DD %": f"{result.metrics['max_drawdown'] * 100:.2f}",
                "Calmar": f"{result.metrics['calmar']:.3f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    require_file(args.data_path, "market data")
    df = load_market_data(args.data_path)

    results = []
    results.extend(evaluate_output_dir("t1", args.t1_dir, df, args))
    results.extend(evaluate_output_dir("position", args.position_dir, df, args))

    best_t1 = choose_best(results, "t1")
    best_position = choose_best(results, "position")

    figures = {
        "returns": args.fig_dir / "return_comparison.png",
        "risk": args.fig_dir / "risk_dashboard.png",
        "equity": args.fig_dir / "equity_curves.png",
    }
    generate_return_chart(results, figures["returns"])
    generate_risk_chart(best_t1, best_position, figures["risk"])
    generate_equity_chart(results, df, args, figures["equity"])

    if args.embed:
        embed_into_html(args.html, figures, best_t1, best_position)

    print_result_summary(results)
    for name, path in figures.items():
        print(f"{name}: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    if args.embed:
        print(f"embedded: {args.html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
