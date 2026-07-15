"""Reconstruct per-strategy equity curves by replaying the trained models.

The training pipeline only persists aggregate metrics (``summary.csv``) and a
rendered PNG, not the per-step equity time series. To draw an *interactive*
equity comparison we reload each model and replay a single deterministic
episode over the test window, which for a deterministic policy reproduces the
exact curve behind the saved metrics.

The environment feature set must match how a run was trained or the observation
shape will not line up with the saved policy. The trend-feature / turbulence
flags are recoverable from the model's observation width; the residual-prior and
regime-scaling flags are not (they change dynamics, not shape), so they are
surfaced to the caller as explicit :class:`EnvSettings` overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rl_stock.dashboard.runs import RunInfo

# Six per-tick portfolio features are always appended to the signal block, so
# ``observation_width - 6`` isolates the technical-signal width, which in turn
# pins down the optional feature groups. Verified empirically against the env:
#   default = 19, +turbulence = 20, +trend = 25, +trend +turbulence = 26.
_OBS_WIDTH_TO_FLAGS: dict[int, dict[str, bool]] = {
    19: {"use_trend_features": False, "use_turbulence_gate": False},
    20: {"use_trend_features": False, "use_turbulence_gate": True},
    25: {"use_trend_features": True, "use_turbulence_gate": False},
    26: {"use_trend_features": True, "use_turbulence_gate": True},
}

RANDOM_LABEL = "Random"
BUY_AT_DIP_LABEL = "Buy at the Dip"
BUY_AND_HOLD_LABEL = "Buy and Hold"

# Rule-based baselines shown alongside the trained RL models on the trade page.
# Keep this order in sync with the requested selector order.
BASELINE_STRATEGIES: tuple[str, ...] = (
    BUY_AT_DIP_LABEL,
    RANDOM_LABEL,
    BUY_AND_HOLD_LABEL,
)


class ReplayUnavailable(RuntimeError):
    """Raised when a run's equity curves cannot be reconstructed."""


@dataclass(frozen=True)
class EnvSettings:
    """Replay-time environment configuration.

    ``use_trend_features`` / ``use_turbulence_gate`` are auto-detected from the
    model when left as ``None``. The remaining flags cannot be inferred from a
    saved policy and default to the training defaults; override them to match a
    run that used the residual trend prior or regime-aware caps.
    """

    data_path: Path
    test_start_date: str = "2024-01-01"
    window_size: int = 30
    seed: int = 42
    use_trend_features: bool | None = None
    use_turbulence_gate: bool | None = None
    use_trend_prior: bool = False
    regime_risk_scaling: bool = False
    initial_cash: float = 1_000_000.0
    max_position: float = 1.0
    trade_fee_rate: float = 0.001
    vol_target: float = 0.02
    residual_band: float = 0.5
    counter_trend_cap: float = 0.2


@dataclass(frozen=True)
class ReplayResult:
    """Equity, position, and price paths plus the settings used to produce them."""

    equity: pd.DataFrame  # index = trade date, one column per strategy
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)  # target position ratio
    price: pd.Series = field(default_factory=pd.Series)  # NASDAQ-100 close over the window
    detected_flags: dict[str, bool] = field(default_factory=dict)
    initial_cash: float = 1_000_000.0

    @property
    def cumulative_return_pct(self) -> pd.DataFrame:
        """Equity re-based to a 0% start, in percentage points."""
        base = self.equity.iloc[0].replace(0.0, np.nan)
        return (self.equity.divide(base) - 1.0) * 100.0

    @property
    def drawdown_pct(self) -> pd.DataFrame:
        """Per-strategy drawdown from the running peak, in percentage points."""
        running_max = self.equity.cummax()
        return (self.equity.divide(running_max) - 1.0) * 100.0


def _algo_class(algo: str):
    """Resolve an algorithm label to its stable-baselines3 class (lazy import)."""
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    classes = {"A2C": A2C, "PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    try:
        return classes[algo.upper()]
    except KeyError as exc:
        raise ReplayUnavailable(f"unknown algorithm: {algo!r}") from exc


def infer_env_flags(obs_width: int) -> dict[str, bool] | None:
    """Map an observation width to known feature flags, or ``None`` if unknown."""
    flags = _OBS_WIDTH_TO_FLAGS.get(int(obs_width))
    return dict(flags) if flags is not None else None


def _build_env(data_path: Path, settings: EnvSettings, flags: dict[str, bool]):
    """Build the position-management test environment for the given settings."""
    from rl_stock.data import load_market_data, split_frame_bounds
    from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig

    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(
        df,
        window_size=settings.window_size,
        test_start_date=settings.test_start_date,
    )
    risk = RiskConfig(
        initial_cash=settings.initial_cash,
        max_position=settings.max_position,
        trade_fee_rate=settings.trade_fee_rate,
        vol_target=settings.vol_target,
        use_trend_features=flags["use_trend_features"],
        use_turbulence_gate=flags["use_turbulence_gate"],
        use_trend_prior=settings.use_trend_prior,
        regime_risk_scaling=settings.regime_risk_scaling,
        residual_band=settings.residual_band,
        counter_trend_cap=settings.counter_trend_cap,
    )
    env = PositionMgmtEnv(
        df=df,
        window_size=settings.window_size,
        frame_bound=test_bound,
        risk_config=risk,
    )
    return env


def _run_episode(env, model, seed: int) -> dict:
    """Replay one deterministic episode and return the terminal ``info`` dict.

    ``model is None`` runs the random baseline (sampling the action space).
    """
    obs, _ = env.reset(seed=seed)
    if model is None:
        env.action_space.seed(seed)
    done = False
    info: dict = {}
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def _dated_series(env, values, name: str) -> pd.Series:
    """Attach the test-window trade dates to a per-step value array."""
    arr = np.asarray(values, dtype=float)
    start = env.frame_bound[0]
    dates = env.df.index[start : start + arr.size]
    return pd.Series(arr, index=dates, name=name)


def _price_series(env) -> pd.Series:
    """The raw NASDAQ-100 close over the test window, aligned to the curves."""
    prices = np.asarray(env.prices[env.window_size :], dtype=float)
    start = env.frame_bound[0]
    dates = env.df.index[start : start + prices.size]
    return pd.Series(prices, index=dates, name="NASDAQ-100")


def _buy_and_hold_paths(
    price: pd.Series, initial_cash: float
) -> tuple[pd.Series, pd.Series]:
    """Return equity and full-long position paths for a buy-and-hold baseline."""
    values = price.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0:
        empty = pd.Series(index=price.index, dtype=float)
        return empty.rename(BUY_AND_HOLD_LABEL), empty.copy().rename(BUY_AND_HOLD_LABEL)

    equity = initial_cash * values / values[0]
    positions = np.ones(values.size, dtype=float)
    return (
        pd.Series(equity, index=price.index, name=BUY_AND_HOLD_LABEL),
        pd.Series(positions, index=price.index, name=BUY_AND_HOLD_LABEL),
    )


def _buy_at_dip_paths(
    price: pd.Series,
    initial_cash: float,
    trade_fee_rate: float,
    *,
    lookback: int = 20,
    entry_discount: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """Return paths for a simple causal buy-the-dip benchmark.

    The strategy enters a full-long position when the close is at least 2%
    below its trailing 20-day moving average, holds through the dip, and exits
    when the close recovers to that moving average.  Trades are charged the same
    proportional fee configured for the replay environment.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not 0.0 <= entry_discount < 1.0:
        raise ValueError("entry_discount must be in [0, 1)")
    if not 0.0 <= trade_fee_rate < 1.0:
        raise ValueError("trade_fee_rate must be in [0, 1)")

    values = price.to_numpy(dtype=float)
    equity_values = np.full(values.size, float(initial_cash), dtype=float)
    position_values = np.zeros(values.size, dtype=float)
    if values.size == 0:
        return (
            pd.Series(equity_values, index=price.index, name=BUY_AT_DIP_LABEL),
            pd.Series(position_values, index=price.index, name=BUY_AT_DIP_LABEL),
        )

    trailing_mean = (
        pd.Series(values)
        .rolling(window=lookback, min_periods=lookback)
        .mean()
        .to_numpy(dtype=float)
    )
    equity = float(initial_cash)
    position = 0.0

    for i, current_price in enumerate(values):
        # Mark the position to market before deciding whether to trade at the
        # current close. This keeps the rule causal and the dates intuitive.
        if i > 0 and np.isfinite(current_price) and values[i - 1] > 0:
            price_return = current_price / values[i - 1] - 1.0
            equity *= 1.0 + position * price_return

        target = position
        reference = trailing_mean[i]
        if np.isfinite(current_price) and np.isfinite(reference) and reference > 0:
            if position == 0.0 and current_price <= reference * (1.0 - entry_discount):
                target = 1.0
            elif position > 0.0 and current_price >= reference:
                target = 0.0

        turnover = abs(target - position)
        if turnover:
            equity *= 1.0 - turnover * trade_fee_rate
        position = target
        equity_values[i] = equity
        position_values[i] = position

    return (
        pd.Series(equity_values, index=price.index, name=BUY_AT_DIP_LABEL),
        pd.Series(position_values, index=price.index, name=BUY_AT_DIP_LABEL),
    )


def replay_run(run: RunInfo, settings: EnvSettings) -> ReplayResult:
    """Reload every model in ``run`` and reconstruct their equity/position paths.

    Also builds Random, Buy at the Dip, and Buy and Hold baselines and captures
    the underlying price series, so callers can overlay every strategy's trades
    on the raw index.

    Raises:
        ReplayUnavailable: if the run has no models, the environment feature set
            cannot be inferred, or model loading fails.
    """
    if not run.has_models:
        raise ReplayUnavailable(f"run {run.label!r} has no saved models to replay")

    models: dict[str, object] = {}
    obs_width: int | None = None
    for algo, model_path in run.model_paths:
        model = _algo_class(algo).load(model_path, device="cpu")
        models[algo] = model
        width = int(model.observation_space.shape[-1])
        if obs_width is None:
            obs_width = width
        elif width != obs_width:
            raise ReplayUnavailable(
                f"run {run.label!r} mixes observation widths ({obs_width} vs {width})"
            )

    detected = infer_env_flags(obs_width) if obs_width is not None else None
    if detected is None:
        raise ReplayUnavailable(
            f"cannot infer env features for observation width {obs_width}; "
            "this run was trained with an unsupported feature set"
        )

    flags = {
        "use_trend_features": (
            detected["use_trend_features"]
            if settings.use_trend_features is None
            else settings.use_trend_features
        ),
        "use_turbulence_gate": (
            detected["use_turbulence_gate"]
            if settings.use_turbulence_gate is None
            else settings.use_turbulence_gate
        ),
    }

    equity_curves: dict[str, pd.Series] = {}
    position_curves: dict[str, pd.Series] = {}

    # Fresh env per strategy keeps each replay hermetic; model inference
    # dominates the per-strategy cost, so rebuilding the env is cheap.
    env = _build_env(settings.data_path, settings, flags)
    info = _run_episode(env, None, settings.seed)
    if not info.get("equity_curve"):
        raise ReplayUnavailable("episode produced an empty equity curve")
    equity_curves[RANDOM_LABEL] = _dated_series(env, info["equity_curve"], RANDOM_LABEL)
    position_curves[RANDOM_LABEL] = _dated_series(env, info["position_history"], RANDOM_LABEL)
    price = _price_series(env)

    dip_equity, dip_positions = _buy_at_dip_paths(
        price,
        settings.initial_cash,
        settings.trade_fee_rate,
    )
    equity_curves[BUY_AT_DIP_LABEL] = dip_equity
    position_curves[BUY_AT_DIP_LABEL] = dip_positions

    hold_equity, hold_positions = _buy_and_hold_paths(price, settings.initial_cash)
    equity_curves[BUY_AND_HOLD_LABEL] = hold_equity
    position_curves[BUY_AND_HOLD_LABEL] = hold_positions

    for algo, model in models.items():
        env = _build_env(settings.data_path, settings, flags)
        info = _run_episode(env, model, settings.seed)
        equity_curves[algo] = _dated_series(env, info["equity_curve"], algo)
        position_curves[algo] = _dated_series(env, info["position_history"], algo)

    return ReplayResult(
        equity=pd.DataFrame(equity_curves),
        positions=pd.DataFrame(position_curves),
        price=price,
        detected_flags=detected,
        initial_cash=settings.initial_cash,
    )
