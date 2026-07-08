"""Long-only continuous-action position management environment with risk control.

``PositionMgmtEnv`` simulates a portfolio account where the agent decides a
*target long position ratio* in [0, 1] each step (0 = flat, 1 = fully
invested; the book is long-only, so no short selling). The environment
translates that target into trades, books transaction costs, optionally
enforces stop-loss / drawdown hard limits (both off by default, so risk is
reward-driven), and rewards the agent with a Differential Sharpe Ratio plus
penalties for drawdown and turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import numpy as np
from gymnasium import spaces

from rl_stock.indicators import (
    compute_adx,
    compute_bollinger_bands,
    compute_bollinger_bandwidth,
    compute_macd,
    compute_macd_cross,
    compute_rsi,
    compute_sma_slope,
    historical_moving_average,
)
from rl_stock.risk import EPS, IncrementalSharpe, compute_atr, compute_turbulence
from rl_stock.trend import (
    TrendRegime,
    classify_trend_regime,
    compute_trend_score,
    regime_to_base_position,
)

N_TECH_FEATURES = 13
N_TREND_FEATURES = 6
N_TURBULENCE_FEATURES = 1
N_PORTFOLIO_FEATURES = 6


@dataclass(frozen=True)
class RiskConfig:
    """Risk control knobs for :class:`PositionMgmtEnv`."""

    initial_cash: float = 1_000_000.0
    max_position: float = 1.0
    # Hard risk rails, both opt-in. Per-trade stop (0.05 = flatten at -5% of
    # entry) and equity drawdown lock (0.15 = de-risk past -15% off the peak).
    # A non-positive value disables that rail; the default 0.0 disables both so
    # the agent manages risk purely through the reward (DSR + penalties).
    stop_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_fee_rate: float = 0.001
    vol_target: float = 0.02
    # Reward shaping. ``reward_mode`` picks the objective:
    #   "return" (default) pays the agent the realized net return each step, so
    #     holding a long into an uptrend -- and sizing it up -- is directly
    #     rewarded (a profit-seeking objective).
    #   "dsr" is the legacy Differential Sharpe Ratio (scale-invariant, so it
    #     rewards smoothness but is indifferent to how much money is made); kept
    #     for comparison only.
    reward_mode: str = "return"
    drawdown_coeff: float = 0.1
    turnover_coeff: float = 0.1
    sharpe_coeff: float = 1.0
    dsr_eta: float = 0.01
    # Volatility position-scaling. When on, the agent's target is multiplied by
    # ``base_atr / current_atr`` (clamped) before clipping -- classic vol
    # targeting. Off by default: the agent already controls size directly in
    # [0, 1], and the legacy scaling silently capped upside (it holds ~0.6 avg
    # exposure for a 1.0 target, roughly halving trend capture).
    use_vol_scaling: bool = False
    use_trend_features: bool = False
    # Trend-aware position management (residual policy). The explicit trend
    # judgment anchors a base position; the agent learns a bounded tilt around
    # it, and a strongly trending regime can cap counter-trend exposure.
    use_trend_prior: bool = False
    trend_base_exposure: float = 0.8
    residual_band: float = 0.5
    regime_risk_scaling: bool = False
    counter_trend_cap: float = 0.2
    trend_align_coeff: float = 0.0
    adx_trend_threshold: float = 20.0
    adx_strong_threshold: float = 40.0
    # Market-level turbulence circuit breaker (Kritzman-Li, single-asset proxy).
    # When on, a turbulence state feature is added and exposure is capped to
    # ``turbulence_exposure_cap`` (0 = flat) on days whose squared return
    # z-score exceeds ``turbulence_threshold`` -- FinRL's crash risk-off rule.
    use_turbulence_gate: bool = False
    turbulence_lookback: int = 252
    turbulence_threshold: float = 9.0  # z**2; 9.0 == a 3-sigma daily move
    turbulence_exposure_cap: float = 0.0


class PositionMgmtEnv(gymnasium.Env):
    """Long-only continuous-action portfolio simulator with embedded risk controls.

    Action space: ``Box(0, 1, shape=(1,))`` -- target long position ratio
    (0 = flat, 1 = fully invested; long-only, no shorting).
    Observation: ``(window_size, n_tech + n_portfolio)`` window of features.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        window_size: int,
        frame_bound: tuple[int, int],
        risk_config: RiskConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if frame_bound[0] < window_size:
            raise ValueError("frame_bound[0] must be >= window_size")
        if frame_bound[1] <= frame_bound[0]:
            raise ValueError("frame_bound[1] must be greater than frame_bound[0]")

        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.risk = risk_config or RiskConfig()
        self.render_mode = render_mode

        self.prices, self.signal_features, self.atr = self._process_data()
        self._base_atr = float(np.mean(self.atr[: self.window_size])) or EPS

        n_tech = (
            N_TECH_FEATURES
            + (N_TREND_FEATURES if self.risk.use_trend_features else 0)
            + (N_TURBULENCE_FEATURES if self.risk.use_turbulence_gate else 0)
        )
        n_features = n_tech + N_PORTFOLIO_FEATURES
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )

        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._dsr = IncrementalSharpe(eta=self.risk.dsr_eta)
        self._reset_state()

    def _process_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_close = self.df.loc[:, "Close"].to_numpy()
        full_high = self.df.loc[:, "High"].to_numpy()
        full_low = self.df.loc[:, "Low"].to_numpy()
        full_open = self.df.loc[:, "Open"].to_numpy()

        ma5 = historical_moving_average(full_close, 5)
        ma10 = historical_moving_average(full_close, 10)
        ma20 = historical_moving_average(full_close, 20)
        ma60 = historical_moving_average(full_close, 60)
        rsi = compute_rsi(full_close, 14)
        _, _, macd_hist = compute_macd(full_close)
        bb_upper, _, bb_lower = compute_bollinger_bands(full_close, 20)
        atr_full = compute_atr(full_high, full_low, full_close, 14)

        if self.risk.use_trend_features:
            sma_slope_full = compute_sma_slope(full_close, period=20, slope_window=5)
            bb_bandwidth_full = compute_bollinger_bandwidth(full_close, period=20)
            adx_full = compute_adx(full_high, full_low, full_close, period=14)
            macd_cross_full = compute_macd_cross(full_close)

        # Explicit trend judgment, computed unconditionally so the residual
        # policy, regime caps, and alignment reward can index it cheaply.
        trend_score_full = compute_trend_score(
            full_high,
            full_low,
            full_close,
            adx_strong=self.risk.adx_strong_threshold,
        )
        regime_full = classify_trend_regime(
            full_high,
            full_low,
            full_close,
            adx_trend=self.risk.adx_trend_threshold,
            adx_strong=self.risk.adx_strong_threshold,
        )
        trend_base_full = regime_to_base_position(
            regime_full, self.risk.trend_base_exposure, long_only=True
        )

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        close = full_close[start:end]
        high = full_high[start:end]
        low = full_low[start:end]
        open_ = full_open[start:end]
        ma5 = ma5[start:end]
        ma10 = ma10[start:end]
        ma20 = ma20[start:end]
        ma60 = ma60[start:end]
        rsi = rsi[start:end]
        macd_hist = macd_hist[start:end]
        bb_upper = bb_upper[start:end]
        bb_lower = bb_lower[start:end]
        atr_window = atr_full[start:end]

        # Per-tick trend judgment aligned to ``self.prices`` (causal, no look-ahead).
        self._regime = regime_full[start:end].astype(np.float32)
        self._trend_base = trend_base_full[start:end].astype(np.float32)
        self._trend_score = trend_score_full[start:end].astype(np.float32)

        # Market-level turbulence aligned to ``self.prices`` (causal). Computed on
        # the full series so the test frame can draw on prior history; only the
        # gate consumes it, so skip the work when it is off.
        if self.risk.use_turbulence_gate:
            turbulence_full = compute_turbulence(
                full_close, lookback=self.risk.turbulence_lookback
            )
            self._turbulence = turbulence_full[start:end].astype(np.float32)
        else:
            self._turbulence = np.zeros(end - start, dtype=np.float32)

        eps = EPS
        pct_change = np.insert(np.diff(close) / np.maximum(close[:-1], eps) * 100, 0, 0)
        ma5_ratio = (close - ma5) / np.maximum(ma5, eps) * 100
        ma10_ratio = (close - ma10) / np.maximum(ma10, eps) * 100
        ma20_ratio = (close - ma20) / np.maximum(ma20, eps) * 100
        ma60_ratio = (close - ma60) / np.maximum(ma60, eps) * 100
        ma5_10_cross = (ma5 - ma10) / np.maximum(ma10, eps) * 100
        ma10_20_cross = (ma10 - ma20) / np.maximum(ma20, eps) * 100
        bb_width = bb_upper - bb_lower
        bb_position = np.clip((close - bb_lower) / np.maximum(bb_width, eps) * 100, 0, 100)
        rsi_norm = rsi / 100.0 * 2 - 1
        macd_max = np.maximum(np.abs(macd_hist).max(), eps)
        macd_norm = macd_hist / macd_max
        amplitude = (high - low) / np.maximum(close, eps) * 100
        body = (close - open_) / np.maximum(close, eps) * 100
        momentum_5 = np.zeros(len(close))
        momentum_5[5:] = (close[5:] - close[:-5]) / np.maximum(close[:-5], eps) * 100

        feature_columns = [
            pct_change,
            ma5_ratio,
            ma10_ratio,
            ma20_ratio,
            ma60_ratio,
            ma5_10_cross,
            ma10_20_cross,
            rsi_norm,
            macd_norm,
            bb_position / 100,
            amplitude,
            body,
            momentum_5,
        ]

        if self.risk.use_trend_features:
            sma_slope_norm = sma_slope_full[start:end] * 100.0  # percent per day
            bb_bandwidth_col = bb_bandwidth_full[start:end]  # ratio, already O(0.1)
            adx_norm = adx_full[start:end] / 100.0  # -> [0, 1]
            macd_cross_col = macd_cross_full[start:end]  # {-1, 0, +1}
            trend_score_col = self._trend_score  # signed strength [-1, 1]
            regime_norm = self._regime / 2.0  # {-1, -.5, 0, .5, 1}
            feature_columns.extend(
                [
                    sma_slope_norm,
                    bb_bandwidth_col,
                    adx_norm,
                    macd_cross_col,
                    trend_score_col,
                    regime_norm,
                ]
            )

        if self.risk.use_turbulence_gate:
            # Bounded, smooth "how turbulent" signal: ~0 when calm, -> 1 at/above
            # the gate threshold. tanh keeps it O(1) like the other features.
            turbulence_norm = np.tanh(
                self._turbulence / max(self.risk.turbulence_threshold, EPS)
            )
            feature_columns.append(turbulence_norm)

        signal_features = np.column_stack(feature_columns)
        signal_features = np.nan_to_num(signal_features, nan=0.0, posinf=10.0, neginf=-10.0)
        signal_features = np.clip(signal_features, -100, 100)
        return (
            close.astype(np.float32),
            signal_features.astype(np.float32),
            atr_window.astype(np.float32),
        )

    def _reset_state(self) -> None:
        self._current_tick = self._start_tick
        self._position_ratio = 0.0
        self._entry_price = 0.0
        self._cash = self.risk.initial_cash
        self._equity = self.risk.initial_cash
        self._peak_equity = self.risk.initial_cash
        self._holding_days = 0
        self._stopped_out = False
        self._dd_locked = False
        self._dsr.reset()
        self._equity_curve: list[float] = [self._equity]
        self._position_history: list[float] = [0.0]
        self._returns: list[float] = []
        self._trade_returns: list[float] = []
        self._stop_loss_events = 0
        self._truncated = False
        self._terminated = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def _get_portfolio_features(self) -> np.ndarray:
        """Build the per-step portfolio state vector tiled across the window."""
        current_price = float(self.prices[self._current_tick])
        unrealized_pnl = 0.0
        if abs(self._position_ratio) > EPS and self._entry_price > EPS:
            unrealized_pnl = (
                (current_price - self._entry_price) / self._entry_price * np.sign(self._position_ratio)
            )

        drawdown = (
            (self._equity - self._peak_equity) / max(self._peak_equity, EPS)
        )
        atr_now = float(self.atr[self._current_tick])
        volatility = atr_now / max(current_price, EPS)
        holding_norm = np.tanh(self._holding_days / 30.0)
        # Neutral (1.0 = far from any stop) when flat or when the hard stop is
        # disabled (stop_loss_pct <= 0); otherwise how close to the stop we are.
        if abs(self._position_ratio) > EPS and self.risk.stop_loss_pct > 0:
            distance_to_stop = 1.0 - abs(unrealized_pnl) / self.risk.stop_loss_pct
        else:
            distance_to_stop = 1.0

        return np.array(
            [
                self._position_ratio,
                np.clip(unrealized_pnl, -1.0, 1.0),
                np.clip(drawdown, -1.0, 1.0),
                float(holding_norm),
                float(volatility),
                float(np.clip(distance_to_stop, -1.0, 2.0)),
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        end = self._current_tick
        start = end - self.window_size
        tech = self.signal_features[start:end]
        portfolio_row = self._get_portfolio_features()
        portfolio_block = np.tile(portfolio_row, (self.window_size, 1))
        return np.concatenate([tech, portfolio_block], axis=1).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "tick": self._current_tick,
            "position_ratio": self._position_ratio,
            "equity": self._equity,
            "cash": self._cash,
            "peak_equity": self._peak_equity,
            "drawdown": (self._equity - self._peak_equity)
            / max(self._peak_equity, EPS),
            "holding_days": self._holding_days,
            "total_profit": self._equity / self.risk.initial_cash,
            "stop_loss_events": self._stop_loss_events,
            "equity_curve": list(self._equity_curve),
            "position_history": list(self._position_history),
            "trade_returns": list(self._trade_returns),
            "returns": list(self._returns),
            "dd_locked": self._dd_locked,
            "regime": float(self._regime[self._current_tick]),
            "trend_score": float(self._trend_score[self._current_tick]),
            "trend_base": float(self._trend_base[self._current_tick]),
            "turbulence": float(self._turbulence[self._current_tick]),
        }

    def _volatility_scaled_target(self, raw_target: float) -> float:
        """Scale the agent's target by current volatility (lower vol = bigger size).

        Long-only: the result is clamped to ``[0, max_position]`` so a target can
        never go short.
        """
        atr_now = float(self.atr[self._current_tick])
        if atr_now < EPS:
            scale = 1.0
        else:
            scale = self._base_atr / atr_now
        scale = float(np.clip(scale, 0.25, 2.0))
        scaled = raw_target * scale
        return float(np.clip(scaled, 0.0, self.risk.max_position))

    def _resolve_target(self, raw_action: float) -> float:
        """Translate the raw agent action into a target long position ratio.

        The action is a long ratio in ``[0, 1]``. With ``use_trend_prior`` the
        explicit trend judgment supplies a base long position (the trend-following
        anchor) and the action becomes an upward *tilt* around it (residual
        policy: ``action == 0`` holds the base, larger actions add exposure);
        otherwise the action is the target directly. The result is optionally
        volatility-scaled (``use_vol_scaling``, off by default), clamped to
        ``[0, max_position]`` (long-only), and, when ``regime_risk_scaling`` is
        on, further capped inside a strong down-trend.
        """
        raw = float(np.clip(raw_action, 0.0, 1.0))
        if self.risk.use_trend_prior:
            prior = float(self._trend_base[self._current_tick])
            combined = prior + raw * self.risk.residual_band
        else:
            combined = raw
        if self.risk.use_vol_scaling:
            target = self._volatility_scaled_target(combined)
        else:
            target = float(np.clip(combined, 0.0, self.risk.max_position))
        if self.risk.regime_risk_scaling:
            target = self._apply_regime_caps(target)
        if self.risk.use_turbulence_gate:
            target = self._apply_turbulence_gate(target)
        return target

    def _apply_regime_caps(self, target: float) -> float:
        """Cap long exposure inside a strong down-trend (long-only: nothing to cap up).

        With shorting removed the only counter-trend risk left is holding a large
        long into a strong down-trend, so the cap clamps the long there to
        ``counter_trend_cap``. A strong up-trend needs no cap (the agent is already
        long-only, so it can never take the counter-trend short side).
        """
        regime = float(self._regime[self._current_tick])
        cap = self.risk.counter_trend_cap
        if regime <= float(TrendRegime.STRONG_DOWN):
            target = min(target, cap)
        return float(target)

    def _is_turbulent(self) -> bool:
        """Whether the current tick's turbulence exceeds the gate threshold."""
        return bool(
            self.risk.use_turbulence_gate
            and self._turbulence[self._current_tick] > self.risk.turbulence_threshold
        )

    def _apply_turbulence_gate(self, target: float) -> float:
        """Risk-off on turbulent days (FinRL's turbulence rule, single-asset).

        When the latest move is an outlier (squared return z-score above the
        threshold) the long target is clamped to ``turbulence_exposure_cap``, so a
        market shock forces de-risking. ``cap == 0`` flattens the book outright.
        Long-only: the floor stays at ``0`` (the gate never opens a short).
        """
        if not self._is_turbulent():
            return float(target)
        cap = abs(self.risk.turbulence_exposure_cap)
        return float(np.clip(target, 0.0, cap))

    def step(self, action):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        raw_action = float(np.asarray(action).flatten()[0])
        target_position = self._resolve_target(raw_action)

        prev_position = self._position_ratio
        current_price = float(self.prices[self._current_tick])

        unrealized_pnl = 0.0
        if abs(prev_position) > EPS and self._entry_price > EPS:
            unrealized_pnl = (
                (current_price - self._entry_price) / self._entry_price * np.sign(prev_position)
            )

        # Hard risk constraints, both opt-in. Stop-loss flattens immediately;
        # the drawdown lock forbids new exposure but lets the agent reduce. A
        # non-positive ``stop_loss_pct`` / ``max_drawdown_pct`` disables that
        # rail entirely (the default: pure reward-driven risk control).
        stop_triggered = (
            self.risk.stop_loss_pct > 0
            and abs(prev_position) > EPS
            and unrealized_pnl <= -self.risk.stop_loss_pct
        )
        if stop_triggered:
            target_position = 0.0
            self._stop_loss_events += 1
            self._stopped_out = True

        drawdown = (self._equity - self._peak_equity) / max(self._peak_equity, EPS)
        if self.risk.max_drawdown_pct > 0 and drawdown <= -self.risk.max_drawdown_pct:
            self._dd_locked = True
        if self._dd_locked and abs(target_position) > abs(prev_position):
            target_position = prev_position * 0.5

        delta_position = target_position - prev_position
        notional = abs(delta_position) * self._equity
        transaction_cost = notional * self.risk.trade_fee_rate

        self._cash -= transaction_cost
        self._equity -= transaction_cost

        if abs(target_position) > EPS:
            if abs(prev_position) < EPS or np.sign(prev_position) != np.sign(target_position):
                self._entry_price = current_price
                self._holding_days = 0
            else:
                self._holding_days += 1
        else:
            self._entry_price = 0.0
            self._holding_days = 0

        prev_tick = self._current_tick
        self._current_tick += 1
        if self._current_tick >= self._end_tick:
            self._truncated = True
            self._current_tick = self._end_tick

        next_price = float(self.prices[self._current_tick])
        price_change_pct = (next_price - current_price) / max(current_price, EPS)
        portfolio_return = target_position * price_change_pct
        cost_drag = transaction_cost / max(self._equity, EPS)
        net_return = portfolio_return - cost_drag

        self._equity *= (1.0 + portfolio_return)
        self._peak_equity = max(self._peak_equity, self._equity)
        self._position_ratio = target_position

        self._equity_curve.append(self._equity)
        self._position_history.append(target_position)
        self._returns.append(net_return)

        flatten = stop_triggered or (abs(prev_position) > EPS and abs(target_position) < EPS)
        if flatten and abs(prev_position) > EPS:
            self._trade_returns.append(unrealized_pnl)

        reward = self._calculate_reward(
            net_return, delta_position, float(self._trend_score[prev_tick])
        )

        if self._equity <= 0:
            self._terminated = True

        observation = self._get_observation()
        info = self._get_info()
        info["prev_tick"] = prev_tick
        info["delta_position"] = delta_position
        info["stop_triggered"] = stop_triggered
        info["turbulence_triggered"] = bool(
            self.risk.use_turbulence_gate
            and self._turbulence[prev_tick] > self.risk.turbulence_threshold
        )
        return observation, reward, self._terminated, self._truncated, info

    def _calculate_reward(
        self, net_return: float, delta_position: float, trend_score: float
    ) -> float:
        drawdown = (self._equity - self._peak_equity) / max(self._peak_equity, EPS)
        # Penalise only the downside of drawdown and only churn; both scale with
        # their coefficients. Drawdown is already <= 0, so ``min(drawdown, 0)``
        # keeps the penalty non-positive.
        drawdown_penalty = min(drawdown, 0.0) * self.risk.drawdown_coeff
        turnover_penalty = -abs(delta_position) * self.risk.turnover_coeff
        # Optional: reward holding a long aligned with the trend judgment
        # (positive when long in an uptrend, negative when long into a
        # downtrend; flat earns nothing). Off by default (coeff 0).
        align_bonus = (
            self.risk.trend_align_coeff
            * float(np.sign(self._position_ratio))
            * trend_score
        )

        if self.risk.reward_mode == "dsr":
            # Legacy Differential Sharpe Ratio: scale-invariant risk-adjusted
            # signal. Rewards smoothness but is nearly indifferent to how much
            # money is made (a half-sized and a full-sized winning position earn
            # almost the same reward), so it under-participates in trends.
            sharpe_component = self._dsr.update(net_return)
            reward = (
                sharpe_component * self.risk.sharpe_coeff
                + drawdown_penalty
                + turnover_penalty
                + align_bonus
            )
        else:
            # Return-driven objective (default): pay the realized net return
            # (in percent) so the agent is directly rewarded for capturing --
            # and for sizing up into -- a profitable move.
            reward = (
                net_return * 100.0
                + drawdown_penalty
                + turnover_penalty
                + align_bonus
            )
        if not np.isfinite(reward):
            return 0.0
        return float(reward)

    def render(self):  # pragma: no cover - rendering handled via visualization module
        return None
