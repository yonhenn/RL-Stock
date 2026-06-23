"""Trading environments for index reinforcement learning (NASDAQ-100)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from gym_anytrading.envs import Actions, Positions, StocksEnv

from rl_stock.indicators import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    historical_moving_average,
)


class IndexTradingEnv(StocksEnv):
    """Enhanced index trading environment from the original notebook."""

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        super().__init__(df, window_size, frame_bound, render_mode)
        self.trade_fee_bid_percent = 0.001
        self.trade_fee_ask_percent = 0.0003

    def _process_data(self):
        full_close = self.df.loc[:, "Close"].to_numpy()
        full_high = self.df.loc[:, "High"].to_numpy()
        full_low = self.df.loc[:, "Low"].to_numpy()
        full_open = self.df.loc[:, "Open"].to_numpy()

        ma5 = np.convolve(full_close, np.ones(5) / 5, mode="same")
        ma10 = np.convolve(full_close, np.ones(10) / 10, mode="same")
        ma20 = np.convolve(full_close, np.ones(20) / 20, mode="same")
        ma60 = np.convolve(full_close, np.ones(60) / 60, mode="same")
        rsi = compute_rsi(full_close, 14)
        _, _, macd_hist = compute_macd(full_close)
        bb_upper, _, bb_lower = compute_bollinger_bands(full_close, 20)

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = full_close[start:end]
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

        eps = 1e-8
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

        signal_features = np.column_stack(
            (
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
            )
        )
        signal_features = np.nan_to_num(signal_features, nan=0.0, posinf=10.0, neginf=-10.0)
        signal_features = np.clip(signal_features, -100, 100)
        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0
        action = int(np.asarray(action).item())

        trade = (
            action == Actions.Buy.value
            and self._position == Positions.Short
            or action == Actions.Sell.value
            and self._position == Positions.Long
        )

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            if self._position == Positions.Long:
                step_reward = price_diff / last_trade_price * 100
        elif self._current_tick > 0:
            current_price = self.prices[self._current_tick]
            prev_price = self.prices[self._current_tick - 1]
            price_change = (current_price - prev_price) / prev_price * 100
            if self._position == Positions.Long:
                step_reward = price_change * 0.1

        return step_reward


class IndexTradingEnvT1(StocksEnv):
    """T+1 constrained discrete index trading environment.

    Note: the T+1 hold-one-day rule is a China A-share constraint and does not
    apply to NASDAQ; kept available for comparison but ``position`` is the
    appropriate env for US markets.
    """

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        self.frame_bound = frame_bound
        self._buy_ticks: list[int] = []
        self._sell_ticks: list[int] = []
        self._hold_days = 0
        super().__init__(df, window_size, frame_bound, render_mode)
        self.trade_fee_bid_percent = 0.001
        self.trade_fee_ask_percent = 0.0003

    def reset(self, seed=None, options=None):
        self._buy_ticks = []
        self._sell_ticks = []
        self._hold_days = 0
        return super().reset(seed=seed, options=options)

    def _process_data(self):
        full_close = self.df.loc[:, "Close"].to_numpy()
        full_high = self.df.loc[:, "High"].to_numpy()
        full_low = self.df.loc[:, "Low"].to_numpy()
        full_open = self.df.loc[:, "Open"].to_numpy()

        ma5 = historical_moving_average(full_close, 5)
        ma10 = historical_moving_average(full_close, 10)
        ma20 = historical_moving_average(full_close, 20)
        rsi = compute_rsi(full_close, 14)
        _, _, macd_hist = compute_macd(full_close)
        bb_upper, _, bb_lower = compute_bollinger_bands(full_close, 20)

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = full_close[start:end]
        close = full_close[start:end]
        high = full_high[start:end]
        low = full_low[start:end]
        open_ = full_open[start:end]
        ma5 = ma5[start:end]
        ma10 = ma10[start:end]
        ma20 = ma20[start:end]
        rsi = rsi[start:end]
        macd_hist = macd_hist[start:end]
        bb_upper = bb_upper[start:end]
        bb_lower = bb_lower[start:end]

        eps = 1e-8
        pct_change = np.insert(np.diff(close), 0, 0)
        ma5_ratio = (close - ma5) / np.maximum(ma5, eps) * 100
        ma10_ratio = (close - ma10) / np.maximum(ma10, eps) * 100
        ma20_ratio = (close - ma20) / np.maximum(ma20, eps) * 100
        ma5_10_cross = (ma5 - ma10) / np.maximum(ma10, eps) * 100
        bb_width = bb_upper - bb_lower
        bb_position = np.clip((close - bb_lower) / np.maximum(bb_width, eps), 0, 1)
        rsi_norm = rsi / 100.0
        macd_max = np.maximum(np.abs(macd_hist).max(), eps)
        macd_norm = macd_hist / macd_max
        amplitude = (high - low) / np.maximum(close, eps) * 100
        body = (close - open_) / np.maximum(close, eps) * 100

        signal_features = np.column_stack(
            (
                pct_change,
                ma5_ratio,
                ma10_ratio,
                ma20_ratio,
                ma5_10_cross,
                rsi_norm,
                macd_norm,
                bb_position,
                amplitude,
                body,
            )
        )
        signal_features = np.nan_to_num(signal_features, nan=0.0, posinf=10.0, neginf=-10.0)
        signal_features = np.clip(signal_features, -100, 100)
        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _get_observation(self):
        end_tick = self._current_tick
        start_tick = end_tick - self.window_size
        return self.signal_features[start_tick:end_tick]

    def step(self, action):
        action = int(np.asarray(action).item())
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        if self._position == Positions.Long:
            self._hold_days += 1

        can_sell = self._position == Positions.Long and self._hold_days >= 1
        can_buy = self._position == Positions.Short

        trade = False
        if action == Actions.Buy.value and can_buy:
            trade = True
            self._buy_ticks.append(self._current_tick)
            self._hold_days = 0
        elif action == Actions.Sell.value and can_sell:
            trade = True
            self._sell_ticks.append(self._current_tick)

        step_reward = 0.0
        if trade and action == Actions.Sell.value:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            gross_return = (current_price - last_trade_price) / last_trade_price
            net_return = gross_return - self.trade_fee_ask_percent - self.trade_fee_bid_percent
            step_reward = net_return * 100

        self._total_reward += step_reward

        if trade:
            self._update_profit(action)
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        info["buy_ticks"] = self._buy_ticks.copy()
        info["sell_ticks"] = self._sell_ticks.copy()
        info["hold_days"] = self._hold_days

        return observation, step_reward, False, self._truncated, info

    def render_with_trades(self, title=None):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.prices, "b-", linewidth=1, label="Price", alpha=0.7)

        if self._buy_ticks:
            buy_prices = [self.prices[t] for t in self._buy_ticks]
            ax.scatter(
                self._buy_ticks,
                buy_prices,
                marker="^",
                c="green",
                s=150,
                label=f"BUY ({len(self._buy_ticks)})",
                zorder=5,
                edgecolors="black",
            )
            for i, tick in enumerate(self._buy_ticks):
                ax.annotate(
                    f"B{i + 1}\n{self.prices[tick]:.0f}",
                    (tick, self.prices[tick]),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha="center",
                    fontsize=8,
                    color="darkgreen",
                    fontweight="bold",
                )

        if self._sell_ticks:
            sell_prices = [self.prices[t] for t in self._sell_ticks]
            ax.scatter(
                self._sell_ticks,
                sell_prices,
                marker="v",
                c="red",
                s=150,
                label=f"SELL ({len(self._sell_ticks)})",
                zorder=5,
                edgecolors="black",
            )
            for i, tick in enumerate(self._sell_ticks):
                ax.annotate(
                    f"S{i + 1}\n{self.prices[tick]:.0f}",
                    (tick, self.prices[tick]),
                    textcoords="offset points",
                    xytext=(0, -20),
                    ha="center",
                    fontsize=8,
                    color="darkred",
                    fontweight="bold",
                )

        for buy_tick, sell_tick in zip(self._buy_ticks, self._sell_ticks):
            buy_price = self.prices[buy_tick]
            sell_price = self.prices[sell_tick]
            pnl = (sell_price - buy_price) / buy_price * 100
            color = "green" if pnl > 0 else "red"
            ax.plot(
                [buy_tick, sell_tick],
                [buy_price, sell_price],
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=2,
            )
            ax.annotate(
                f"{pnl:+.1f}%",
                ((buy_tick + sell_tick) / 2, (buy_price + sell_price) / 2),
                fontsize=9,
                color=color,
                fontweight="bold",
            )

        title = title or "Trading Strategy"
        ax.set_title(
            f"{title}\nTotal Profit: {self._total_profit:.4f} "
            f"({(self._total_profit - 1) * 100:.2f}%)",
            fontsize=14,
        )
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("NASDAQ-100")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
