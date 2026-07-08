"""Plotly figure builders shared by the Streamlit dashboard.

Kept free of Streamlit imports so the figures can be unit-tested or reused in
notebooks. All builders return ``plotly.graph_objects.Figure`` instances and
never mutate their inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# A position below this ratio counts as "flat" for trade-event detection.
POSITION_EPS = 1e-3

# Buy/sell are encoded by marker fill colour (universal green/red convention).
BUY_COLOR = "#2CA02C"
SELL_COLOR = "#D62728"

# Qualitative palette for combos (experiment x algorithm) on the trade chart,
# where a single algorithm may appear across several experiments. On the price
# chart it is the marker *outline* (algorithm identity); on the position chart it
# is the line colour. A distinct marker *shape* per combo reinforces identity so
# the same algorithm across experiments stays separable.
_COMBO_PALETTE: tuple[str, ...] = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
)
_MARKER_SHAPES: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "star",
    "pentagon",
    "hexagon",
    "cross",
    "x",
    "star-diamond",
)


def marker_shape(index: int) -> str:
    """Stable marker shape for the n-th selected combo on the trade chart."""
    return _MARKER_SHAPES[index % len(_MARKER_SHAPES)]

# Stable per-strategy colours so a strategy reads the same across every chart.
STRATEGY_COLORS: dict[str, str] = {
    "Random": "#9AA0A6",
    "SAC": "#4ECDC4",
    "PPO": "#45B7D1",
    "TD3": "#F4A261",
    "DDPG": "#9B5DE5",
    "A2C": "#FF6B6B",
}
_FALLBACK_COLOR = "#7F7F7F"

# Metrics where a larger value is better (used for value-label colouring).
_HIGHER_IS_BETTER: dict[str, bool] = {
    "Return Rate": True,
    "Average Profit": True,
    "Sharpe": True,
    "Calmar": True,
    "Max DD": False,
}


def color_for(strategy: str) -> str:
    """Return the stable colour for a strategy label."""
    return STRATEGY_COLORS.get(strategy, _FALLBACK_COLOR)


def _unit_for(metric: str) -> str:
    return "%" if metric in ("Return Rate", "Max DD") else ""


def metric_bar(summary: pd.DataFrame, metric: str) -> go.Figure:
    """Bar chart of one metric across strategies, coloured per strategy."""
    if metric not in summary.columns:
        raise ValueError(f"metric {metric!r} not in summary columns")

    data = summary[["Strategy", metric]].dropna(subset=[metric])
    strategies = data["Strategy"].tolist()
    values = data[metric].tolist()
    unit = _unit_for(metric)
    colors = [color_for(name) for name in strategies]
    text = [f"{value:+.2f}{unit}" if unit else f"{value:.3f}" for value in values]

    fig = go.Figure(
        go.Bar(
            x=strategies,
            y=values,
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + f"{metric}: " + "%{y:.3f}" + unit + "<extra></extra>",
        )
    )
    better = _HIGHER_IS_BETTER.get(metric)
    subtitle = ""
    if better is not None:
        subtitle = "  (higher is better)" if better else "  (lower is better)"
    fig.update_layout(
        title=f"{metric} by strategy{subtitle}",
        xaxis_title=None,
        yaxis_title=f"{metric}{f' ({unit})' if unit else ''}",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        showlegend=False,
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="rgba(0,0,0,0.25)")
    return fig


def equity_lines(cumulative_return_pct: pd.DataFrame) -> go.Figure:
    """Overlay of each strategy's cumulative return (%) over the test window."""
    fig = go.Figure()
    for strategy in cumulative_return_pct.columns:
        series = cumulative_return_pct[strategy]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=strategy,
                line=dict(color=color_for(strategy), width=2),
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>"
                "Return: %{y:+.2f}%<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.35)")
    fig.update_layout(
        title="Cumulative return over the test window",
        xaxis_title=None,
        yaxis_title="Cumulative return (%)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def drawdown_lines(drawdown_pct: pd.DataFrame) -> go.Figure:
    """Overlay of each strategy's drawdown (%) from its running peak."""
    fig = go.Figure()
    for strategy in drawdown_pct.columns:
        series = drawdown_pct[strategy]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=strategy,
                line=dict(color=color_for(strategy), width=1.5),
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>"
                "Drawdown: %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="Drawdown from running peak",
        xaxis_title=None,
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def combo_color(index: int) -> str:
    """Stable colour for the n-th selected combo on the trade chart."""
    return _COMBO_PALETTE[index % len(_COMBO_PALETTE)]


def derive_trades(position: pd.Series) -> dict[str, list]:
    """Turn a continuous position path into discrete entry/exit trade events.

    A *buy* is the first day exposure rises above :data:`POSITION_EPS` from flat;
    a *sell* is the first day it returns to flat. Holding spans are the number of
    trading days each position is carried (the final open position counts up to
    the last day). Returns entry/exit dates and holding-day lengths.
    """
    values = position.to_numpy(dtype=float)
    dates = list(position.index)
    buys: list = []
    sells: list = []
    holding_days: list[int] = []

    in_position = False
    entry_i = 0
    for i, value in enumerate(values):
        held = value > POSITION_EPS
        if held and not in_position:
            buys.append(dates[i])
            entry_i = i
            in_position = True
        elif not held and in_position:
            sells.append(dates[i])
            holding_days.append(i - entry_i)
            in_position = False
    if in_position:
        holding_days.append((len(values) - 1) - entry_i)

    return {"buys": buys, "sells": sells, "holding_days": holding_days}


def price_with_trades(price: pd.Series, combos: list[tuple[str, pd.Series]]) -> go.Figure:
    """Raw NASDAQ-100 price with each combo's buy/sell markers overlaid.

    Triple encoding so both requested distinctions are visible at once:
    marker *fill* = action (green buy / red sell), marker *outline* = algorithm
    colour (matched to the position chart), marker *shape* = combo identity.
    ``combos`` is an ordered ``(label, position_series)`` list.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            mode="lines",
            name=price.name or "NASDAQ-100",
            line=dict(color="#9AA0A6", width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>价格 %{y:.0f}<extra></extra>",
        )
    )

    # Action legend (fill colour): what green vs red means.
    for name, fill in (("买入 Buy", BUY_COLOR), ("卖出 Sell", SELL_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=name, legendgroup="_action",
                marker=dict(color=fill, symbol="square", size=11),
                hoverinfo="skip",
            )
        )

    for index, (label, position) in enumerate(combos):
        algo_color = combo_color(index)
        shape = marker_shape(index)
        trades = derive_trades(position.reindex(price.index))

        # Algorithm legend (outline colour + shape): which combo is which.
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=label, legendgroup=label,
                marker=dict(
                    color=algo_color, symbol=shape, size=11,
                    line=dict(color=algo_color, width=1),
                ),
                hoverinfo="skip",
            )
        )

        for action, fill, dates in (
            ("买入", BUY_COLOR, trades["buys"]),
            ("卖出", SELL_COLOR, trades["sells"]),
        ):
            if not dates:
                continue
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=price.reindex(dates).values,
                    mode="markers",
                    legendgroup=label,
                    showlegend=False,
                    marker=dict(
                        color=fill, symbol=shape, size=11,
                        line=dict(color=algo_color, width=1.4),
                    ),
                    hovertemplate=(
                        f"<b>{label}</b> {action}<br>%{{x|%Y-%m-%d}}<br>%{{y:.0f}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="NASDAQ-100 价格与买卖点 · 绿=买 红=卖，描边色/形状=算法",
        yaxis_title="指数价格 Price",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=90),
        height=560,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
    )
    return fig


def position_lines(combos: list[tuple[str, pd.Series]]) -> go.Figure:
    """Target position ratio (0..1) over time for each combo, colours matched."""
    fig = go.Figure()
    for index, (label, position) in enumerate(combos):
        fig.add_trace(
            go.Scatter(
                x=position.index,
                y=position.values,
                mode="lines",
                name=label,
                line=dict(color=combo_color(index), width=1.6),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>仓位 %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="持仓比例 Position ratio over time",
        yaxis_title="仓位 Position (0-1)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=70),
        height=340,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def cumulative_return_lines(combos: list[tuple[str, pd.Series]]) -> go.Figure:
    """Cumulative return (%) over time for each combo, colours matched.

    ``combos`` is an ordered ``(label, cumulative_return_pct_series)`` list; the
    n-th combo reuses :func:`combo_color`, so a combo reads as the same colour on
    this chart, :func:`price_with_trades`, and :func:`position_lines`. A dashed
    0% baseline marks break-even. Unlike :func:`equity_lines` (coloured by
    strategy name for the per-experiment page), this keys off combo order so the
    same algorithm stays distinct across experiments.
    """
    fig = go.Figure()
    for index, (label, returns) in enumerate(combos):
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns.values,
                mode="lines",
                name=label,
                line=dict(color=combo_color(index), width=1.8),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>"
                "累计收益 %{y:+.2f}%<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.35)")
    fig.update_layout(
        title="累计收益对比 Cumulative return by combo (%)",
        yaxis_title="累计收益 Cumulative return (%)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=70),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    return fig


def trade_stats(combos: list[tuple[str, pd.Series]], finals: dict[str, float]) -> pd.DataFrame:
    """Per-combo trade summary: counts, holding time, time-in-market, final return."""
    rows: list[dict] = []
    for label, position in combos:
        trades = derive_trades(position)
        holding = trades["holding_days"]
        held_days = int((position.to_numpy(dtype=float) > POSITION_EPS).sum())
        total_days = int(len(position))
        rows.append(
            {
                "参数组合 Combo": label,
                "买入次数 Buys": len(trades["buys"]),
                "卖出次数 Sells": len(trades["sells"]),
                "平均持有(天) Avg hold": round(float(np.mean(holding)), 1) if holding else 0.0,
                "最长持有(天) Max hold": max(holding) if holding else 0,
                "持仓占比 In-market": f"{(held_days / total_days * 100) if total_days else 0:.0f}%",
                "最终收益 Return": f"{finals.get(label, float('nan')):+.2f}%",
            }
        )
    return pd.DataFrame(rows)
