"""Page: buy/sell points and holding time on the raw NASDAQ-100 curve.

One experiment and one trained or rule-based strategy are selected at a time;
their buy/sell markers and position path are shown on the raw price curve.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rl_stock.dashboard.figures import (
    cumulative_return_lines,
    position_lines,
    price_with_trades,
    trade_stats,
)
from rl_stock.dashboard.replay import (
    BASELINE_STRATEGIES,
    BUY_AT_DIP_LABEL,
    EnvSettings,
    ReplayUnavailable,
)
from rl_stock.dashboard.runs import KNOWN_ALGORITHMS, RunInfo, discover_runs
from rl_stock.dashboard.session import cached_replay

_COMBO_SEP = " / "
_DEFAULT_EXPERIMENT = "position_trend"
_DEFAULT_ALGORITHM = "PPO"


def _sidebar() -> tuple[list[RunInfo], EnvSettings]:
    """Data-source and env controls; returns discovered runs and replay settings."""
    st.sidebar.header("Trade Analysis")
    outputs_dir = st.sidebar.text_input("Outputs directory", value="outputs")
    data_path = st.sidebar.text_input("Data file", value="nasdaq100.xlsx")
    test_start = st.sidebar.text_input("Test start date", value="2024-01-01")
    with st.sidebar.expander("Advanced: training environment settings"):
        st.caption(
            "Enable these settings if the selected run used a residual trend prior or "
            "regime position limits during training."
        )
        use_prior = st.checkbox("Residual trend prior (use_trend_prior)", value=False)
        use_regime = st.checkbox(
            "Regime position limits (regime_risk_scaling)", value=False
        )

    settings = EnvSettings(
        data_path=Path(data_path),
        test_start_date=test_start,
        use_trend_prior=use_prior,
        regime_risk_scaling=use_regime,
    )
    return discover_runs(outputs_dir), settings


def _algorithms_in(experiments: list[RunInfo]) -> list[str]:
    """Available trained algorithms followed by the rule-based baselines."""
    present = {algo for run in experiments for algo, _ in run.model_paths}
    if not present:
        return []
    trained = [algo for algo in KNOWN_ALGORITHMS if algo in present]
    return [*trained, *BASELINE_STRATEGIES]


def _pairs(experiments: list[RunInfo], algorithms: list[str]) -> list[tuple[RunInfo, str]]:
    """Every existing (experiment, algorithm) pair, experiment-major for stable colours."""
    pairs: list[tuple[RunInfo, str]] = []
    for run in experiments:
        run_algos = {algo for algo, _ in run.model_paths}
        for algo in algorithms:
            if algo in run_algos or algo in BASELINE_STRATEGIES:
                pairs.append((run, algo))
    return pairs


def _collect(
    pairs: list[tuple[RunInfo, str]], settings: EnvSettings
) -> tuple[pd.Series | None, list[tuple[str, pd.Series]], list[tuple[str, pd.Series]]]:
    """Replay each pair (grouping by run so each run loads once) and gather paths.

    Returns the shared price series, an ordered ``(label, position)`` list, and a
    parallel ``(label, cumulative_return_pct)`` list in the same order (so the
    returns chart, position chart, and trade table all agree on combo identity).
    Runs that cannot be replayed (e.g. the discrete T+1 env) are reported once
    and skipped.
    """
    results: dict[str, object] = {}
    failed: set[str] = set()
    price: pd.Series | None = None
    combos: list[tuple[str, pd.Series]] = []
    returns: list[tuple[str, pd.Series]] = []

    for run, algo in pairs:
        if run.label in failed:
            continue
        if run.label not in results:
            try:
                with st.spinner(f"Replaying {run.label}..."):
                    results[run.label] = cached_replay(run, settings)
            except ReplayUnavailable as exc:
                st.warning(f"{run.label}: trade replay is unavailable ({exc})")
                failed.add(run.label)
                continue
            except Exception as exc:  # noqa: BLE001 - surface any load/replay error
                st.error(f"{run.label}: replay failed: {type(exc).__name__}: {exc}")
                failed.add(run.label)
                continue

        result = results[run.label]
        if price is None:
            price = result.price
        if algo not in result.positions.columns:
            continue
        label = f"{run.label}{_COMBO_SEP}{algo}"
        combos.append((label, result.positions[algo].reindex(price.index)))
        returns.append((label, result.cumulative_return_pct[algo].reindex(price.index)))

    return price, combos, returns


def render() -> None:
    """Entry point used by the multipage router."""
    st.title("Buy/Sell & Holding Analysis on the Raw NASDAQ-100")
    st.caption(
        "Display buy/sell points and holding periods for the selected experiment and "
        "algorithm (or baseline) on the raw NASDAQ-100 price curve. In a continuous-"
        "position environment, moving from zero to a positive position is a buy, "
        "returning to zero is a sell, and the uninterrupted interval is the holding period."
    )

    runs, settings = _sidebar()
    if not runs:
        st.error("No experiment directories were found. Run `python train.py` first.")
        return

    run_by_label = {run.label: run for run in runs}
    experiment_labels = list(run_by_label)
    experiment_index = (
        experiment_labels.index(_DEFAULT_EXPERIMENT)
        if _DEFAULT_EXPERIMENT in run_by_label
        else 0
    )
    experiment = st.selectbox(
        "Experiment",
        experiment_labels,
        index=experiment_index,
    )
    selected_runs = [run_by_label[experiment]]

    # Axis 2: algorithms available across the chosen experiments.
    available = _algorithms_in(selected_runs)
    if not available:
        st.info("The selected experiment has no replayable algorithm models.")
        return
    algorithm_index = (
        available.index(_DEFAULT_ALGORITHM)
        if _DEFAULT_ALGORITHM in available
        else 0
    )
    algorithm = st.selectbox("Algorithm", available, index=algorithm_index)
    st.caption(
        f"{BUY_AT_DIP_LABEL}: buy when the close is 2% below its 20-day moving average "
        "and sell when it recovers to that average. Random uses random positions; "
        "Buy and Hold remains fully invested."
    )

    pairs = _pairs(selected_runs, [algorithm])
    if not pairs:
        st.info("The selected experiment and algorithm do not form a valid combination.")
        return

    price, combos, returns = _collect(pairs, settings)
    if price is None or not combos:
        st.info(
            "The selected combination cannot be replayed. Trade markers are not yet "
            "supported for the discrete T+1 environment."
        )
        return

    span = f"{price.index.min():%Y-%m-%d} to {price.index.max():%Y-%m-%d}"
    st.caption(f"{len(combos)} selection - Test window: {span}")

    # Put the summary directly below the selectors so the headline performance
    # and risk metrics are visible before the detailed charts.
    return_paths = dict(returns)
    st.subheader("Trade Summary")
    st.dataframe(
        trade_stats(combos, return_paths),
        width="stretch",
        hide_index=True,
    )

    st.plotly_chart(price_with_trades(price, combos), width="stretch")
    st.plotly_chart(position_lines(combos), width="stretch")

    st.subheader("Cumulative Return")
    st.caption(
        "Cumulative return (%) over the test window. Colors match the trade-marker and "
        "position charts above."
    )
    st.plotly_chart(cumulative_return_lines(returns), width="stretch")
