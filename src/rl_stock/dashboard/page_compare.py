"""Page: per-experiment algorithm comparison (saved metrics + live equity)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rl_stock.dashboard.figures import drawdown_lines, equity_lines, metric_bar
from rl_stock.dashboard.replay import RANDOM_LABEL, EnvSettings, ReplayUnavailable
from rl_stock.dashboard.runs import RunInfo, discover_runs, parse_summary_csv
from rl_stock.dashboard.session import cached_replay

_METRIC_OPTIONS = ("Return Rate", "Sharpe", "Calmar", "Max DD", "Average Profit")
_HIGHLIGHT_MAX = ("Return Rate", "Average Profit", "Highest", "Sharpe", "Calmar")
_HIGHLIGHT_MIN = ("Max DD", "Lowest", "Standard Deviation")


def _style_summary(summary: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Highlight the best value in each metric column."""
    numeric = summary.select_dtypes("number").columns
    styler = summary.style.format(precision=3, na_rep="-")
    max_cols = [c for c in _HIGHLIGHT_MAX if c in numeric]
    min_cols = [c for c in _HIGHLIGHT_MIN if c in numeric]
    if max_cols:
        styler = styler.highlight_max(subset=max_cols, color="#c6efce", axis=0)
    if min_cols:
        styler = styler.highlight_min(subset=min_cols, color="#c6efce", axis=0)
    return styler


def _sidebar(runs: list[RunInfo]) -> tuple[RunInfo, EnvSettings, bool]:
    """Sidebar controls; returns the chosen run, replay settings, and replay toggle."""
    st.sidebar.header("Experiment")
    labels = [run.label for run in runs]
    choice = st.sidebar.selectbox("Run", labels, index=0)
    run = runs[labels.index(choice)]

    if run.has_models:
        algos = ", ".join(algo for algo, _ in run.model_paths)
        st.sidebar.success(f"Replayable models: {algos}")
    else:
        st.sidebar.info("This run has no saved models; only saved metrics are available.")

    st.sidebar.divider()
    st.sidebar.header("Equity Replay")
    replay_enabled = st.sidebar.checkbox(
        "Replay equity curves from saved models",
        value=run.has_models,
        disabled=not run.has_models,
    )
    data_path = st.sidebar.text_input("Data file", value="nasdaq100.xlsx")
    test_start = st.sidebar.text_input("Test start date", value="2024-01-01")
    with st.sidebar.expander("Advanced: training environment settings"):
        st.caption(
            "Trend features and the turbulence gate are inferred automatically from the "
            "model observation width. Enable the following settings if they were used "
            "during training; they cannot be inferred from a saved model."
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
    return run, settings, replay_enabled


def _metrics(summary: pd.DataFrame, run: RunInfo) -> None:
    """Render the metric KPI row, bar chart, and ranked table."""
    ranked = summary.sort_values("Return Rate", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    worst_dd = summary.loc[summary["Max DD"].idxmin()] if "Max DD" in summary else None

    cols = st.columns(3)
    cols[0].metric("Best strategy", str(best["Strategy"]), f"{best['Return Rate']:+.2f}%")
    if "Sharpe" in summary.columns:
        cols[1].metric("Best strategy Sharpe", f"{best['Sharpe']:.3f}")
    if worst_dd is not None:
        cols[2].metric(
            "Lowest max drawdown",
            str(worst_dd["Strategy"]),
            f"{worst_dd['Max DD']:.2f}%",
            delta_color="off",
        )

    metric = st.selectbox("Metric", _METRIC_OPTIONS, index=0)
    if metric in summary.columns:
        st.plotly_chart(metric_bar(summary, metric), width="stretch")
    else:
        st.info(f"The summary.csv for this run does not contain '{metric}'.")

    st.subheader("Summary Table")
    st.dataframe(_style_summary(ranked), width="stretch", hide_index=True)

    if run.returns_png is not None:
        with st.expander("View saved return comparison (strategy_returns.png)"):
            st.image(str(run.returns_png), width="stretch")


def _equity(run: RunInfo, settings: EnvSettings) -> None:
    """Replay the models and render the interactive equity comparison."""
    if not run.has_models:
        st.info("This run has no saved models, so its equity curves cannot be replayed.")
        return
    try:
        with st.spinner("Loading models and replaying equity curves..."):
            result = cached_replay(run, settings)
    except ReplayUnavailable as exc:
        st.warning(f"Equity replay is unavailable: {exc}")
        if run.returns_png is not None:
            st.image(str(run.returns_png), width="stretch")
        return
    except Exception as exc:  # noqa: BLE001 - surface any load/replay error to the UI
        st.error(f"Replay failed: {type(exc).__name__}: {exc}")
        return

    flags = result.detected_flags
    st.warning(
        "These curves replay the saved models against the current nasdaq100.xlsx. "
        "The dataset was extended after some runs were trained and now covers data "
        "through June 2026, so final returns may differ from the saved metrics because "
        "the test windows differ. Use this view for daily performance on the latest data "
        "and the saved-metrics tab for each run's original evaluation results."
    )
    st.caption(
        "Detected environment features: "
        f"trend features={flags.get('use_trend_features')}, "
        f"turbulence gate={flags.get('use_turbulence_gate')}. "
        "These are inferred from observation width; match the residual prior and regime "
        "position limits in Advanced settings if they were enabled during training."
    )

    # Trade-view rule baselines share the cached replay result, but this page
    # keeps its existing scope: trained models plus the original Random baseline.
    model_strategies = {algo for algo, _ in run.model_paths}
    strategies = [
        strategy
        for strategy in result.equity.columns
        if strategy == RANDOM_LABEL or strategy in model_strategies
    ]
    selected = st.multiselect("Strategies", strategies, default=strategies)
    if not selected:
        st.info("Select at least one strategy.")
        return

    returns = result.cumulative_return_pct[selected]
    drawdowns = result.drawdown_pct[selected]

    finals = returns.iloc[-1].sort_values(ascending=False)
    kpi_cols = st.columns(len(finals))
    for col, (strategy, value) in zip(kpi_cols, finals.items()):
        col.metric(strategy, f"{value:+.2f}%")

    st.plotly_chart(equity_lines(returns), width="stretch")
    st.plotly_chart(drawdown_lines(drawdowns), width="stretch")


def render() -> None:
    """Entry point used by the multipage router."""
    st.title("RL Trading Algorithm Performance")
    st.caption(
        "Compare RL algorithm performance on the NASDAQ-100. Saved metrics come from "
        "the summary.csv produced during each experiment; live equity curves replay the "
        "saved models on the current dataset, so their test windows may differ."
    )

    outputs_dir = st.sidebar.text_input("Outputs directory", value="outputs")
    runs = discover_runs(outputs_dir)
    if not runs:
        st.error(
            f"No experiment directories containing summary.csv were found under "
            f"'{outputs_dir}'. Run `python train.py` to generate results."
        )
        return

    run, settings, replay_enabled = _sidebar(runs)

    try:
        summary = parse_summary_csv(run.summary_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Failed to read summary.csv: {exc}")
        return

    st.subheader(f"Run: `{run.label}`")
    metrics_tab, equity_tab = st.tabs(["Saved Metrics", "Live Equity Replay"])
    with metrics_tab:
        _metrics(summary, run)
    with equity_tab:
        if replay_enabled:
            _equity(run, settings)
        else:
            st.info("Enable equity replay in the sidebar to view daily equity curves.")
