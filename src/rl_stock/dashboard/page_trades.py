"""Page: buy/sell points and holding time on the raw NASDAQ-100 curve.

One shared price line, with any number of parameter combinations overlaid as
buy/sell markers plus position ribbons. Experiment and algorithm are chosen on
separate axes, so the same algorithm can be compared across experiments and vice
versa; every existing (experiment, algorithm) pair becomes one overlaid combo.
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
from rl_stock.dashboard.replay import RANDOM_LABEL, EnvSettings, ReplayUnavailable
from rl_stock.dashboard.runs import KNOWN_ALGORITHMS, RunInfo, discover_runs
from rl_stock.dashboard.session import cached_replay

_COMBO_SEP = " · "


def _sidebar() -> tuple[list[RunInfo], EnvSettings]:
    """Data-source and env controls; returns discovered runs and replay settings."""
    st.sidebar.header("买卖点分析 Trade view")
    outputs_dir = st.sidebar.text_input("outputs 目录", value="outputs")
    data_path = st.sidebar.text_input("数据文件 Data path", value="nasdaq100.xlsx")
    test_start = st.sidebar.text_input("测试起始日 Test start", value="2024-01-01")
    with st.sidebar.expander("高级：训练时的环境设置"):
        st.caption("如某实验训练时启用过残差趋势先验 / regime 限仓，勾选以精确复现其买卖行为。")
        use_prior = st.checkbox("残差趋势先验 use_trend_prior", value=False)
        use_regime = st.checkbox("regime限仓 regime_risk_scaling", value=False)

    settings = EnvSettings(
        data_path=Path(data_path),
        test_start_date=test_start,
        use_trend_prior=use_prior,
        regime_risk_scaling=use_regime,
    )
    return discover_runs(outputs_dir), settings


def _algorithms_in(experiments: list[RunInfo]) -> list[str]:
    """Union of algorithms across the selected experiments, in canonical order."""
    present = {algo for run in experiments for algo, _ in run.model_paths}
    return [algo for algo in KNOWN_ALGORITHMS if algo in present]


def _pairs(experiments: list[RunInfo], algorithms: list[str]) -> list[tuple[RunInfo, str]]:
    """Every existing (experiment, algorithm) pair, experiment-major for stable colours."""
    pairs: list[tuple[RunInfo, str]] = []
    for run in experiments:
        run_algos = {algo for algo, _ in run.model_paths}
        for algo in algorithms:
            if algo in run_algos:
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
                with st.spinner(f"回放 {run.label} …"):
                    results[run.label] = cached_replay(run, settings)
            except ReplayUnavailable as exc:
                st.warning(f"{run.label}: 无法回放买卖点（{exc}）")
                failed.add(run.label)
                continue
            except Exception as exc:  # noqa: BLE001 - surface any load/replay error
                st.error(f"{run.label}: 回放失败 {type(exc).__name__}: {exc}")
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
    st.title("🎯 买卖点分析 Buy/Sell & holding on the raw NASDAQ-100")
    st.caption(
        "在原始 NASDAQ-100 价格曲线上叠加不同【实验】与【算法】组合的买点、卖点与持仓时间。"
        "实验与算法分开选，可同实验比算法、也可同算法比实验，所有存在的组合会叠在同一张图中。"
        "连续仓位环境下：仓位由 0 变为持有=买入，回到 0=卖出，连续持有天数=持有时间。"
    )

    runs, settings = _sidebar()
    if not runs:
        st.error("未找到任何实验目录，请先运行 `python train.py`。")
        return

    run_by_label = {run.label: run for run in runs}

    # Axis 1: experiments (runs are discovered richest-first, so the default leads).
    exp_labels = list(run_by_label)
    experiments = st.multiselect("实验 Experiments", exp_labels, default=exp_labels[:1])
    if not experiments:
        st.info("请至少选择一个实验。")
        return
    selected_runs = [run_by_label[label] for label in experiments]

    # Axis 2: algorithms available across the chosen experiments.
    available = _algorithms_in(selected_runs)
    if not available:
        st.info("所选实验没有可回放的算法模型。")
        return
    algo_default = [a for a in available if a in ("SAC", "PPO")] or available
    algorithms = st.multiselect("算法 Algorithms", available, default=algo_default)
    if not algorithms:
        st.info("请至少选择一个算法。")
        return

    pairs = _pairs(selected_runs, algorithms)
    if not pairs:
        st.info("所选实验与算法没有交集，请调整选择。")
        return

    price, combos, returns = _collect(pairs, settings)
    if price is None or not combos:
        st.info("所选组合均无法回放（离散 T+1 实验暂不支持买卖点），请换一组。")
        return

    span = f"{price.index.min():%Y-%m-%d} … {price.index.max():%Y-%m-%d}"
    st.caption(
        f"共 {len(combos)} 个组合 · 测试区间 Test window: {span}"
        f"（{RANDOM_LABEL} 基准未纳入买卖点对比）"
    )

    st.plotly_chart(price_with_trades(price, combos), width="stretch")
    st.plotly_chart(position_lines(combos), width="stretch")

    finals = {label: float(series.iloc[-1]) for label, series in returns}
    st.subheader("交易统计 Trade summary")
    st.dataframe(trade_stats(combos, finals), width="stretch", hide_index=True)

    st.subheader("累计收益对比 Cumulative return")
    st.caption(
        "每个【实验 · 算法】组合在测试区间内的累计收益(%)，颜色与上方买卖点、持仓图一致。"
    )
    st.plotly_chart(cumulative_return_lines(returns), width="stretch")
