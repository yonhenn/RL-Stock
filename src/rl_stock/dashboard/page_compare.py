"""Page: per-experiment algorithm comparison (saved metrics + live equity)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rl_stock.dashboard.figures import drawdown_lines, equity_lines, metric_bar
from rl_stock.dashboard.replay import EnvSettings, ReplayUnavailable
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
    st.sidebar.header("实验选择 Experiment")
    labels = [run.label for run in runs]
    choice = st.sidebar.selectbox("输出目录 Run", labels, index=0)
    run = runs[labels.index(choice)]

    if run.has_models:
        algos = ", ".join(algo for algo, _ in run.model_paths)
        st.sidebar.success(f"可回放模型 Models: {algos}")
    else:
        st.sidebar.info("此实验无已保存模型，仅展示指标。")

    st.sidebar.divider()
    st.sidebar.header("收益曲线 Equity replay")
    replay_enabled = st.sidebar.checkbox(
        "从模型回放收益曲线", value=run.has_models, disabled=not run.has_models
    )
    data_path = st.sidebar.text_input("数据文件 Data path", value="nasdaq100.xlsx")
    test_start = st.sidebar.text_input("测试起始日 Test start", value="2024-01-01")
    with st.sidebar.expander("高级：训练时的环境设置 (仅当回放曲线异常时调整)"):
        st.caption(
            "趋势特征/湍流门控会自动从模型观测维度识别。以下两项无法从模型推断，"
            "如训练时启用过残差趋势先验或regime限仓，请勾选以精确复现。"
        )
        use_prior = st.checkbox("残差趋势先验 use_trend_prior", value=False)
        use_regime = st.checkbox("regime限仓 regime_risk_scaling", value=False)

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
    cols[0].metric("最佳策略 Best strategy", str(best["Strategy"]), f"{best['Return Rate']:+.2f}%")
    if "Sharpe" in summary.columns:
        cols[1].metric("其 Sharpe", f"{best['Sharpe']:.3f}")
    if worst_dd is not None:
        cols[2].metric(
            "最小回撤 Lowest Max DD",
            str(worst_dd["Strategy"]),
            f"{worst_dd['Max DD']:.2f}%",
            delta_color="off",
        )

    metric = st.selectbox("对比指标 Metric", _METRIC_OPTIONS, index=0)
    if metric in summary.columns:
        st.plotly_chart(metric_bar(summary, metric), width="stretch")
    else:
        st.info(f"该实验的 summary.csv 不含 '{metric}' 列。")

    st.subheader("指标明细 Summary table")
    st.dataframe(_style_summary(ranked), width="stretch", hide_index=True)

    if run.returns_png is not None:
        with st.expander("查看已保存的收益对比图 (strategy_returns.png)"):
            st.image(str(run.returns_png), width="stretch")


def _equity(run: RunInfo, settings: EnvSettings) -> None:
    """Replay the models and render the interactive equity comparison."""
    if not run.has_models:
        st.info("此实验没有已保存的模型，无法回放收益曲线。")
        return
    try:
        with st.spinner("加载模型并回放收益曲线中…"):
            result = cached_replay(run, settings)
    except ReplayUnavailable as exc:
        st.warning(f"无法回放收益曲线：{exc}")
        if run.returns_png is not None:
            st.image(str(run.returns_png), width="stretch")
        return
    except Exception as exc:  # noqa: BLE001 - surface any load/replay error to the UI
        st.error(f"回放失败：{type(exc).__name__}: {exc}")
        return

    flags = result.detected_flags
    st.warning(
        "⚠️ 本页曲线由已保存模型在【当前】 nasdaq100.xlsx 上实时回放得到。"
        "该数据集在部分实验训练之后被扩展（现覆盖至 2026-06），"
        "因此这里的最终收益可能与“保存的评估指标”页不一致——两者针对的是不同的测试区间。"
        "本页用于查看模型在最新数据上的逐日表现，指标页才是各实验训练时的评估结果。"
    )
    st.caption(
        "自动识别的环境特征："
        f"趋势特征={flags.get('use_trend_features')}, "
        f"湍流门控={flags.get('use_turbulence_gate')}"
        "（趋势/湍流由观测维度自动识别；残差先验与 regime 限仓可在侧栏“高级”中匹配）。"
    )

    strategies = list(result.equity.columns)
    selected = st.multiselect("显示策略 Strategies", strategies, default=strategies)
    if not selected:
        st.info("请至少选择一个策略。")
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
    st.title("📈 强化学习交易算法收益对比")
    st.caption(
        "对比不同 RL 算法在 NASDAQ-100 上的收益表现。"
        "「保存的评估指标」来自各实验训练时写出的 summary.csv（权威、自洽）；"
        "「收益曲线」由已保存模型在当前数据上实时回放（可交互，但区间可能不同）。"
    )

    outputs_dir = st.sidebar.text_input("outputs 目录", value="outputs")
    runs = discover_runs(outputs_dir)
    if not runs:
        st.error(
            f"在 '{outputs_dir}' 下未找到任何包含 summary.csv 的实验目录。"
            " 请先运行 `python train.py` 生成结果。"
        )
        return

    run, settings, replay_enabled = _sidebar(runs)

    try:
        summary = parse_summary_csv(run.summary_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"读取 summary.csv 失败：{exc}")
        return

    st.subheader(f"实验 Run: `{run.label}`")
    metrics_tab, equity_tab = st.tabs(
        ["📊 保存的评估指标 Saved metrics", "📈 收益曲线（当前数据回放）Equity (live)"]
    )
    with metrics_tab:
        _metrics(summary, run)
    with equity_tab:
        if replay_enabled:
            _equity(run, settings)
        else:
            st.info("在左侧勾选“从模型回放收益曲线”以查看逐日收益曲线。")
