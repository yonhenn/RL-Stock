"""Unit tests for the dashboard's pure data-handling logic.

These cover the parts that run without the heavy RL stack: summary parsing,
run discovery, and the observation-width -> feature-flag inference. The live
model replay is exercised via the app, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rl_stock.dashboard.figures import (
    combo_color,
    cumulative_return_lines,
    derive_trades,
    trade_stats,
)
from rl_stock.dashboard.replay import infer_env_flags
from rl_stock.dashboard.runs import discover_runs, parse_summary_csv

_SUMMARY = (
    "Strategy,Average Profit,Return Rate,Highest,Lowest,Standard Deviation,Sharpe,Max DD,Calmar\n"
    "Random,0.8814,-11.86%,1.0273,0.8324,0.0463,-1.104,15.19%,-0.325\n"
    "SAC,0.9513,-4.87%,0.9513,0.9513,0.0000,-0.211,9.94%,-0.172\n"
    "PPO,1.0287,2.87%,1.0287,1.0287,0.0000,0.173,11.71%,0.134\n"
)


# --------------------------------------------------------------------------- #
# parse_summary_csv
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_parse_summary_converts_percent_columns(tmp_path: Path):
    path = tmp_path / "summary.csv"
    path.write_text(_SUMMARY)

    frame = parse_summary_csv(path)

    assert list(frame["Strategy"]) == ["Random", "SAC", "PPO"]
    # Percent strings become numeric percentage points.
    assert frame.loc[frame["Strategy"] == "PPO", "Return Rate"].item() == pytest.approx(2.87)
    assert frame.loc[frame["Strategy"] == "SAC", "Max DD"].item() == pytest.approx(9.94)
    # Plain numeric columns are floats, not strings.
    assert frame["Sharpe"].dtype.kind == "f"
    assert frame["Return Rate"].dtype.kind == "f"


@pytest.mark.unit
def test_parse_summary_tolerates_missing_optional_columns(tmp_path: Path):
    path = tmp_path / "summary.csv"
    path.write_text("Strategy,Return Rate\nSAC,5.00%\nPPO,-1.00%\n")

    frame = parse_summary_csv(path)

    assert "Sharpe" not in frame.columns
    assert frame.loc[frame["Strategy"] == "SAC", "Return Rate"].item() == pytest.approx(5.0)


@pytest.mark.unit
def test_parse_summary_missing_strategy_column_raises(tmp_path: Path):
    path = tmp_path / "summary.csv"
    path.write_text("Foo,Bar\n1,2\n")
    with pytest.raises(ValueError):
        parse_summary_csv(path)


@pytest.mark.unit
def test_parse_summary_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_summary_csv(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# discover_runs
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_discover_runs_finds_root_and_children_and_orders_models_first(tmp_path: Path):
    # Root-level run without models.
    (tmp_path / "summary.csv").write_text(_SUMMARY)
    # Child run with a model archive.
    child = tmp_path / "with_models"
    child.mkdir()
    (child / "summary.csv").write_text(_SUMMARY)
    (child / "sac.zip").write_bytes(b"stub")
    (child / "ppo.zip").write_bytes(b"stub")
    (child / "strategy_returns.png").write_bytes(b"stub")
    # A folder without a summary is ignored.
    (tmp_path / "empty").mkdir()

    runs = discover_runs(tmp_path)

    labels = [run.label for run in runs]
    assert len(runs) == 2
    # Model-bearing run sorts first.
    assert runs[0].label == "with_models"
    assert runs[0].has_models
    assert {algo for algo, _ in runs[0].model_paths} == {"SAC", "PPO"}
    assert runs[0].returns_png is not None
    assert any(label.endswith("(root)") for label in labels)
    assert not runs[1].has_models


@pytest.mark.unit
def test_discover_runs_missing_dir_returns_empty(tmp_path: Path):
    assert discover_runs(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------- #
# infer_env_flags
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "width,trend,turb",
    [
        (19, False, False),
        (20, False, True),
        (25, True, False),
        (26, True, True),
    ],
)
def test_infer_env_flags_known_widths(width: int, trend: bool, turb: bool):
    flags = infer_env_flags(width)
    assert flags == {"use_trend_features": trend, "use_turbulence_gate": turb}


@pytest.mark.unit
def test_infer_env_flags_unknown_width_returns_none():
    assert infer_env_flags(23) is None


# --------------------------------------------------------------------------- #
# derive_trades / trade_stats
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_derive_trades_detects_entries_exits_and_holding():
    # flat, flat, long, long, flat, long, flat
    position = pd.Series([0.0, 0.0, 0.5, 0.5, 0.0, 0.8, 0.0])
    trades = derive_trades(position)
    assert trades["buys"] == [2, 5]
    assert trades["sells"] == [4, 6]
    assert trades["holding_days"] == [2, 1]


@pytest.mark.unit
def test_derive_trades_counts_open_position_at_end():
    position = pd.Series([0.0, 0.6, 0.6])  # still holding on the last day
    trades = derive_trades(position)
    assert trades["buys"] == [1]
    assert trades["sells"] == []  # never exited
    assert trades["holding_days"] == [1]


@pytest.mark.unit
def test_trade_stats_summarizes_each_combo():
    position = pd.Series([0.0, 0.5, 0.5, 0.0])
    stats = trade_stats([("regime_study · SAC", position)], {"regime_study · SAC": 7.6})
    row = stats.iloc[0]
    assert row["买入次数 Buys"] == 1
    assert row["卖出次数 Sells"] == 1
    assert row["最终收益 Return"] == "+7.60%"
    assert row["持仓占比 In-market"] == "50%"


# --------------------------------------------------------------------------- #
# cumulative_return_lines
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cumulative_return_lines_one_trace_per_combo_with_matched_colours():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    combos = [
        ("regime_study · PPO", pd.Series([0.0, 1.5, 3.2], index=idx)),
        ("regime_study · SAC", pd.Series([0.0, -0.5, 2.1], index=idx)),
    ]

    fig = cumulative_return_lines(combos)

    # One line per combo; the 0% baseline is a shape, not a data trace.
    assert len(fig.data) == len(combos)
    assert [trace.name for trace in fig.data] == [label for label, _ in combos]
    # Colours track the shared combo palette used by the price/position charts,
    # so a combo reads the same across all three charts on the Trades page.
    assert fig.data[0].line.color == combo_color(0)
    assert fig.data[1].line.color == combo_color(1)
    # Return values are carried through untouched.
    assert list(fig.data[1].y) == [0.0, -0.5, 2.1]


@pytest.mark.unit
def test_cumulative_return_lines_handles_no_combos():
    fig = cumulative_return_lines([])
    assert len(fig.data) == 0
