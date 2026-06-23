"""Generate a self-contained HTML comparing the three RL trading improvements.

Compares strategies on the same test window, all starting from the same initial
capital:

  * Buy & Hold                  -- passive benchmark
  * Discrete baseline           -- legacy buy/sell env (improvement #1 "before")
  * Continuous <algo> (base)    -- continuous position management (improvement #1)
  * Continuous <algo> + Trend   -- continuous + trend-aware state features (#2)

The continuous configs are paired *per algorithm* (PPO base vs PPO+Trend, SAC
base vs SAC+Trend) so the trend-feature contribution is isolated rather than
confounded by the choice of algorithm. It loads PRE-TRAINED models from the
output directories (it does not train) and reuses the evaluation/plotting
helpers in ``generate_feature_presentation_images.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_feature_presentation_images as fp  # noqa: E402
from rl_stock.data import load_market_data, split_frame_bounds  # noqa: E402
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig  # noqa: E402
from rl_stock.risk import (  # noqa: E402
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


BENCHMARK_COLOR = "#8c8c8c"
DISCRETE_COLOR = "#d8a657"
# (base color, trend color) per continuous algorithm.
ALGO_COLORS = {
    "PPO": ("#5a7fa0", "#56b4e9"),
    "SAC": ("#6f8f6f", "#7ee787"),
    "TD3": ("#8f6f9f", "#b07ed1"),
    "A2C": ("#a0795a", "#e9a456"),
}
CONTINUOUS_ALGO_ORDER = ["PPO", "SAC", "TD3", "A2C"]
TREND_FEATURE_CARDS = [
    ("SMA-20 slope", "Rolling slope of the 20-day moving average -> trend direction."),
    ("Bollinger width", "Normalized band width -> volatility / trend-strength regime."),
    ("ADX", "Average Directional Index -> raw trend strength, direction-agnostic."),
    ("MACD cross", "Golden/death-cross regime -> rule-based trading prior."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "nasdaq100.xlsx")
    parser.add_argument("--t1-dir", type=Path, default=ROOT / "outputs" / "t1")
    parser.add_argument("--position-dir", type=Path, default=ROOT / "outputs" / "position")
    parser.add_argument("--trend-dir", type=Path, default=ROOT / "outputs" / "position_trend")
    parser.add_argument("--html", type=Path, default=ROOT / "improvements_comparison.html")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "outputs" / "improvement_comparison" / "figures",
    )
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--num-eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def build_trend_test_env(df: pd.DataFrame, args: argparse.Namespace) -> PositionMgmtEnv:
    """Position env with trend features ON (matches the 23-feature trend models)."""
    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
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
        use_trend_features=True,
    )
    return PositionMgmtEnv(
        df=df, window_size=args.window_size, frame_bound=test_bound, risk_config=risk
    )


def evaluate_trend_dir(
    output_dir: Path, df: pd.DataFrame, args: argparse.Namespace
) -> tuple[list[fp.StrategyEval], PositionMgmtEnv]:
    summary = fp.load_summary(output_dir, "position")
    env = build_trend_test_env(df, args)
    results = []
    for strategy in fp.sorted_strategies(summary):
        model = None if strategy == "Random" else fp.load_model(strategy, output_dir, env)
        results.append(fp.evaluate_strategy("position", strategy, model, env, args))
    return results, env


def position_strategy_map(
    output_dir: Path, df: pd.DataFrame, args: argparse.Namespace, trend: bool
) -> tuple[dict[str, fp.StrategyEval], Any, str | None]:
    """Return {algo: StrategyEval} for the trained (non-random) continuous models."""
    label = "Continuous + Trend" if trend else "Continuous (base)"
    try:
        if trend:
            results, env = evaluate_trend_dir(output_dir, df, args)
        else:
            results = fp.evaluate_output_dir("position", output_dir, df, args)
            env = fp.build_test_env("position", df, args)
        mapping = {r.strategy: r for r in results if r.strategy != "Random"}
        if not mapping:
            raise ValueError("no trained strategies in summary.csv")
        return mapping, env, None
    except (FileNotFoundError, ValueError) as exc:
        msg = f"{label}: skipped ({exc})"
        if trend:
            msg += (
                ". Train it with: python train.py --env-type position "
                "--use-trend-features --initial-cash 100000 --output-dir outputs/position_trend"
            )
        return {}, None, msg


def to_yuan(curve: np.ndarray, initial_cash: float) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return np.asarray([initial_cash])
    base = curve[0] if curve[0] != 0 else 1.0
    return curve / base * initial_cash


def buy_hold_series(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, float]]:
    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    close = df["Close"].iloc[test_bound[0] : test_bound[1]].to_numpy(dtype=float)
    curve = close / close[0] if close.size else np.asarray([1.0])
    returns = np.diff(curve) / np.maximum(curve[:-1], 1e-8)
    metrics = {
        "return": float(curve[-1] - 1.0),
        "sharpe": compute_sharpe_ratio(returns),
        "sortino": compute_sortino_ratio(returns),
        "max_drawdown": compute_max_drawdown(curve),
        "calmar": compute_calmar_ratio(returns, curve),
        "win_rate": float("nan"),
        "turnover": float("nan"),
    }
    return curve, metrics


def continuous_entry(
    algo: str, group: str, result: fp.StrategyEval, env: Any, args: argparse.Namespace
) -> dict[str, Any]:
    base_color, trend_color = ALGO_COLORS.get(algo, ("#9aa3b2", "#f2f4f8"))
    is_trend = group == "trend"
    return {
        "key": f"{algo}_{group}",
        "name": f"Continuous {algo}" + (" + Trend" if is_trend else " (base)"),
        "color": trend_color if is_trend else base_color,
        "linestyle": "-" if is_trend else "--",
        "linewidth": 2.1 if is_trend else 1.6,
        "curve": to_yuan(fp.representative_curve(result, env), args.initial_cash),
        "metrics": result.metrics,
        "group": group,
    }


def discrete_entry(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict | None, str | None]:
    try:
        results = fp.evaluate_output_dir("t1", args.t1_dir, df, args)
        best = fp.choose_best(results, "t1")
        # The legacy discrete (gym-anytrading) env reports realized trade P&L but
        # does not keep a mark-to-market equity curve, so its risk metrics are not
        # comparable. Keep the (real) return, mark the rest n/a, and keep it out of
        # the equity chart so a flat reconstruction does not mislead.
        metrics = dict(best.metrics)
        for key in ("sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "turnover"):
            metrics[key] = float("nan")
        return {
            "key": "discrete",
            "name": f"Discrete baseline ({best.strategy})",
            "color": DISCRETE_COLOR,
            "linestyle": "-",
            "linewidth": 1.6,
            "curve": np.asarray([args.initial_cash]),
            "metrics": metrics,
            "group": "discrete",
            "in_chart": False,
        }, None
    except (FileNotFoundError, ValueError) as exc:
        return None, f"Discrete baseline: skipped ({exc})"


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def yuan_formatter(initial_cash: float) -> FuncFormatter:
    if initial_cash >= 1_000_000:
        return FuncFormatter(lambda x, _: f"¥{x / 1e6:.2f}M")
    return FuncFormatter(lambda x, _: f"¥{x / 1e3:.0f}k")


def generate_equity_chart(
    entries: list[dict[str, Any]], initial_cash: float, path: Path
) -> None:
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(11.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08},
    )
    for entry in entries:
        if not entry.get("in_chart", True):
            continue
        curve = entry["curve"]
        if curve.size == 0:
            continue
        x = np.arange(curve.size)
        ax_eq.plot(
            x,
            curve,
            label=entry["name"],
            color=entry["color"],
            linewidth=entry["linewidth"],
            linestyle=entry["linestyle"],
        )
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        ax_dd.plot(
            x,
            drawdown * 100.0,
            color=entry["color"],
            linewidth=max(entry["linewidth"] - 0.5, 0.9),
            linestyle=entry["linestyle"],
            alpha=0.85,
        )

    ax_eq.axhline(initial_cash, color="#ffffff", linewidth=0.9, alpha=0.5, linestyle="--")
    ax_eq.yaxis.set_major_formatter(yuan_formatter(initial_cash))
    ax_eq.set_ylabel(f"Equity (start ¥{initial_cash:,.0f})")
    ax_eq.set_title("Equity curves on the test window", loc="left")
    ax_eq.legend(frameon=False, ncols=2, loc="upper left", fontsize=8)
    fp.clean_axis(ax_eq)

    ax_dd.axhline(0, color="#ffffff", linewidth=0.9, alpha=0.45)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel("Test step")
    fp.clean_axis(ax_dd)
    fp.save_figure(fig, path)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def fmt_yuan(value: float) -> str:
    return f"¥{value:,.0f}"


def fmt_ret(value: float) -> str:
    return f"{value * 100:+.2f}%"


def fmt_num(value: float) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


def fmt_pct(value: float) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"


def metrics_table(entries: list[dict[str, Any]], initial_cash: float) -> str:
    head = (
        "<tr><th>Strategy</th><th>Final equity</th><th>Return</th><th>Sharpe</th>"
        "<th>Sortino</th><th>Max DD</th><th>Calmar</th><th>Win rate</th>"
        "<th>Turnover</th></tr>"
    )
    rows = [head]
    for entry in entries:
        m = entry["metrics"]
        cls = ' class="hero"' if entry.get("group") == "trend" else ""
        final_equity = initial_cash * (1.0 + m.get("return", 0.0))
        rows.append(
            f"<tr{cls}><td>{entry['name']}</td>"
            f"<td>{fmt_yuan(final_equity)}</td>"
            f"<td>{fmt_ret(m.get('return', 0.0))}</td>"
            f"<td>{fmt_num(m.get('sharpe'))}</td>"
            f"<td>{fmt_num(m.get('sortino'))}</td>"
            f"<td>{fmt_pct(m.get('max_drawdown'))}</td>"
            f"<td>{fmt_num(m.get('calmar'))}</td>"
            f"<td>{fmt_pct(m.get('win_rate'))}</td>"
            f"<td>{fmt_num(m.get('turnover'))}</td></tr>"
        )
    return "\n".join(rows)


def delta_pills(
    base_map: dict[str, fp.StrategyEval], trend_map: dict[str, fp.StrategyEval]
) -> str:
    """One pill per algorithm: the effect of adding trend features (trend - base)."""
    pills: list[str] = []
    for algo in CONTINUOUS_ALGO_ORDER:
        if algo not in base_map or algo not in trend_map:
            continue
        bm, tm = base_map[algo].metrics, trend_map[algo].metrics
        ret_d = tm.get("return", 0.0) - bm.get("return", 0.0)
        sharpe_d = tm.get("sharpe", 0.0) - bm.get("sharpe", 0.0)
        dd_d = tm.get("max_drawdown", 0.0) - bm.get("max_drawdown", 0.0)
        # Better = higher return, higher Sharpe, lower drawdown.
        helped = (ret_d > 0) + (sharpe_d > 0) + (dd_d < 0)
        cls = "good" if helped >= 2 else "bad"
        pills.append(
            f'<span class="pill {cls}"><b>{algo}</b> +Trend: '
            f"Return Δ {fmt_ret(ret_d)} · Sharpe Δ {sharpe_d:+.2f} · "
            f"MaxDD Δ {dd_d * 100:+.2f}pp</span>"
        )
    if not pills:
        return '<span class="pill muted">No paired base/trend algorithms to compare.</span>'
    return "\n".join(pills)


def improvement_cards() -> str:
    trend_sub = "".join(
        f"<li><b>{name}</b> — {desc}</li>" for name, desc in TREND_FEATURE_CARDS
    )
    return f"""
      <div class="card">
        <div class="tag">Improvement #1</div>
        <h3>Revise the action space</h3>
        <p>Discrete buy/sell → continuous target position <code>a_t ∈ [-1, 1]</code>.
        The agent sizes exposure continuously (RL-based position management) instead
        of flipping all-in / all-out.</p>
      </div>
      <div class="card">
        <div class="tag">Improvement #2</div>
        <h3>Upgrade the state space</h3>
        <p>Four trend-aware features injected into the observation:</p>
        <ul>{trend_sub}</ul>
      </div>
      <div class="card">
        <div class="tag">Improvement #3</div>
        <h3>Enhance performance indicators</h3>
        <p>Max Drawdown, Sharpe, Sortino, Calmar, win rate and turnover are computed
        per episode and reported in the table below.</p>
      </div>
    """


HTML_STYLE = """
  :root {
    --bg:#050608; --surface:#0e1014; --line:rgba(255,255,255,.12);
    --text:#f2f4f8; --muted:#9aa3b2; --hero:#7ee787; --good:#7ee787; --bad:#f4a261;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    line-height:1.55; }
  .wrap { max-width:1080px; margin:0 auto; padding:48px 24px 80px; }
  header h1 { font-size:2.1rem; margin:0 0 .4rem; letter-spacing:-.5px; }
  header p.sub { color:var(--muted); margin:0 0 8px; font-size:1.02rem; }
  .eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:.72rem;
    color:var(--hero); font-weight:800; }
  section { margin-top:44px; }
  h2 { font-size:1.35rem; margin:0 0 16px; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
  .card { background:var(--surface); border:1px solid var(--line);
    border-radius:14px; padding:18px 20px; }
  .card h3 { margin:.3rem 0 .5rem; font-size:1.05rem; }
  .card p, .card li { color:var(--muted); font-size:.92rem; }
  .card ul { margin:.4rem 0 0; padding-left:1.1rem; }
  .tag { font-size:.7rem; font-weight:800; letter-spacing:1px; color:var(--hero);
    text-transform:uppercase; }
  code { background:#1b1f27; padding:1px 6px; border-radius:5px; font-size:.85em; }
  figure { margin:0; background:var(--surface); border:1px solid var(--line);
    border-radius:14px; padding:14px; }
  figure img { width:100%; display:block; border-radius:8px; }
  figcaption { color:var(--muted); font-size:.85rem; margin-top:10px; }
  table { width:100%; border-collapse:collapse; background:var(--surface);
    border:1px solid var(--line); border-radius:14px; overflow:hidden; font-size:.9rem; }
  th, td { padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; }
  tr:first-child th { background:#12151b; color:var(--text); font-weight:700; }
  tr.hero td { background:rgba(126,231,135,.08); color:var(--hero); font-weight:600; }
  .pills { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
  .pill { background:#12151b; border:1px solid var(--line); border-radius:999px;
    padding:7px 14px; font-size:.86rem; font-weight:600; }
  .pill.good { border-color:rgba(126,231,135,.5); color:var(--good); }
  .pill.bad { border-color:rgba(244,162,97,.5); color:var(--bad); }
  .pill.muted { color:var(--muted); }
  .banner { background:rgba(244,162,97,.1); border:1px solid rgba(244,162,97,.4);
    border-radius:10px; padding:10px 14px; margin:8px 0; color:#f4a261; font-size:.9rem; }
  footer { margin-top:48px; color:var(--muted); font-size:.82rem;
    border-top:1px solid var(--line); padding-top:18px; }
"""


def build_html(
    args: argparse.Namespace,
    fig_uri: str,
    entries: list[dict[str, Any]],
    pills_html: str,
    banners: list[str],
    test_period: str,
) -> str:
    banner_html = "".join(f'<div class="banner">{b}</div>' for b in banners)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Trading — Improvements Comparison</title>
<style>{HTML_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">RL-Stock · TUM</div>
      <h1>Trading Agent Improvements — Before vs After</h1>
      <p class="sub">Continuous position management + trend-aware state space, evaluated
      against a discrete baseline and Buy &amp; Hold on the same test window.</p>
      <p class="sub">Initial capital <b>{fmt_yuan(args.initial_cash)}</b> · Test period {test_period}</p>
    </header>
    {banner_html}

    <section>
      <h2>What changed</h2>
      <div class="grid">{improvement_cards()}</div>
    </section>

    <section>
      <h2>Equity comparison</h2>
      <figure>
        <img alt="Equity curves" src="{fig_uri}">
        <figcaption>Each strategy starts from {fmt_yuan(args.initial_cash)}. Top: portfolio
        equity; bottom: drawdown. Continuous configs are paired per algorithm —
        <b>dashed = base</b>, <b>solid = + trend features</b>.</figcaption>
      </figure>
    </section>

    <section>
      <h2>Performance metrics</h2>
      <table>{metrics_table(entries, args.initial_cash)}</table>
      <h2 style="margin-top:28px;">Effect of trend features (per algorithm)</h2>
      <div class="pills">{pills_html}</div>
    </section>

    <footer>
      Generated from saved training artifacts (deterministic evaluation). Reinforcement-learning
      results are stochastic and reflect a single training seed per configuration, so the
      trend-feature effect varies by algorithm; the features add trend awareness to the state
      space but do not guarantee a higher return. Treat this as a methodology demonstration.
      The discrete baseline reports realized trade P&amp;L only — the legacy env keeps no
      mark-to-market equity curve, so its risk metrics are shown as “—” and it is omitted from
      the equity chart (a key motivation for improvement #3). Regenerate with
      <code>python scripts/generate_improvement_comparison.py</code>.
    </footer>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    fp.configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fp.require_file(args.data_path, "market data")
    df = load_market_data(args.data_path)

    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    test_period = f"{df.index[test_bound[0]].date()} → {df.index[-1].date()}"

    entries: list[dict[str, Any]] = []
    banners: list[str] = []

    # Buy & Hold always anchors the chart.
    bh_curve, bh_metrics = buy_hold_series(df, args)
    entries.append(
        {
            "key": "buy_hold",
            "name": "Buy & Hold",
            "color": BENCHMARK_COLOR,
            "linestyle": ":",
            "linewidth": 2.0,
            "curve": to_yuan(bh_curve, args.initial_cash),
            "metrics": bh_metrics,
            "group": "benchmark",
        }
    )

    disc_entry, disc_banner = discrete_entry(df, args)
    if disc_entry:
        entries.append(disc_entry)
    elif disc_banner:
        banners.append(disc_banner)

    base_map, base_env, base_banner = position_strategy_map(args.position_dir, df, args, trend=False)
    trend_map, trend_env, trend_banner = position_strategy_map(args.trend_dir, df, args, trend=True)
    if base_banner:
        banners.append(base_banner)
    if trend_banner:
        banners.append(trend_banner)

    # Pair per algorithm: base then +trend, so the ablation reads left-to-right.
    for algo in CONTINUOUS_ALGO_ORDER:
        if algo in base_map:
            entries.append(continuous_entry(algo, "base", base_map[algo], base_env, args))
        if algo in trend_map:
            entries.append(continuous_entry(algo, "trend", trend_map[algo], trend_env, args))

    fig_path = args.fig_dir / "equity_comparison.png"
    generate_equity_chart(entries, args.initial_cash, fig_path)

    pills_html = delta_pills(base_map, trend_map)
    document = build_html(args, fp.image_uri(fig_path), entries, pills_html, banners, test_period)
    args.html.write_text(document, encoding="utf-8")

    print("Strategies compared:")
    for entry in entries:
        m = entry["metrics"]
        print(
            f"  {entry['name']:<28} return={fmt_ret(m.get('return', 0.0)):>9}  "
            f"sharpe={fmt_num(m.get('sharpe')):>6}  maxDD={fmt_pct(m.get('max_drawdown')):>8}"
        )
    for banner in banners:
        print(f"  [skip] {banner}")
    print(f"\nchart: {fig_path.relative_to(ROOT)}")
    print(f"html:  {args.html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
