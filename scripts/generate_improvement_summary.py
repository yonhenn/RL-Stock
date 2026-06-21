"""Generate a comprehensive, self-contained HTML summary of the RL improvements.

Consolidates the whole story into one shareable page:
  * the three improvements (action space, state space, performance indicators);
  * an illustration of the four new trend features on the real test data;
  * the experimental setup;
  * the results (equity curves, per-algorithm metric bars, metrics table).

Reuses the evaluation/plotting helpers in ``generate_improvement_comparison.py``
(which in turn reuse ``generate_feature_presentation_images.py``). Loads
pre-trained models only -- it does not train.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_improvement_comparison as cmp  # noqa: E402
from rl_stock.data import load_market_data, split_frame_bounds  # noqa: E402
from rl_stock.indicators import (  # noqa: E402
    compute_adx,
    compute_bollinger_bandwidth,
    compute_macd_cross,
    compute_sma_slope,
    historical_moving_average,
)
from rl_stock.visualization import plot_risk_dashboard  # noqa: E402

fp = cmp.fp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "output.xlsx")
    parser.add_argument("--t1-dir", type=Path, default=ROOT / "outputs" / "t1")
    parser.add_argument("--position-dir", type=Path, default=ROOT / "outputs" / "position")
    parser.add_argument("--trend-dir", type=Path, default=ROOT / "outputs" / "position_trend")
    parser.add_argument("--html", type=Path, default=ROOT / "improvements_summary.html")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "outputs" / "improvement_summary" / "figures",
    )
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
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
# Charts
# --------------------------------------------------------------------------- #
def generate_feature_illustration(df: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    """Show the 4 new trend features computed on the real test-window data."""
    full_close = df["Close"].to_numpy(dtype=float)
    full_high = df["High"].to_numpy(dtype=float)
    full_low = df["Low"].to_numpy(dtype=float)

    sma20 = historical_moving_average(full_close, 20)
    slope = compute_sma_slope(full_close) * 100.0
    bandwidth = compute_bollinger_bandwidth(full_close)
    adx = compute_adx(full_high, full_low, full_close)
    cross = compute_macd_cross(full_close)

    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    s, e = test_bound[0], test_bound[1]
    x = df.index[s:e]

    fig, axes = plt.subplots(
        5, 1, figsize=(11.0, 9.6), sharex=True, gridspec_kw={"hspace": 0.14}
    )
    axes[0].plot(x, full_close[s:e], color="#cdd6e0", linewidth=1.3, label="Close")
    axes[0].plot(x, sma20[s:e], color="#56b4e9", linewidth=1.6, label="SMA-20")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Price & 20-day SMA", loc="left")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].axhline(0, color="#ffffff", linewidth=0.8, alpha=0.4)
    axes[1].plot(x, slope[s:e], color="#7ee787", linewidth=1.3)
    axes[1].set_ylabel("%/day")
    axes[1].set_title("① SMA-20 slope — trend direction", loc="left")

    axes[2].plot(x, bandwidth[s:e], color="#e9a456", linewidth=1.3)
    axes[2].set_ylabel("width/SMA")
    axes[2].set_title("② Bollinger bandwidth — volatility regime", loc="left")

    axes[3].axhline(25, color="#ffffff", linewidth=0.8, alpha=0.35, linestyle="--")
    axes[3].plot(x, adx[s:e], color="#b07ed1", linewidth=1.3)
    axes[3].set_ylabel("ADX")
    axes[3].set_title("③ ADX — trend strength (25 = trending threshold)", loc="left")

    cross_slice = cross[s:e]
    axes[4].axhline(0, color="#ffffff", linewidth=0.8, alpha=0.35)
    axes[4].step(x, cross_slice, color="#f2f4f8", linewidth=1.2, where="mid")
    axes[4].fill_between(x, cross_slice, 0, where=cross_slice > 0, color="#7ee787", alpha=0.25, step="mid")
    axes[4].fill_between(x, cross_slice, 0, where=cross_slice < 0, color="#f4a261", alpha=0.25, step="mid")
    axes[4].set_yticks([-1, 0, 1])
    axes[4].set_ylabel("regime")
    axes[4].set_title("④ MACD cross — golden (+1) / death (−1) regime", loc="left")

    for ax in axes:
        cmp.fp.clean_axis(ax)
    fig.autofmt_xdate()
    cmp.fp.save_figure(fig, path)


def generate_metric_bars(
    base_map: dict[str, Any], trend_map: dict[str, Any], path: Path
) -> bool:
    """Per-algorithm base-vs-trend bars for Return, Sharpe and Max Drawdown."""
    algos = [a for a in cmp.CONTINUOUS_ALGO_ORDER if a in base_map and a in trend_map]
    if not algos:
        return False

    specs = [("Return %", "return", 100.0), ("Sharpe", "sharpe", 1.0), ("Max Drawdown %", "max_drawdown", 100.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    width = 0.36
    x = np.arange(len(algos))
    for ax, (title, key, scale) in zip(axes, specs):
        base_vals = [base_map[a].metrics.get(key, 0.0) * scale for a in algos]
        trend_vals = [trend_map[a].metrics.get(key, 0.0) * scale for a in algos]
        ax.bar(x - width / 2, base_vals, width, label="base", color="#5a6b7a", edgecolor="#ffffff", linewidth=0.5)
        ax.bar(x + width / 2, trend_vals, width, label="+ trend", color="#56b4e9", edgecolor="#ffffff", linewidth=0.5)
        for xi, (bv, tv) in enumerate(zip(base_vals, trend_vals)):
            for off, val in ((-width / 2, bv), (width / 2, tv)):
                ax.text(xi + off, val, f"{val:.1f}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=7, color="#e8e8e8")
        ax.axhline(0, color="#ffffff", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_title(title, loc="left")
        cmp.fp.clean_axis(ax)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Effect of trend features per algorithm (base vs + trend)", x=0.012, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cmp.fp.save_figure(fig, path)
    return True


def generate_agent_dashboards(
    base_map: dict[str, Any],
    base_env: Any,
    trend_map: dict[str, Any],
    trend_env: Any,
    fig_dir: Path,
) -> list[tuple[str, str]]:
    """Dark-theme position / equity / drawdown dashboard for each continuous agent."""
    plans: list[tuple[str, Any, Any, str]] = []
    for algo in cmp.CONTINUOUS_ALGO_ORDER:
        if algo in base_map:
            plans.append((f"{algo} · base", base_env, base_map[algo], f"dash_{algo.lower()}_base.png"))
        if algo in trend_map:
            plans.append((f"{algo} · + trend", trend_env, trend_map[algo], f"dash_{algo.lower()}_trend.png"))

    dashboards: list[tuple[str, str]] = []
    for label, env, result, fname in plans:
        if not result.infos or env is None:
            continue
        fig = plot_risk_dashboard(env, result.infos[0], label)
        path = fig_dir / fname
        cmp.fp.save_figure(fig, path)
        dashboards.append((label, cmp.fp.image_uri(path)))
    return dashboards


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def takeaway_items(base_map: dict[str, Any], trend_map: dict[str, Any], bh_return: float) -> str:
    items: list[str] = []
    for algo in cmp.CONTINUOUS_ALGO_ORDER:
        if algo not in base_map or algo not in trend_map:
            continue
        bm, tm = base_map[algo].metrics, trend_map[algo].metrics
        ret_d = tm.get("return", 0.0) - bm.get("return", 0.0)
        sharpe_d = tm.get("sharpe", 0.0) - bm.get("sharpe", 0.0)
        dd_d = tm.get("max_drawdown", 0.0) - bm.get("max_drawdown", 0.0)
        helped = (ret_d > 0) + (sharpe_d > 0) + (dd_d < 0)
        verb = "improved" if helped >= 2 else "did not help"
        items.append(
            f"<li>Trend features <b>{verb} {algo}</b>: return {cmp.fmt_ret(ret_d)}, "
            f"Sharpe {sharpe_d:+.2f}, max drawdown {dd_d * 100:+.2f}pp.</li>"
        )
    items.append(
        "<li>The <b>continuous action space</b> turns binary buy/sell into graded position "
        "sizing, and the <b>risk metrics</b> (Sharpe, Sortino, Max DD, Calmar) are now first-class "
        "outputs rather than post-hoc guesses.</li>"
    )
    items.append(
        f"<li><b>Honest context:</b> the 2024–2026 test window was a strong bull market "
        f"(Buy &amp; Hold {cmp.fmt_ret(bh_return)}), so every learned agent trails passive holding "
        "here; the value shown is the methodology and the isolated per-algorithm effect, on a single seed.</li>"
    )
    return "\n".join(items)


SUMMARY_STYLE = (
    cmp.HTML_STYLE
    + """
  .hero-pills { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
  .stat { background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:12px 16px; }
  .stat .v { font-size:1.35rem; font-weight:800; }
  .stat .k { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
  .setup { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  .setup .stat .v { font-size:1.02rem; }
  ul.takeaways { margin:0; padding-left:1.1rem; }
  ul.takeaways li { margin:.5rem 0; }
  .dash-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
  @media (max-width:820px) {
    .dash-grid, .grid, .setup { grid-template-columns:1fr; }
  }
"""
)


def stat_block(value: str, label: str) -> str:
    return f'<div class="stat"><div class="v">{value}</div><div class="k">{label}</div></div>'


def build_summary_html(
    args: argparse.Namespace,
    figs: dict[str, str],
    entries: list[dict[str, Any]],
    base_map: dict[str, Any],
    trend_map: dict[str, Any],
    bh_return: float,
    data_range: str,
    test_period: str,
    banners: list[str],
    dashboards: list[tuple[str, str]],
) -> str:
    banner_html = "".join(f'<div class="banner">{b}</div>' for b in banners)
    pills_html = cmp.delta_pills(base_map, trend_map)
    metric_bars = (
        f'<figure style="margin-top:16px;"><img alt="metric bars" src="{figs["bars"]}">'
        "<figcaption>Base vs + trend, per algorithm. Higher Return/Sharpe is better; "
        "lower Max Drawdown is better.</figcaption></figure>"
        if "bars" in figs
        else ""
    )
    if dashboards:
        cards = "".join(
            f'<figure><img alt="{label}" src="{uri}"><figcaption>{label}</figcaption></figure>'
            for label, uri in dashboards
        )
        dashboards_section = (
            '<section><h2>5 · How each agent traded</h2>'
            '<p class="sub">Target position vs price (top), equity curve (middle), and drawdown '
            '(bottom) for each continuous agent on one deterministic test episode.</p>'
            f'<div class="dash-grid">{cards}</div></section>'
        )
    else:
        dashboards_section = ""
    takeaways_no = "6" if dashboards else "5"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Trading — Improvements Summary</title>
<style>{SUMMARY_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">RL-Stock · TUM · Results Summary</div>
      <h1>Reinforcement-Learning Trading Improvements</h1>
      <p class="sub">Revised action space, upgraded trend-aware state space, and richer
      performance indicators — implemented, trained and evaluated end-to-end.</p>
      <div class="hero-pills">
        {stat_block("3", "improvements")}
        {stat_block("19 → 23", "observation features")}
        {stat_block(cmp.fmt_yuan(args.initial_cash), "initial capital")}
        {stat_block(f"{args.total_timesteps:,}", "steps / model")}
      </div>
    </header>
    {banner_html}

    <section>
      <h2>1 · What changed</h2>
      <div class="grid">{cmp.improvement_cards()}</div>
    </section>

    <section>
      <h2>2 · What the agent now sees</h2>
      <figure>
        <img alt="trend feature illustration" src="{figs["features"]}">
        <figcaption>The four trend-aware features (improvement #2) computed on the real
        test window. They give the agent explicit signals for trend direction, volatility
        regime, trend strength, and the MACD golden/death-cross prior.</figcaption>
      </figure>
    </section>

    <section>
      <h2>3 · Experimental setup</h2>
      <div class="setup">
        {stat_block("Shanghai Index", "instrument")}
        {stat_block(data_range, "data range")}
        {stat_block(test_period, "test window")}
        {stat_block(f"{args.window_size} days", "observation window")}
        {stat_block("PPO · SAC", "continuous algos")}
        {stat_block("A2C / PPO", "discrete baseline")}
        {stat_block("Stop-loss + max-DD lock", "risk controls")}
        {stat_block("Differential Sharpe", "reward signal")}
      </div>
    </section>

    <section>
      <h2>4 · Results</h2>
      <figure>
        <img alt="equity curves" src="{figs["equity"]}">
        <figcaption>Equity from {cmp.fmt_yuan(args.initial_cash)}. Continuous configs paired
        per algorithm — <b>dashed = base</b>, <b>solid = + trend</b>. The discrete baseline is
        omitted here (no mark-to-market curve; see table).</figcaption>
      </figure>
      {metric_bars}
      <table style="margin-top:16px;">{cmp.metrics_table(entries, args.initial_cash)}</table>
      <h2 style="margin-top:28px;">Trend-feature effect (per algorithm)</h2>
      <div class="pills">{pills_html}</div>
    </section>

    {dashboards_section}

    <section>
      <h2>{takeaways_no} · Takeaways</h2>
      <ul class="takeaways">{takeaway_items(base_map, trend_map, bh_return)}</ul>
    </section>

    <footer>
      Self-contained page generated from saved training artifacts (deterministic evaluation).
      Single training seed per configuration; results are stochastic and illustrative. Regenerate
      with <code>python scripts/generate_improvement_summary.py</code>.
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
    cmp.fp.configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    cmp.fp.require_file(args.data_path, "market data")
    df = load_market_data(args.data_path)

    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    data_range = f"{df.index.min().date()} → {df.index.max().date()}"
    test_period = f"{df.index[test_bound[0]].date()} → {df.index[-1].date()}"

    entries: list[dict[str, Any]] = []
    banners: list[str] = []

    bh_curve, bh_metrics = cmp.buy_hold_series(df, args)
    entries.append(
        {
            "key": "buy_hold",
            "name": "Buy & Hold",
            "color": cmp.BENCHMARK_COLOR,
            "linestyle": ":",
            "linewidth": 2.0,
            "curve": cmp.to_yuan(bh_curve, args.initial_cash),
            "metrics": bh_metrics,
            "group": "benchmark",
        }
    )

    disc_entry, disc_banner = cmp.discrete_entry(df, args)
    if disc_entry:
        entries.append(disc_entry)
    elif disc_banner:
        banners.append(disc_banner)

    base_map, base_env, base_banner = cmp.position_strategy_map(args.position_dir, df, args, trend=False)
    trend_map, trend_env, trend_banner = cmp.position_strategy_map(args.trend_dir, df, args, trend=True)
    if base_banner:
        banners.append(base_banner)
    if trend_banner:
        banners.append(trend_banner)

    for algo in cmp.CONTINUOUS_ALGO_ORDER:
        if algo in base_map:
            entries.append(cmp.continuous_entry(algo, "base", base_map[algo], base_env, args))
        if algo in trend_map:
            entries.append(cmp.continuous_entry(algo, "trend", trend_map[algo], trend_env, args))

    figs: dict[str, str] = {}
    feat_path = args.fig_dir / "feature_illustration.png"
    generate_feature_illustration(df, args, feat_path)
    figs["features"] = cmp.fp.image_uri(feat_path)

    equity_path = args.fig_dir / "equity_comparison.png"
    cmp.generate_equity_chart(entries, args.initial_cash, equity_path)
    figs["equity"] = cmp.fp.image_uri(equity_path)

    bars_path = args.fig_dir / "metric_bars.png"
    if generate_metric_bars(base_map, trend_map, bars_path):
        figs["bars"] = cmp.fp.image_uri(bars_path)

    dashboards = generate_agent_dashboards(base_map, base_env, trend_map, trend_env, args.fig_dir)

    document = build_summary_html(
        args, figs, entries, base_map, trend_map,
        bh_metrics["return"], data_range, test_period, banners, dashboards,
    )
    args.html.write_text(document, encoding="utf-8")

    print("Summary written.")
    for entry in entries:
        m = entry["metrics"]
        print(
            f"  {entry['name']:<28} return={cmp.fmt_ret(m.get('return', 0.0)):>9}  "
            f"sharpe={cmp.fmt_num(m.get('sharpe')):>6}  maxDD={cmp.fmt_pct(m.get('max_drawdown')):>8}"
        )
    for banner in banners:
        print(f"  [skip] {banner}")
    print(f"\nfigures: {args.fig_dir.relative_to(ROOT)}")
    print(f"html:    {args.html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
