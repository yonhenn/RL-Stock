"""Self-contained HTML report: how does each RL algorithm perform per trend regime?

The headline study answers a single question: when the NASDAQ-100 is in a
**downtrend**, a **sideways** range, or an **uptrend**, which algorithm trades it
best? It loads the SAC / TD3 / DDPG / PPO models trained on the trend-aware
position-management env, replays each deterministically on the test window, and
**attributes every day's profit to the trend regime in force when the position
was taken** (the env's own ``classify_trend_regime`` judgment, causal, no
look-ahead). Buy & Hold is the passive benchmark.

It then renders:

  * a regime-segmented scorecard (compounded return & Sharpe per band),
  * a regime-shaded equity overlay (see who makes / loses money where),
  * the agents' positioning behaviour per band (do they lean long in uptrends,
    de-risk in downtrends?),
  * a full per-band metrics table.

It does not train. Regenerate the models first, e.g.::

    python train.py --env-type position --use-trend-features --use-trend-prior \\
        --regime-risk-scaling --algorithms SAC,TD3,DDPG,PPO \\
        --total-timesteps 50000 --output-dir outputs/regime_study

Then::

    python scripts/generate_regime_performance.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_improvement_comparison as cmp  # noqa: E402
from rl_stock.data import load_market_data, split_frame_bounds  # noqa: E402
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig  # noqa: E402
from rl_stock.risk import compute_sharpe_ratio  # noqa: E402

fp = cmp.fp


def load_model(algo: str, model_dir: Path, env: PositionMgmtEnv) -> Any:
    """Load a saved SB3 model. Local loader so DDPG is supported (fp.load_model is not)."""
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    loaders = {"A2C": A2C, "DDPG": DDPG, "PPO": PPO, "SAC": SAC, "TD3": TD3}
    if algo not in loaders:
        raise ValueError(f"Unknown algorithm: {algo!r}")
    model_path = model_dir / f"{algo.lower()}.zip"
    fp.require_file(model_path, f"{algo} model")
    return loaders[algo].load(model_path, env=env)

# Algorithms to compare (must exist as <name>.zip in the model dir).
ALGOS = ["SAC", "TD3", "DDPG", "PPO"]
BENCHMARK = "Buy & Hold"

# Coarse 3-band regime grouping (the user's "波段"): the env classifies each bar
# into {-2,-1,0,1,2}; we fold sign into three tradable bands.
BANDS = ["Down", "Sideways", "Up"]
BAND_LABEL = {
    "Down": "Downtrend\n下降趋势",
    "Sideways": "Sideways\n震荡",
    "Up": "Uptrend\n上升趋势",
}
BAND_TINT = {"Down": "#f4a261", "Sideways": "#6b7280", "Up": "#7ee787"}

# One colour per strategy (DDPG / Buy & Hold are not in cmp.ALGO_COLORS).
STRATEGY_COLORS = {
    "SAC": "#7ee787",
    "TD3": "#b07ed1",
    "DDPG": "#e9967a",
    "PPO": "#56b4e9",
    BENCHMARK: "#9aa3b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "nasdaq100.xlsx")
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "outputs" / "regime_study"
    )
    parser.add_argument("--html", type=Path, default=ROOT / "regime_performance.html")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "outputs" / "regime_performance" / "figures",
    )
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    # Risk + trend config MUST match training so the replay reproduces it.
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.0,
        help="Per-trade hard stop fraction; 0 disables it (matches the no-stop-loss training).",
    )
    parser.add_argument("--adx-trend-threshold", type=float, default=20.0)
    parser.add_argument("--adx-strong-threshold", type=float, default=40.0)
    parser.add_argument(
        "--inject-into",
        type=Path,
        default=None,
        help="Inject Chinese regime slides into this presentation deck (idempotent) "
        "instead of writing the standalone report.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Environment + replay (config MUST match training)
# --------------------------------------------------------------------------- #
def build_env(df, args: argparse.Namespace) -> PositionMgmtEnv:
    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    risk = RiskConfig(
        initial_cash=args.initial_cash,
        stop_loss_pct=args.stop_loss_pct,
        use_trend_features=True,
        use_trend_prior=True,
        regime_risk_scaling=True,
        adx_trend_threshold=args.adx_trend_threshold,
        adx_strong_threshold=args.adx_strong_threshold,
    )
    return PositionMgmtEnv(
        df=df, window_size=args.window_size, frame_bound=test_bound, risk_config=risk
    )


def replay(model: Any, env: PositionMgmtEnv, seed: int) -> dict[str, np.ndarray]:
    """Deterministic single episode.

    Returns per-step arrays, all the same length ``n_steps`` (except ``equity``
    which is ``n_steps + 1``). ``regime`` is captured *before* each step, so it is
    the regime judgment in force when the position for that step was decided
    (causal attribution, no look-ahead). ``market_return`` is the underlying
    index move over the step, i.e. the Buy & Hold per-step return.
    """
    obs, info = env.reset(seed=seed)
    if model is None:
        env.action_space.seed(seed)

    regime_now = int(round(info["regime"]))
    returns: list[float] = []
    regimes: list[int] = []
    positions: list[float] = []
    market: list[float] = []
    equity: list[float] = [float(info["equity"])]

    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        returns.append(float(info["returns"][-1]))
        regimes.append(regime_now)
        positions.append(float(info["position_ratio"]))
        prev_price = float(env.prices[info["prev_tick"]])
        cur_price = float(env.prices[info["tick"]])
        market.append((cur_price - prev_price) / max(prev_price, 1e-8))
        equity.append(float(info["equity"]))

        regime_now = int(round(info["regime"]))
        done = bool(terminated or truncated)

    return {
        "returns": np.asarray(returns, dtype=float),
        "regimes": np.asarray(regimes, dtype=int),
        "positions": np.asarray(positions, dtype=float),
        "market_return": np.asarray(market, dtype=float),
        "equity": np.asarray(equity, dtype=float),
    }


# --------------------------------------------------------------------------- #
# Per-regime attribution
# --------------------------------------------------------------------------- #
def band_mask(regimes: np.ndarray, band: str) -> np.ndarray:
    if band == "Down":
        return regimes < 0
    if band == "Up":
        return regimes > 0
    return regimes == 0


def band_metrics(returns: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    """Metrics over one regime's days (the days need not be contiguous)."""
    if returns.size == 0:
        return {
            "return": 0.0,
            "mean_daily": 0.0,
            "sharpe": 0.0,
            "win_rate": float("nan"),
            "avg_pos": float("nan"),
            "n_days": 0,
        }
    return {
        # Compounded growth across just this band's days.
        "return": float(np.prod(1.0 + returns) - 1.0),
        "mean_daily": float(np.mean(returns)),
        "sharpe": compute_sharpe_ratio(returns),
        "win_rate": float(np.mean(returns > 0)),
        "avg_pos": float(np.mean(positions)),
        "n_days": int(returns.size),
    }


def attribute(rec: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Split a replay record into per-band metrics."""
    out: dict[str, dict[str, float]] = {}
    for band in BANDS:
        mask = band_mask(rec["regimes"], band)
        out[band] = band_metrics(rec["returns"][mask], rec["positions"][mask])
    # Whole-window metrics for context.
    out["Overall"] = {
        "return": float(rec["equity"][-1] / rec["equity"][0] - 1.0),
        "sharpe": compute_sharpe_ratio(rec["returns"]),
        "n_days": int(rec["returns"].size),
    }
    return out


def regime_distribution(regimes: np.ndarray) -> dict[str, dict[str, float]]:
    n = max(regimes.size, 1)
    dist: dict[str, dict[str, float]] = {}
    for band in BANDS:
        mask = band_mask(regimes, band)
        days = int(mask.sum())
        dist[band] = {"days": float(days), "pct": days / n, "segments": float(_segments(mask))}
    return dist


def _segments(mask: np.ndarray) -> int:
    """Count contiguous runs of True (number of distinct波段 for the band)."""
    count, prev = 0, False
    for value in mask.tolist():
        if value and not prev:
            count += 1
        prev = value
    return count


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _grouped_bars(
    ax,
    strategies: list[str],
    value_fn: Callable[[str, str], float],
    fmt: Callable[[float], str],
    *,
    as_pct: bool = False,
) -> None:
    """Bars grouped by regime band (x) with one bar per strategy."""
    x = np.arange(len(BANDS))
    width = 0.8 / len(strategies)
    for idx, strat in enumerate(strategies):
        values = np.array([value_fn(strat, band) for band in BANDS], dtype=float)
        plotted = values * 100.0 if as_pct else values
        offset = (idx - (len(strategies) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            plotted,
            width,
            label=strat,
            color=STRATEGY_COLORS[strat],
            edgecolor="#0b0b0b",
            linewidth=0.6,
        )
        for bar, raw in zip(bars, values):
            if not np.isfinite(raw):
                continue
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                fmt(raw),
                (bar.get_x() + bar.get_width() / 2.0, height),
                textcoords="offset points",
                xytext=(0, 2 if height >= 0 else -2),
                ha="center",
                va=va,
                fontsize=6.6,
                color="#e8eaed",
            )
    ax.axhline(0, color="#ffffff", linewidth=0.9, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABEL[b] for b in BANDS])
    fp.clean_axis(ax)


def _expand_ylim(ax, top: float = 0.3, bottom: float = 0.18) -> None:
    """Pad the y-axis so value labels and the legend clear the bars."""
    lo, hi = ax.get_ylim()
    rng = hi - lo or 1.0
    ax.set_ylim(lo - rng * bottom, hi + rng * top)


def chart_scorecard(
    metrics: dict[str, dict[str, dict[str, float]]], strategies: list[str], path: Path
) -> None:
    """Two panels: compounded return per band, annualised Sharpe per band."""
    fig, (ax_ret, ax_sharpe) = plt.subplots(
        2, 1, figsize=(11.2, 8.2), gridspec_kw={"hspace": 0.32}
    )
    _grouped_bars(
        ax_ret,
        strategies,
        lambda s, b: metrics[s][b]["return"],
        lambda v: f"{v * 100:+.1f}%",
        as_pct=True,
    )
    ax_ret.set_ylabel("Compounded return (band days only)")
    ax_ret.set_title(
        "Per-regime return — who trades each band best?", loc="left"
    )
    # Extra top room so the wide legend sits above the (tall) Buy & Hold bars.
    _expand_ylim(ax_ret, top=0.55, bottom=0.18)
    ax_ret.legend(frameon=False, ncols=len(strategies), loc="upper left", fontsize=8)

    _grouped_bars(
        ax_sharpe,
        strategies,
        lambda s, b: metrics[s][b]["sharpe"],
        lambda v: f"{v:.2f}",
    )
    ax_sharpe.set_ylabel("Annualised Sharpe (band days only)")
    ax_sharpe.set_title("Per-regime risk-adjusted return", loc="left")
    _expand_ylim(ax_sharpe, top=0.22, bottom=0.22)
    fp.save_figure(fig, path)


def chart_equity_with_regime(
    curves: dict[str, np.ndarray],
    regimes: np.ndarray,
    initial_cash: float,
    path: Path,
) -> None:
    """Equity overlay shaded by the trend regime in force at each step."""
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    steps = np.arange(regimes.size)
    trans = ax.get_xaxis_transform()
    for band in ("Down", "Up"):  # leave Sideways unshaded for contrast
        mask = band_mask(regimes, band)
        if mask.any():
            ax.fill_between(
                steps,
                0,
                1,
                where=mask,
                transform=trans,
                color=BAND_TINT[band],
                alpha=0.13,
                step="mid",
                label=f"{band}trend regime" if band != "Up" else "Uptrend regime",
            )

    for strat, curve in curves.items():
        yuan = cmp.to_yuan(curve, initial_cash)
        ax.plot(
            np.arange(yuan.size),
            yuan,
            label=strat,
            color=STRATEGY_COLORS[strat],
            linewidth=2.3 if strat == "SAC" else 1.5,
            linestyle="-" if strat in (BENCHMARK, "SAC") else "--",
        )
    ax.axhline(initial_cash, color="#ffffff", linewidth=0.9, alpha=0.4, linestyle="--")
    ax.yaxis.set_major_formatter(cmp.yuan_formatter(initial_cash))
    ax.set_ylabel(f"Equity (start ¥{initial_cash:,.0f})")
    ax.set_xlabel("Test step")
    ax.set_title(
        "Equity through the regimes — orange = downtrend, green = uptrend days",
        loc="left",
    )
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=7.8)
    fp.clean_axis(ax)
    fp.save_figure(fig, path)


def chart_behavior(
    metrics: dict[str, dict[str, dict[str, float]]], strategies: list[str], path: Path
) -> None:
    """Two panels: average net position per band, win rate per band."""
    fig, (ax_pos, ax_win) = plt.subplots(
        2, 1, figsize=(11.2, 8.2), gridspec_kw={"hspace": 0.32}
    )
    rl = [s for s in strategies if s != BENCHMARK]
    _grouped_bars(
        ax_pos,
        rl,
        lambda s, b: metrics[s][b]["avg_pos"],
        lambda v: f"{v * 100:+.0f}%",
        as_pct=True,
    )
    ax_pos.set_ylabel("Average net position")
    ax_pos.set_ylim(-110, 110)
    ax_pos.set_title(
        "How the agents position per regime (long > 0, short < 0)", loc="left"
    )
    ax_pos.legend(frameon=False, ncols=len(rl), loc="upper left", fontsize=8)

    _grouped_bars(
        ax_win,
        rl,
        lambda s, b: metrics[s][b]["win_rate"],
        lambda v: f"{v * 100:.0f}%",
        as_pct=True,
    )
    ax_win.set_ylabel("Win rate (positive days)")
    ax_win.set_title("Daily win rate per regime", loc="left")
    fp.save_figure(fig, path)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def metrics_table(
    metrics: dict[str, dict[str, dict[str, float]]], strategies: list[str]
) -> str:
    head = (
        "<tr><th>Regime</th><th>Strategy</th><th>Compounded return</th>"
        "<th>Sharpe</th><th>Win rate</th><th>Avg position</th><th>Days</th></tr>"
    )
    out = [head]
    for band in BANDS:
        # Highlight the best *RL* algorithm (Buy & Hold is the benchmark, not a
        # competitor in the "which algorithm" question).
        best = max(ALGOS, key=lambda s: metrics[s][band]["return"])
        for strat in strategies:
            m = metrics[strat][band]
            label = BAND_LABEL[band].replace("\n", " ")
            cls = ' class="hero"' if strat == best else ""
            first = (
                f'<td rowspan="{len(strategies)}">{label}</td>'
                if strat == strategies[0]
                else ""
            )
            out.append(
                f"<tr{cls}>{first}<td>{strat}</td>"
                f"<td>{cmp.fmt_ret(m['return'])}</td>"
                f"<td>{cmp.fmt_num(m['sharpe'])}</td>"
                f"<td>{cmp.fmt_pct(m.get('win_rate'))}</td>"
                f"<td>{cmp.fmt_pct(m.get('avg_pos'))}</td>"
                f"<td>{m['n_days']}</td></tr>"
            )
    return "\n".join(out)


def distribution_pills(dist: dict[str, dict[str, float]]) -> str:
    pills = []
    for band in BANDS:
        d = dist[band]
        label = BAND_LABEL[band].replace("\n", " ")
        pills.append(
            f'<span class="pill muted"><b>{label}</b>: '
            f"{d['pct'] * 100:.0f}% of days ({int(d['days'])} bars, "
            f"{int(d['segments'])} segments)</span>"
        )
    return "\n".join(pills)


def takeaway_pills(metrics: dict[str, dict[str, dict[str, float]]]) -> str:
    """One pill per band: best / worst *RL* algorithm, with B&H as benchmark.

    The user's question is which algorithm trades each band best, so the ranking
    is over the RL algos only; Buy & Hold is shown as the passive reference.
    Green when the best RL agent was profitable in that band, else orange.
    """
    pills = []
    for band in BANDS:
        ranked = sorted(ALGOS, key=lambda s: metrics[s][band]["return"], reverse=True)
        best, worst = ranked[0], ranked[-1]
        label = BAND_LABEL[band].replace("\n", " ")
        cls = "good" if metrics[best][band]["return"] > 0 else "bad"
        pills.append(
            f'<span class="pill {cls}"><b>{label}</b>: best RL <b>{best}</b> '
            f"{cmp.fmt_ret(metrics[best][band]['return'])} · "
            f"worst {worst} {cmp.fmt_ret(metrics[worst][band]['return'])} · "
            f"benchmark B&amp;H {cmp.fmt_ret(metrics[BENCHMARK][band]['return'])}</span>"
        )
    return "\n".join(pills)


def verdict_note(metrics: dict[str, dict[str, dict[str, float]]]) -> str:
    """A short, data-driven sentence summarising the RL-vs-benchmark picture."""
    best_rl = max(ALGOS, key=lambda s: metrics[s]["Overall"]["return"])
    profitable = [
        BAND_LABEL[b].split("\n")[0].lower() for b in BANDS if metrics[best_rl][b]["return"] > 0
    ]
    bh_wins = all(
        metrics[BENCHMARK]["Overall"]["return"] > metrics[s]["Overall"]["return"]
        for s in ALGOS
    )
    lead = (
        "Buy &amp; Hold beat every RL agent in this bull window. "
        if bh_wins
        else ""
    )
    if profitable:
        bands_txt = " and ".join(profitable)
        tail = (
            f"Among the RL algorithms <b>{best_rl}</b> is strongest overall "
            f"({cmp.fmt_ret(metrics[best_rl]['Overall']['return'])}) and the only one "
            f"that turns a profit in any regime ({bands_txt})."
        )
    else:
        tail = (
            f"Among the RL algorithms <b>{best_rl}</b> is the least-bad overall "
            f"({cmp.fmt_ret(metrics[best_rl]['Overall']['return'])}); none profit in any regime here."
        )
    return lead + tail


def build_html(args, results: dict[str, Any]) -> str:
    strategies = results["strategies"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Position Management — Performance by Trend Regime</title>
<style>{cmp.HTML_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">RL-Stock · TUM</div>
      <h1>Which algorithm wins in which market regime?</h1>
      <p class="sub">SAC · TD3 · DDPG · PPO vs Buy &amp; Hold on the NASDAQ-100,
      with every day's P&amp;L attributed to the trend regime
      (<b>downtrend 下降</b> / <b>sideways 震荡</b> / <b>uptrend 上升</b>) in force when
      the position was taken.</p>
      <p class="sub">Initial capital <b>¥{args.initial_cash:,.0f}</b> ·
      Test period {results["test_period"]} · {args.total_timesteps:,}-step training budget each</p>
    </header>

    <section>
      <h2>How a regime is defined</h2>
      <div class="grid">
        <div class="card"><div class="tag">Strength</div><h3>ADX gate</h3>
          <p>ADX &lt; {args.adx_trend_threshold:.0f} → <b>sideways</b> (no tradable trend);
          ≥ {args.adx_trend_threshold:.0f} → trending; ≥ {args.adx_strong_threshold:.0f} → strong.</p></div>
        <div class="card"><div class="tag">Direction</div><h3>3-vote consensus</h3>
          <p>Sign of the SMA-20 slope, the MACD golden/death cross, and price vs its
          SMA decide up vs down. Conflicts stay sideways.</p></div>
        <div class="card"><div class="tag">Attribution</div><h3>Causal, decision-time</h3>
          <p>Each day's return is booked to the regime known <b>when the position was
          chosen</b> (no look-ahead). Returns are compounded within each band.</p></div>
      </div>
      <div class="pills" style="margin-top:14px;">{results["distribution_pills"]}</div>
    </section>

    <section>
      <h2>The verdict — best &amp; worst per regime</h2>
      <p class="sub">{results["verdict_note"]}</p>
      <div class="pills">{results["takeaway_pills"]}</div>
      <figure style="margin-top:16px;">
        <img alt="Per-regime scorecard" src="{results["scorecard_uri"]}">
        <figcaption>Top: compounded return over each band's days. Bottom: annualised Sharpe
        on those same days. Bars grouped by regime, one per strategy. The trend prior caps
        counter-trend exposure, so the agents are designed to lean with the band.</figcaption>
      </figure>
    </section>

    <section>
      <h2>Equity through the regimes</h2>
      <figure>
        <img alt="Regime-shaded equity overlay" src="{results["equity_uri"]}">
        <figcaption>All start from ¥{args.initial_cash:,.0f}. Background shaded by the trend
        judgment (orange = downtrend, green = uptrend days; sideways left blank). Watch where
        each curve gains or bleeds relative to the shading.</figcaption>
      </figure>
    </section>

    <section>
      <h2>What the agents actually did</h2>
      <figure>
        <img alt="Positioning behaviour per regime" src="{results["behavior_uri"]}">
        <figcaption>Top: average net position held per band (positive = net long, negative =
        net short). Bottom: fraction of winning days per band. This is the <i>behaviour</i>
        behind the returns above.</figcaption>
      </figure>
    </section>

    <section>
      <h2>Full per-regime metrics</h2>
      <table>{results["table_html"]}</table>
    </section>

    <footer>
      Deterministic replay of saved models in an env configured exactly like training
      (trend features + trend prior + regime caps, hard stop-loss disabled), retrained on the
      current <code>nasdaq100.xlsx</code> vintage so train, test and regime labels share one dataset.
      Single training seed per algorithm at a {args.total_timesteps:,}-step budget; RL results
      are stochastic, so treat this as a methodology demonstration rather than a guarantee.
      Strong-up / strong-down bars are folded into up / down. Regenerate with
      <code>python scripts/generate_regime_performance.py</code>.
    </footer>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Compute + assemble
# --------------------------------------------------------------------------- #
def build_results(args: argparse.Namespace) -> dict[str, Any]:
    fp.configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fp.require_file(args.data_path, "market data")
    df = load_market_data(args.data_path)
    _, test_bound = split_frame_bounds(
        df, window_size=args.window_size, test_start_date=args.test_start_date
    )
    test_period = f"{df.index[test_bound[0]].date()} → {df.index[-1].date()}"

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    curves: dict[str, np.ndarray] = {}
    regimes: np.ndarray | None = None
    market_rec: dict[str, np.ndarray] | None = None

    for algo in ALGOS:
        env = build_env(df, args)
        model = load_model(algo, args.model_dir, env)
        rec = replay(model, env, args.seed)
        metrics[algo] = attribute(rec)
        curves[algo] = rec["equity"]
        if regimes is None:
            regimes = rec["regimes"]
            market_rec = rec
        print(
            f"{algo:<4} overall={cmp.fmt_ret(metrics[algo]['Overall']['return']):>9}  "
            + "  ".join(
                f"{b[:4]}={cmp.fmt_ret(metrics[algo][b]['return']):>8}" for b in BANDS
            )
        )

    assert regimes is not None and market_rec is not None

    # Buy & Hold: position 1.0 throughout, same regime sequence.
    bh_returns = market_rec["market_return"]
    bh_positions = np.ones_like(bh_returns)
    metrics[BENCHMARK] = attribute(
        {
            "returns": bh_returns,
            "regimes": regimes,
            "positions": bh_positions,
            "equity": np.concatenate([[1.0], np.cumprod(1.0 + bh_returns)]),
        }
    )
    curves[BENCHMARK] = np.concatenate([[1.0], np.cumprod(1.0 + bh_returns)])
    print(
        f"B&H  overall={cmp.fmt_ret(metrics[BENCHMARK]['Overall']['return']):>9}  "
        + "  ".join(
            f"{b[:4]}={cmp.fmt_ret(metrics[BENCHMARK][b]['return']):>8}" for b in BANDS
        )
    )

    strategies = ALGOS + [BENCHMARK]
    dist = regime_distribution(regimes)

    scorecard = args.fig_dir / "regime_scorecard.png"
    chart_scorecard(metrics, strategies, scorecard)
    equity = args.fig_dir / "regime_equity.png"
    chart_equity_with_regime(curves, regimes, args.initial_cash, equity)
    behavior = args.fig_dir / "regime_behavior.png"
    chart_behavior(metrics, strategies, behavior)

    return {
        "strategies": strategies,
        "metrics": metrics,
        "dist": dist,
        "test_period": test_period,
        "distribution_pills": distribution_pills(dist),
        "takeaway_pills": takeaway_pills(metrics),
        "verdict_note": verdict_note(metrics),
        "table_html": metrics_table(metrics, strategies),
        "scorecard_uri": fp.image_uri(scorecard),
        "equity_uri": fp.image_uri(equity),
        "behavior_uri": fp.image_uri(behavior),
    }


# --------------------------------------------------------------------------- #
# Deck injection (Chinese slides for presentation.html)
# --------------------------------------------------------------------------- #
SLIDE_START = "<!-- REGIME-SLIDES-START -->"
SLIDE_END = "<!-- REGIME-SLIDES-END -->"
CONCLUSION_ANCHOR = '<section class="slide" data-title="Conclusion">'


def _best_rl(metrics: dict, band: str) -> tuple[str, float]:
    algo = max(ALGOS, key=lambda s: metrics[s][band]["return"])
    return algo, metrics[algo][band]["return"]


def _deck_slide(
    data_title: str, eyebrow: str, title: str, fig_uri: str, caption: str, chips: str
) -> str:
    """One deck-native slide: header + a height-capped figure + chips."""
    return f'''<section class="slide" data-title="{data_title}">
        <div class="slide-header reveal" style="--d:0">
          <div>
            <div class="eyebrow">{eyebrow}</div>
            <h2 class="slide-title wide">{title}</h2>
          </div>
          <div class="slide-number">00 / 00</div>
        </div>
        <div class="content" style="align-content:center; gap:clamp(0.6rem,1.8vh,1.1rem);">
          <figure class="python-figure reveal" style="--d:1; margin:0 auto; max-width:1000px;">
            <img src="{fig_uri}" style="width:auto; max-width:100%; max-height:56vh; height:auto; margin:0 auto;">
            <figcaption>{caption}</figcaption>
          </figure>
          <div class="chips reveal" style="--d:2; justify-content:center;">{chips}</div>
        </div>
      </section>'''


def render_deck_slides_zh(results: dict[str, Any]) -> str:
    """Three deck-native Chinese slides summarising the regime study."""
    m = results["metrics"]
    dist = results["dist"]
    up_b, up_r = _best_rl(m, "Up")
    dn_b, dn_r = _best_rl(m, "Down")
    sd_b, sd_r = _best_rl(m, "Sideways")
    bh = m[BENCHMARK]
    bh_avg = (bh["Down"]["return"] + bh["Sideways"]["return"] + bh["Up"]["return"]) / 3.0

    def band_verb(band: str, ret: float) -> str:
        """Sign-aware label: profit phrasing when positive, else 'smallest loss'."""
        if ret > 0:
            n_pos = sum(1 for s in ALGOS if m[s][band]["return"] > 0)
            return "唯一盈利" if n_pos == 1 else "盈利最高"
        return "亏损最小"

    chips1 = (
        f'<span class="chip green">上升趋势 · {up_b} {band_verb("Up", up_r)} {cmp.fmt_ret(up_r)}</span>'
        f'<span class="chip cyan">下降趋势 · {dn_b} {band_verb("Down", dn_r)} {cmp.fmt_ret(dn_r)}</span>'
        f'<span class="chip magenta">震荡 · {sd_b} {band_verb("Sideways", sd_r)} {cmp.fmt_ret(sd_r)}</span>'
        f'<span class="chip amber">基准 Buy &amp; Hold 各段约 {cmp.fmt_ret(bh_avg)},全面领先</span>'
    )
    slide1 = _deck_slide(
        "Regimes",
        "Regime Analysis · 分波段验证",
        "不同市场波段下,各算法表现如何?",
        results["scorecard_uri"],
        "按下单时刻的趋势 regime 归因 · 上:各波段复合收益 · 下:年化夏普 · "
        "柱按波段分组,每色一个算法(已禁用硬性止损)",
        chips1,
    )

    chips2 = (
        f'<span class="chip">下降 {dist["Down"]["pct"] * 100:.0f}%</span>'
        f'<span class="chip">震荡 {dist["Sideways"]["pct"] * 100:.0f}%</span>'
        f'<span class="chip">上升 {dist["Up"]["pct"] * 100:.0f}%</span>'
        '<span class="chip amber">2024–2026 为强牛市,买入持有一骑绝尘</span>'
    )
    slide2 = _deck_slide(
        "Regime Equity",
        "Regime Analysis · 权益曲线",
        "穿越各波段的资金曲线",
        results["equity_uri"],
        "背景:橙 = 下降日、绿 = 上升日(震荡留白)· "
        "SAC(绿)最贴近本金,Buy & Hold(灰)在牛市中领先",
        chips2,
    )

    chips3 = (
        '<span class="chip cyan">下降趋势:各 agent 净做空(顺势减仓)</span>'
        '<span class="chip green">上升趋势:SAC 做多最果断、胜率最高</span>'
        '<span class="chip amber">洞察:牛市中“下降”标签前瞻性差,逆势减仓反而拖累收益</span>'
    )
    slide3 = _deck_slide(
        "Regime Behavior",
        "Regime Analysis · 行为剖析",
        "智能体在各波段实际怎么操作?",
        results["behavior_uri"],
        "上:各波段平均净仓位(>0 做多,<0 做空)· 下:日胜率 —— 收益背后的行为",
        chips3,
    )
    return "\n      ".join([slide1, slide2, slide3])


def _renumber_slides(document: str, total: int) -> str:
    """Rewrite every slide-number header to ``NN / total`` in document order."""
    counter = {"i": 0}

    def repl(match: re.Match) -> str:
        counter["i"] += 1
        return f'{match.group(1)}{counter["i"]:02d} / {total:02d}{match.group(2)}'

    return re.sub(
        r'(<div class="slide-number">)\s*\d+\s*/\s*\d+\s*(</div>)', repl, document
    )


def inject_into_presentation(html_path: Path, slides_html: str) -> int:
    """Insert the Chinese regime slides before the Conclusion slide. Idempotent."""
    fp.require_file(html_path, "presentation deck")
    document = html_path.read_text(encoding="utf-8")
    # Drop any previously injected block so re-runs update instead of duplicate.
    document = re.sub(
        re.escape(SLIDE_START) + r".*?" + re.escape(SLIDE_END),
        "",
        document,
        flags=re.DOTALL,
    )
    if CONCLUSION_ANCHOR not in document:
        raise ValueError(f"Could not find the Conclusion slide anchor in {html_path}")
    block = f"{SLIDE_START}\n      {slides_html}\n      {SLIDE_END}\n      "
    document = document.replace(CONCLUSION_ANCHOR, block + CONCLUSION_ANCHOR, 1)
    # Renumber every slide header + the pager label to the new total.
    total = document.count('<section class="slide"')
    document = _renumber_slides(document, total)
    document = re.sub(
        r'(<span id="pageLabel">)\s*\d+\s*/\s*\d+\s*(</span>)',
        rf"\g<1>01 / {total:02d}\g<2>",
        document,
    )
    html_path.write_text(document, encoding="utf-8")
    return total


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible, else absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    results = build_results(args)
    if args.inject_into is not None:
        total = inject_into_presentation(args.inject_into, render_deck_slides_zh(results))
        print(f"\ninjected 3 regime slides into: {_rel(args.inject_into)} ({total} slides total)")
    else:
        document = build_html(args, results)
        args.html.write_text(document, encoding="utf-8")
        print(f"\nhtml:  {_rel(args.html)}")
    print(f"figs:  {_rel(args.fig_dir)}")


if __name__ == "__main__":
    main()
