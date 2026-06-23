"""Light-themed HTML showing what *position management* does: off vs on.

Frames the story simply:
  * OFF  -- discrete buy/sell agent: the position is binary (0% or 100% in market).
  * ON   -- continuous agent: it *sizes* the position (0-100%, can short) and adds
            risk controls (stop-loss, drawdown lock, volatility scaling).

It loads pre-trained models, replays one deterministic test episode, and renders a
concrete, annotated example of the position-management agent in action. Reuses the
evaluation helpers in ``generate_improvement_comparison.py``.
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
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_improvement_comparison as cmp  # noqa: E402
from rl_stock.data import load_market_data, split_frame_bounds  # noqa: E402
from rl_stock.risk import compute_max_drawdown  # noqa: E402

fp = cmp.fp

LONG_C = "#1565c0"
SHORT_C = "#c62828"
HOLD_C = "#2e7d32"
WARN_C = "#e65100"
INK = "#1a1a1a"
MUTED = "#5b6470"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "nasdaq100.xlsx")
    parser.add_argument("--t1-dir", type=Path, default=ROOT / "outputs" / "t1")
    parser.add_argument("--position-dir", type=Path, default=ROOT / "outputs" / "position")
    parser.add_argument("--trend-dir", type=Path, default=ROOT / "outputs" / "position_trend")
    parser.add_argument("--html", type=Path, default=ROOT / "position_management.html")
    parser.add_argument(
        "--fig-dir", type=Path, default=ROOT / "outputs" / "position_demo" / "figures"
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


def configure_light() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#cfcfcf",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": "#ededed",
            "grid.linewidth": 0.8,
        }
    )


def clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Episode replay (capture per-step events)
# --------------------------------------------------------------------------- #
def replay_continuous(model: Any, env: Any, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    price, position, equity, stop, lock = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        price.append(float(env.prices[info["tick"]]))
        position.append(float(info["position_ratio"]))
        equity.append(float(info["equity"]))
        stop.append(bool(info.get("stop_triggered", False)))
        lock.append(bool(info.get("dd_locked", False)))
        done = term or trunc
    return {
        "price": np.array(price),
        "position": np.array(position),
        "equity": np.array(equity),
        "stop": np.array(stop),
        "lock": np.array(lock),
    }


def replay_discrete_exposure(model: Any, env: Any, seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    done = False
    info: dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    positions = info.get("position_history", []) or []
    return np.array([fp.position_to_exposure(p) for p in positions], dtype=float)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_in_action(rec: dict[str, np.ndarray], label: str, stop_pct: float, path: Path) -> None:
    steps = np.arange(rec["position"].size)
    pos = rec["position"] * 100.0
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.12}
    )
    ax1.plot(steps, rec["price"], color="#444", linewidth=1.3)
    ax1.set_ylabel("Index price")
    ax1.set_title(f"Position management in action — {label}", loc="left")

    ax2.fill_between(steps, 0, pos, where=pos >= 0, color=LONG_C, alpha=0.45, label="Long exposure")
    ax2.fill_between(steps, 0, pos, where=pos < 0, color=SHORT_C, alpha=0.45, label="Short exposure")
    ax2.axhline(0, color="#999", linewidth=0.8)
    ax2.set_ylabel("Position %")
    ax2.set_ylim(-110, 110)
    ax2.set_xlabel("Trading day (test window)")

    stops = np.flatnonzero(rec["stop"])
    if stops.size:
        ax1.scatter(stops, rec["price"][stops], marker="v", color=SHORT_C, s=45, zorder=5)
        ax1.annotate(
            f"Stop-loss (−{stop_pct * 100:.0f}%) → sell to cash",
            (stops[0], rec["price"][stops[0]]),
            textcoords="offset points",
            xytext=(8, 16),
            color=SHORT_C,
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=SHORT_C, lw=1.1),
        )

    locks = np.flatnonzero(rec["lock"])
    if locks.size:
        ax2.axvspan(locks[0], locks[-1], color=WARN_C, alpha=0.12)
        ax2.annotate(
            "Drawdown lock: no new exposure",
            (locks[0], -95),
            color=WARN_C,
            fontsize=9,
            fontweight="bold",
        )

    peak = int(np.argmax(pos))
    ax2.annotate(
        f"Scaled up to {pos[peak]:.0f}%",
        (peak, pos[peak]),
        textcoords="offset points",
        xytext=(6, -14),
        color=LONG_C,
        fontsize=9,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", frameon=False, fontsize=8, ncols=2)
    for ax in (ax1, ax2):
        clean(ax)
    save(fig, path)


def chart_exposure(disc: np.ndarray, cont: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axhline(100, color=HOLD_C, linewidth=1.4, linestyle="--", label="Buy & Hold — always 100%")
    if disc.size:
        ax.step(np.arange(disc.size), disc * 100.0, color=WARN_C, linewidth=1.4,
                where="post", label="Without mgmt — binary 0% / 100%")
    ax.plot(np.arange(cont.size), cont * 100.0, color=LONG_C, linewidth=1.8,
            label="With mgmt — graded 0–100%")
    ax.set_ylabel("Capital in market %")
    ax.set_xlabel("Trading day")
    ax.set_ylim(-110, 120)
    ax.set_title("How much capital is in the market", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncols=3)
    clean(ax)
    save(fig, path)


def chart_distribution(disc: np.ndarray, cont: np.ndarray, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    a.hist(cont * 100.0, bins=20, color=LONG_C, alpha=0.85, edgecolor="white")
    a.set_title("With mgmt — graded sizes", loc="left")
    a.set_xlabel("Position %")
    a.set_ylabel("Days")
    a.set_xlim(-110, 110)
    if disc.size:
        b.hist(disc * 100.0, bins=[-5, 5, 95, 105], color=WARN_C, alpha=0.85, edgecolor="white")
    b.set_title("Without mgmt — only 0% or 100%", loc="left")
    b.set_xlabel("Position %")
    b.set_xlim(-10, 110)
    for ax in (a, b):
        clean(ax)
    save(fig, path)


def chart_equity(bh_yuan: np.ndarray, cont_yuan: np.ndarray, initial: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axhline(initial, color="#bbbbbb", linewidth=1.0, linestyle="--")
    ax.plot(np.arange(bh_yuan.size), bh_yuan, color=HOLD_C, linewidth=1.8, label="Without mgmt — Buy & Hold")
    ax.plot(np.arange(cont_yuan.size), cont_yuan, color=LONG_C, linewidth=1.8, label="With mgmt — RL position sizing")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x / 1e3:.0f}k"))
    ax.set_ylabel(f"Equity (start ¥{initial:,.0f})")
    ax.set_xlabel("Trading day")
    ax.set_title("Equity: with vs without position management", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    clean(ax)
    save(fig, path)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def mechanism_cards(stop_pct: float, dd_pct: float) -> str:
    cards = [
        ("Graded sizing", "Hold any size, not just all-in / all-out.",
         "Moderate signal → hold 60%, not 100%."),
        ("Volatility scaling", "Bigger size when calm, smaller when choppy.",
         "Low volatility → scale the target position up; high → shrink it."),
        ("Stop-loss", "Cut a losing trade automatically.",
         f"Trade down {stop_pct * 100:.0f}% → sell to cash immediately."),
        ("Drawdown lock", "Protect the account after a slump.",
         f"Account down {dd_pct * 100:.0f}% from peak → stop adding, only reduce."),
    ]
    return "".join(
        f'<div class="card"><h3>{t}</h3><p>{d}</p><p class="ex">e.g. {e}</p></div>'
        for t, d, e in cards
    )


def fmt_pct(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.1f}%"


def fmt_ret(v: float) -> str:
    return f"{v * 100:+.1f}%"


def result_table(rows: list[tuple[str, str, str, str, str, bool]]) -> str:
    head = (
        "<tr><th>Setup</th><th>Position sizing</th><th>Risk controls</th>"
        "<th>Return</th><th>Max drawdown</th></tr>"
    )
    cells = []
    for name, sizing, controls, ret, dd, on in rows:
        tr = '<tr class="on">' if on else "<tr>"
        cells.append(
            f"{tr}<td>{name}</td><td>{sizing}</td>"
            f"<td>{controls}</td><td>{ret}</td><td>{dd}</td></tr>"
        )
    return head + "".join(cells)


STYLE = """
  * { box-sizing:border-box; }
  body { margin:0; background:#ffffff; color:#1a1a1a; line-height:1.5;
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:980px; margin:0 auto; padding:44px 24px 72px; }
  h1 { font-size:2rem; margin:0 0 .3rem; letter-spacing:-.3px; }
  h2 { font-size:1.25rem; margin:40px 0 14px; }
  p.sub { color:#5b6470; margin:.2rem 0; font-size:1.02rem; }
  .eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:.72rem;
    font-weight:800; color:#1565c0; }
  .vs { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }
  .pane { border:1px solid #e6e6e6; border-radius:14px; padding:18px 20px; background:#fafbfc; }
  .pane.on { border-color:#1565c0; background:#f3f8ff; }
  .pane .lab { font-weight:800; font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
  .pane.off .lab { color:#e65100; }
  .pane.on .lab { color:#1565c0; }
  .pane .big { font-size:1.45rem; font-weight:800; margin:.3rem 0; }
  .pane p { color:#5b6470; margin:.2rem 0 0; font-size:.92rem; }
  .cards { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
  .card { border:1px solid #e9e9e9; border-radius:12px; padding:16px 18px; background:#fff; }
  .card h3 { margin:0 0 .3rem; font-size:1.03rem; }
  .card p { color:#5b6470; margin:.15rem 0; font-size:.92rem; }
  .card .ex { color:#1565c0; font-weight:600; }
  figure { margin:18px 0 0; border:1px solid #ededed; border-radius:14px; padding:12px; background:#fff; }
  figure img { width:100%; display:block; border-radius:8px; }
  figcaption { color:#5b6470; font-size:.88rem; margin-top:10px; }
  table { width:100%; border-collapse:collapse; margin-top:14px; font-size:.92rem;
    border:1px solid #e6e6e6; border-radius:12px; overflow:hidden; }
  th, td { padding:10px 12px; border-bottom:1px solid #eee; text-align:left; }
  thead, tr:first-child th { background:#f5f7fa; font-weight:700; }
  tr.on td { background:#f3f8ff; font-weight:600; }
  footer { margin-top:40px; color:#8a929c; font-size:.82rem; border-top:1px solid #eee; padding-top:16px; }
  code { background:#f0f2f5; padding:1px 6px; border-radius:5px; font-size:.85em; }
"""


def build_html(args, figs, rows, test_period, banners) -> str:
    banner_html = "".join(f'<p class="sub" style="color:#e65100;">{b}</p>' for b in banners)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Position Management — Off vs On</title>
<style>{STYLE}</style>
</head>
<body>
  <div class="wrap">
    <div class="eyebrow">RL-Stock · TUM</div>
    <h1>What does position management do?</h1>
    <p class="sub">Turning the binary buy/sell action into a graded, risk-controlled position.</p>
    <p class="sub">Start ¥{args.initial_cash:,.0f} · test window {test_period}</p>
    {banner_html}

    <h2>Off vs On</h2>
    <div class="vs">
      <div class="pane off">
        <div class="lab">Without position management</div>
        <div class="big">Buy / Sell</div>
        <p>Binary action — you are either fully in (100%) or fully out (0%). No sizing, no
        automatic risk limits.</p>
      </div>
      <div class="pane on">
        <div class="lab">With position management</div>
        <div class="big">Hold 0–100% (or short)</div>
        <p>The agent chooses <i>how much</i> to hold each day and adds stop-loss, drawdown
        lock and volatility scaling.</p>
      </div>
    </div>

    <h2>The four things it does</h2>
    <div class="cards">{mechanism_cards(args.stop_loss_pct, args.max_drawdown_pct)}</div>

    <h2>Example: the agent in action</h2>
    <figure>
      <img alt="position in action" src="{figs["action"]}">
      <figcaption>One test episode. Top: index price (▼ = a stop-loss sell). Bottom: how much
      the agent holds each day — it scales the position up and down instead of staying all-in.</figcaption>
    </figure>

    <h2>How much capital is in the market</h2>
    <figure>
      <img alt="exposure" src="{figs["exposure"]}">
      <figcaption>Without management the exposure is flat (Buy &amp; Hold) or a 0/100 switch
      (discrete). With management it varies smoothly with the agent's conviction.</figcaption>
    </figure>

    <h2>Position sizes actually used</h2>
    <figure>
      <img alt="distribution" src="{figs["dist"]}">
      <figcaption>The managed agent uses a whole range of sizes; the discrete agent only ever
      uses 0% or 100%.</figcaption>
    </figure>

    <h2>Result: with vs without</h2>
    <figure>
      <img alt="equity" src="{figs["equity"]}">
      <figcaption>Equity from ¥{args.initial_cash:,.0f} on the test window.</figcaption>
    </figure>
    <table>{result_table(rows)}</table>
    <p class="sub" style="margin-top:10px;">This 2024–2026 test window was a strong bull market, so
    staying fully invested (Buy &amp; Hold) won on return; position management trades some upside for
    control over sizing and downside — its real value shows in choppier or falling markets.</p>

    <footer>
      Self-contained page generated from saved training artifacts (deterministic replay).
      Single seed; illustrative. Regenerate with <code>python scripts/generate_position_demo.py</code>.
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
    configure_light()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fp.require_file(args.data_path, "market data")
    df = load_market_data(args.data_path)
    _, test_bound = split_frame_bounds(df, window_size=args.window_size, test_start_date=args.test_start_date)
    test_period = f"{df.index[test_bound[0]].date()} → {df.index[-1].date()}"

    banners: list[str] = []

    # --- With position management: pick the most illustrative continuous agent. ---
    base_map, base_env, base_banner = cmp.position_strategy_map(args.position_dir, df, args, trend=False)
    if base_banner:
        banners.append(base_banner)

    demo_label, demo_rec, demo_return = "n/a", None, float("nan")
    best_score = -1.0
    for algo, result in base_map.items():
        model = cmp.fp.load_model(algo, args.position_dir, base_env)
        rec = replay_continuous(model, base_env, args.seed)
        score = float(np.std(rec["position"])) + 0.05 * int(rec["stop"].sum())
        if score > best_score:
            best_score = score
            demo_label, demo_rec = f"{algo} agent", rec
            demo_return = result.metrics.get("return", float("nan"))

    # --- Without position management: discrete buy/sell agent. ---
    disc_expo = np.asarray([])
    disc_return = float("nan")
    try:
        disc_results = cmp.fp.evaluate_output_dir("t1", args.t1_dir, df, args)
        disc_best = cmp.fp.choose_best(disc_results, "t1")
        disc_env = cmp.fp.build_test_env("t1", df, args)
        disc_model = cmp.fp.load_model(disc_best.strategy, args.t1_dir, disc_env)
        disc_expo = replay_discrete_exposure(disc_model, disc_env, args.seed)
        disc_return = disc_best.metrics.get("return", float("nan"))
        disc_name = f"Discrete buy/sell ({disc_best.strategy})"
    except (FileNotFoundError, ValueError) as exc:
        disc_name = "Discrete buy/sell"
        banners.append(f"Discrete baseline unavailable ({exc}).")

    # --- Buy & Hold reference. ---
    bh_curve, bh_metrics = cmp.buy_hold_series(df, args)
    bh_yuan = cmp.to_yuan(bh_curve, args.initial_cash)

    if demo_rec is None:
        raise SystemExit(
            "No continuous (position-management) models found. Train them with: "
            "python train.py --env-type position --initial-cash 100000 --output-dir outputs/position"
        )

    # --- Charts. ---
    figs: dict[str, str] = {}
    action_path = args.fig_dir / "position_in_action.png"
    chart_in_action(demo_rec, demo_label, args.stop_loss_pct, action_path)
    figs["action"] = cmp.fp.image_uri(action_path)

    exposure_path = args.fig_dir / "exposure.png"
    chart_exposure(disc_expo, demo_rec["position"], exposure_path)
    figs["exposure"] = cmp.fp.image_uri(exposure_path)

    dist_path = args.fig_dir / "distribution.png"
    chart_distribution(disc_expo, demo_rec["position"], dist_path)
    figs["dist"] = cmp.fp.image_uri(dist_path)

    cont_yuan = demo_rec["equity"] / demo_rec["equity"][0] * args.initial_cash
    equity_path = args.fig_dir / "equity.png"
    chart_equity(bh_yuan, cont_yuan, args.initial_cash, equity_path)
    figs["equity"] = cmp.fp.image_uri(equity_path)

    # --- Result table. ---
    cont_dd = compute_max_drawdown(demo_rec["equity"])
    rows = [
        ("Buy & Hold", "Fixed 100%", "None", fmt_ret(bh_metrics["return"]), fmt_pct(bh_metrics["max_drawdown"]), False),
        (disc_name, "Binary 0% / 100%", "None", fmt_ret(disc_return), "—", False),
        (f"Position mgmt ({demo_label})", "Graded 0–100% + short",
         "Stop-loss · DD-lock · vol-scaling", fmt_ret(demo_return), fmt_pct(cont_dd), True),
    ]

    document = build_html(args, figs, rows, test_period, banners)
    args.html.write_text(document, encoding="utf-8")

    print(f"Featured position-management agent: {demo_label}")
    print(f"  stop-loss events: {int(demo_rec['stop'].sum())}  "
          f"position std: {np.std(demo_rec['position']):.3f}  return: {fmt_ret(demo_return)}")
    for banner in banners:
        print(f"  [note] {banner}")
    print(f"\nfigures: {args.fig_dir.relative_to(ROOT)}")
    print(f"html:    {args.html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
