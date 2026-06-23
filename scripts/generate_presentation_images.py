"""Generate and embed presentation figures from the NASDAQ-100 index dataset."""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "nasdaq100.xlsx"
OUT_DIR = ROOT / "assets" / "generated"
HTML_PATH = ROOT / "presentation.html"
TEST_START = pd.Timestamp("2024-01-01")


def load_market_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").set_index("date")
    return df


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1
    return float(-drawdown.min())


def sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))


def sortino_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(returns) < 2 or downside.empty:
        return 0.0
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev == 0:
        return 0.0
    return float(returns.mean() / downside_dev * np.sqrt(252))


def figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.labelcolor": "#333333",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "text.color": "#111111",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e8e8e8", linewidth=0.6)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def generate_market_overview(df: pd.DataFrame, path: Path) -> None:
    close = df["close"]
    volume = df["vol"]
    recent = df.loc[df.index >= TEST_START]

    fig = plt.figure(figsize=(8.6, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 0.9, 0.12])
    ax_price = fig.add_subplot(gs[0])
    ax_recent = fig.add_subplot(gs[1], sharex=None)
    ax_note = fig.add_subplot(gs[2])

    ax_price.plot(close.index, close.values, color="#111111", linewidth=1.35)
    ax_price.axvspan(TEST_START, close.index.max(), color="#f0f0f0", zorder=0)
    ax_price.set_title("NASDAQ-100 Index close, 2014-2026", loc="left")
    ax_price.set_ylabel("Index")
    ax_price.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    clean_axis(ax_price)

    recent_close = recent["close"]
    ax_recent.plot(recent_close.index, recent_close / recent_close.iloc[0], color="#111111", linewidth=1.2, label="Close indexed")
    ax_recent.plot(
        recent_close.index,
        moving_average(recent_close, 60) / recent_close.iloc[0],
        color="#888888",
        linewidth=1,
        label="60D MA",
    )
    ax_recent.bar(
        recent.index,
        volume.loc[recent.index] / volume.loc[recent.index].max() * 0.18,
        width=1,
        color="#d0d0d0",
        label="Volume scaled",
    )
    ax_recent.set_title("Test window snapshot from 2024", loc="left")
    ax_recent.set_ylabel("Indexed")
    ax_recent.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_recent.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    clean_axis(ax_recent)
    ax_recent.legend(frameon=False, loc="upper left", ncols=3)

    ax_note.axis("off")
    ax_note.text(
        0,
        0.5,
        f"{len(df):,} daily bars | {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d} | source: nasdaq100.xlsx",
        fontsize=8,
        color="#555555",
        va="center",
    )
    save(fig, path)


def generate_feature_snapshot(df: pd.DataFrame, path: Path) -> None:
    recent = df.loc[df.index >= TEST_START].copy()
    close = recent["close"]
    ma20 = moving_average(close, 20)
    ma60 = moving_average(close, 60)
    std20 = close.rolling(20, min_periods=1).std().fillna(0)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    rsi14 = rsi(close)
    _, _, hist = macd(close)

    fig = plt.figure(figsize=(8.4, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.1, 0.75, 0.75, 0.75])
    ax_price = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
    ax_groups = fig.add_subplot(gs[3])

    ax_price.plot(close.index, close, color="#111111", linewidth=1.2, label="Close")
    ax_price.plot(ma20.index, ma20, color="#666666", linewidth=0.95, label="MA20")
    ax_price.plot(ma60.index, ma60, color="#aaaaaa", linewidth=0.95, label="MA60")
    ax_price.fill_between(close.index, lower.to_numpy(), upper.to_numpy(), color="#e5e5e5", label="Bollinger band")
    ax_price.set_title("Technical indicator snapshot, 2024-2026", loc="left")
    ax_price.set_ylabel("Index")
    clean_axis(ax_price)
    ax_price.legend(frameon=False, loc="upper left", ncols=4)

    ax_rsi.plot(rsi14.index, rsi14, color="#111111", linewidth=1)
    ax_rsi.axhline(70, color="#999999", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="#999999", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    clean_axis(ax_rsi)

    colors = np.where(hist >= 0, "#111111", "#b7b7b7")
    ax_macd.bar(hist.index, hist, color=colors, width=1.3)
    ax_macd.axhline(0, color="#111111", linewidth=0.8)
    ax_macd.set_ylabel("MACD")
    ax_macd.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    clean_axis(ax_macd)

    ax_groups.axis("off")
    groups = [
        ("Trend", "MA5/10/20/60 deviation"),
        ("Momentum", "return, RSI, MACD, 5D momentum"),
        ("Crossovers", "MA5-10, MA10-20, Bollinger position"),
        ("Price action", "amplitude, candle body"),
    ]
    for i, (title, body) in enumerate(groups):
        x = i / 4
        ax_groups.add_patch(plt.Rectangle((x + 0.01, 0.12), 0.225, 0.74, fill=False, edgecolor="#cfcfcf", linewidth=0.8))
        ax_groups.text(x + 0.025, 0.62, title, fontsize=8, weight="bold", transform=ax_groups.transAxes)
        ax_groups.text(x + 0.025, 0.34, body, fontsize=6.5, color="#555555", transform=ax_groups.transAxes, wrap=True)
    save(fig, path)


def generate_evaluation_snapshot(df: pd.DataFrame, path: Path) -> None:
    test = df.loc[df.index >= TEST_START].copy()
    returns = test["close"].pct_change().fillna(0)
    buyhold = (1 + returns).cumprod()

    rng = np.random.default_rng(42)
    runs = []
    for _ in range(250):
        positions = rng.choice([-1.0, 0.0, 1.0], size=len(returns), p=[0.25, 0.25, 0.5])
        turnover = np.abs(np.diff(np.insert(positions, 0, 0.0)))
        net = positions * returns.to_numpy() - 0.001 * turnover
        runs.append(np.cumprod(1 + net))
    random_equity = pd.DataFrame(np.column_stack(runs), index=test.index)
    random_median = random_equity.median(axis=1)
    random_p10 = random_equity.quantile(0.1, axis=1)
    random_p90 = random_equity.quantile(0.9, axis=1)

    bh_returns = buyhold.pct_change().fillna(0)
    random_returns = random_median.pct_change().fillna(0)
    metrics = pd.DataFrame(
        {
            "Buy & Hold": [
                buyhold.iloc[-1] - 1,
                sharpe_ratio(bh_returns),
                sortino_ratio(bh_returns),
                max_drawdown(buyhold),
            ],
            "Random median": [
                random_median.iloc[-1] - 1,
                sharpe_ratio(random_returns),
                sortino_ratio(random_returns),
                max_drawdown(random_median),
            ],
        },
        index=["Return", "Sharpe", "Sortino", "Max DD"],
    )

    fig = plt.figure(figsize=(8.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.25, 1])
    ax_eq = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_eq)
    ax_bar = fig.add_subplot(gs[1, 1])

    ax_eq.plot(buyhold.index, buyhold, color="#111111", linewidth=1.4, label="Buy & Hold")
    ax_eq.plot(random_median.index, random_median, color="#777777", linewidth=1.1, label="Random median")
    ax_eq.fill_between(random_median.index, random_p10.to_numpy(), random_p90.to_numpy(), color="#dddddd", label="Random p10-p90")
    ax_eq.set_title("Python-generated benchmark snapshot, 2024-2026", loc="left")
    ax_eq.set_ylabel("Equity")
    clean_axis(ax_eq)
    ax_eq.legend(frameon=False, loc="upper left", ncols=3)

    bh_dd = buyhold / buyhold.cummax() - 1
    random_dd = random_median / random_median.cummax() - 1
    ax_dd.fill_between(bh_dd.index, bh_dd * 100, 0, color="#111111", alpha=0.18, label="Buy & Hold DD")
    ax_dd.plot(random_dd.index, random_dd * 100, color="#777777", linewidth=1, label="Random DD")
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    clean_axis(ax_dd)

    values = [metrics.loc["Return", "Buy & Hold"] * 100, metrics.loc["Return", "Random median"] * 100]
    ax_bar.bar(["Buy & Hold", "Random"], values, color=["#111111", "#9a9a9a"], width=0.55)
    ax_bar.axhline(0, color="#111111", linewidth=0.8)
    ax_bar.set_title("Return comparison", loc="left")
    ax_bar.set_ylabel("%")
    clean_axis(ax_bar)
    for idx, value in enumerate(values):
        ax_bar.text(idx, value + (1 if value >= 0 else -1), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    save(fig, path)


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def replace_img_src(html: str, alt_text: str, uri: str) -> str:
    pattern = rf'(<img\b(?=[^>]*\balt="{re.escape(alt_text)}")[^>]*\bsrc=")[^"]*(")'
    new_html, count = re.subn(pattern, lambda match: f"{match.group(1)}{uri}{match.group(2)}", html)
    if count != 1:
        raise RuntimeError(f"Expected to replace one image with alt={alt_text!r}, replaced {count}.")
    return new_html


def embed_images(html_path: Path, generated: dict[str, Path]) -> None:
    html = html_path.read_text(encoding="utf-8")
    html = replace_img_src(
        html,
        "Python-generated NASDAQ-100 Index overview chart",
        image_uri(generated["market"]),
    )
    html = replace_img_src(
        html,
        "Python-generated technical indicator and feature snapshot",
        image_uri(generated["features"]),
    )
    html = replace_img_src(
        html,
        "Python-generated buy-and-hold and random baseline evaluation chart",
        image_uri(generated["evaluation"]),
    )
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true", help="Embed generated PNGs into presentation.html as data URIs.")
    args = parser.parse_args()

    figure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_market_data()

    generated = {
        "market": OUT_DIR / "market_overview.png",
        "features": OUT_DIR / "feature_snapshot.png",
        "evaluation": OUT_DIR / "evaluation_snapshot.png",
    }
    generate_market_overview(df, generated["market"])
    generate_feature_snapshot(df, generated["features"])
    generate_evaluation_snapshot(df, generated["evaluation"])

    if args.embed:
        embed_images(HTML_PATH, generated)

    for name, path in generated.items():
        print(f"{name}: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
