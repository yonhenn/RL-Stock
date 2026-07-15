"""Streamlit dashboard entry point: compare RL trading algorithms.

Run with::

    streamlit run streamlit_app.py

Two pages (switch in the sidebar):

- **Algorithm Comparison**: per-experiment saved metrics plus a live equity
  replay of every algorithm.
- **Trade Analysis**: the raw NASDAQ-100 price with buy/sell markers, a
  cumulative-return comparison, position ribbons, and holding-time stats for any
  mix of experiment x algorithm combinations, overlaid on shared charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_stock.dashboard import page_compare, page_trades  # noqa: E402

st.set_page_config(page_title="RL Trading Dashboard", layout="wide")

navigation = st.navigation(
    [
        st.Page(
            page_compare.render,
            title="Algorithm Comparison",
            url_path="compare",
            default=True,
        ),
        st.Page(page_trades.render, title="Trade Analysis", url_path="trades"),
    ]
)
navigation.run()
