"""Streamlit dashboard for comparing RL trading algorithms.

The package is split into small, focused modules:

- :mod:`rl_stock.dashboard.runs` discovers experiment output folders and parses
  their ``summary.csv`` metric tables (pure, no heavy dependencies).
- :mod:`rl_stock.dashboard.replay` reloads the trained models and replays a
  deterministic episode to reconstruct each strategy's equity curve.
- :mod:`rl_stock.dashboard.figures` builds the Plotly figures shared by the app.

The Streamlit entry point lives in ``streamlit_app.py`` at the project root.
"""

from __future__ import annotations

from rl_stock.dashboard.runs import RunInfo, discover_runs, parse_summary_csv

__all__ = ["RunInfo", "discover_runs", "parse_summary_csv"]
