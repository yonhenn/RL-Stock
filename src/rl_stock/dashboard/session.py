"""Shared Streamlit session helpers (replay caching) used across pages."""

from __future__ import annotations

import streamlit as st

from rl_stock.dashboard.replay import EnvSettings, ReplayResult, replay_run
from rl_stock.dashboard.runs import RunInfo


def _model_mtimes(run: RunInfo) -> tuple[tuple[str, float], ...]:
    """Stable cache key: each model path with its modification time."""
    return tuple((str(path), path.stat().st_mtime) for _, path in run.model_paths)


def cached_replay(run: RunInfo, settings: EnvSettings) -> ReplayResult:
    """Replay a run once per (models, settings) combination, held in session state.

    Model loading and episode replay are the expensive parts; keeping the result
    in ``st.session_state`` means switching pages, metrics, or selections re-uses
    the same curves instead of reloading torch models every rerun.
    """
    cache: dict[str, ReplayResult] = st.session_state.setdefault("_equity_cache", {})
    key = f"{run.path}|{_model_mtimes(run)}|{settings}"
    if key not in cache:
        cache[key] = replay_run(run, settings)
    return cache[key]
