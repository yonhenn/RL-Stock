"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """Generate deterministic OHLC data long enough for window+frame tests."""
    rng = np.random.default_rng(seed=7)
    n = 300
    base = 3000 + np.cumsum(rng.normal(0, 5, n))
    high = base + rng.uniform(1, 10, n)
    low = base - rng.uniform(1, 10, n)
    open_ = base + rng.normal(0, 2, n)
    close = base + rng.normal(0, 2, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.Index(dates, name="Date"),
    )
    return df


@pytest.fixture
def trending_ohlc() -> pd.DataFrame:
    """Deterministic, strongly *uptrending* OHLC data (high ADX, STRONG_UP).

    A near-linear ramp with mild noise so the trend judgment resolves to a
    confident uptrend -- used to exercise the trend-prior residual policy and
    regime-aware caps, which the noisy random-walk ``synthetic_ohlc`` leaves
    in a SIDEWAYS regime.
    """
    rng = np.random.default_rng(seed=3)
    n = 240
    base = np.linspace(3000.0, 4800.0, n) + rng.normal(0, 3, n)
    high = base + rng.uniform(2, 8, n)
    low = base - rng.uniform(2, 8, n)
    open_ = base + rng.normal(0, 2, n)
    close = base + rng.normal(0, 2, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.Index(dates, name="Date"),
    )
    return df
