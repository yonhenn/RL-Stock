"""Data loading and train/test split helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RENAME_COLUMNS = {
    "open": "Open",
    "close": "Close",
    "high": "High",
    "low": "Low",
    "vol": "Volume",
}


def load_market_data(data_path: str | Path) -> pd.DataFrame:
    """Load and normalize the Shanghai Index Excel dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_excel(data_path)
    df = df.copy().rename(columns=RENAME_COLUMNS)

    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)
        dates = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if dates.isna().any():
            dates = pd.to_datetime(df["trade_date"], errors="coerce")
        df["Date"] = dates
    elif "Date" in df.columns:
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("Data must contain either 'trade_date' or 'Date'.")

    if df["Date"].isna().any():
        raise ValueError("Some dates could not be parsed.")

    required_columns = {"Open", "Close", "High", "Low"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.set_index("Date")
    return df


def split_frame_bounds(
    df: pd.DataFrame,
    window_size: int,
    test_start_date: str = "2024-01-01",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return gym-anytrading frame bounds for train and test periods."""
    df_reset = df.reset_index()
    matching_rows = df_reset[df_reset["Date"] >= test_start_date]
    if matching_rows.empty:
        raise ValueError(f"No rows found on or after test_start_date={test_start_date!r}.")

    test_start_idx = int(matching_rows.index[0])
    if test_start_idx <= window_size:
        raise ValueError(
            "test_start_date leaves too little training history for "
            f"window_size={window_size}."
        )

    total_len = len(df)
    return (window_size, test_start_idx), (test_start_idx, total_len)
