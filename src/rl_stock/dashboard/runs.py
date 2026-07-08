"""Discover experiment runs and parse their ``summary.csv`` metric tables.

This module is intentionally dependency-light (pandas only) so the metric
comparison view works even when the heavy RL stack (torch / stable-baselines3)
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Model archives a run may contain, keyed by the algorithm label used elsewhere.
KNOWN_ALGORITHMS: tuple[str, ...] = ("A2C", "PPO", "SAC", "TD3", "DDPG")

# Columns in ``summary.csv`` stored as percent strings (e.g. ``"-11.86%"``).
_PERCENT_COLUMNS: tuple[str, ...] = ("Return Rate", "Max DD")

# Columns that should be numeric floats when present.
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "Average Profit",
    "Highest",
    "Lowest",
    "Standard Deviation",
    "Sharpe",
    "Calmar",
)


@dataclass(frozen=True)
class RunInfo:
    """One experiment output folder and the artifacts it exposes."""

    label: str
    path: Path
    summary_path: Path
    model_paths: tuple[tuple[str, Path], ...]
    returns_png: Path | None

    @property
    def has_models(self) -> bool:
        return len(self.model_paths) > 0

    @property
    def model_map(self) -> dict[str, Path]:
        return dict(self.model_paths)


def _percent_to_float(series: pd.Series) -> pd.Series:
    """Convert a ``"12.34%"`` string column to a numeric percentage (``12.34``)."""
    cleaned = series.astype(str).str.strip().str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce")


def parse_summary_csv(path: str | Path) -> pd.DataFrame:
    """Read a ``summary.csv`` into a tidy, fully numeric frame.

    Percent columns (``Return Rate``, ``Max DD``) are converted to floats in
    percentage points. Missing optional columns are tolerated. ``Strategy`` is
    preserved as-is and used as the natural key.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file has no ``Strategy`` column.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")

    raw = pd.read_csv(path)
    if "Strategy" not in raw.columns:
        raise ValueError(f"{path} is missing the required 'Strategy' column")

    frame = raw.copy()
    for column in _PERCENT_COLUMNS:
        if column in frame.columns:
            frame[column] = _percent_to_float(frame[column])
    for column in _NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["Strategy"] = frame["Strategy"].astype(str)
    return frame.reset_index(drop=True)


def _find_models(run_dir: Path) -> tuple[tuple[str, Path], ...]:
    """Return ``(ALGO, path)`` pairs for model archives found in ``run_dir``."""
    found: list[tuple[str, Path]] = []
    for algo in KNOWN_ALGORITHMS:
        candidate = run_dir / f"{algo.lower()}.zip"
        if candidate.is_file():
            found.append((algo, candidate))
    return tuple(found)


def _label_for(run_dir: Path, outputs_root: Path) -> str:
    """Human-friendly label: the folder name, or ``outputs (root)`` for the top level."""
    if run_dir.resolve() == outputs_root.resolve():
        return f"{outputs_root.name} (root)"
    try:
        return str(run_dir.relative_to(outputs_root))
    except ValueError:
        return run_dir.name


def discover_runs(outputs_dir: str | Path) -> list[RunInfo]:
    """Find every folder under ``outputs_dir`` that holds a ``summary.csv``.

    The search is one level deep plus the root itself, matching how the
    training CLI writes results. Runs are returned sorted by trained-model count
    (descending) so the fullest algorithm comparison leads, then alphabetically.
    """
    outputs_root = Path(outputs_dir)
    if not outputs_root.is_dir():
        return []

    candidates: list[Path] = []
    if (outputs_root / "summary.csv").is_file():
        candidates.append(outputs_root)
    candidates.extend(
        sorted(
            child
            for child in outputs_root.iterdir()
            if child.is_dir() and (child / "summary.csv").is_file()
        )
    )

    runs: list[RunInfo] = []
    for run_dir in candidates:
        returns_png = run_dir / "strategy_returns.png"
        runs.append(
            RunInfo(
                label=_label_for(run_dir, outputs_root),
                path=run_dir,
                summary_path=run_dir / "summary.csv",
                model_paths=_find_models(run_dir),
                returns_png=returns_png if returns_png.is_file() else None,
            )
        )

    # Richest comparison first: the run with the most trained models leads, so
    # the default view already shows the fullest algorithm line-up (e.g. the
    # 5-way Random/PPO/SAC/TD3/DDPG runs). Ties break alphabetically; runs with
    # no models sort last.
    runs.sort(key=lambda run: (-len(run.model_paths), run.label.lower()))
    return runs
