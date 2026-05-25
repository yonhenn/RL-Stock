"""Experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path
    test_start_date: str = "2024-01-01"
    window_size: int = 30
    total_timesteps: int = 100_000
    num_eval_episodes: int = 20
    seed: int = 42
    output_dir: Path = Path("outputs")
    show_plots: bool = False
    save_models: bool = True
