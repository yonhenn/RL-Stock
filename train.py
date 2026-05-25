"""Command-line entry point for the Shanghai Index RL experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL trading agents for Shanghai Index data.")
    parser.add_argument("--data-path", type=Path, default=Path("output.xlsx"))
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--skip-trade-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from rl_stock.config import ExperimentConfig
        from rl_stock.evaluation import print_summary
        from rl_stock.training import run_experiment
        from rl_stock.visualization import configure_matplotlib, plot_strategy_returns, visualize_trading
    except ModuleNotFoundError as exc:
        missing_module = exc.name
        raise SystemExit(
            f"Missing dependency: {missing_module}. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    configure_matplotlib()

    config = ExperimentConfig(
        data_path=args.data_path,
        test_start_date=args.test_start_date,
        window_size=args.window_size,
        total_timesteps=args.total_timesteps,
        num_eval_episodes=args.num_eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        save_models=not args.no_save_models,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = run_experiment(config)
    test_start = artifacts.df.index[artifacts.test_frame_bound[0]]
    test_end = artifacts.df.index[-1]
    test_period = f"{test_start.date()} to {test_end.date()}"

    summary_df = print_summary(
        artifacts.results,
        total_timesteps=config.total_timesteps,
        test_period=test_period,
    )
    summary_df.to_csv(config.output_dir / "summary.csv", index=False)

    plot_strategy_returns(
        artifacts.results,
        save_path=config.output_dir / "strategy_returns.png",
        show=config.show_plots,
    )

    if config.save_models:
        for model_name, model in artifacts.models.items():
            model.save(config.output_dir / model_name.lower())

    if not args.skip_trade_plots:
        strategies = {"Random": None, **artifacts.models}
        for model_name, model in strategies.items():
            visualize_trading(
                model,
                artifacts.test_env,
                model_name,
                seed=config.seed,
                save_path=config.output_dir / f"trades_{model_name.lower()}.png",
                show=config.show_plots,
            )

    print(f"\nOutputs written to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
