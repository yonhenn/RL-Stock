"""Command-line entry point for the Shanghai Index RL experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ALGOS = {
    "t1": "A2C,PPO",
    # SAC first: off-policy, entropy-regularized actor-critic is the natural fit
    # for continuous position sizing (sample-efficient exploration of sizes).
    "position": "SAC,PPO",
}


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

    parser.add_argument(
        "--env-type",
        choices=["t1", "position"],
        default="t1",
        help="t1 = legacy discrete env, position = continuous risk-managed env.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated list of algorithms (A2C,PPO,SAC,TD3). Defaults depend on --env-type.",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.15)
    parser.add_argument("--trade-fee-rate", type=float, default=0.001)
    parser.add_argument("--vol-target", type=float, default=0.02)
    parser.add_argument("--drawdown-coeff", type=float, default=0.5)
    parser.add_argument("--turnover-coeff", type=float, default=0.1)
    parser.add_argument("--sharpe-coeff", type=float, default=1.0)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument(
        "--use-trend-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add trend-aware state features (SMA slope, BB width, ADX, MACD cross, "
        "trend score, regime). Only affects --env-type position.",
    )
    parser.add_argument(
        "--use-trend-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Residual policy: anchor the position to a trend-following base from the "
        "regime judgment and let the agent tilt around it. --env-type position only.",
    )
    parser.add_argument("--trend-base-exposure", type=float, default=0.8)
    parser.add_argument("--residual-band", type=float, default=0.5)
    parser.add_argument(
        "--regime-risk-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cap counter-trend exposure inside a strongly trending regime.",
    )
    parser.add_argument("--counter-trend-cap", type=float, default=0.2)
    parser.add_argument("--trend-align-coeff", type=float, default=0.0)
    parser.add_argument("--adx-trend-threshold", type=float, default=20.0)
    parser.add_argument("--adx-strong-threshold", type=float, default=40.0)
    return parser.parse_args()


def _resolve_algos(env_type: str, raw: str | None) -> tuple[str, ...]:
    text = raw if raw is not None else DEFAULT_ALGOS[env_type]
    return tuple(name.strip().upper() for name in text.split(",") if name.strip())


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

    algorithms = _resolve_algos(args.env_type, args.algorithms)
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
        env_type=args.env_type,
        algorithms=algorithms,
        initial_cash=args.initial_cash,
        max_position=args.max_position,
        stop_loss_pct=args.stop_loss_pct,
        max_drawdown_pct=args.max_drawdown_pct,
        trade_fee_rate=args.trade_fee_rate,
        vol_target=args.vol_target,
        drawdown_coeff=args.drawdown_coeff,
        turnover_coeff=args.turnover_coeff,
        sharpe_coeff=args.sharpe_coeff,
        dsr_eta=args.dsr_eta,
        use_trend_features=args.use_trend_features,
        use_trend_prior=args.use_trend_prior,
        trend_base_exposure=args.trend_base_exposure,
        residual_band=args.residual_band,
        regime_risk_scaling=args.regime_risk_scaling,
        counter_trend_cap=args.counter_trend_cap,
        trend_align_coeff=args.trend_align_coeff,
        adx_trend_threshold=args.adx_trend_threshold,
        adx_strong_threshold=args.adx_strong_threshold,
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
