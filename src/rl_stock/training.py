"""Training orchestration for the Shanghai Index experiment."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import A2C, PPO

from rl_stock.config import ExperimentConfig
from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs import ShanghaiIndexEnvT1
from rl_stock.evaluation import ProgressBarCallback, evaluate_model, print_stats


@dataclass
class ExperimentArtifacts:
    df: object
    train_env: object
    test_env: object
    models: dict
    results: dict
    train_frame_bound: tuple[int, int]
    test_frame_bound: tuple[int, int]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_a2c(train_env, seed: int, total_timesteps: int):
    """Train the optimized A2C model from the notebook."""
    model = A2C(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback())
    return model


def train_ppo(train_env, seed: int, total_timesteps: int):
    """Train the optimized PPO model from the notebook."""
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback())
    return model


def run_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    """Load data, create environments, train models, and evaluate them."""
    set_seed(config.seed)

    df = load_market_data(config.data_path)
    train_frame_bound, test_frame_bound = split_frame_bounds(
        df,
        window_size=config.window_size,
        test_start_date=config.test_start_date,
    )

    train_env = ShanghaiIndexEnvT1(
        df=df,
        window_size=config.window_size,
        frame_bound=train_frame_bound,
    )
    test_env = ShanghaiIndexEnvT1(
        df=df,
        window_size=config.window_size,
        frame_bound=test_frame_bound,
    )

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Training frame: {train_frame_bound}")
    print(f"Test frame: {test_frame_bound}")
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space}")

    results = {}
    models = {}

    print("\nEvaluating random strategy baseline...")
    random_rewards, random_profits = evaluate_model(
        None,
        test_env,
        config.num_eval_episodes,
        config.seed,
    )
    results["Random"] = {"rewards": random_rewards, "profits": random_profits}
    print_stats("Random", random_rewards, random_profits)

    print("\n" + "=" * 50)
    print("Training A2C Model (Optimized Hyperparameters)")
    print("=" * 50)
    a2c_model = train_a2c(train_env, config.seed, config.total_timesteps)
    models["A2C"] = a2c_model
    a2c_rewards, a2c_profits = evaluate_model(
        a2c_model,
        test_env,
        config.num_eval_episodes,
        config.seed,
    )
    results["A2C"] = {"rewards": a2c_rewards, "profits": a2c_profits}
    print_stats("A2C", a2c_rewards, a2c_profits)

    print("\n" + "=" * 50)
    print("Training PPO Model (Optimized Hyperparameters)")
    print("=" * 50)
    ppo_model = train_ppo(train_env, config.seed, config.total_timesteps)
    models["PPO"] = ppo_model
    ppo_rewards, ppo_profits = evaluate_model(
        ppo_model,
        test_env,
        config.num_eval_episodes,
        config.seed,
    )
    results["PPO"] = {"rewards": ppo_rewards, "profits": ppo_profits}
    print_stats("PPO", ppo_rewards, ppo_profits)

    return ExperimentArtifacts(
        df=df,
        train_env=train_env,
        test_env=test_env,
        models=models,
        results=results,
        train_frame_bound=train_frame_bound,
        test_frame_bound=test_frame_bound,
    )
