import sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path("/Users/rickchen/Antigravity/RL wit Nasdaq")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.bc_inference import BCAgent
from rl_stock.bc_model import BehaviorCloning
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib

def run_agent_history(agent, env):
    obs, _ = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    positions = []
    prices = []
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        positions.append(info.get("position_ratio", 0.0))
        prices.append(env.prices[info.get("tick")])
        
    return np.array(positions), np.array(prices)

def detect_signals(positions):
    buys = []
    sells = []
    
    for t in range(1, len(positions)):
        w_prev = positions[t-1]
        w_curr = positions[t]
        
        # Buy: Go from <= 0 to > 0, or from < 0 to >= 0
        if (w_prev <= 0 and w_curr > 0) or (w_prev < 0 and w_curr >= 0):
            buys.append(t)
        # Sell: Go from >= 0 to < 0, or from > 0 to <= 0
        elif (w_prev >= 0 and w_curr < 0) or (w_prev > 0 and w_curr <= 0):
            sells.append(t)
            
    return buys, sells

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot trading signals overlay.")
    parser.add_argument("--data-path", type=str, default="nasdaq100.xlsx")
    parser.add_argument("--dt-checkpoint", type=str, default="outputs/dt/dt_model.pt")
    parser.add_argument("--bc-checkpoint", type=str, default="outputs/bc/bc_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--test-start-date", type=str, default="2024-01-01")
    parser.add_argument(
        "--long-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to restrict action space to [0, 1] (long-only). Default is True.",
    )
    parser.add_argument("--target-rtg-low", type=float, default=0.3)
    parser.add_argument("--target-rtg-high", type=float, default=2.4)
    args = parser.parse_args()

    configure_matplotlib()
    
    data_path = Path(args.data_path)
    dt_ckpt_path = Path(args.dt_checkpoint)
    bc_ckpt_path = Path(args.bc_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date=args.test_start_date)
    
    risk_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=args.long_only,
    )
    env = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=risk_config)
    
    # Load DT
    dt_checkpoint = torch.load(dt_ckpt_path, map_location="cpu", weights_only=False)
    dt_model = DecisionTransformer(
        state_dim=dt_checkpoint["state_dim"],
        action_dim=1,
        context_length=dt_checkpoint["context_length"],
        max_episode_len=dt_checkpoint.get("max_episode_len", dt_checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
    )
    dt_model.load_state_dict(dt_checkpoint["state_dict"])
    
    dt_agent_03 = DTAgent(dt_model, args.target_rtg_low, dt_checkpoint["state_mean"], dt_checkpoint["state_std"], dt_checkpoint["rtg_scale"])
    dt_agent_08 = DTAgent(dt_model, args.target_rtg_high, dt_checkpoint["state_mean"], dt_checkpoint["state_std"], dt_checkpoint["rtg_scale"])
    
    # Load BC
    bc_checkpoint = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
    bc_model = BehaviorCloning(
        state_dim=bc_checkpoint["state_dim"],
        action_dim=1,
        context_length=bc_checkpoint["context_length"],
        max_episode_len=bc_checkpoint.get("max_episode_len", bc_checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
    )
    bc_model.load_state_dict(bc_checkpoint["state_dict"])
    bc_agent = BCAgent(bc_model, bc_checkpoint["state_mean"], bc_checkpoint["state_std"])
    
    # Run evaluations
    print("Running BC Agent...")
    bc_pos, bc_prices = run_agent_history(bc_agent, env)
    print(f"Running DT (RTG={args.target_rtg_low:.2f}) Agent...")
    dt03_pos, dt03_prices = run_agent_history(dt_agent_03, env)
    print(f"Running DT (RTG={args.target_rtg_high:.2f}) Agent...")
    dt08_pos, dt08_prices = run_agent_history(dt_agent_08, env)
    
    # Detect buy/sell ticks
    bc_buys, bc_sells = detect_signals(bc_pos)
    dt03_buys, dt03_sells = detect_signals(dt03_pos)
    dt08_buys, dt08_sells = detect_signals(dt08_pos)
    
    dates = df.index[test_bound[0] : test_bound[0] + len(bc_prices)]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    configs = [
        ("Behavior Cloning (BC - No RTG)", bc_prices, bc_pos, bc_buys, bc_sells, axes[0], "#ff7f0e"),
        (f"Decision Transformer (DT - RTG = {args.target_rtg_low:.2f})", dt03_prices, dt03_pos, dt03_buys, dt03_sells, axes[1], "#aec7e8"),
        (f"Decision Transformer (DT - RTG = {args.target_rtg_high:.2f} - Sweet Spot)", dt08_prices, dt08_pos, dt08_buys, dt08_sells, axes[2], "#1f77b4")
    ]
    
    for title, prices, positions, buys, sells, ax, model_color in configs:
        # Plot price on primary y-axis
        ax.plot(dates, prices, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")
        
        # Plot Buy / Sell scatter points
        if buys:
            ax.scatter(dates[buys], prices[buys], color="green", marker="^", s=70, label="Buy / Cover", zorder=5)
        if sells:
            ax.scatter(dates[sells], prices[sells], color="red", marker="v", s=70, label="Sell / Short", zorder=5)
            
        ax.set_ylabel("NASDAQ Close Price", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        
        # Twin axis for position ratio
        ax2 = ax.twinx()
        ax2.plot(dates, positions, color=model_color, linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
        ax2.fill_between(dates, 0, positions, color=model_color, alpha=0.1)
        ax2.set_ylabel("Target Position Ratio", color=model_color, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=model_color)
        if args.long_only:
            ax2.set_ylim(-0.05, 1.05)
        else:
            ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        
    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    
    plot_path = output_dir / "dt_bc_trading_signals.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved signals plot to {plot_path}")

if __name__ == "__main__":
    main()
