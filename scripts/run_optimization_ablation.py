#!/usr/bin/env python3
"""Resumable overnight DT/BC optimization-intensity ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.bc_inference import evaluate_bc_model
from rl_stock.bc_model import BehaviorCloning
from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_dataset import TrajectoryDataset
from rl_stock.dt_inference import evaluate_dt_model
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio, compute_turnover


OUT = ROOT / "outputs" / "optimization_ablation"
DATASETS = {
    "long_only": ROOT / "outputs" / "long_only_v2_mvp" / "dataset_v2.pkl",
    "symmetric": ROOT / "outputs" / "symmetric_v2_mvp" / "dataset_v2.pkl",
}
SCHEDULES = (("e20_s20", 20, 20), ("e40_s10", 40, 10), ("e60_s5", 60, 5))


@dataclass(frozen=True)
class RunConfig:
    space: str
    model: str
    run_id: str
    epochs: int
    stride: int
    checkpoint_every: int = 5
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    context_length: int = 20

    @property
    def directory(self) -> Path:
        return OUT / self.space / self.model / self.run_id


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def log(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def configs() -> list[RunConfig]:
    ordered: list[RunConfig] = []
    for model in ("dt", "bc"):
        for space in ("long_only", "symmetric"):
            for run_id, epochs, stride in SCHEDULES:
                ordered.append(RunConfig(space, model, run_id, epochs, stride))
    return ordered


def load_raw(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dataset_and_metadata(cfg: RunConfig):
    raw = load_raw(DATASETS[cfg.space])
    rtg_abs = [abs(step["rtg"]) for traj in raw["trajectories"] for step in traj]
    rtg_scale = max(rtg_abs) if rtg_abs and max(rtg_abs) >= 1e-6 else 1.0
    ds = TrajectoryDataset(
        DATASETS[cfg.space], context_length=cfg.context_length,
        rtg_scale=rtg_scale, step_stride=cfg.stride,
    )
    max_t = max(step["timestep"] for traj in raw["trajectories"] for step in traj)
    return ds, raw, rtg_scale, max_t + 128


def make_model(cfg: RunConfig, state_dim: int, max_len: int):
    activation = "sigmoid" if cfg.space == "long_only" else "tanh"
    cls = DecisionTransformer if cfg.model == "dt" else BehaviorCloning
    return cls(
        state_dim=state_dim, action_dim=1, context_length=cfg.context_length,
        max_episode_len=max_len, action_activation=activation,
    )


def scheduler_for(optimizer, total_steps: int, warmup: int):
    def factor(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("epoch_*.pt"))
    return paths[-1] if paths else None


def checkpoint_payload(cfg, model, optimizer, scheduler, epoch, step, history, ds, rtg_scale, max_len):
    return {
        "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "epoch": epoch,
        "global_step": step, "history": history, "config": asdict(cfg),
        "state_mean": ds.state_mean, "state_std": ds.state_std,
        "state_dim": ds.metadata["state_dim"], "context_length": cfg.context_length,
        "max_episode_len": max_len, "rtg_scale": rtg_scale,
        "action_activation": "sigmoid" if cfg.space == "long_only" else "tanh",
        "long_only": cfg.space == "long_only", "selection_rule": "final_epoch",
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
    }


def train_run(cfg: RunConfig, stop_after_epoch: int | None = None) -> Path:
    cfg.directory.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs" / f"{cfg.space}_{cfg.model}_{cfg.run_id}.log"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    ds, _, rtg_scale, max_len = dataset_and_metadata(cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = make_model(cfg, ds.metadata["state_dim"], max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(loader) * cfg.epochs
    scheduler = scheduler_for(optimizer, total_steps, cfg.warmup_steps)
    history = {"epoch": [], "train_loss": [], "learning_rate": [], "global_step": []}
    start_epoch, global_step = 1, 0
    resume = latest_checkpoint(cfg.directory)
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["epoch"] >= cfg.epochs:
            log(f"[{cfg.space}][{cfg.model.upper()}][{cfg.run_id}] already complete", log_path)
            return resume
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"]); history = saved["history"]
        start_epoch, global_step = saved["epoch"] + 1, saved["global_step"]
        torch.set_rng_state(saved["torch_rng_state"]); np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        log(f"Resuming {cfg.space}/{cfg.model}/{cfg.run_id} from epoch {saved['epoch']}", log_path)
    loss_fn = torch.nn.MSELoss(reduction="none")
    started = time.time(); final_epoch = min(stop_after_epoch or cfg.epochs, cfg.epochs)
    for epoch in range(start_epoch, final_epoch + 1):
        model.train(); losses = []
        for states, actions, rtgs, timesteps, mask in loader:
            if cfg.model == "dt":
                predictions = model(states, actions, rtgs, timesteps, attention_mask=mask)
            else:
                predictions = model(states, actions, timesteps, attention_mask=mask)
            raw_loss = loss_fn(predictions, actions)
            loss = (raw_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step(); scheduler.step(); global_step += 1; losses.append(loss.item())
        mean_loss = float(np.mean(losses)); lr = float(scheduler.get_last_lr()[0])
        history["epoch"].append(epoch); history["train_loss"].append(mean_loss)
        history["learning_rate"].append(lr); history["global_step"].append(global_step)
        elapsed = time.time() - started; done = epoch - start_epoch + 1
        eta = elapsed / max(done, 1) * max(final_epoch - epoch, 0)
        log(f"[{cfg.space}][{cfg.model.upper()}][{cfg.run_id}] Epoch {epoch}/{cfg.epochs} step={global_step} loss={mean_loss:.6f} lr={lr:.2e} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", log_path)
        atomic_json(cfg.directory / "history.json", history)
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs or epoch == final_epoch:
            payload = checkpoint_payload(cfg, model, optimizer, scheduler, epoch, global_step, history, ds, rtg_scale, max_len)
            path = cfg.directory / f"epoch_{epoch:03d}.pt"; torch.save(payload, path)
            log(f"Saved checkpoint: {path}", log_path)
    return latest_checkpoint(cfg.directory)  # type: ignore[return-value]


def env_for(space: str):
    df = load_market_data(ROOT / "nasdaq100.xlsx")
    _, bound = split_frame_bounds(df, 30, "2024-01-01")
    risk = RiskConfig(flat_obs=True, stop_loss_pct=.05, max_drawdown_pct=.15,
                      use_trend_features=True, long_only=space == "long_only")
    return PositionMgmtEnv(df, 30, bound, risk)


def metrics(info: dict[str, Any]) -> dict[str, Any]:
    returns = np.asarray(info["returns"]); positions = np.asarray(info["position_history"])
    return {
        "profit": float(info["total_profit"] - 1), "sharpe": compute_sharpe_ratio(returns),
        "max_drawdown": compute_max_drawdown(np.asarray(info["equity_curve"])),
        "avg_raw_action": float(np.mean(info["raw_action_history"])),
        "avg_executed_position": float(np.mean(info["executed_action_history"])),
        "turnover": compute_turnover(positions), "stop_loss_events": int(info["stop_loss_events"]),
    }


def evaluate_checkpoint(path: Path, target: float | None = None) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False); cfg = RunConfig(**ck["config"])
    model = make_model(cfg, ck["state_dim"], ck["max_episode_len"]); model.load_state_dict(ck["state_dict"])
    env = env_for(cfg.space)
    if cfg.model == "dt":
        _, _, infos = evaluate_dt_model(model, env, float(target), ck["state_mean"], ck["state_std"], ck["rtg_scale"], num_episodes=1, seed=42)
    else:
        _, _, infos = evaluate_bc_model(model, env, ck["state_mean"], ck["state_std"], num_episodes=1, seed=42)
    return {"action_space": cfg.space, "model": cfg.model, "run_id": cfg.run_id,
            "epoch": ck["epoch"], "global_step": ck["global_step"],
            "learning_rate": ck["history"]["learning_rate"][-1],
            "train_loss": ck["history"]["train_loss"][-1], **metrics(infos[0])}


def starting_rtgs(space: str) -> np.ndarray:
    raw = load_raw(DATASETS[space])
    return np.asarray([traj[0]["rtg"] for traj in raw["trajectories"]], dtype=float)


def evaluate_all() -> list[dict[str, Any]]:
    all_rows = []
    for cfg in configs():
        target = float(np.percentile(starting_rtgs(cfg.space), 90)) if cfg.model == "dt" else None
        rows = []
        for path in sorted(cfg.directory.glob("epoch_*.pt")):
            row = evaluate_checkpoint(path, target); row["target_rtg_percentile"] = 90 if target is not None else ""
            row["target_rtg_value"] = target if target is not None else ""; rows.append(row); all_rows.append(row)
            log(f"Evaluated {cfg.space}/{cfg.model}/{cfg.run_id}/epoch {row['epoch']}: profit={row['profit']:+.4f}")
        write_csv(OUT / cfg.space / f"{cfg.model}_checkpoint_evaluations.csv", [r for r in all_rows if r["action_space"] == cfg.space and r["model"] == cfg.model])
    write_csv(OUT / "comparison.csv", all_rows)
    return all_rows


def rtg_sweeps() -> list[dict[str, Any]]:
    rows = []
    for space in DATASETS:
        cfg = RunConfig(space, "dt", "e60_s5", 60, 5); path = cfg.directory / "epoch_060.pt"
        if not path.exists():
            cfg = RunConfig(space, "dt", "e40_s10", 40, 10); path = cfg.directory / "epoch_040.pt"
        starts = starting_rtgs(space)
        for p in range(10, 100, 10):
            target = float(np.percentile(starts, p)); row = evaluate_checkpoint(path, target)
            row["rtg_percentile"] = p; row["rtg_value"] = target; rows.append(row)
            log(f"RTG sweep {space} P{p}: profit={row['profit']:+.4f} exec={row['avg_executed_position']:+.3f}")
        write_csv(OUT / space / "rtg_sweep.csv", [r for r in rows if r["action_space"] == space])
    return rows


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def dataset_analysis(space: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = load_raw(DATASETS[space]); trajectories = raw["trajectories"]
    summaries = []; step_rtg=[]; raw_actions=[]; executed=[]
    for i, traj in enumerate(trajectories):
        r=np.asarray([s["net_return"] for s in traj]); a=np.asarray([s["raw_action"] for s in traj]); e=np.asarray([s["executed_action"] for s in traj])
        summaries.append({"trajectory_id": traj[0].get("trajectory_id", i), "policy_name": traj[0].get("policy_name", "unknown"),
            "starting_rtg": float(traj[0]["rtg"]), "avg_raw_action": float(a.mean()), "avg_executed_position": float(e.mean()),
            "executed_turnover": float(np.mean(np.abs(np.diff(e)))), "sum_net_returns": float(r.sum()),
            "compound_return": float(np.prod(1+r)-1), "episode_sharpe": float(traj[0].get("episode_sharpe", 0)),
            "stop_events": int(sum(bool(s.get("stop_triggered")) for s in traj)), "dd_lock_proportion": float(np.mean([bool(s.get("dd_locked")) for s in traj]))})
        step_rtg.extend(s["rtg"] for s in traj); raw_actions.extend(a); executed.extend(e)
    stats=[]
    arrays={"starting_rtg":np.asarray([x["starting_rtg"] for x in summaries]),"step_rtg":np.asarray(step_rtg),"raw_action":np.asarray(raw_actions),"executed_position":np.asarray(executed)}
    for name,x in arrays.items():
        row={"variable":name,"count":len(x),"mean":x.mean(),"std":x.std(),"min":x.min(),"max":x.max()}
        row.update({f"p{p}":np.percentile(x,p) for p in (5,10,20,30,40,50,60,70,80,90,95)}); stats.append(row)
    write_csv(OUT/space/"dataset_statistics"/"trajectory_summary.csv",summaries); write_csv(OUT/space/"dataset_statistics"/"descriptive_statistics.csv",stats)
    sdf=pd.DataFrame(summaries); correlations=[]
    for y in ("avg_raw_action","avg_executed_position","compound_return","executed_turnover","sum_net_returns"):
        x=sdf["starting_rtg"].to_numpy(); z=sdf[y].to_numpy(); correlations.append({"x":"starting_rtg","y":y,"pearson":float(np.corrcoef(x,z)[0,1]),"spearman":rank_corr(x,z)})
    write_csv(OUT/space/"dataset_statistics"/"correlations.csv",correlations)
    fig,axes=plt.subplots(2,2,figsize=(12,8));
    for ax,(name,x) in zip(axes.flat,arrays.items()): ax.hist(x,bins=40,alpha=.8); ax.set_title(name); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/space/"dataset_statistics"/"dataset_histograms.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9)); colors=pd.factorize(sdf["policy_name"])[0]
    for ax,y in zip(axes.flat,("avg_raw_action","avg_executed_position","compound_return","executed_turnover")):
        ax.scatter(sdf["starting_rtg"],sdf[y],c=colors,cmap="tab10"); ax.set_xlabel("starting RTG"); ax.set_ylabel(y); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/space/"dataset_statistics"/"correlation_scatter.png",dpi=150); plt.close(fig)
    return summaries, correlations


def plots(checkpoints: list[dict[str, Any]], sweeps: list[dict[str, Any]]) -> None:
    df=pd.DataFrame(checkpoints)
    for space in DATASETS:
        fig,axes=plt.subplots(2,2,figsize=(13,9)); sub=df[df.action_space==space]
        for model in ("dt","bc"):
            for run_id,_,_ in SCHEDULES:
                x=sub[(sub.model==model)&(sub.run_id==run_id)]
                if x.empty: continue
                label=f"{model.upper()} {run_id}"; axes[0,0].plot(x.epoch,x.train_loss,marker="o",label=label); axes[0,1].plot(x.epoch,x.profit,marker="o",label=label); axes[1,0].plot(x.epoch,x.sharpe,marker="o",label=label); axes[1,1].plot(x.epoch,x.avg_executed_position,marker="o",label=label)
        for ax,title in zip(axes.flat,("Training loss","Test profit (diagnostic)","Test Sharpe (diagnostic)","Average executed position")): ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,ncol=2); fig.suptitle(f"Optimization curves — {space}"); fig.tight_layout(); fig.savefig(OUT/space/"optimization_curves.png",dpi=160); plt.close(fig)
        s=pd.DataFrame([r for r in sweeps if r["action_space"]==space]); fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,y in zip(axes.flat,("profit","sharpe","avg_raw_action","avg_executed_position")): ax.plot(s.rtg_percentile,s[y],marker="o"); ax.set_xlabel("RTG percentile"); ax.set_ylabel(y); ax.grid(alpha=.2)
        fig.suptitle(f"Dense RTG sweep — {space}"); fig.tight_layout(); fig.savefig(OUT/space/"rtg_sweep.png",dpi=160); plt.close(fig)


def report(checkpoints, sweeps, correlations) -> None:
    df=pd.DataFrame(checkpoints); lines=["# Optimization Ablation Results","","## Scope","","Retrospective diagnostic; intermediate test metrics were not used for checkpoint selection.","","## Final checkpoint comparison","", "| Space | Model | Run | Loss | Profit | Sharpe | Avg executed |", "|---|---|---|---:|---:|---:|---:|"]
    finals=df.sort_values("epoch").groupby(["action_space","model","run_id"]).tail(1)
    for _,r in finals.iterrows(): lines.append(f"| {r.action_space} | {r.model.upper()} | {r.run_id} | {r.train_loss:.5f} | {r.profit*100:+.2f}% | {r.sharpe:.3f} | {r.avg_executed_position:.3f} |")
    lines += ["","## Correlations",""]
    for space,rows in correlations.items():
        lines += [f"### {space}","","| Relationship | Pearson | Spearman |","|---|---:|---:|"]
        for r in rows: lines.append(f"| RTG vs {r['y']} | {r['pearson']:.3f} | {r['spearman']:.3f} |")
        lines.append("")
    lines += ["## Interpretation placeholder","","Final interpretation must compare decreasing training loss with checkpoint profit/exposure curves; no checkpoint is selected by test performance.","","## Reproduction","","```bash","PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_optimization_ablation.py","```",""]
    (ROOT/"docs"/"optimization_ablation_results.md").write_text("\n".join(lines),encoding="utf-8")


def smoke() -> None:
    smoke_root = OUT / "_smoke"; old_out = globals()["OUT"]
    globals()["OUT"] = smoke_root
    try:
        for model in ("dt","bc"):
            cfg=RunConfig("long_only",model,"resume_smoke",2,500,checkpoint_every=1)
            train_run(cfg,stop_after_epoch=1); train_run(cfg)
            ck=torch.load(cfg.directory/"epoch_002.pt",map_location="cpu",weights_only=False)
            required={"optimizer_state","scheduler_state","global_step","history","torch_rng_state"}
            assert required.issubset(ck) and ck["epoch"]==2
        log("Smoke checkpoint/resume tests passed")
    finally: globals()["OUT"] = old_out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--analysis-only",action="store_true"); parser.add_argument("--stop-after-runs",type=int,default=None); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.smoke: smoke(); return
    if not args.analysis_only:
        manifest_path = OUT / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": {}}
        completed=0
        for cfg in configs():
            key=f"{cfg.space}/{cfg.model}/{cfg.run_id}"; manifest["runs"][key]={**asdict(cfg),"status":"training"}; atomic_json(manifest_path,manifest)
            try: path=train_run(cfg); manifest["runs"][key].update(status="trained",latest_checkpoint=str(path),latest_epoch=cfg.epochs,error=None)
            except Exception as exc: manifest["runs"][key].update(status="failed",error=repr(exc)); atomic_json(manifest_path,manifest); raise
            atomic_json(manifest_path,manifest); completed+=1
            if args.stop_after_runs and completed>=args.stop_after_runs: return
    checkpoints=evaluate_all(); sweeps=rtg_sweeps(); correlations={}
    for space in DATASETS: _,correlations[space]=dataset_analysis(space)
    plots(checkpoints,sweeps); report(checkpoints,sweeps,correlations)
    atomic_json(OUT/"completion.json",{"status":"complete","completed_at":time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("Optimization ablation complete")


if __name__ == "__main__": main()
