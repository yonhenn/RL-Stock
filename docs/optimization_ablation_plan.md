# DT/BC Optimization-Intensity Ablation — Overnight Implementation Plan

## 1. Objective

本實驗的唯一研究問題是：

> 目前 Decision Transformer（DT）與 Behavior Cloning（BC）的 underperformance，主要來自 optimization 不足，還是模型已經因 Dataset V2 與 environment constraints 而飽和？

本實驗只改變 optimization intensity，不修改資料生成與交易環境。

### Permitted variables

- Training epochs
- Dataset step stride
- 因 total optimizer steps 改變而連帶變化的 cosine learning-rate schedule
- Intermediate checkpoint frequency

### Locked variables

- Dataset V2 contents
- Environment、reward、ATR scaling、stop-loss與drawdown lock
- Model architecture、peak learning rate、optimizer與weight decay
- Context length、batch size與training seed
- Action semantics與SAC/PPO checkpoints
- Train/test date split與dynamic RTG inference

本實驗定位為 **retrospective optimization diagnostic**。2024–2026 test period 已被過往實驗使用，因此 intermediate test curves 只能診斷 optimization dynamics，不得宣稱為 untouched out-of-sample evidence。

---

## 2. Action spaces and datasets

兩個 action spaces 分開執行完整實驗。

### Long-only `[0,1]`

```text
Dataset: outputs/long_only_v2_mvp/dataset_v2.pkl
Model output: sigmoid
Environment: long_only=True
Primary DT diagnostic target: Dataset starting-RTG P90
```

### Symmetric `[-1,1]`

```text
Dataset: outputs/symmetric_v2_mvp/dataset_v2.pkl
Model output: tanh
Environment: long_only=False
Primary DT diagnostic target: Dataset starting-RTG P90
```

不得合併兩個 datasets，也不得跨 action space 使用 normalization statistics。

---

## 3. Optimization matrix

每個 action space、每個 model 均執行三組 independent training runs：

| Run ID | Epochs | Step stride | Relative load |
|---|---:|---:|---:|
| `e20_s20` | 20 | 20 | 1× |
| `e40_s10` | 40 | 10 | 約4× |
| `e60_s5` | 60 | 5 | 約12× |

總計：

```text
2 action spaces × 2 models × 3 schedules = 12 training runs
```

此設計稱為 **optimization-intensity ablation**。由於 epochs 與 stride 同時改變，結果不能單獨歸因於 epochs 或 stride。

### Locked training settings

```text
Seed: 42
Context length: 20
Batch size: 64
Peak learning rate: 1e-4
Weight decay: 1e-4
Warmup steps: 500
Checkpoint selection: final epoch
Architecture: unchanged
```

Cosine scheduler 依每組 run 的 total optimizer steps 計算。因此 intermediate table 必須保存 global step 與 learning rate。

---

## 4. Execution priority

若 overnight process 中斷，依以下順序確保最重要結果優先完成：

1. Long-only DT：`e20_s20 → e40_s10 → e60_s5`
2. Symmetric DT：`e20_s20 → e40_s10 → e60_s5`
3. Long-only BC：`e20_s20 → e40_s10 → e60_s5`
4. Symmetric BC：`e20_s20 → e40_s10 → e60_s5`
5. Intermediate-checkpoint batch evaluation
6. Dense RTG sweeps
7. Dataset statistics、correlations、plots與report

若時間不足，DT 優先於 BC；不得為趕時間改變 architecture 或 dataset。

---

## 5. Git and output isolation

### Branch

從目前 branch 建立：

```text
codex/optimization-ablation
```

目前既有 unstaged changes 視為使用者工作，不得執行 destructive Git commands。只 stage 與 optimization ablation 直接相關的檔案。

### Output directory

```text
outputs/optimization_ablation/
├── run_manifest.json
├── logs/
├── long_only/
│   ├── dt/e20_s20, e40_s10, e60_s5
│   ├── bc/{e20_s20,e40_s10,e60_s5}/
│   ├── bc/{e20_s20,e40_s10,e60_s5}/
│   ├── checkpoint_evaluations.csv
│   ├── rtg_sweep.csv
│   └── dataset_statistics/
└── cross_space_comparison.csv
```

不得覆蓋既有 V2 或 matched-audit outputs。

---

## 6. Resumable orchestrator

新增：

```text
scripts/run_optimization_ablation.py
```

功能：

1. 建立 12 組 configs。
2. 依優先順序 training。
3. 每 5 epochs 保存 checkpoint。
4. 保存 history、config、logs。
5. 以 atomic replace 更新 manifest/CSV。
6. 跳過 completed runs。
7. 從最新 checkpoint 恢復 interrupted run。
8. Training 完成後才批次 evaluation。
9. 產生 tables、plots、report inputs。

### Manifest states

```text
pending → training → trained → evaluating → evaluated → complete
failed
```

Manifest entry：

```json
{
  "action_space": "long_only",
  "model": "dt",
  "run_id": "e60_s5",
  "epochs": 60,
  "step_stride": 5,
  "status": "training",
  "latest_epoch": 35,
  "latest_checkpoint": "epoch_035.pt",
  "error": null
}
```

---

## 7. Checkpoint implementation

DT/BC training API 加入：

```python
checkpoint_dir: Path | None
checkpoint_every: int = 5
resume_checkpoint: Path | None
run_metadata: dict | None
```

每個 checkpoint 保存：

- Model、optimizer、scheduler states
- Epoch、global step、LR、latest loss、loss history
- State normalization、DT RTG scale
- Action activation、long_only、selection rule
- Complete run config

檔名：

```text
epoch_005.pt ... epoch_060.pt
```

Training 階段不得建立或呼叫 test environment。

---

## 8. Terminal and logs

正式執行使用 unbuffered TTY。每 epoch 顯示：

```text
[long_only][DT][e60_s5] Epoch 17/60
Global step / loss / LR / elapsed / ETA
```

每次 checkpoint 顯示完整路徑。Logs 保存至：

```text
outputs/optimization_ablation/logs/<space>_<model>_<run>.log
```

必要時用特定 `caffeinate` wrapper 防止 macOS sleep。

---

## 9. Preflight smoke tests

1. Full pytest。
2. 1-epoch DT smoke run。
3. 1-epoch BC smoke run。
4. 驗證 checkpoint 含 optimizer/scheduler/global step。
5. 從 smoke checkpoint resume 至 epoch 2。
6. 確認 history/global step 連續。
7. 確認 training log 無 test metrics。

Smoke artifacts 不混入正式 results。

---

## 10. Intermediate checkpoint evaluation

Training 全部完成後才批次載入 checkpoints。

### DT target

每個 action space 預先使用其 Dataset starting-RTG P90。不得因 test 表現換 target。

### Metrics

```text
action_space, model, run_id, epoch, global_step, learning_rate,
train_loss, target_rtg_percentile, target_rtg_value,
profit, sharpe, max_drawdown,
avg_raw_action, avg_executed_position, turnover, stop_loss_events
```

輸出：

```text
outputs/optimization_ablation/<space>/checkpoint_evaluations.csv
```

Intermediate test metrics 只診斷 plateau、exposure與overfitting，不選 checkpoint、不回頭調參。

---

## 11. Dense RTG sweep

預先指定：

```text
e60_s5 / epoch 60 final DT checkpoint
```

只有技術性失敗才降級為 `e40_s10` final，且記錄原因。

Targets 由 trajectory starting-RTG 計算：

```text
P10, P20, P30, P40, P50, P60, P70, P80, P90
```

保存 percentile、absolute RTG、profit、Sharpe、Max DD、raw/executed position、turnover與stops。

---

## 12. Dataset statistics

只讀 Dataset V2，不修改內容。

### RTG

分開分析：

1. Trajectory starting RTG
2. All step-level RTGs

各自輸出 histogram、count、mean、std、min/max 與 P5/P10/.../P90/P95。Step RTGs 高度 autocorrelated，不視為 independent samples。

### Actions and positions

- Overall/per-policy raw-action distributions
- Overall/per-policy executed-position distributions
- Long/flat/short step proportions

### Per-trajectory table

```text
trajectory_id, policy_name, starting_rtg,
avg_raw_action, avg_executed_position, executed_turnover,
sum_net_returns, compound_return, episode_sharpe,
stop_events, dd_lock_proportion
```

Definitions：

```python
executed_turnover = mean(abs(diff(executed_position)))
compound_return = product(1 + net_return) - 1
```

---

## 13. Correlations

兩個 action spaces 分別輸出 Pearson 與 Spearman：

- Starting RTG vs average raw action
- Starting RTG vs average executed position
- Starting RTG vs compound return
- Starting RTG vs executed turnover
- Starting RTG vs sum net returns（integrity check）

Scatter points 依 policy_name 著色。Mixed-policy data 可能群聚/非線性，因此不得只報 Pearson。

---

## 14. Required plots

### Optimization

- Epoch/global step vs training loss
- Epoch vs profit
- Epoch vs Sharpe
- Epoch vs raw/executed position
- Training loss vs test profit

### RTG sweep

- RTG percentile vs profit/Sharpe
- RTG percentile vs raw/executed position
- RTG percentile vs turnover

### Dataset

- Starting-RTG histogram
- Step-RTG histogram
- Raw-action histogram
- Executed-position histogram
- Correlation scatter panel

圖中標示 action space、dataset、seed、run schedule。

---

## 15. Decision rules

### Optimization matters

較高 intensity 同時帶來 lower loss、持續提升的 profit/Sharpe、合理 exposure，且 epoch 60 尚未 plateau。

### Dataset/environment dominates

Loss 持續下降，但 profit/Sharpe 快速 plateau或下降；exposure ceiling 不變；RTG control 平滑但最高收益仍低。

### Overfitting

Intermediate performance 先改善後惡化，training loss 仍下降，turnover/action saturation 上升。

不設定任意 profit threshold；依完整 curves 與metrics判斷。

---

## 16. Deliverables

### Data

```text
outputs/optimization_ablation/comparison.csv
outputs/optimization_ablation/cross_space_comparison.csv
outputs/optimization_ablation/<space>/checkpoint_evaluations.csv
outputs/optimization_ablation/<space>/rtg_sweep.csv
outputs/optimization_ablation/<space>/dataset_statistics/trajectory_summary.csv
outputs/optimization_ablation/<space>/dataset_statistics/descriptive_statistics.csv
outputs/optimization_ablation/<space>/dataset_statistics/correlations.csv
```

### Plots

```text
training_curves.png
checkpoint_profit.png
checkpoint_sharpe.png
checkpoint_exposure.png
loss_vs_profit.png
rtg_sweep.png
dataset_histograms.png
correlation_scatter.png
```

### Report

```text
docs/optimization_ablation_results.md
```

Report 包含 exact configs、manifest、checkpoint tables、RTG sweep、statistics、correlations、結論、limitations與reproduction commands。

---

## 17. Runtime and resources

| Phase | Estimate |
|---|---:|
| Code/tests/smoke | 45–75 min |
| Long-only DT | 75–100 min |
| Symmetric DT | 75–100 min |
| Long-only BC | 45–65 min |
| Symmetric BC | 45–65 min |
| Evaluation/sweep | 30–60 min |
| Statistics/report | 30–60 min |
| **Total** | **約 5–7 hours** |

Disk：約 400–700 MB。Mac 應插電、保持 task 開啟並防止 sleep。

---

## 18. Model and permissions

### Codex

```text
GPT-5 Codex — High reasoning
```

LLM 只負責 implementation、monitoring與analysis；PyTorch model不因Codex選擇改變。

### Required

- Workspace read/write
- Execute `.venv/bin/python`、`.venv/bin/pytest`
- Write `outputs/optimization_ablation/`
- Create Git branch、stage/commit ablation files
- Optional specific `caffeinate` command

### Not required

- Network、package installation、sudo/admin
- Full-disk access、destructive Git、Git push

---

## 19. Planned commits

```text
docs: add optimization-ablation implementation plan
feat: add resumable checkpointed optimization ablation
test: cover checkpoint resume and statistics
docs: report optimization-ablation results
```

不 push、不建立 submission tag，除非另行要求。

---

## 20. Definition of done

- [ ] 12 runs完成，或failure已記錄且可resume。
- [ ] 每5 epochs checkpoints齊全。
- [ ] Training logs無test evaluation。
- [ ] Intermediate checkpoints已批次評估且未用於選模。
- [ ] Dense RTG sweep使用預先指定final `e60_s5`。
- [ ] 兩個action spaces的statistics與correlations齊全。
- [ ] Required CSV、plots、report齊全。
- [ ] Full pytest與`git diff --check`通過。
- [ ] Code與results分開commit。
- [ ] Unrelated unstaged changes未被覆蓋或提交。
