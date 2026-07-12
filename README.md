# RL in Stock Market: An Improvement

This project improves the previous online A2C/PPO stock-trading experiment by
moving to continuous position control and offline sequence modeling on the
**NASDAQ-100 index** (`.NDX`). The main model is a Decision Transformer (DT)
conditioned on Return-to-Go (RTG), with an architecture-matched Behavior
Cloning (BC) baseline.

The semester objectives are:

- replace discrete buy/sell decisions with continuous exposure;
- add trend-aware state features and multi-day temporal context;
- train from a fixed offline replay dataset without online exploration;
- compare DT against BC, PPO, SAC, random, heuristic, and fixed-exposure baselines;
- test whether RTG conditioning changes policy behavior.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data

The default dataset is `nasdaq100.xlsx`: NASDAQ-100 daily OHLC bars
(2014-present) in the project's schema (`trade_date` as `YYYYMMDD` plus lowercase
`open/close/high/low/vol`). It was generated with `akshare`:

```python
import akshare as ak
ak.index_us_stock_sina(symbol=".NDX")  # then mapped to the project schema
```

Point `--data-path` at any Excel file with the same columns to train on a
different instrument (e.g. the original Shanghai Composite `output.xlsx`).

## Run

```bash
python train.py
```

Defaults to `--data-path nasdaq100.xlsx` and `--env-type position` (the
continuous risk-managed environment). The discrete `t1` env enforces a
China-specific T+1 hold rule that does not apply to NASDAQ, so `position` is the
right choice for US markets.

Common options:

```bash
python train.py \
  --data-path nasdaq100.xlsx \
  --env-type position \
  --algorithms SAC,PPO \
  --test-start-date 2024-01-01 \
  --window-size 30 \
  --total-timesteps 100000 \
  --num-eval-episodes 20
```

Outputs are written to `outputs/` by default:

- `summary.csv`
- `strategy_returns.png`
- `trades_<strategy>.png`
- trained model files when `--no-save-models` is not set

## Reproduce the offline DT experiment

### 1. Train continuous online baselines

Long-only:

```bash
python train.py \
  --env-type position \
  --algorithms SAC,PPO \
  --output-dir outputs/long_only \
  --use-trend-features \
  --stop-loss-pct 0.05 \
  --max-drawdown-pct 0.15 \
  --long-only
```

Symmetric long/short:

```bash
python train.py \
  --env-type position \
  --algorithms SAC,PPO \
  --output-dir outputs/symmetric \
  --use-trend-features \
  --stop-loss-pct 0.05 \
  --max-drawdown-pct 0.15 \
  --no-long-only
```

### 2. Build the offline replay datasets

```bash
python scripts/build_mvp_dataset.py \
  --sac-model outputs/long_only/sac.zip \
  --ppo-model outputs/long_only/ppo.zip \
  --output outputs/long_only_v2_mvp/dataset_v2.pkl \
  --qa-output outputs/long_only_v2_mvp/dataset_qa.json \
  --long-only

python scripts/build_mvp_dataset.py \
  --sac-model outputs/symmetric/sac.zip \
  --ppo-model outputs/symmetric/ppo.zip \
  --output outputs/symmetric_v2_mvp/dataset_v2.pkl \
  --qa-output outputs/symmetric_v2_mvp/dataset_qa.json \
  --no-long-only
```

Each Dataset V2 contains stochastic PPO and SAC trajectories, random-policy
trajectories, and fixed-exposure coverage anchors. Exact duplicate action paths
are removed and recorded in the QA JSON.

### 3. Train DT and BC

```bash
python scripts/train_dt.py \
  --dataset outputs/long_only_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/long_only_v2_mvp/dt \
  --long-only

python scripts/train_bc.py \
  --dataset outputs/long_only_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/long_only_v2_mvp/bc \
  --long-only
```

Use `--no-long-only` and the symmetric dataset for the symmetric experiment.

### 4. Run the controlled optimization comparison

```bash
python scripts/run_optimization_ablation.py
```

The ablation compares DT and BC using the same replay data, final-epoch model
selection, common RTG targets, and multiple optimization schedules.

## Tests

```bash
pytest -q
```

The test suite covers continuous environments, risk controls, trend features,
trajectory collection, RTG construction, DT/BC models, and padding-mask logic.

## Repository structure

```text
src/rl_stock/       Core environments, features, data, DT, BC, and evaluation
scripts/            Reproducible training, ablation, and visualization commands
tests/              Unit and integration tests
docs/               Experiment plans and result summaries
nasdaq100.xlsx      NASDAQ-100 OHLC data used by the submitted experiments
```

Generated models, replay datasets, and plots are written to `outputs/` and are
not tracked by Git.
