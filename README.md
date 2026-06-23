# RL Stock

Reinforcement-learning trading agents for the **NASDAQ-100 index** (`.NDX`).
Originally adapted from `shanghai_index_training.ipynb`; the default instrument
is now NASDAQ-100.

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
