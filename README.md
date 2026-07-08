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

## Dashboard

An interactive Streamlit app compares the algorithms:

```bash
streamlit run streamlit_app.py
```

It is a two-page app (switch in the sidebar):

**算法收益对比 / Compare** has two tabs, keyed off the run selector:

- **保存的评估指标 / Saved metrics**: reads each run's `summary.csv` (the exact
  evaluated numbers) and renders return-rate, Sharpe, Calmar, and max-drawdown
  bar charts plus a ranked table with the best value in each column highlighted.
  It also shows the saved `strategy_returns.png`.
- **收益曲线 / Equity (live)**: reloads the saved models and replays a
  deterministic episode to draw an interactive cumulative-return and drawdown
  overlay. The trend-feature and turbulence flags are auto-detected from each
  model's observation width; residual-prior and regime-scaling can be matched in
  the sidebar "高级" expander.

**买卖点分析 / Trades** plots the raw NASDAQ-100 price and overlays buy/sell
markers, position ribbons, holding-time stats, and a per-combo cumulative-return
comparison for any mix of experiment-times-algorithm combinations, so several
parameter combinations can be compared on one chart. Position env is continuous,
so a *buy* is the first
day exposure rises off zero, a *sell* is the first day it returns to zero, and
holding time is the number of days each position is carried.

Note: the equity curves are recomputed on the *current* `nasdaq100.xlsx`, whose
test window may differ from the one a run was trained on, so those final returns
can differ from the saved-metrics view. The saved-metrics view is the source of
truth for each run's reported numbers.

The dashboard code lives in `src/rl_stock/dashboard/` as small modules: run
discovery (`runs`), model replay (`replay`), Plotly figures (`figures`), the
shared replay cache (`session`), and one module per page (`page_compare`,
`page_trades`). `streamlit_app.py` is just the multipage router.
