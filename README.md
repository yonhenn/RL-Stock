# RL Stock

This is a structured Python version of `shanghai_index_training.ipynb`.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python train.py --data-path output.xlsx
```

The original notebook used `/content/drive/MyDrive/RL/output.xlsx` in Colab.
Pass that path with `--data-path` if you still run from Colab, or place the
Excel file in this directory as `output.xlsx`.

Common options:

```bash
python train.py \
  --data-path output.xlsx \
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
