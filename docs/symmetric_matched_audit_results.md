# Symmetric `[-1,1]` Matched Audit Results

## Run definition

- Branch: `codex/symmetric-matched-audit`
- Experiment type: matched retrospective audit
- Train period: 2014–2023
- Test period: 2024-01-02 to 2026-06-22
- Dataset: 35 unique trajectories, 0 exact duplicates
- BC/DT: 20 epochs, context 20, stride 20, seed 42
- Output activation: `tanh`
- Checkpoint selection: final epoch
- DT inference: dynamic RTG at training starting-RTG P10/P50/P90
- Test metrics were not used during training

## Dataset QA

| Source | Unique trajectories |
|---|---:|
| Random | 10 |
| SAC stochastic | 10 |
| PPO stochastic | 10 |
| Fixed raw targets `-1/-.5/0/.5/1` | 5 |
| **Total** | **35** |

- Raw-action range: `[-1,1]`
- Mean raw action: +0.054
- Mean executed position: +0.044
- Executed steps: 13.8% short, 62.8% flat, 23.4% long
- Starting RTG P10: -0.1616
- Starting RTG P50: -0.1344
- Starting RTG P90: +0.3786

The high flat proportion shows that the existing ATR/stop/drawdown overlay strongly transforms symmetric raw actions. This audit intentionally keeps that overlay unchanged to avoid adding a second experimental variable.

## Symmetric test results

| Strategy | Profit | Sharpe | Max DD | Avg raw | Avg executed |
|---|---:|---:|---:|---:|---:|
| Always Short direct | -51.14% | -1.286 | 52.12% | -1.000 | -1.000 |
| Fixed -50% direct | -29.13% | -1.286 | 29.86% | -0.500 | -0.500 |
| Cash | 0.00% | 0.000 | 0.00% | 0.000 | 0.000 |
| Fixed +50% direct | +37.20% | 1.282 | 12.04% | 0.500 | 0.500 |
| Passive Buy & Hold | +83.25% | 1.282 | 22.93% | 1.000 | 1.000 |
| SAC V1 | -12.61% | -1.036 | 15.10% | 0.014 | 0.007 |
| PPO V1 | -9.07% | -0.666 | 15.12% | 0.246 | 0.072 |
| BC V2 | -6.83% | -0.972 | 9.49% | 0.056 | 0.034 |
| DT V2 P10 | -10.87% | -1.626 | 12.71% | -0.092 | -0.052 |
| DT V2 P50 | -10.26% | -1.545 | 12.10% | -0.077 | -0.043 |
| **DT V2 P90** | **+7.67%** | **0.592** | **6.43%** | **0.335** | **0.210** |

## Paired 100-Random distributions

Seeds 42–141 use the same uniform draws `u_t`:

```text
long-only action = u_t
symmetric action = 2u_t - 1
```

| Action space | Metric | Mean ± std | P5 | P50 | P95 |
|---|---|---:|---:|---:|---:|
| `[0,1]` | Profit | +4.91% ± 5.94% | -4.72% | +5.00% | +13.82% |
| `[0,1]` | Sharpe | 0.312 ± 0.365 | -0.277 | 0.324 | 0.832 |
| `[0,1]` | Max DD | 9.55% ± 2.42% | 6.27% | 9.29% | 13.59% |
| `[-1,1]` | Profit | -13.02% ± 3.14% | -15.87% | -14.00% | -7.57% |
| `[-1,1]` | Sharpe | -1.175 ± 0.453 | -1.874 | -1.172 | -0.463 |
| `[-1,1]` | Max DD | 15.30% ± 0.82% | 14.56% | 15.29% | 16.19% |

Long-only random actions carry positive beta: mean raw action 0.500 and mean executed position 0.309. Symmetric random actions have near-zero mean exposure but higher turnover, transaction costs and occasional stop events. Therefore the two action spaces require separate random reference distributions.

## Model empirical percentiles

Percentiles use Laplace smoothing and compare each model only with the 100 random paths from the same action space.

| Model | Space | Profit pct. | Sharpe pct. | Low-DD pct. |
|---|---|---:|---:|---:|
| SAC V1 | `[0,1]` | 97.1 | 98.0 | 95.1 |
| PPO V1 | `[0,1]` | 97.1 | 91.2 | 6.9 |
| BC V2 | `[0,1]` | 22.5 | 52.0 | 99.0 |
| DT V2 P75 | `[0,1]` | 72.5 | 68.6 | 41.2 |
| DT V2 P90 | `[0,1]` | 75.5 | 68.6 | 40.2 |
| SAC V1 | `[-1,1]` | 65.7 | 62.7 | 77.5 |
| PPO V1 | `[-1,1]` | 87.3 | 85.3 | 72.5 |
| BC V2 | `[-1,1]` | 94.1 | 66.7 | 98.0 |
| DT V2 P10 | `[-1,1]` | 82.4 | 17.6 | 97.1 |
| DT V2 P50 | `[-1,1]` | 84.3 | 23.5 | 97.1 |
| **DT V2 P90** | `[-1,1]` | **99.0** | **99.0** | **99.0** |

`99.0` means the model exceeded all 100 observed random paths after finite-sample smoothing. It is not a 99% confidence statement and does not measure future-market generalization.

## Matched DT comparison

| DT setting | Profit | Sharpe | Max DD | Avg executed | Profit random pct. |
|---|---:|---:|---:|---:|---:|
| Long-only P90 | **+8.68%** | 0.480 | 10.03% | 0.262 | 75.5 |
| Symmetric P90 | +7.67% | **0.592** | **6.43%** | 0.210 | **99.0** |

The symmetric model has slightly lower absolute profit but better Sharpe and drawdown. Its empirical random rank is much higher because symmetric random policies perform extremely poorly in this bull market.

## Main conclusions

### 1. The old symmetric peak did not reproduce

The historical test-selected symmetric result (`+22.37%` at RTG 0.8 in the stored comparison, and higher exploratory figures in earlier padding reports) did not reproduce under final-epoch, in-support, dynamic-RTG evaluation. Symmetric DT V2 peaks at +7.67% at the pre-specified P90 target.

This is consistent with the hypothesis that earlier results were materially affected by test-based checkpoint selection, RTG sweeping, duplicated trajectories and historical implementation differences.

### 2. RTG conditioning controls direction

Symmetric DT changes from negative exposure at P10/P50 to positive exposure at P90:

```text
P10 executed position: -0.052
P50 executed position: -0.043
P90 executed position: +0.210
```

This is clearer directional controllability than the long-only model, although it does not produce competitive absolute beta capture.

### 3. A high random percentile is relative, not absolute

Symmetric DT P90 is above every symmetric random path, but its +7.67% return remains far below passive Buy & Hold at +83.25%. The high percentile mainly shows that the model avoids the severe short/turnover losses of symmetric random actions.

### 4. Action-space flexibility did not increase absolute return

Long-only DT P90 returns +8.68%; symmetric DT P90 returns +7.67%. Under this protocol, allowing negative actions improves risk-adjusted performance but does not increase absolute profit.

## Limitations

- One training seed per BC/DT model
- Same market path for all random policies
- Previously reused test period; retrospective audit only
- Existing experts retain unstable DSR reward
- Ten-year training trajectories versus 2.5-year evaluation horizon
- ATR/stop/drawdown overlay transforms actions and creates many flat steps
- Different expert checkpoints are used for the two action spaces
- No multiple-testing correction for earlier exploratory RTG sweeps

## Reproduction

```bash
PYTHONUNBUFFERED=1 .venv/bin/python scripts/build_mvp_dataset.py \
  --no-long-only \
  --sac-model outputs/v2_causal_symmetric/sac.zip \
  --ppo-model outputs/v2_causal_symmetric/ppo.zip \
  --output outputs/symmetric_v2_mvp/dataset_v2.pkl \
  --qa-output outputs/symmetric_v2_mvp/dataset_qa.json

PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_bc.py \
  --dataset outputs/symmetric_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/symmetric_v2_mvp/bc \
  --n-epochs 20 --step-stride 20 \
  --checkpoint-selection final_epoch --no-long-only

PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_dt.py \
  --dataset outputs/symmetric_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/symmetric_v2_mvp/dt \
  --n-epochs 20 --step-stride 20 \
  --checkpoint-selection final_epoch --no-long-only

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/evaluate_mvp.py --no-long-only \
  --dataset outputs/symmetric_v2_mvp/dataset_v2.pkl \
  --dt-checkpoint outputs/symmetric_v2_mvp/dt/dt_model.pt \
  --bc-checkpoint outputs/symmetric_v2_mvp/bc/bc_model.pt \
  --sac-model outputs/v2_causal_symmetric/sac.zip \
  --ppo-model outputs/v2_causal_symmetric/ppo.zip \
  --v1-comparison outputs/comparison_dt_bc.pkl \
  --output-dir outputs/symmetric_v2_mvp

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/evaluate_paired_random.py
```

