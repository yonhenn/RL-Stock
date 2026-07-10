# Long-Only MVP Results

## Run definition

- Branch: `codex/long-only-mvp-audit`
- Dataset: 35 unique trajectories, no exact duplicates
- Training period: 2014–2023
- Test period: 2024-01-02 to 2026-06-22
- BC/DT epochs: 20
- Context length: 20
- Step stride: 20
- Training seed: 42
- Action output: sigmoid (`[0, 1]`)
- Checkpoint selection: final epoch
- Test metrics were not used during training
- DT evaluation: dynamic RTG at Dataset V2 starting-RTG P50/P75/P90

## Dataset V2 QA

| Source | Unique trajectories |
|---|---:|
| Random | 10 |
| SAC stochastic | 10 |
| PPO stochastic | 10 |
| Fixed raw targets 0/25/50/75/100% | 5 |
| **Total** | **35** |

- Exact duplicates removed: 0
- Raw-action range: 0.0–1.0
- Executed-position range: 0.0–1.0
- Starting RTG P50: 1.2929
- Starting RTG P75: 2.2652
- Starting RTG P90: 2.2960

## Test results

| Strategy | Profit | Sharpe | Max DD | Avg raw action | Avg executed position |
|---|---:|---:|---:|---:|---:|
| Random, seed 42 | +20.48% | 1.237 | 5.30% | 0.501 | 0.313 |
| Fixed 25% direct | +17.53% | 1.282 | 6.17% | 0.250 | 0.250 |
| Fixed 50% direct | +37.20% | 1.282 | 12.04% | 0.500 | 0.500 |
| Fixed 75% direct | +59.09% | 1.282 | 17.63% | 0.750 | 0.750 |
| Passive Buy & Hold | +83.25% | 1.282 | 22.93% | 1.000 | 1.000 |
| SAC V1 | +17.34% | 1.103 | 5.93% | 0.456 | 0.283 |
| PPO V1 | +16.01% | 0.770 | 13.45% | 0.565 | 0.374 |
| BC V2 | +0.86% | 0.347 | 1.20% | 0.046 | 0.028 |
| DT V2, RTG P50 | -0.56% | -0.098 | 3.57% | 0.064 | 0.041 |
| DT V2, RTG P75 | +8.43% | 0.468 | 9.98% | 0.425 | 0.260 |
| DT V2, RTG P90 | +8.68% | 0.480 | 10.03% | 0.427 | 0.262 |

## Findings

### 1. Dataset diversity did not recover absolute performance

BC V2 and DT V2 did not exceed the historical V1 models. This means duplicated trajectories were a real data-quality defect, but removing duplicates and adding fixed-target action demonstrations was not sufficient to solve the long-only performance problem. The fixed targets were still transformed by the existing ATR overlay during collection; they expand raw-action coverage but are not direct-position demonstrations.

The V2 result must not be replaced by a test-selected earlier checkpoint. The final-epoch result is retained as the pre-specified MVP outcome.

### 2. DT retained return-conditioning controllability

DT V2 clearly changed its exposure with commanded RTG:

- P50: average executed position 0.041
- P75: average executed position 0.260
- P90: average executed position 0.262

The large P50-to-P75 change shows that RTG conditioning affects behavior. The near-identical P75/P90 paths show saturation near the upper portion of the training RTG distribution rather than smooth monotonic control.

### 3. BC showed an out-of-sample low-exposure collapse

BC V2 predicted an average raw action of only 0.046 on the test period even though Dataset V2's global mean raw action was approximately 0.489. Possible causes include state-distribution shift, autoregressive error accumulation, final-epoch generalization, and the mixed-policy action target being multi-modal.

Diagnosing or correcting this after seeing test performance would create another test-selection loop, so it is recorded as a finding rather than tuned away in the MVP.

### 4. The historical Buy & Hold baseline was not true Buy & Hold

The historical comparison sent action `1.0` through ATR volatility scaling. Its average executed exposure was only approximately 0.626, producing the previously reported +38.14% result.

The corrected passive baseline bypasses volatility scaling and disables hard stop/drawdown rails. It keeps executed exposure at 1.0 and returns +83.25%. Therefore, the earlier +38.14% line should be renamed `100% raw target with volatility scaling`, not Buy & Hold.

### 5. Raw actions and portfolio positions are materially different

For SAC V1, average raw action was 0.456 while average executed position was 0.283. For PPO V1 the corresponding values were 0.565 and 0.374. Any action-space comparison must report both quantities or disable the hidden ATR overlay.

## Interpretation

The one-day MVP improves experimental correctness more than trading performance. Its strongest contributions are:

1. Long-only outputs are implemented at the model level.
2. The expert dataset contains genuinely distinct paths.
3. Test performance no longer selects checkpoints during training.
4. RTG targets are selected from training support and evaluated dynamically.
5. Passive and fixed-exposure baselines now mean what their labels claim.
6. Raw model actions are separated from executed portfolio exposure.

The result supports a cautious conclusion: DT exhibits return-conditioned exposure control, but the current reward, trajectory horizon, environment overlay and mixed-policy dataset do not produce competitive long-only out-of-sample returns.

## Reproduction commands

```bash
.venv/bin/python scripts/build_mvp_dataset.py

.venv/bin/python scripts/train_bc.py \
  --dataset outputs/long_only_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/long_only_v2_mvp/bc \
  --n-epochs 20 --step-stride 20 \
  --checkpoint-selection final_epoch --long-only

.venv/bin/python scripts/train_dt.py \
  --dataset outputs/long_only_v2_mvp/dataset_v2.pkl \
  --output-dir outputs/long_only_v2_mvp/dt \
  --n-epochs 20 --step-stride 20 \
  --checkpoint-selection final_epoch --long-only

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/evaluate_mvp.py

.venv/bin/pytest -q
```

## Verification

- Full test suite: 107 passed
- `git diff --check`: clean after whitespace cleanup
- Dataset exact duplicates: 0
- New checkpoint metadata:
  - `action_activation = sigmoid`
  - `long_only = true`
  - `selection_rule = final_epoch`

## Remaining limitations

- Single training seed
- One long bullish test period
- Ten-year trajectories and 2.5-year evaluation still have horizon mismatch
- Existing SAC/PPO experts still use unstable DSR reward
- Agent action remains modified by ATR volatility scaling
- Dataset policies share the same historical market path
- Hyperparameter choices were informed by earlier exploratory test iterations
