# Symmetric `[-1, 1]` Matched Audit and 100-Random Percentile Plan

## 1. Purpose

本計畫延伸目前完成的 Long-Only MVP，以相同方法重新測試 symmetric action space `[-1, 1]`，並在兩種 action spaces 下各執行 100 條 Random policy paths，建立 empirical null distribution。

主要研究問題：

1. 在修正 padding、checkpoint selection、action semantics 與資料重複後，symmetric DT 是否仍優於 long-only DT？
2. DT、BC、SAC、PPO 在相同行情下，相對於 100 條 random action paths 位於哪個百分位？
3. 舊 symmetric DT `+42% ~ +44%` 的優勢，在不使用 test-based checkpoint/RTG selection 時是否仍然存在？

本實驗定位為：

> **Matched retrospective audit**

2024–2026 test period 已在過往實驗中被反覆使用，因此不得重新宣稱為 untouched out-of-sample test。

---

## 2. Why run 100 Random policies?

100 Random policies 可以回答：

> 在固定的 2024–2026 市場路徑與同一個 environment 下，一個模型的 profit、Sharpe 或 drawdown，相對於隨機 action paths 位於第幾百分位？

例如：

```text
DT profit = +8.7%
100 Random policies 中有 82 條 profit <= +8.7%
DT empirical profit percentile ≈ 82nd percentile
```

這是一個有用的 empirical reference distribution，特別適合判斷模型是否只是碰巧優於單一 Random seed。

### 2.1 What it does not prove

100 Random action seeds 不等於：

- 100 個 DT training seeds
- 100 個獨立市場樣本
- 100 個不同 test periods
- 模型結果具有 95% statistical confidence
- 策略可以泛化到未來市場

所有 Random policies 仍使用同一段 NASDAQ-100 2024–2026 market path。它們只改變 action randomness，不改變市場報酬路徑。

因此正確措辭是：

> The model ranks at the Xth percentile of 100 random-action policies on the same historical market path.

不應寫成：

> The model is statistically significant at 95% confidence.

### 2.2 Why deterministic DT/BC are not evaluated 100 times

DT、BC、deterministic SAC/PPO 在 deterministic environment 中，改變 evaluation seed 不會改變：

- market observations
- model actions
- equity curve
- final metrics

因此重複執行 deterministic DT 100 次只會得到同一個數字 100 次，沒有新的統計資訊。

若要估計模型本身的不確定性，需要：

```text
Train DT with seeds 42, 43, 44, ...
```

而不是：

```text
Evaluate the same DT checkpoint with seeds 42, 43, 44, ...
```

100 Random policies 與 multi-training-seed experiments 應分開報告。

---

## 3. Matched experimental design

除 action bounds 與 output activation 外，兩組設定必須一致。

| Setting | Long-only | Symmetric |
|---|---|---|
| Action space | `[0, 1]` | `[-1, 1]` |
| DT/BC output activation | Sigmoid | Tanh |
| Padding mask | Fixed | Fixed |
| Context length | 20 | 20 |
| Epochs | 20 | 20 |
| Step stride | 20 | 20 |
| Batch size | 64 | 64 |
| Learning rate | `1e-4` | `1e-4` |
| Training seed | 42 | 42 |
| Checkpoint selection | Final epoch | Final epoch |
| Test-time model selection | None | None |
| DT inference | Dynamic RTG | Dynamic RTG |
| Expert trajectories | Stochastic SAC/PPO | Stochastic SAC/PPO |
| Unique trajectory QA | Required | Required |
| Random evaluation paths | 100 | 100 |

本輪不修改 DSR reward、不重新調整 reward coefficients。否則無法將差異歸因於 action space。

---

## 4. Git and output isolation

建議從目前完成的 commit 建立新 branch：

```text
codex/long-only-mvp-audit
└── codex/symmetric-matched-audit
```

新輸出目錄：

```text
outputs/symmetric_v2_mvp/
outputs/action_space_matched_audit/
```

不得覆蓋：

```text
outputs/long_only/
outputs/long_only_v2_mvp/
outputs/v2_causal_symmetric/
```

預計 commits：

```text
docs: add symmetric matched-audit implementation plan
feat: parameterize MVP collection for symmetric actions
feat: add paired 100-random action-space evaluation
docs: report matched action-space audit results
```

---

## 5. Implementation tasks

### Task 1 — Parameterize Dataset V2 collector

目前 `build_mvp_dataset.py` 的 long-only 設定需要改為 CLI parameter：

```bash
--long-only
--no-long-only
```

環境設定：

```python
RiskConfig(long_only=args.long_only)
```

Model paths 必須與 action space 相容：

```text
Long-only SAC/PPO: outputs/long_only/*.zip
Symmetric SAC/PPO: outputs/v2_causal_symmetric/*.zip
```

載入 checkpoint 後驗證：

```python
assert env.action_space.low[0] == expected_low
assert env.action_space.high[0] == 1.0
assert model.action_space.low[0] == expected_low
```

若 checkpoint observation/action shape 不相容，立即停止，不得自動 clipping 後繼續。

### Task 2 — Build symmetric Dataset V2

建議組成：

| Source | Trajectories |
|---|---:|
| Random symmetric | 10 |
| SAC stochastic | 10 |
| PPO stochastic | 10 |
| Fixed targets `-1, -.5, 0, .5, 1` | 5 |
| **Total requested** | **35** |

Fixed targets 是 raw-action demonstrations。由於 agent environment 仍包含 ATR overlay，必須另外記錄 executed positions，不得將 raw target 直接稱為實際 exposure。

Dataset QA 必須輸出：

- requested trajectories
- unique action hashes
- duplicate count
- policy distribution
- raw-action min/mean/max
- executed-position min/mean/max
- long/flat/short step proportions
- starting RTG percentiles
- episode profit/Sharpe distribution

**Acceptance criteria**

- 至少 25 條 unique trajectories
- Raw-action range 覆蓋負值與正值
- Dataset metadata 明確記錄 `long_only = false`
- 所有 state/action dimensions 與 symmetric checkpoints 相容

### Task 3 — Train symmetric BC V2 and DT V2

訓練命令使用：

```bash
--no-long-only
--checkpoint-selection final_epoch
```

模型 activation：

```text
action_activation = tanh
```

訓練過程不得執行 2024–2026 test evaluation。

建議設定：

```text
Epochs: 20
Context length: 20
Step stride: 20
Batch size: 64
Seed: 42
```

Checkpoint metadata 必須包含：

```json
{
  "action_activation": "tanh",
  "long_only": false,
  "selection_rule": "final_epoch"
}
```

### Task 4 — Select in-support symmetric RTGs

由 symmetric Dataset V2 的 starting RTGs 計算：

```text
P10
P50
P90
```

使用 P10/P50/P90，而不是只使用高 percentiles，原因是 symmetric dataset 可能同時包含：

- negative-return/short trajectories
- neutral trajectories
- positive-return/long trajectories

正式報告需列出 target 的實際 RTG 數值，不得只寫 percentile label。

禁止使用 `RTG=5.0` 等 training support 外推值作主要結論。

### Task 5 — Correct symmetric baselines

Agent evaluation environment 保留既有 ATR overlay，以維持與 expert training 相容。

另外建立 direct-position baseline environment，繞過 ATR overlay 並關閉 hard risk rails：

```text
Always Short: -100%
Fixed Short: -50%
Cash: 0%
Fixed Long: +50%
Passive Buy & Hold: +100%
```

Direct baselines 與 agent strategies 必須明確分開標示。

每個模型與 baseline 都要報告：

- cumulative return
- annualized Sharpe
- maximum drawdown
- average raw action
- average executed position
- turnover
- stop-loss events

---

## 6. Paired 100-Random evaluation

### 6.1 Seeds

使用與舊實驗一致的 seeds：

```text
42–141
```

### 6.2 Paired action generation

為了讓 long-only 與 symmetric random paths 具有可比較的底層亂數，對每一 seed 先生成：

```python
u_t ~ Uniform(0, 1)
```

再映射：

```python
long_only_action_t = u_t
symmetric_action_t = 2 * u_t - 1
```

如此每一組 seed 使用相同的 quantile draws：

- `u=0.9` 對應 long-only `+0.9`
- `u=0.9` 對應 symmetric `+0.8`
- `u=0.1` 對應 long-only `+0.1`
- `u=0.1` 對應 symmetric `-0.8`

這是 paired random design，可降低兩組 action-space 比較中的 RNG 差異。

### 6.3 Random-policy outputs

每個 action space 保存 100 rows：

```text
seed
action_space
profit
sharpe
max_drawdown
avg_raw_action
avg_executed_position
turnover
stop_loss_events
```

Aggregate statistics：

- mean
- standard deviation
- min/max
- P5/P25/P50/P75/P95
- histogram or empirical CDF

### 6.4 Empirical percentile calculation

對「越高越好」的指標，例如 profit、Sharpe：

```python
percentile = 100 * (1 + count(random_metric <= model_metric)) / (N + 2)
```

使用 Laplace smoothing（分子 `+1`、分母 `+2`），避免有限樣本中產生精確 0th 或 100th percentile。

對「越低越好」的 max drawdown：

```python
percentile = 100 * (1 + count(random_max_dd >= model_max_dd)) / (N + 2)
```

Turnover 與 average exposure 沒有單一「越高越好」方向，只報告 distribution position，不轉成 performance percentile。

### 6.5 Model table

輸出格式：

| Model | Action space | Profit | Profit percentile | Sharpe | Sharpe percentile | Max DD | DD percentile |
|---|---|---:|---:|---:|---:|---:|---:|
| BC | `[0,1]` | ... | ... | ... | ... | ... | ... |
| DT P50 | `[0,1]` | ... | ... | ... | ... | ... | ... |
| DT P90 | `[0,1]` | ... | ... | ... | ... | ... | ... |
| BC | `[-1,1]` | ... | ... | ... | ... | ... | ... |
| DT P10 | `[-1,1]` | ... | ... | ... | ... | ... | ... |
| DT P90 | `[-1,1]` | ... | ... | ... | ... | ... | ... |

每個模型只能和相同 action space 的 random distribution 比較。

---

## 7. Statistical interpretation

### Permitted claims

- `DT is above the 90th percentile of random-action policies on this market path.`
- `Symmetric random policies have a lower median return than long-only random policies.`
- `DT exhibits a higher empirical rank than BC relative to the same random-policy reference.`

### Prohibited claims

- `DT is significant at the 99% confidence level.`
- `There is a 99% probability DT will outperform in the future.`
- `100 random seeds prove market generalization.`
- `Changing the evaluation seed provides 100 independent DT results.`

### Future confirmatory statistics

若要建立更強的統計證據，未來需加入：

1. 3–5 個 model training seeds
2. 多個 non-overlapping test windows
3. Bull/bear/sideways regime splits
4. Block bootstrap of daily returns
5. White's Reality Check 或其他 multiple-testing adjustment（若大量 sweep RTG/hyperparameters）

---

## 8. Expected outcomes

### Scenario A — Symmetric DT remains strong

若 final-epoch symmetric DT 仍顯著高於 symmetric Random P95，且優於 long-only DT：

> 提供證據顯示 negative exposure 擴大了 DT 可檢索與組合的策略行為，action-space flexibility 是歷史改善的一部分來源。

仍不可宣稱完全因果，因為 symmetric 與 long-only experts 本身也不同。

### Scenario B — Symmetric DT falls substantially

若 symmetric V2 遠低於舊版 +42%～+44%：

> 舊結果可能主要受到 test-selected checkpoint、RTG sweep、duplicated trajectories 或 historical implementation differences 影響。

### Scenario C — DT rank is high but absolute return is low

若 DT absolute profit 不高，但仍位於 Random P90/P95：

> 模型相對 random actions 有學習訊號，但 environment/reward/data constraints 限制其 beta capture 與 absolute performance。

### Scenario D — Long-only Random dominates symmetric Random

在 2024–2026 強牛市中，這是合理預期：

- Long-only random actions 平均帶有 positive beta
- Symmetric random actions 的平均 raw exposure 接近 0
- Symmetric random 策略容易錯過市場上行

因此兩種 random distributions 不能混合成同一個 null distribution。

---

## 9. Artifacts

預計輸出：

```text
outputs/symmetric_v2_mvp/dataset_v2.pkl
outputs/symmetric_v2_mvp/dataset_qa.json
outputs/symmetric_v2_mvp/bc/bc_model.pt
outputs/symmetric_v2_mvp/dt/dt_model.pt
outputs/symmetric_v2_mvp/comparison.csv
outputs/symmetric_v2_mvp/equity_curves.png
outputs/symmetric_v2_mvp/run_config.json

outputs/action_space_matched_audit/random_100_paths.csv
outputs/action_space_matched_audit/random_summary.csv
outputs/action_space_matched_audit/model_percentiles.csv
outputs/action_space_matched_audit/random_profit_ecdf.png
outputs/action_space_matched_audit/action_space_comparison.png
```

Git 中提交小型 artifacts：

```text
docs/symmetric_matched_audit_results.md
docs/random_100_summary.csv
docs/model_random_percentiles.csv
```

大型 checkpoints、pickle datasets 與中間圖片保留於 ignored `outputs/`。

---

## 10. Tests and acceptance criteria

新增 tests：

- Collector 正確建立 `[0,1]` 與 `[-1,1]` action spaces
- Fixed symmetric targets 包含負 action
- Paired random mapping 符合 `long=u`、`symmetric=2u-1`
- 同 seed 可重現相同 random action path
- 不同 seeds 產生不同 action hashes
- Model percentile 計算方向正確
- Max-drawdown percentile 使用 lower-is-better 方向
- Symmetric checkpoint 使用 tanh
- Long-only checkpoint 使用 sigmoid
- Training default 不使用 test Sharpe selection

Definition of done：

- [x] Symmetric Dataset V2 至少 25 unique paths
- [x] Symmetric BC/DT final-epoch checkpoints 已產生
- [x] DT P10/P50/P90 已使用 dynamic RTG 評估
- [x] Long-only 與 symmetric 各有 100 paired Random paths
- [x] Model percentile table 已產生
- [x] 所有 baselines 的 raw/executed position 已分開
- [x] Full test suite 通過
- [x] `git diff --check` 通過
- [x] 報告未將 empirical percentile 誤稱為 confidence level

---

## 11. Estimated execution time

| Task | Estimate |
|---|---:|
| Parameterize collection/evaluation and tests | 45–75 min |
| Collect symmetric Dataset V2 | 5–15 min |
| Train symmetric BC/DT | 10–20 min |
| Evaluate symmetric models | 5–10 min |
| Run paired 100 Random paths | 5–15 min |
| Generate tables, plots and report | 30–45 min |
| **Total** | **Approximately 2–3 hours** |

若時間不足，優先順序：

1. Symmetric Dataset V2 + BC/DT
2. 100 paired Random distributions
3. Percentile tables
4. Additional plots
5. Multi-training-seed experiments 留作 future work
