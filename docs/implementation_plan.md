# Long-Only Decision Transformer：One-Day MVP Implementation Plan

## 1. Project objective

本計畫的目標是在一天內完成一個可重現、可解釋、適合 school project 報告的 Long-Only 實驗版本，而不是在截止前重構整套 Online RL 系統。

本次 MVP 聚焦回答以下研究問題：

> 在動作空間限制為 `[0, 1]` 時，修正模型輸出範圍、移除重複專家軌跡並加入完整 raw-action coverage 後，Decision Transformer（DT）相對於 Behavior Cloning（BC）的表現是否改善？

本次實驗保留既有 SAC/PPO checkpoint，不重新設計 reward 或重訓 online experts。DSR 數值不穩定、RTG horizon mismatch、volatility scaling 等問題會被記錄為已確認的限制與 future improvements。

---

## 2. Background and experimental interpretation

專案目前經歷三個主要階段：

1. **Architecture debugging**：修正 Transformer padding mask，並觀察到 symmetric action-space 模型回測改善。
2. **Long-only adaptation**：將 action space 改為 `[0, 1]`，發現 BC 優於 DT，且策略普遍維持低曝險。
3. **Experimental audit**：確認 DSR reward spikes、Sharpe scale-invariance 所造成的低曝險退化解、重複 expert trajectories、RTG horizon mismatch、Long-only output head mismatch，以及 test-based checkpoint selection。

Padding 修正前後的數值應描述為「修正、固定 seed 並重新訓練後觀察到改善」，不應宣稱收益提升完全由 padding mask 單獨造成。既有最高回測結果也應定位為 exploratory result，因為 checkpoint 與 RTG target 曾使用 2024–2026 test period 進行選擇。

---

## 3. Scope

### 3.1 Must-have changes

本日必須完成：

1. Long-only DT/BC output head 改為 `[0, 1]`。
2. 評估結果同時記錄 raw action 與 executed position。
3. 正式 `Avg Position` 改由 `position_history` 計算。
4. 建立包含 stochastic experts 與 fixed-target demonstrations 的 Dataset V2。
5. 對 Dataset V2 執行 duplicate/action/RTG quality audit。
6. 使用 Dataset V2 各訓練一個 BC V2 與 DT V2。
7. 新模型不使用 test Sharpe 選 checkpoint。
8. 使用 dynamic RTG 評估 DT。
9. 產生 V1/V2 比較表與至少一張 equity-curve 圖。

### 3.2 Explicitly out of scope today

今天不執行：

- 不修改 DSR reward 後重訓 SAC/PPO。
- 不直接加入未校準的固定 `epsilon=1e-4`。
- 不重構 volatility scaling。
- 不建立 rolling 252-day trajectory pipeline。
- 不跑完整 multi-training-seed study。
- 不跑大範圍 RTG extrapolation sweep。
- 不將 deterministic evaluation seeds 誤稱為模型統計顯著性。

---

## 4. Implementation tasks

### Task 0 — Preserve V1 artifacts

建立獨立輸出目錄：

```text
outputs/long_only_v1_original/
outputs/long_only_v2_mvp/
```

V1 需要保留：

- DT/BC/SAC/PPO checkpoints
- 原始 dataset
- comparison pickle/CSV
- RTG sensitivity results
- reward decomposition results
- 既有圖表

新程式不得覆蓋 V1 artifacts。

**Acceptance criteria**

- V1 的 DT `+12.98%` 與 BC `+18.41%` 結果仍可由保存的 artifacts 查閱。
- 所有 V2 outputs 都寫入 `outputs/long_only_v2_mvp/`。

### Task 1 — Make action output genuinely long-only

目前 DT 與 BC prediction head 使用 `Tanh()`。MVP 將模型建構函式加入 `long_only` 或 `action_activation` 設定：

```python
if long_only:
    output_activation = nn.Sigmoid()
else:
    output_activation = nn.Tanh()
```

Checkpoint metadata 必須保存 action mode，避免載入模型時使用錯誤 activation。

新增或更新測試：

```python
assert predictions.min() >= 0.0
assert predictions.max() <= 1.0
```

**Expected result**

- Long-only DT/BC 不再預測負 action。
- 不再依賴 environment clipping 修正模型輸出。
- Action loss 完整用於擬合有效 `[0, 1]` 區間。

### Task 2 — Correct action and position reporting

評估時分別記錄：

```text
avg_raw_action
avg_executed_position
turnover
action_saturation
stop_loss_events
```

正式報告中的 `Avg Position` 必須由：

```python
np.mean(info["position_history"])
```

計算，而不是 agent raw action 的平均值。

Buy & Hold 也要同時報告 raw target 與 executed exposure，以揭露 volatility scaling 是否改寫滿倉指令。

**Expected result**

- 能清楚區分「模型想持有多少」與「環境實際執行多少」。
- 避免把 raw action 誤解成 portfolio exposure。

### Task 3 — Build Dataset V2

沿用現有 SAC/PPO checkpoints，但使用 stochastic collection 取得不同 action paths。

建議資料組成：

| Source | Trajectories | Collection mode |
|---|---:|---|
| Random | 10 | Random action-space sampling |
| SAC | 10 | `deterministic=False` |
| PPO | 10 | `deterministic=False` |
| Fixed 0% | 1 | Constant `0.00` |
| Fixed 25% | 1 | Constant `0.25` |
| Fixed 50% | 1 | Constant `0.50` |
| Fixed 75% | 1 | Constant `0.75` |
| Buy & Hold | 1 | Constant `1.00` |

總計 35 條 trajectories。

每條 trajectory 必須加入 metadata：

```text
trajectory_id
policy_name
collection_seed
deterministic
action_hash
episode_length
starting_rtg
episode_return
episode_sharpe
```

使用 action sequence hash 檢查重複：

```python
sha256(np.asarray(actions, dtype=np.float32).tobytes())
```

如果 stochastic SAC/PPO 仍產生重複 paths，保留一條並在 QA report 中列出，不得用複製增加 sample count。

**Acceptance criteria**

- 至少 25 條 unique action paths。
- Dataset 包含從 0% 到 100% 的 raw-target demonstrations，並另外報告 ATR overlay 改寫後的 executed exposure。
- Dataset QA 輸出 policy distribution、duplicate count、action distribution、executed-position distribution 與 starting-RTG distribution。

### Task 4 — Prevent new test-based checkpoint selection

2024–2026 period 僅在 V2 模型完成後評估一次。

One-day MVP 採用最簡單且透明的規則：

```text
Train for 20 epochs
Save final epoch
Do not evaluate on the test environment during training
```

如果需要 early stopping，只能使用 held-out action-prediction validation loss；不得使用 test profit 或 test Sharpe。

建議設定：

```text
Epochs: 20
Context length: 20
Step stride: 20
Batch size: 64
Learning rate: 1e-4
Seed: 42
```

**Acceptance criteria**

- Training log 中不出現 2024–2026 test profit/Sharpe。
- Checkpoint metadata 記錄 `selection_rule = "final_epoch"`。

### Task 5 — Train BC V2 and DT V2

BC 與 DT 必須使用：

- 同一 Dataset V2
- 同一 state normalization
- 同一 context length
- 同一 training epochs
- 同一 optimizer settings
- Long-only sigmoid output

如果訓練超時，降級順序為：

1. Epochs 由 20 降至 15。
2. Step stride 由 20 提高至 30。
3. 保留 model size，不臨時修改架構以免增加新變因。

### Task 6 — Evaluate with in-support RTG targets

從 Dataset V2 的 trajectory starting RTG 計算：

```python
targets = np.percentile(starting_rtgs, [50, 75, 90])
```

正式評估只測 P50、P75、P90，不使用 `RTG=5.0` 等明顯超出 training support 的 target。

DT 使用 dynamic RTG：

```python
next_rtg = current_rtg - realised_net_return / rtg_scale
```

保留 V1 `RTG=2.4` 作歷史比較，但不得將其描述成 V2 的「100% expert intensity」。

### Task 7 — Produce final artifacts

至少輸出：

```text
outputs/long_only_v2_mvp/dataset_v2.pkl
outputs/long_only_v2_mvp/dataset_qa.json
outputs/long_only_v2_mvp/bc/bc_model.pt
outputs/long_only_v2_mvp/dt/dt_model.pt
outputs/long_only_v2_mvp/comparison.csv
outputs/long_only_v2_mvp/equity_curves.png
outputs/long_only_v2_mvp/run_config.json
```

比較表包含：

- Buy & Hold
- Fixed 25%、50%、75%
- SAC、PPO
- BC V1、DT V1
- BC V2
- DT V2 P50/P75/P90

指標包含：

- Cumulative return
- Annualized Sharpe
- Maximum drawdown
- Average raw action
- Average executed position
- Turnover
- Stop-loss events

---

## 5. Expected outcomes

### 5.1 Primary expected outcome

Dataset V2 加入完整 raw-target demonstrations 後，預期 DT 與 BC 的 action coverage 會提高，DT 的 RTG conditioning 也應比 V1 更容易在 P50/P75/P90 之間產生可觀察的 exposure 差異。由於既有環境仍包含 ATR overlay，raw target 不保證等於 executed exposure。

預期改善應主要出現在：

- 負 action 消失。
- Unique expert paths 增加。
- 高 RTG target 對應較高曝險的關係變清楚。
- DT 不再只能檢索同一條 SAC 或 PPO trajectory。
- 報告中的平均持倉能與 environment position history 對上。

### 5.2 Performance expectation

不預先承諾 DT 一定超越 BC 或 Buy & Hold。合理的成功標準是：

1. V2 實驗具備更好的實驗有效性與可重現性。
2. DT 對不同 in-support RTG targets 產生有系統的行為差異。
3. V2 DT 相較 V1 DT 提高 return、Sharpe 或 exposure controllability 中至少一項。

若 DT V2 仍輸給 BC，仍可形成有效結論：在 long-only 牛市與有限資料下，簡單的行為模仿可能比 return-conditioned sequence modeling 更穩健。

若加入 100% raw-target demonstrations 後所有模型都接近固定目標策略，則代表資料 coverage 對結果的影響大於模型架構；這同樣是有價值的 ablation finding。

### 5.3 Failure-safe deliverable

即使模型訓練結果沒有改善，今天仍必須交付：

- Correctness fixes
- Dataset duplicate audit
- V1/V2 methodology comparison
- DSR decomposition evidence
- Honest limitations section
- Reproducible commands and configs

School project 的主要貢獻可以定位為系統性 debugging 與 experimental audit，而不只追求最高回測收益。

---

## 6. Future improvements

以下項目留待 deadline 後或報告中的 future work：

### 6.1 Reward V2 and expert retraining

重新設計 online reward：

```python
reward = (
    return_coeff * net_return
    + dsr_coeff * clipped_dsr
    - turnover_coeff * abs(delta_position)
    - drawdown_coeff * drawdown_increase
)
```

建議加入：

- 20–60 step DSR warm-up
- calibrated variance floor
- DSR clipping，例如 `[-2, 2]`
- drawdown-increase penalty，而非每一步重複懲罰全部歷史 drawdown
- reward-component scale monitoring

修正後需重新訓練 SAC/PPO，再重建 offline dataset。

### 6.2 Horizon-consistent trajectories and RTG

將十年 trajectory 改成固定 252 或 504 trading-day rolling windows，並使用固定 horizon 的 cumulative log-return RTG。如此可減少 10 年 training RTG 與 2.5 年 test RTG 的尺度錯配。

RTG scaling 可由 `max(abs(RTG))` 改為 robust percentile scaling 或 standardization，降低 outlier 支配。

### 6.3 Remove or isolate volatility scaling

目前 raw action 會被 ATR-based scaling 改寫。未來應比較：

- Direct position control：action 就是實際 target position。
- Volatility overlay：action 經 volatility scaling。

兩者應作為獨立 ablation，不應混在 action-space 比較中。

同時應修正或移除目前未實際控制 scaling 的 `vol_target` 設定。

### 6.4 Proper validation and multiple training seeds

建立完全獨立的 train/validation/test workflow：

```text
Train: 2014–2021
Validation: 2022–2023
Test: 2024–2026
```

Dataset 也必須依日期切分，避免 validation dates 已存在於 supervised DT training samples。

最終模型至少訓練三個 seeds，例如 `42, 43, 44`，報告 model-training mean ± standard deviation。Deterministic policy 不應藉由更換 evaluation seed 假裝成 multi-seed robustness。

### 6.5 Regime-aware evaluation

增加 bull、bear、sideways 子期間，避免單一牛市使 Buy & Hold 天然占優。報告每個 regime 的 return、Sharpe、drawdown、exposure 與 turnover。

---

## 7. Git version-control plan

### 7.1 Current repository caution

目前 worktree 已存在多個未提交修改與新增檔案。這些修改應視為使用者既有工作，不可使用：

```text
git reset --hard
git checkout -- <file>
git clean -fd
```

也不要在未確認修改歸屬前直接執行 `git stash`，以免把不同實驗混在一起。

### 7.2 Recommended branch structure

建議使用：

```text
main
└── feature/long-only-action-space      # 目前歷史工作
    └── codex/long-only-mvp-audit       # 今日 MVP
```

由於目前 worktree 是 dirty state，安全流程為：

1. 先執行 `git status --short` 與 `git diff --stat`。
2. 確認目前 long-only 修改是否應先建立 checkpoint commit。
3. 將目前相關修改提交為明確的基準 commit。
4. 再建立 `codex/long-only-mvp-audit` branch。
5. 在新 branch 完成 MVP。

若不希望提交目前半成品，可直接從目前 branch 建立新 branch；未提交修改會跟著工作目錄移動。此方式可行，但第一個 commit 必須清楚區分「既有修改」與「本次 MVP 修改」。

### 7.3 Suggested commits

每個 commit 只處理一個可驗證主題：

```text
docs: add one-day long-only MVP implementation plan
fix: enforce long-only DT and BC action outputs
fix: report executed positions separately from raw actions
feat: collect diverse long-only trajectories and fixed exposures
fix: stop selecting DT checkpoints on the test period
feat: add reproducible long-only V2 evaluation
docs: document ablation results and experimental limitations
```

不要提交大型 generated artifacts，例如 `.pt`、`.pkl` 與高解析度圖，除非課程規定 repository 必須包含模型成果。建議在 `.gitignore` 排除：

```gitignore
outputs/**/*.pt
outputs/**/*.pkl
outputs/**/*.zip
outputs/**/*.png
```

但應提交小型、可讀的結果：

```text
run_config.json
dataset_qa.json
comparison.csv
Markdown report
```

如果圖表是作業報告的一部分，可以只提交最終精選圖，而不是所有中間圖。

### 7.4 Pre-commit verification

每個程式 commit 前至少執行：

```bash
.venv/bin/pytest -q
git diff --check
git status --short
```

模型訓練 commit 或 final report commit 前另執行：

```text
Dataset QA
Checkpoint metadata inspection
One deterministic evaluation reproduction
Artifact path verification
```

### 7.5 Tagging the submitted version

課程提交版本確認後建立 annotated tag：

```bash
git tag -a school-project-v1.0 -m "Long-only DT/BC school project submission"
```

只有在確認 commit、report 與 artifacts 都是最終版本後才建立 tag。若 repository 有 remote，再推送 branch 與 tag。

---

## 8. Execution ownership: Codex or Gemini

### Recommendation

**程式修改、測試、資料 QA、訓練 orchestration 與 Git 操作由 Codex 執行較合適；Gemini 可協助第二視角檢查圖表與潤飾報告。**

原因：

- Codex 目前已直接檢查 repository、dataset、checkpoint 與實際 evaluation path。
- 已確認 duplicate trajectories、DSR distributions、dynamic RTG 差異與 test-based checkpoint selection 的程式位置。
- 可以在同一個 workspace 中修改程式、跑 tests、監看訓練並保持 Git changes 分批可追蹤。
- 可避免把 Gemini 產生的報告結論與實際程式版本再次對不齊。

適合交給 Gemini 的工作：

- 檢查圖表敘事是否容易理解。
- 將已驗證的技術結果改寫成簡報語言。
- 針對最終 Markdown 提供第二輪語句校對。

不建議兩邊同時修改同一批程式檔案，否則容易產生環境設定、checkpoint metadata 與結果版本不一致。若需要共同工作，應由 Codex 先完成並 commit 程式版本，再把固定 commit hash、CSV 與圖表交給 Gemini 做報告 review。

---

## 9. One-day schedule and stop conditions

| Time | Deliverable |
|---|---|
| Hour 1 | Preserve V1, action-output fix, metric fix, tests |
| Hour 2–3 | Dataset V2 collection and QA |
| Hour 4–5 | BC V2 and DT V2 training |
| Hour 6 | P50/P75/P90 evaluation and comparison artifacts |
| Hour 7 | Report update, limitations and reproduction check |

Stop conditions：

- 若 stochastic expert collection 無法在 30 分鐘內正常產生 unique paths，改用現有 expert 各一條，加上 noisy fixed-exposure policies，並在報告揭露。
- 若 20 epochs 無法於時限內完成，降至 15 epochs；不臨時更改模型架構。
- 若 V2 performance 下降，不回頭用 test set 挑 checkpoint；保留結果並分析 dataset/model trade-off。
- 若時間只夠完成一個新模型，優先完成 DT V2，BC 使用既有 V1 作參考，但必須將比較限制寫清楚。

---

## 10. Definition of done

本 MVP 完成需要同時滿足：

- [ ] V1 artifacts 未被覆蓋。
- [ ] Long-only DT/BC prediction 位於 `[0, 1]`。
- [ ] Avg executed position 計算正確。
- [ ] Dataset V2 與 duplicate QA 已產生。
- [ ] 新模型 training 不使用 test Sharpe 選 checkpoint。
- [ ] BC V2 與 DT V2 checkpoint 已保存，或依 stop condition 清楚記錄降級結果。
- [ ] DT 使用 in-support targets 與 dynamic RTG 評估。
- [ ] Comparison CSV、run config 與 equity curve 已產生。
- [ ] Tests 通過，Git diff 無 whitespace errors。
- [ ] 報告清楚區分 observed result、causal claim、limitation 與 future work。
