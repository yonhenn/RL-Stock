# RL 交易系統視覺化儀表板：可行性說明

> **專案**: RL wit Nasdaq — NASDAQ-100 強化學習交易系統  
> **目標**: 建立統一的視覺化模組，支援 Online RL（PPO/SAC/TD3/DDPG）與 Offline RL（Decision Transformer）的成果比較  
> **日期**: 2025-06-25

---

## 1. 現有視覺化能力 vs 缺口

### 已有（`visualization.py` + `evaluation.py`）

| 圖表 | 函數 | 狀態 |
|------|------|------|
| Strategy Return 柱狀圖 | `plot_strategy_returns()` | ✅ |
| Position vs Price 疊合圖 | `plot_position_history()` | ✅ |
| Equity Curve + Drawdown | `plot_equity_curve()` | ✅ |
| Risk Dashboard（三面板） | `plot_risk_dashboard()` | ✅ |
| Summary 指標表 | `print_summary()` | ✅ 文字 |

### 缺少

| 類別 | 缺少的圖表 | 重要性 |
|------|-----------|--------|
| **Action 行為分析** | Action 分佈直方圖、Action Saturation 分析、Action 自相關 | 🔴 |
| **Reward 診斷** | 各 reward 分量分解、累積 reward 曲線、Reward 分佈 | 🔴 |
| **風險深度分析** | Rolling Sharpe/Sortino、Drawdown 持續時間、VaR/CVaR | 🟡 |
| **Online vs Offline 對比** | 多模型並排比較、雷達圖、分佈對比 | 🔴 |
| **訓練過程** | Learning curve、Policy entropy/loss | 🟡 |
| **市場條件分析** | 不同 regime 下的績效、Turbulence 期間行為 | 🟡 |

---

## 2. 規劃的視覺化圖表：完整清單

### Panel A：Performance Overview（績效總覽）

#### A1. 多策略 Equity Curve 疊合圖
- **內容**: 所有策略（Random, SAC, PPO, DT 等）的 equity curve 畫在同一張圖，可直接比較報酬軌跡
- **資料來源**: `info["equity_curve"]`（每個策略的 evaluation episode）
- **附加**: Buy & Hold baseline 作為參考線
- **技術**: matplotlib, 不同顏色/線型區分策略

#### A2. 績效雷達圖 (Radar Chart)
- **內容**: 每個策略的多維績效指標繪製在同一張雷達圖上
- **維度**: Total Return, Sharpe, Sortino, Max Drawdown⁻¹, Win Rate, Calmar
- **目的**: 一眼看出各策略的強項和弱項，特別適合 Online vs Offline RL 的對比
- **技術**: matplotlib polar projection

#### A3. 績效指標表（熱力圖版）
- **內容**: 將目前文字版的 `print_summary()` 升級為 color-coded 表格
- **欄位**: Return, Sharpe, Sortino, Max DD, Calmar, Win Rate, Turnover, Stop Events
- **顏色**: 綠（好）→ 紅（差），按欄位 rank normalize
- **技術**: matplotlib table 或 seaborn heatmap

---

### Panel B：Action Behavior Analysis（動作行為分析）

#### B1. Action 分佈直方圖
- **內容**: 每個策略的 action（target_position）分佈
- **呈現**: 多策略並排直方圖或 KDE 曲線疊合
- **資料來源**: `info["position_history"]`
- **診斷目標**: 觀察 agent 是否偏好特定倉位（如永遠做多）

#### B2. Action Saturation 分析
- **定義**: Action 飽和 = agent 持續輸出接近邊界值（±1.0）的比例
- **內容**:
  - 飽和率柱狀圖：`|action| > 0.95` 的 timestep 比例
  - 時序飽和圖：X 軸時間、Y 軸 `|action|`，標記飽和區間
- **診斷目標**: PPO 常見的 action collapse 問題——如果飽和率 > 50%，policy 可能已退化為 bang-bang 控制
- **閾值建議**: `< 20%` 健康、`20~50%` 需注意、`> 50%` 問題嚴重

#### B3. Action 變化率 (Turnover) 時序圖
- **內容**: `|action_t - action_{t-1}|` 隨時間的變化
- **附加**: Rolling mean 平滑線 + 總 turnover 標註
- **資料來源**: `np.abs(np.diff(position_history))`
- **診斷目標**: 過度交易 vs 適度調倉

#### B4. Action 自相關圖 (ACF)
- **內容**: Action 序列的 autocorrelation function
- **診斷目標**: 高自相關 = agent 傾向長期持有；低自相關 = agent 頻繁反轉
- **技術**: `statsmodels.graphics.tsaplots.plot_acf` 或手動計算

---

### Panel C：Reward Diagnostics（獎勵診斷）

#### C1. Reward 分量分解堆疊圖
- **內容**: 每一步 reward 拆成 DSR、drawdown penalty、turnover penalty、trend align bonus 四個分量，用堆疊面積圖呈現
- **資料需求**: **新增** — 需要在 `step()` 回傳的 `info` 中記錄各分量
  ```python
  info["reward_components"] = {
      "dsr": sharpe_component * self.risk.sharpe_coeff,
      "drawdown": drawdown_penalty * self.risk.drawdown_coeff,
      "turnover": turnover_penalty * self.risk.turnover_coeff,
      "align": align_bonus,
  }
  ```
- **診斷目標**: 了解 reward 由哪個分量主導；如果 turnover penalty 永遠 dominate，agent 可能學到「不動」

#### C2. 累積 Reward 曲線
- **內容**: 各策略的 `cumsum(rewards)` 隨 timestep 的變化
- **附加**: 標記 stop-loss 事件和 drawdown lock 時間段
- **資料來源**: `info["returns"]` 的累積和

#### C3. Reward 分佈比較
- **內容**: 各策略的 per-step reward 分佈（violin plot 或 box plot）
- **診斷目標**: 比較 Online RL 和 DT 的 reward 分佈是否相似

---

### Panel D：Risk Deep Dive（風險深度分析）

#### D1. Rolling Sharpe Ratio
- **內容**: 滑動窗口（如 60 天）的 Sharpe ratio 時序圖
- **附加**: 標記 Sharpe < 0 的區間（紅色陰影）
- **診斷目標**: 策略是否在特定時段失效

#### D2. Drawdown 瀑布圖
- **內容**: 每次 drawdown 的深度和持續時間
- **呈現**: 橫軸 = drawdown 開始日、縱軸 = drawdown 深度、寬度 = 持續天數
- **附加**: 標註 Max Drawdown 事件
- **資料來源**: 從 `info["equity_curve"]` 計算 drawdown episodes

#### D3. Return 分佈 + VaR/CVaR
- **內容**: Daily return 的直方圖，標記 5% VaR 和 CVaR 線
- **公式**:
  - $\text{VaR}_{5\%} = \text{quantile}(returns, 0.05)$
  - $\text{CVaR}_{5\%} = \mathbb{E}[r \mid r \leq \text{VaR}_{5\%}]$
- **診斷目標**: 尾部風險評估

#### D4. 月度報酬熱力圖
- **內容**: X 軸 = 月份、Y 軸 = 年份、顏色 = 該月報酬率
- **技術**: seaborn heatmap
- **診斷目標**: 是否有季節性表現差異

---

### Panel E：Online vs Offline 比較（DT 專用）

#### E1. 條件績效比較（By RTG Level）
- **內容**: DT 在不同 RTG conditioning level 下的實際 achieved return
- **呈現**: X 軸 = 設定的 RTG、Y 軸 = 實際 achieved return
- **附加**: 理想線（y = x）作為參考
- **診斷目標**: DT 是否能有效回應不同的 RTG 目標

#### E2. Action 分佈對比：Behavior Policy vs DT
- **內容**: 收集資料用的 behavior policy（如 SAC）和 DT 的 action 分佈並排比較
- **診斷目標**: DT 是否學到了與 behavior policy 相似或更好的行為模式

#### E3. Attention 視覺化（DT 專屬）
- **內容**: Transformer 的 attention weight heatmap
- **呈現**: X/Y 軸 = context 中的 timestep、顏色 = attention weight
- **診斷目標**: DT 在做決策時「關注」哪些過去的 timestep——是看最近幾天？還是更遠的趨勢？
- **技術**: 從 DT forward pass 中提取 attention weights

#### E4. Train Dataset 覆蓋率圖
- **內容**: t-SNE 或 PCA 降維後的 state space 散點圖
- **顏色維度**: 不同 behavior policy 的 trajectory 用不同顏色
- **附加**: DT inference 時訪問到的 state 疊加在上面
- **診斷目標**: DT inference 是否超出了 training data 的覆蓋範圍（distribution shift）

---

### Panel F：Market Regime Analysis（市場條件分析）

#### F1. Regime 分段績效
- **內容**: 按 trend regime（Strong Up / Up / Neutral / Down / Strong Down）分段統計各策略的 Sharpe / Return
- **呈現**: Grouped bar chart
- **資料來源**: `info["regime"]` + 對應區間的 returns
- **診斷目標**: 哪些策略在哪些市場環境下表現最好

#### F2. Turbulence 回應圖
- **內容**: Turbulence index 時序圖，疊加 agent 的 position
- **標記**: Turbulence gate 觸發的時間點
- **診斷目標**: Agent 是否在高波動期適當減倉

---

### Panel G：訓練過程（Online RL 專屬）

#### G1. Learning Curve
- **內容**: 訓練過程中的 episode reward 隨 timestep 的變化
- **資料需求**: **新增** — 需要擴展 `ProgressBarCallback` 或使用 SB3 的 `EvalCallback` 記錄中間評估
- **呈現**: 原始值（淺色）+ 平滑線（深色）

#### G2. Policy Entropy 曲線（PPO/SAC）
- **內容**: SAC 的 entropy coefficient 或 PPO 的 policy entropy 隨訓練的變化
- **資料需求**: 從 SB3 的 logger 提取
- **診斷目標**: Entropy 過快衰減 = 過早收斂；不衰減 = 學不到東西

---

## 3. 資料需求與 Env 修改

### 需要新增到 `info` dict 的欄位

```python
# envs_v2.py step() 的 info 中新增以下欄位：

# 1. Reward 分量（Panel C1 需要）
info["reward_components"] = {
    "dsr": sharpe_component * self.risk.sharpe_coeff,
    "drawdown": drawdown_penalty * self.risk.drawdown_coeff,
    "turnover": turnover_penalty * self.risk.turnover_coeff,
    "align": align_bonus,
}

# 2. Raw action vs executed action（Panel B, E2 需要）
info["raw_action"] = raw_action        # agent 原始輸出
info["executed_action"] = target_position  # 轉換後實際執行

# 3. 各分量原始值（計算用）
info["net_return"] = net_return
info["price"] = current_price
```

### 需要擴展的 Callback（Panel G 需要）

```python
# evaluation.py 新增
class EvalLoggingCallback(BaseCallback):
    """記錄訓練過程中的中間評估指標"""
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5):
        ...
    # 每 eval_freq 步做一次中間評估，記錄 Sharpe / Return / Equity
```

---

## 4. 技術實現方案

### 輸出格式選項

| 格式 | 優點 | 缺點 | 推薦場景 |
|------|------|------|----------|
| **靜態 PNG/PDF** (matplotlib) | 已有基礎、可嵌入報告 | 無互動 | 論文、報告 |
| **互動式 HTML** (Plotly) | 可縮放、hover 看數值、可分享 | 需要額外依賴 | 日常分析、展示 |
| **Streamlit Dashboard** | 即時互動、可選策略/參數 | 需要 server | 完整儀表板 |
| **Jupyter Notebook** | 混合程式碼與圖表 | 不便分享 | 開發階段 |

### 推薦方案：雙軌制

1. **核心圖表**用 matplotlib 產出 PNG（嵌入現有 pipeline，`train.py` 自動輸出）
2. **進階互動分析**用 Plotly 產出 standalone HTML（像你現有的 `algo_comparison.html`）

### 模組架構

```
src/rl_stock/
├── visualization.py          # 現有，保持向後相容
├── viz/                      # 新增視覺化子模組
│   ├── __init__.py
│   ├── action_analysis.py    # Panel B: Action 行為分析
│   ├── reward_diagnostics.py # Panel C: Reward 診斷
│   ├── risk_analysis.py      # Panel D: 風險深度分析
│   ├── comparison.py         # Panel A + E: 績效總覽 + Online vs Offline
│   ├── regime_analysis.py    # Panel F: 市場條件分析
│   ├── training_curves.py    # Panel G: 訓練過程
│   └── dashboard.py          # 整合入口：一鍵生成完整 HTML 報告
```

---

## 5. 各圖表可行性與優先級

| 圖表 | 可行性 | 資料就緒？ | 需要改 Env？ | 優先級 |
|------|--------|-----------|-------------|--------|
| A1. Equity Curve 疊合 | ✅ 簡單 | ✅ 有 | ❌ | P0 |
| A2. 雷達圖 | ✅ 簡單 | ✅ 有 | ❌ | P0 |
| A3. 指標熱力圖 | ✅ 簡單 | ✅ 有 | ❌ | P0 |
| B1. Action 分佈 | ✅ 簡單 | ✅ 有 `position_history` | ❌ | P0 |
| B2. Action Saturation | ✅ 簡單 | ✅ 有 | ❌ | P0 |
| B3. Turnover 時序 | ✅ 簡單 | ✅ 有 | ❌ | P1 |
| B4. Action ACF | ✅ 中等 | ✅ 有 | ❌ | P2 |
| C1. Reward 分量分解 | ✅ 中等 | ❌ 缺 | ✅ 需要新增 `info` 欄位 | P1 |
| C2. 累積 Reward | ✅ 簡單 | ✅ 有 `returns` | ❌ | P1 |
| C3. Reward 分佈 | ✅ 簡單 | ✅ 有 | ❌ | P1 |
| D1. Rolling Sharpe | ✅ 中等 | ✅ 有 `returns` | ❌ | P1 |
| D2. Drawdown 瀑布 | ✅ 中等 | ✅ 有 `equity_curve` | ❌ | P1 |
| D3. VaR/CVaR | ✅ 簡單 | ✅ 有 | ❌ | P2 |
| D4. 月度熱力圖 | ✅ 中等 | ⚠️ 需要日期對應 | ❌ | P2 |
| E1. RTG vs Achieved | ✅ 中等 | ❌ 待 DT 實作 | ❌ | P2 (DT 之後) |
| E2. Action 對比 | ✅ 簡單 | ❌ 待 DT 實作 | ❌ | P2 (DT 之後) |
| E3. Attention 視覺化 | 🟡 較複雜 | ❌ 待 DT 實作 | ❌ | P3 (DT 之後) |
| E4. State 覆蓋率 | ✅ 中等 | ❌ 待 DT 實作 | ❌ | P3 (DT 之後) |
| F1. Regime 分段績效 | ✅ 中等 | ✅ 有 `regime` | ❌ | P1 |
| F2. Turbulence 回應 | ✅ 中等 | ✅ 有 `turbulence` | ❌ | P1 |
| G1. Learning Curve | ✅ 中等 | ❌ 缺 | ✅ 需要 Callback | P2 |
| G2. Entropy 曲線 | 🟡 較複雜 | ❌ 缺 | ✅ 需要 Logger | P3 |

---

## 6. 建議實施路線圖

### Phase 1 (P0)：立即可做，無需改 Env

> 使用現有 `info` dict 中已有的資料

- A1. 多策略 Equity Curve 疊合
- A2. 績效雷達圖
- A3. 指標熱力圖
- B1. Action 分佈直方圖
- B2. Action Saturation 分析

**預估工作量**: 1~2 天  
**交付物**: `viz/comparison.py` + `viz/action_analysis.py`

### Phase 2 (P1)：小幅改 Env，大幅增強洞察

> 在 `info` 中新增 `reward_components` 欄位

- C1. Reward 分量分解
- C2. 累積 Reward 曲線
- C3. Reward 分佈
- D1. Rolling Sharpe
- D2. Drawdown 瀑布圖
- B3. Turnover 時序圖
- F1. Regime 分段績效
- F2. Turbulence 回應圖

**預估工作量**: 2~3 天  
**交付物**: `viz/reward_diagnostics.py` + `viz/risk_analysis.py` + `viz/regime_analysis.py`

### Phase 3 (P2)：進階分析

- B4. Action ACF
- D3. VaR/CVaR
- D4. 月度熱力圖
- G1. Learning Curve

**預估工作量**: 1~2 天  
**交付物**: 擴展現有模組

### Phase 4 (P3)：DT 實作後

> 需要 Decision Transformer 模型先完成

- E1. RTG vs Achieved Return
- E2. Behavior vs DT Action 對比
- E3. Attention 視覺化
- E4. State 覆蓋率 (t-SNE)
- G2. Entropy 曲線

**預估工作量**: 2~3 天  
**交付物**: `viz/dt_analysis.py` + `viz/training_curves.py`

---

## 7. 整合入口：一鍵生成報告

最終目標是在 `train.py` 加入一個選項，訓練結束後自動生成完整的 HTML 分析報告：

```bash
python train.py --data-path nasdaq100.xlsx --generate-report

# 輸出：
# outputs/
# ├── report.html          ← 完整互動式儀表板
# ├── strategy_returns.png
# ├── action_analysis.png
# ├── risk_dashboard.png
# └── summary.csv
```

或獨立執行：

```bash
python -m rl_stock.viz.dashboard \
    --results outputs/results.pkl \
    --output outputs/report.html
```
