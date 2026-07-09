# Decision Transformer 遷移建議書

> **專案**: RL wit Nasdaq — NASDAQ-100 強化學習交易系統  
> **目標**: 從目前的 PPO/SAC + Sliding Window 架構遷移至 Decision Transformer (DT)  
> **日期**: 2025-06-25  
> **狀態**: 設計階段

---

## 目錄

1. [背景與動機](#1-背景與動機)
2. [架構差異總覽](#2-架構差異總覽)
3. [Issue #1：雙重 Sliding Window 冗餘](#3-issue-1雙重-sliding-window-冗餘)
4. [Issue #2：RTG 與 DSR 的根本衝突](#4-issue-2rtg-與-dsr-的根本衝突)
5. [Issue #3：Portfolio Features 分佈漂移](#5-issue-3portfolio-features-分佈漂移)
6. [Issue #4：_resolve_target 轉換層位置](#6-issue-4_resolve_target-轉換層位置)
7. [Issue #5：離線資料集品質與收集](#7-issue-5離線資料集品質與收集)
8. [Issue #6：有狀態元件的重播問題](#8-issue-6有狀態元件的重播問題)
9. [Issue #7：Context Length K 的選擇](#9-issue-7context-length-k-的選擇)
10. [遷移 Checklist](#10-遷移-checklist)
11. [附錄：程式碼修改範例](#11-附錄程式碼修改範例)

---

## 1. 背景與動機

目前系統使用 `PositionMgmtEnv`（`envs_v2.py`）搭配 PPO/SAC 進行線上訓練。環境透過 **sliding window**（`window_size=30`）將過去 30 天的技術指標堆疊成 2D observation `(30, n_features)`，作為 Markovian agent 的唯一時序資訊來源。

Decision Transformer 是一種 **離線 RL** 方法，將 RL 問題重新框架為 sequence modeling：

$$\hat{a}_t = \text{DT}\left(\hat{R}_{t-K}, s_{t-K}, a_{t-K}, \ldots, \hat{R}_t, s_t\right)$$

其中 $\hat{R}_t$ 是 Return-to-Go（期望未來累積回報），DT 透過 Transformer 架構處理過去 K 步的 (RTG, state, action) 序列。

由於 DT **自身即具備時序建模能力**，直接套用目前的 sliding-window 環境會產生多個架構衝突。本文件逐一分析這些衝突並提出解決方案。

---

## 2. 架構差異總覽

|  | PPO / SAC（目前） | Decision Transformer |
|---|---|---|
| **學習範式** | Online RL — 與環境互動訓練 | Offline RL — 從固定 dataset 學習 |
| **時序資訊來源** | Env 的 sliding window（唯一來源） | DT 的 context window (K 步序列) |
| **策略條件** | $\pi(a_t \mid s_t)$ — 只看當前 state | $\pi(a_t \mid s_{t-K:t}, a_{t-K:t-1}, \hat{R}_{t-K:t})$ — 看序列 |
| **Reward 用途** | 直接驅動 policy gradient 更新 | 計算 RTG 作為 conditioning input |
| **探索機制** | Entropy bonus / noise | 無（受限於 dataset 品質） |
| **State 設計需求** | 需要包含歷史（window） | 單步即可（DT 提供時序） |

---

## 3. Issue #1：雙重 Sliding Window 冗餘

### 問題描述

目前每個 observation 是一個 `(30, n_features)` 的 2D tensor（env 的 sliding window）。如果 DT 再用 K=20 的 context window 包裝這些 observation，就變成 **window of windows**：

```
┌─────────────────── DT context (K=20) ───────────────────┐
│                                                          │
│  s_{t-19}      s_{t-18}      ...      s_{t-1}      s_t  │
│  ┌──────┐      ┌──────┐              ┌──────┐      ┌──┐ │
│  │30 days│      │30 days│              │30 days│      │30│ │
│  │  of   │      │  of   │              │  of   │      │  │ │
│  │ feat  │      │ feat  │              │ feat  │      │  │ │
│  └──────┘      └──────┘              └──────┘      └──┘ │
│  (30×19)        (30×19)               (30×19)       (30×19) │
└──────────────────────────────────────────────────────────┘

實際看到的唯一天數: 30 + 19 = 49 天
Input 大小: 20 × 30 × 19 = 11,400 維
冗餘率: 相鄰 state 重疊 29/30 行 ≈ 96.7%
```

### 影響

- **計算浪費**：Transformer attention 是 O(K²)，每個 token 本身又是 570~780 維
- **資訊冗餘**：96.7% 的數據是重複的
- **阻礙學習**：DT 的 self-attention 會被大量重複資訊淹沒

### 解決方案

**將 state 切回 single-step（1D 向量）**，讓 DT 的 context window 取代 env 的 sliding window：

| 模式 | Observation shape | 用途 |
|------|------------------|------|
| Window mode（目前） | `(window_size, n_features)` = `(30, 19~26)` | PPO / SAC |
| Flat mode（DT 用） | `(n_features,)` = `(19~26,)` | Decision Transformer |

**資訊損失評估**：幾乎沒有損失。技術指標本身已經 encode 了歷史資訊：

| 指標 | 隱含的歷史長度 |
|------|--------------|
| RSI(14) | 14 天 |
| MA60 ratio | 60 天 |
| MACD | 12/26 天 |
| Bollinger Bands(20) | 20 天 |
| Momentum(5) | 5 天 |
| ADX(14) | 14 天 |

DT 的 K=20 步序列再補上 step-to-step 的時序 pattern，資訊充足。

---

## 4. Issue #2：RTG 與 DSR 的根本衝突

### 問題描述

DT 的核心機制是 **conditioning on Return-to-Go (RTG)**：

$$\hat{R}_t = \sum_{k=t}^{T} r_k$$

但目前的 reward function 由多個分量組成：

```python
# envs_v2.py, _calculate_reward()
reward = (
    sharpe_component * sharpe_coeff      # DSR, ≈ O(0.01~1.0)
    + drawdown_penalty * drawdown_coeff  # ≤ 0, ≈ O(0~-0.5)
    + turnover_penalty * turnover_coeff  # ≤ 0, ≈ O(0~-0.1)
    + align_bonus                        # ≈ O(-0.3~0.3)
)
```

#### 問題 A：DSR 不可加

`IncrementalSharpe`（`risk.py`）是一個有狀態的統計量，其輸出取決於內部的 running mean `a` 和 second moment `b`。同一個 return 值在 episode 不同位置產生完全不同的 DSR：

```
Episode 開頭: return = +0.01 → DSR ≈ 0.0  (統計量未穩定)
Episode 中段: return = +0.01 → DSR ≈ 0.5  (統計量穩定)
Episode 結尾: return = +0.01 → DSR ≈ 0.3  (均值已被拉高)
```

把這些 DSR 值加總成 RTG **沒有直觀的物理意義**——它不代表「未來的 Sharpe ratio 還有多好」。

#### 問題 B：混合 reward 的尺度模糊

三個 reward 分量的數值範圍和物理意義不同，加總後的 RTG 難以被 DT 理解為一個有意義的目標。

### 解決方案

**訓練 DT 時改用可加且有直觀意義的 reward 來計算 RTG**，把 DSR/penalties 留給 PPO/SAC：

| RTG 候選 | 公式 | 優點 | 缺點 | 推薦 |
|----------|------|------|------|------|
| **累積 net return** | $\hat{R}_t = \sum_{k=t}^T r_k^{\text{net}}$ | 最直觀、可加、DT 原論文用法 | 沒有風險調整 | ⭐⭐⭐⭐⭐ |
| **累積 log return** | $\hat{R}_t = \sum_{k=t}^T \log(1 + r_k)$ | 可加 + 近似 geometric growth | 略複雜 | ⭐⭐⭐⭐ |
| **Final Sharpe** | RTG = episode 結束時的 annualized Sharpe（全程統一值） | 風險調整 | 不可分解到每步 | ⭐⭐⭐ |
| **Windowed Sharpe** | RTG = 未來 N 步的 rolling Sharpe | 折衷 | 引入新超參數 | ⭐⭐ |

**推薦方案**：使用環境中已有的 `net_return`（`envs_v2.py` step 函數中計算的 `portfolio_return - cost_drag`）的累積和作為 RTG。在 inference 時透過設定高 RTG 值來「要求」DT 產出好策略。

---

## 5. Issue #3：Portfolio Features 分佈漂移

### 問題描述

State 中的 6 個 portfolio features（`_get_portfolio_features()`）是 **endogenous（內生的）**——它們取決於 agent 自己過去的 action：

| Feature | 來源 | 內生性 |
|---------|------|--------|
| `position_ratio` | 上一步的 target_position | 🔴 完全內生 |
| `unrealized_pnl` | position × 價格變化 | 🟡 部分內生 |
| `drawdown` | equity vs peak | 🟡 部分內生 |
| `holding_norm` | 持倉天數 | 🔴 完全內生 |
| `volatility` | ATR / price | 🟢 外生 |
| `distance_to_stop` | PnL vs stop_loss_pct | 🟡 部分內生 |

**在 DT 離線訓練中**：training data 來自 behavior policy（例如 SAC），portfolio features 反映的是 SAC 的決策結果。在 DT inference 時，DT 的不同決策會導致完全不同的 portfolio state，產生 **distribution shift**：

```
訓練資料（SAC 收集）: t=50, position_ratio = +0.7, pnl = +0.03
DT 推理:               t=50, position_ratio = -0.3, pnl = -0.01
→ DT 從未在訓練中見過這種 state 組合 → 行為不可預測
```

### 解決方案

**方案 A（推薦）：移除高度冗餘特徵，保留核心倉位特徵**

由於環境存在硬性風險控制覆寫機制（如止損與回撤鎖定），實際倉位 `position_ratio` 並不總是等於上一步的 action $a_{t-1}$。如果直接移除，Agent 在被強制平倉後將無法感知自身已是空倉，因此**必須保留 `position_ratio`**。

我們僅移除完全可由動作歷史推斷或高度冗餘的特徵：

| Feature | 動作 | 理由 |
|---------|------|------|
| `position_ratio` | ✅ 保留 | **關鍵**：防止止損/回撤鎖定覆寫 action 後，模型產生資訊落差 |
| `holding_norm` | ❌ 移除 | 可從連續同向 action 推斷 |
| `unrealized_pnl` | ✅ 保留 | 需要 entry_price 和當前價格，action 不夠推斷 |
| `drawdown` | ✅ 保留 | 需要 equity curve 計算 |
| `volatility` | ✅ 保留 | 外生，純市場資訊 |
| `distance_to_stop` | ❌ 移除 | 冗餘特徵，可由 stop_flag 代替 |

**方案 B：多策略資料收集**

用多個不同品質 / 風格的 policy 收集 dataset，增加 portfolio state space 的覆蓋率。詳見 Issue #5。

---

## 6. Issue #4：_resolve_target 轉換層位置

### 問題描述

`_resolve_target()` 在 raw action 和實際執行的倉位之間做了多層轉換：

```
raw_action ∈ [-1, 1]
  ├─ trend prior (residual policy)      ← use_trend_prior
  ├─ volatility scaling                 ← 永遠啟用
  ├─ regime caps                        ← regime_risk_scaling
  └─ turbulence gate                    ← use_turbulence_gate
  → actual target_position
```

**DT 應該預測哪一個？**

### 方案比較

| | 預測 raw action | 預測 resolved target |
|---|---|---|
| **行為** | DT 輸出 → `_resolve_target()` → 實際倉位 | DT 直接輸出實際倉位 |
| **安全機制** | ✅ vol scaling, regime caps, turbulence gate 保留 | ❌ 全部繞過 |
| **訓練資料** | 記錄 behavior policy 的 raw action | 記錄 resolved target |
| **action 範圍** | 固定在 [-1, 1]，好 normalize | 可能被 vol scaling 放大，範圍不固定 |
| **推薦** | ✅ **推薦** | ⚠️ 危險 |

### 解決方案

**讓 DT 預測 raw action，保留 `_resolve_target()` 轉換層。** 收集訓練資料時記錄 raw action（即 behavior policy 的直接輸出），不是被轉換後的 target_position。

---

## 7. Issue #5：離線資料集品質與收集

### 問題描述

DT 是 offline RL——不跟環境互動，只從固定 dataset 學習。**DT 的表現上限受限於 dataset 中最好的 trajectory。**

如果 dataset 中最好的 episode Sharpe = 1.2，那 DT 設 RTG 再高也很難超過 1.2。這與 PPO/SAC 的 online exploration 根本不同。

### 資料收集策略

```
┌─────────────────────────────────────────────────────┐
│                  資料收集 Pipeline                    │
│                                                     │
│  1. 訓練多組 SAC/PPO（不同超參數）                     │
│     ├─ SAC (default)                                │
│     ├─ SAC (high sharpe_coeff)                      │
│     ├─ PPO (default)                                │
│     └─ PPO (high drawdown_coeff)                    │
│                                                     │
│  2. 每個 policy 跑 N 個 episodes，記錄完整 trajectory  │
│     ├─ state (flat, single-step)                    │
│     ├─ raw_action                                   │
│     ├─ executed_action (resolved target)            │
│     ├─ net_return                                   │
│     ├─ stop_triggered (bool)                        │
│     ├─ dd_locked (bool)                             │
│     └─ episode_sharpe (最終績效標記)                  │
│                                                     │
│  3. 混入 random policy trajectories                  │
│     └─ 增加 state space coverage                     │
│                                                     │
│  4. 計算 RTG = cumulative net_return from t to T     │
│                                                     │
│  5. 組裝成 DT 訓練格式                               │
│     └─ sequences of (RTG, state, action)            │
└─────────────────────────────────────────────────────┘
```

### 資料量建議

| 資料來源 | Episodes | 目的 |
|----------|----------|------|
| SAC (多組超參數) | 各 50~100 | 高品質 trajectory |
| PPO (多組超參數) | 各 50~100 | 不同風格的策略 |
| Random policy | 50~100 | State space 覆蓋 |
| 規則策略 (trend-following) | 50~100 | Baseline trajectory |
| **合計** | 400~800 | — |

### 注意事項

- **不要只用最好的 trajectory**：DT 需要看到好壞對比，才能學會「高 RTG 對應好策略」
- **標記品質**：每個 episode 附上最終 Sharpe / 總報酬，方便後續分析
- **避免 data leakage**：收集資料用 training period，DT 的 evaluation 用 test period

---

## 8. Issue #6：有狀態元件的重播問題

### 問題描述

環境中有多個有狀態元件，在 DT 離線場景下需要特殊處理：

| 元件 | 狀態變數 | 問題 |
|------|----------|------|
| `IncrementalSharpe` | running `a`, `b` | DSR 值依賴 episode 歷史順序，不能亂序重播 |
| `_stopped_out` | `bool` | 隱藏狀態，DT 的 state 中看不到 |
| `_dd_locked` | `bool` | 隱藏狀態，會強制覆寫 agent 的 action |
| `_entry_price` | `float` | Path-dependent，影響 unrealized PnL 計算 |

### Action 被覆寫的問題

```python
# envs_v2.py step() 中：
if stop_triggered:
    target_position = 0.0    # ← agent 輸出 0.8，被強制改成 0.0

if self._dd_locked and abs(target_position) > abs(prev_position):
    target_position = prev_position * 0.5  # ← 被強制縮減
```

訓練資料中的 (state, action, reward) 可能出現不一致：
- 如果記錄 **被覆寫後的 action**：DT 學到「在這種 state 下應該 flatten」，但不知道為什麼
- 如果記錄 **原始 action**：reward 和 next_state 是根據被覆寫的 action 計算的，action 和結果對不上

### 解決方案

**在訓練資料中同時記錄兩個 action，並將隱藏狀態暴露為 state features：**

```python
# 每一步記錄的完整資訊
trajectory_step = {
    # State（DT 用）
    "state": flat_observation,           # (n_features,)
    
    # Action（DT 訓練用 raw_action）
    "raw_action": raw_action,            # agent 原始輸出
    "executed_action": target_position,  # 環境實際執行的
    
    # 隱藏狀態暴露為額外 state features
    "stop_triggered": bool,              # 加入 state
    "dd_locked": bool,                   # 加入 state
    
    # Reward / RTG
    "net_return": net_return,            # 用於計算 RTG
    "reward": shaped_reward,             # 原始 reward（記錄但不用於 RTG）
}
```

**具體做法**：
1. DT 訓練時用 `raw_action` 作為 action label
2. 將 `stop_triggered` 和 `dd_locked` 作為 **binary features 加入 state vector**
3. RTG 使用 `net_return` 的累積和（不用 shaped reward）
4. 這樣 DT 可以學到：「當 `dd_locked=1` 時，即使我想加倉，環境也會限制我」

---

## 9. Issue #7：Context Length K 的選擇

### PPO/SAC 的 window vs DT 的 K

| | Env window (`window_size`) | DT context (`K`) |
|---|---|---|
| **看到什麼** | 過去 N 天的技術指標 | 過去 K 步的 (RTG, state, action) |
| **學到什麼** | 指標的時間 pattern | 策略的 temporal dependency |
| **典型值** | 20~60 | 10~30 |
| **計算成本** | 線性 O(N × features) | Quadratic O(3K)² for attention |

### K 的選擇考量

| K 值 | 優點 | 缺點 |
|------|------|------|
| K < 10 | 計算快 | 看不到足夠的 action-reward pattern |
| K = 20 | DT 原論文常用值；覆蓋約一個月交易日 | 平衡點 |
| K = 30~50 | 覆蓋更長趨勢 | 計算量增加；金融序列的有效 dependency 通常 < 30 天 |
| K > 50 | — | 計算量 O(K²) 過大；可能 overfit |

### 建議

- **起始值：K = 20**
- 用 validation set 做 ablation study（K ∈ {10, 15, 20, 30}）
- 注意：K 需要小於 episode 長度（你的訓練資料約 1969 步，所以 K 有很大空間）

---

## 10. 遷移 Checklist

| # | Issue | 嚴重度 | 動作 | 狀態 |
|---|-------|--------|------|------|
| 1 | 雙重 sliding window | 🔴 Critical | State 切成 single-step `(n_features,)` | ☐ 未開始 |
| 2 | DSR 不適合做 RTG | 🔴 Critical | 改用累積 net return 作為 RTG | ☐ 未開始 |
| 3 | Portfolio features 分佈漂移 | 🟡 Medium | 保留 `position_ratio`，移除 `holding_norm` 與 `distance_to_stop`；加入 stop/dd flags | ☐ 未開始 |
| 4 | `_resolve_target` 位置 | 🟡 Medium | DT 預測 raw action，保留轉換層 | ☐ 未開始 |
| 5 | 離線資料集品質 | 🔴 Critical | 多策略收集 + 品質標記；建立資料收集 pipeline | ☐ 未開始 |
| 6 | 有狀態元件覆寫 | 🟡 Medium | 記錄 raw + executed action；暴露 `stop_triggered` / `dd_locked` 為 state features | ☐ 未開始 |
| 7 | Context length K | 🟢 Low | 從 K=20 開始，validation 調整 | ☐ 未開始 |

### 建議實施順序

```
Phase 1: 環境改造
├─ [1] 加入 flat_obs mode（single-step state）
├─ [6] 暴露 stop_triggered / dd_locked 為 state features
└─ [3] 調整 portfolio features（移除內生 features）

Phase 2: 資料收集
├─ [5] 建立多策略資料收集 pipeline
├─ [4] 確保記錄 raw_action
└─ [2] 計算累積 net return 作為 RTG

Phase 3: DT 模型
├─ [7] 設定 K=20，建立 DT 模型
├─ 訓練 + validation
└─ 與 PPO/SAC baseline 比較
```

---

## 11. 附錄：程式碼修改範例

### A. 為 PositionMgmtEnv 加入 flat_obs 模式

```python
# config 新增
@dataclass(frozen=True)
class RiskConfig:
    ...
    flat_obs: bool = False  # True → single-step state for DT

# __init__ 修改 observation_space
if self.risk.flat_obs:
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(n_features,),
        dtype=np.float32,
    )
else:
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(window_size, n_features),
        dtype=np.float32,
    )

# _get_observation 修改
def _get_observation(self) -> np.ndarray:
    if self.risk.flat_obs:
        tech = self.signal_features[self._current_tick]     # (n_tech,)
        portfolio = self._get_portfolio_features()           # (n_portfolio,)
        return np.concatenate([tech, portfolio]).astype(np.float32)
    else:
        # 原有 sliding window 邏輯
        end = self._current_tick
        start = end - self.window_size
        tech = self.signal_features[start:end]
        portfolio_row = self._get_portfolio_features()
        portfolio_block = np.tile(portfolio_row, (self.window_size, 1))
        return np.concatenate([tech, portfolio_block], axis=1).astype(np.float32)
```

### B. 資料收集腳本骨架

```python
def collect_trajectories(env, policy, n_episodes=100):
    """收集 DT 訓練用的 trajectory 資料。"""
    dataset = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        trajectory = []
        done = False
        
        while not done:
            raw_action = policy.predict(obs)  # agent 原始輸出
            obs_next, reward, terminated, truncated, info = env.step(raw_action)
            
            trajectory.append({
                "state": obs,                                    # flat (n_features,)
                "raw_action": float(raw_action),                 # 原始 action
                "executed_action": info["position_ratio"],       # 實際倉位
                "net_return": info["returns"][-1] if info["returns"] else 0.0,
                "stop_triggered": info.get("stop_triggered", False),
                "dd_locked": info.get("dd_locked", False),
            })
            
            obs = obs_next
            done = terminated or truncated
        
        # 計算 RTG (cumulative net return from t to T)
        net_returns = [step["net_return"] for step in trajectory]
        rtgs = np.cumsum(net_returns[::-1])[::-1].tolist()
        for i, step in enumerate(trajectory):
            step["rtg"] = rtgs[i]
        
        # 標記 episode 品質
        episode_sharpe = compute_sharpe_ratio(np.array(net_returns))
        for step in trajectory:
            step["episode_sharpe"] = episode_sharpe
        
        dataset.extend(trajectory)
    
    return dataset
```

### C. DT 用的 Portfolio Features 調整

```python
def _get_portfolio_features_for_dt(self) -> np.ndarray:
    """DT 專用：移除高度冗餘特徵，保留核心倉位與隱藏狀態 flags。"""
    current_price = float(self.prices[self._current_tick])
    
    # 保留：需要價格計算、action history 不夠推斷的
    unrealized_pnl = 0.0
    if abs(self._position_ratio) > EPS and self._entry_price > EPS:
        unrealized_pnl = (
            (current_price - self._entry_price)
            / self._entry_price
            * np.sign(self._position_ratio)
        )
    
    drawdown = (self._equity - self._peak_equity) / max(self._peak_equity, EPS)
    atr_now = float(self.atr[self._current_tick])
    volatility = atr_now / max(current_price, EPS)
    
    # 新增：暴露隱藏狀態
    stop_flag = 1.0 if self._stopped_out else 0.0
    dd_lock_flag = 1.0 if self._dd_locked else 0.0
    
    return np.array([
        self._position_ratio,
        np.clip(unrealized_pnl, -1.0, 1.0),
        np.clip(drawdown, -1.0, 1.0),
        float(volatility),
        stop_flag,
        dd_lock_flag,
    ], dtype=np.float32)
```

---

> **附註**：本文件為設計階段文件，實際實施時可能需要根據實驗結果調整。建議每完成一個 Phase 後做一次中間驗證，確認方向正確再進入下一個 Phase。
