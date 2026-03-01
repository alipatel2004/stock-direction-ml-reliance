# 📈 Stock Market Direction Prediction — Reliance Industries (NSE)

> **Finance Domain · Machine Learning Project**
> Binary classification model predicting whether Reliance Industries' closing price will go **UP or DOWN** the next trading day.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![yfinance](https://img.shields.io/badge/Data-yfinance%20live-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)
![Domain](https://img.shields.io/badge/Domain-Finance-gold)

---

## 🗂️ Project Structure

```
stock-ml-reliance/
├── ML_Project.ipynb                  # Main Jupyter Notebook (full pipeline)
├── README.md                         # This file
└── Reliance_Stock_ML_Report.pdf      # Conclusion & Project Report
```

---

## 📌 Project Overview

| Field | Details |
|---|---|
| **Domain** | Finance — Stock Market Price Direction Prediction |
| **Dataset** | Yahoo Finance via `yfinance` (live, no manual download needed) |
| **Ticker** | `RELIANCE.NS` — Reliance Industries Ltd., NSE India |
| **Date Range** | January 1 2015 → December 31 2024 (~2,500 trading days) |
| **Task** | Binary Classification: `1` = next-day price UP · `0` = DOWN or FLAT |
| **Train/Test** | 80/20 chronological split (time-series aware, no look-ahead bias) |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| `yfinance` | Download live OHLCV stock data from Yahoo Finance |
| `pandas` / `numpy` | Data wrangling and feature engineering |
| `matplotlib` / `seaborn` | EDA visualisations |
| `scikit-learn` | Logistic Regression, Random Forest, SVM classifiers |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/stock-ml-reliance.git
cd stock-ml-reliance
```

### 2. Install dependencies
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the notebook
```bash
jupyter notebook ML_Project.ipynb
```

> **No dataset file needed** — data downloads automatically from Yahoo Finance when you run the first cell.

---

## 📊 Pipeline

### Step 1 — Data Collection
```python
import yfinance as yf
df = yf.download("RELIANCE.NS", start="2015-01-01", end="2024-12-31")
```

### Step 2 — Feature Engineering

| Feature | Calculation | Purpose |
|---|---|---|
| `Return` | `Close.pct_change()` | Daily % price change |
| `SMA_10` | `Close.rolling(10).mean()` | Short-term trend (2-week) |
| `SMA_50` | `Close.rolling(50).mean()` | Medium-term trend (10-week) |
| `Volatility` | `Return.rolling(10).std()` | 10-day realised volatility |
| `RSI` | 14-period Wilder RSI | Momentum oscillator (0–100) |

### Step 3 — Target Variable
```python
df['Target'] = (df['Return'].shift(-1) > 0).astype(int)  # 1=UP, 0=DOWN
df.dropna(inplace=True)   # removes NaN rows from rolling features
```

### Step 4 — EDA
- **Returns histogram** — near-normal distribution centred at zero, slight positive skew
- **Correlation heatmap** — SMA features correlated with price; RSI and Volatility are stationary

### Step 5 — Train / Test Split (80/20 chronological)
```python
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

X_train = train.drop('Target', axis=1)
y_train = train['Target']
X_test  = test.drop('Target', axis=1)
y_test  = test['Target']

# Secondary NaN/Inf clean-up (83 residual cells found)
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]
X_test  = X_test.replace([np.inf, -np.inf], np.nan).dropna()
y_test  = y_test.loc[X_test.index]
```

### Step 6 — Model Training & Results

| Model | Config | Test Accuracy |
|---|---|---|
| Logistic Regression | `max_iter=1000` | 50.62% |
| Random Forest | `n_estimators=200` | 50.21% |
| **SVM (Best)** | `SVC()` RBF kernel | **51.03%** |

### Step 7 — Random Forest Detailed Evaluation
```
Confusion Matrix:
[[180  58]    ← 180 correct DOWN predictions,  58 false UP signals
 [183  63]]   ← 183 missed UP days,            63 correct UP predictions

Classification Report:
              precision  recall  f1-score  support
0  (DOWN)       0.50      0.76     0.60      238
1  (UP)         0.52      0.26     0.34      246
accuracy                           0.50      484
macro avg       0.51      0.51     0.47      484
```

### Step 8 — Backtesting Strategy
```python
# Long-only: buy when model predicts UP, else hold cash
test['Strategy_Return']   = y_pred_rf * test['Return']
test['Cumulative_Strategy'] = (1 + test['Strategy_Return']).cumprod()
test['Cumulative_Market']   = (1 + test['Return']).cumprod()
test[['Cumulative_Strategy', 'Cumulative_Market']].plot()
```

---

## 📉 Results Summary

- **Best model:** SVM at **51.03%** test accuracy (484 test samples, ~2023–2024)
- All three models fall in the **50–51% accuracy band** — consistent with semi-efficient markets
- RF achieves **76% recall on DOWN days** but only **26% on UP days** (threshold tuning can rebalance)
- Backtesting shows the strategy **reduces drawdown** by avoiding predicted-DOWN days

> ℹ️ **Why ~50% is OK:** Daily stock direction is near a coin flip for a single equity on price features alone. Even a 1–5% edge, applied consistently with proper risk management, can compound into meaningful alpha over time.

---

## 🔑 Key Implementation Decisions

| Decision | Reason |
|---|---|
| Chronological 80/20 split | Prevents look-ahead bias — essential for time-series ML |
| `df.dropna()` after feature engineering | SMA-50 and RSI leave NaN in first ~50 rows |
| Secondary NaN/Inf clean-up | 83 residual NaN cells remained in X_train after initial dropna |
| No SMOTE required | Class split is ~49% UP / 51% DOWN — already near-balanced |
| `max_iter=1000` for LR | Default 100 iterations insufficient for convergence on this data |

---

## 📈 Potential Improvements

- Add MACD, Bollinger Bands, VWAP, OBV as additional features
- Use `TimeSeriesSplit` for k-fold cross-validation
- Try **XGBoost** or **LightGBM** — typically outperform RF on tabular financial data
- Explore **LSTM** / Transformer for sequence-based temporal modelling
- Lower decision threshold to ~0.45 to improve UP-day recall
- Add **transaction costs** (0.1–0.2% per round-trip) to backtest
- Extend to **Nifty 50** stocks for cross-sectional ranking strategies
- Include **Sharpe Ratio, Max Drawdown, Calmar Ratio** in backtest metrics

---

## 🌍 Real-World Applications

- Algorithmic trading engines on NSE/BSE
- Robo-advisory platforms with automated rebalancing signals
- Quantitative hedge funds using ML factor models for Indian equities
- Retail investor apps with predicted UP-day push notifications
- Risk dashboards flagging high-volatility market regimes

---


---

