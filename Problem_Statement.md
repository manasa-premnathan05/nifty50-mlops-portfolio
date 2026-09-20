# Experiment 1: Case Study Framing & Dataset Preparation
## Problem Statement & Data Versioning Architecture

**Domain:** Financial Machine Learning / Quantitative Equity Modeling  
**Target Asset Universe:** NIFTY 50 Benchmark Constituents (National Stock Exchange of India)  
**Author:** Manasa Premnathan  

---

### 1. Domain Problem Statement
Financial market volatility and non-stationary price dynamics present significant challenges for institutional risk management, portfolio rebalancing, and tactical capital allocation. Traditional econometric time series models (e.g., ARIMA, GARCH) struggle to capture non-linear interactions across technical indicators, multi-day momentum shifts, and cross-asset volatility.

The objective of this project is to develop an automated, reproducible machine learning classification pipeline to predict next-day equity price movement direction ($Direction \in \{1: \text{UP}, 0: \text{DOWN}\}$) across the NIFTY 50 constituent equities, evaluated over chronological trading sessions from 2018 through 2021.

---

### 2. Success Metrics & Benchmarks
- **Directional Accuracy Target:** $\ge 52.0\%$ (In efficient financial markets, a directional edge of 52%–54% delivers substantial positive Sharpe ratios after transaction cost modeling).
- **ROC-AUC Target:** $\ge 0.540$
- **High-Risk Segment Recall:** $\ge 54.0\%$
- **Disparity Ratio Ceiling:** Algorithmic fairness disparity across asset risk tiers must remain below $5\%$ post-mitigation.

---

### 3. Dataset Description & Acquisition
- **Acquisition Source:** Historical secondary market end-of-day equity quotes gathered via financial market APIs and exchange records (NSE).
- **Raw Merged Dataset (`raw_merged_dataset.csv`):** 60,550 historical observations with uncleaned dates, missing values, duplicates, and non-aligned trading calendars.
- **Cleaned Dataset (`final_nifty50_dataset.csv`):** 29,310 curated, chronological trading records across 42 constituent stocks with 16 continuous technical indicators (`Open`, `High`, `Low`, `Close`, `AdjClose`, `Volume`, `Daily_Return`, `MA_20`, `MA_50`, `Volatility_20D`, `RSI`, `MACD`, `Return_Lag_1`, `Return_Lag_2`, `Return_Lag_5`, `Volume_Change`).

---

### 4. Data Version Control (DVC) Plan
Large datasets and binary model weights should not be stored directly in Git repositories to prevent repository bloat and bandwidth bottlenecks.
- **Version Control Engine:** Data Version Control (DVC) tracks large data files (`raw_merged_dataset.csv`) using lightweight metadata pointer files (`.dvc`).
- **Storage Strategy:** Git tracks code, notebooks, and `.dvc` pointers, while DVC manages the underlying raw dataset in external storage.
