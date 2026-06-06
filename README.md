# 📊 User Churn Prediction System
> End-to-end big data pipeline for telecom customer churn prediction  
> 电信用户流失预测 · 大数据全栈项目 · 应用统计学 × 数据工程

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange?logo=apachespark)
![HBase](https://img.shields.io/badge/HBase-2.x-red)
![HDFS](https://img.shields.io/badge/HDFS-3.2.1-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

---

## 🎯 Project Overview

This project builds a **production-style churn prediction system** combining applied statistics with big data engineering. The pipeline spans data ingestion, distributed feature engineering, multi-model training, online serving, and an interactive dashboard — all containerized with Docker.

**Key Results:**

| Model | AUC | F1 |
|---|---|---|
| Logistic Regression | 0.8291 | **0.7985** |
| Random Forest | **0.8300** | 0.7798 |
| GBT | 0.8274 | 0.7932 |

> All three models achieve AUC ≈ 0.83, demonstrating robust feature engineering. Logistic Regression achieves the highest F1, suggesting the churn signal is largely linear — a key statistical insight.

---

## 🏗️ Architecture

```
Data Source (Kaggle CSV)
        │
        ▼
┌─────────────────────────────────┐
│         HDFS Storage            │
│  /raw  /cleaned  /features      │
│  /predictions  (Parquet)        │
└──────────────┬──────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌─────────────┐
│ Spark ETL  │   │    Hive     │
│ Feature    │   │  Data       │
│ Engineering│   │  Warehouse  │
└──────┬─────┘   └──────┬──────┘
       └────────┬────────┘
                ▼
┌───────────────────────────────┐
│     Statistical Modeling      │
│  Logistic Regression (LR)     │
│  Random Forest (RF)           │
│  Gradient Boosted Trees (GBT) │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│     HBase Online Serving      │
│  RowKey: user_XXXXXX          │
│  Columns: prob, risk, model   │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│    Streamlit Dashboard        │
│  Overview / Models / Features │
│  / Insights (4 tabs)          │
└───────────────────────────────┘
```

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Containerization | Docker Compose | One-command cluster setup |
| Distributed Storage | HDFS 3.2.1 | Raw + feature + prediction data |
| Batch Processing | Apache Spark 3.5.0 | Feature engineering, ML training |
| Data Warehouse | Apache Hive | ODS / DWD / DWS layered modeling |
| Online Serving | Apache HBase | Low-latency prediction lookup |
| Modeling | Spark MLlib | LR, Random Forest, GBT |
| Dashboard | Streamlit | Interactive visualization |
| Language | Python 3.11 | PySpark, happybase, pandas, matplotlib |

---

## 📁 Project Structure

```
churn-prediction/
├── docker/
│   ├── docker-compose.yml      # Full cluster definition
│   └── hadoop.env              # HDFS configuration
├── data/
│   ├── raw/                    # Original Kaggle CSV
│   ├── cleaned/                # After Spark cleaning
│   ├── features/               # Feature tables (Parquet)
│   └── predictions/            # Model output
├── notebooks/
│   └── churn_prediction_full.ipynb   # Full pipeline notebook
├── spark_jobs/                 # Standalone Spark scripts
├── modeling/                   # Saved model artifacts
├── app/
│   └── dashboard.py            # Streamlit dashboard
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (WSL2 backend, 20GB RAM allocated)
- Windows 10/11 or macOS/Linux

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction
```

### 2. Download dataset
Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the CSV in `data/raw/`.

### 3. Start the cluster
```bash
cd docker
docker compose up -d
```

Wait ~30 seconds, then verify services:

| Service | URL |
|---|---|
| HDFS NameNode UI | http://localhost:9870 |
| Spark Master UI  | http://localhost:8080 |
| HBase Master UI  | http://localhost:16010 |
| JupyterLab       | http://localhost:8888 |

### 4. Upload data to HDFS
```bash
docker exec namenode hdfs dfs -mkdir -p /churn/raw
docker exec namenode hdfs dfs -put \
  /data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv /churn/raw/
```

### 5. Run the notebook
Open JupyterLab at `http://localhost:8888`, navigate to `work/churn_prediction_full.ipynb`, and run all cells.

### 6. Launch dashboard
```bash
streamlit run app/dashboard.py
```
Open `http://localhost:8501`

---

## 📊 Dataset

**Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Watson)

| Property | Value |
|---|---|
| Rows | 7,043 |
| Features | 21 (raw) → 45 (after OHE) |
| Target | `Churn` (Yes/No) |
| Class ratio | 73.5% No Churn / 26.5% Churn |
| Split | 80% train / 20% test (seed=42) |

**Key features:**
- `tenure` — months the customer has stayed
- `Contract` — Month-to-month / One year / Two year
- `MonthlyCharges`, `TotalCharges` — billing information
- `InternetService`, `OnlineSecurity`, `TechSupport` — service subscriptions

---

## 🔬 Statistical Methodology

### 1. Data Cleaning
- `TotalCharges` contained whitespace strings — cast to `DoubleType`, imputed as `MonthlyCharges × tenure`
- Target variable `Churn` encoded as binary label (1 = churned)
- No rows dropped; all 7,043 records retained

### 2. Feature Engineering (Spark MLlib Pipeline)
```
StringIndexer → OneHotEncoder → VectorAssembler
```
- 15 categorical features encoded via OHE → 41 binary dimensions
- 4 numerical features appended directly
- Final feature vector: **45 dimensions**

### 3. Class Imbalance
- Positive class (churn) = 26.5% — moderate imbalance
- Evaluation uses **AUC-ROC** and **F1** rather than accuracy
- A naive "predict No Churn" baseline achieves 73.5% accuracy but AUC = 0.5

### 4. Model Training & Evaluation
All models trained with `seed=42` for reproducibility:

**Logistic Regression** — interpretable baseline; coefficients reveal odds ratios per feature.

**Random Forest** — 100 trees; feature importance via Gini impurity decrease. `tenure` (0.1591) and `Contract_0` (0.1545) dominate.

**GBT (Gradient Boosted Trees)** — 50 iterations; strongest on recall. Used for HBase prediction storage.

### 5. HBase Schema Design
```
Table: churn_predictions
RowKey: user_XXXXXX  (zero-padded index, lexicographic ordering)

Column Family: cf
  cf:actual_label   — ground truth (0/1)
  cf:prediction     — model prediction (0/1)
  cf:churn_prob     — probability of churn (0.0000–1.0000)
  cf:risk_level     — HIGH / MEDIUM / LOW
  cf:model          — model used (GBT)
```

---

## 📈 Key Findings

**① Contract type is the strongest business predictor**  
Month-to-month customers churn at ~42%, vs ~11% (one year) and ~3% (two year). Migrating customers to annual contracts is the highest-leverage retention action.

**② Tenure drives early churn**  
Churn is heavily concentrated in the first 1–6 months. Early onboarding programs targeting new users can significantly reduce lifetime churn.

**③ Value-added services reduce churn by ~2×**  
Customers without OnlineSecurity or TechSupport churn at nearly double the rate. Bundling increases customer stickiness.

**④ Logistic Regression matches complex models**  
Despite being the simplest model, LR achieves the highest F1 (0.7985). This suggests the churn signal is largely linear — adding model complexity yields diminishing returns, a classic bias-variance tradeoff insight.

**⑤ AUC ≈ 0.83 across all models**  
The consistency indicates the feature engineering quality is the primary driver of predictive performance, not model choice.

---

## 🖥️ Dashboard Preview

The Streamlit dashboard includes 4 interactive tabs:

- **Overview** — Churn distribution, contract vs churn, tenure histogram
- **Model Comparison** — AUC/F1/Precision/Recall bar charts with model selector and AUC threshold filter
- **Feature Importance** — Color-coded by feature category (Numerical / Contract / Service / Payment) with category filter
- **Insights** — Statistical findings with risk level breakdown from HBase

---

## 📝 Resume Description (EN)

> Built an end-to-end user churn prediction system on 7,043 Telco records using a Dockerized big data stack (HDFS, Spark, HBase). Engineered 45-dimensional feature vectors via Spark MLlib Pipeline (StringIndexer, OHE, VectorAssembler). Compared Logistic Regression, Random Forest, and GBT; all achieved AUC ≈ 0.83, with LR yielding the best F1 (0.7985) — interpreted via the bias-variance tradeoff. Stored predictions in HBase (RowKey design for sequential scan efficiency) and deployed an interactive Streamlit dashboard with model comparison, feature importance, and statistical insights.

## 📝 简历描述（中文）

> 基于 7,043 条电信用户数据，构建端到端用户流失预测系统（Docker 容器化，HDFS + Spark + HBase 全栈）。使用 Spark MLlib Pipeline 构建 45 维特征向量（StringIndexer + OHE + VectorAssembler）；对比逻辑回归、随机森林、GBT 三种模型，AUC 均达 0.83，逻辑回归 F1 最高（0.7985），体现偏差-方差权衡分析能力。预测结果按 RowKey 设计写入 HBase，Streamlit 部署四标签交互式 Dashboard，展示模型对比、特征重要性与统计洞察。

---

## 📜 License

MIT License — feel free to use this project as a portfolio template.

---

*Built with ❤️ as a portfolio project combining Applied Statistics and Big Data Engineering.*
