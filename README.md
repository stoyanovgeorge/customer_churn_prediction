# 📊 Customer Churn Prediction with MLflow, Docker, Terraform & FastAPI

This project tackles customer churn prediction using a **Random Forest classifier** on a real-world telecom dataset from [Maven Analytics on Kaggle](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics).

---

## 🚀 Project Overview

### ✅ Core Objectives
- Train models to predict customer churn.
- Compare results across two dataset sizes using **MLflow**.
- Package the best model into a **Dockerized FastAPI app**.
- Manage infrastructure with **Terraform** (local Docker).
- Optionally orchestrate training with **Airflow**.
- Basic monitoring of the deployed service.

---

## 📂 Dataset

- **Source**: [Telecom Customer Churn](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics)
- **Format**: Tabular CSV
- **Target variable**: `Churn`

We prepare two versions of the dataset:
- `data_sample_1000.csv` – a sample of 1,000 rows
- `data_full.csv` – the full dataset

---

## 🧪 ML Experiment Tracking with MLflow

### Run MLflow Server Locally

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```