# 📊 Customer Churn Prediction with MLflow, Airflow, Docker, Terraform & FastAPI

A robust end‑to‑end pipeline for **customer churn prediction** using a **Random Forest classifier** on real telecom data from Maven Analytics (Kaggle).

---

## 🚀 Project Overview

### 🎯 Objectives
- Train and compare churn classifiers on two dataset sizes (sample vs full) using **MLflow**.
- Package the best-performing model into a **Dockerized FastAPI** service.
- Orchestrate training jobs using **Airflow**.
- Provision infrastructure (MLflow tracking server, local Docker registry) with **Terraform**.
- Enable basic monitoring for the deployed endpoint.

---

## 📂 Dataset

- **Source**: [Maven Analytics – Telecom Customer Churn (CSV)](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data)
- **Target**: `Customer Status`  
- **Versions**:
  - `data/data_sample_1000.csv` – 1,000‑row sample  
  - `data/data_full.csv` – full dataset  

---

## 🧩 Project Structure

| File                                | Description |
|-------------------------------------|-------------|
| `.env.template`                     | Environment file for Docker Compose specifying Airflow configuration parameters. |
| `airflow-compose.yaml`              | Simple Docker compose file to spin up Airflow with `LocalExecutor`. |
| `app/main.py`                       | FastAPI app serving `/predict`, `/health`, `/logs`. |
| `fastapi_requirements.txt`          | Dependencies for the FastAPI container. |
| `churn_prediction.ipynb`            | Exploratory analysis, visualization, RF + GridSearchCV. |
| `churn_prediction_mlflow.ipynb`     | Skips exploration: focuses on MLflow logging. |
| `mlflow_logging.py`                 | CLI script: train RF, log model + metrics + artifacts (confusion matrix, feature importances). |
| `download_best_model.py`            | Fetches and downloads the top MLflow run model. |
| `download_dataset.py`               | Downloads sample/full datasets. |
| `random_forest_automation.sh`       | Batch runner of `mlflow_logging.py` with differing hyperparameters. |
| `Dockerfile`                        | Builds the FastAPI service on Python 3.13.5‑slim. |
| `build_docker_image.sh`             | Automates Docker build/tag/push. |
| `docker-compose.yaml`               | Spins up Airflow (LocalExecutor) + FastAPI service. |
| `terraform/variables.tf`            | Terraform input variable definitions. |
| `terraform/dev.tfvars`              | Local override values for variables.tf. |
| `terraform/main.tf`                 | Provisions Docker network, registry, MLflow server. |
| `.env.template`                     | Template for Docker Compose environment variables. |

---

## 🔧 Infrastructure (Terraform)

### Prerequisites
- Install [Terraform](https://developer.hashicorp.com/terraform/install).

### Setup
1. Edit `terraform/dev.tfvars`, replacing placeholders:
   - `mlflow_image`, `external_mlflow_port`, `docker_network_name`, `uid`, `gid`, etc.
2. Deploy:
   ```bash
   terraform plan --var-file="dev.tfvars"
   terraform apply --var-file="dev.tfvars"
  ```
This sets up a Docker network, registry (e.g. at 127.0.0.1:5000), and MLflow tracking UI (e.g. at 127.0.0.1:7000).

---

## 🧪 MLflow Tracking
Visit MLflow UI at: `http://localhost:<external_mlflow_port>`

Use `mlflow_logging.py` to log RF runs:
```bash
python mlflow_logging.py \
  -D data/data_full.csv \
  --n_estimators 50 \
  --max_depth 10 \
  --min_ssplit 2 \
  --mlflow_uri http://localhost:<external_mlflow_port>
```
* The script creates a new Experiment named: `RF Churn Prediction` and every execution will create a new Run, that will be of the format: `Full DF | NE=200 | MD=10 | MSS=2`. This translates to:

  * Dataset: Full dataset (`data_full.csv`) used
  * NE: 200 Random Forest estimators (`n_estimators=200`)
  * MD: Maximum tree depth set to 10 (`max_depth=10`)
  * MSS: Minimum samples required to split an internal node is 2 (`min_samples_split=2`)

* In addition on every run it logs: 
  * The trained model itself
  * A feature importance bar chart (vertical layout)
  * A confusion matrix
  * Model parameters
  * Performance metrics (e.g., accuracy)
* You can also run:
```bash
./random_forest_automation.sh
```
to generate a grid of runs. 
* To retrieve the best model:
  * Use MLflow UI to manually download `pipeline.pkl` from the Runs' Artifacts, or
  * Run:
  ```bash
  python download_best_model.py --mlflow_uri http://localhost:<port>
  ```
  which fetches the model with the highest training accuracy into `app/pipeline.pkl`.

---

## 🐳 Dockerize & Deploy FastAPI Prediction App
### Build & Push the Model Server
```bash
docker build -t rf-api:1.0.0 .
docker tag rf-api:1.0.0 127.0.0.1:5000/rf-api:1.0.0
docker push 127.0.0.1:5000/rf-api:1.0.0
```
Or you can use:
```bash
./build_docker_image.sh 1.0.0
```
* Dockerfile:
  * Based on `python:3.13.5-slim`
  * Installs dependencies, exposes port 8080.
  * Runs `uvicorn main:app --host 0.0.0.0 --port 8080`
* Confirm local registry content:
```bash
curl http://127.0.0.1:5000/v2/_catalog
curl http://127.0.0.1:5000/v2/rf-api/tags/list
```
### Run the Container
```bash
docker run -d --name rf-api \
  -p 8080:8080 --restart unless-stopped \
  127.0.0.1:5000/rf-api:1.0.0
```
* Alternatively, you can also use the provided `docker-compose.yaml` file to spin up the container. 
```bash
docker compose up -d
```
> [!TIP]
> The version tab on your build and in the `docker-compose.yaml` file must match. 
Check:
* If the container is running with:
  ```bash
  docker ps -a
  ```
* The log files:
  ```bash
  docker logs rf-api
  ```
* Access the shell:
  ```bash
  docker exec -it rf-api /bin/sh
  ```

---

## 🔌 API Endpoints

* `GET /api/health` - Checks if service is running and model is loaded.
* `GET /api/logs?limit=50` - Returns up to limit recent log entries (default: 100).
* `POST /api/predictions` - Sends JSON (single object or array), returns:
```json
{
  "prediction": 0,
  "label": "Churned",
  "probabilities": [
    0.6516209691830841,
    0.07573502309365371,
    0.27264400772326197
  ],
  "available_classes": {
    "0": "Churned",
    "1": "Joined",
    "2": "Stayed"
  }
}
```
* Explore interactive docs at http://127.0.0.1:8080/docs.

---

## 🧩 Airflow (Local Execution)
* Standalone `airflow-compose.yaml` launches:
  * Airflow with `LocalExecutor`
  * No Redis, Celery, or Flower — only core services activated.

* Before startup, create directories:
```bash
mkdir -p terraform/airflow/{dags,logs,config,plugins,scripts}
```
* Copy variables:
```bash
cp .env.template .env
```
and fill in:
* `AIRFLOW_PROJ_DIR`
* `AIRFLOW_UID`
* `AIRFLOW_IMAGE_NAME`
* `_AIRFLOW_WWW_USER_USERNAME`
* `_AIRFLOW_WWW_USER_PASSWORD`
* `AIRFLOW_FERNET_KEY`

> [!IMPORTANT]
> Airflow uses Fernet to encrypt passwords in the connection configuration and the variable configuration. It guarantees that a password encrypted using it cannot be manipulated or read without the key. Fernet is an implementation of symmetric (also known as “secret key”) authenticated cryptography.
* You can generate the Airflow Fernet key with the following command:
```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Once you're satisfied with the Airflow configuration, you can start the Docker containers using the `airflow-compose.yaml` file:

```bash
docker compose -f airflow-compose.yaml up -d
```

> [!IMPORTANT]
> Since the Docker Compose file does not use the default name (`docker-compose.yaml`), you must specify its filename explicitly with the `-f` flag.

---

## 🛠 Troubleshooting & Tips
* Validate Docker Compose Setup:
  ```bash
  docker compose config
  ```
* If services restart unexpectedly:
  * Check logs:
  ```bash
  docker logs <containerName>
  ```
  * Open Shell to debug
  ```bash
  docker exec -it <container> /bin/sh
  ```
* Ensure that MLflow and Docker local registry are up and running
* Ensure that you are not having port collisions when starting new services

---

## 🎯 Summary Workflow

1. Terraform apply → infra up (registry, MLflow)
2. Download data → run ML experiments → log results → retrieve best model
3. Build & push FastAPI Docker image
4. Start services via Docker Compose
5. Use API endpoints or deploy Airflow orchestration

---

## 📋 References
* [Maven Analytics – Telecom Customer Churn (CSV)](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data)
* [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
* [MLflow Documentation](https://mlflow.org/docs/latest/)
* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

--- 

## ✅ Contributors & License

* Author: Georgi Stoyanov
* License: MIT