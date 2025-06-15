# 📊 Customer Churn Prediction with MLflow, Airflow, Docker, Terraform & FastAPI

This project tackles customer churn prediction using a **Random Forest classifier** on a real-world telecom dataset from [Maven Analytics on Kaggle](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics).

---

## 🚀 Project Overview

### ✅ Core Objectives
- Train models to predict customer churn.
- Compare results across two dataset sizes using **MLflow**.
- Package the best model into a **Dockerized FastAPI app**.
- Manage infrastructure with **Terraform** (local Docker).
- Orchestrate training with **Airflow**.
- Basic monitoring of the deployed service.

---

## 📂 Dataset

- **Source**: [Telecom Customer Churn](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics)
- **Format**: Tabular CSV
- **Target variable**: `Customer Status`

We prepare two versions of the dataset:
- `data_sample_1000.csv` – a sample of 1,000 rows
- `data_full.csv` – the full dataset

---

## Terraform Setup

### 📥 Terraform Installation

To get started, make sure Terraform is installed on your machine. You can refer to the official Terraform documentation for installation instructions tailored to your operating system: [Terrfaform Installation](https://developer.hashicorp.com/terraform/install)

### 🔧 Terraform Environment Variables

Open the `dev.tfvars` file in your preferred editor and replace all placeholder values with actual configuration values. Key variables to update include:

* `mlflow_image` - The version tag of the MLflow Docker image being used
* `external_mlflow_port` - The host machine’s external port that receives incoming MLflow requests
* `docker_network_name` - Name of the Docker Network
* `uid` - User ID
* `gid` - Group ID
* and others defined in the `dev.tfvars` file

> [!IMPORTANT] 
> Double-check all values before moving on. Incorrect values may lead to deployment failures or misconfiguration.

### 🚀 Terraform Deployment

Once your `dev.tfvars` is properly configured, you're ready to deploy the infrastructure.

1. Review the plan to see what changes Terraform will make:
```bash
terraform plan --var-file="dev.tfvars"
```
2. Apply the plan to provision the resources:
```bash
terraform apply --var-file="dev.tfvars"
```
You’ll be prompted to confirm the changes. Type yes to proceed.

> [!NOTE]
> This action will start a Docker network, a local Docker registry, and an MLflow tracking server. To access them, refer to the configured values for `external_registry_port` and `external_mlflow_port`.

## 🧪 ML Experiment Tracking with MLflow

You can access the web UI of MLflow by navigating to `http://localhost:{external_mlflow_port}`. I have configured the 

## Docker Compose

The `docker-compose.yaml` file defines services for launching an Airflow instance and a FastAPI application. The FastAPI app exposes a single `POST /predict` endpoint, which allows you to use the best model to make classification predictions for the `Customer Status` column.

### Docker Compose Environment Variables

The Docker Compose file relies on multiple environment variables. Before proceeding, copy the `.env.template` file to `.env`:
```bash
cp terraform/.env.template terraform/.env
```
And update the values as needed:
* `AIRFLOW_PROJ_DIR` - The root directory of your Airflow project.
* `AIRFLOW_UID` - The user ID (UID) of your local user.
* `AIRFLOW_IMAGE_NAME` - The Airflow Docker image name [Docker Images](https://hub.docker.com/r/apache/airflow)
* `_AIRFLOW_WWW_USER_USERNAME` - The Airflow webserver login username.
* `_AIRFLOW_WWW_USER_PASSWORD` - The Airflow webserver login password.
* `AIRFLOW_FERNET_KEY` - The secret key used by Airflow to encrypt sensitive data. See below for how to generate it.

## 🚀 Airflow DAG (Directed Acyclic Graph)



### 🔧 Airflow Installation

You can follow the official Airflow installation tutorial using Docker Compose: [How to Install Apache Airflow with Docker Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html). 

I've simplified the docker-compose.yaml file for local execution. Since we’re running Airflow locally, we can safely switch the executor to LocalExecutor, which doesn’t require a distributed message broker like Redis. As a result, we can remove the Redis service as well as Flower, which is used for monitoring Celery workers — both of which are unnecessary in this setup.

Additionally, we can also remove the following components to further simplify the configuration:

* Celery workers (if present): only needed for CeleryExecutor
* Worker queue configuration: relevant only for distributed setups
* Volumes for Redis logs/data
* Environment variables related to Celery or Redis, such as:
  * `AIRFLOW__CELERY__BROKER_URL`
  * `AIRFLOW__CELERY__RESULT_BACKEND`
* Ports and healthchecks related to Redis or Flower

You can find the final docker-compose.yaml file in the terraform/ directory.

Before proceeding, make sure to create the required directory structure for Airflow by running:
```bash
mkdir -p terraform/airflow/{dags,logs,config,plugins,includes}
```
This command will create the following subdirectories under `terraform/airflow/`:
```text
airflow/
├── config
├── dags
├── logs
└── plugins
```
> [!TIP]
> You can inspect your final Docker Compose configuration by running: `docker compose config`. This command resolves and prints the fully rendered configuration by merging your docker-compose.yaml with the .env file and substituting all environment variables with their actual values.

#### 🔐 Airflow Fernet Key
Airflow uses Fernet to encrypt passwords in the connection configuration and the variable configuration. It guarantees that a password encrypted using it cannot be manipulated or read without the key. Fernet is an implementation of symmetric (also known as “secret key”) authenticated cryptography.

You can generate Airflow Fernet key with the following script:

```python
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 🐳 Building and Pushing the Docker Container to Local Registry

This section describes how to build your FastAPI-based Random Forest API prediction endpoint, tag it properly, and push it to your local Docker registry (127.0.0.1:5000).

1. Build the Docker Image

Use the following command to build the Docker image using the `Dockerfile` and assign it a versioned tag (1.0.0):
```bash
docker build -t rf-api:1.0.0 .
```
This command will:
* Use the current directory (.) as the build context
* Create a Docker image named rf-api with the tag 1.0.0

2. Tag the Image for Your Local Registry
To push the image to your local registry, you need to tag it with the registry address (127.0.0.1:5000):
```bash
docker tag rf-api:1.0.0 127.0.0.1:5000/rf-api:1.0.0
```
> [!INFO]
> This step simply creates a new reference to the same image so that Docker knows which registry to push it to.

3. Push the Image to the Local Registry
Push the tagged image to your local Docker registry:
```bash
docker push 127.0.0.1:5000/rf-api:1.0.0
```
If successful, the image will now be available in your local registry and can be deployed by other services via docker run or docker compose.

**Optional**: To confirm the image is available run:
```bash
$ curl http://127.0.0.1:5000/v2/_catalog
{"repositories":["rf-api"]}
```
You should see your repository listed here. 
To check available tags for the image:
```bash
curl http://127.0.0.1:5000/v2/rf-api/tags/list
{"name":"rf-api","tags":["1.0.0"]}
```