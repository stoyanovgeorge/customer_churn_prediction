# Docker Network
docker_network_name                     = "pipeline_network"

# Registry
registry_image                          = "registry:2"
external_registry_port                  = 5000

# MLflow
# You can change the MLflow image version here
# MLFlow Docu: https://mlflow.org/docs/latest/ml/docker
# Releases: https://github.com/mlflow/mlflow/pkgs/container/mlflow
mlflow_image                            = "ghcr.io/mlflow/mlflow:v3.0.0"
# External port for accessing the MLflow tracking server
external_mlflow_port                    = 7000
mlflow_backend_store_uri                = "sqlite:////mlflow/mlflow.db"

# Setting the User and Group IDs
uid                                     = 1000
gid                                     = 1000