terraform {
  required_providers {
    docker = {
      source = "kreuzwerker/docker"
      version = "3.6.1" # Latest Docker Provider Version https://registry.terraform.io/providers/kreuzwerker/docker/latest
    }
  }
}

provider "docker" {}

# Create a Docker network
resource "docker_network" "mlflow_net" {
  name = "mlflow_network"
}

# Local Docker registry container
resource "docker_image" "registry_image" {
  name = "registry:2"
}

resource "docker_container" "registry" {
  name  = "registry"
  image = docker_image.registry_image.name
  networks_advanced {
    name = docker_network.mlflow_net.name
  }

  ports {
    internal = 5000
    external = 5000
  }
}

# MLflow server container 
# MLFlow Docu: https://mlflow.org/docs/latest/ml/docker
# Releases: https://github.com/mlflow/mlflow/pkgs/container/mlflow
resource "docker_image" "mlflow_image" {
  name = "ghcr.io/mlflow/mlflow:v3.0.0"
}

resource "docker_container" "mlflow" {
  name  = "mlflow"
  image = docker_image.mlflow_image.name

  networks_advanced {
    name = docker_network.mlflow_net.name
  }

# IMPORTANT: I have changed the external port to 7000
# You can access MLflow using http://127.0.0.1:7000
  ports {
    internal = 5000
    external = 7000
  }

  # Simple local backend and artifact store
  volumes {
    host_path      = abspath("${path.module}/mlruns")
    container_path = "/mlflow/mlruns"
  }

  command = [
    "mlflow", "server",
    "--backend-store-uri", "file:/mlflow/mlruns",
    "--default-artifact-root", "file:/mlflow/mlruns",
    "--host", "0.0.0.0"
  ]
}