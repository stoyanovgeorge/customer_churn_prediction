terraform {
  # Releases: https://registry.terraform.io/providers/kreuzwerker/docker/latest
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.6.1"
    }
  }
}

provider "docker" {}

# Create a Docker network
resource "docker_network" "tf_net" {
  name = var.docker_network_name
}

# Local Docker registry container
resource "docker_image" "registry_image" {
  name = var.registry_image
}

resource "docker_container" "registry" {
  name  = "registry"
  image = docker_image.registry_image.name

  networks_advanced {
    name = docker_network.tf_net.name
  }

  ports {
    internal = 5000
    external = var.external_registry_port
  }
}

# MLflow server
resource "docker_image" "mlflow_image" {
  name = var.mlflow_image
}

resource "docker_container" "mlflow" {
  name  = "mlflow"
  image = docker_image.mlflow_image.name

  networks_advanced {
    name = docker_network.tf_net.name
  }

  ports {
    internal = 5000
    external = var.external_mlflow_port
  }

  volumes {
    host_path      = abspath("${path.module}/mlflow/mlruns")
    container_path = var.mlflow_container_path
  }

  command = [
    "mlflow", "server",
    "--backend-store-uri", var.mlflow_backend_store_uri,
    "--default-artifact-root", var.mlflow_artifact_root,
    "--host", "0.0.0.0"    
  ]
}

# Change recursively the user and group permissions of both the `airflow/` and `mlflow/` directories
resource "null_resource" "fix_mlflow_folder_ownership" {
  provisioner "local-exec" {
    command = "sudo chown -R ${var.uid}:${var.gid} ${abspath("${path.module}")}/mlflow/"
  }
}