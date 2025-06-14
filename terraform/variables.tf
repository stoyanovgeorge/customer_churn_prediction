# Docker Network
variable "docker_network_name" {
  type        = string
  description = "Name of the Docker network for MLflow"
}

# Registry
variable "registry_image" {
  type        = string
  description = "Docker image for the local registry"
}

variable "external_registry_port" {
  type        = number
  description = "Port exposed for the local Docker registry"
  default     = 5000
}

variable "uid" {
  default = 1000
}

variable "gid" {
  default = 1000
}

# MLflow
variable "mlflow_image" {
  type        = string
  description = "MLflow Docker image to use"
}

variable "external_mlflow_port" {
  type        = number
  description = "External port to access the MLflow tracking server"
}

variable "mlflow_container_path" {
  description = "Container path for volume mount"
  type        = string
  default     = "/mlflow/mlruns"  
}

variable "mlflow_backend_store_uri" {
  type        = string
  description = "Backend store URI for MLflow tracking server"
}

variable "mlflow_artifact_root" {
  type        = string
  description = "Default artifact root directory for MLflow"
}