import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:7000")

EXPERIMENT_NAME = "RF Churn Prediction"
METRIC = "accuracy"  
PIPELINE_DIR = "app/"  # Model export path

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"No experiment found with the name: '{EXPERIMENT_NAME}'")
    exit(1)

runs = client.search_runs(experiment.experiment_id)

runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.{METRIC} DESC"], max_results=1)

if not runs:
    raise ValueError("No runs found")

accuracy = runs[0].data.metrics["test_accuracy"]

best_run = runs[0]

run_id = best_run.info.run_id

local_path = client.download_artifacts(
    run_id=run_id,
    path="pipeline.pkl",
    dst_path=PIPELINE_DIR
)

print(f"The best pipeline of Run ID: {run_id} is saved to: {local_path}")