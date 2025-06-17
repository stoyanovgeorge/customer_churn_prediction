#!/bin/bash

# Define parameter values
N_ESTIMATORS_LIST=(50 100 200)
MAX_DEPTH_LIST=(None 5 10)
MIN_SAMPLES_SPLIT_LIST=(2 4)


# Dataset and MLflow URI
DATASET="data/data_full.csv"
MLFLOW_URI="http://localhost:7000"

# Loop over all combinations
for N_ESTITMATORS in "${N_ESTIMATORS_LIST[@]}"; do
  for MAX_DEPTH in "${MAX_DEPTH_LIST[@]}"; do
    for MIN_SAMPLES_SPLIT in "${MIN_SAMPLES_SPLIT_LIST[@]}"; do

      echo "Running with n_estimators=$N_ESTITMATORS, max_depth=$MAX_DEPTH, min_samples_split=$MIN_SAMPLES_SPLIT"

      python mlflow_logging.py \
        -D "$DATASET" \
        --n_estimators "$N_ESTITMATORS" \
        --max_depth "$MAX_DEPTH" \
        --min_ssplit "$MIN_SAMPLES_SPLIT" \
        --mlflow_uri "$MLFLOW_URI"

    done
  done
done