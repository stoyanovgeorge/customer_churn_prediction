#!/bin/bash


for n in 50 100 150; do
  for d in "None" 5 10 15; do
    echo "Running with n_estimators=$n, max_depth=$d"
    
    # Build base command
    cmd="python train_rf_model_mlflow.py -D ./data/data_full.csv --n_estimators $n --mlflow_uri=http://127.0.0.1:7000"
    
    # Append max_depth only if it's not None
    if [ "$d" != "None" ]; then
      cmd="$cmd --max_depth $d"
    fi

    # Run the command
    eval "$cmd"
  done
done