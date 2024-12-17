#!/bin/bash

# connect to the remote server
ssh -L 5000:127.0.0.1:5000 strivaud@ssh.saclay.inria.fr
cd /home/tau/strivaud/gromo || { echo "Failed to change directory"; exit 1; }

# launching the mlflow server
backend_store_uri="logs/mlp_run"
artifact_store_uri="logs/mlp_run"
mlflow server \
  --backend-store-uri $backend_store_uri \
  --default-artifact-root $artifact_store_uri \
  --host
