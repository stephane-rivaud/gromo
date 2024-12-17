#!/bin/bash

# connect to the remote server
port=6000
ssh -L $port:127.0.0.1:$port strivaud@ssh.saclay.inria.fr
ssh -A $port:127.0.0.1:$port titanic
cd /home/tau/strivaud/gromo || {
  echo "Failed to change directory"
  exit 1
}

# launching the mlflow server
backend_store_uri="logs/mlp_run"
artifact_store_uri="logs/mlp_run"
mlflow server \
  --backend-store-uri $backend_store_uri \
  --default-artifact-root $artifact_store_uri \
  --host 127.0.0.1 \
  --port $port
