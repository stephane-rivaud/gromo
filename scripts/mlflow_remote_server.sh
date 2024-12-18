#!/bin/bash

# Define the port to use for port forwarding
local_port=6000
remote_port=6000

# Connect to the remote server with port forwarding
ssh -L $local_port:127.0.0.1:$remote_port strivaud@ssh.saclay.inria.fr << 'EOF'
  # Connect to the titanic server
  ssh -L $remote_port:127.0.0.1:$remote_port 127.0.0.1 << 'EOT'
    # Navigate to the project directory
    cd /home/tau/strivaud/gromo || {
      echo "Failed to change directory"
      exit 1
    }

    # Launch the mlflow server
    backend_store_uri="logs/mlp_run"
    artifact_store_uri="logs/mlp_run"
    mlflow server \
      --backend-store-uri $backend_store_uri \
      --default-artifact-root $artifact_store_uri \
      --host 127.0.0.1 \
      --port $remote_port
  EOT
EOF