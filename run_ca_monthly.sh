#!/bin/sh
set -e

# Find the project directory dynamically
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Source the central conda initializer (adjust path to match the machine's conda installation)
source /cpc/home/mgoss/miniconda3/etc/profile.d/conda.sh

# Activate the local environment using its explicit prefix path
conda activate ./env

# Run the python orchestrator and pass along any bash arguments
python run_ca_monthly.py "$@"
