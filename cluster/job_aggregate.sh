#!/usr/bin/env bash
#SBATCH --job-name=aif-aggregate
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --gpus=1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

source cluster/setup_env.sh

# OUTPUT_DIR and N_EPISODES are passed via --export from submit_all.sh
python scripts/minigrid/aggregate_episodes.py \
    --output-dir "$OUTPUT_DIR" \
    --n-episodes "$N_EPISODES"
