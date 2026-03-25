#!/usr/bin/env bash
#SBATCH --job-name=aif-convergence
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/convergence_%j.out
#SBATCH --error=logs/convergence_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

# Read params from params.yaml
read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

source cluster/setup_env.sh

GRID_SIZE=$(read_param "['minigrid']['grid_size']")
FOV_SIZE=$(read_param "['minigrid']['fov_size']")
N_OPT_STEPS=$(read_param "['minigrid']['n_optimization_steps']")
LEARNING_RATE=$(read_param "['minigrid']['learning_rate']")
HORIZON=$(read_param "['minigrid']['planning_horizon']")
SEED=$(read_param "['minigrid']['seed']")
GOAL_SCALE=$(read_param "['minigrid']['goal_scale']")
THRESHOLD=$(read_param "['convergence']['threshold']")

echo "Running convergence analysis on $(hostname)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python scripts/minigrid/convergence.py \
    --grid-size "$GRID_SIZE" \
    --fov-size "$FOV_SIZE" \
    --n-opt-steps "$N_OPT_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --horizon "$HORIZON" \
    --seed "$SEED" \
    --freeze-obs-and-transitions \
    --goal-scale "$GOAL_SCALE" \
    --convergence-threshold "$THRESHOLD" \
    --output-dir data/minigrid/convergence
