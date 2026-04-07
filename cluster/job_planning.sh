#!/usr/bin/env bash
#SBATCH --job-name=aif-planning
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-99
#SBATCH --output=logs/planning_%A_%a.out
#SBATCH --error=logs/planning_%A_%a.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

# Read params from params.yaml
read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

GRID_SIZE=$(read_param "['minigrid']['grid_size']")
N_EPISODES=$(read_param "['minigrid']['n_episodes']")
MAX_STEPS=$(read_param "['minigrid']['max_steps']")
PLANNING_HORIZON=$(read_param "['minigrid']['planning_horizon']")
N_OPT_STEPS=$(read_param "['minigrid']['n_optimization_steps']")
LEARNING_RATE=$(read_param "['minigrid']['learning_rate']")
OPTIMIZER_TYPE=$(read_param "['minigrid']['optimizer_type']")
FOV_SIZE=$(read_param "['minigrid']['fov_size']")
SEED=$(read_param "['minigrid']['seed']")
GOAL_SCALE=$(read_param "['minigrid']['goal_scale']")
RECORD=$(read_param "['minigrid']['record']")

echo "Running temporal VFE planning episode ${SLURM_ARRAY_TASK_ID} on $(hostname)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

python scripts/minigrid/experiment.py \
    --inference-mode planning \
    --grid-size "$GRID_SIZE" \
    --episodes "$N_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --planning-horizon "$PLANNING_HORIZON" \
    --n-opt-steps "$N_OPT_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --optimizer-type "$OPTIMIZER_TYPE" \
    --fov-size "$FOV_SIZE" \
    --seed "$SEED" \
    --freeze-obs-and-transitions \
    --receding-horizon \
    --goal-scale "$GOAL_SCALE" \
    --record "$RECORD" \
    --output-dir data/minigrid/planning \
    --episode-index "$SLURM_ARRAY_TASK_ID" 
