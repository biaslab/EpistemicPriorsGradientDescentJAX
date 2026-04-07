#!/usr/bin/env bash
#SBATCH --job-name=aif-tmaze
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tmaze_%x_%j.out
#SBATCH --error=logs/tmaze_%x_%j.err

# Generic T-maze job. Dispatches based on STAGE_TYPE env var:
#   convergence  — requires ANALYSIS (curves|lr_sweep|budget|variance|policy_stability|scenario_curves)
#   experiment   — requires INFERENCE_MODE (marginal|active|planning)
#   figures      — no extra vars needed

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

echo "Running T-maze ${STAGE_TYPE} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

case "${STAGE_TYPE:?STAGE_TYPE not set}" in
    convergence)
        python scripts/tmaze/convergence_analysis.py \
            --analysis "${ANALYSIS:?ANALYSIS not set}" \
            --output-dir data/tmaze/convergence
        ;;
    experiment)
        python scripts/tmaze/experiment.py \
            --inference-mode "${INFERENCE_MODE:?INFERENCE_MODE not set}" \
            --output-dir "data/tmaze/${INFERENCE_MODE}"
        ;;
    figures)
        python scripts/tmaze/plot_convergence.py \
            --input-dir data/tmaze/convergence \
            --output-dir data/tmaze/convergence/figures
        ;;
    *)
        echo "ERROR: unknown STAGE_TYPE='${STAGE_TYPE}' (expected convergence|experiment|figures)"
        exit 1
        ;;
esac

echo "T-maze ${STAGE_TYPE} completed at $(date)"
