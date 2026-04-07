#!/usr/bin/env bash
#SBATCH --job-name=aif-epistemic
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/epistemic_%x_%j.out
#SBATCH --error=logs/epistemic_%x_%j.err

# Generic epistemic maze job. Dispatches based on STAGE_TYPE env var:
#   experiment   — requires STRATEGY (temporal|sophisticated|vanilla)
#                  and INFERENCE_MODE (planning|active|marginal) when STRATEGY=temporal
#   convergence  — requires ANALYSIS (curves|lr_sweep|budget|variance|scenario_curves)
#   figures      — no extra vars needed

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"
mkdir -p logs

export JAX_PLATFORMS="cuda"

source cluster/setup_env.sh

# Read params from params.yaml (same pattern as minigrid job scripts)
read_param() {
    python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p$1)"
}

echo "Running epistemic maze ${STAGE_TYPE} on $(hostname) at $(date)"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# --- Common environment params ---
N_EPISODES=$(read_param "['epistemic_maze']['n_episodes']")
HORIZON=$(read_param "['epistemic_maze']['horizon']")
MAX_STEPS=$(read_param "['epistemic_maze']['max_steps']")
N_THETA=$(read_param "['epistemic_maze']['n_theta']")
GOAL_TEMP=$(read_param "['epistemic_maze']['goal_temperature']")
CUE_ACC=$(read_param "['epistemic_maze']['cue_observation_accuracy']")
SEED=$(read_param "['epistemic_maze']['seed']")
RECEDING_HORIZON=$(read_param "['epistemic_maze']['receding_horizon']")

# Build receding horizon flag
RECEDING_FLAG=""
if [ "$RECEDING_HORIZON" = "False" ] || [ "$RECEDING_HORIZON" = "false" ]; then
    RECEDING_FLAG="--no-receding-horizon"
fi

case "${STAGE_TYPE:?STAGE_TYPE not set}" in
    experiment)
        STRATEGY="${STRATEGY:?STRATEGY not set}"

        if [ "$STRATEGY" = "temporal" ]; then
            # Temporal strategy needs optimization params + inference mode
            N_OPT_STEPS=$(read_param "['epistemic_maze']['n_optimization_steps']")
            LR=$(read_param "['epistemic_maze']['learning_rate']")

            python scripts/epistemic_maze/experiment.py \
                --strategy temporal \
                --inference-mode "${INFERENCE_MODE:?INFERENCE_MODE not set}" \
                --output-dir data/epistemic_maze \
                --n-episodes "$N_EPISODES" \
                --horizon "$HORIZON" \
                --max-steps "$MAX_STEPS" \
                --n-opt-steps "$N_OPT_STEPS" \
                --learning-rate "$LR" \
                --n-theta "$N_THETA" \
                --goal-temperature "$GOAL_TEMP" \
                --cue-accuracy "$CUE_ACC" \
                --seed "$SEED" \
                $RECEDING_FLAG
        else
            # Sophisticated/vanilla: no optimization params, no inference mode
            python scripts/epistemic_maze/experiment.py \
                --strategy "$STRATEGY" \
                --output-dir data/epistemic_maze \
                --n-episodes "$N_EPISODES" \
                --horizon "$HORIZON" \
                --max-steps "$MAX_STEPS" \
                --n-theta "$N_THETA" \
                --goal-temperature "$GOAL_TEMP" \
                --cue-accuracy "$CUE_ACC" \
                --seed "$SEED"
        fi
        ;;

    convergence)
        ANALYSIS="${ANALYSIS:?ANALYSIS not set}"

        # Common args shared by all convergence analyses
        COMMON_ARGS=(
            --output-dir data/epistemic_maze/convergence
            --n-theta "$N_THETA"
            --horizon "$HORIZON"
            --goal-temperature "$GOAL_TEMP"
            --cue-accuracy "$CUE_ACC"
            --max-steps "$MAX_STEPS"
        )

        case "$ANALYSIS" in
            curves)
                CONV_N_EPS=$(read_param "['epistemic_maze']['convergence']['curves']['n_episodes']")
                CONV_N_OPT=$(read_param "['epistemic_maze']['convergence']['curves']['n_optimization_steps']")
                CONV_LR=$(read_param "['epistemic_maze']['convergence']['curves']['learning_rate']")
                CONV_SEED=$(read_param "['epistemic_maze']['convergence']['curves']['seed']")
                python scripts/epistemic_maze/convergence_analysis.py \
                    --analysis curves \
                    --n-episodes "$CONV_N_EPS" \
                    --n-opt-steps "$CONV_N_OPT" \
                    --learning-rate "$CONV_LR" \
                    --seed "$CONV_SEED" \
                    "${COMMON_ARGS[@]}"
                ;;
            lr_sweep)
                CONV_N_EPS=$(read_param "['epistemic_maze']['convergence']['lr_sweep']['n_episodes']")
                CONV_N_OPT=$(read_param "['epistemic_maze']['convergence']['lr_sweep']['n_optimization_steps']")
                CONV_SEED=$(read_param "['epistemic_maze']['convergence']['lr_sweep']['seed']")
                python scripts/epistemic_maze/convergence_analysis.py \
                    --analysis lr_sweep \
                    --n-episodes "$CONV_N_EPS" \
                    --n-opt-steps "$CONV_N_OPT" \
                    --seed "$CONV_SEED" \
                    "${COMMON_ARGS[@]}"
                ;;
            budget)
                CONV_N_EPS=$(read_param "['epistemic_maze']['convergence']['budget']['n_episodes']")
                CONV_LR=$(read_param "['epistemic_maze']['convergence']['budget']['learning_rate']")
                CONV_SEED=$(read_param "['epistemic_maze']['convergence']['budget']['seed']")
                python scripts/epistemic_maze/convergence_analysis.py \
                    --analysis budget \
                    --n-episodes "$CONV_N_EPS" \
                    --learning-rate "$CONV_LR" \
                    --seed "$CONV_SEED" \
                    "${COMMON_ARGS[@]}"
                ;;
            variance)
                CONV_N_EPS=$(read_param "['epistemic_maze']['convergence']['variance']['n_episodes']")
                CONV_N_OPT=$(read_param "['epistemic_maze']['convergence']['variance']['n_optimization_steps']")
                CONV_LR=$(read_param "['epistemic_maze']['convergence']['variance']['learning_rate']")
                python scripts/epistemic_maze/convergence_analysis.py \
                    --analysis variance \
                    --n-episodes "$CONV_N_EPS" \
                    --n-opt-steps "$CONV_N_OPT" \
                    --learning-rate "$CONV_LR" \
                    "${COMMON_ARGS[@]}" \
                    --seed "$SEED"
                ;;
            scenario_curves)
                CONV_N_OPT=$(read_param "['epistemic_maze']['convergence']['curves']['n_optimization_steps']")
                CONV_LR=$(read_param "['epistemic_maze']['convergence']['curves']['learning_rate']")
                CONV_SEED=$(read_param "['epistemic_maze']['convergence']['curves']['seed']")
                python scripts/epistemic_maze/convergence_analysis.py \
                    --analysis scenario_curves \
                    --n-opt-steps "$CONV_N_OPT" \
                    --learning-rate "$CONV_LR" \
                    --seed "$CONV_SEED" \
                    "${COMMON_ARGS[@]}"
                ;;
            *)
                echo "ERROR: unknown ANALYSIS='${ANALYSIS}'"
                exit 1
                ;;
        esac
        ;;

    figures)
        python scripts/epistemic_maze/plot_convergence.py \
            --input-dir data/epistemic_maze/convergence \
            --output-dir data/epistemic_maze/convergence/figures
        ;;

    *)
        echo "ERROR: unknown STAGE_TYPE='${STAGE_TYPE}' (expected experiment|convergence|figures)"
        exit 1
        ;;
esac

echo "Epistemic maze ${STAGE_TYPE} completed at $(date)"
