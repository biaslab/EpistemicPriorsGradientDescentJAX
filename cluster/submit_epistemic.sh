#!/usr/bin/env bash
# Submit all epistemic maze DVC stages as individual SLURM jobs.
# Convergence stages run in parallel; the figures stage depends on all of them.
# Experiment stages are independent.
# Stages whose output files already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

JOB_SCRIPT="cluster/job_epistemic.sh"

# submit_stage <label> <env_vars> [dependency_flag]
submit_stage() {
    local label="$1"
    local env_vars="$2"
    local dep_flag="${3:-}"

    sbatch --parsable \
        --job-name="$label" \
        --export="ALL,${env_vars}" \
        ${dep_flag} \
        "$JOB_SCRIPT"
}

echo "=== Epistemic Maze stages ==="

# --- Temporal experiment stages (3, parallel) ---
for mode in planning active marginal; do
    OUTPUT="data/epistemic_maze/${mode}/results.json"
    if [ -f "$OUTPUT" ]; then
        echo "  experiment/temporal/${mode}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "ep-exp-${mode}" "STAGE_TYPE=experiment,STRATEGY=temporal,INFERENCE_MODE=${mode}")
    echo "  experiment/temporal/${mode}: $JID"
done

# --- pymdp experiment stages (2, parallel) ---
for strategy in sophisticated vanilla; do
    OUTPUT="data/epistemic_maze/${strategy}/results.json"
    if [ -f "$OUTPUT" ]; then
        echo "  experiment/${strategy}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "ep-exp-${strategy}" "STAGE_TYPE=experiment,STRATEGY=${strategy}")
    echo "  experiment/${strategy}: $JID"
done

# --- Convergence analyses (5 stages, all parallel) ---
CONV_JOBS=""
for analysis in curves lr_sweep budget variance scenario_curves; do
    OUTPUT="data/epistemic_maze/convergence/${analysis}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  convergence/${analysis}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "ep-conv-${analysis}" "STAGE_TYPE=convergence,ANALYSIS=${analysis}")
    echo "  convergence/${analysis}: $JID"
    CONV_JOBS="${CONV_JOBS:+${CONV_JOBS}:}${JID}"
done

# --- Figures (depends on ALL convergence stages) ---
if [ -d "data/epistemic_maze/convergence/figures" ] && [ -n "$(ls -A data/epistemic_maze/convergence/figures 2>/dev/null)" ]; then
    echo "  figures: skipped (data/epistemic_maze/convergence/figures/ exists)"
elif [ -n "$CONV_JOBS" ]; then
    FIG_JID=$(submit_stage "ep-figures" "STAGE_TYPE=figures" "--dependency=afterok:${CONV_JOBS}")
    echo "  figures: $FIG_JID (after convergence)"
else
    # All convergence outputs exist, but figures dir is missing — run figures immediately
    FIG_JID=$(submit_stage "ep-figures" "STAGE_TYPE=figures")
    echo "  figures: $FIG_JID"
fi

echo ""
echo "Epistemic maze jobs submitted. Monitor with: squeue -u \$USER"
