#!/usr/bin/env bash
# Submit all T-maze DVC stages as individual SLURM jobs.
# Convergence stages run in parallel; the figures stage depends on all of them.
# Experiment stages are independent.
# Stages whose output files already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

JOB_SCRIPT="cluster/job_tmaze.sh"

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

echo "=== T-Maze stages ==="

# --- Convergence analyses (6 stages, all parallel) ---
CONV_JOBS=""
for analysis in curves lr_sweep budget variance policy_stability scenario_curves; do
    OUTPUT="data/tmaze/convergence/${analysis}.json"
    if [ -f "$OUTPUT" ]; then
        echo "  convergence/${analysis}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "tm-conv-${analysis}" "STAGE_TYPE=convergence,ANALYSIS=${analysis}")
    echo "  convergence/${analysis}: $JID"
    CONV_JOBS="${CONV_JOBS:+${CONV_JOBS}:}${JID}"
done

# --- Figures (depends on ALL convergence stages) ---
if [ -d "data/tmaze/convergence/figures" ] && [ -n "$(ls -A data/tmaze/convergence/figures 2>/dev/null)" ]; then
    echo "  figures: skipped (data/tmaze/convergence/figures/ exists)"
elif [ -n "$CONV_JOBS" ]; then
    FIG_JID=$(submit_stage "tm-figures" "STAGE_TYPE=figures" "--dependency=afterok:${CONV_JOBS}")
    echo "  figures: $FIG_JID (after convergence)"
else
    # All convergence outputs exist, but figures dir is missing — run figures immediately
    FIG_JID=$(submit_stage "tm-figures" "STAGE_TYPE=figures")
    echo "  figures: $FIG_JID"
fi

# --- Experiments (3 stages, all parallel) ---
for mode in marginal active planning; do
    OUTPUT="data/tmaze/${mode}/results.json"
    if [ -f "$OUTPUT" ]; then
        echo "  experiment/${mode}: skipped (${OUTPUT} exists)"
        continue
    fi
    JID=$(submit_stage "tm-exp-${mode}" "STAGE_TYPE=experiment,INFERENCE_MODE=${mode}")
    echo "  experiment/${mode}: $JID"
done

echo ""
echo "T-maze jobs submitted. Monitor with: squeue -u \$USER"
