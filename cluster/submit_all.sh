#!/usr/bin/env bash
# Submit all SLURM array jobs with resume support and dependent aggregation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

# Max concurrent array tasks per experiment (courtesy to other users)
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"

# Read n_episodes from params.yaml
N_EPISODES=$(python -c "import yaml; p=yaml.safe_load(open('params.yaml')); print(p['minigrid']['n_episodes'])")
echo "N_EPISODES=$N_EPISODES, MAX_CONCURRENT=$MAX_CONCURRENT"
echo ""

# submit_array_job <job_script> <output_dir> <label>
# Checks for missing episodes, submits array job + aggregation job.
submit_array_job() {
    local job_script="$1"
    local output_dir="$2"
    local label="$3"

    # Find missing episodes
    local missing
    missing=$(python scripts/minigrid/find_missing_episodes.py "$output_dir" "$N_EPISODES" 2>/dev/null || echo "0-$((N_EPISODES - 1))")

    if [ -z "$missing" ]; then
        echo "  $label: all $N_EPISODES episodes complete, skipping"
        return
    fi

    echo "  $label: submitting array=$missing"

    # Submit array job (override default --array from script)
    local array_job_id
    array_job_id=$(sbatch --parsable --array="${missing}%${MAX_CONCURRENT}" "$job_script")
    echo "    array job: $array_job_id"

    # Submit aggregation job dependent on array completion
    local agg_job_id
    agg_job_id=$(sbatch --parsable \
        --dependency="afterany:${array_job_id}" \
        --export="ALL,OUTPUT_DIR=${output_dir},N_EPISODES=${N_EPISODES}" \
        cluster/job_aggregate.sh)
    echo "    aggregate job: $agg_job_id (after $array_job_id)"
}

echo "Submitting jobs..."

submit_array_job cluster/job_active.sh data/minigrid/active active
submit_array_job cluster/job_planning.sh data/minigrid/planning planning
submit_array_job cluster/job_marginal.sh data/minigrid/marginal marginal

# Convergence job is not episode-based, submit as before
JOB_CONV=$(sbatch --parsable cluster/job_convergence.sh)
echo "  convergence: $JOB_CONV"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/*_\${JOBID}_*.out"
