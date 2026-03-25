# Running Temporal VFE Experiments on Snellius

This guide explains how to run the three temporal VFE inference experiments — **active**, **marginal**, and **planning** — on the [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) HPC cluster (SURF).

## Prerequisites

- A Snellius account with access to the `gpu` partition
- SSH access configured (`ssh snellius.surf.nl`)
- This repository cloned on Snellius

## Getting started

SSH into Snellius and clone the repository:

```bash
ssh snellius.surf.nl
git clone <repo-url>
cd EpistemicPriorsExperiments
```

### Directory structure

```
cluster/
├── setup_env.sh          # Shared environment bootstrap (modules + venv)
├── job_active.sh         # SLURM job: temporal VFE + epistemic priors
├── job_marginal.sh       # SLURM job: temporal VFE without epistemic priors
├── job_planning.sh       # SLURM job: temporal VFE + entropy priors
├── job_convergence.sh    # SLURM job: convergence analysis
├── job_aggregate.sh      # SLURM job: aggregate episode results
└── submit_all.sh         # Submit all jobs at once
```

## Configuration

All experiment parameters are controlled by `params.yaml` in the project root. The job scripts read from this file at runtime, so you only need to edit it once.

Key parameters for the temporal VFE experiments:

```yaml
minigrid:
  grid_size: 3                # MiniGrid DoorKey grid size
  n_episodes: 10              # Number of episodes to run
  max_steps: 12               # Maximum steps per episode
  fov_size: 3                 # Field-of-view size
  seed: 42                    # Random seed
  goal_scale: 1.0             # Goal reward scaling
  record: "first,last"        # Which episodes to record ("first,last", "all", "none")

  planning_horizon: 12        # Policy planning horizon (timesteps)
  n_optimization_steps: 1500  # Adam iterations per planning step
  learning_rate: 0.01         # Optimizer learning rate
  optimizer_type: "adafactor" # Optimizer type
```

Edit `params.yaml` before submitting jobs. All three temporal VFE jobs share the same parameters — only the `--inference-mode` flag differs between them.

## Environment setup

The script `cluster/setup_env.sh` handles module loading and virtual environment creation. It:

1. Loads `Python/3.11.3-GCCcore-12.3.0`
2. Creates a GPU venv at `.venvs/venv-gpu/` with the project installed in editable mode plus `jax[cuda12]`
3. On subsequent runs, activates the existing venv

**Important:** Run the environment setup once on the login node before submitting jobs, since compute nodes may lack internet access:

```bash
source cluster/setup_env.sh
```

This creates the venv and installs all dependencies. You only need to do this once (or after changing dependencies).

## Running individual experiments

Each experiment has its own SLURM job script. All three use identical resource allocations:

| Resource       | Value      |
|----------------|------------|
| Partition      | `gpu_a100` |
| GPUs           | 1          |
| CPUs per task  | 18         |
| Memory         | 32 GB      |
| Time limit     | 1 hour     |

### Active inference

Temporal VFE with epistemic priors (control, state, and observation priors for information-seeking behavior):

```bash
sbatch cluster/job_active.sh
```

- Job name: `aif-active`
- Inference mode: `active`
- Output directory: `data/minigrid/active/`
- Logs: `logs/active_<jobid>.out` and `logs/active_<jobid>.err`

### Marginal inference

Standard temporal VFE without epistemic priors (baseline — no information-seeking):

```bash
sbatch cluster/job_marginal.sh
```

- Job name: `aif-marginal`
- Inference mode: `marginal`
- Output directory: `data/minigrid/marginal/`
- Logs: `logs/marginal_<jobid>.out` and `logs/marginal_<jobid>.err`

### Planning inference

Temporal VFE with entropy-based priors (prefer states with high observation entropy):

```bash
sbatch cluster/job_planning.sh
```

- Job name: `aif-planning`
- Inference mode: `planning`
- Output directory: `data/minigrid/planning/`
- Logs: `logs/planning_<jobid>.out` and `logs/planning_<jobid>.err`

### What the job scripts do

Each job script follows the same structure:

1. Sets JAX/XLA environment variables (`JAX_PLATFORMS=gpu`, XLA Triton softmax fusion)
2. Sources `cluster/setup_env.sh` to activate the GPU venv
3. Reads all parameters from `params.yaml` using a Python YAML helper
4. Runs `python scripts/minigrid/experiment.py` with the appropriate `--inference-mode` and all flags from `params.yaml`

All three jobs also pass `--freeze-obs-and-transitions` and `--receding-horizon` flags.

## Running all experiments at once

To submit all temporal VFE jobs (plus convergence) in parallel:

```bash
bash cluster/submit_all.sh
```

This submits 4 independent jobs and prints their job IDs:

```
Submitting all jobs...
  active:          12345
  planning:        12346
  marginal:        12347
  convergence:     12350
```

All jobs run independently — there are no dependencies between them.

## Monitoring jobs

### Check job status

```bash
squeue -u $USER
```

### Follow log output in real time

```bash
# Substitute the job ID printed at submission
tail -f logs/active_12345.out
tail -f logs/marginal_12346.out
tail -f logs/planning_12347.out
```

### Cancel a job

```bash
scancel <jobid>

# Cancel all your jobs
scancel -u $USER
```

## Output structure

Each experiment writes its results to a subdirectory under `data/minigrid/`:

```
data/minigrid/
├── active/
│   ├── results.json          # Episode metrics (rewards, steps, VFE values)
│   └── recordings/           # mp4 videos + per-frame PNGs
├── marginal/
│   ├── results.json
│   └── recordings/
└── planning/
    ├── results.json
    └── recordings/
```

The `record` parameter in `params.yaml` controls which episodes get recorded. With `"first,last"`, only the first and last episodes produce video output.

## Troubleshooting

### Module not found

If `module load` fails, check available versions:

```bash
module spider Python
module spider CUDA
```

The scripts expect `Python/3.11.3-GCCcore-12.3.0`. If unavailable, update the `PYTHON_MODULE` variable at the top of `cluster/setup_env.sh`.

### Venv creation fails on compute node

Compute nodes may not have internet access. Always create the venv on the login node first:

```bash
source cluster/setup_env.sh
```

### No GPU available / long queue times

The `gpu` partition can be busy. Check queue status with:

```bash
sinfo -p gpu
```

If wait times are long, you can check estimated start time:

```bash
squeue -u $USER --start
```

### JAX does not see the GPU

The job logs print JAX device info at the start. If you see `CpuDevice` instead of `GpuDevice`, verify that:

1. The job is running on the `gpu` partition (check `#SBATCH --partition=gpu`)
2. `jax[cuda12]` is installed in the venv (check `setup_env.sh` was sourced)
3. `jax[cuda12]` is installed in the venv (`pip list | grep jax`)

### Stale venv after dependency changes

If you update `pyproject.toml` or dependencies, delete and recreate the venv:

```bash
rm -rf .venvs/venv-gpu
source cluster/setup_env.sh
```
