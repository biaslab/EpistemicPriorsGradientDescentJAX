# Running Experiments on Snellius

This guide explains how to run T-maze, epistemic maze, and MiniGrid experiments on the [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) HPC cluster (SURF).

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
├── setup_env.sh            # Shared environment bootstrap (modules + venv)
├── job_smoke_test.sh       # Smoke test: validate GPU + deps before real runs
├── submit_all.sh           # Submit jobs: all, or by environment
├── submit_tmaze.sh         # Submit all T-maze stages
├── submit_epistemic.sh     # Submit all epistemic maze stages
├── job_tmaze.sh            # SLURM job: T-maze (convergence/experiment/figures)
├── job_epistemic.sh        # SLURM job: epistemic maze (experiment/convergence/figures)
├── job_active.sh           # SLURM job: MiniGrid active (array, per-episode)
├── job_marginal.sh         # SLURM job: MiniGrid marginal (array, per-episode)
├── job_planning.sh         # SLURM job: MiniGrid planning (array, per-episode)
├── job_convergence.sh      # SLURM job: MiniGrid convergence
└── job_aggregate.sh        # SLURM job: MiniGrid aggregate episode results
```

## Configuration

All experiment parameters are controlled by `params.yaml` in the project root. Job scripts read from this file at runtime, so you only need to edit it once.

## Environment setup

The script `cluster/setup_env.sh` handles module loading and virtual environment creation. It:

1. Loads `Python/3.11.3-GCCcore-12.3.0`
2. Creates a GPU venv at `.venvs/venv-gpu/` with the project installed in editable mode plus `jax[cuda12]`
3. On subsequent runs, activates the existing venv

**Important:** Run the environment setup once on the login node before submitting jobs, since compute nodes may lack internet access:

```bash
source cluster/setup_env.sh
```

## Submitting jobs

### Smoke test (run first)

Validate that GPU, JAX, pymdp, and project imports all work before submitting real jobs:

```bash
sbatch cluster/job_smoke_test.sh
# wait for completion, then check:
cat logs/smoke_test_<jobid>.out
```

The smoke test checks:
1. JAX sees a GPU device
2. A JAX matmul actually executes on GPU
3. `inferactively-pymdp` is installed (and reports whether it's from the pinned git commit or PyPI)
4. All project modules are importable

### All environments

```bash
bash cluster/submit_all.sh          # submit everything
bash cluster/submit_all.sh tmaze    # T-maze only
bash cluster/submit_all.sh epistemic # epistemic maze only
bash cluster/submit_all.sh minigrid  # MiniGrid only
```

### T-maze

T-maze stages are lightweight (seconds to minutes each). Each DVC stage runs as its own SLURM job — no array jobs needed.

```bash
bash cluster/submit_tmaze.sh
```

This submits 10 jobs:

| Stage | Type | Dependencies |
|-------|------|--------------|
| 6 convergence analyses | GPU | None (parallel) |
| 1 convergence figures | GPU | After all 6 convergence analyses |
| 3 experiments (marginal, active, planning) | GPU | None (parallel) |

```
Parallel:                                    After all converge:
  tm-conv-curves              ─┐
  tm-conv-lr_sweep            ─┤
  tm-conv-budget              ─┼─ afterok ──> tm-figures
  tm-conv-variance            ─┤
  tm-conv-policy_stability    ─┤
  tm-conv-scenario_curves     ─┘
  tm-exp-marginal    (independent)
  tm-exp-active      (independent)
  tm-exp-planning    (independent)
```

### Epistemic maze

Similar to T-maze: one SLURM job per DVC stage.

```bash
bash cluster/submit_epistemic.sh
```

This submits 11 jobs:

| Stage | Type | Dependencies |
|-------|------|--------------|
| 3 temporal experiments (planning, active, marginal) | GPU | None (parallel) |
| 2 pymdp experiments (sophisticated, vanilla) | GPU | None (parallel) |
| 5 convergence analyses | GPU | None (parallel) |
| 1 convergence figures | GPU | After all 5 convergence analyses |

```
Parallel:                                         After all converge:
  ep-exp-planning   (temporal)
  ep-exp-active     (temporal)
  ep-exp-marginal   (temporal)
  ep-exp-sophisticated (pymdp)
  ep-exp-vanilla       (pymdp)
  ep-conv-curves              ─┐
  ep-conv-lr_sweep            ─┤
  ep-conv-budget              ─┼─ afterok ──> ep-figures
  ep-conv-variance            ─┤
  ep-conv-scenario_curves     ─┘
```

### MiniGrid

MiniGrid episodes are expensive, so they use SLURM array jobs (1 task = 1 episode) with resume support and dependent aggregation.

```bash
bash cluster/submit_all.sh minigrid
```

## Resource allocation

| Environment | Stages | Partition | GPU | CPUs | Mem | Time |
|---|---|---|---|---|---|---|
| T-maze | all 10 | gpu_a100 | 1 | 18 | 16G | 30min |
| Epistemic maze | all 11 | gpu_a100 | 1 | 18 | 16G | 30min |
| MiniGrid episodes | 3 array jobs | gpu_a100 | 1 | 18 | 64G | 30min-1hr |
| MiniGrid convergence | 1 job | gpu_a100 | 1 | 18 | 32G | 1hr |
| MiniGrid aggregation | 3 jobs | gpu_a100 | 1 | 1 | 4G | 10min |

## Running individual stages

You can submit individual stages by passing env vars directly:

```bash
# Single T-maze convergence analysis
sbatch --export=ALL,STAGE_TYPE=convergence,ANALYSIS=curves cluster/job_tmaze.sh

# Single T-maze experiment
sbatch --export=ALL,STAGE_TYPE=experiment,INFERENCE_MODE=active cluster/job_tmaze.sh

# Single epistemic maze temporal experiment
sbatch --export=ALL,STAGE_TYPE=experiment,STRATEGY=temporal,INFERENCE_MODE=active cluster/job_epistemic.sh

# Single epistemic maze pymdp experiment
sbatch --export=ALL,STAGE_TYPE=experiment,STRATEGY=sophisticated cluster/job_epistemic.sh

# Single epistemic maze convergence analysis
sbatch --export=ALL,STAGE_TYPE=convergence,ANALYSIS=lr_sweep cluster/job_epistemic.sh
```

## Monitoring jobs

### Check job status

```bash
squeue -u $USER
```

### Follow log output in real time

```bash
tail -f logs/tmaze_tm-conv-curves_12345.out
tail -f logs/epistemic_ep-exp-planning_12346.out
```

### Cancel a job

```bash
scancel <jobid>

# Cancel all your jobs
scancel -u $USER
```

## Output structure

```
data/
├── tmaze/
│   ├── marginal/     # results.json, stats.json, episode.mp4, frames/
│   ├── active/
│   ├── planning/
│   └── convergence/  # curves.json, lr_sweep.json, ..., figures/
├── epistemic_maze/
│   ├── planning/     # results.json
│   ├── active/
│   ├── marginal/
│   ├── sophisticated/
│   ├── vanilla/
│   └── convergence/  # curves.json, lr_sweep.json, ..., figures/
└── minigrid/
    ├── active/       # episodes/, results.json, recordings/
    ├── marginal/
    ├── planning/
    └── convergence/
```

## Troubleshooting

### Module not found

If `module load` fails, check available versions:

```bash
module spider Python
```

Update `PYTHON_MODULE` in `cluster/setup_env.sh` if needed.

### Venv creation fails on compute node

Compute nodes may not have internet access. Always create the venv on the login node first:

```bash
source cluster/setup_env.sh
```

### No GPU available / long queue times

```bash
sinfo -p gpu
squeue -u $USER --start    # estimated start time
```

### JAX does not see the GPU

The job logs print JAX device info at the start. If you see `CpuDevice` instead of `GpuDevice`, verify that:

1. The job is running on the `gpu` partition
2. `jax[cuda12]` is installed in the venv (`pip list | grep jax`)

### Stale venv after dependency changes

```bash
rm -rf .venvs/venv-gpu
source cluster/setup_env.sh
```
