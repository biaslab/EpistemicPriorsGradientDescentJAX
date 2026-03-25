# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase comparing inference modes for Active Inference planning via variational free energy (VFE) minimization in JAX. Implements three environments (T-Maze, Epistemic Maze, MiniGrid) with three inference modes (marginal, active/epistemic priors, planning correction).

## Commands

```bash
# Setup
uv sync

# Run tests
uv run python -m pytest tests/ -v

# Run experiments (T-Maze)
uv run python scripts/tmaze/experiment.py --inference-mode marginal
uv run python scripts/tmaze/experiment.py --inference-mode active
uv run python scripts/tmaze/experiment.py --inference-mode planning

# Run experiments (Epistemic Maze)
uv run python scripts/epistemic_maze/experiment.py --strategy temporal --inference-mode planning

# Run experiments (MiniGrid)
uv run python scripts/minigrid/experiment.py

# DVC pipeline (all experiments)
dvc repro

# Single DVC stage
dvc repro -s tmaze_experiment_marginal
```

All experiment parameters are centralized in `params.yaml`. CLI arguments override `params.yaml` values.

## Architecture

### Two VFE Factorization Strategies

**Factorized VFE** (`src/objectives/factorized_vfe.py`) — used by T-Maze only:
- Exhaustive enumeration of all state/action sequences
- `q(y,x,u,θ) = q(x|u) q(y,θ|x) q(u)` — 180K parameters for T=4
- Optimizer: `src/planning/factorized_optimizer.py`

**Temporal VFE** (`src/objectives/temporal_vfe.py`) — used by Epistemic Maze and MiniGrid:
- Markovian (Bethe) factorization, scales linearly with planning horizon
- `q(x₀:T, u₁:T, y₁:T, θ) = q(θ) · ∏ q(uₜ|xₜ₋₁) · q(xₜ|xₜ₋₁,uₜ,θ) · q(yₜ|xₜ,θ)`
- θ-independent policy: `q(u,θ) = q(u)q(θ)`
- Optimizer: `src/planning/temporal_optimizer.py`

### Inference Modes

Each mode adds different terms to the VFE objective:
- **marginal**: Standard VFE only
- **active**: VFE + epistemic priors (state uncertainty, observation informativeness, belief updates)
- **planning**: VFE + entropy correction for planning-as-inference

### Module Relationships

Environments (`src/environments/`) provide transition/observation tensors → Objectives (`src/objectives/`) compute JIT-compiled VFE loss → Planners (`src/planning/`) run Adam optimization via Optax → Experiment scripts (`scripts/`) orchestrate the agent-environment loop.

### Key Dependencies

- **JAX** + **Optax**: all numerical computation and optimization are JIT-compiled
- **pymdp**: custom git pinned fork (`inferactively-pymdp==0.0.7.1`) for sophisticated planning baseline
- **DVC**: experiment reproducibility pipeline (`dvc.yaml`)

### Conventions

- Categorical distributions are stored as JAX arrays in log-space or probability simplex
- Entropy/KL utilities live in `src/distributions/entropy.py`
- Color scheme for plots defined in `params.yaml` under `colors` (ColorBrewer Paired palette)
- Results output to `data/<environment>/<mode>/`
