# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Active Inference planning via variational free energy (VFE) minimization in JAX. Compares three inference modes across three environments (T-Maze, Epistemic Maze, MiniGrid) to study the role of epistemic priors in planning.

## Commands

Always use `uv` to run Python code (never bare `python`).

```bash
uv sync                                # install dependencies
uv run python -m pytest tests/ -v      # run tests
uv run python scripts/tmaze/experiment.py --inference-mode marginal
dvc repro                              # run full DVC pipeline
```

Experiment parameters live in `params.yaml`; CLI args override them.

## What Lives Where

- `src/environments/` — environment definitions (transition/observation tensors): `tmaze.py`, `epistemic_maze.py`, `minigrid.py`
- `src/objectives/` — VFE loss functions (JIT-compiled): `factorized_vfe.py` (T-Maze), `temporal_vfe.py` (Epistemic Maze, MiniGrid)
- `src/planning/` — Optax/Adam optimizers that minimize VFE: `factorized_optimizer.py`, `temporal_optimizer.py`
- `src/distributions/` — entropy and KL utilities
- `scripts/{tmaze,epistemic_maze,minigrid}/` — experiment runners, convergence analyses, diagnostics
- `data/` — DVC-tracked results, organized by `<environment>/<mode>/`

## Architecture

Data flow: Environments → Objectives → Planners → Scripts.

Two VFE factorizations:
- **Factorized** (`factorized_vfe.py` + `factorized_optimizer.py`): exhaustive sequence enumeration, T-Maze only.
- **Temporal** (`temporal_vfe.py` + `temporal_optimizer.py`): Markovian/Bethe factorization, used by Epistemic Maze and MiniGrid.

Three inference modes (`--inference-mode`): `marginal` (standard VFE), `active` (+ epistemic priors), `planning` (+ entropy correction).

pymdp dependency is a pinned git fork (see `pyproject.toml` `[tool.uv.sources]`).
