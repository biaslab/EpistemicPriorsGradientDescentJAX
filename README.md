# Epistemic Priors in Active Inference Planning (JAX)

Experiments comparing inference modes for Active Inference planning via variational free energy (VFE) minimization. Three environments of increasing complexity test whether epistemic priors are necessary for goal-directed exploration.

## Environments

### T-Maze

```
State 2 (left arm) --- State 3 (top) --- State 4 (right arm)
                           |
                       State 1 (junction)
                           |
                       State 0 (cue)
```

- Reward at State 2 or State 4 (randomized per episode)
- Agent starts at State 1; visiting State 0 reveals reward location
- Uses **factorized** VFE with exhaustive sequence enumeration

### Epistemic Maze

A larger grid maze with cue locations and hidden reward, requiring multi-step information gathering before committing to a goal.

- Uses **temporal** (Markovian/Bethe) VFE factorization

### MiniGrid DoorKey

A `MiniGrid-DoorKey` environment with partial observability (field-of-view), requiring the agent to find a key, unlock a door, and reach a goal.

- Uses **temporal** VFE factorization with a Gymnasium wrapper

## Inference Modes

Three modes implementing different forms of epistemic priors (`--inference-mode`):

| Mode | VFE Components |
|------|----------------|
| **marginal** | Standard VFE: `-H[q] + E_q[-log p(u,x,y,theta,goal)]` |
| **active** | Standard VFE + **Epistemic Priors** that encourage exploration |
| **planning** | Standard VFE + **Entropy Correction** for planning-as-inference |

Additionally, **sophisticated** and **vanilla** strategies (via `--strategy`) use pymdp's tree-search planner as baselines.

## Usage

```bash
# Install
uv sync

# Run T-Maze experiments
uv run python scripts/tmaze/experiment.py --inference-mode marginal
uv run python scripts/tmaze/experiment.py --inference-mode active
uv run python scripts/tmaze/experiment.py --inference-mode planning

# Run Epistemic Maze experiments
uv run python scripts/epistemic_maze/experiment.py --strategy temporal --inference-mode planning

# Run MiniGrid experiments
uv run python scripts/minigrid/experiment.py

# Run tests
uv run python -m pytest tests/ -v

# Run full DVC pipeline
dvc repro
```

Experiment parameters live in `params.yaml`; CLI args override them.

## Structure

```
scripts/
├── tmaze/
│   ├── experiment.py              # T-Maze experiment runner
│   ├── convergence_analysis.py    # Convergence and budget sweeps
│   ├── plot_convergence.py        # Generate convergence figures
│   └── diagnostics.py             # Diagnostic tools
├── epistemic_maze/
│   ├── experiment.py
│   ├── convergence_analysis.py
│   ├── plot_convergence.py
│   └── diagnostics.py
└── minigrid/
    ├── experiment.py
    ├── convergence.py
    ├── aggregate_episodes.py
    └── diagnostics.py
src/
├── environments/
│   ├── tmaze.py                   # T-Maze transition/observation tensors
│   ├── epistemic_maze.py          # Epistemic Maze environment
│   ├── minigrid.py                # MiniGrid wrapper
│   └── environment_protocol.py    # Shared environment protocol
├── objectives/
│   ├── factorized_vfe.py          # Factorized VFE (T-Maze)
│   └── temporal_vfe.py            # Temporal/Bethe VFE (Epistemic Maze, MiniGrid)
├── planning/
│   ├── factorized_optimizer.py    # Adam optimizer for factorized VFE
│   ├── temporal_optimizer.py      # Adam optimizer for temporal VFE
│   ├── temporal_optimizer_minigrid.py
│   └── sophisticated_planner.py   # pymdp tree-search baseline
├── distributions/
│   └── entropy.py                 # Entropy and KL utilities
└── visualization/
    └── tmaze_viz.py               # Video generation
data/                              # DVC-tracked results
├── tmaze/{marginal,active,planning,sophisticated,vanilla}/
├── epistemic_maze/{marginal,active,planning,sophisticated,vanilla}/
└── minigrid/
tests/
├── test_tmaze.py
├── test_temporal_vfe_regression.py
└── test_minigrid_fov.py
```

## DVC Pipeline

```bash
# Run all experiments and convergence analyses
dvc repro

# Run specific stages
dvc repro -s tmaze_experiment_marginal
dvc repro -s epistemic_planning
```

See `dvc.yaml` for the full pipeline definition.
