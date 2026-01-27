# Epistemic Priors in T-Maze (JAX)

Experiments comparing inference modes for Active Inference planning in a T-Maze environment.

## Environment

T-shaped maze with 5 states:

```
State 2 (left arm) --- State 3 (top) --- State 4 (right arm)
                           |
                       State 1 (junction)
                           |
                       State 0 (cue)
```

- Reward at State 2 or State 4 (randomized per episode)
- Agent starts at State 1
- Visiting State 0 reveals reward location via observation

## Variational Parametrization

Full joint distribution `q(x_{1:T}, u_{1:T}, r)` parametrized as logits with shape:

```
(n_states^T, n_actions^T, n_reward_locs) = (5^T, 4^T, 2)
```

For horizon T=4: `(625, 256, 2)` parameters, optimized via Adam.

## Inference Modes

| Mode | Description |
|------|-------------|
| `marginal` | Standard VFE: entropy + action prior + transition + goal + reward prior |
| `active` | VFE + epistemic state prior (low observation entropy) + control prior (high transition entropy) |
| `planning` | VFE + planning entropy correction: `∑_t H[q(x_{t-1}, u_t)] - H[q(x_{t-1})]` |

## Usage

```bash
# Install
uv sync

# Run experiments
uv run python scripts/tmaze_experiment.py --inference-mode marginal
uv run python scripts/tmaze_experiment.py --inference-mode active
uv run python scripts/tmaze_experiment.py --inference-mode planning
```

### Options

```
--inference-mode    marginal | active | planning (default: marginal)
--n-episodes        Number of episodes (default: 50)
--max-steps         Max steps per episode (default: 4)
--planning-horizon  Planning horizon T (default: 4)
--n-opt-steps       Optimization steps per planning call (default: 100)
--learning-rate     Adam learning rate (default: 0.1)
--seed              Random seed (default: 42)
--verbose, -v       Print per-step details
--output-dir        Output directory (default: data)
--no-video          Skip video generation
--no-receding-horizon  Use fixed horizon instead of receding
```

## Output

Results saved to `data/`:
- `results_{mode}_{timestamp}.json` — episode data and summary statistics
- `episode_{mode}_{timestamp}.mp4` — video of last episode

## Structure

```
├── scripts/
│   ├── tmaze_experiment.py    # Main experiment script
│   └── diagnostics.py         # Diagnostic tools for tuning
├── src/
│   ├── environments/
│   │   └── tmaze.py           # T-Maze environment and tensors
│   ├── objectives/
│   │   └── factorized_vfe.py  # VFE with factorized q(x|u)q(y,θ|x)q(u)
│   ├── planning/
│   │   └── factorized_optimizer.py  # Adam optimization
│   └── visualization/
│       └── tmaze_viz.py       # Video generation
└── tests/
    └── test_tmaze.py
```
