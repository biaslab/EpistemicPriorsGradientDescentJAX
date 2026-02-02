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

The joint distribution is **factorized** as:

```
q(y_{1:T}, x_{1:T}, u_{1:T}, θ) = q(x|u) q(y,θ|x) q(u)
```

Where:
- `q(u)`: distribution over action sequences — shape `(n_actions^T,)` = `(256,)` for T=4
- `q(x|u)`: states given actions — shape `(n_states^T, n_actions^T)` = `(625, 256)`
- `q(y,θ|x)`: observations and reward location given states — shape `(n_obs^T, n_theta, n_states^T)` = `(16, 2, 625)`

**Total parameters for T=4**: 256 + 160,000 + 20,000 = **180,256 parameters**, optimized via Adam.

Entropy decomposes as:
```
H[q] = H[q(u)] + E_{q(u)}[H[q(x|u)]] + E_{q(x)}[H[q(y,θ|x)]]
```

## Inference Modes

Three modes implementing different forms of epistemic priors:

| Mode | VFE Components |
|------|----------------|
| **marginal** | Standard VFE: `-H[q] + E_q[-log p(u,x,y,θ,goal)]` |
| **active** | Standard VFE + **Epistemic Priors**:<br>• `p̃(u) ∝ exp(H[q(x\|u)])` — prefer uncertain state outcomes<br>• `p̃(x) ∝ exp(-H[q(y\|x)])` — prefer informative observations<br>• `p̃(y,x) ∝ exp(KL[q(θ\|y,x) ‖ q(θ\|x)])` — prefer belief updates |
| **planning** | Standard VFE + **Entropy Correction**: `∑_t H[q(x_{t-1}, u_t)] - H[q(x_{t-1})]`<br>Corrects for planning-as-inference structure |

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
   marginal | active | planning (default: marginal)
--n-episodes           Number of episodes (default: from params.yaml: 100)
--max-steps            Max steps per episode (default: from params.yaml: 4)
--planning-horizon     Planning horizon T (default: from params.yaml: 4)
--n-opt-steps          Optimization steps per planning call (default: from params.yaml: 5000)
--learning-rate        Adam learning rate (default: from params.yaml: 0.05)
--seed                 Random seed (default: from params.yaml: 18)
--verbose, -v          Print per-step details
--output-dir           Output directory (default: data/<inference-mode>)
--no-video             Skip video generation
--no-tikz              Skip TikZ frame generation
--no-receding-horizon  Use fixed horizon instead of receding
```

Note: CLI arguments override values from `params.yaml`.o-video          Skip video generation
--no-receding-horizon  Use fixed horizon instead of receding
```
are organized by inference mode in `data/<inference-mode>/`:

```
data/
├── marginal/
│   ├── results.json           # Full episode data and trajectories
│   ├── stats.json             # Summary statistics (mean reward, success rate, etc.)
│   ├── episode.mp4            # Video of last episode with planning visualization
│   └── frames/                # TikZ frames for LaTeX (frame_XX.tex, frame_XX_arrows.tex)
├── active/
│   └── ... (same structure)
├── planning/
│   └── ... (same structure)
├── convergence/               # Convergence analysis results
├── tmaze.tex                  # Reference T-maze diagram for papers
└── colors.tex                 # Color scheme definitions
```

### DVC Pipeline

The project uses DVC for reproducible experiments:

```bash
# Run all three inference modes
dvc repro

# Run specific stage
dvc repro -s marginal_experiment
dvc repro -s active_experiment
dvc repro -s planning_experiment
```
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
