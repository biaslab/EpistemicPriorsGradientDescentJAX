# JAX Active Inference

Active Inference planning via variational optimization in JAX with **TRUE FULL JOINT** factorization.

## Overview

This package implements Active Inference planning by directly optimizing the **full joint** variational distribution:

```
q(x_{1:T}, u_{1:T}, r)
```

Where:
- `x_{1:T}`: State trajectory (determined by dynamics)
- `u_{1:T}`: Action sequence  
- `r`: Hidden reward/cue location

Since dynamics are deterministic, we parameterize `q(u_{1:T}, r)` with `4^T × 2` parameters. The full joint is then:

```
q(x, u, r) = δ(x = f(x_0, u)) × q(u, r)
```

**Why the full joint matters:**
- `x` ties `u` and `r` together through observations
- `q(x, r)` captures correlation between states and hidden reward location
- `q(r|x)` tells us how reward belief depends on state
- This is essential for computing the ambiguity term that drives cue-seeking

## Epistemic Priors

**Exploration** (prior on actions):
```
p̃(u) ∝ exp(H[q(x|u)])
```
With deterministic dynamics, this is 0. Information-seeking comes from ambiguity.

**Ambiguity** (prior on states):
```
p̃(x) ∝ exp(-H[q(y|x)])

where q(y|x) = Σ_r p(y|x,r) q(r|x)
```

This is the KEY term! At the cue state (x=0):
- `p(y|x=0, r)` perfectly distinguishes `r`
- So `H[q(y|x=0)]` depends on `q(r|x=0)`
- The ambiguity term encourages visiting states where observations are informative

This drives the agent toward the cue state, which in turn biases `q(u)` toward going South.

## Two Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Active Inference** | (default) | VFE + epistemic priors |
| **Marginal Inference** | `--marginal` | Standard KL only |

## Installation

```bash
cd jax_ai
uv sync
```

## Usage

```bash
# Active inference (with epistemic priors - drives cue-seeking)
uv run python scripts/tmaze_experiment.py --n-episodes 5 --verbose

# Marginal inference (no epistemic priors)
uv run python scripts/tmaze_experiment.py --n-episodes 5 --verbose --marginal
```

## Mathematical Details

### Full Joint VFE

```
F[q(u,r)] = -E_q[log p(u)]           # Action prior
          + KL[q(r) || p(r)]         # Reward location prior
          - E_q[log p(x_T | r)]      # Goal term
          - H[q(u, r)]               # Joint entropy
          + E_q(x)[H[q(y|x)]]        # Ambiguity (epistemic)
```

The ambiguity term is computed as:
```
E_q(x)[H[q(y|x)]] = Σ_t Σ_x q(x_t) H[Σ_r p(y|x_t,r) q(r|x_t)]
```

Where `q(x_t, r)` is marginalized from the full joint.

## Project Structure

```
jax_ai/
├── src/
│   ├── objectives/
│   │   └── full_joint_vfe.py    # TRUE full joint VFE
│   ├── planning/
│   │   └── full_joint_optimizer.py  # Optimizes q(u,r)
│   ├── environments/
│   │   └── tmaze.py
│   └── distributions/
└── scripts/
    └── tmaze_experiment.py
```
