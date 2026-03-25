#!/usr/bin/env python3
"""
Epistemic diagnostics for the MiniGrid DoorKey environment.

Prints per-step epistemic state preferences as ASCII grids:
- FOV informativeness: u~(x) ~ exp(-E_θ[H[q(y_fov|x,θ)]])
- FOV information gain: u~(x) ~ exp(E_y[KL[q(theta|y,x) || q(theta|x)]])
- Reward informativeness: u~(x) ~ exp(-E_θ[H[q(y_reward|x,θ)]])
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
from dataclasses import dataclass, replace
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.minigrid import (
    ActionType,
    Orientation,
    CellType,
    N_CELL_TYPES,
    N_ORIENTATIONS,
    N_ACTIONS,
    N_DOOR_KEY_STATES,
    state_to_coords,
    unflatten_state_index,
    key_position,
    door_position,
    create_minigrid_env_tensors,
)
from src.environments.gym_wrapper import MiniGridWrapper, StepResult
from src.agents.temporal_vfe_agent import create_temporal_vfe_agent, TemporalVFEAgent
from src.planning.temporal_optimizer_minigrid import (
    TemporalPlanningConfig,
    plan_actions_temporal,
)
from src.objectives.temporal_vfe import (
    compute_forward_marginals,
    compute_epistemic_priors,
    ModalityGroup,
)

EPS = 1e-8

ACTION_NAMES = [a.name for a in ActionType]

ORIENTATION_ARROWS = {
    Orientation.RIGHT: "->",
    Orientation.DOWN: "v",
    Orientation.LEFT: "<-",
    Orientation.UP: "^",
}

DKS_NAMES = ["no_key", "has_key", "door_open"]


@dataclass
class DiagnosticConfig:
    grid_size: int = 3
    fov_size: int = 3
    planning_horizon: int = 5
    max_steps: int = 10
    n_optimization_steps: int = 500
    learning_rate: float = 0.01
    seed: int = 42
    inference_mode: str = "active"
    freeze_obs_and_transitions: bool = False
    policy_init_scale: float = 1.0
    goal_scale: float = 1.0


def decode_theta(theta_idx: int, n: int, valid_configs: list[tuple[int, int]]) -> str:
    """Decode theta index to human-readable (key_x, key_y, door_x, door_y)."""
    key_pos, door_pos = valid_configs[theta_idx]
    kx, ky = key_position(key_pos, n)
    dx, dy = door_position(door_pos, n)
    return f"key=({kx},{ky}) door=({dx},{dy})"


def decode_state(flat_idx: int, n: int) -> str:
    """Decode flat state index to human-readable string."""
    n_states = n * n
    loc, orient, dks = unflatten_state_index(flat_idx, n_states, N_ORIENTATIONS, N_DOOR_KEY_STATES)
    x, y = state_to_coords(loc, n)
    return f"({x},{y}) {ORIENTATION_ARROWS.get(orient, '?')} {DKS_NAMES[dks]}"


def marginalize_to_grid(state_probs: jnp.ndarray, n: int) -> np.ndarray:
    """Marginalize (n_total_states,) over orientation and door_key_state -> (n, n) grid."""
    n_locations = n * n
    grid = np.zeros((n, n))
    probs = np.asarray(state_probs)
    for flat_idx in range(len(probs)):
        loc, _, _ = unflatten_state_index(flat_idx, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
        x, y = state_to_coords(loc, n)
        grid[y, x] += probs[flat_idx]  # y=row, x=col
    return grid


def print_grid(grid: np.ndarray, n: int, label: str):
    """Print (n, n) probability grid as ASCII."""
    print(f"\n  {label}:")
    header = "    " + "  ".join(f" {c}" for c in range(n))
    print(header)
    for row in range(n):
        cells = "  ".join(f"{grid[row, col]:.2f}" for col in range(n))
        print(f"  {row} {cells}")


def compute_fov_epistemic_drives(
    fov_gen_tensor: jnp.ndarray,      # (n_patterns, n_states, n_theta) - collapsed FOV
    reward_gen_tensor: jnp.ndarray,   # (2, n_states, n_theta)
    theta_belief: jnp.ndarray,        # (n_theta,)
    state_belief: jnp.ndarray = None, # (n_states,) — if provided, q(θ|x) = q(x,θ)/q(x)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute epistemic state priors from generative model + current theta belief.

    Args:
        fov_gen_tensor: (n_patterns, n_states, n_theta) collapsed FOV generative tensor
        reward_gen_tensor: (2, n_states, n_theta)
        theta_belief: (n_theta,)
        state_belief: (n_states,) — used to form q(θ|x) for info gain.
            When agent stores factored beliefs, q(x,θ)=q(x)q(θ) so q(θ|x)=q(θ).

    Returns:
        fov_informativeness: (n_states,) - FOV state prior
        fov_info_gain: (n_states,) - FOV information gain prior
        reward_informativeness: (n_states,) - reward state prior
    """
    theta_belief = jnp.clip(theta_belief, EPS, 1.0)
    theta_belief = theta_belief / jnp.sum(theta_belief)

    # q(θ|x) from joint q(x,θ) = q(x) q(θ) (factored agent beliefs)
    if state_belief is not None:
        state_belief = jnp.clip(state_belief, EPS, 1.0)
        state_belief = state_belief / jnp.sum(state_belief)
        q_x_theta = state_belief[:, None] * theta_belief[None, :]  # (n_states, n_theta)
        q_x = jnp.sum(q_x_theta, axis=1)  # (n_states,)
        q_theta_given_x = q_x_theta / (q_x[:, None] + EPS)  # (n_states, n_theta)
    else:
        # Fallback: assume q(θ|x) = q(θ)
        n_states = fov_gen_tensor.shape[1]
        q_theta_given_x = jnp.broadcast_to(theta_belief[None, :], (n_states, theta_belief.shape[0]))

    # ---- 1. FOV informativeness: u~(x) ~ exp(-E_θ[H[q(y|x,θ)]]) ----
    # E_θ[H[q(y|x,θ)]] = sum_theta q(theta) H[q(y|x,theta)]
    fov_safe = jnp.clip(fov_gen_tensor, EPS, 1.0)
    h_fov_per_theta = -jnp.sum(fov_safe * jnp.log(fov_safe), axis=0)  # (n_states, n_theta)
    h_fov = jnp.sum(h_fov_per_theta * theta_belief[None, :], axis=1)  # (n_states,)
    fov_informativeness = jax.nn.softmax(-h_fov)

    # ---- 2. FOV information gain: u~(x) ~ exp(E_y[KL[q(theta|y,x) || q(theta|x)]]) ----
    # q(θ|y,x) ∝ q(y|x,θ) q(θ|x)
    q_theta_yx_unnorm = fov_gen_tensor * q_theta_given_x[None, :, :]  # (n_patterns, n_states, n_theta)
    q_theta_yx_norm = jnp.sum(q_theta_yx_unnorm, axis=2, keepdims=True)
    q_theta_given_yx = q_theta_yx_unnorm / (q_theta_yx_norm + EPS)

    # KL[q(θ|y,x) || q(θ|x)]
    log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(q_theta_given_x[None, :, :] + EPS)
    kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=2)  # (n_patterns, n_states)

    # q(y|x) = Σ_θ q(y|x,θ) q(θ|x)
    q_y_x = jnp.sum(fov_gen_tensor * q_theta_given_x[None, :, :], axis=2)  # (n_patterns, n_states)
    expected_ig = jnp.sum(q_y_x * kl_yx, axis=0)  # (n_states,)
    fov_info_gain = jax.nn.softmax(expected_ig)

    # ---- 3. Reward informativeness: u~(x) ~ exp(-E_θ[H[q(y_r|x,θ)]]) ----
    reward_safe = jnp.clip(reward_gen_tensor, EPS, 1.0)
    h_reward_per_theta = -jnp.sum(reward_safe * jnp.log(reward_safe), axis=0)  # (n_states, n_theta)
    h_reward = jnp.sum(h_reward_per_theta * theta_belief[None, :], axis=1)  # (n_states,)
    reward_informativeness = jax.nn.softmax(-h_reward)

    return fov_informativeness, fov_info_gain, reward_informativeness


def print_theta_belief(theta_belief: jnp.ndarray, n: int, valid_configs: list[tuple[int, int]], top_k: int = 5):
    """Print top-k theta hypotheses decoded to key/door positions."""
    probs = np.asarray(theta_belief)
    top_indices = np.argsort(probs)[-top_k:][::-1]
    for idx in top_indices:
        desc = decode_theta(int(idx), n, valid_configs)
        print(f"    theta={idx:3d} ({desc}): {probs[idx]:.4f}")


def compute_goal_prior(goal_mapping: jnp.ndarray, theta_belief: jnp.ndarray, goal_scale: float) -> jnp.ndarray:
    """Effective goal state prior: softmax(goal_scale * E_theta[log p(goal|x,theta)])."""
    log_goal = jnp.log(jnp.clip(goal_mapping, EPS, 1.0))
    weighted_log_goal = goal_scale * jnp.sum(log_goal * theta_belief[None, :], axis=1)  # (n_states,)
    return jax.nn.softmax(weighted_log_goal)


def compute_vfe_decomposition(plan_result, env_tensors, effective_horizon, goal_scale, inference_mode):
    """Decompose VFE into individual terms for the frozen path.

    Returns dict with each term value and derived metrics.
    """
    q_theta = plan_result.q_theta
    q_u_given_x = plan_result.q_u_given_x
    q_x_theta = plan_result.q_x_theta

    # --- Entropy terms (frozen path: only theta + policy) ---
    q_theta_safe = jnp.clip(q_theta, EPS, 1.0)
    h_theta = -jnp.sum(q_theta_safe * jnp.log(q_theta_safe))

    policy_safe = jnp.clip(q_u_given_x, EPS, 1.0)
    h_u = -jnp.sum(policy_safe * jnp.log(policy_safe), axis=2)  # (horizon, n_states)
    q_x_prev = jnp.sum(q_x_theta[:-1], axis=2)  # (horizon, n_states)
    policy_entropy = jnp.sum(q_x_prev * h_u)

    # --- Energy terms ---
    theta_prior_safe = jnp.clip(env_tensors.theta_prior, EPS, 1.0)
    theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior_safe))

    action_prior_safe = jnp.clip(env_tensors.action_prior, EPS, 1.0)
    q_u_theta = jnp.einsum('hsa,hst->hat', q_u_given_x, q_x_theta[:-1])
    q_u_all = jnp.sum(q_u_theta, axis=2)
    action_energy = -jnp.sum(q_u_all * jnp.log(action_prior_safe)[None, :])

    goal_mapping_safe = jnp.clip(env_tensors.goal_mapping, EPS, 1.0)
    goal_energy_unscaled = -jnp.sum(q_x_theta[effective_horizon] * jnp.log(goal_mapping_safe))
    goal_energy = goal_energy_unscaled * goal_scale

    # --- Epistemic priors ---
    epistemic_energy = 0.0
    if inference_mode == "active":
        planning_mods = env_tensors.planning_modalities
        buckets = {}
        for mod in planning_mods:
            buckets.setdefault(mod.n_obs, []).append(mod)
        modality_groups = []
        for n_obs_val, mods in buckets.items():
            gen_batch = jnp.stack([m.generative_tensor for m in mods])
            log_gen_batch = jnp.log(jnp.clip(gen_batch, EPS, 1.0))
            modality_groups.append(ModalityGroup(
                n_obs=n_obs_val,
                q_obs_batch=gen_batch[:, None],  # frozen: q_obs = p_obs
                gen_batch=gen_batch,
                log_gen_batch=log_gen_batch,
            ))

        use_transition_index = env_tensors.transition_index is not None
        # Recompute forward marginals with stopped theta for epistemic priors
        q_x_theta_ep, _, _ = compute_forward_marginals(
            q_theta=jax.lax.stop_gradient(q_theta),
            q_u_given_x=q_u_given_x,
            q_x_given_xu_theta=None,
            initial_state=jnp.sum(q_x_theta[0], axis=1),  # initial state from q_x_theta
            horizon=effective_horizon,
            constant_transitions=True,
            transition_index=env_tensors.transition_index if use_transition_index else None,
            use_transition_index=use_transition_index,
            frozen=True,
        )

        epistemic_energy = compute_epistemic_priors(
            q_theta=q_theta,
            q_x_theta=q_x_theta_ep,
            modality_groups=modality_groups,
            horizon=effective_horizon,
            constant_obs=True,
        )

    return {
        'h_theta': float(h_theta),
        'policy_entropy': float(policy_entropy),
        'theta_energy': float(theta_energy),
        'action_energy': float(action_energy),
        'goal_energy': float(goal_energy),
        'goal_energy_unscaled': float(goal_energy_unscaled),
        'goal_scale': goal_scale,
        'epistemic_energy': float(epistemic_energy),
        'total_vfe': float(-h_theta - policy_entropy + theta_energy + action_energy + goal_energy + float(epistemic_energy)),
    }


def print_vfe_decomposition(decomp):
    """Print VFE decomposition table."""
    print(f"\n--- VFE DECOMPOSITION ---")
    print(f"  -H[q(theta)]:             {-decomp['h_theta']:8.2f}")
    print(f"  -H[q(u|x)]:               {-decomp['policy_entropy']:8.2f}")
    print(f"  E[-log p(theta)]:          {decomp['theta_energy']:8.2f}")
    print(f"  E[-log p(a)]:              {decomp['action_energy']:8.2f}")
    print(f"  E[-log goal] * scale:      {decomp['goal_energy']:8.2f}  (unscaled: {decomp['goal_energy_unscaled']:.2f}, scale: {decomp['goal_scale']:.1f})")
    if decomp['epistemic_energy'] != 0.0:
        print(f"  Epistemic priors:          {decomp['epistemic_energy']:8.2f}")
    print(f"  {'─' * 37}")
    print(f"  Total VFE:                 {decomp['total_vfe']:8.2f}")

    if decomp['epistemic_energy'] != 0.0:
        ratio = abs(decomp['goal_energy']) / max(abs(decomp['epistemic_energy']), EPS)
        label = "goal-dominant" if ratio > 1 else "exploration-dominant"
        print(f"\n  Goal/Epistemic ratio:      {ratio:8.2f}  (>1 = goal-dominant, <1 = exploration-dominant)")
        print(f"  => {label}")


def run_goal_scale_sweep(config: DiagnosticConfig, sweep_scales: list[float]):
    """Run planning with multiple goal_scale values on step 0 and compare."""
    n = config.grid_size
    fov_size = config.fov_size

    env_tensors = create_minigrid_env_tensors(n=n, fov_size=fov_size)
    fov_pattern_map = env_tensors.metadata.get("fov_pattern_map", {})

    # Create agent to get initial state
    agent = create_temporal_vfe_agent(
        env_tensors=env_tensors,
        planning_horizon=config.planning_horizon,
        n_optimization_steps=config.n_optimization_steps,
        learning_rate=config.learning_rate,
        inference_mode=config.inference_mode,
        init_seed=config.seed,
        fov_size=fov_size,
        fov_pattern_map=fov_pattern_map,
        freeze_obs_and_transitions=config.freeze_obs_and_transitions,
        policy_init_scale=config.policy_init_scale,
        goal_scale=config.goal_scale,
    )
    agent = agent.reset()
    initial_state = agent.state_belief
    theta_logits = agent.theta_logits

    print(f"\n{'='*80}")
    print(f"GOAL SCALE SWEEP")
    print(f"{'='*80}")
    print(f"Grid: {n}x{n}, Inference: {config.inference_mode}, Horizon: {config.planning_horizon}")
    print(f"Scales: {sweep_scales}")
    print()

    header = f"{'goal_scale':>10s} | {'VFE':>8s} | {'Goal E':>8s} | {'Epistemic':>9s} | {'G/E ratio':>9s} | {'Expl.Score':>10s} | {'Best Action':>12s}"
    print(header)
    print("-" * len(header))

    effective_horizon = min(config.max_steps, config.planning_horizon)

    for gs in sweep_scales:
        step_config = TemporalPlanningConfig(
            planning_horizon=effective_horizon,
            n_states=env_tensors.n_states,
            n_actions=env_tensors.n_actions,
            n_theta=env_tensors.n_theta,
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            inference_mode=config.inference_mode,
            init_seed=config.seed,
            freeze_obs_and_transitions=config.freeze_obs_and_transitions,
            policy_init_scale=config.policy_init_scale,
            goal_scale=gs,
        )
        plan_result = plan_actions_temporal(
            initial_state=initial_state,
            env_tensors=env_tensors,
            config=step_config,
            prior_theta_logits=theta_logits,
        )

        decomp = compute_vfe_decomposition(
            plan_result, env_tensors, effective_horizon, gs, config.inference_mode,
        )

        # Exploration score: normalized first-action entropy
        q_fa = plan_result.q_first_action
        q_fa_safe = jnp.clip(q_fa, EPS, 1.0)
        fa_entropy = -float(jnp.sum(q_fa_safe * jnp.log(q_fa_safe)))
        max_entropy = float(jnp.log(jnp.array(env_tensors.n_actions, dtype=jnp.float32)))
        expl_score = fa_entropy / max_entropy if max_entropy > 0 else 0.0

        best_action = ACTION_NAMES[int(jnp.argmax(q_fa))]

        ge_ratio = abs(decomp['goal_energy']) / max(abs(decomp['epistemic_energy']), EPS)
        ep_str = f"{decomp['epistemic_energy']:9.2f}" if decomp['epistemic_energy'] != 0.0 else "      N/A"
        ratio_str = f"{ge_ratio:9.2f}" if decomp['epistemic_energy'] != 0.0 else "      N/A"

        print(f"{gs:10.2f} | {decomp['total_vfe']:8.2f} | {decomp['goal_energy']:8.2f} | {ep_str} | {ratio_str} | {expl_score:10.2f} | {best_action:>12s}")


def run_episode_with_diagnostics(config: DiagnosticConfig):
    """Run a single episode with detailed epistemic diagnostics."""
    n = config.grid_size
    fov_size = config.fov_size

    # Create environment tensors
    env_tensors = create_minigrid_env_tensors(
        n=n, fov_size=fov_size,
    )

    # Extract modality tensors for diagnostics
    fov_gen_tensor = None
    reward_gen_tensor = None
    orient_gen_tensor = None
    for mod in env_tensors.observation_modalities:
        if mod.name == "fov":
            fov_gen_tensor = mod.generative_tensor
        elif mod.name == "reward":
            reward_gen_tensor = mod.generative_tensor
        elif mod.name == "orientation":
            orient_gen_tensor = mod.generative_tensor

    fov_pattern_map = env_tensors.metadata.get("fov_pattern_map", {})
    valid_configs = env_tensors.metadata.get("valid_configs", [])

    # Create agent
    agent = create_temporal_vfe_agent(
        env_tensors=env_tensors,
        planning_horizon=config.planning_horizon,
        n_optimization_steps=config.n_optimization_steps,
        learning_rate=config.learning_rate,
        inference_mode=config.inference_mode,
        init_seed=config.seed,
        fov_size=fov_size,
        fov_pattern_map=fov_pattern_map,
        freeze_obs_and_transitions=config.freeze_obs_and_transitions,
        policy_init_scale=config.policy_init_scale,
        goal_scale=config.goal_scale,
    )

    # Create gymnasium environment
    gym_size = n + 2  # MiniGrid adds walls
    env_name = f"MiniGrid-DoorKey-{gym_size}x{gym_size}-v0"
    env = MiniGridWrapper(
        env_name=env_name,
        render_mode=None,
        max_steps=config.max_steps,
        fov_size=fov_size,
    )

    # Reset
    step_result = env.reset(seed=config.seed)
    agent = agent.reset()

    # Get true environment state for display
    inner_env = env.env
    while hasattr(inner_env, "env"):
        inner_env = inner_env.env

    print(f"\n{'='*60}")
    print(f"MINIGRID DOORKEY EPISTEMIC DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Grid: {n}x{n} (gym: {gym_size}x{gym_size}), FOV: {fov_size}x{fov_size}")
    print(f"Inference mode: {config.inference_mode}")
    print(f"n_states: {env_tensors.n_states}, n_theta: {env_tensors.n_theta}, n_actions: {env_tensors.n_actions}")
    print(f"Horizon: {config.planning_horizon}, Opt steps: {config.n_optimization_steps}")
    print(f"Freeze obs/trans: {config.freeze_obs_and_transitions}")
    print(f"Policy init scale: {config.policy_init_scale}, Goal scale: {config.goal_scale}")
    print(f"FOV patterns: {len(fov_pattern_map)}")

    for step in range(config.max_steps):
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")

        # Current agent position from gymnasium env
        try:
            agent_pos = inner_env.agent_pos
            agent_dir = inner_env.agent_dir
            print(f"True agent: pos=({agent_pos[0]-1},{agent_pos[1]-1}) orient={ORIENTATION_ARROWS.get(agent_dir, '?')}")
        except Exception:
            pass

        # Current beliefs
        theta_probs = jax.nn.softmax(agent.theta_logits)
        print(f"\nTop theta hypotheses:")
        print_theta_belief(theta_probs, n, valid_configs, top_k=5)

        # State belief as grid
        state_grid = marginalize_to_grid(agent.state_belief, n)
        print_grid(state_grid, n, "State belief q(x) marginalized to grid")

        # Compute epistemic drives
        fov_info, fov_ig, reward_info = compute_fov_epistemic_drives(
            fov_gen_tensor, reward_gen_tensor, theta_probs,
            state_belief=agent.state_belief,
        )

        # Display as grids
        fov_info_grid = marginalize_to_grid(fov_info, n)
        print_grid(fov_info_grid, n, "FOV informativeness u~(x) ~ exp(-E_θ[H[q(y|x,θ)]])")

        fov_ig_grid = marginalize_to_grid(fov_ig, n)
        print_grid(fov_ig_grid, n, "FOV info gain u~(x) ~ exp(E[KL[q(theta|y,x)||q(theta|x)]])")

        reward_info_grid = marginalize_to_grid(reward_info, n)
        print_grid(reward_info_grid, n, "Reward informativeness u~(x) ~ exp(-E_θ[H[q(y_r|x,θ)]])")

        combined_fov = fov_info * fov_ig
        combined_fov = combined_fov / jnp.sum(combined_fov)
        combined_fov_grid = marginalize_to_grid(combined_fov, n)
        print_grid(combined_fov_grid, n, "Combined FOV prior (informativeness * info gain)")

        # Goal prior grid
        goal_prob = compute_goal_prior(env_tensors.goal_mapping, theta_probs, config.goal_scale)
        goal_grid = marginalize_to_grid(goal_prob, n)
        print_grid(goal_grid, n, "Effective goal prior softmax(scale * E_theta[log p(goal|x,theta)])")

        # --- Belief update from observation ---
        # Use collapsed FOV pattern lookup
        fov_pattern = tuple(int(x) for x in jnp.argmax(step_result.vision_obs, axis=-1).reshape(-1))
        orient_obs = jnp.argmax(step_result.orientation_obs)

        pattern_idx = fov_pattern_map.get(fov_pattern)
        if pattern_idx is not None:
            log_fov = jnp.log(jnp.clip(fov_gen_tensor[pattern_idx, :, :], EPS, 1.0))
            print(f"\n  FOV pattern matched: idx={pattern_idx}")
        else:
            log_fov = jnp.zeros((env_tensors.n_states, env_tensors.n_theta))
            print(f"\n  FOV pattern NOT matched (uniform likelihood)")

        log_orient = jnp.log(jnp.clip(orient_gen_tensor[orient_obs, :], EPS, 1.0))

        log_joint = (
            jnp.log(jnp.clip(agent.state_belief, EPS, 1.0))[:, None]
            + agent.theta_logits[None, :]
            + log_fov
            + log_orient[:, None]
        )
        log_joint = log_joint - jax.scipy.special.logsumexp(log_joint)
        joint = jnp.exp(log_joint)

        new_state_belief = jnp.sum(joint, axis=1)
        new_state_belief = new_state_belief / jnp.sum(new_state_belief)
        new_theta_probs = jnp.sum(joint, axis=0)
        new_theta_probs = new_theta_probs / jnp.sum(new_theta_probs)
        new_theta_logits = jnp.log(jnp.clip(new_theta_probs, EPS, 1.0))

        print(f"\nAfter observation update:")
        print(f"  Top theta hypotheses:")
        print_theta_belief(new_theta_probs, n, valid_configs, top_k=3)
        updated_grid = marginalize_to_grid(new_state_belief, n)
        print_grid(updated_grid, n, "Updated state belief")

        # --- Planning ---
        effective_horizon = min(config.max_steps - step, config.planning_horizon)
        effective_horizon = max(effective_horizon, 1)

        step_config = TemporalPlanningConfig(
            planning_horizon=effective_horizon,
            n_states=env_tensors.n_states,
            n_actions=env_tensors.n_actions,
            n_theta=env_tensors.n_theta,
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            inference_mode=config.inference_mode,
            init_seed=config.seed + step,
            freeze_obs_and_transitions=config.freeze_obs_and_transitions,
            policy_init_scale=config.policy_init_scale,
            goal_scale=config.goal_scale,
        )
        plan_result = plan_actions_temporal(
            initial_state=new_state_belief,
            env_tensors=env_tensors,
            config=step_config,
            prior_theta_logits=new_theta_logits,
        )

        # Print planned state trajectories as grids
        print(f"\n--- PLANNED STATE TRAJECTORY ---")
        for t in range(min(effective_horizon, 3)):  # Show first 3 timesteps
            q_x_t = jnp.sum(plan_result.q_x_theta[t + 1] * plan_result.q_theta[None, :], axis=1)
            traj_grid = marginalize_to_grid(q_x_t, n)
            print_grid(traj_grid, n, f"q(x_{{t+{t+1}}}) planned state")

        # Action distribution
        print(f"\n--- ACTION SELECTION ---")
        sorted_actions = jnp.argsort(plan_result.q_first_action)[::-1]
        for a_idx in sorted_actions[:5]:
            prob = float(plan_result.q_first_action[a_idx])
            if prob > 0.01:
                print(f"  {ACTION_NAMES[int(a_idx)]:12s}: {prob:.4f}")

        action = int(jnp.argmax(plan_result.q_first_action))
        print(f"\nSelected: {ACTION_NAMES[action]}")
        print(f"Final loss: {plan_result.final_loss:.4f}")

        # Exploration indicators
        q_fa = plan_result.q_first_action
        q_fa_safe = jnp.clip(q_fa, EPS, 1.0)
        fa_entropy = -float(jnp.sum(q_fa_safe * jnp.log(q_fa_safe)))
        max_entropy = float(jnp.log(jnp.array(env_tensors.n_actions, dtype=jnp.float32)))
        expl_score = fa_entropy / max_entropy if max_entropy > 0 else 0.0
        print(f"  First-action entropy: {fa_entropy:.4f}  Exploration score: {expl_score:.2f} (0=deterministic, 1=uniform)")

        # Average policy entropy per step
        policy_safe = jnp.clip(plan_result.q_u_given_x, EPS, 1.0)
        h_per_step = -jnp.sum(policy_safe * jnp.log(policy_safe), axis=2)  # (H, n_states)
        q_x_prev = jnp.sum(plan_result.q_x_theta[:-1], axis=2)  # (H, n_states)
        avg_policy_h = float(jnp.sum(q_x_prev * h_per_step)) / effective_horizon
        print(f"  Avg policy entropy/step: {avg_policy_h:.4f}")

        # VFE decomposition (frozen path only)
        if config.freeze_obs_and_transitions:
            decomp = compute_vfe_decomposition(
                plan_result, env_tensors, effective_horizon, config.goal_scale, config.inference_mode,
            )
            print_vfe_decomposition(decomp)

        # Goal overlap grid: q(x_T) * goal_prob
        q_x_T = jnp.sum(plan_result.q_x_theta[effective_horizon] * plan_result.q_theta[None, :], axis=1)
        goal_overlap = q_x_T * goal_prob
        goal_overlap_grid = marginalize_to_grid(goal_overlap, n)
        print_grid(goal_overlap_grid, n, f"Goal overlap: q(x_T) * goal_prior (sum={float(jnp.sum(goal_overlap)):.4f})")

        # Update agent state for next step (predict next state)
        if env_tensors.transition_index is not None:
            idx = env_tensors.transition_index[:, action, :]  # (n_states, n_theta)
            weights = new_state_belief[:, None] * new_theta_probs[None, :]
            predicted_state = jnp.zeros(env_tensors.n_states).at[idx.ravel()].add(weights.ravel())
        elif env_tensors.theta_dependent_transitions:
            T = env_tensors.transition_tensor
            T_action = T[:, :, :, action]
            predicted_state = jnp.einsum("ijk,j,k->i", T_action, new_state_belief, new_theta_probs)
        else:
            T = env_tensors.transition_tensor
            T_action = T[:, :, action]
            predicted_state = T_action @ new_state_belief
        predicted_state = predicted_state / jnp.sum(predicted_state)

        agent = replace(
            agent,
            state_belief=predicted_state,
            theta_logits=new_theta_logits,
            step_count=agent.step_count + 1,
        )

        # Step environment
        step_result = env.step(action)

        print(f"\n--- ENVIRONMENT ---")
        print(f"Reward: {step_result.reward}, Done: {step_result.terminated or step_result.truncated}")

        if step_result.terminated or step_result.truncated:
            print(f"\n{'='*60}")
            print(f"Episode complete! Reward: {step_result.reward}")
            if step_result.terminated and step_result.reward > 0:
                print("SUCCESS!")
            elif step_result.truncated:
                print("TRUNCATED (max steps reached)")
            print(f"{'='*60}")
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(description="MiniGrid DoorKey Epistemic Diagnostics")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--fov-size", type=int, default=3)
    parser.add_argument("--n-opt-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--inference-mode", type=str, default="active",
                        choices=["marginal", "active", "planning"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-obs-and-transitions", action="store_true")
    parser.add_argument("--policy-init-scale", type=float, default=1.0)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--sweep-goal-scales", type=str, default=None,
                        help="Comma-separated goal scales to sweep (e.g., 0.1,0.5,1.0,2.0,5.0)")

    args = parser.parse_args()
    config = DiagnosticConfig(
        grid_size=args.grid_size,
        fov_size=args.fov_size,
        n_optimization_steps=args.n_opt_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        inference_mode=args.inference_mode,
        planning_horizon=args.horizon,
        max_steps=args.max_steps,
        freeze_obs_and_transitions=args.freeze_obs_and_transitions,
        policy_init_scale=args.policy_init_scale,
        goal_scale=args.goal_scale,
    )

    if args.sweep_goal_scales:
        sweep_scales = [float(s.strip()) for s in args.sweep_goal_scales.split(",")]
        run_goal_scale_sweep(config, sweep_scales)
    else:
        run_episode_with_diagnostics(config)


if __name__ == "__main__":
    main()
