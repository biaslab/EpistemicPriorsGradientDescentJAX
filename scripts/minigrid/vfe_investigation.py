#!/usr/bin/env python3
"""
Investigate why the VFE optimizer produces near-uniform policies for MiniGrid.

Decomposes the VFE into individual terms, checks gradient magnitudes,
verifies the generative model, and compares inference modes.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
import sys
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environments.minigrid import (
    create_minigrid_env_tensors,
    generate_transition_tensor,
    get_valid_static_configs,
    N_ORIENTATIONS,
    N_DOOR_KEY_STATES,
    N_ACTIONS,
    ActionType,
    state_to_coords,
    unflatten_state_index,
    flatten_state_index,
    coords_to_state,
)
from src.objectives.temporal_vfe import (
    temporal_vfe_jit,
    compute_forward_marginals,
    compute_entropy_terms,
    compute_energy_terms,
    compute_epistemic_priors,
    group_modalities_for_jit,
    ModalityGroup,
)
from src.planning.temporal_optimizer_minigrid import TemporalPlanningConfig, plan_actions_temporal

EPS = 1e-8
ACTION_NAMES = [a.name for a in ActionType]


def investigate_generative_model(env_tensors, n):
    """Check 1: Is the generative model sensible?"""
    print("\n" + "=" * 70)
    print("CHECK 1: GENERATIVE MODEL SANITY")
    print("=" * 70)

    n_locations = n * n
    T = env_tensors.transition_tensor  # (s_next, s_prev, theta, action)

    # Pick a specific theta and check reachability
    theta_idx = 0
    T_theta = T[:, :, theta_idx, :]  # (s_next, s_prev, action)

    # Start at state (0,0), orientation RIGHT, dks=no_key
    start_flat = flatten_state_index(0, 0, 0, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)

    # Goal at (n-1, n-1)
    goal_loc = coords_to_state(n - 1, n - 1, n)

    print(f"\nTransition tensor shape: {T.shape}")
    print(f"  (n_states_next={T.shape[0]}, n_states_prev={T.shape[1]}, n_theta={T.shape[2]}, n_actions={T.shape[3]})")

    # Check: from start state, what happens with each action?
    print(f"\nFrom state {start_flat} = (0,0) RIGHT no_key, θ={theta_idx}:")
    for a in range(N_ACTIONS):
        next_dist = T_theta[:, start_flat, a]
        top_idx = int(jnp.argmax(next_dist))
        top_prob = float(next_dist[top_idx])
        loc, orient, dks = unflatten_state_index(top_idx, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
        x, y = state_to_coords(loc, n)
        print(f"  {ACTION_NAMES[a]:12s} → state {top_idx} ({x},{y}) orient={orient} dks={dks}  p={top_prob:.3f}")

    # Check: is transition deterministic?
    nonzero_per_col = jnp.sum(T_theta > 0.01, axis=0)  # (s_prev, action)
    print(f"\nTransition sparsity (avg nonzero next-states per (s,a)): {float(jnp.mean(nonzero_per_col)):.2f}")
    print(f"  Max: {int(jnp.max(nonzero_per_col))}, Min: {int(jnp.min(nonzero_per_col))}")

    # BFS reachability from start
    reachable = set()
    frontier = {start_flat}
    depth = 0
    while frontier and depth < 20:
        reachable |= frontier
        next_frontier = set()
        for s in frontier:
            for a in range(N_ACTIONS):
                next_dist = np.asarray(T_theta[:, s, a])
                for s_next in np.where(next_dist > 0.01)[0]:
                    if s_next not in reachable:
                        next_frontier.add(int(s_next))
        frontier = next_frontier
        depth += 1

    # Check if any goal state is reachable
    goal_states = set()
    for orient in range(N_ORIENTATIONS):
        for dks in range(N_DOOR_KEY_STATES):
            gs = flatten_state_index(goal_loc, orient, dks, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
            goal_states.add(gs)

    reachable_goals = reachable & goal_states
    print(f"\nReachability (θ={theta_idx}, max depth=20):")
    print(f"  Total reachable states: {len(reachable)} / {env_tensors.n_states}")
    print(f"  Goal states reachable: {len(reachable_goals)} / {len(goal_states)}")
    if not reachable_goals:
        print("  *** WARNING: NO GOAL STATES REACHABLE! ***")

    # Goal mapping check
    goal_mapping = env_tensors.goal_mapping  # (n_states, n_theta)
    print(f"\nGoal mapping shape: {goal_mapping.shape}")
    goal_mass = float(jnp.sum(goal_mapping[goal_loc * N_ORIENTATIONS * N_DOOR_KEY_STATES:
                                            (goal_loc + 1) * N_ORIENTATIONS * N_DOOR_KEY_STATES, 0]))
    print(f"  Goal mass at location ({n-1},{n-1}): {goal_mass:.4f}")
    print(f"  Max goal prob: {float(jnp.max(goal_mapping[:, 0])):.4f}")
    print(f"  Min goal prob: {float(jnp.min(goal_mapping[:, 0])):.4f}")


def investigate_vfe_decomposition(env_tensors, n, horizon, n_opt_steps, lr, inference_mode, seed,
                                   gradient_scale_factor=1.0,
                                   freeze_obs_and_transitions=False, policy_init_scale=1.0,
                                   goal_scale=1.0):
    """Check 2: Decompose VFE into individual terms before and after optimization."""
    print("\n" + "=" * 70)
    print("CHECK 2: VFE TERM DECOMPOSITION")
    print("=" * 70)

    n_locations = n * n

    # Concentrated start state
    initial_state = jnp.zeros(env_tensors.n_states)
    start_flat = flatten_state_index(0, 0, 0, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
    initial_state = initial_state.at[start_flat].set(1.0)

    config = TemporalPlanningConfig(
        planning_horizon=horizon,
        n_states=env_tensors.n_states,
        n_actions=env_tensors.n_actions,
        n_theta=env_tensors.n_theta,
        n_optimization_steps=n_opt_steps,
        learning_rate=lr,
        inference_mode=inference_mode,
        init_seed=seed,
        gradient_scale_factor=gradient_scale_factor,
        freeze_obs_and_transitions=freeze_obs_and_transitions,
        policy_init_scale=policy_init_scale,
        goal_scale=goal_scale,
    )

    result = plan_actions_temporal(
        initial_state=initial_state,
        env_tensors=env_tensors,
        config=config,
    )

    # Compute individual terms from the optimized result
    from jax.nn import softmax

    q_theta = result.q_theta
    q_u_given_x = result.q_u_given_x
    q_x_given_xu_theta = result.q_x_given_xu_theta
    q_x_theta = result.q_x_theta
    q_u_theta = result.q_u_theta

    # Rebuild modality groups from optimized obs
    q_obs_list = result.q_obs
    obs_mods = env_tensors.observation_modalities
    modality_groups = []
    from src.objectives.temporal_vfe import group_modalities
    # We need to construct ModalityGroup from the optimized q_obs
    # Filter to planning modalities (theta-dependent only)
    planning_obs = [(q, m) for q, m in zip(q_obs_list, obs_mods) if m.theta_dependent]
    buckets = {}
    for q_obs, mod in planning_obs:
        key = mod.n_obs
        if key not in buckets:
            buckets[key] = {'q_obs': [], 'gen': []}
        buckets[key]['q_obs'].append(q_obs)
        buckets[key]['gen'].append(mod.generative_tensor)

    modality_groups = []
    for n_obs_val, data in buckets.items():
        q_obs_batch = jnp.stack(data['q_obs'])
        gen_batch = jnp.stack(data['gen'])
        log_gen_batch = jnp.log(jnp.clip(gen_batch, EPS, 1.0))
        modality_groups.append(ModalityGroup(
            n_obs=n_obs_val,
            q_obs_batch=q_obs_batch,
            gen_batch=gen_batch,
            log_gen_batch=log_gen_batch,
        ))

    # Rebuild xu_theta — policy is (H, S, A), broadcast over θ
    q_xu_theta_recon = q_u_given_x[:, :, None, :] * q_x_theta[:-1, :, :, None]  # (H, S, T, A)
    q_xu_theta_recon = jnp.transpose(q_xu_theta_recon, (0, 1, 3, 2))  # (H, S, A, T)

    # Entropy
    entropy = compute_entropy_terms(
        q_theta, q_u_given_x, q_x_given_xu_theta,
        modality_groups, q_x_theta, q_xu_theta_recon, horizon,
    )

    # Energy terms
    energies = compute_energy_terms(
        q_theta, q_x_given_xu_theta, modality_groups,
        q_x_theta, q_xu_theta_recon, q_u_theta,
        env_tensors.transition_tensor, env_tensors.goal_mapping,
        env_tensors.action_prior, env_tensors.theta_prior, horizon,
    )

    # Epistemic priors
    if inference_mode == "active":
        epistemic = compute_epistemic_priors(
            q_theta, q_x_theta, modality_groups, horizon,
        )
    else:
        epistemic = 0.0

    base_vfe = -float(entropy) + float(energies['action']) + float(energies['transition']) + float(energies['observation']) + float(energies['goal']) + float(energies['theta'])

    print(f"\nAfter {n_opt_steps} optimization steps (mode={inference_mode}):")
    print(f"  Final VFE: {result.final_loss:.2f}")
    print(f"  Reconstructed: {base_vfe + float(epistemic):.2f}")
    print(f"\n  Term decomposition:")
    print(f"    -H[q] (neg entropy):  {-float(entropy):10.2f}")
    print(f"    E[-log p(θ)]:         {float(energies['theta']):10.2f}")
    print(f"    E[-log p(a)]:         {float(energies['action']):10.2f}")
    print(f"    E[-log p(x'|x,u,θ)]: {float(energies['transition']):10.2f}")
    print(f"    E[-log p(y|x,θ)]:     {float(energies['observation']):10.2f}")
    print(f"    E[-log p(goal|xT,θ)]: {float(energies['goal']):10.2f}")
    if inference_mode == "active":
        print(f"    Epistemic priors:     {float(epistemic):10.2f}")
    print(f"    ─────────────────────────────────")
    print(f"    Total:                {base_vfe + float(epistemic):10.2f}")

    # Relative magnitudes
    total_energy = sum(float(v) for v in energies.values())
    print(f"\n  Energy breakdown (% of total energy {total_energy:.1f}):")
    for name, val in energies.items():
        pct = 100 * float(val) / total_energy if total_energy > 0 else 0
        print(f"    {name:15s}: {pct:5.1f}%")

    return result, initial_state


def investigate_gradient_magnitudes(env_tensors, n, horizon, inference_mode, seed):
    """Check 3: Gradient magnitudes on policy logits."""
    print("\n" + "=" * 70)
    print("CHECK 3: GRADIENT MAGNITUDES")
    print("=" * 70)

    n_locations = n * n
    initial_state = jnp.zeros(env_tensors.n_states)
    start_flat = flatten_state_index(0, 0, 0, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
    initial_state = initial_state.at[start_flat].set(1.0)

    # Setup params at initialization (before any optimization)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3 + len(env_tensors.observation_modalities))

    q_theta_logits = jnp.log(env_tensors.theta_prior + EPS)
    q_u_logits = jax.random.normal(keys[0], shape=(horizon, env_tensors.n_states, N_ACTIONS)) * 0.01

    log_transition = jnp.log(jnp.clip(env_tensors.transition_tensor, EPS, 1.0))
    log_trans_init = jnp.transpose(log_transition, (0, 1, 3, 2))
    q_x_logits = jax.random.normal(keys[1], shape=(horizon, env_tensors.n_states, env_tensors.n_states, N_ACTIONS, env_tensors.n_theta)) * 0.01 + log_trans_init

    planning_mods = [m for m in env_tensors.observation_modalities if m.theta_dependent]
    q_obs_logits_init = []
    for i, mod in enumerate(planning_mods):
        log_gen = jnp.log(jnp.clip(mod.generative_tensor, EPS, 1.0))
        noise = jax.random.normal(keys[3 + i], shape=(horizon, mod.n_obs, env_tensors.n_states, env_tensors.n_theta)) * 0.01
        q_obs_logits_init.append(noise + log_gen[None, :, :, :])

    _, gen_tensor_groups, log_gen_tensor_groups, modality_index_groups = \
        group_modalities_for_jit(q_obs_logits_init, planning_mods)

    q_obs_logits_groups = []
    for mod_indices in modality_index_groups:
        q_obs_logits_groups.append(jnp.stack([q_obs_logits_init[i] for i in mod_indices]))

    params = {
        'q_theta_logits': q_theta_logits,
        'q_u_given_x_logits': q_u_logits,
        'q_x_given_xu_theta_logits': q_x_logits,
        'q_obs_logits_groups': q_obs_logits_groups,
    }

    def loss_fn(params):
        return temporal_vfe_jit(
            q_theta_logits=params['q_theta_logits'],
            q_u_given_x_logits=params['q_u_given_x_logits'],
            q_x_given_xu_theta_logits=params['q_x_given_xu_theta_logits'],
            q_obs_logits_groups=params['q_obs_logits_groups'],
            initial_state=initial_state,
            transition_tensor=env_tensors.transition_tensor,
            gen_tensor_groups=gen_tensor_groups,
            log_gen_tensor_groups=log_gen_tensor_groups,
            goal_mapping=env_tensors.goal_mapping,
            action_prior=env_tensors.action_prior,
            theta_prior=env_tensors.theta_prior,
            horizon=horizon,
            inference_mode=inference_mode,
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)

    print(f"\nAt initialization (mode={inference_mode}):")
    print(f"  VFE loss: {float(loss):.2f}")
    print(f"\n  Gradient norms:")
    for key_name, grad_val in grads.items():
        if isinstance(grad_val, list):
            for i, g in enumerate(grad_val):
                print(f"    {key_name}[{i}]:  norm={float(jnp.linalg.norm(g)):.6f}  "
                      f"max={float(jnp.max(jnp.abs(g))):.6f}  shape={g.shape}")
        else:
            print(f"    {key_name}:  norm={float(jnp.linalg.norm(grad_val)):.6f}  "
                  f"max={float(jnp.max(jnp.abs(grad_val))):.6f}  shape={grad_val.shape}")

    # Focus on policy gradients at t=0
    policy_grad = grads['q_u_given_x_logits']
    print(f"\n  Policy gradient analysis (t=0):")
    g0 = policy_grad[0]  # (n_states, n_actions)
    print(f"    Shape: {g0.shape}")
    print(f"    Norm: {float(jnp.linalg.norm(g0)):.6f}")
    print(f"    Max abs: {float(jnp.max(jnp.abs(g0))):.6f}")

    # Gradient at the start state specifically
    g0_start = g0[start_flat]  # (n_actions,)
    print(f"\n    At start state (flat={start_flat}):")
    print(f"      Gradient per action:")
    g0_arr = np.asarray(g0_start)
    for a in range(N_ACTIONS):
        print(f"        {ACTION_NAMES[a]:12s}: {g0_arr[a]:+.6f}")

    # Per-timestep gradient magnitude
    print(f"\n  Per-timestep policy gradient norms:")
    for t in range(horizon):
        gt = policy_grad[t]
        print(f"    t={t}: norm={float(jnp.linalg.norm(gt)):.6f}  max={float(jnp.max(jnp.abs(gt))):.6f}")


def investigate_state_trajectory(result, n, horizon):
    """Check 4: Where does the planned state distribution go?"""
    print("\n" + "=" * 70)
    print("CHECK 4: PLANNED STATE TRAJECTORY")
    print("=" * 70)

    n_locations = n * n
    q_x_theta = result.q_x_theta  # (horizon+1, n_states, n_theta)
    q_theta = result.q_theta

    # Marginalize over θ
    q_x = jnp.sum(q_x_theta * q_theta[None, None, :], axis=2)  # (horizon+1, n_states)

    for t in range(min(horizon + 1, 6)):
        q_x_t = q_x[t]
        # Marginalize to grid
        grid = np.zeros((n, n))
        probs = np.asarray(q_x_t)
        for flat_idx in range(len(probs)):
            loc, _, _ = unflatten_state_index(flat_idx, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
            x, y = state_to_coords(loc, n)
            grid[y, x] += probs[flat_idx]

        total = grid.sum()
        print(f"\n  t={t} (total mass: {total:.4f}):")
        header = "    " + "  ".join(f" {c}" for c in range(n))
        print(header)
        for row in range(n):
            cells = "  ".join(f"{grid[row, col]:.3f}" for col in range(n))
            print(f"  {row} {cells}")

    # Action distribution per timestep
    print(f"\n  Action distributions per timestep:")
    q_u_theta = result.q_u_theta  # (horizon, n_actions, n_theta)
    q_u = jnp.sum(q_u_theta * q_theta[None, None, :], axis=2)  # (horizon, n_actions)
    for t in range(min(horizon, 6)):
        actions_str = "  ".join(f"{ACTION_NAMES[a][:4]}={float(q_u[t, a]):.3f}" for a in range(N_ACTIONS))
        print(f"    t={t}: {actions_str}")


def investigate_mode_comparison(env_tensors, n, horizon, n_opt_steps, lr, seed,
                                gradient_scale_factor=1.0,
                                freeze_obs_and_transitions=False, policy_init_scale=1.0,
                                goal_scale=1.0):
    """Check 5: Compare inference modes."""
    print("\n" + "=" * 70)
    print("CHECK 5: INFERENCE MODE COMPARISON")
    print("=" * 70)

    n_locations = n * n
    initial_state = jnp.zeros(env_tensors.n_states)
    start_flat = flatten_state_index(0, 0, 0, n_locations, N_ORIENTATIONS, N_DOOR_KEY_STATES)
    initial_state = initial_state.at[start_flat].set(1.0)

    for mode in ["marginal", "active", "planning"]:
        config = TemporalPlanningConfig(
            planning_horizon=horizon,
            n_states=env_tensors.n_states,
            n_actions=env_tensors.n_actions,
            n_theta=env_tensors.n_theta,
            n_optimization_steps=n_opt_steps,
            learning_rate=lr,
            inference_mode=mode,
            init_seed=seed,
            gradient_scale_factor=gradient_scale_factor,
            freeze_obs_and_transitions=freeze_obs_and_transitions,
            policy_init_scale=policy_init_scale,
            goal_scale=goal_scale,
        )
        result = plan_actions_temporal(
            initial_state=initial_state, env_tensors=env_tensors, config=config,
        )
        q_u1 = np.asarray(result.q_first_action)
        entropy = -np.sum(q_u1 * np.log(q_u1 + 1e-10))
        max_action = ACTION_NAMES[int(np.argmax(q_u1))]
        print(f"\n  {mode:10s}: loss={result.final_loss:.1f}  "
              f"q(u1)=[{', '.join(f'{v:.3f}' for v in q_u1)}]  "
              f"H={entropy:.3f}  best={max_action}({np.max(q_u1):.3f})")


def investigate_parameter_counts(env_tensors, horizon):
    """Check 6: Parameter space size."""
    print("\n" + "=" * 70)
    print("CHECK 6: PARAMETER SPACE")
    print("=" * 70)

    ns = env_tensors.n_states
    na = env_tensors.n_actions
    nt = env_tensors.n_theta

    n_policy = horizon * ns * na
    n_transition = horizon * ns * ns * na * nt
    n_theta = nt

    n_obs = 0
    for mod in env_tensors.observation_modalities:
        if mod.theta_dependent:
            n_obs += horizon * mod.n_obs * ns * nt
        else:
            n_obs += horizon * mod.n_obs * ns

    total = n_policy + n_transition + n_theta + n_obs

    print(f"\n  n_states={ns}, n_actions={na}, n_theta={nt}, horizon={horizon}")
    print(f"\n  Parameter counts:")
    print(f"    q(u|x) policy:          {n_policy:>12,d}  ({100*n_policy/total:.1f}%)")
    print(f"    q(x'|x,u,θ) transition: {n_transition:>12,d}  ({100*n_transition/total:.1f}%)")
    print(f"    q(θ):                   {n_theta:>12,d}  ({100*n_theta/total:.1f}%)")
    print(f"    q(y|x,θ) observations:  {n_obs:>12,d}  ({100*n_obs/total:.1f}%)")
    print(f"    ─────────────────────────────────")
    print(f"    Total:                  {total:>12,d}")
    print(f"\n  Ratio of goal-relevant params to total:")
    print(f"    Policy logits at t=0: {ns * na:,d} / {total:,d} = {100*ns*na/total:.4f}%")


def main():
    parser = argparse.ArgumentParser(description="Investigate MiniGrid VFE optimization")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--fov-size", type=int, default=3)
    parser.add_argument("--n-opt-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--inference-mode", type=str, default="active",
                        choices=["marginal", "active", "planning"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-scale-factor", type=float, default=1.0,
                        help="Exponential scaling factor for earlier-timestep policy gradients")
    parser.add_argument("--freeze-obs-and-transitions", action="store_true",
                        help="Freeze observation and transition logits (optimize only policy and theta)")
    parser.add_argument("--policy-init-scale", type=float, default=1.0,
                        help="Scale for random policy logit initialization (default: 1.0)")
    parser.add_argument("--goal-scale", type=float, default=1.0,
                        help="Multiplier for goal energy term (default: 1.0)")
    parser.add_argument("--checks", type=str, default="all",
                        help="Comma-separated checks to run: model,decompose,gradient,trajectory,modes,params,all")

    args = parser.parse_args()
    n = args.grid_size
    checks = set(args.checks.split(","))

    print(f"MiniGrid VFE Investigation")
    print(f"  n={n}, fov={args.fov_size}, horizon={args.horizon}, mode={args.inference_mode}")
    print(f"  opt_steps={args.n_opt_steps}, lr={args.learning_rate}")

    env_tensors = create_minigrid_env_tensors(n=n, fov_size=args.fov_size)

    # Build dense transition tensor on demand for investigation/analysis
    if env_tensors.transition_tensor is None:
        valid_configs = env_tensors.metadata.get("valid_configs", get_valid_static_configs(n))
        T_dense = generate_transition_tensor(n, valid_configs)
        env_tensors.transition_tensor = jnp.array(T_dense, dtype=jnp.float32)

    print(f"  n_states={env_tensors.n_states}, n_theta={env_tensors.n_theta}, n_actions={env_tensors.n_actions}")

    if "all" in checks or "params" in checks:
        investigate_parameter_counts(env_tensors, args.horizon)

    if "all" in checks or "model" in checks:
        investigate_generative_model(env_tensors, n)

    if "all" in checks or "gradient" in checks:
        investigate_gradient_magnitudes(env_tensors, n, args.horizon, args.inference_mode, args.seed)

    if "all" in checks or "decompose" in checks:
        result, initial_state = investigate_vfe_decomposition(
            env_tensors, n, args.horizon, args.n_opt_steps, args.learning_rate,
            args.inference_mode, args.seed,
            gradient_scale_factor=args.gradient_scale_factor,
            freeze_obs_and_transitions=args.freeze_obs_and_transitions,
            policy_init_scale=args.policy_init_scale,
            goal_scale=args.goal_scale,
        )
        if "all" in checks or "trajectory" in checks:
            investigate_state_trajectory(result, n, args.horizon)

    if "all" in checks or "modes" in checks:
        investigate_mode_comparison(
            env_tensors, n, args.horizon, args.n_opt_steps, args.learning_rate, args.seed,
            gradient_scale_factor=args.gradient_scale_factor,
            freeze_obs_and_transitions=args.freeze_obs_and_transitions,
            policy_init_scale=args.policy_init_scale,
            goal_scale=args.goal_scale,
        )


if __name__ == "__main__":
    main()
