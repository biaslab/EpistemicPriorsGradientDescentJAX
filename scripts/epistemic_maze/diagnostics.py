#!/usr/bin/env python3
"""
Epistemic diagnostics for the Epistemic Maze.

Prints per-step epistemic state preferences to contextualize cue drive:
- p~(x) ∝ exp(-H[q(y|x)])
- p~(x) ∝ exp(E_y[KL(q(θ|y,x) || q(θ|x))])
"""

import argparse
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments import (
    EpistemicMaze,
    create_epistemic_maze_tensors,
    N_STATES,
    N_ACTIONS,
    N_LOCATIONS,
    state_to_components,
    Location,
)
from src.planning import plan_actions_temporal, TemporalPlanningConfig

EPS = 1e-8

ACTION_NAMES = [
    "NAV_0", "NAV_1", "NAV_2", "NAV_3", "NAV_4",
    "DEC_KNOB", "INC_KNOB", "VISIT_CUE"
]


@dataclass
class DiagnosticConfig:
    planning_horizon: int = 7
    max_steps: int = 7
    n_optimization_steps: int = 500
    learning_rate: float = 0.01
    seed: int = 42
    inference_mode: str = "active"
    n_theta: int = 2


def location_name(loc: int) -> str:
    """Get human-readable location name."""
    if loc == Location.CUE:
        return "CUE"
    elif loc == Location.SAFE_SINK:
        return "SAFE_SINK"
    else:
        return f"NAV_{loc}"


def compute_epistemic_state_drives(
    theta_observation_tensor: jnp.ndarray,  # (n_obs_theta, n_states, n_theta)
    location_observation_tensor: jnp.ndarray,  # (n_locations, n_states)
    theta_belief: jnp.ndarray,  # (n_theta,)
    top_k: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute epistemic state preferences using temporal VFE epistemic priors.

    Computes three state priors from variational observation models:
    1. Theta observation informativeness: ũ(x) ∝ exp(-H[q(y_theta|x)])
    2. Theta information gain (observation prior): ũ(y,x) ∝ exp(KL[q(θ|y,x) || q(θ)])
    3. Location observation informativeness: ũ(x) ∝ exp(-H[q(y_location|x)])

    Returns:
        p_tilde_theta_info: State prior from theta observation informativeness
        p_tilde_theta_ig: Marginalized state prior from theta information gain
        p_tilde_location_info: State prior from location observation informativeness
    """
    n_obs_theta, n_states, n_theta = theta_observation_tensor.shape
    n_locations = location_observation_tensor.shape[0]
    
    theta_belief = jnp.clip(theta_belief, EPS, 1.0)
    theta_belief = theta_belief / jnp.sum(theta_belief)

    # ============ 1. State prior (THETA): ũ(x) ∝ exp(-H[q(y_theta | x)]) ============
    # Prefer states with low observation entropy (informative)
    
    # q(y_theta | x) = ∑_θ p(y_theta | x, θ) q(θ)
    q_y_theta_given_x = jnp.sum(
        theta_observation_tensor * theta_belief[None, None, :],
        axis=2  # Sum over θ
    )  # (n_obs_theta, n_states)

    q_y_theta_given_x_safe = jnp.clip(q_y_theta_given_x, EPS, 1.0)

    # H[q(y_theta | x)] for each x
    h_y_theta_given_x = -jnp.sum(
        q_y_theta_given_x_safe * jnp.log(q_y_theta_given_x_safe),
        axis=0  # Sum over observations
    )  # (n_states,)

    # State prior: ũ(x) ∝ exp(-H[...]) - prefer LOW entropy (informative) states
    p_tilde_theta_info = jax.nn.softmax(-h_y_theta_given_x)

    # ============ 2. Observation prior (THETA): ũ(y, x) ∝ exp(KL[q(θ|y,x) || q(θ)]) ============
    # Information gain: how much would observing y at state x update θ beliefs

    # q(θ | y, x) ∝ p(y | x, θ) q(θ)
    # Unnormalized posterior
    q_theta_yx_unnorm = theta_observation_tensor * theta_belief[None, None, :]  # (n_obs_theta, n_states, n_theta)

    # Normalize over θ to get q(θ | y, x)
    q_theta_yx_norm = jnp.sum(q_theta_yx_unnorm, axis=2, keepdims=True)  # (n_obs_theta, n_states, 1)
    q_theta_given_yx = q_theta_yx_unnorm / (q_theta_yx_norm + EPS)  # (n_obs_theta, n_states, n_theta)

    # KL[q(θ|y,x) || q(θ)] for each (y, x)
    log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(theta_belief[None, None, :] + EPS)
    kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=2)  # (n_obs_theta, n_states)

    # Prior: ũ(y, x) ∝ exp(KL) - prefer (y,x) pairs with high information gain
    yx_prior = jax.nn.softmax(kl_yx.flatten()).reshape(kl_yx.shape)  # (n_obs_theta, n_states)
    
    # Marginalize over observations to get state preference
    # Weighted by q(y|x) to get expected information gain per state
    expected_ig_per_state = jnp.sum(q_y_theta_given_x * kl_yx, axis=0)  # (n_states,)
    p_tilde_theta_ig = jax.nn.softmax(expected_ig_per_state)

    # ============ 3. State prior (LOCATION): ũ(x) ∝ exp(-H[q(y_location | x)]) ============
    # Prefer states with informative location observations
    
    # Location observations are deterministic in generative model, so entropy is typically low
    # But we compute it for consistency with the VFE formulation
    loc_obs_safe = jnp.clip(location_observation_tensor, EPS, 1.0)
    
    # H[q(y_location | x)] for each x
    h_loc_given_x = -jnp.sum(
        loc_obs_safe * jnp.log(loc_obs_safe),
        axis=0  # Sum over locations
    )  # (n_states,)

    # Location informativeness prior: ũ(x) ∝ exp(-H[y_location | x])
    p_tilde_location_info = jax.nn.softmax(-h_loc_given_x)

    return p_tilde_theta_info, p_tilde_theta_ig, p_tilde_location_info



def print_top_states(probabilities: jnp.ndarray, top_k: int = 10):
    """Print top-k states by probability."""
    top_indices = jnp.argsort(probabilities)[-top_k:][::-1]
    for idx in top_indices:
        loc, knob = state_to_components(int(idx))
        prob = float(probabilities[idx])
        marker = " <- CUE" if loc == Location.CUE else ""
        print(f"    {location_name(loc):10s} knob={knob}: {prob:.6f}{marker}")


def run_episode_with_diagnostics(
    config: DiagnosticConfig,
    env: EpistemicMaze,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    location_observation_tensor: jnp.ndarray,
    goal_mapping: jnp.ndarray,
    action_prior: jnp.ndarray,
    theta_prior: jnp.ndarray,
    theta_belief: jnp.ndarray,
    location_observation_accuracy: float = 0.90,
):
    """Run episode with detailed epistemic diagnostics."""
    rng = np.random.default_rng(config.seed)

    def sample_observation(context_obs) -> int:
        probs = np.asarray(context_obs, dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

    def bayesian_theta_update(prior_theta, obs_idx, state_idx):
        likelihood = observation_tensor[obs_idx, state_idx, :]
        posterior = likelihood * prior_theta
        posterior = posterior / (jnp.sum(posterior) + EPS)
        return posterior

    def sample_location_observation(location_obs) -> int:
        probs = np.asarray(location_obs, dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

    def bayesian_location_update(prior_location, location_obs_idx, loc_obs_acc):
        from src.environments import N_LOCATIONS
        likelihood = np.ones(N_LOCATIONS) * (1.0 - loc_obs_acc) / (N_LOCATIONS - 1)
        likelihood[location_obs_idx] = loc_obs_acc
        posterior = likelihood * prior_location
        posterior = posterior / (np.sum(posterior) + EPS)
        return posterior

    def bayesian_state_update_from_location(prior_state, action, location_obs_idx):
        """
        Update state belief using transition dynamics and location observation.

        Steps:
        1) Predict: q(s_t) = sum_{s_{t-1}} p(s_t | s_{t-1}, a_{t-1}) q(s_{t-1})
        2) Update:  q(s_t | o_t) proportional to p(o_t | s_t) q(s_t)

        Args:
            prior_state: Previous state belief (n_states,)
            action: Action taken
            location_obs_idx: Observed location index

        Returns:
            Posterior state belief (n_states,)
        """
        # Predict with transition model: transition_tensor is (n_states_next, n_states_prev, n_actions)
        transition_action = transition_tensor[:, :, action]  # (n_states_next, n_states_prev)
        prior_next = transition_action @ prior_state  # (n_states,)

        # Update with location observation likelihood
        # location_observation_tensor: (n_locations, n_states)
        likelihood = location_observation_tensor[location_obs_idx, :]  # (n_states,)
        posterior_state = likelihood * prior_next
        posterior_state = posterior_state / (np.sum(posterior_state) + EPS)

        return posterior_state

    # Initialize location belief as uniform
    from src.environments import N_LOCATIONS
    location_belief = np.ones(N_LOCATIONS) / N_LOCATIONS

    # Initialize state belief: uniform over locations, but concentrated at knob=4 (starting knob)
    # State index = loc + N_LOCATIONS * knob, so knob=4 states are indices 28-34
    state_belief = np.zeros(N_STATES)
    for loc in range(N_LOCATIONS):
        s = loc + N_LOCATIONS * 4  # knob=4
        state_belief[s] = 1.0 / N_LOCATIONS
    state_belief = state_belief / (np.sum(state_belief) + EPS)

    print(f"\n--- EPISODE ROLLOUT ---")
    print(f"True θ: {env.theta}")
    loc, knob = state_to_components(env.state_idx)
    print(f"Start: Location={location_name(loc)}, Knob={knob}")

    for step in range(config.max_steps):
        loc, knob = state_to_components(env.state_idx)
        
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")
        print(f"Current state: {env.state_idx} ({location_name(loc)}, knob={knob})")
        print(f"Current θ belief: ", end="")
        for i in range(config.n_theta):
            print(f"θ={i}: {float(theta_belief[i]):.3f}  ", end="")
        print()

        # Plan and act
        current_state_onehot = jnp.zeros(N_STATES)
        current_state_onehot = current_state_onehot.at[env.state_idx].set(1.0)
        
        effective_horizon = min(config.max_steps - step, config.planning_horizon)
        planning_config = TemporalPlanningConfig(
            planning_horizon=effective_horizon,
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            n_theta=config.n_theta,
            n_obs=config.n_theta + 1,
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            inference_mode=config.inference_mode,
            init_seed=config.seed + step,
        )

        # Use current theta belief as prior for next step
        prior_theta_logits = jnp.log(theta_belief + EPS)
        
        result = plan_actions_temporal(
            initial_state=current_state_onehot,
            transition_tensor=transition_tensor,
            theta_observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            config=planning_config,
            prior_theta_logits=prior_theta_logits,
            location_observation_tensor=location_observation_tensor,
        )
        
        # Compute epistemic state preferences using FINAL OPTIMIZED q_theta
        optimized_q_theta = result.q_theta
        p_tilde_theta_info, p_tilde_theta_ig, p_tilde_location_info = compute_epistemic_state_drives(
            theta_observation_tensor=observation_tensor,
            location_observation_tensor=location_observation_tensor,
            theta_belief=optimized_q_theta,
            top_k=10
        )
        
        print(f"\n--- EPISTEMIC STATE PREFERENCES (after optimization) ---")
        print(f"Optimized θ belief: ", end="")
        for i in range(config.n_theta):
            print(f"θ={i}: {float(optimized_q_theta[i]):.3f}  ", end="")
        print()
        
        print("\nTop 5 states by theta observation informativeness:")
        print_top_states(p_tilde_theta_info, top_k=5)
        
        print("\nTop 5 states by theta information gain:")
        print_top_states(p_tilde_theta_ig, top_k=5)
        
        print("\nTop 5 states by location observation informativeness:")
        print_top_states(p_tilde_location_info, top_k=5)
        
        # Print action posteriors
        print("\n--- ACTION SELECTION ---")
        print("q(u) action posteriors:")
        sorted_actions = jnp.argsort(result.q_first_action)[::-1]
        for a_idx in sorted_actions[:5]:  # Top 5 actions
            prob = float(result.q_first_action[a_idx])
            print(f"  {ACTION_NAMES[a_idx]:12s}: {prob:.4f}")
        
        action = int(jnp.argmax(result.q_first_action))
        print(f"\nSelected action: {ACTION_NAMES[action]}")

        # Step env
        print(f"\n--- ENVIRONMENT TRANSITION ---")
        location_obs, context_obs, reward, done = env.step(action)
        
        new_loc, new_knob = state_to_components(env.state_idx)
        print(f"New state: {env.state_idx} ({location_name(new_loc)}, knob={new_knob})")

        # Bayesian update of θ belief based on observation
        print(f"\n--- BELIEF UPDATES ---")
        obs_idx = sample_observation(context_obs)
        theta_belief = bayesian_theta_update(theta_belief, obs_idx, env.state_idx)
        print(f"Observed y_theta={obs_idx} → updated θ belief: ", end="")
        for i in range(config.n_theta):
            print(f"θ={i}: {float(theta_belief[i]):.3f}  ", end="")
        print()

        # Bayesian update of location belief based on observation
        location_obs_idx = sample_location_observation(location_obs)
        location_belief = bayesian_location_update(location_belief, location_obs_idx, location_observation_accuracy)
        print(f"Observed y_location={location_obs_idx} → updated location belief: ", end="")
        for i in range(N_LOCATIONS):
            loc_name = location_name(i) if i < N_LOCATIONS else f"Loc{i}"
            print(f"{loc_name}: {float(location_belief[i]):.3f}  ", end="")
        print()

        # Bayesian state update from location observation (using transition model)
        state_belief = bayesian_state_update_from_location(state_belief, action, location_obs_idx)
        print(f"\nState belief after transition + y_location={location_obs_idx} (top 5):")
        top_state_indices = np.argsort(state_belief)[-5:][::-1]
        for s_idx in top_state_indices:
            s_loc, s_knob = state_to_components(int(s_idx))
            prob = float(state_belief[s_idx])
            marker = " ← ACTUAL" if s_idx == env.state_idx else ""
            print(f"  State {s_idx} ({location_name(s_loc)}, knob={s_knob}): {prob:.4f}{marker}")

        if done:
            print(f"\n{'='*60}")
            print(f"Episode complete! Final reward: {env.get_final_reward()}")
            print(f"{'='*60}")
            break


def main():
    parser = argparse.ArgumentParser(description="Epistemic Maze Epistemic Diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-opt-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--inference-mode", type=str, default="active",
                        choices=["marginal", "active", "planning"])
    parser.add_argument("--n-theta", type=int, default=2, 
                        help="Number of context values (2 or 5)")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "unknown_nav", "unknown_cue", "known_nav"])

    args = parser.parse_args()
    config = DiagnosticConfig(
        n_optimization_steps=args.n_opt_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        inference_mode=args.inference_mode,
        n_theta=args.n_theta,
        planning_horizon=args.horizon,
        max_steps=args.horizon,
    )

    # Load params.yaml if available
    params_path = project_root / "params.yaml"
    cue_accuracy = 0.95
    goal_temperature = 1.0
    cue_cost_epsilon = 1.0
    location_observation_accuracy = 0.90
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f) or {}
        maze_params = params.get('epistemic_maze', {})
        cue_accuracy = maze_params.get('cue_observation_accuracy', cue_accuracy)
        goal_temperature = maze_params.get('goal_temperature', goal_temperature)
        cue_cost_epsilon = maze_params.get('cue_cost_epsilon', cue_cost_epsilon)
        location_observation_accuracy = maze_params.get('location_observation_accuracy', location_observation_accuracy)

    # Create tensors
    transition_tensor, location_observation_tensor, theta_observation_tensor, goal_mapping, action_prior, theta_prior = create_epistemic_maze_tensors(
        n_theta=args.n_theta,
        cue_accuracy=cue_accuracy,
        location_observation_accuracy=location_observation_accuracy,
        goal_temperature=goal_temperature,
        cue_cost_epsilon=cue_cost_epsilon,
    )
    # Use theta observation tensor for planning (same as old observation_tensor)
    observation_tensor = theta_observation_tensor

    print("=" * 70)
    print("EPISTEMIC MAZE - EPISTEMIC STATE PREFERENCES")
    print(f"Inference mode: {args.inference_mode}")
    print(f"n_theta: {args.n_theta}")
    print("=" * 70)

    if args.scenario in ["all", "unknown_nav"]:
        print("\n" + "=" * 70)
        print("Scenario: START AT NAV_0, θ UNKNOWN")
        print("=" * 70)
        env = EpistemicMaze.create(
            theta=None,
            start_location=0,
            n_theta=args.n_theta,
            cue_accuracy=cue_accuracy,
            location_accuracy=location_observation_accuracy,
        )
        theta_belief = jnp.ones(args.n_theta) / args.n_theta
        run_episode_with_diagnostics(
            config=config,
            env=env,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            theta_belief=theta_belief,
            location_observation_accuracy=location_observation_accuracy,
        )

    if args.scenario in ["all", "unknown_cue"]:
        print("\n" + "=" * 70)
        print("Scenario: START AT CUE (via VISIT_CUE), θ UNKNOWN")
        print("=" * 70)
        env = EpistemicMaze.create(
            theta=None,
            start_location=int(Location.CUE),
            n_theta=args.n_theta,
            cue_accuracy=cue_accuracy,
            location_accuracy=location_observation_accuracy,
        )
        theta_belief = jnp.ones(args.n_theta) / args.n_theta
        # Update belief immediately since starting at cue (Bayesian update from observation)
        init_obs = env._get_context_observation()
        init_obs_idx = int(np.random.default_rng(config.seed).choice(len(init_obs), p=np.asarray(init_obs)))
        likelihood = observation_tensor[init_obs_idx, env.state_idx, :]
        theta_belief = likelihood * theta_belief
        theta_belief = theta_belief / (jnp.sum(theta_belief) + EPS)
        print(f"\n*** Starting at CUE - observed y={init_obs_idx} ***")
        run_episode_with_diagnostics(
            config=config,
            env=env,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            theta_belief=theta_belief,
            location_observation_accuracy=location_observation_accuracy,
        )

    if args.scenario in ["all", "known_nav"]:
        print("\n" + "=" * 70)
        print("Scenario: START AT NAV_0, θ KNOWN (θ=0 at 95%)")
        print("=" * 70)
        env = EpistemicMaze.create(
            theta=0,
            start_location=0,
            n_theta=args.n_theta,
            cue_accuracy=cue_accuracy,
            location_accuracy=location_observation_accuracy,
        )
        theta_belief = jnp.zeros(args.n_theta).at[0].set(0.95)
        for i in range(1, args.n_theta):
            theta_belief = theta_belief.at[i].set(0.05 / (args.n_theta - 1) if args.n_theta > 1 else 0.0)
        run_episode_with_diagnostics(
            config=config,
            env=env,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            theta_belief=theta_belief,
            location_observation_accuracy=location_observation_accuracy,
        )


if __name__ == "__main__":
    main()
