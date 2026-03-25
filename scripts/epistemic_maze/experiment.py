#!/usr/bin/env python3
"""
Epistemic Maze experiment using the TEMPORAL factorization planner.

This applies the temporal factorization approach to the Epistemic Maze
environment with multi-goal epistemic uncertainty.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

from src.environments import (
    EpistemicMaze,
    create_epistemic_maze_tensors,
    N_STATES,
    N_ACTIONS,
    state_to_components,
    components_to_state,
    Location,
)
from src.planning.temporal_optimizer import (
    plan_actions_temporal,
    select_action_temporal,
    TemporalPlanningConfig,
)
from src.planning.sophisticated_planner import (
    SophisticatedPlanningConfig,
    SophisticatedPlanner,
)
from src.environments import N_LOCATIONS


def sample_observation_fn(context_obs) -> int:
    """Sample an observation index from context observation probabilities."""
    probs = jnp.asarray(context_obs)
    probs = probs / jnp.sum(probs)
    cumulative = jnp.cumsum(probs)
    r = random.random()
    return int(jnp.searchsorted(cumulative, r))


def sample_location_observation_fn(location_obs) -> int:
    """Sample a location index from location observation probabilities."""
    probs = jnp.asarray(location_obs)
    probs = probs / jnp.sum(probs)
    cumulative = jnp.cumsum(probs)
    r = random.random()
    return int(jnp.searchsorted(cumulative, r))


def bayesian_theta_update_fn(observation_tensor, prior_theta, obs_idx, state_idx):
    """Update θ belief using Bayesian filtering."""
    likelihood = observation_tensor[obs_idx, state_idx, :]
    posterior = likelihood * prior_theta
    posterior = posterior / (jnp.sum(posterior) + 1e-8)
    return posterior


def bayesian_location_update_fn(prior_location, location_obs_idx, location_observation_accuracy):
    """Update location belief using Bayesian filtering."""
    # Likelihood: P(obs|location)
    # High probability for observed location, uniform for others
    likelihood = jnp.ones(N_LOCATIONS) * (1.0 - location_observation_accuracy) / (N_LOCATIONS - 1)
    likelihood = likelihood.at[location_obs_idx].set(location_observation_accuracy)
    posterior = likelihood * prior_location
    posterior = posterior / (jnp.sum(posterior) + 1e-8)
    return posterior


def bayesian_state_update_from_location_fn(
    prior_state,
    action,
    transition_tensor,
    location_observation_tensor,
    location_obs_idx,
):
    r"""
    Update state belief using transition dynamics and a location observation.

    Steps:
    1) Predict: q(s_t) = \sum_{s_{t-1}} p(s_t | s_{t-1}, a_{t-1}) q(s_{t-1})
    2) Update:  q(s_t | o_t) ∝ p(o_t | s_t) q(s_t)
    """
    # Predict with transition model (transition_tensor: (n_states, n_states, n_actions))
    transition_action = transition_tensor[:, :, action]  # (n_states_next, n_states_prev)
    prior_next = transition_action @ prior_state  # (n_states,)

    # Update with location observation likelihood (location_observation_tensor: (n_locations, n_states))
    likelihood = location_observation_tensor[location_obs_idx, :]  # (n_states,)
    posterior_state = likelihood * prior_next
    posterior_state = posterior_state / (jnp.sum(posterior_state) + 1e-8)

    return posterior_state


def get_observation(env: EpistemicMaze, is_final_step: bool = False) -> list:
    """
    Get current observation from environment state.

    Args:
        env: The epistemic maze environment
        is_final_step: If True, include reward observation; otherwise neutral

    Returns list of [location_obs_idx, theta_obs_idx, reward_obs_idx]
    """
    loc, knob = state_to_components(env.state_idx)

    # Location observation (sample from distribution)
    if loc == Location.CUE:
        location_obs_idx = int(Location.CUE)
    else:
        # Noisy observation - high prob on true location
        if random.random() < env.location_accuracy:
            location_obs_idx = loc
        else:
            # Random other location (excluding CUE and true location)
            other_locs = [i for i in range(N_LOCATIONS) if i != loc and i != Location.CUE]
            location_obs_idx = random.choice(other_locs)

    # Theta observation
    if loc == Location.CUE:
        # Informative observation at cue
        if random.random() < env.cue_accuracy:
            theta_obs_idx = env.theta
        else:
            # Random other theta
            other_thetas = [i for i in range(env.n_theta) if i != env.theta]
            theta_obs_idx = random.choice(other_thetas) if other_thetas else env.theta
    else:
        # Neutral observation (index = n_theta)
        theta_obs_idx = env.n_theta

    # Reward observation: Only reveal at final step (like T-maze)
    # During episode, pass None to skip belief update for reward modality
    # This prevents the A matrix from incorrectly ruling out knob=4 states
    if is_final_step:
        if loc == Location.SAFE_SINK:
            reward_obs_idx = 3
        elif loc < 5 and knob == 4:  # Nav state with max knob at final step
            reward_obs_idx = 1 if loc == env.theta else 2
        else:
            reward_obs_idx = 0
    else:
        reward_obs_idx = None  # Don't update beliefs on reward during episode

    return [location_obs_idx, theta_obs_idx, reward_obs_idx]


def run_episode_sophisticated(
    env: EpistemicMaze,
    transition_tensor,
    observation_tensor,
    location_observation_tensor,
    goal_mapping,
    action_prior,
    theta_prior,
    config: SophisticatedPlanningConfig,
    location_observation_accuracy: float = 0.90,  # noqa: ARG001
    max_steps: int = 7,
    verbose: bool = False,
) -> Dict:
    """
    Run a single episode using sophisticated (pymdp) active inference.

    Follows the standard pymdp pattern:
        for t in range(T):
            qs = agent.infer_states(obs)
            q_pi, efe = agent.infer_policies()
            action = agent.sample_action()
            obs = env.step(action)
    """
    trajectory = [env.state_idx]
    actions = []
    planning_history = []

    # Create sophisticated planner
    planner = SophisticatedPlanner(
        transition_tensor=transition_tensor,
        theta_observation_tensor=observation_tensor,
        location_observation_tensor=location_observation_tensor,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        config=config,
    )

    planner.agent.reset()

    # Get initial observation (not final step)
    obs = get_observation(env, is_final_step=False)

    for step in range(max_steps):
        # === 1) Infer states from observation ===
        planner.infer_states(obs)
        remaining = min(max_steps - step, config.inference_horizon)
        if config.sophisticated:
            # Receding-horizon control for sophisticated inference
            if hasattr(planner.agent, "si_horizon"):
                planner.agent.si_horizon = remaining
        else:
            # Rebuild policies with new length for unsophisticated
            planner.rebuild_policies(remaining)
        # 2) Infer policies (compute EFE) - replan every step for both sophisticated and vanilla
        q_pi, efe = planner.infer_policies()

        # 3) Sample action - this also records prev_actions for state transitions
        action_arr = planner.sample_action()
        action = int(action_arr[0])  # First (and only) controllable factor

        if verbose:
            loc, knob = state_to_components(env.state_idx)
            q_theta = planner.agent.qs[1]
            action_name = ["NAV_0", "NAV_1", "NAV_2", "NAV_3", "NAV_4", "DEC_KNOB", "INC_KNOB", "VISIT_CUE"][action]
            print(f"  Step {step}: State={env.state_idx} (loc={loc}, knob={knob}), Action={action} ({action_name})")
            print(f"    q(theta)={q_theta}")

        # Record planning
        planning_history.append({
            'step': step,
            'state': env.state_idx,
            'q_pi': q_pi.tolist() if hasattr(q_pi, 'tolist') else list(q_pi),
            'q_theta': planner.agent.qs[1].tolist(),
            'efe': efe.tolist() if efe is not None and hasattr(efe, 'tolist') else None,
            'action': action,
        })

        # 4) Execute action in environment
        env.step(action)

        # 5) Get new observation (reward only revealed at final step)
        is_final = (step == max_steps - 1)
        obs = get_observation(env, is_final_step=is_final)

        trajectory.append(env.state_idx)
        actions.append(action)

    # Get final reward and outcome
    final_reward = env.get_final_reward()
    final_loc, final_knob = state_to_components(trajectory[-1])

    # Check if visited cue
    visited_cue = any(state_to_components(s)[0] == Location.CUE for s in trajectory[1:])

    return {
        'total_reward': float(final_reward),
        'n_steps': len(actions),
        'reached_goal': (final_loc == env.theta and final_knob == 4),
        'reached_safe': (final_loc == Location.SAFE_SINK),
        'outcome': env.get_outcome(),
        'trajectory': trajectory,
        'actions': actions,
        'final_state': trajectory[-1],
        'theta': env.theta,
        'visited_cue': visited_cue,
        'planning_history': planning_history,
    }


def run_episode(
    env: EpistemicMaze,
    transition_tensor,
    observation_tensor,
    location_observation_tensor,
    goal_mapping,
    action_prior,
    theta_prior,
    config: TemporalPlanningConfig,
    location_observation_accuracy: float = 0.90,
    max_steps: int = 7,
    receding_horizon: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Run a single episode in the Epistemic Maze using temporal planner.
    """
    trajectory = [env.state_idx]
    actions = []
    planning_history = []

    # Track belief about θ
    # Initialize to uniform prior
    prior_theta_logits = jnp.log(theta_prior + 1e-8)

    # Track belief about location
    # Initialize to uniform prior over all locations
    prior_location_logits = jnp.log(jnp.ones(N_LOCATIONS) / N_LOCATIONS)

    # Track belief about full state (location x knob)
    # Initialize to uniform over all locations with knob fixed at 4
    state_belief = jnp.zeros(config.n_states)
    for loc in range(N_LOCATIONS):
        s = components_to_state(loc, 4)
        state_belief = state_belief.at[s].set(1.0 / N_LOCATIONS)
    state_belief = state_belief / jnp.sum(state_belief + 1e-8)
    for step in range(max_steps):
        # Receding horizon
        if receding_horizon:
            effective_horizon = min(max_steps - step, config.planning_horizon)
        else:
            effective_horizon = config.planning_horizon
        if effective_horizon <= 0:
            break

        # Current state belief (posterior from previous step)
        current_state_dist = state_belief

        # Update config for effective horizon
        step_config = TemporalPlanningConfig(
            planning_horizon=effective_horizon,
            n_states=config.n_states,
            n_actions=config.n_actions,
            n_theta=config.n_theta,
            n_obs=config.n_obs,
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            inference_mode=config.inference_mode,
            init_seed=config.init_seed + step,  # Different seed per step
        )

        # Plan actions
        planning_result = plan_actions_temporal(
            initial_state=current_state_dist,
            transition_tensor=transition_tensor,
            theta_observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            config=step_config,
            prior_theta_logits=prior_theta_logits,
            location_observation_tensor=location_observation_tensor,
        )

        # Select action
        action = select_action_temporal(
            result=planning_result,
            current_state_idx=env.state_idx,
            timestep=0,
        )

        if verbose:
            loc, knob = state_to_components(env.state_idx)
            print(f"  Step {step}: State={env.state_idx} (loc={loc}, knob={knob}), Action={action}")
            print(f"    q(u1)={planning_result.q_first_action}")
            print(f"    q(θ)={planning_result.q_theta}")

        # Record planning
        planning_history.append({
            'step': step,
            'state': env.state_idx,
            'q_first_action': planning_result.q_first_action.tolist(),
            'q_theta': planning_result.q_theta.tolist(),
            'action': int(action),
            'all_losses': planning_result.all_losses,
            'loss_history': planning_result.loss_history,
            'final_loss': planning_result.final_loss,
        })

        # Execute action
        location_obs, context_obs, _reward, done = env.step(action)

        # Bayesian update of θ belief based on observation
        obs_idx = sample_observation_fn(context_obs)
        prior_theta = jax.nn.softmax(prior_theta_logits)
        posterior_theta = bayesian_theta_update_fn(observation_tensor, prior_theta, obs_idx, env.state_idx)
        prior_theta_logits = jnp.log(posterior_theta + 1e-8)

        # Bayesian update of location belief based on observation
        location_obs_idx = sample_location_observation_fn(location_obs)
        prior_location = jax.nn.softmax(prior_location_logits)
        posterior_location = bayesian_location_update_fn(prior_location, location_obs_idx, location_observation_accuracy)
        prior_location_logits = jnp.log(posterior_location + 1e-8)

        # Bayesian update of full state belief using transition and location observation
        state_belief = bayesian_state_update_from_location_fn(
            prior_state=state_belief,
            action=action,
            transition_tensor=transition_tensor,
            location_observation_tensor=location_observation_tensor,
            location_obs_idx=location_obs_idx,
        )

        trajectory.append(env.state_idx)
        actions.append(int(action))

        if done:
            break

    # Get final reward and outcome
    final_reward = env.get_final_reward()
    final_loc, final_knob = state_to_components(trajectory[-1])

    # Check if visited cue (location 6)
    visited_cue = any(state_to_components(s)[0] == Location.CUE for s in trajectory[1:])

    return {
        'total_reward': float(final_reward),
        'n_steps': len(actions),
        'reached_goal': (final_loc == env.theta and final_knob == 4),
        'reached_safe': (final_loc == Location.SAFE_SINK),
        'outcome': env.get_outcome(),
        'trajectory': trajectory,
        'actions': actions,
        'final_state': trajectory[-1],
        'theta': env.theta,
        'visited_cue': visited_cue,
        'planning_history': planning_history,
    }


def run_experiment(
    n_episodes: int,
    inference_mode: str,
    seed: int,
    n_theta: int = 2,
    horizon: int = 7,
    max_steps: int = 5,
    n_optimization_steps: int = 2000,
    learning_rate: float = 0.01,
    goal_temperature: float = 1.0,
    cue_observation_accuracy: float = 1.0,
    cue_cost_epsilon: float = 0.01,
    location_observation_accuracy: float = 0.90,
    verbose: bool = False,
    strategy: str = "temporal",
    policy_len: int = None,
    receding_horizon: bool = True,
) -> Dict:
    """Run full experiment.

    Args:
        strategy: Planning strategy to use:
            - "temporal": Temporal VFE factorization (with inference_mode)
            - "sophisticated": pymdp-based sophisticated active inference (tree-search)
            - "vanilla": pymdp-based vanilla active inference (single-step EFE)
        policy_len: Length of action sequences for pymdp strategies.
            Default: horizon for sophisticated, 1 for vanilla.
    """
    random.seed(seed)

    # Create Epistemic Maze tensors
    transition_tensor, location_observation_tensor, theta_observation_tensor, goal_mapping, action_prior, theta_prior = create_epistemic_maze_tensors(
        n_theta=n_theta,
        cue_accuracy=cue_observation_accuracy,
        location_observation_accuracy=location_observation_accuracy,
        goal_temperature=goal_temperature,
        cue_cost_epsilon=cue_cost_epsilon,
    )
    # Use theta observation tensor for planning (same as old observation_tensor)
    observation_tensor = theta_observation_tensor

    # Create config based on strategy
    if strategy in ("sophisticated", "vanilla"):
        # Both use pymdp, but differ in whether tree-search is enabled
        is_sophisticated = (strategy == "sophisticated")
        # Determine policy length:
        # - Sophisticated: policy_len=1 (required by pymdp), use si_horizon for tree search
        # - Vanilla: policy_len=horizon for one-shot planning
        if policy_len is not None:
            effective_policy_len = policy_len
        else:
            effective_policy_len = 1 if is_sophisticated else horizon
        config = SophisticatedPlanningConfig(
            planning_horizon=horizon,
            n_states=N_STATES,  # 35
            n_actions=N_ACTIONS,  # 8
            n_theta=n_theta,
            policy_len=effective_policy_len,
            inference_horizon=horizon,  # Tree search depth (only used if sophisticated)
            use_utility=True,
            use_states_info_gain=True,  # Info gain on x (states)
            use_param_info_gain=True,  # Info gain on theta (parameters)
            action_selection="deterministic",
            gamma=16.0,
            sophisticated=is_sophisticated,  # Tree-search vs single-step EFE
            include_reward_modality=True,  # Enable goal-seeking via reward preferences
            goal_temperature=goal_temperature,  # Control goal preference strength
        )
        run_episode_fn = run_episode_sophisticated
        desc_label = f"{strategy} (policy_len={effective_policy_len}, si_horizon={horizon if is_sophisticated else 1})"
    else:
        # Temporal VFE strategy
        n_obs = n_theta + 1  # Context observations + neutral
        config = TemporalPlanningConfig(
            planning_horizon=horizon,
            n_states=N_STATES,  # 35
            n_actions=N_ACTIONS,  # 8
            n_theta=n_theta,
            n_obs=n_obs,
            n_optimization_steps=n_optimization_steps,
            learning_rate=learning_rate,
            inference_mode=inference_mode,
            init_seed=seed,
        )
        run_episode_fn = run_episode
        desc_label = inference_mode

    episodes = []
    for episode_idx in tqdm(range(n_episodes), desc=f"Running {desc_label} episodes"):
        # Create new environment with random start location (nav states 0-4)
        env = EpistemicMaze.create(
            theta=None,
            start_location=None,  # Random nav state
            n_theta=n_theta,
            cue_accuracy=cue_observation_accuracy,
            location_accuracy=location_observation_accuracy,
        )

        if verbose:
            print(f"\nEpisode {episode_idx + 1}: Theta={env.theta}")

        # Run episode
        episode_kwargs = dict(
            env=env,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            config=config,
            location_observation_accuracy=location_observation_accuracy,
            max_steps=max_steps,
            verbose=verbose,
        )
        # Only temporal strategy supports receding horizon
        if run_episode_fn is run_episode:
            episode_kwargs["receding_horizon"] = receding_horizon
        episode_data = run_episode_fn(**episode_kwargs)

        episodes.append(episode_data)

    # Compute statistics
    total_rewards = [ep['total_reward'] for ep in episodes]
    cue_visits = sum(1 for ep in episodes if ep['visited_cue'])
    successes = sum(1 for ep in episodes if ep['reached_goal'])
    safe_visits = sum(1 for ep in episodes if ep['reached_safe'])

    stats = {
        'strategy': strategy,
        'inference_mode': inference_mode if strategy == "temporal" else None,
        'n_episodes': n_episodes,
        'mean_reward': float(sum(total_rewards) / len(total_rewards)),
        'success_rate': successes / n_episodes,
        'safe_rate': safe_visits / n_episodes,
        'cue_visit_rate': cue_visits / n_episodes,
        'cue_visits': cue_visits,
    }

    return {
        'config': {
            'strategy': strategy,
            'n_episodes': n_episodes,
            'n_theta': n_theta,
            'horizon': horizon,
            'max_steps': max_steps,
            'n_optimization_steps': n_optimization_steps if strategy == "temporal" else None,
            'learning_rate': learning_rate if strategy == "temporal" else None,
            'policy_len': config.policy_len if strategy in ("sophisticated", "vanilla") else None,
            'sophisticated': config.sophisticated if strategy in ("sophisticated", "vanilla") else None,
            'goal_temperature': goal_temperature,
            'cue_observation_accuracy': cue_observation_accuracy,
            'cue_cost_epsilon': cue_cost_epsilon,
            'location_observation_accuracy': location_observation_accuracy,
            'seed': seed,
        },
        'stats': stats,
        'episodes': episodes,
    }


def load_params_from_yaml(params_path: Path) -> dict:
    """Load experiment parameters from params.yaml if it exists."""
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('epistemic_maze', {})
    return {}


def main():
    parser = argparse.ArgumentParser(description="Epistemic Maze Experiment")
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['temporal', 'sophisticated', 'vanilla'],
        default='temporal',
        help='Planning strategy: temporal (VFE factorization), sophisticated (pymdp tree-search), or vanilla (pymdp single-step EFE)'
    )
    parser.add_argument(
        '--inference-mode',
        type=str,
        choices=['marginal', 'active', 'planning'],
        default='marginal',
        help='Inference mode for temporal strategy (ignored for sophisticated)'
    )
    parser.add_argument('--n-episodes', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--n-opt-steps', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--n-theta', type=int, default=2, help='Number of possible theta values (2 or 6)')
    parser.add_argument('--policy-len', type=int, default=None, help='Policy length for pymdp strategies (default: 1 for sophisticated, horizon for vanilla)')
    parser.add_argument('--goal-temperature', type=float, default=None)
    parser.add_argument('--cue-accuracy', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-receding-horizon', action='store_true',
                        help='Disable receding horizon (use full horizon throughout episode)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--output-dir', type=str, default='data/epistemic_maze')

    args = parser.parse_args()

    # Load defaults from params.yaml
    project_root = Path(__file__).parent.parent.parent
    params_path = project_root / "params.yaml"
    yaml_params = load_params_from_yaml(params_path)

    # Merge: CLI args override yaml params, yaml params override hardcoded defaults
    defaults = {
        'n_episodes': 1,
        'horizon': 7,
        'max_steps': 5,
        'n_optimization_steps': 2000,
        'learning_rate': 0.01,
        'seed': 42,
        'goal_temperature': 1.0,
        'cue_observation_accuracy': 1.0,
        'cue_cost_epsilon': 0.01,
        'location_observation_accuracy': 0.90,
    }

    n_episodes = args.n_episodes if args.n_episodes is not None else yaml_params.get('n_episodes', defaults['n_episodes'])
    horizon = args.horizon if args.horizon is not None else yaml_params.get('horizon', defaults['horizon'])
    max_steps = args.max_steps if args.max_steps is not None else yaml_params.get('max_steps', defaults['max_steps'])
    n_optimization_steps = args.n_opt_steps if args.n_opt_steps is not None else yaml_params.get('n_optimization_steps', defaults['n_optimization_steps'])
    learning_rate = args.learning_rate if args.learning_rate is not None else yaml_params.get('learning_rate', defaults['learning_rate'])
    seed = args.seed if args.seed is not None else yaml_params.get('seed', defaults['seed'])
    goal_temperature = args.goal_temperature if args.goal_temperature is not None else yaml_params.get('goal_temperature', defaults['goal_temperature'])
    cue_accuracy = args.cue_accuracy if args.cue_accuracy is not None else yaml_params.get('cue_observation_accuracy', defaults['cue_observation_accuracy'])
    cue_cost_epsilon = yaml_params.get('cue_cost_epsilon', defaults['cue_cost_epsilon'])
    location_observation_accuracy = yaml_params.get('location_observation_accuracy', defaults['location_observation_accuracy'])

    print(f"\nEpistemic Maze Experiment")
    print(f"Strategy: {args.strategy}")
    if args.strategy == "temporal":
        print(f"Inference mode: {args.inference_mode}")
        print(f"Optimization steps: {n_optimization_steps}")
    elif args.strategy in ("sophisticated", "vanilla"):
        default_policy_len = '1' if args.strategy == 'sophisticated' else 'horizon'
        print(f"Policy length: {args.policy_len or default_policy_len}")
    print(f"Episodes: {n_episodes}")
    print(f"Horizon: {horizon}")
    print(f"N_theta: {args.n_theta}\n")

    results = run_experiment(
        n_episodes=n_episodes,
        inference_mode=args.inference_mode,
        seed=seed,
        n_theta=args.n_theta,
        horizon=horizon,
        max_steps=max_steps,
        n_optimization_steps=n_optimization_steps,
        learning_rate=learning_rate,
        goal_temperature=goal_temperature,
        cue_observation_accuracy=cue_accuracy,
        cue_cost_epsilon=cue_cost_epsilon,
        location_observation_accuracy=location_observation_accuracy,
        verbose=args.verbose,
        strategy=args.strategy,
        policy_len=args.policy_len,
        receding_horizon=not args.no_receding_horizon,
    )

    # Create output directory
    if args.strategy in ("sophisticated", "vanilla"):
        output_subdir = args.strategy
    else:
        output_subdir = args.inference_mode
    output_dir = Path(args.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("\nSummary Statistics:")
    print(f"  Mean reward: {results['stats']['mean_reward']:.3f}")
    print(f"  Success rate: {results['stats']['success_rate']:.2%}")
    print(f"  Safe sink rate: {results['stats']['safe_rate']:.2%}")
    print(f"  Cue visit rate: {results['stats']['cue_visit_rate']:.2%}")

    # Show episode details for debugging
    if args.verbose or n_episodes <= 5:
        print("\nEpisode details:")
        for i, ep in enumerate(results['episodes']):
            loc, knob = state_to_components(ep['final_state'])
            print(f"  Ep {i+1}: theta={ep['theta']}, visited_cue={ep['visited_cue']}, final_loc={loc}, final_knob={knob}, reward={ep['total_reward']}, outcome={ep['outcome']}")


if __name__ == '__main__':
    main()
