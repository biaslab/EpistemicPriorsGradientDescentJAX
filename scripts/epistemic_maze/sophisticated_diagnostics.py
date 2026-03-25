#!/usr/bin/env python3
"""
Diagnostics for sophisticated vs. unsophisticated pymdp planning.

Prints per-step policy posterior and action posterior for the Epistemic Maze.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List
import random

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
    N_LOCATIONS,
    N_STATES,
    N_ACTIONS,
    state_to_components,
    Location,
)
from src.planning import SophisticatedPlanner, SophisticatedPlanningConfig

ACTION_NAMES = [
    "NAV_0", "NAV_1", "NAV_2", "NAV_3", "NAV_4",
    "DEC_KNOB", "INC_KNOB", "VISIT_CUE"
]


@dataclass
class DiagnosticConfig:
    planning_horizon: int = 7
    max_steps: int = 7
    n_theta: int = 2
    action_selection: str = "deterministic"
    sophisticated: bool = True
    goal_temperature: float = 1.0


def get_observation(env: EpistemicMaze, is_final_step: bool = False) -> List[int]:
    """
    Get current observation from environment state.

    Returns list of [location_obs_idx, theta_obs_idx, reward_obs_idx]
    """
    loc, knob = state_to_components(env.state_idx)

    # Location observation
    if loc == Location.CUE:
        location_obs_idx = int(Location.CUE)
    else:
        if random.random() < env.location_accuracy:
            location_obs_idx = loc
        else:
            other_locs = [i for i in range(N_LOCATIONS) if i != loc and i != Location.CUE]
            location_obs_idx = random.choice(other_locs)

    # Theta observation
    if loc == Location.CUE:
        if random.random() < env.cue_accuracy:
            theta_obs_idx = env.theta
        else:
            other_thetas = [i for i in range(env.n_theta) if i != env.theta]
            theta_obs_idx = random.choice(other_thetas) if other_thetas else env.theta
    else:
        theta_obs_idx = env.n_theta

    # Reward observation: Only reveal at final step
    # During episode, pass None to skip belief update for reward modality
    # This prevents the A matrix from incorrectly ruling out knob=4 states
    if is_final_step:
        if loc == 5:  # SAFE_SINK
            reward_obs_idx = 3
        elif loc < 5 and knob == 4:  # Nav state with max knob
            reward_obs_idx = 1 if loc == env.theta else 2
        else:
            reward_obs_idx = 0
    else:
        reward_obs_idx = None  # Don't update beliefs on reward during episode

    return [location_obs_idx, theta_obs_idx, reward_obs_idx]


def compute_action_posterior(q_pi: np.ndarray, policies: np.ndarray, n_actions: int) -> np.ndarray:
    """Marginalize policy posterior to action posterior for t=0."""
    q_u = np.zeros(n_actions)
    for p_idx, policy in enumerate(policies):
        action = int(policy[0, 0])
        q_u[action] += q_pi[p_idx]
    q_u = q_u / (np.sum(q_u) + 1e-16)
    return q_u


def run_episode_diagnostics(
    env: EpistemicMaze,
    transition_tensor,
    observation_tensor,
    location_observation_tensor,
    goal_mapping,
    action_prior,
    theta_prior,
    config: DiagnosticConfig,
) -> Dict:
    """Run a single episode and print policy/action posteriors."""
    history = []

    planner = SophisticatedPlanner(
        transition_tensor=transition_tensor,
        theta_observation_tensor=observation_tensor,
        location_observation_tensor=location_observation_tensor,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        config=SophisticatedPlanningConfig(
            planning_horizon=config.planning_horizon,
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            n_theta=config.n_theta,
            policy_len=1 if config.sophisticated else config.planning_horizon,
            inference_horizon=config.planning_horizon,
            action_selection=config.action_selection,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            gamma=16.0,
            sophisticated=config.sophisticated,
            include_reward_modality=True,
            goal_temperature=config.goal_temperature,
        ),
    )

    planner.agent.reset()

    obs = get_observation(env, is_final_step=False)

    for step in range(config.max_steps):
        # Update state beliefs from observation
        planner.infer_states(obs)

        # Receding-horizon control for both sophisticated and unsophisticated
        remaining = min(config.max_steps - step, config.planning_horizon)
        if config.sophisticated:
            if hasattr(planner.agent, "si_horizon"):
                planner.agent.si_horizon = remaining
        else:
            # Rebuild policies with new length for unsophisticated
            planner.rebuild_policies(remaining)

        # Infer policies (compute EFE) - replan every step for both sophisticated and vanilla
        q_pi, efe = planner.infer_policies()
        
        # Compute action posterior for diagnostics
        q_u = compute_action_posterior(q_pi, planner.agent.policies, N_ACTIONS)

        # Sample action - this also records prev_actions for state transitions
        action_arr = planner.sample_action()
        action = int(action_arr[0])

        loc, knob = state_to_components(env.state_idx)
        q_theta = planner.agent.qs[1]

        print("\n" + "=" * 60)
        print(f"STEP {step}  (remaining horizon: {remaining})")
        print("=" * 60)
        print(f"True State: {env.state_idx} (loc={loc}, knob={knob})")
        print("Theta belief:", " ".join([f"θ={i}:{q_theta[i]:.3f}" for i in range(config.n_theta)]))

        # Print state belief (top 5 states)
        q_state = planner.agent.qs[0]
        top_states = np.argsort(q_state)[::-1][:5]
        print("State belief q(s) top 5:")
        for s_idx in top_states:
            s_loc, s_knob = state_to_components(s_idx)
            print(f"  s={s_idx} (loc={s_loc}, knob={s_knob}): {q_state[s_idx]:.4f}")

        top_actions = np.argsort(q_u)[::-1]
        print("Action posterior q(u) for first action:")
        for a_idx in top_actions[:5]:
            print(f"  {ACTION_NAMES[a_idx]:12s}: {q_u[a_idx]:.4f}")

        top_policies = np.argsort(q_pi)[::-1]
        n_policies = len(q_pi)
        policy_len = planner.agent.policies[0].shape[0]
        print(f"Top policy posterior q(pi) (total {n_policies} policies, policy_len={policy_len}, sum={np.sum(q_pi):.4f}):")
        for p_idx in top_policies[:5]:
            # Only show controllable factor (factor 0) - policies shape is (policy_len, num_factors)
            policy = planner.agent.policies[p_idx]
            action_indices = [int(policy[t, 0]) for t in range(policy.shape[0])]
            action_names = [ACTION_NAMES[a] for a in action_indices]
            print(f"  pi[{p_idx}]={q_pi[p_idx]:.4f}  actions={action_names}")

        history.append({
            "step": step,
            "state": int(env.state_idx),
            "q_pi": q_pi.tolist(),
            "q_u": q_u.tolist(),
            "q_theta": q_theta.tolist(),
            "efe": efe.tolist() if efe is not None and hasattr(efe, "tolist") else None,
            "action": action,
        })

        env.step(action)
        is_final = (step == config.max_steps - 1)
        obs = get_observation(env, is_final_step=is_final)

    return {"history": history}


def load_params_from_yaml(params_path: Path) -> dict:
    """Load experiment parameters from params.yaml if it exists."""
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('epistemic_maze', {})
    return {}


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for pymdp planning on Epistemic Maze")
    parser.add_argument("--n-theta", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-selection", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"])
    parser.add_argument("--mode", type=str, default="sophisticated",
                        choices=["sophisticated", "unsophisticated"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--goal-temperature", type=float, default=None)
    parser.add_argument("--cue-accuracy", type=float, default=None)
    parser.add_argument("--location-accuracy", type=float, default=None)
    parser.add_argument("--cue-cost-epsilon", type=float, default=None)
    args = parser.parse_args()

    # Load defaults from params.yaml
    params_path = project_root / "params.yaml"
    yaml_params = load_params_from_yaml(params_path)

    # Hardcoded defaults (fallback if not in yaml)
    defaults = {
        'horizon': 7,
        'seed': 42,
        'goal_temperature': 1.0,
        'cue_observation_accuracy': 1.0,
        'cue_cost_epsilon': 0.01,
        'location_observation_accuracy': 0.90,
    }

    # Merge: CLI args override yaml params, yaml params override hardcoded defaults
    horizon = args.horizon if args.horizon is not None else yaml_params.get('horizon', defaults['horizon'])
    max_steps = args.max_steps if args.max_steps is not None else horizon
    seed = args.seed if args.seed is not None else yaml_params.get('seed', defaults['seed'])
    goal_temperature = args.goal_temperature if args.goal_temperature is not None else yaml_params.get('goal_temperature', defaults['goal_temperature'])
    cue_accuracy = args.cue_accuracy if args.cue_accuracy is not None else yaml_params.get('cue_observation_accuracy', defaults['cue_observation_accuracy'])
    location_accuracy = args.location_accuracy if args.location_accuracy is not None else yaml_params.get('location_observation_accuracy', defaults['location_observation_accuracy'])
    cue_cost_epsilon = args.cue_cost_epsilon if args.cue_cost_epsilon is not None else yaml_params.get('cue_cost_epsilon', defaults['cue_cost_epsilon'])

    random.seed(seed)
    np.random.seed(seed)

    transition_tensor, location_observation_tensor, theta_observation_tensor, goal_mapping, action_prior, theta_prior = create_epistemic_maze_tensors(
        n_theta=args.n_theta,
        cue_accuracy=cue_accuracy,
        location_observation_accuracy=location_accuracy,
        goal_temperature=goal_temperature,
        cue_cost_epsilon=cue_cost_epsilon,
    )
    observation_tensor = theta_observation_tensor

    env = EpistemicMaze.create(
        theta=None,
        start_location=0,
        n_theta=args.n_theta,
        cue_accuracy=cue_accuracy,
        location_accuracy=location_accuracy,
    )

    config = DiagnosticConfig(
        planning_horizon=horizon,
        max_steps=max_steps,
        n_theta=args.n_theta,
        action_selection=args.action_selection,
        sophisticated=(args.mode == "sophisticated"),
        goal_temperature=goal_temperature,
    )

    print("=" * 70)
    print(f"MODE: {args.mode}")
    print(f"n_theta: {args.n_theta}  horizon: {horizon}  max_steps: {max_steps}")
    print(f"goal_temperature: {goal_temperature}  cue_accuracy: {cue_accuracy}")
    print(f"location_accuracy: {location_accuracy}  cue_cost_epsilon: {cue_cost_epsilon}")
    print("=" * 70)

    run_episode_diagnostics(
        env=env,
        transition_tensor=transition_tensor,
        observation_tensor=observation_tensor,
        location_observation_tensor=location_observation_tensor,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        config=config,
    )


if __name__ == "__main__":
    main()
