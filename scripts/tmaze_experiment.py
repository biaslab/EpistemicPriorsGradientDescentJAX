#!/usr/bin/env python3
"""
T-Maze experiment using JAX-based planning via VFE minimization.

Vanilla VFE: minimize E_q[log q] - E_q[log p] with a goal prior.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import random

import jax.numpy as jnp
from jax import Array

from pathlib import Path
import sys

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.environments import TMaze, create_tmaze_tensors
from src.planning import plan_actions, select_action, PlanningConfig


@dataclass
class ExperimentConfig:
    """Configuration for TMaze experiment."""
    n_episodes: int = 50
    max_steps: int = 4
    planning_horizon: int = 4
    receding_horizon: bool = True
    n_optimization_steps: int = 200
    learning_rate: float = 0.1
    seed: int = 42
    verbose: bool = False
    marginal_inference: bool = False


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    total_reward: float
    n_steps: int
    reached_goal: bool
    trajectory: List[int]
    actions: List[int]
    final_state: int


def run_episode(
    env: TMaze,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    config: ExperimentConfig,
) -> EpisodeResult:
    """Run a single episode in the T-maze."""
    total_reward = 0.0
    trajectory = [env.agent_state]
    actions = []
    
    for step in range(config.max_steps):
        time_remaining = config.max_steps - step
        if config.receding_horizon:
            effective_horizon = min(time_remaining, config.planning_horizon)
        else:
            effective_horizon = config.planning_horizon
        
        if effective_horizon <= 0:
            break
        
        prior_state = jnp.zeros(5).at[env.agent_state].set(1.0)
        
        # Reward location prior based on cue observation
        if env.has_seen_cue:
            if env.reward_location == 'left':
                prior_reward_location = jnp.array([1.0, 0.0])
            else:
                prior_reward_location = jnp.array([0.0, 1.0])
        else:
            prior_reward_location = jnp.array([0.5, 0.5])
        
        planning_config = PlanningConfig(
            planning_horizon=effective_horizon,
            n_states=5,
            n_actions=4,
            n_reward_locs=2,
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            verbose=False,
            marginal_inference=config.marginal_inference,
        )
        result = plan_actions(
            prior_state=prior_state,
            prior_reward_location=prior_reward_location,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            config=planning_config,
        )
        action = select_action(result)
        actions.append(action)
        
        if config.verbose:
            print(f"  Step {step}: State={env.agent_state}, Action={action}, horizon={effective_horizon}")
            print(f"    q(u1)={result.first_action_probs}")
            print(f"    q(r)={result.reward_location_probs}")
        
        _, _, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(env.agent_state)
        
        if done:
            break
    
    return EpisodeResult(
        total_reward=total_reward,
        n_steps=len(actions),
        reached_goal=(total_reward > 0),
        trajectory=trajectory,
        actions=actions,
        final_state=env.agent_state,
    )


def run_experiment(config: ExperimentConfig) -> Tuple[float, float, List[EpisodeResult]]:
    """Run the full T-maze experiment."""
    random.seed(config.seed)
    
    transition_tensor, observation_tensor, goal_mapping = create_tmaze_tensors()
    
    results = []
    
    for episode in range(config.n_episodes):
        env = TMaze.create(reward_location=None, start_state=1)
        
        if config.verbose:
            print(f"\nEpisode {episode + 1}: Reward at {env.reward_location}")
        
        result = run_episode(
            env=env,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            config=config,
        )
        results.append(result)
        
        if config.verbose:
            print(f"  Result: reward={result.total_reward}, steps={result.n_steps}, goal={result.reached_goal}")
    
    rewards = [r.total_reward for r in results]
    mean_reward = sum(rewards) / len(rewards)
    success_rate = sum(1 for r in results if r.reached_goal) / len(results)
    
    return mean_reward, success_rate, results


def main():
    parser = argparse.ArgumentParser(description="Run T-maze VFE planning experiment")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--planning-horizon", type=int, default=4)
    parser.add_argument("--no-receding-horizon", action="store_true")
    parser.add_argument("--n-opt-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--marginal-inference", action="store_true",
                        help="Skip epistemic state energy term (marginal inference only)")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        planning_horizon=args.planning_horizon,
        receding_horizon=not args.no_receding_horizon,
        n_optimization_steps=args.n_opt_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=args.verbose,
        marginal_inference=args.marginal_inference,
    )
    
    print("=" * 60)
    print("T-Maze VFE Planning")
    print("=" * 60)
    print(f"Episodes: {config.n_episodes}")
    print(f"Max steps: {config.max_steps}")
    print(f"Planning horizon: {config.planning_horizon}")
    print(f"Optimization steps: {config.n_optimization_steps}")
    print("=" * 60)
    
    mean_reward, success_rate, results = run_experiment(config)
    
    print("\nRESULTS")
    print("=" * 60)
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    
    cue_visits = sum(1 for r in results if 0 in r.trajectory[1:])
    print(f"Cue visits: {cue_visits}/{len(results)}")
    print(f"Avg steps: {sum(r.n_steps for r in results) / len(results):.2f}")


if __name__ == "__main__":
    main()
