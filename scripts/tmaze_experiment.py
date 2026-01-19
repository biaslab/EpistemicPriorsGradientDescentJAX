#!/usr/bin/env python3
"""
T-Maze experiment using JAX-based planning via VFE minimization.

Vanilla VFE: minimize E_q[log q] - E_q[log p] with a goal prior.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional
import random

import jax.numpy as jnp
from jax import Array
import yaml

from pathlib import Path
import sys

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.environments import TMaze, create_tmaze_tensors
from src.planning import plan_actions, select_action, PlanningConfig
from src.visualization import plot_tmaze_frame, create_episode_video


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
    inference_mode: str = "marginal"


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    total_reward: float
    n_steps: int
    reached_goal: bool
    trajectory: List[int]
    actions: List[int]
    final_state: int
    reward_location: str = ""  # 'left' or 'right'


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
            inference_mode=config.inference_mode,
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
        reward_location=env.reward_location,
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


def save_results(
    config: ExperimentConfig,
    mean_reward: float,
    success_rate: float,
    results: List[EpisodeResult],
    output_dir: Path,
) -> Path:
    """Save experiment results to JSON file."""
    # Determine inference type tag
    inference_type = config.inference_mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute aggregate stats
    cue_visits = sum(1 for r in results if 0 in r.trajectory[1:])
    avg_steps = sum(r.n_steps for r in results) / len(results)
    
    # Prepare results data
    results_data = {
        "metadata": {
            "inference_type": inference_type,
            "timestamp": timestamp,
            "config": asdict(config),
        },
        "summary": {
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "cue_visits": cue_visits,
            "avg_steps": avg_steps,
        },
        "episodes": [asdict(r) for r in results],
    }
    
    # Save full results JSON file (fixed filename for DVC)
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    
    # Save aggregate stats separately (for easy comparison)
    stats_data = {
        "inference_type": inference_type,
        "n_episodes": config.n_episodes,
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "cue_visits": cue_visits,
        "cue_visit_rate": cue_visits / len(results),
        "avg_steps": avg_steps,
        "seed": config.seed,
    }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"Stats saved to: {stats_path}")
    
    return json_path


def create_last_episode_video(
    result: EpisodeResult,
    output_dir: Path,
    inference_type: str,
) -> Path:
    """Create a video of the last episode."""
    # Fixed filename for DVC
    video_path = output_dir / "episode.mp4"
    
    create_episode_video(
        trajectory=result.trajectory,
        actions=result.actions,
        reward_location=result.reward_location,
        output_path=video_path,
        fps=2,
    )
    
    return video_path


def load_params_from_yaml(params_path: Path) -> dict:
    """Load experiment parameters from params.yaml if it exists."""
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('experiment', {})
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run T-maze VFE planning experiment")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--planning-horizon", type=int, default=None)
    parser.add_argument("--no-receding-horizon", action="store_true")
    parser.add_argument("--n-opt-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--inference-mode", type=str, default="marginal",
                        choices=["marginal", "active", "planning"],
                        help="Inference mode: marginal, active, or planning")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for results and videos")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video generation")
    
    args = parser.parse_args()
    
    # Load defaults from params.yaml (for DVC pipeline)
    params_path = script_dir / "params.yaml"
    yaml_params = load_params_from_yaml(params_path)
    
    # Merge: CLI args override yaml params, yaml params override hardcoded defaults
    defaults = {
        'n_episodes': 50,
        'max_steps': 4,
        'planning_horizon': 4,
        'receding_horizon': True,
        'n_optimization_steps': 100,
        'learning_rate': 0.1,
        'seed': 42,
    }
    
    config = ExperimentConfig(
        n_episodes=args.n_episodes if args.n_episodes is not None else yaml_params.get('n_episodes', defaults['n_episodes']),
        max_steps=args.max_steps if args.max_steps is not None else yaml_params.get('max_steps', defaults['max_steps']),
        planning_horizon=args.planning_horizon if args.planning_horizon is not None else yaml_params.get('planning_horizon', defaults['planning_horizon']),
        receding_horizon=not args.no_receding_horizon if args.no_receding_horizon else yaml_params.get('receding_horizon', defaults['receding_horizon']),
        n_optimization_steps=args.n_opt_steps if args.n_opt_steps is not None else yaml_params.get('n_optimization_steps', defaults['n_optimization_steps']),
        learning_rate=args.learning_rate if args.learning_rate is not None else yaml_params.get('learning_rate', defaults['learning_rate']),
        seed=args.seed if args.seed is not None else yaml_params.get('seed', defaults['seed']),
        verbose=args.verbose,
        inference_mode=args.inference_mode,
    )
    
    # Determine inference type for display
    inference_type_display = {"marginal": "Marginal Inference", "active": "Active Inference", "planning": "Planning Inference"}
    inference_type = inference_type_display.get(config.inference_mode, config.inference_mode)
    
    print("=" * 60)
    print(f"T-Maze VFE Planning - {inference_type}")
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
    
    # Save results to disk
    output_dir = Path(args.output_dir)
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    save_results(config, mean_reward, success_rate, results, output_dir)
    
    # Create video of last episode
    if not args.no_video and results:
        inference_tag = config.inference_mode
        try:
            video_path = create_last_episode_video(results[-1], output_dir, inference_tag)
            print(f"Video saved to: {video_path}")
        except ImportError as e:
            print(f"Warning: Could not create video - {e}")
            print("Install imageio with: pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()
