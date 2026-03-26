#!/usr/bin/env python3
"""
T-Maze experiment using JAX-based planning via VFE minimization.

Vanilla VFE: minimize E_q[log q] - E_q[log p] with a goal prior.
"""

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Tuple, Optional
import random

import jax.numpy as jnp
from jax import Array
import yaml
from tqdm import tqdm

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments import TMaze, create_tmaze_tensors
from src.planning import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    SophisticatedPlanningConfig,
    SophisticatedPlanner,
    convert_tmaze_tensors_to_pymdp,
)
from src.visualization import (
    plot_tmaze_frame,
    create_episode_video,
    save_episode_tikz_frames,
    save_tmaze_reference,
    PlanData,
)


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
class PlanningSnapshot:
    """Snapshot of planning state at a single time step."""
    all_action_probs: List[List[float]]  # (horizon, n_actions) - q(u_t) for all future t
    all_state_probs: List[List[float]]   # (horizon, n_states) - q(x_t) for all future t


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
    planning_history: Optional[List[PlanningSnapshot]] = None  # Planning snapshots at each step


def get_observation_tmaze(env: TMaze, is_final_step: bool = False) -> list:
    """
    Convert T-Maze environment state to pymdp observation indices.

    Returns list of [location_obs_idx, cue_obs_idx, reward_obs_idx].
    """
    # Location observation: deterministic (T-Maze has perfect location obs)
    location_obs_idx = env.agent_state

    # Cue observation: informative at BOTTOM (state 0), ambiguous elsewhere
    if env.agent_state == 0:  # BOTTOM/cue location
        cue_obs_idx = 0 if env.reward_location == 'left' else 1
    else:
        # Ambiguous: both cue indices are equally likely (A[1] = 0.5 at non-cue locations)
        cue_obs_idx = random.randint(0, 1)

    # Reward observation: only reveal at final step
    if is_final_step:
        if env.agent_state == 2:  # TOP_LEFT
            reward_obs_idx = 1 if env.reward_location == 'left' else 2
        elif env.agent_state == 4:  # TOP_RIGHT
            reward_obs_idx = 1 if env.reward_location == 'right' else 2
        else:
            reward_obs_idx = 0  # no reward
    else:
        reward_obs_idx = None  # skip reward belief update during episode

    return [location_obs_idx, cue_obs_idx, reward_obs_idx]


def run_episode_pymdp(
    env: TMaze,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    config: SophisticatedPlanningConfig,
    max_steps: int = 4,
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single T-Maze episode using pymdp active inference."""
    import numpy as np

    trajectory = [env.agent_state]
    actions = []
    total_reward = 0.0

    # Convert tensors to pymdp format
    theta_prior = jnp.array([0.5, 0.5])
    A, B, C, D = convert_tmaze_tensors_to_pymdp(
        transition_tensor=transition_tensor,
        reward_obs_tensor=observation_tensor,
        goal_mapping=goal_mapping,
        theta_prior=theta_prior,
        goal_temperature=config.goal_temperature,
    )

    # Create planner
    planner = SophisticatedPlanner.from_pymdp_arrays(A, B, C, D, config)
    planner.agent.reset()

    # Get initial observation
    obs = get_observation_tmaze(env, is_final_step=False)

    for step in range(max_steps):
        # 1) Infer states from observation
        planner.infer_states(obs)

        # Receding horizon
        remaining = min(max_steps - step, config.inference_horizon)
        if config.sophisticated:
            if hasattr(planner.agent, "si_horizon"):
                planner.agent.si_horizon = remaining
        else:
            planner.rebuild_policies(remaining)

        # 2) Infer policies (compute EFE)
        q_pi, efe = planner.infer_policies()

        # 3) Sample action
        action_arr = planner.sample_action()
        action = int(action_arr[0])

        if verbose:
            q_theta = planner.agent.qs[1]
            print(f"  Step {step}: State={env.agent_state}, Action={action}")
            print(f"    q(theta)={q_theta}")

        # 4) Execute action
        _, _, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(env.agent_state)
        actions.append(action)

        # 5) Get new observation
        is_final = (step == max_steps - 1)
        obs = get_observation_tmaze(env, is_final_step=is_final)

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


def run_episode(
    env: TMaze,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    config: ExperimentConfig,
    record_planning: bool = False,
) -> EpisodeResult:
    """Run a single episode in the T-maze."""
    total_reward = 0.0
    trajectory = [env.agent_state]
    actions = []
    planning_history = [] if record_planning else None
    
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
        
        planning_config = FactorizedPlanningConfig(
            planning_horizon=effective_horizon,
            n_obs=2,  # T-maze has 2 observation types (cue for left/right)
            n_states=5,
            n_actions=4,
            n_theta=2,  # 2 possible reward locations (left/right)
            n_optimization_steps=config.n_optimization_steps,
            learning_rate=config.learning_rate,
            verbose=False,
            inference_mode=config.inference_mode,
        )
        result = plan_actions_factorized(
            prior_state=prior_state,
            prior_reward_location=prior_reward_location,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            config=planning_config,
        )
        action = select_action_factorized(result)
        actions.append(action)
        
        # Record planning snapshot for visualization
        if record_planning:
            snapshot = PlanningSnapshot(
                all_action_probs=result.all_action_probs.tolist(),
                all_state_probs=result.all_state_probs.tolist(),
            )
            planning_history.append(snapshot)
        
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
        planning_history=planning_history,
    )


def run_experiment(
    config: ExperimentConfig,
    record_planning_for_last: bool = False,
    strategy: str = "factorized",
) -> Tuple[float, float, List[EpisodeResult]]:
    """Run the full T-maze experiment.

    Args:
        strategy: Planning strategy to use:
            - "factorized": Factorized VFE (gradient-based, with inference_mode)
            - "sophisticated": pymdp tree-search (EFE)
            - "vanilla": pymdp single-step EFE
    """
    random.seed(config.seed)

    transition_tensor, observation_tensor, goal_mapping = create_tmaze_tensors()

    # Build pymdp config if needed
    pymdp_config = None
    if strategy in ("sophisticated", "vanilla"):
        is_sophisticated = (strategy == "sophisticated")
        pymdp_config = SophisticatedPlanningConfig(
            planning_horizon=config.max_steps,
            n_states=5,
            n_actions=4,
            n_theta=2,
            policy_len=1 if is_sophisticated else config.max_steps,
            inference_horizon=config.max_steps,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            action_selection="deterministic",
            gamma=16.0,
            sophisticated=is_sophisticated,
            include_reward_modality=True,
            goal_temperature=1.0,
        )

    results = []
    desc = strategy if strategy != "factorized" else config.inference_mode

    for episode in tqdm(range(config.n_episodes), desc=f"Running {desc} episodes", disable=config.verbose):
        env = TMaze.create(reward_location=None, start_state=1)

        if config.verbose:
            print(f"\nEpisode {episode + 1}: Reward at {env.reward_location}")

        if strategy in ("sophisticated", "vanilla"):
            result = run_episode_pymdp(
                env=env,
                transition_tensor=transition_tensor,
                observation_tensor=observation_tensor,
                goal_mapping=goal_mapping,
                config=pymdp_config,
                max_steps=config.max_steps,
                verbose=config.verbose,
            )
        else:
            # Record planning for last episode if requested (for visualization)
            is_last_episode = (episode == config.n_episodes - 1)
            record_planning = record_planning_for_last and is_last_episode

            result = run_episode(
                env=env,
                transition_tensor=transition_tensor,
                observation_tensor=observation_tensor,
                goal_mapping=goal_mapping,
                config=config,
                record_planning=record_planning,
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


def _convert_planning_history_to_plan_data(
    planning_history: Optional[List[PlanningSnapshot]],
) -> Optional[List[PlanData]]:
    """Convert PlanningSnapshot list to PlanData list for visualization."""
    if planning_history is None:
        return None
    
    return [
        PlanData(
            all_action_probs=snapshot.all_action_probs,
            all_state_probs=snapshot.all_state_probs,
        )
        for snapshot in planning_history
    ]


def create_last_episode_video(
    result: EpisodeResult,
    output_dir: Path,
    inference_type: str,
) -> Path:
    """Create a video of the last episode."""
    video_path = output_dir / "episode.mp4"
    
    planning_history = _convert_planning_history_to_plan_data(result.planning_history)
    
    create_episode_video(
        trajectory=result.trajectory,
        actions=result.actions,
        reward_location=result.reward_location,
        output_path=video_path,
        fps=2,
        planning_history=planning_history,
    )
    
    return video_path


def create_last_episode_tikz_frames(
    result: EpisodeResult,
    output_dir: Path,
) -> Path:
    """Create TikZ frames of the last episode for LaTeX papers.
    
    Saves both versions: frame_XX.tex (no arrows) and frame_XX_arrows.tex (with arrows).
    """
    frames_dir = output_dir / "frames"
    
    planning_history = _convert_planning_history_to_plan_data(result.planning_history)
    
    save_episode_tikz_frames(
        trajectory=result.trajectory,
        actions=result.actions,
        reward_location=result.reward_location,
        output_dir=frames_dir,
        planning_history=planning_history,
    )
    
    return frames_dir


def load_params_from_yaml(params_path: Path) -> dict:
    """Load experiment parameters from params.yaml if it exists."""
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('tmaze', {}).get('experiment', {})
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
    parser.add_argument("--strategy", type=str, default="factorized",
                        choices=["factorized", "sophisticated", "vanilla"],
                        help="Planning strategy: factorized (VFE), sophisticated (pymdp tree-search), or vanilla (pymdp single-step EFE)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for results and videos")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video generation")
    parser.add_argument("--no-tikz", action="store_true",
                        help="Skip TikZ frame generation")
    
    args = parser.parse_args()
    
    # Load defaults from params.yaml (for DVC pipeline)
    params_path = project_root / "params.yaml"
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
    
    # Determine display label
    if args.strategy in ("sophisticated", "vanilla"):
        inference_type = f"pymdp ({args.strategy})"
    else:
        inference_type_display = {"marginal": "Marginal Inference", "active": "Active Inference", "planning": "Planning Inference"}
        inference_type = inference_type_display.get(config.inference_mode, config.inference_mode)

    print("=" * 60)
    print(f"T-Maze Planning - {inference_type}")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Max steps: {config.max_steps}")
    if args.strategy == "factorized":
        print(f"Planning horizon: {config.planning_horizon}")
        print(f"Optimization steps: {config.n_optimization_steps}")
    print("=" * 60)

    # Always record planning for last episode (for visualization, factorized only)
    mean_reward, success_rate, results = run_experiment(
        config,
        record_planning_for_last=(args.strategy == "factorized"),
        strategy=args.strategy,
    )
    
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

    # Create video of last episode (only for factorized strategy which records planning)
    if not args.no_video and results and args.strategy == "factorized":
        inference_tag = config.inference_mode
        try:
            video_path = create_last_episode_video(
                results[-1], output_dir, inference_tag,
            )
            print(f"Video saved to: {video_path}")
        except ImportError as e:
            print(f"Warning: Could not create video - {e}")
            print("Install imageio with: pip install imageio[ffmpeg]")

    # Create TikZ frames for LaTeX papers (only for factorized strategy)
    if not args.no_tikz and results and args.strategy == "factorized":
        frames_dir = create_last_episode_tikz_frames(
            results[-1], output_dir,
        )
        print(f"TikZ frames saved to: {frames_dir}")

        # Save reference T-maze figure with legend (only once in data/)
        data_dir = output_dir.parent if output_dir.name in ['marginal', 'active', 'planning'] else output_dir
        tmaze_ref_path = data_dir / "tmaze.tex"
        if not tmaze_ref_path.exists():
            save_tmaze_reference(tmaze_ref_path)


if __name__ == "__main__":
    main()
