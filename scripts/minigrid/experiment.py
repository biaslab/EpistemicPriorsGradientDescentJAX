#!/usr/bin/env python3
"""MiniGrid DoorKey experiment using temporal VFE planning."""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from src.environments.minigrid import create_minigrid_env_tensors
from src.environments.gym_wrapper import run_experiment, run_episode, MiniGridWrapper, save_video, save_frames
from src.agents import create_temporal_vfe_agent


def parse_record_spec(spec: str, n_episodes: int) -> list[int]:
    """Parse record specification like 'first,last,0,9' into episode indices."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if part == "first":
            indices.append(0)
        elif part == "last":
            indices.append(n_episodes - 1)
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _build_config(args) -> dict:
    """Build the config dict from CLI args (shared between monolithic and single-episode modes)."""
    return {
        "grid_size": args.grid_size,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "planning_horizon": args.planning_horizon,
        "receding_horizon": args.receding_horizon,
        "n_opt_steps": args.n_opt_steps,
        "learning_rate": args.learning_rate,
        "inference_mode": args.inference_mode,
        "fov_size": args.fov_size,
        "seed": args.seed,
        "goal_scale": args.goal_scale,
    }


def run_single_episode(args, agent, env_name, output_dir, video_dir, record_episodes):
    """Run a single episode and write per-episode JSON. For SLURM array jobs."""
    i = args.episode_index
    seed = args.seed + i

    if output_dir is None:
        print("Error: --output-dir is required with --episode-index", file=sys.stderr)
        sys.exit(1)

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episodes_dir / f"episode_{i:04d}.json"

    # Skip if output file already exists (resume support)
    if episode_path.exists():
        print(f"Episode {i} already complete, skipping ({episode_path})")
        return

    # Determine if this episode should be recorded
    should_record = record_episodes is not None and i in record_episodes
    render_mode = "rgb_array" if should_record else None

    env = MiniGridWrapper(
        env_name=env_name,
        render_mode=render_mode,
        max_steps=args.max_steps,
        fov_size=args.fov_size,
    )

    print(f"Running episode {i} (seed={seed})...")
    result = run_episode(
        agent=agent,
        env=env,
        seed=seed,
        receding_horizon=args.receding_horizon,
        verbose=args.verbose,
        record=should_record,
        no_orientation=args.no_orientation,
    )
    env.close()

    # Save recording if applicable
    if should_record and "frames" in result and video_dir:
        frames = result.pop("frames")
        frames_dir = str(Path(video_dir) / f"frames_episode_{i:03d}")
        save_frames(frames, frames_dir, i)
        video_path = str(Path(video_dir) / f"episode_{i:03d}.mp4")
        save_video(frames, video_path)

    # Write per-episode JSON
    episode_data = {
        "episode_index": i,
        "seed": seed,
        "total_reward": result["total_reward"],
        "steps": result["steps"],
        "success": result["success"],
        "terminated": result["terminated"],
        "truncated": result["truncated"],
        "wall_clock_s": result["wall_clock_s"],
    }
    with open(episode_path, "w") as f:
        json.dump(episode_data, f, indent=2)
    print(f"Episode {i}: success={result['success']}, steps={result['steps']}, "
          f"reward={result['total_reward']:.3f}, time={result['wall_clock_s']:.1f}s")
    print(f"Saved to {episode_path}")

    # Write config.json (idempotent — all array tasks write identical content)
    config_path = output_dir / "config.json"
    config_data = _build_config(args)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MiniGrid DoorKey experiment")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="Internal grid size (env will be grid_size+2 x grid_size+2)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--planning-horizon", type=int, default=15, help="Lookahead depth")
    parser.add_argument("--receding-horizon", action="store_true",
                        help="Decrease horizon as time runs out")
    parser.add_argument("--n-opt-steps", type=int, default=2000,
                        help="Temporal VFE optimization steps")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Optimizer learning rate")
    parser.add_argument("--optimizer-type", type=str, default="adam",
                        choices=["adam", "adafactor"], help="Optimizer type")
    parser.add_argument("--inference-mode", choices=["marginal", "active", "planning"],
                        default="active", help="Inference mode")
    parser.add_argument("--fov-size", type=int, default=7,
                        help="Field-of-view size (odd, >= 3)")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Replace orientation obs with uniform")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Debug output")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (saves results.json + recordings/)")
    parser.add_argument("--record", type=str, default=None,
                        help='Record episodes (e.g. "first,last,0,9")')
    parser.add_argument("--freeze-obs-and-transitions", action="store_true",
                        help="Freeze observation and transition variational factors")
    parser.add_argument("--policy-init-scale", type=float, default=1.0,
                        help="Scale for policy logit initialization")
    parser.add_argument("--goal-scale", type=float, default=1.0,
                        help="Scale for goal energy term")
    parser.add_argument("--episode-index", type=int, default=None,
                        help="Run only this episode (0-based). For SLURM array jobs.")

    args = parser.parse_args()

    # Validate fov_size
    if args.fov_size < 3 or args.fov_size % 2 == 0:
        parser.error("--fov-size must be odd and >= 3")

    print(f"Configuration:")
    print(f"  Grid size: {args.grid_size} (env: {args.grid_size + 2}x{args.grid_size + 2})")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Planning horizon: {args.planning_horizon}")
    print(f"  Receding horizon: {args.receding_horizon}")
    print(f"  Inference mode: {args.inference_mode}")
    print(f"  FOV size: {args.fov_size}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate environment tensors
    print("Generating environment tensors...")
    env_tensors = create_minigrid_env_tensors(
        n=args.grid_size, fov_size=args.fov_size,
    )
    print(f"  States: {env_tensors.n_states}, Actions: {env_tensors.n_actions}, "
          f"Theta: {env_tensors.n_theta}, Modalities: {len(env_tensors.observation_modalities)}")

    # Create agent
    agent = create_temporal_vfe_agent(
        env_tensors=env_tensors,
        planning_horizon=args.planning_horizon,
        n_optimization_steps=args.n_opt_steps,
        learning_rate=args.learning_rate,
        inference_mode=args.inference_mode,
        init_seed=args.seed,
        fov_size=args.fov_size,
        fov_pattern_map=env_tensors.metadata.get("fov_pattern_map"),
        freeze_obs_and_transitions=args.freeze_obs_and_transitions,
        policy_init_scale=args.policy_init_scale,
        goal_scale=args.goal_scale,
        optimizer_type=args.optimizer_type,
    )
    print(f"  Planning method: temporal ({args.inference_mode})")

    # Environment name
    env_size = args.grid_size + 2
    env_name = f"MiniGrid-DoorKey-{env_size}x{env_size}-v0"

    # Resolve output paths
    output_dir = Path(args.output_dir) if args.output_dir else None
    video_dir = str(output_dir / "recordings") if output_dir else None

    # Parse record spec
    record_episodes = None
    if args.record:
        record_episodes = parse_record_spec(args.record, args.episodes)
        print(f"  Recording episodes: {record_episodes}")

    # --- Single-episode mode (for SLURM array jobs) ---
    if args.episode_index is not None:
        run_single_episode(args, agent, env_name, output_dir, video_dir, record_episodes)
        return

    # --- Monolithic mode (original behavior) ---
    print(f"\nRunning {args.episodes} episodes on {env_name}...")
    experiment_start = time.perf_counter()
    stats = run_experiment(
        agent=agent,
        env_name=env_name,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        receding_horizon=args.receding_horizon,
        seed_start=args.seed,
        verbose=args.verbose,
        record_episodes=record_episodes,
        video_dir=video_dir if record_episodes else None,
        fov_size=args.fov_size,
        no_orientation=args.no_orientation,
    )

    experiment_wall_clock_s = time.perf_counter() - experiment_start

    # Print results
    print(f"\nResults:")
    print(f"  Success rate: {stats['success_rate']:.1%} ({stats['successes']}/{stats['n_episodes']})")
    print(f"  Avg steps: {stats['avg_steps']:.1f}")
    print(f"  Avg reward: {stats['avg_reward']:.3f}")
    print(f"  Wall clock: {experiment_wall_clock_s:.1f}s total, "
          f"{stats['avg_wall_clock_s_per_episode']:.1f}s/episode")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.json"

        save_data = {
            "config": {
                "grid_size": args.grid_size,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "planning_horizon": args.planning_horizon,
                "receding_horizon": args.receding_horizon,
                "n_opt_steps": args.n_opt_steps,
                "learning_rate": args.learning_rate,
                "inference_mode": args.inference_mode,
                "fov_size": args.fov_size,
                "seed": args.seed,
                "goal_scale": args.goal_scale,
            },
            "summary": {
                "success_rate": stats["success_rate"],
                "successes": stats["successes"],
                "n_episodes": stats["n_episodes"],
                "avg_steps": stats["avg_steps"],
                "avg_reward": stats["avg_reward"],
            },
            "timing": {
                "total_wall_clock_s": stats["total_wall_clock_s"],
                "avg_wall_clock_s_per_episode": stats["avg_wall_clock_s_per_episode"],
            },
            "episodes": stats["episode_results"],
        }
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
