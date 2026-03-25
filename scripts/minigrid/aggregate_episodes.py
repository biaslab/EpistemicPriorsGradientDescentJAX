#!/usr/bin/env python3
"""Combine per-episode JSON files into a single results.json."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-episode results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing episodes/ and config.json")
    parser.add_argument("--n-episodes", type=int, required=True,
                        help="Expected number of episodes")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow incomplete runs (missing episodes)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes_dir = output_dir / "episodes"
    config_path = output_dir / "config.json"

    # Load config
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    # Load per-episode files
    episodes = []
    missing = []
    for i in range(args.n_episodes):
        ep_path = episodes_dir / f"episode_{i:04d}.json"
        if ep_path.exists():
            with open(ep_path) as f:
                episodes.append(json.load(f))
        else:
            missing.append(i)

    if missing:
        print(f"Warning: {len(missing)} missing episodes: {missing}", file=sys.stderr)
        if not args.allow_partial:
            print("Use --allow-partial to aggregate anyway", file=sys.stderr)
            sys.exit(1)

    if not episodes:
        print("Error: no episode files found", file=sys.stderr)
        sys.exit(1)

    # Sort by episode index
    episodes.sort(key=lambda e: e["episode_index"])

    # Compute summary stats
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    total_steps = sum(e["steps"] for e in episodes)
    total_reward = sum(e["total_reward"] for e in episodes)
    total_wall_clock = sum(e["wall_clock_s"] for e in episodes)

    # Build episode_results in the same format as the monolithic run
    episode_results = [
        {
            "total_reward": e["total_reward"],
            "steps": e["steps"],
            "success": e["success"],
            "terminated": e["terminated"],
            "truncated": e["truncated"],
            "wall_clock_s": e["wall_clock_s"],
        }
        for e in episodes
    ]

    save_data = {
        "config": config,
        "summary": {
            "success_rate": successes / n,
            "successes": successes,
            "n_episodes": n,
            "avg_steps": total_steps / n,
            "avg_reward": total_reward / n,
        },
        "timing": {
            "total_wall_clock_s": total_wall_clock,
            "avg_wall_clock_s_per_episode": total_wall_clock / n,
        },
        "episodes": episode_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Aggregated {n}/{args.n_episodes} episodes")
    print(f"  Success rate: {successes/n:.1%} ({successes}/{n})")
    print(f"  Avg steps: {total_steps/n:.1f}")
    print(f"  Avg reward: {total_reward/n:.3f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
