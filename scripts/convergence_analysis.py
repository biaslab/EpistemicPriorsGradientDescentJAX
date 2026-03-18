#!/usr/bin/env python3
"""
Convergence analysis for T-Maze VFE optimization.

Dispatcher script with 5 analysis functions that produce JSON data files
consumed by plot_convergence.py.
"""

import argparse
import json
import random
from pathlib import Path
import sys

import jax.numpy as jnp
import yaml

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.environments import TMaze, create_tmaze_tensors
from src.planning import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
)

MODES = ["active", "marginal", "planning"]


def load_params():
    """Load convergence parameters from params.yaml."""
    params_path = script_dir / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)["convergence"]


def run_episodes_with_losses(
    n_episodes, n_optimization_steps, learning_rate,
    planning_horizon, max_steps, seed, inference_mode,
):
    """Run episodes capturing loss histories at every planning step.

    Mimics run_episode() from tmaze_experiment.py but records
    result.loss_history for each planning call.
    """
    random.seed(seed)
    transition, obs, goal = create_tmaze_tensors()

    episodes = []
    for _ in range(n_episodes):
        env = TMaze.create(reward_location=None, start_state=1)
        total_reward = 0.0
        planning_steps = []
        n_actions = 0

        for step in range(max_steps):
            effective_horizon = min(max_steps - step, planning_horizon)
            if effective_horizon <= 0:
                break

            prior_state = jnp.zeros(5).at[env.agent_state].set(1.0)
            if env.has_seen_cue:
                if env.reward_location == "left":
                    prior_rl = jnp.array([1.0, 0.0])
                else:
                    prior_rl = jnp.array([0.0, 1.0])
            else:
                prior_rl = jnp.array([0.5, 0.5])

            config = FactorizedPlanningConfig(
                planning_horizon=effective_horizon,
                n_obs=2,
                n_states=5,
                n_actions=4,
                n_theta=2,
                n_optimization_steps=n_optimization_steps,
                learning_rate=learning_rate,
                inference_mode=inference_mode,
            )

            result = plan_actions_factorized(
                prior_state=prior_state,
                prior_reward_location=prior_rl,
                transition_tensor=transition,
                observation_tensor=obs,
                goal_mapping=goal,
                config=config,
            )

            action = select_action_factorized(result)
            planning_steps.append({"all_losses": result.loss_history})
            n_actions += 1

            _, _, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        episodes.append({
            "total_reward": total_reward,
            "reached_goal": total_reward > 0,
            "n_steps": n_actions,
            "cue_visited": env.has_seen_cue,
            "planning_steps": planning_steps,
        })

    return episodes


def run_aggregate_stats(
    n_episodes, n_optimization_steps, learning_rate,
    planning_horizon, max_steps, seed, inference_mode,
):
    """Run episodes and return aggregate statistics."""
    episodes = run_episodes_with_losses(
        n_episodes, n_optimization_steps, learning_rate,
        planning_horizon, max_steps, seed, inference_mode,
    )

    success_rate = sum(1 for e in episodes if e["reached_goal"]) / len(episodes)
    mean_reward = sum(e["total_reward"] for e in episodes) / len(episodes)
    cue_visit_rate = sum(1 for e in episodes if e["cue_visited"]) / len(episodes)

    final_losses = []
    for e in episodes:
        for ps in e["planning_steps"]:
            final_losses.append(ps["all_losses"][-1])

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "cue_visit_rate": cue_visit_rate,
        "final_losses": final_losses,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analysis_curves(output_dir):
    """Per-episode loss histories for all modes."""
    params = load_params()["curves"]

    results = {}
    for mode in MODES:
        print(f"  Running {mode}...")
        results[mode] = run_episodes_with_losses(
            n_episodes=params["n_episodes"],
            n_optimization_steps=params["n_optimization_steps"],
            learning_rate=params["learning_rate"],
            planning_horizon=params["planning_horizon"],
            max_steps=params["max_steps"],
            seed=params["seed"],
            inference_mode=mode,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "curves.json"
    with open(path, "w") as f:
        json.dump({"params": params, "results": results}, f, indent=2)
    print(f"Saved: {path}")


def analysis_lr_sweep(output_dir):
    """Learning-rate sensitivity sweep."""
    params = load_params()["lr_sweep"]

    results = {}
    for mode in MODES:
        mode_results = []
        for lr in params["learning_rates"]:
            print(f"  Running {mode} lr={lr}...")
            stats = run_aggregate_stats(
                n_episodes=params["n_episodes"],
                n_optimization_steps=params["n_optimization_steps"],
                learning_rate=lr,
                planning_horizon=params["planning_horizon"],
                max_steps=params["max_steps"],
                seed=params["seed"],
                inference_mode=mode,
            )
            mode_results.append({"learning_rate": lr, **stats})
        results[mode] = mode_results

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lr_sweep.json"
    with open(path, "w") as f:
        json.dump({"params": params, "results": results}, f, indent=2)
    print(f"Saved: {path}")


def analysis_budget(output_dir):
    """Compute-budget (optimization steps) analysis."""
    params = load_params()["budget"]

    results = {}
    for mode in MODES:
        mode_results = []
        for budget in params["budgets"]:
            print(f"  Running {mode} budget={budget}...")
            stats = run_aggregate_stats(
                n_episodes=params["n_episodes"],
                n_optimization_steps=budget,
                learning_rate=params["learning_rate"],
                planning_horizon=params["planning_horizon"],
                max_steps=params["max_steps"],
                seed=params["seed"],
                inference_mode=mode,
            )
            mode_results.append({"n_opt_steps": budget, **stats})
        results[mode] = mode_results

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "budget.json"
    with open(path, "w") as f:
        json.dump({"params": params, "results": results}, f, indent=2)
    print(f"Saved: {path}")


def analysis_variance(output_dir):
    """Seed-variance analysis across multiple random seeds."""
    params = load_params()["variance"]

    results = {}
    for mode in MODES:
        mode_results = []
        for seed in params["seeds"]:
            print(f"  Running {mode} seed={seed}...")
            stats = run_aggregate_stats(
                n_episodes=params["n_episodes"],
                n_optimization_steps=params["n_optimization_steps"],
                learning_rate=params["learning_rate"],
                planning_horizon=params["planning_horizon"],
                max_steps=params["max_steps"],
                seed=seed,
                inference_mode=mode,
            )
            mode_results.append({"seed": seed, **stats})
        results[mode] = mode_results

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "variance.json"
    with open(path, "w") as f:
        json.dump({"params": params, "results": results}, f, indent=2)
    print(f"Saved: {path}")


def analysis_policy_stability(output_dir):
    """Policy stability: p(u_1) vs optimization budget for a single planning call."""
    params = load_params()["policy_stability"]

    transition, obs, goal = create_tmaze_tensors()
    prior_state = jnp.zeros(5).at[1].set(1.0)  # MIDDLE
    prior_theta = jnp.array([0.5, 0.5])  # uniform

    results = {}
    for mode in MODES:
        mode_results = []
        for budget in params["budgets"]:
            print(f"  Running {mode} budget={budget}...")
            config = FactorizedPlanningConfig(
                planning_horizon=params["planning_horizon"],
                n_obs=2,
                n_states=5,
                n_actions=4,
                n_theta=2,
                n_optimization_steps=budget,
                learning_rate=params["learning_rate"],
                inference_mode=mode,
                init_seed=params["seed"],
            )

            result = plan_actions_factorized(
                prior_state=prior_state,
                prior_reward_location=prior_theta,
                transition_tensor=transition,
                observation_tensor=obs,
                goal_mapping=goal,
                config=config,
            )

            mode_results.append({
                "n_opt_steps": budget,
                "first_action_probs": [float(p) for p in result.first_action_probs],
            })
        results[mode] = mode_results

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "policy_stability.json"
    with open(path, "w") as f:
        json.dump({"params": params, "results": results}, f, indent=2)
    print(f"Saved: {path}")


def analysis_scenario_curves(output_dir):
    """Single planning calls under different theta priors."""
    params = load_params()["scenario_curves"]

    transition, obs, goal = create_tmaze_tensors()
    prior_state = jnp.zeros(5).at[1].set(1.0)  # MIDDLE

    scenarios = {
        "theta_unknown": jnp.array([0.5, 0.5]),
        "theta_known": jnp.array([0.95, 0.05]),
    }

    results = {}
    for scenario_name, prior_theta in scenarios.items():
        scenario_results = {}
        for mode in MODES:
            print(f"  Running {scenario_name} {mode}...")
            config = FactorizedPlanningConfig(
                planning_horizon=params["planning_horizon"],
                n_obs=2,
                n_states=5,
                n_actions=4,
                n_theta=2,
                n_optimization_steps=params["n_optimization_steps"],
                learning_rate=params["learning_rate"],
                inference_mode=mode,
                init_seed=params["seed"],
            )

            result = plan_actions_factorized(
                prior_state=prior_state,
                prior_reward_location=prior_theta,
                transition_tensor=transition,
                observation_tensor=obs,
                goal_mapping=goal,
                config=config,
            )

            scenario_results[mode] = {
                "all_losses": result.loss_history,
                "final_loss": float(result.final_loss),
            }
        results[scenario_name] = scenario_results

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scenario_curves.json"
    with open(path, "w") as f:
        json.dump({"params": params, "scenarios": results}, f, indent=2)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

ANALYSES = {
    "curves": analysis_curves,
    "lr_sweep": analysis_lr_sweep,
    "budget": analysis_budget,
    "variance": analysis_variance,
    "policy_stability": analysis_policy_stability,
    "scenario_curves": analysis_scenario_curves,
}


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for T-Maze VFE")
    parser.add_argument(
        "--analysis", required=True, choices=ANALYSES.keys(),
        help="Which analysis to run",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/convergence",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Running {args.analysis} analysis...")
    ANALYSES[args.analysis](output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
