#!/usr/bin/env python3
"""
Convergence analysis for temporal VFE optimization in the Epistemic Maze.

Analyzes how hard it is to converge to a decent solution across inference modes
(marginal, active, planning) and scenarios. Four analysis modes:

  curves   — Loss vs optimization step
  lr_sweep — Learning rate sensitivity
  budget   — Performance vs compute (n_opt_steps)
  variance — Robustness across random initializations
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure scripts/epistemic_maze/ and project root are on sys.path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

import yaml
import jax.numpy as jnp

from experiment import run_experiment
from src.environments import (
    create_epistemic_maze_tensors,
    get_initial_state_distribution,
    N_STATES,
    N_ACTIONS,
)
from src.planning.temporal_optimizer import plan_actions_temporal, TemporalPlanningConfig


INFERENCE_MODES = ["marginal", "active", "planning"]


def run_for_modes(*, n_episodes, n_theta, horizon, max_steps, n_optimization_steps,
                  learning_rate, goal_temperature, cue_observation_accuracy,
                  cue_cost_epsilon, location_observation_accuracy, seed,
                  verbose=False):
    """Run experiment for all three inference modes, return dict keyed by mode."""
    results = {}
    for mode in INFERENCE_MODES:
        result = run_experiment(
            n_episodes=n_episodes,
            inference_mode=mode,
            seed=seed,
            n_theta=n_theta,
            horizon=horizon,
            max_steps=max_steps,
            n_optimization_steps=n_optimization_steps,
            learning_rate=learning_rate,
            goal_temperature=goal_temperature,
            cue_observation_accuracy=cue_observation_accuracy,
            cue_cost_epsilon=cue_cost_epsilon,
            location_observation_accuracy=location_observation_accuracy,
            verbose=verbose,
            strategy="temporal",
        )
        results[mode] = result
    return results


def extract_loss_curves(result):
    """Extract per-episode, per-planning-step all_losses from experiment result."""
    episodes_data = []
    for ep in result["episodes"]:
        steps_data = []
        for ph in ep["planning_history"]:
            steps_data.append({
                "step": ph["step"],
                "final_loss": ph["final_loss"],
                "all_losses": ph["all_losses"],
            })
        episodes_data.append({
            "theta": ep["theta"],
            "total_reward": ep["total_reward"],
            "reached_goal": ep["reached_goal"],
            "reached_safe": ep["reached_safe"],
            "outcome": ep["outcome"],
            "visited_cue": ep["visited_cue"],
            "planning_steps": steps_data,
        })
    return episodes_data


def extract_performance(result):
    """Extract aggregate performance metrics from experiment result."""
    return {
        "mean_reward": result["stats"]["mean_reward"],
        "success_rate": result["stats"]["success_rate"],
        "safe_rate": result["stats"]["safe_rate"],
        "cue_visit_rate": result["stats"]["cue_visit_rate"],
        "final_losses": [
            ph["final_loss"]
            for ep in result["episodes"]
            for ph in ep["planning_history"]
        ],
    }


def analysis_curves(args, base_config):
    """Loss vs optimization step for each inference mode."""
    print("=== Convergence Curves Analysis ===")
    mode_results = run_for_modes(**base_config, n_optimization_steps=args.n_opt_steps,
                                  learning_rate=args.learning_rate)
    output = {
        "analysis": "curves",
        "base_config": {
            "n_episodes": base_config["n_episodes"],
            "n_opt_steps": args.n_opt_steps,
            "learning_rate": args.learning_rate,
            "n_theta": base_config["n_theta"],
            "horizon": base_config["horizon"],
        },
        "results": {
            mode: extract_loss_curves(result)
            for mode, result in mode_results.items()
        },
    }
    return output, "curves.json"


def analysis_lr_sweep(args, base_config):
    """Learning rate sensitivity for each inference mode."""
    print("=== Learning Rate Sweep Analysis ===")
    learning_rates = args.learning_rates
    output = {
        "analysis": "lr_sweep",
        "base_config": {
            "n_episodes": base_config["n_episodes"],
            "n_opt_steps": args.n_opt_steps,
            "learning_rates": learning_rates,
            "n_theta": base_config["n_theta"],
            "horizon": base_config["horizon"],
        },
        "results": {},
    }
    for mode in INFERENCE_MODES:
        mode_data = []
        for lr in learning_rates:
            print(f"  {mode} lr={lr}")
            result = run_experiment(
                n_episodes=base_config["n_episodes"],
                inference_mode=mode,
                seed=base_config["seed"],
                n_theta=base_config["n_theta"],
                horizon=base_config["horizon"],
                max_steps=base_config["max_steps"],
                n_optimization_steps=args.n_opt_steps,
                learning_rate=lr,
                goal_temperature=base_config["goal_temperature"],
                cue_observation_accuracy=base_config["cue_observation_accuracy"],
                cue_cost_epsilon=base_config["cue_cost_epsilon"],
                location_observation_accuracy=base_config["location_observation_accuracy"],
                strategy="temporal",
            )
            perf = extract_performance(result)
            perf["learning_rate"] = lr
            perf["loss_curves"] = extract_loss_curves(result)
            mode_data.append(perf)
        output["results"][mode] = mode_data
    return output, "lr_sweep.json"


def analysis_budget(args, base_config):
    """Performance vs compute budget (n_opt_steps) for each inference mode."""
    print("=== Optimization Budget Analysis ===")
    budgets = args.optimization_budgets
    output = {
        "analysis": "budget",
        "base_config": {
            "n_episodes": base_config["n_episodes"],
            "optimization_budgets": budgets,
            "learning_rate": args.learning_rate,
            "n_theta": base_config["n_theta"],
            "horizon": base_config["horizon"],
        },
        "results": {},
    }
    for mode in INFERENCE_MODES:
        mode_data = []
        for n_steps in budgets:
            print(f"  {mode} n_opt_steps={n_steps}")
            result = run_experiment(
                n_episodes=base_config["n_episodes"],
                inference_mode=mode,
                seed=base_config["seed"],
                n_theta=base_config["n_theta"],
                horizon=base_config["horizon"],
                max_steps=base_config["max_steps"],
                n_optimization_steps=n_steps,
                learning_rate=args.learning_rate,
                goal_temperature=base_config["goal_temperature"],
                cue_observation_accuracy=base_config["cue_observation_accuracy"],
                cue_cost_epsilon=base_config["cue_cost_epsilon"],
                location_observation_accuracy=base_config["location_observation_accuracy"],
                strategy="temporal",
            )
            perf = extract_performance(result)
            perf["n_opt_steps"] = n_steps
            mode_data.append(perf)
        output["results"][mode] = mode_data
    return output, "budget.json"


def analysis_variance(args, base_config):
    """Robustness across random initializations for each inference mode."""
    print("=== Variance Analysis ===")
    seeds = args.seeds
    output = {
        "analysis": "variance",
        "base_config": {
            "n_episodes": base_config["n_episodes"],
            "n_opt_steps": args.n_opt_steps,
            "learning_rate": args.learning_rate,
            "seeds": seeds,
            "n_theta": base_config["n_theta"],
            "horizon": base_config["horizon"],
        },
        "results": {},
    }
    for mode in INFERENCE_MODES:
        mode_data = []
        for seed in seeds:
            print(f"  {mode} seed={seed}")
            result = run_experiment(
                n_episodes=base_config["n_episodes"],
                inference_mode=mode,
                seed=seed,
                n_theta=base_config["n_theta"],
                horizon=base_config["horizon"],
                max_steps=base_config["max_steps"],
                n_optimization_steps=args.n_opt_steps,
                learning_rate=args.learning_rate,
                goal_temperature=base_config["goal_temperature"],
                cue_observation_accuracy=base_config["cue_observation_accuracy"],
                cue_cost_epsilon=base_config["cue_cost_epsilon"],
                location_observation_accuracy=base_config["location_observation_accuracy"],
                strategy="temporal",
            )
            perf = extract_performance(result)
            perf["seed"] = seed
            mode_data.append(perf)
        output["results"][mode] = mode_data
    return output, "variance.json"


SCENARIOS = {
    "theta_unknown_knob4": {"theta_known": False, "knob": 4,
                            "label": "θ unknown, knob=4"},
    "theta_known_knob4":   {"theta_known": True,  "knob": 4,
                            "label": "θ known, knob=4"},
    "theta_unknown_knob0": {"theta_known": False, "knob": 0,
                            "label": "θ unknown, knob=0"},
    "theta_known_knob0":   {"theta_known": True,  "knob": 0,
                            "label": "θ known, knob=0"},
}


def analysis_scenario_curves(args, base_config):
    """Absolute loss curves under controlled 2×2 scenarios (single seed)."""
    print("=== Scenario Convergence Curves ===")
    n_theta = base_config["n_theta"]

    # Create environment tensors once
    tensors = create_epistemic_maze_tensors(
        n_theta=n_theta,
        cue_accuracy=base_config["cue_observation_accuracy"],
        location_observation_accuracy=base_config["location_observation_accuracy"],
        goal_temperature=base_config["goal_temperature"],
        cue_cost_epsilon=base_config["cue_cost_epsilon"],
    )
    transition, location_obs, theta_obs, goal_mapping, action_prior, theta_prior = tensors
    n_obs = n_theta + 1

    scenarios_output = {}
    for scenario_name, spec in SCENARIOS.items():
        print(f"\n  Scenario: {spec['label']}")
        initial_state = get_initial_state_distribution(
            start_location=0, start_knob=spec["knob"],
        )

        # θ prior: uniform or 95% concentrated on θ=0
        if spec["theta_known"]:
            concentrated = jnp.ones(n_theta) * (0.05 / (n_theta - 1))
            concentrated = concentrated.at[0].set(0.95)
            prior_theta_logits = jnp.log(concentrated)
        else:
            prior_theta_logits = None  # uses uniform from theta_prior

        mode_results = {}
        for mode in INFERENCE_MODES:
            config = TemporalPlanningConfig(
                planning_horizon=base_config["horizon"],
                n_states=N_STATES,
                n_actions=N_ACTIONS,
                n_theta=n_theta,
                n_obs=n_obs,
                n_optimization_steps=args.n_opt_steps,
                learning_rate=args.learning_rate,
                inference_mode=mode,
                init_seed=base_config["seed"],
            )
            print(f"    {mode} ...", end=" ", flush=True)
            result = plan_actions_temporal(
                initial_state=initial_state,
                transition_tensor=transition,
                theta_observation_tensor=theta_obs,
                goal_mapping=goal_mapping,
                action_prior=action_prior,
                theta_prior=theta_prior,
                config=config,
                prior_theta_logits=prior_theta_logits,
                location_observation_tensor=location_obs,
            )
            print(f"final_loss={result.final_loss:.4f}")
            mode_results[mode] = {
                "all_losses": result.all_losses,
                "final_loss": result.final_loss,
            }
        scenarios_output[scenario_name] = mode_results

    output = {
        "analysis": "scenario_curves",
        "base_config": {
            "n_opt_steps": args.n_opt_steps,
            "learning_rate": args.learning_rate,
            "n_theta": n_theta,
            "horizon": base_config["horizon"],
            "seed": base_config["seed"],
        },
        "scenarios": scenarios_output,
    }
    return output, "scenario_curves.json"


ANALYSES = {
    "curves": analysis_curves,
    "lr_sweep": analysis_lr_sweep,
    "budget": analysis_budget,
    "variance": analysis_variance,
    "scenario_curves": analysis_scenario_curves,
}


def main():
    parser = argparse.ArgumentParser(description="Convergence Analysis for Temporal VFE")
    parser.add_argument("--analysis", type=str, required=True,
                        choices=list(ANALYSES.keys()),
                        help="Analysis type to run")
    parser.add_argument("--output-dir", type=str, default="data/epistemic_maze/convergence")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--n-opt-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rates", type=str, default=None,
                        help="Comma-separated list of learning rates for lr_sweep")
    parser.add_argument("--optimization-budgets", type=str, default=None,
                        help="Comma-separated list of n_opt_steps for budget analysis")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--goal-temperature", type=float, default=None)
    parser.add_argument("--cue-accuracy", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load defaults from params.yaml
    params_path = project_root / "params.yaml"
    yaml_params = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            all_params = yaml.safe_load(f)
        yaml_params = all_params.get("epistemic_maze", {})
        conv_params = yaml_params.get("convergence", {})
    else:
        conv_params = {}

    # Resolve parameters: CLI > yaml > defaults
    def resolve(cli_val, yaml_dict, key, default):
        if cli_val is not None:
            return cli_val
        return yaml_dict.get(key, default)

    # Per-analysis sub-sections (matching tmaze convergence structure)
    analysis_params = conv_params.get(args.analysis, {})

    args.n_opt_steps = resolve(args.n_opt_steps, analysis_params, "n_optimization_steps", 1000)
    args.learning_rate = resolve(args.learning_rate, analysis_params, "learning_rate", 0.01)

    # Learning rates: CLI is comma-separated string, yaml is a list
    if args.learning_rates is not None:
        args.learning_rates = [float(x) for x in args.learning_rates.split(",")]
    else:
        lr_sweep_params = conv_params.get("lr_sweep", {})
        args.learning_rates = lr_sweep_params.get("learning_rates", [0.005, 0.01, 0.05, 0.1])

    # Budgets: CLI is comma-separated string, yaml is a list
    if args.optimization_budgets is not None:
        args.optimization_budgets = [int(x) for x in args.optimization_budgets.split(",")]
    else:
        budget_params = conv_params.get("budget", {})
        args.optimization_budgets = budget_params.get("budgets", [50, 100, 250, 500, 1000, 2000])

    # Seeds: CLI is n_seeds (int), yaml is explicit list
    variance_params = conv_params.get("variance", {})
    if args.n_seeds is not None:
        base_seed = resolve(args.seed, yaml_params, "seed", 42)
        args.seeds = [base_seed + i for i in range(args.n_seeds)]
    else:
        args.seeds = variance_params.get("seeds", [42, 43, 44, 45, 46])

    base_config = {
        "n_episodes": resolve(args.n_episodes, analysis_params, "n_episodes", 20),
        "n_theta": resolve(args.n_theta, yaml_params, "n_theta", 2),
        "horizon": resolve(args.horizon, yaml_params, "horizon", 7),
        "max_steps": resolve(args.max_steps, yaml_params, "max_steps", 5),
        "goal_temperature": resolve(args.goal_temperature, yaml_params, "goal_temperature", 1.0),
        "cue_observation_accuracy": resolve(args.cue_accuracy, yaml_params, "cue_observation_accuracy", 1.0),
        "cue_cost_epsilon": yaml_params.get("cue_cost_epsilon", 0.01),
        "location_observation_accuracy": yaml_params.get("location_observation_accuracy", 0.90),
        "seed": resolve(args.seed, analysis_params, "seed", resolve(None, yaml_params, "seed", 42)),
        "verbose": args.verbose,
    }

    print(f"Analysis: {args.analysis}")
    print(f"Base config: n_episodes={base_config['n_episodes']}, "
          f"n_theta={base_config['n_theta']}, horizon={base_config['horizon']}")

    # Run analysis
    analysis_fn = ANALYSES[args.analysis]
    output, filename = analysis_fn(args, base_config)

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
