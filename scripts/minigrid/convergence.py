#!/usr/bin/env python3
"""
Comprehensive VFE convergence analysis for MiniGrid DoorKey planning.

Grids over 2 optimizers (adam, adafactor) x 3 inference modes x 5 scenarios,
producing PNG + PGF plots and a structured JSON summary.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environments.minigrid import (
    create_minigrid_env_tensors,
    flatten_state_index,
    coords_to_state,
    get_valid_static_configs,
    N_ORIENTATIONS,
    N_DOOR_KEY_STATES,
)
from src.planning.temporal_optimizer_minigrid import TemporalPlanningConfig, plan_actions_temporal

OPTIMIZERS = ["adam", "adafactor"]
MODES = ["marginal", "active", "planning"]


def build_scenarios(n, n_states, n_theta, default_horizon):
    """Build the 5 convergence scenarios.

    Returns list of dicts with keys: name, initial_state, theta_logits, horizon.
    """
    n_locs = n * n

    # Uniform state
    uniform_state = jnp.ones(n_states) / n_states

    # Concentrated state at (0,0), facing RIGHT, dks=no_key
    concentrated_state = jnp.zeros(n_states)
    idx_origin = flatten_state_index(
        coords_to_state(0, 0, n), 0, 0, n_locs, N_ORIENTATIONS, N_DOOR_KEY_STATES
    )
    concentrated_state = concentrated_state.at[idx_origin].set(1.0)

    # Near-goal state: (2,1) facing DOWN, dks=door_open (for n=3)
    # For larger grids: (n-1, n-2) facing DOWN, dks=door_open
    near_goal_state = jnp.zeros(n_states)
    goal_adj_loc = coords_to_state(n - 1, n - 2, n)
    idx_near_goal = flatten_state_index(
        goal_adj_loc, 1, 2, n_locs, N_ORIENTATIONS, N_DOOR_KEY_STATES  # orient=DOWN, dks=door_open
    )
    near_goal_state = near_goal_state.at[idx_near_goal].set(1.0)

    # Theta priors
    uniform_theta = jnp.zeros(n_theta)
    concentrated_theta = jnp.zeros(n_theta).at[0].set(5.0)  # ~99% on config 0

    scenarios = [
        {
            "name": "Tabula rasa",
            "short_name": "tabula_rasa",
            "initial_state": uniform_state,
            "theta_logits": uniform_theta,
            "horizon": default_horizon,
        },
        {
            "name": "Known position",
            "short_name": "known_pos",
            "initial_state": concentrated_state,
            "theta_logits": uniform_theta,
            "horizon": default_horizon,
        },
        {
            "name": "Known layout",
            "short_name": "known_layout",
            "initial_state": uniform_state,
            "theta_logits": concentrated_theta,
            "horizon": default_horizon,
        },
        {
            "name": "Near goal",
            "short_name": "near_goal",
            "initial_state": near_goal_state,
            "theta_logits": concentrated_theta,
            "horizon": 3,
        },
        {
            "name": f"Long horizon (2H={2 * default_horizon})",
            "short_name": "long_horizon",
            "initial_state": concentrated_state,
            "theta_logits": uniform_theta,
            "horizon": 2 * default_horizon,
        },
    ]
    return scenarios


def run_single(env_tensors, scenario, mode, optimizer_type, n_opt_steps, learning_rate,
               seed, freeze, goal_scale):
    """Run a single convergence test. Returns dict with metrics."""
    config = TemporalPlanningConfig(
        planning_horizon=scenario["horizon"],
        n_states=env_tensors.n_states,
        n_actions=env_tensors.n_actions,
        n_theta=env_tensors.n_theta,
        n_optimization_steps=n_opt_steps,
        learning_rate=learning_rate,
        inference_mode=mode,
        init_seed=seed,
        freeze_obs_and_transitions=freeze,
        goal_scale=goal_scale,
        optimizer_type=optimizer_type,
    )

    result = plan_actions_temporal(
        initial_state=scenario["initial_state"],
        env_tensors=env_tensors,
        config=config,
        prior_theta_logits=scenario["theta_logits"],
    )

    loss_history = result.loss_history
    steps = list(range(0, n_opt_steps, 100))[:len(loss_history)]

    initial_loss = loss_history[0] if loss_history else float("nan")
    final_loss = result.final_loss
    total_drop = initial_loss - final_loss

    # Convergence: range of last 5 samples (= last 500 steps)
    n_tail = min(5, len(loss_history))
    if n_tail >= 2:
        tail = loss_history[-n_tail:]
        delta_last_500 = max(tail) - min(tail)
    else:
        delta_last_500 = float("inf")

    return {
        "steps": steps,
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "total_drop": total_drop,
        "delta_last_500": delta_last_500,
        "q_first_action": [float(x) for x in np.array(result.q_first_action)],
    }


def load_colors():
    """Load color scheme from params.yaml if available."""
    try:
        with open(Path(__file__).parent.parent.parent / "params.yaml") as f:
            params = yaml.safe_load(f)
        colors = params.get("colors", {})
        return {
            "active": colors.get("active", "#1F78B4"),
            "planning": colors.get("planning", "#FF7F00"),
            "marginal": "#666666",
        }
    except Exception:
        return {"active": "#1F78B4", "planning": "#FF7F00", "marginal": "#666666"}


def make_knowledge_figure(results, scenarios, colors, threshold):
    """Create 3x4 knowledge scenarios figure (scenarios #0-3)."""
    knowledge_scenarios = scenarios[:4]
    fig, axes = plt.subplots(
        len(MODES), len(knowledge_scenarios),
        figsize=(4 * len(knowledge_scenarios), 3 * len(MODES)),
        squeeze=False,
    )

    for row, mode in enumerate(MODES):
        for col, scenario in enumerate(knowledge_scenarios):
            ax = axes[row, col]
            for opt_type in OPTIMIZERS:
                key = (opt_type, mode, scenario["short_name"])
                if key not in results:
                    continue
                r = results[key]
                ls = "-" if opt_type == "adam" else "--"
                ax.plot(r["steps"], r["loss_history"], ls, linewidth=1.2,
                        color=colors[mode], label=opt_type)

            ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel(f"{mode}\nVFE")
            if row == 0:
                ax.set_title(scenario["name"], fontsize=9)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == len(knowledge_scenarios) - 1:
                ax.legend(fontsize=7)

    fig.suptitle("VFE Convergence - Knowledge Scenarios", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def make_horizon_figure(results, scenarios, colors, default_horizon):
    """Create 3x2 horizon comparison figure (scenarios #1 default vs #4 long)."""
    # scenario[1] = Known position (default H), scenario[4] = Long horizon (2H)
    horizon_scenarios = [scenarios[1], scenarios[4]]
    fig, axes = plt.subplots(
        len(MODES), len(horizon_scenarios),
        figsize=(4 * len(horizon_scenarios), 3 * len(MODES)),
        squeeze=False,
    )

    for row, mode in enumerate(MODES):
        for col, scenario in enumerate(horizon_scenarios):
            ax = axes[row, col]
            for opt_type in OPTIMIZERS:
                key = (opt_type, mode, scenario["short_name"])
                if key not in results:
                    continue
                r = results[key]
                ls = "-" if opt_type == "adam" else "--"
                ax.plot(r["steps"], r["loss_history"], ls, linewidth=1.2,
                        color=colors[mode], label=opt_type)

            ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel(f"{mode}\nVFE")
            if row == 0:
                ax.set_title(f"H={scenario['horizon']}", fontsize=9)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == len(horizon_scenarios) - 1:
                ax.legend(fontsize=7)

    fig.suptitle("VFE Convergence - Horizon Comparison", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def save_figure(fig, path_stem):
    """Save figure as PNG and PGF."""
    fig.savefig(f"{path_stem}.png", dpi=150, bbox_inches="tight")
    try:
        fig.savefig(f"{path_stem}.pgf", backend="pgf", bbox_inches="tight")
    except Exception as e:
        print(f"  Warning: PGF save failed ({e}), skipping .pgf output")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VFE convergence analysis")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--fov-size", type=int, default=3)
    parser.add_argument("--n-opt-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-obs-and-transitions", action="store_true")
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--convergence-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.grid_size
    print(f"Creating env tensors (n={n}, fov={args.fov_size})...")
    env_tensors = create_minigrid_env_tensors(n=n, fov_size=args.fov_size)
    print(f"  n_states={env_tensors.n_states}, n_theta={env_tensors.n_theta}, "
          f"n_actions={env_tensors.n_actions}, modalities={len(env_tensors.observation_modalities)}")

    scenarios = build_scenarios(n, env_tensors.n_states, env_tensors.n_theta, args.horizon)

    # Run all combinations
    results = {}
    total = len(OPTIMIZERS) * len(MODES) * len(scenarios)
    count = 0
    total_start = time.perf_counter()

    for opt_type in OPTIMIZERS:
        for mode in MODES:
            for scenario in scenarios:
                count += 1
                key = (opt_type, mode, scenario["short_name"])
                print(f"\n[{count}/{total}] {opt_type} | {mode} | {scenario['name']} (H={scenario['horizon']})")

                t0 = time.perf_counter()
                r = run_single(
                    env_tensors, scenario, mode, opt_type,
                    args.n_opt_steps, args.learning_rate, args.seed,
                    args.freeze_obs_and_transitions, args.goal_scale,
                )
                r["wall_clock_s"] = time.perf_counter() - t0
                results[key] = r

                converged = r["delta_last_500"] < args.convergence_threshold
                status = "PASS" if converged else "FAIL"
                print(f"  Loss: {r['initial_loss']:.2f} -> {r['final_loss']:.2f} "
                      f"(drop={r['total_drop']:.2f}, d500={r['delta_last_500']:.4f}) [{status}] "
                      f"({r['wall_clock_s']:.1f}s)")

    # Print summary table
    print(f"\n{'=' * 95}")
    print(f"{'Optimizer':<12} {'Mode':<12} {'Scenario':<22} {'H':>3} "
          f"{'Final Loss':>11} {'d_last500':>10} {'Status':>6}")
    print(f"{'-' * 95}")
    for opt_type in OPTIMIZERS:
        for mode in MODES:
            for scenario in scenarios:
                key = (opt_type, mode, scenario["short_name"])
                r = results[key]
                converged = r["delta_last_500"] < args.convergence_threshold
                status = "PASS" if converged else "FAIL"
                print(f"{opt_type:<12} {mode:<12} {scenario['name']:<22} {scenario['horizon']:>3} "
                      f"{r['final_loss']:>11.4f} {r['delta_last_500']:>10.4f} {status:>6}")
    print(f"{'=' * 95}")

    total_wall_clock_s = time.perf_counter() - total_start

    # Save JSON results
    json_results = {
        "config": {
            "grid_size": n,
            "fov_size": args.fov_size,
            "n_opt_steps": args.n_opt_steps,
            "learning_rate": args.learning_rate,
            "default_horizon": args.horizon,
            "seed": args.seed,
            "freeze_obs_and_transitions": args.freeze_obs_and_transitions,
            "goal_scale": args.goal_scale,
            "convergence_threshold": args.convergence_threshold,
            "total_wall_clock_s": total_wall_clock_s,
        },
        "results": {},
    }
    for (opt_type, mode, sname), r in results.items():
        json_key = f"{opt_type}__{mode}__{sname}"
        json_results["results"][json_key] = {
            "optimizer": opt_type,
            "mode": mode,
            "scenario": sname,
            "initial_loss": r["initial_loss"],
            "final_loss": r["final_loss"],
            "total_drop": r["total_drop"],
            "delta_last_500": r["delta_last_500"],
            "converged": r["delta_last_500"] < args.convergence_threshold,
            "wall_clock_s": r["wall_clock_s"],
            "q_first_action": r["q_first_action"],
            "loss_history": r["loss_history"],
        }

    with open(output_dir / "convergence_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON results saved to {output_dir / 'convergence_results.json'}")

    # Generate figures
    colors = load_colors()

    # Configure matplotlib for PGF
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": True,
    })

    fig_knowledge = make_knowledge_figure(results, scenarios, colors, args.convergence_threshold)
    save_figure(fig_knowledge, str(output_dir / "convergence_knowledge"))
    plt.close(fig_knowledge)
    print(f"Knowledge figure saved to {output_dir / 'convergence_knowledge.png'}")

    fig_horizon = make_horizon_figure(results, scenarios, colors, args.horizon)
    save_figure(fig_horizon, str(output_dir / "convergence_horizon"))
    plt.close(fig_horizon)
    print(f"Horizon figure saved to {output_dir / 'convergence_horizon.png'}")


if __name__ == "__main__":
    main()
