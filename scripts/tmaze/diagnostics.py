#!/usr/bin/env python3
"""
Diagnostic tests for epistemic priors in T-Maze planning.

This script helps tune:
- Goal prior temperature (how strongly the agent is pulled to the goal)
- Number of optimization iterations (convergence analysis)

Run with: uv run python scripts/diagnostics.py [options]
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import time

import jax
import jax.numpy as jnp
from jax.nn import softmax

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.planning import plan_actions_factorized, FactorizedPlanningConfig
from src.objectives.factorized_vfe import (
    enumerate_state_sequences,
    enumerate_action_sequences,
    enumerate_obs_sequences,
)

# Constants
STATE_NAMES = ["BOTTOM", "MIDDLE", "TOP_LEFT", "TOP_MID", "TOP_RIGHT"]
ACTION_NAMES = ["NORTH", "EAST", "SOUTH", "WEST"]
EPS = 1e-10


def create_tmaze_tensors_with_goal_temp(goal_temp: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create T-maze tensors with custom goal temperature."""
    # Transition tensor (deterministic)
    transition = jnp.zeros((5, 5, 4))
    
    # From BOTTOM (state 0)
    transition = transition.at[1, 0, 0].set(1.0)  # North -> Middle
    transition = transition.at[0, 0, 1].set(1.0)  # East -> stay
    transition = transition.at[0, 0, 2].set(1.0)  # South -> stay
    transition = transition.at[0, 0, 3].set(1.0)  # West -> stay
    
    # From MIDDLE (state 1)
    transition = transition.at[3, 1, 0].set(1.0)  # North -> Top middle
    transition = transition.at[1, 1, 1].set(1.0)  # East -> stay
    transition = transition.at[0, 1, 2].set(1.0)  # South -> Bottom
    transition = transition.at[1, 1, 3].set(1.0)  # West -> stay
    
    # From TOP_LEFT (state 2)
    transition = transition.at[2, 2, 0].set(1.0)  # North -> stay
    transition = transition.at[3, 2, 1].set(1.0)  # East -> Top middle
    transition = transition.at[1, 2, 2].set(1.0)  # South -> Middle
    transition = transition.at[2, 2, 3].set(1.0)  # West -> stay
    
    # From TOP_MIDDLE (state 3)
    transition = transition.at[3, 3, 0].set(1.0)  # North -> stay
    transition = transition.at[4, 3, 1].set(1.0)  # East -> Top right
    transition = transition.at[1, 3, 2].set(1.0)  # South -> Middle
    transition = transition.at[2, 3, 3].set(1.0)  # West -> Top left
    
    # From TOP_RIGHT (state 4)
    transition = transition.at[4, 4, 0].set(1.0)  # North -> stay
    transition = transition.at[4, 4, 1].set(1.0)  # East -> stay
    transition = transition.at[1, 4, 2].set(1.0)  # South -> Middle
    transition = transition.at[3, 4, 3].set(1.0)  # West -> Top middle
    
    # Observation tensor: cue reveals θ at BOTTOM only
    observation = jnp.ones((2, 5, 2)) * 0.5
    observation = observation.at[:, 0, 0].set(jnp.array([1.0, 0.0]))  # Left reward -> left cue
    observation = observation.at[:, 0, 1].set(jnp.array([0.0, 1.0]))  # Right reward -> right cue
    
    # Goal mapping with custom temperature
    goal_logits = jnp.array([
        [0, 0],
        [0, 0],
        [goal_temp, 0],
        [0, 0],
        [0, goal_temp],
    ])
    goal_mapping = jax.nn.softmax(goal_logits, axis=0)
    
    return transition, observation, goal_mapping


@dataclass
class DiagnosticResult:
    """Result from a single diagnostic run."""
    goal_temp: float
    n_opt_steps: int
    inference_mode: str
    scenario: str
    first_action_probs: List[float]
    final_state_probs: List[float]
    p_reach_goal: float
    p_visit_cue: float
    final_loss: float
    runtime_ms: float


def run_scenario(
    goal_temp: float,
    n_opt_steps: int,
    inference_mode: str,
    scenario: str,
    learning_rate: float = 0.05,
    horizon_override: int = None,
) -> DiagnosticResult:
    """Run a single diagnostic scenario."""
    transition, observation, goal_mapping = create_tmaze_tensors_with_goal_temp(goal_temp)
    
    if scenario == "unknown_theta_middle":
        prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
        prior_theta = jnp.array([0.5, 0.5])
        horizon = horizon_override if horizon_override else 4
    elif scenario == "known_theta_bottom":
        prior_state = jnp.array([1, 0, 0, 0, 0], dtype=jnp.float32)
        prior_theta = jnp.array([1.0, 0.0])
        horizon = horizon_override if horizon_override else 3
    elif scenario == "known_theta_middle":
        prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
        prior_theta = jnp.array([1.0, 0.0])
        horizon = 3
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    config = FactorizedPlanningConfig(
        planning_horizon=horizon,
        n_obs=2,
        n_states=5,
        n_actions=4,
        n_theta=2,
        n_optimization_steps=n_opt_steps,
        learning_rate=learning_rate,
        inference_mode=inference_mode,
    )
    
    start = time.perf_counter()
    result = plan_actions_factorized(
        prior_state=prior_state,
        prior_reward_location=prior_theta,
        transition_tensor=transition,
        observation_tensor=observation,
        goal_mapping=goal_mapping,
        config=config,
    )
    runtime_ms = (time.perf_counter() - start) * 1000
    
    # Compute metrics
    final_state_probs = result.all_state_probs[-1]
    
    if scenario == "unknown_theta_middle":
        # P(correct goal) = 0.5 * P(TOP_LEFT) + 0.5 * P(TOP_RIGHT)
        p_reach_goal = 0.5 * final_state_probs[2] + 0.5 * final_state_probs[4]
    else:
        # θ=left, so goal is TOP_LEFT
        p_reach_goal = final_state_probs[2]
    
    p_visit_cue = result.all_state_probs[0][0]  # P(BOTTOM at t=1)
    
    return DiagnosticResult(
        goal_temp=goal_temp,
        n_opt_steps=n_opt_steps,
        inference_mode=inference_mode,
        scenario=scenario,
        first_action_probs=result.first_action_probs.tolist(),
        final_state_probs=final_state_probs.tolist(),
        p_reach_goal=float(p_reach_goal),
        p_visit_cue=float(p_visit_cue),
        final_loss=result.final_loss,
        runtime_ms=runtime_ms,
    )


def print_table_header(title: str, columns: List[str]):
    """Print a formatted table header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    header = " | ".join(f"{c:>12}" for c in columns)
    print(header)
    print("-" * len(header))


def run_goal_temp_sweep(
    goal_temps: List[float],
    n_opt_steps: int = 2000,
    scenarios: List[str] = None,
    inference_modes: List[str] = None,
):
    """Sweep over goal temperatures."""
    if scenarios is None:
        scenarios = ["unknown_theta_middle", "known_theta_bottom"]
    if inference_modes is None:
        inference_modes = ["active", "marginal"]
    
    for scenario in scenarios:
        for mode in inference_modes:
            scenario_name = scenario.replace("_", " ").title()
            print_table_header(
                f"GOAL TEMP SWEEP: {scenario_name} ({mode.upper()})",
                ["goal_temp", "p(goal_prior)", "P(goal)", "P(cue@t1)", "SOUTH%", "loss"]
            )
            
            for temp in goal_temps:
                result = run_scenario(temp, n_opt_steps, mode, scenario)
                
                # Goal prior for reference
                _, _, goal_mapping = create_tmaze_tensors_with_goal_temp(temp)
                p_goal_prior = goal_mapping[2, 0]  # P(goal|TOP_LEFT, θ=left)
                
                south_pct = result.first_action_probs[2] * 100
                
                row = f"{temp:12.1f} | {p_goal_prior:12.4f} | {result.p_reach_goal:12.4f} | {result.p_visit_cue:12.4f} | {south_pct:12.1f} | {result.final_loss:12.2f}"
                print(row)


def run_opt_steps_sweep(
    opt_steps_list: List[int],
    goal_temp: float = 6.0,
    scenarios: List[str] = None,
    inference_modes: List[str] = None,
):
    """Sweep over optimization steps to check convergence."""
    if scenarios is None:
        scenarios = ["unknown_theta_middle", "known_theta_bottom"]
    if inference_modes is None:
        inference_modes = ["active"]
    
    for scenario in scenarios:
        for mode in inference_modes:
            scenario_name = scenario.replace("_", " ").title()
            print_table_header(
                f"OPT STEPS SWEEP: {scenario_name} ({mode.upper()}, goal_temp={goal_temp})",
                ["n_steps", "P(goal)", "P(cue@t1)", "loss", "time_ms"]
            )
            
            for n_steps in opt_steps_list:
                result = run_scenario(goal_temp, n_steps, mode, scenario)
                
                row = f"{n_steps:12d} | {result.p_reach_goal:12.4f} | {result.p_visit_cue:12.4f} | {result.final_loss:12.4f} | {result.runtime_ms:12.1f}"
                print(row)


def run_horizon_sweep(
    horizons: List[int],
    goal_temps: List[float],
    n_opt_steps: int = 10000,
):
    """Sweep over horizons and goal temps to find the best explore/exploit balance."""
    from itertools import product as iterproduct
    
    # First show combinatorics
    print_table_header(
        "COMBINATORICS: explore vs direct paths by horizon",
        ["horizon", "explore_paths", "direct_paths", "ratio"]
    )
    
    def next_state(state, action):
        transitions = {
            (0, 0): 1, (0, 1): 0, (0, 2): 0, (0, 3): 0,
            (1, 0): 3, (1, 1): 1, (1, 2): 0, (1, 3): 1,
            (2, 0): 2, (2, 1): 3, (2, 2): 1, (2, 3): 2,
            (3, 0): 3, (3, 1): 4, (3, 2): 1, (3, 3): 2,
            (4, 0): 4, (4, 1): 4, (4, 2): 1, (4, 3): 3,
        }
        return transitions.get((int(state), int(action)), state)
    
    for horizon in horizons:
        action_sequences = list(iterproduct(range(4), repeat=horizon))
        cue_goal, direct_goal = 0, 0
        
        for action_seq in action_sequences:
            state = 1  # MIDDLE
            trajectory = [state]
            for a in action_seq:
                state = next_state(state, a)
                trajectory.append(state)
            
            if trajectory[-1] in [2, 4]:  # ends at goal
                if 0 in trajectory:  # visits cue
                    cue_goal += 1
                else:
                    direct_goal += 1
        
        ratio = direct_goal / max(cue_goal, 1)
        print(f"{horizon:12d} | {cue_goal:12d} | {direct_goal:12d} | {ratio:12.1f}:1")
    
    # Now sweep horizon x goal_temp
    print_table_header(
        f"HORIZON x GOAL_TEMP SWEEP (n_opt_steps={n_opt_steps}, start=MIDDLE, θ unknown)",
        ["horizon", "goal_temp", "P(cue@t1)", "P(goal)", "q(SOUTH)"]
    )
    
    for horizon in horizons:
        for goal_temp in goal_temps:
            result = run_scenario(
                goal_temp, n_opt_steps, "active", "unknown_theta_middle",
                horizon_override=horizon
            )
            
            print(f"{horizon:12d} | {goal_temp:12.1f} | {result.p_visit_cue:12.4f} | {result.p_reach_goal:12.4f} | {result.first_action_probs[2]:12.4f}")


def run_epistemic_prior_analysis(goal_temp: float = 6.0, n_opt_steps: int = 2000):
    """Analyze the epistemic priors after optimization."""
    print("\n" + "=" * 80)
    print(f"EPISTEMIC PRIOR ANALYSIS (goal_temp={goal_temp}, n_opt_steps={n_opt_steps})")
    print("=" * 80)
    
    transition, observation, goal_mapping = create_tmaze_tensors_with_goal_temp(goal_temp)
    
    for scenario, prior_state, prior_theta, horizon, desc in [
        ("unknown_theta_middle", jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32), 
         jnp.array([0.5, 0.5]), 4, "θ UNKNOWN, start MIDDLE"),
        ("known_theta_bottom", jnp.array([1, 0, 0, 0, 0], dtype=jnp.float32),
         jnp.array([1.0, 0.0]), 3, "θ KNOWN (LEFT), start BOTTOM"),
    ]:
        print(f"\n>>> Scenario: {desc}")
        
        config = FactorizedPlanningConfig(
            planning_horizon=horizon, n_obs=2, n_states=5, n_actions=4, n_theta=2,
            n_optimization_steps=n_opt_steps, learning_rate=0.05, inference_mode="active",
        )
        
        result = plan_actions_factorized(
            prior_state=prior_state,
            prior_reward_location=prior_theta,
            transition_tensor=transition,
            observation_tensor=observation,
            goal_mapping=goal_mapping,
            config=config,
        )
        
        # Reconstruct distributions
        n_obs_seqs = 2 ** horizon
        n_state_seqs = 5 ** horizon
        
        q_u = softmax(result.q_u_logits)
        q_x_given_u = softmax(result.q_x_given_u_logits, axis=0)
        q_y_theta_given_x_flat = result.q_y_theta_given_x_logits.reshape(-1, n_state_seqs)
        q_y_theta_given_x = softmax(q_y_theta_given_x_flat, axis=0).reshape(n_obs_seqs, 2, n_state_seqs)
        
        q_xu = q_x_given_u * q_u[None, :]
        q_x = jnp.sum(q_xu, axis=1)
        
        # Epistemic state prior p̃(x) ∝ exp(-H[q(y|x)])
        q_y_given_x = jnp.sum(q_y_theta_given_x, axis=1)
        h_y_given_x = -jnp.sum(q_y_given_x * jnp.log(q_y_given_x + EPS), axis=0)
        epistemic_x_prior = softmax(-h_y_given_x)
        
        # Marginalize to first timestep
        state_sequences = enumerate_state_sequences(5, horizon)
        state_onehot = jax.nn.one_hot(state_sequences, 5)
        epistemic_x_t1 = jnp.einsum('s,stn->tn', epistemic_x_prior, state_onehot)[0]
        
        print(f"\n    Epistemic state prior p̃(x_1) (marginalized):")
        for s, p in enumerate(epistemic_x_t1):
            print(f"      {STATE_NAMES[s]:10s}: {p:.4f}")
        
        print(f"\n    Actual q(x_1):")
        for s, p in enumerate(result.all_state_probs[0]):
            if p > 0.01:
                print(f"      {STATE_NAMES[s]:10s}: {p:.4f}")
        
        print(f"\n    First action q(u_1):")
        for a, p in enumerate(result.first_action_probs):
            if p > 0.05:
                print(f"      {ACTION_NAMES[a]:10s}: {p:.4f}")
        
        print(f"\n    Final state q(x_T):")
        for s, p in enumerate(result.all_state_probs[-1]):
            if p > 0.01:
                marker = " ← GOAL" if (s == 2 and prior_theta[0] > 0.5) or (s == 4 and prior_theta[1] > 0.5) else ""
                print(f"      {STATE_NAMES[s]:10s}: {p:.4f}{marker}")


def run_active_vs_marginal_comparison(goal_temp: float = 6.0, n_opt_steps: int = 2000):
    """Compare active vs marginal inference behavior."""
    print("\n" + "=" * 80)
    print(f"ACTIVE vs MARGINAL COMPARISON (goal_temp={goal_temp})")
    print("=" * 80)
    
    transition, observation, goal_mapping = create_tmaze_tensors_with_goal_temp(goal_temp)
    
    scenario = "unknown_theta_middle"
    prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
    prior_theta = jnp.array([0.5, 0.5])
    horizon = 4
    
    print(f"\nScenario: Start MIDDLE, θ UNKNOWN")
    print("-" * 60)
    
    for mode in ["marginal", "active"]:
        config = FactorizedPlanningConfig(
            planning_horizon=horizon, n_obs=2, n_states=5, n_actions=4, n_theta=2,
            n_optimization_steps=n_opt_steps, learning_rate=0.05, inference_mode=mode,
        )
        
        result = plan_actions_factorized(
            prior_state=prior_state,
            prior_reward_location=prior_theta,
            transition_tensor=transition,
            observation_tensor=observation,
            goal_mapping=goal_mapping,
            config=config,
        )
        
        print(f"\n>>> {mode.upper()}:")
        print(f"    q(u_1): ", end="")
        for a, p in enumerate(result.first_action_probs):
            if p > 0.05:
                print(f"{ACTION_NAMES[a]}={p:.3f} ", end="")
        print()
        
        print(f"    P(visit cue at t=1): {result.all_state_probs[0][0]:.4f}")
        print(f"    P(final TOP_LEFT):   {result.all_state_probs[-1][2]:.4f}")
        print(f"    P(final TOP_RIGHT):  {result.all_state_probs[-1][4]:.4f}")
        
        # Exploration metric: how much more does active prefer SOUTH?
        if mode == "marginal":
            marginal_south = result.first_action_probs[2]
        else:
            active_south = result.first_action_probs[2]
    
    exploration_boost = active_south / (marginal_south + EPS)
    print(f"\n>>> Exploration boost (active/marginal SOUTH): {exploration_boost:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tests for epistemic priors in T-Maze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all diagnostics with default settings
  uv run python scripts/diagnostics.py --all

  # Sweep goal temperatures
  uv run python scripts/diagnostics.py --goal-temp-sweep

  # Sweep optimization steps
  uv run python scripts/diagnostics.py --opt-steps-sweep

  # Quick test with specific goal temp
  uv run python scripts/diagnostics.py --goal-temp 8 --quick

  # Full analysis with custom settings
  uv run python scripts/diagnostics.py --goal-temp 6 --n-opt-steps 5000 --epistemic-analysis
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")
    parser.add_argument("--goal-temp-sweep", action="store_true", help="Sweep goal temperatures")
    parser.add_argument("--opt-steps-sweep", action="store_true", help="Sweep optimization steps")
    parser.add_argument("--horizon-sweep", action="store_true", help="Sweep horizons and goal temps")
    parser.add_argument("--epistemic-analysis", action="store_true", help="Analyze epistemic priors")
    parser.add_argument("--active-vs-marginal", action="store_true", help="Compare active vs marginal")
    parser.add_argument("--quick", action="store_true", help="Quick test with current settings")
    
    parser.add_argument("--goal-temp", type=float, default=4.0, help="Goal temperature (default: 4.0)")
    parser.add_argument("--n-opt-steps", type=int, default=10000, help="Optimization steps (default: 10000)")
    parser.add_argument("--goal-temps", type=str, default="2,3,4,5,6",
                        help="Comma-separated goal temps for sweep")
    parser.add_argument("--opt-steps-list", type=str, default="1000,2000,5000,10000",
                        help="Comma-separated opt steps for sweep")
    parser.add_argument("--horizons", type=str, default="4,5,6",
                        help="Comma-separated horizons for sweep")
    
    args = parser.parse_args()
    
    # Parse lists
    goal_temps = [float(x) for x in args.goal_temps.split(",")]
    opt_steps_list = [int(x) for x in args.opt_steps_list.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    
    print("=" * 80)
    print("T-MAZE EPISTEMIC PRIOR DIAGNOSTICS")
    print("=" * 80)
    
    if args.all or args.goal_temp_sweep:
        run_goal_temp_sweep(goal_temps, args.n_opt_steps)
    
    if args.all or args.opt_steps_sweep:
        run_opt_steps_sweep(opt_steps_list, args.goal_temp)
    
    if args.all or args.horizon_sweep:
        run_horizon_sweep(horizons, goal_temps, args.n_opt_steps)
    
    if args.all or args.epistemic_analysis:
        run_epistemic_prior_analysis(args.goal_temp, args.n_opt_steps)
    
    if args.all or args.active_vs_marginal:
        run_active_vs_marginal_comparison(args.goal_temp, args.n_opt_steps)
    
    if args.quick:
        print(f"\n>>> Quick test with goal_temp={args.goal_temp}, n_opt_steps={args.n_opt_steps}")
        
        for scenario in ["unknown_theta_middle", "known_theta_bottom"]:
            result = run_scenario(args.goal_temp, args.n_opt_steps, "active", scenario)
            
            scenario_name = scenario.replace("_", " ").title()
            print(f"\n{scenario_name}:")
            print(f"  P(reach goal): {result.p_reach_goal:.4f}")
            print(f"  P(cue at t=1): {result.p_visit_cue:.4f}")
            print(f"  First action: ", end="")
            for a, p in enumerate(result.first_action_probs):
                if p > 0.05:
                    print(f"{ACTION_NAMES[a]}={p:.3f} ", end="")
            print()
    
    # If no flags specified, run all
    if not any([args.all, args.goal_temp_sweep, args.opt_steps_sweep, args.horizon_sweep,
                args.epistemic_analysis, args.active_vs_marginal, args.quick]):
        print("\nNo flags specified. Running --all diagnostics...\n")
        run_goal_temp_sweep(goal_temps, args.n_opt_steps)
        run_opt_steps_sweep(opt_steps_list, args.goal_temp)
        run_horizon_sweep(horizons, goal_temps, args.n_opt_steps)
        run_epistemic_prior_analysis(args.goal_temp, args.n_opt_steps)
        run_active_vs_marginal_comparison(args.goal_temp, args.n_opt_steps)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
