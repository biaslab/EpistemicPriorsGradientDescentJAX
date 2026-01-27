#!/usr/bin/env python3
"""
Convergence analysis for T-Maze VFE optimization.

Analyzes how the VFE loss converges during optimization
for different inference modes (active, marginal, planning).
"""

import argparse
from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.environments import create_tmaze_tensors
from src.planning import plan_actions, PlanningConfig


def run_convergence_analysis(
    n_optimization_steps: int = 500,
    planning_horizon: int = 4,
) -> dict:
    """
    Run convergence analysis for different inference modes.
    
    Returns:
        Dictionary with results for each inference mode.
    """
    transition, obs, goal = create_tmaze_tensors()
    prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)  # Start at MIDDLE
    theta_prior = jnp.array([0.5, 0.5])  # Uniform prior
    
    modes = ["active", "marginal", "planning"]
    results = {}
    
    for mode in modes:
        config = PlanningConfig(
            planning_horizon=planning_horizon,
            n_obs=2,
            n_states=5,
            n_actions=4,
            n_theta=2,
            n_optimization_steps=n_optimization_steps,
            inference_mode=mode,
        )
        
        result = plan_actions(
            prior_state=prior_state,
            prior_reward_location=theta_prior,
            transition_tensor=transition,
            observation_tensor=obs,
            goal_mapping=goal,
            config=config,
        )
        
        results[mode] = {
            "loss_history": result.loss_history,
            "first_action_probs": result.first_action_probs,
            "final_loss": result.final_loss,
        }
    
    return results


def create_convergence_plots(
    results: dict,
    output_dir: Path,
    save_tikz: bool = True,
    save_png: bool = True,
) -> None:
    """
    Create convergence plot and save to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {
        "active": "#1f78b4",    # blue
        "marginal": "#33a02c",  # green
        "planning": "#ff7f00",  # orange
    }
    labels = {
        "active": "Active Inference",
        "marginal": "Marginal Inference", 
        "planning": "Planning Inference",
    }
    
    for mode, data in results.items():
        ax.plot(
            data["loss_history"], 
            label=labels[mode],
            color=colors[mode],
            linewidth=1.5,
        )
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Variational Free Energy")
    ax.set_title("Free Energy Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_png:
        png_path = output_dir / "convergence.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {png_path}")
    
    plt.close()
    
    if save_tikz:
        _create_tikz_convergence_plot(results, output_dir)


def _create_tikz_convergence_plot(results: dict, output_dir: Path) -> None:
    """Generate TikZ code for convergence plot."""
    
    # Generate data file for pgfplots
    data_lines = ["iteration,active,marginal,planning"]
    
    n_steps = len(results["active"]["loss_history"])
    
    # Sample every 10 iterations to keep file size manageable
    for i in range(0, n_steps, 10):
        active = results["active"]["loss_history"][i]
        marginal = results["marginal"]["loss_history"][i]
        planning = results["planning"]["loss_history"][i]
        data_lines.append(f"{i},{active:.6f},{marginal:.6f},{planning:.6f}")
    
    # Always include last point
    if (n_steps - 1) % 10 != 0:
        active = results["active"]["loss_history"][-1]
        marginal = results["marginal"]["loss_history"][-1]
        planning = results["planning"]["loss_history"][-1]
        data_lines.append(f"{n_steps-1},{active:.6f},{marginal:.6f},{planning:.6f}")
    
    data_path = output_dir / "convergence_data.csv"
    with open(data_path, 'w') as f:
        f.write("\n".join(data_lines))
    print(f"Saved: {data_path}")
    
    # Generate TikZ standalone document
    tikz_code = r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[dvipsnames]{xcolor}

% ColorBrewer colors
\definecolor{CBblue}{HTML}{1f78b4}
\definecolor{CBgreen}{HTML}{33a02c}
\definecolor{CBorange}{HTML}{ff7f00}

\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=10cm,
    height=6cm,
    xlabel={Iteration},
    ylabel={Variational Free Energy},
    title={Free Energy Convergence},
    legend pos=north east,
    grid=major,
    grid style={gray!30},
]
\addplot[CBblue, thick] table[x=iteration, y=active, col sep=comma] {convergence_data.csv};
\addlegendentry{Active Inference}
\addplot[CBgreen, thick] table[x=iteration, y=marginal, col sep=comma] {convergence_data.csv};
\addlegendentry{Marginal Inference}
\addplot[CBorange, thick] table[x=iteration, y=planning, col sep=comma] {convergence_data.csv};
\addlegendentry{Planning Inference}
\end{axis}
\end{tikzpicture}
\end{document}
"""
    
    tikz_path = output_dir / "convergence.tex"
    with open(tikz_path, 'w') as f:
        f.write(tikz_code)
    print(f"Saved: {tikz_path}")


def create_policy_stability_analysis(
    n_optimization_steps: int = 500,
    planning_horizon: int = 4,
    output_dir: Path = None,
) -> None:
    """
    Analyze how the policy (action probabilities) stabilizes during optimization.
    """
    transition, obs, goal = create_tmaze_tensors()
    prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
    theta_prior = jnp.array([0.5, 0.5])
    
    checkpoints = [50, 100, 200, 300, 500]
    checkpoints = [c for c in checkpoints if c <= n_optimization_steps]
    
    results = {mode: {} for mode in ["active", "marginal", "planning"]}
    
    for mode in ["active", "marginal", "planning"]:
        for n_steps in checkpoints:
            config = PlanningConfig(
                planning_horizon=planning_horizon,
                n_obs=2,
                n_states=5,
                n_actions=4,
                n_theta=2,
                n_optimization_steps=n_steps,
                inference_mode=mode,
            )
            
            result = plan_actions(
                prior_state=prior_state,
                prior_reward_location=theta_prior,
                transition_tensor=transition,
                observation_tensor=obs,
                goal_mapping=goal,
                config=config,
            )
            
            results[mode][n_steps] = {
                "p_north": float(result.first_action_probs[0]),
                "p_south": float(result.first_action_probs[2]),
            }
    
    # Save as CSV if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_lines = ["iterations,mode,p_north,p_south"]
        for mode in ["active", "marginal", "planning"]:
            for n_steps in checkpoints:
                p_n = results[mode][n_steps]["p_north"]
                p_s = results[mode][n_steps]["p_south"]
                csv_lines.append(f"{n_steps},{mode},{p_n:.4f},{p_s:.4f}")
        
        csv_path = output_dir / "policy_stability.csv"
        with open(csv_path, 'w') as f:
            f.write("\n".join(csv_lines))
        print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for T-Maze VFE")
    parser.add_argument("--n-opt-steps", type=int, default=500,
                        help="Number of optimization steps")
    parser.add_argument("--planning-horizon", type=int, default=4,
                        help="Planning horizon")
    parser.add_argument("--output-dir", type=str, default="data/convergence",
                        help="Output directory for figures")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip PNG generation")
    parser.add_argument("--no-tikz", action="store_true",
                        help="Skip TikZ generation")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Run main convergence analysis
    results = run_convergence_analysis(
        n_optimization_steps=args.n_opt_steps,
        planning_horizon=args.planning_horizon,
    )
    
    # Create plots
    create_convergence_plots(
        results,
        output_dir,
        save_tikz=not args.no_tikz,
        save_png=not args.no_png,
    )
    


if __name__ == "__main__":
    main()
