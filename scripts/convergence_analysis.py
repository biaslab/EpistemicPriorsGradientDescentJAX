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
import yaml

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.environments import create_tmaze_tensors
from src.planning import plan_actions_factorized, FactorizedPlanningConfig


def load_colors() -> dict:
    """Load color scheme from params.yaml."""
    params_path = script_dir / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("colors", {
        "active": "#1F78B4",
        "marginal": "#33A02C", 
        "planning": "#FF7F00",
    })


def run_convergence_analysis(
    n_optimization_steps: int = 1000,
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
        config = FactorizedPlanningConfig(
            planning_horizon=planning_horizon,
            n_obs=2,
            n_states=5,
            n_actions=4,
            n_theta=2,
            n_optimization_steps=n_optimization_steps,
            learning_rate=0.05,
            inference_mode=mode,
        )
        
        result = plan_actions_factorized(
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
    
    colors = load_colors()
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
    colors = load_colors()
    
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
    
    # Convert colors to TikZ format (remove # prefix)
    c_active = colors["active"].lstrip("#")
    c_marginal = colors["marginal"].lstrip("#")
    c_planning = colors["planning"].lstrip("#")
    
    # Generate TikZ standalone document
    tikz_code = rf"""\documentclass[tikz,border=5pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepackage[dvipsnames]{{xcolor}}

\definecolor{{activecolor}}{{HTML}}{{{c_active}}}
\definecolor{{marginalcolor}}{{HTML}}{{{c_marginal}}}
\definecolor{{planningcolor}}{{HTML}}{{{c_planning}}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=10cm,
    height=6cm,
    xlabel={{Iteration}},
    ylabel={{Variational Free Energy}},
    title={{Free Energy Convergence}},
    legend pos=north east,
    grid=major,
    grid style={{gray!30}},
]
\addplot[activecolor, thick] table[x=iteration, y=active, col sep=comma] {{convergence_data.csv}};
\addlegendentry{{Active Inference}}
\addplot[marginalcolor, thick] table[x=iteration, y=marginal, col sep=comma] {{convergence_data.csv}};
\addlegendentry{{Marginal Inference}}
\addplot[planningcolor, thick] table[x=iteration, y=planning, col sep=comma] {{convergence_data.csv}};
\addlegendentry{{Planning Inference}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    
    tikz_path = output_dir / "convergence.tex"
    with open(tikz_path, 'w') as f:
        f.write(tikz_code)
    print(f"Saved: {tikz_path}")


def create_policy_stability_analysis(
    n_optimization_steps: int = 1000,
    planning_horizon: int = 4,
    output_dir: Path = None,
) -> None:
    """
    Analyze how the policy (action probabilities) stabilizes during optimization.
    """
    transition, obs, goal = create_tmaze_tensors()
    prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
    theta_prior = jnp.array([0.5, 0.5])
    
    checkpoints = [100, 500, 1000, 2000, 3000, 5000]
    checkpoints = [c for c in checkpoints if c <= n_optimization_steps]
    
    results = {mode: {} for mode in ["active", "marginal", "planning"]}
    
    for mode in ["active", "marginal", "planning"]:
        for n_steps in checkpoints:
            config = FactorizedPlanningConfig(
                planning_horizon=planning_horizon,
                n_obs=2,
                n_states=5,
                n_actions=4,
                n_theta=2,
                n_optimization_steps=n_steps,
                learning_rate=0.05,
                inference_mode=mode,
            )
            
            result = plan_actions_factorized(
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
    
    # Save outputs if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
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
        
        # Create PNG plot
        _create_policy_stability_png(results, checkpoints, output_dir)
        
        # Create TikZ plot
        _create_policy_stability_tikz(results, checkpoints, output_dir)


def _create_policy_stability_png(results: dict, checkpoints: list, output_dir: Path) -> None:
    """Generate PNG plot for policy stability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = load_colors()
    labels = {
        "active": "Active Inference",
        "marginal": "Marginal Inference",
        "planning": "Planning Inference",
    }
    
    for mode in ["active", "marginal", "planning"]:
        p_south_values = [results[mode][n]["p_south"] for n in checkpoints]
        ax.plot(
            checkpoints,
            p_south_values,
            label=labels[mode],
            color=colors[mode],
            linewidth=2,
            marker='o',
            markersize=6,
        )
    
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("P(South)")
    ax.set_title("Policy Stability: P(South) vs Optimization Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    png_path = output_dir / "policy_stability.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()


def _create_policy_stability_tikz(results: dict, checkpoints: list, output_dir: Path) -> None:
    """Generate TikZ code for policy stability plot."""
    colors = load_colors()
    
    # Convert colors to TikZ format (remove # prefix)
    c_active = colors["active"].lstrip("#")
    c_marginal = colors["marginal"].lstrip("#")
    c_planning = colors["planning"].lstrip("#")
    
    tikz_code = rf"""\documentclass[tikz,border=5pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepackage[dvipsnames]{{xcolor}}

\definecolor{{activecolor}}{{HTML}}{{{c_active}}}
\definecolor{{marginalcolor}}{{HTML}}{{{c_marginal}}}
\definecolor{{planningcolor}}{{HTML}}{{{c_planning}}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=10cm,
    height=6cm,
    xlabel={{Optimization Steps}},
    ylabel={{$P(\mathrm{{South}})$}},
    title={{Policy Stability}},
    legend pos=north east,
    grid=major,
    grid style={{gray!30}},
    ymin=0,
    ymax=1,
]
"""
    
    # Add plots for each mode - now showing P(South) for exploration
    for mode, color in [("active", "activecolor"), ("marginal", "marginalcolor"), ("planning", "planningcolor")]:
        coords = " ".join(
            f"({n},{results[mode][n]['p_south']:.4f})"
            for n in checkpoints
        )
        label = {"active": "Active Inference", "marginal": "Marginal Inference", "planning": "Planning Inference"}[mode]
        tikz_code += f"\\addplot[{color}, thick, mark=*] coordinates {{{coords}}};\n"
        tikz_code += f"\\addlegendentry{{{label}}}\n"
    
    tikz_code += r"""\end{axis}
\end{tikzpicture}
\end{document}
"""
    
    tikz_path = output_dir / "policy_stability.tex"
    with open(tikz_path, 'w') as f:
        f.write(tikz_code)
    print(f"Saved: {tikz_path}")


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for T-Maze VFE")
    parser.add_argument("--n-opt-steps", type=int, default=5000,
                        help="Number of optimization steps (default: 5000)")
    parser.add_argument("--planning-horizon", type=int, default=4,
                        help="Planning horizon (default: 4)")
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
    
    # Run policy stability analysis
    create_policy_stability_analysis(
        n_optimization_steps=args.n_opt_steps,
        planning_horizon=args.planning_horizon,
        output_dir=output_dir,
    )



if __name__ == "__main__":
    main()
