#!/usr/bin/env python3
"""
Publication-quality convergence figures for T-Maze VFE optimization.

Reads JSON data files produced by convergence_analysis.py and generates
combined + individual panel figures in PNG and PGF formats.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
})

MODES = ["active", "marginal", "planning"]
MODE_LABELS = {
    "active": "Active",
    "marginal": "Marginal",
    "planning": "Planning",
}
SCENARIO_ORDER = ["theta_unknown", "theta_known"]
SCENARIO_TITLES = {
    "theta_unknown": r"$\theta$ unknown",
    "theta_known": r"$\theta$ known",
}


def load_colors():
    """Load color scheme from params.yaml."""
    params_path = project_root / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("colors", {
        "active": "#1F78B4",
        "marginal": "#33A02C",
        "planning": "#FF7F00",
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_figure(fig, output_dir, name):
    """Save figure as both PNG and PGF."""
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pgf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png, {name}.pgf")


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def plot_loss_curves(data, colors, ax):
    """Optimization loss curves with median + IQR across episodes."""
    for mode in MODES:
        episodes = data["results"][mode]
        all_histories = []
        for ep in episodes:
            for ps in ep["planning_steps"]:
                all_histories.append(ps["all_losses"])

        if not all_histories:
            continue

        min_len = min(len(h) for h in all_histories)
        arr = np.array([h[:min_len] for h in all_histories])

        median = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)

        steps = np.arange(min_len)
        ax.plot(steps, median, label=MODE_LABELS[mode],
                color=colors[mode], linewidth=1.0)
        ax.fill_between(steps, q25, q75, color=colors[mode], alpha=0.2)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("VFE loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_lr_sensitivity(data, colors, ax):
    """Success rate vs learning rate."""
    for mode in MODES:
        entries = data["results"][mode]
        lrs = [e["learning_rate"] for e in entries]
        rates = [e["success_rate"] for e in entries]
        ax.plot(lrs, rates, marker="o", markersize=3,
                label=MODE_LABELS[mode], color=colors[mode], linewidth=1.0)

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Success rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_compute_budget(data, colors, ax):
    """Success rate vs optimization budget."""
    for mode in MODES:
        entries = data["results"][mode]
        budgets = [e["n_opt_steps"] for e in entries]
        rates = [e["success_rate"] for e in entries]
        ax.plot(budgets, rates, marker="o", markersize=3,
                label=MODE_LABELS[mode], color=colors[mode], linewidth=1.0)

    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Success rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_seed_variance(data, colors, ax):
    """Box + strip plot of mean rewards across seeds."""
    positions = np.arange(len(MODES))
    width = 0.6

    for i, mode in enumerate(MODES):
        entries = data["results"][mode]
        rewards = [e["mean_reward"] for e in entries]

        ax.boxplot(
            [rewards], positions=[positions[i]], widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[mode], alpha=0.3,
                          edgecolor=colors[mode]),
            medianprops=dict(color=colors[mode], linewidth=1.5),
            whiskerprops=dict(color=colors[mode]),
            capprops=dict(color=colors[mode]),
            flierprops=dict(marker=".", markerfacecolor=colors[mode],
                            markersize=3),
        )

        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(rewards))
        ax.scatter(
            positions[i] + jitter, rewards,
            color=colors[mode], s=8, alpha=0.7, zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES])
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")


def plot_policy_stability(data, colors, ax):
    """P(South) vs optimization steps for each mode."""
    for mode in MODES:
        entries = data["results"][mode]
        budgets = [e["n_opt_steps"] for e in entries]
        p_south = [e["first_action_probs"][2] for e in entries]
        ax.plot(budgets, p_south, marker="o", markersize=3,
                label=MODE_LABELS[mode], color=colors[mode], linewidth=1.0)

    ax.set_xlabel("Optimization steps")
    ax.set_ylabel(r"$P(\mathrm{South})$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_scenario_loss(scenario_data, colors, ax, title=""):
    """Loss curves for a single scenario (one curve per mode)."""
    for mode in MODES:
        losses = scenario_data[mode]["all_losses"]
        steps = np.arange(len(losses))
        ax.plot(steps, losses, label=MODE_LABELS[mode],
                color=colors[mode], linewidth=1.0)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("VFE loss")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot convergence figures")
    parser.add_argument("--input-dir", type=str, default="data/convergence")
    parser.add_argument("--output-dir", type=str,
                        default="data/convergence/figures")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = load_colors()

    # Load all data files
    curves_data = load_json(input_dir / "curves.json")
    lr_data = load_json(input_dir / "lr_sweep.json")
    budget_data = load_json(input_dir / "budget.json")
    variance_data = load_json(input_dir / "variance.json")
    policy_stability_data = load_json(input_dir / "policy_stability.json")
    scenario_data = load_json(input_dir / "scenario_curves.json")

    # --- Combined 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    plot_loss_curves(curves_data, colors, axes[0, 0])
    plot_lr_sensitivity(lr_data, colors, axes[0, 1])
    plot_compute_budget(budget_data, colors, axes[1, 0])
    plot_seed_variance(variance_data, colors, axes[1, 1])
    for ax, label in zip(axes.flat, "abcd"):
        ax.set_title(f"({label})", loc="left", fontsize=8, fontweight="bold")
    save_figure(fig, output_dir, "convergence")

    # --- Individual panels ---
    panel_funcs = [
        ("loss_curves", plot_loss_curves, curves_data),
        ("lr_sensitivity", plot_lr_sensitivity, lr_data),
        ("compute_budget", plot_compute_budget, budget_data),
        ("seed_variance", plot_seed_variance, variance_data),
        ("policy_stability", plot_policy_stability, policy_stability_data),
    ]
    for name, func, panel_data in panel_funcs:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        func(panel_data, colors, ax)
        save_figure(fig, output_dir, name)

    # --- Scenario combined figure (1x2) ---
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    for ax, scenario_name in zip(axes, SCENARIO_ORDER):
        plot_scenario_loss(
            scenario_data["scenarios"][scenario_name], colors, ax,
            title=SCENARIO_TITLES[scenario_name],
        )
    save_figure(fig, output_dir, "scenario_convergence")

    # --- Individual scenario panels ---
    for scenario_name in SCENARIO_ORDER:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        plot_scenario_loss(
            scenario_data["scenarios"][scenario_name], colors, ax,
            title=SCENARIO_TITLES[scenario_name],
        )
        save_figure(fig, output_dir, f"scenario_{scenario_name}")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
