#!/usr/bin/env python3
"""
Visualize convergence analysis results as a publication-quality 2×2 figure.

Reads 4 JSON files produced by convergence_analysis.py and exports:
  - Combined 2×2 figure (convergence.pgf + convergence.png)
  - Individual panels (loss_curves, lr_sensitivity, compute_budget, seed_variance)

Panels:
  (a) Loss curves — normalized VFE vs optimization step (median + IQR)
  (b) LR sensitivity — success rate vs learning rate (log scale)
  (c) Compute budget — success rate vs optimization steps (log scale)
  (d) Seed variance — mean reward per mode (box + strip)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Backend setup: try PGF (LaTeX-native), fall back to Agg (PNG only)
# ---------------------------------------------------------------------------
USE_PGF = True
try:
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    # Test that pdflatex is available
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
except Exception:
    USE_PGF = False
    warnings.warn("PGF backend unavailable (LaTeX not installed?). Falling back to Agg — PNG only.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

MODES = ["marginal", "active", "planning"]
MODE_LABELS = {"marginal": "Marginal", "active": "Active", "planning": "Planning"}


def load_json(path):
    """Load JSON file or return None if missing."""
    p = Path(path)
    if not p.exists():
        warnings.warn(f"Missing data file: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_colors(project_root):
    """Load color scheme from params.yaml."""
    params_path = project_root / "params.yaml"
    defaults = {"active": "#1F78B4", "marginal": "#33A02C", "planning": "#FF7F00"}
    if not params_path.exists():
        return defaults
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("colors", defaults)


# ---------------------------------------------------------------------------
# Panel plotting functions
# ---------------------------------------------------------------------------

def plot_loss_curves(ax, data, colors):
    """Panel (a): Normalized VFE vs optimization step (median + IQR shading)."""
    for mode in MODES:
        if mode not in data["results"]:
            continue
        episodes = data["results"][mode]
        # Collect all loss curves across episodes and planning steps
        all_curves = []
        for ep in episodes:
            for ps in ep["planning_steps"]:
                all_curves.append(ps["all_losses"])
        if not all_curves:
            continue
        # Pad to same length (in case of different lengths)
        max_len = max(len(c) for c in all_curves)
        padded = np.full((len(all_curves), max_len), np.nan)
        for i, c in enumerate(all_curves):
            padded[i, :len(c)] = c
        # Normalize each curve to [0, 1]
        for i in range(len(all_curves)):
            row = padded[i]
            valid = ~np.isnan(row)
            lo, hi = np.nanmin(row), np.nanmax(row)
            if hi - lo > 1e-12:
                padded[i, valid] = (row[valid] - lo) / (hi - lo)
            else:
                padded[i, valid] = 0.0
        steps = np.arange(max_len)
        median = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)
        ax.plot(steps, median, color=colors[mode], label=MODE_LABELS[mode], linewidth=1.2)
        ax.fill_between(steps, q25, q75, color=colors[mode], alpha=0.2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Normalized VFE")
    ax.set_title("(a) Convergence trajectories")
    ax.legend(frameon=False)


def plot_lr_sensitivity(ax, data, colors):
    """Panel (b): Success rate vs learning rate (log scale)."""
    for mode in MODES:
        if mode not in data["results"]:
            continue
        entries = data["results"][mode]
        lrs = [e["learning_rate"] for e in entries]
        success = [e["success_rate"] for e in entries]
        ax.plot(lrs, success, "o-", color=colors[mode], label=MODE_LABELS[mode],
                linewidth=1.2, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Success rate")
    ax.set_title("(b) LR sensitivity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)


def plot_compute_budget(ax, data, colors):
    """Panel (c): Success rate vs optimization steps (log scale)."""
    for mode in MODES:
        if mode not in data["results"]:
            continue
        entries = data["results"][mode]
        steps = [e["n_opt_steps"] for e in entries]
        success = [e["success_rate"] for e in entries]
        ax.plot(steps, success, "s-", color=colors[mode], label=MODE_LABELS[mode],
                linewidth=1.2, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Success rate")
    ax.set_title("(c) Compute budget")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)


def plot_seed_variance(ax, data, colors):
    """Panel (d): Mean reward per mode (box + strip plot)."""
    positions = []
    box_data = []
    tick_labels = []
    for i, mode in enumerate(MODES):
        if mode not in data["results"]:
            continue
        entries = data["results"][mode]
        rewards = [e["mean_reward"] for e in entries]
        positions.append(i)
        box_data.append(rewards)
        tick_labels.append(MODE_LABELS[mode])
    if not box_data:
        return
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, zorder=2)
    for i, (patch, mode) in enumerate(zip(bp["boxes"], [m for m in MODES if m in data["results"]])):
        c = colors[mode]
        patch.set_facecolor(c)
        patch.set_alpha(0.3)
        patch.set_edgecolor(c)
        for key in ["whiskers", "caps"]:
            # Each box has 2 whiskers/caps
            bp[key][2 * i].set_color(c)
            bp[key][2 * i + 1].set_color(c)
        bp["medians"][i].set_color(c)
        bp["medians"][i].set_linewidth(1.5)
    # Strip (jittered points)
    rng = np.random.default_rng(0)
    for i, mode in enumerate([m for m in MODES if m in data["results"]]):
        rewards = [e["mean_reward"] for e in data["results"][mode]]
        jitter = rng.uniform(-0.15, 0.15, size=len(rewards))
        ax.scatter(np.full(len(rewards), positions[i]) + jitter, rewards,
                   color=colors[mode], s=12, alpha=0.6, zorder=3, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Mean reward")
    ax.set_title("(d) Seed variance")


# ---------------------------------------------------------------------------
# Panel specs: (filename_stem, plot_function, data_key)
# ---------------------------------------------------------------------------
PANELS = [
    ("loss_curves", plot_loss_curves, "curves"),
    ("lr_sensitivity", plot_lr_sensitivity, "lr_sweep"),
    ("compute_budget", plot_compute_budget, "budget"),
    ("seed_variance", plot_seed_variance, "variance"),
]


def save_figure(fig, output_dir, stem):
    """Save figure as .png (always) and .pgf (if backend supports it)."""
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if USE_PGF:
        fig.savefig(output_dir / f"{stem}.pgf", bbox_inches="tight")


SCENARIO_ORDER = [
    "theta_unknown_knob4",
    "theta_known_knob4",
    "theta_unknown_knob0",
    "theta_known_knob0",
]
SCENARIO_TITLES = {
    "theta_unknown_knob4": r"$\theta$ unknown, knob$=4$" if USE_PGF else "θ unknown, knob=4",
    "theta_known_knob4":   r"$\theta$ known, knob$=4$"   if USE_PGF else "θ known, knob=4",
    "theta_unknown_knob0": r"$\theta$ unknown, knob$=0$" if USE_PGF else "θ unknown, knob=0",
    "theta_known_knob0":   r"$\theta$ known, knob$=0$"   if USE_PGF else "θ known, knob=0",
}


def plot_scenario_panel(ax, scenario_data, colors, title):
    """Single panel: absolute loss curves for 3 modes (single seed, no aggregation)."""
    for mode in MODES:
        if mode not in scenario_data:
            continue
        losses = scenario_data[mode]["all_losses"]
        steps = np.arange(len(losses))
        ax.plot(steps, losses, color=colors[mode], label=MODE_LABELS[mode], linewidth=1.2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss (absolute)")
    ax.set_title(title)
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser(description="Plot convergence analysis results")
    parser.add_argument("--input-dir", type=str, default="data/epistemic_maze/convergence")
    parser.add_argument("--output-dir", type=str, default="data/epistemic_maze/convergence/figures")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.parent.parent
    colors = load_colors(project_root)

    # Load data files
    datasets = {
        "curves": load_json(input_dir / "curves.json"),
        "lr_sweep": load_json(input_dir / "lr_sweep.json"),
        "budget": load_json(input_dir / "budget.json"),
        "variance": load_json(input_dir / "variance.json"),
    }

    scenario_data = load_json(input_dir / "scenario_curves.json")

    available = {k: v for k, v in datasets.items() if v is not None}
    if not available and scenario_data is None:
        print("No data files found — nothing to plot.")
        return

    # --- Combined 2×2 figure ---
    if available:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
        ax_map = {
            "curves": axes[0, 0],
            "lr_sweep": axes[0, 1],
            "budget": axes[1, 0],
            "variance": axes[1, 1],
        }
        for stem, plot_fn, data_key in PANELS:
            ax = ax_map[data_key]
            if data_key in available:
                plot_fn(ax, available[data_key], colors)
            else:
                ax.set_visible(False)

        save_figure(fig, output_dir, "convergence")
        plt.close(fig)
        print(f"Saved combined figure to {output_dir}/convergence.png" +
              (" + .pgf" if USE_PGF else ""))

        # --- Individual panels ---
        for stem, plot_fn, data_key in PANELS:
            if data_key not in available:
                continue
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(3.5, 2.5), constrained_layout=True)
            plot_fn(ax_ind, available[data_key], colors)
            save_figure(fig_ind, output_dir, stem)
            plt.close(fig_ind)
            print(f"Saved {stem}.png" + (" + .pgf" if USE_PGF else ""))

    # --- Scenario convergence 2×2 figure ---
    if scenario_data is not None:
        scenarios = scenario_data["scenarios"]
        # Combined 2×2
        fig_sc, axes_sc = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
        for idx, scenario_name in enumerate(SCENARIO_ORDER):
            if scenario_name not in scenarios:
                axes_sc.flat[idx].set_visible(False)
                continue
            plot_scenario_panel(
                axes_sc.flat[idx], scenarios[scenario_name], colors,
                SCENARIO_TITLES.get(scenario_name, scenario_name),
            )
        save_figure(fig_sc, output_dir, "scenario_convergence")
        plt.close(fig_sc)
        print(f"Saved scenario_convergence.png" + (" + .pgf" if USE_PGF else ""))

        # Individual panels
        for scenario_name in SCENARIO_ORDER:
            if scenario_name not in scenarios:
                continue
            fig_s, ax_s = plt.subplots(1, 1, figsize=(3.5, 2.5), constrained_layout=True)
            plot_scenario_panel(
                ax_s, scenarios[scenario_name], colors,
                SCENARIO_TITLES.get(scenario_name, scenario_name),
            )
            save_figure(fig_s, output_dir, f"scenario_{scenario_name}")
            plt.close(fig_s)
            print(f"Saved scenario_{scenario_name}.png" + (" + .pgf" if USE_PGF else ""))


if __name__ == "__main__":
    main()
