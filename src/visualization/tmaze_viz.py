"""
T-Maze visualization utilities.

Creates matplotlib visualizations of the T-maze environment,
similar to the Julia Plots.jl implementation.
Also supports TikZ output for LaTeX papers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
import numpy as np


# ColorBrewer Paired_9 palette (matching Julia ColorSchemes.jl)
# Index: 1=#a6cee3, 2=#1f78b4, 3=#b2df8a, 4=#33a02c, 5=#fb9a99, 
#        6=#e31a1c, 7=#fdbf6f, 8=#ff7f00, 9=#cab2d6
PAIRED_9 = {
    1: "#a6cee3",  # light blue
    2: "#1f78b4",  # blue
    3: "#b2df8a",  # light green
    4: "#33a02c",  # green
    5: "#fb9a99",  # light red/pink
    6: "#e31a1c",  # red
    7: "#fdbf6f",  # light orange
    8: "#ff7f00",  # orange
    9: "#cab2d6",  # light purple
}


@dataclass
class PlanData:
    """
    Planning data for visualization.
    
    Contains the full plan: expected states and action distributions
    at each future time step.
    """
    all_action_probs: List[List[float]]  # (horizon, n_actions) - q(u_t) for each future t
    all_state_probs: List[List[float]]   # (horizon, n_states) - q(x_t) for each future t


# State coordinates for the T-maze
STATE_COORDS = {
    0: (1.5, 1.5),   # BOTTOM (cue)
    1: (1.5, 2.5),   # MIDDLE
    2: (0.5, 3.5),   # TOP_LEFT
    3: (1.5, 3.5),   # TOP_MIDDLE
    4: (2.5, 3.5),   # TOP_RIGHT
}

# Action directions (dx, dy) for arrows
# Actions: 0=N, 1=E, 2=S, 3=W
ACTION_DIRECTIONS = {
    0: (0, 0.35),    # North
    1: (0.35, 0),    # East
    2: (0, -0.35),   # South
    3: (-0.35, 0),   # West
}

# Valid transitions from each state
VALID_ACTIONS = {
    0: [0],           # From CUE: can only go North
    1: [0, 2],        # From MIDDLE: North or South
    2: [],            # From TOP_LEFT: sink state (no valid actions)
    3: [1, 3],        # From TOP_MIDDLE: East or West
    4: [],            # From TOP_RIGHT: sink state (no valid actions)
}


@dataclass
class MazeTheme:
    """
    Color theme for T-maze visualization.
    
    Default colors use ColorBrewer Paired_9 palette for consistency
    with academic publications.
    """
    background: str = "#ffffff"
    corridor: str = "#ffffff"
    wall: str = "#000000"
    agent: str = PAIRED_9[2]           # blue
    reward_positive: str = PAIRED_9[4]  # green
    reward_negative: str = PAIRED_9[6]  # red
    cue: str = PAIRED_9[7]              # light orange


@dataclass
class TikzTheme:
    """
    Color theme for TikZ T-maze visualization.
    
    Uses HTML hex colors that will be defined in the TikZ preamble.
    Default colors use ColorBrewer Paired_9 palette.
    """
    background: str = "white"
    corridor: str = "white"
    wall: str = "black"
    agent: str = "pairedblue"           # #1f78b4
    reward_positive: str = "pairedgreen" # #33a02c
    reward_negative: str = "pairedred"   # #e31a1c
    cue: str = "pairedorange"            # #fdbf6f


def get_tikz_color_definitions() -> str:
    """
    Get TikZ/xcolor color definitions for the Paired_9 palette.
    
    Returns:
        LaTeX code defining the custom colors.
    """
    return r"""\definecolor{pairedlightblue}{HTML}{A6CEE3}
\definecolor{pairedblue}{HTML}{1F78B4}
\definecolor{pairedlightgreen}{HTML}{B2DF8A}
\definecolor{pairedgreen}{HTML}{33A02C}
\definecolor{pairedlightred}{HTML}{FB9A99}
\definecolor{pairedred}{HTML}{E31A1C}
\definecolor{pairedorange}{HTML}{FDBF6F}
\definecolor{paireddarkorange}{HTML}{FF7F00}
\definecolor{pairedpurple}{HTML}{CAB2D6}"""


def _draw_plan_arrows_mpl(
    ax: plt.Axes,
    plan_data: PlanData,
    theme: MazeTheme,
    current_state: int,
    arrow_color: str = "#8B4513",  # saddle brown for plan arrows
) -> None:
    """
    Draw plan arrows on matplotlib axes.
    
    Shows the full planned trajectory with arrows indicating
    the planned action distribution at each time step.
    
    The mapping is:
    - t=0: Arrows at current_state using all_action_probs[0]
    - t=1: Arrows at expected states all_state_probs[0] using all_action_probs[1]
    - t=2: Arrows at expected states all_state_probs[1] using all_action_probs[2]
    - etc.
    """
    all_state_probs = np.array(plan_data.all_state_probs)
    all_action_probs = np.array(plan_data.all_action_probs)
    horizon = len(all_state_probs)
    
    for t in range(horizon):
        action_probs = all_action_probs[t]
        
        if t == 0:
            # First action is from current state (deterministic)
            state_probs = np.zeros(5)
            state_probs[current_state] = 1.0
        else:
            # Subsequent actions are from expected states at previous timestep
            state_probs = all_state_probs[t - 1]
        
        # Draw arrows from each state weighted by state probability
        for state_idx, state_prob in enumerate(state_probs):
            if state_prob < 0.01:  # Skip very low probability states
                continue
            
            x, y = STATE_COORDS[state_idx]
            valid_actions = VALID_ACTIONS[state_idx]
            
            for action_idx in valid_actions:
                action_prob = action_probs[action_idx]
                if action_prob < 0.01:  # Skip very low probability actions
                    continue
                
                # Combined probability
                combined_prob = state_prob * action_prob
                
                # Sigmoid alpha: low probs -> invisible, high probs -> very visible
                alpha = combined_prob
                
                dx, dy = ACTION_DIRECTIONS[action_idx]
                
                # Draw arrow
                ax.annotate(
                    '',
                    xy=(x + dx, y + dy),
                    xytext=(x, y),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=arrow_color,
                        alpha=alpha,
                        lw=2,
                        mutation_scale=15,
                    ),
                    zorder=4,
                )


def plot_tmaze_frame(
    agent_state: int,
    reward_location: str,
    has_seen_cue: bool = False,
    theme: Optional[MazeTheme] = None,
    step: Optional[int] = None,
    action: Optional[int] = None,
    figsize: tuple = (6, 6),
    plan_data: Optional[PlanData] = None,
) -> Figure:
    """
    Create a visualization of the T-maze environment.
    
    Args:
        agent_state: Current state of the agent (0-4).
            0 = BOTTOM (cue), 1 = MIDDLE, 2 = TOP_LEFT, 
            3 = TOP_MIDDLE, 4 = TOP_RIGHT
        reward_location: 'left' or 'right' indicating reward arm.
        has_seen_cue: Whether agent has visited the cue location.
        theme: Color theme for visualization.
        step: Optional step number to display.
        action: Optional action taken (0=N, 1=E, 2=S, 3=W).
        figsize: Figure size in inches.
        plan_data: Optional planning data for showing plan arrows.
        
    Returns:
        Matplotlib Figure object.
    """
    if theme is None:
        theme = MazeTheme()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(0.8, 4.2)
    ax.axis('off')
    fig.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)
    
    # Draw corridors (filled rectangles)
    # Vertical corridor: [1, 2] x [1, 3]
    vertical_corridor = patches.Rectangle(
        (1, 1), 1, 2,
        facecolor=theme.corridor,
        edgecolor='none'
    )
    ax.add_patch(vertical_corridor)
    
    # Horizontal corridor at top: [0, 3] x [3, 4]
    horizontal_corridor = patches.Rectangle(
        (0, 3), 3, 1,
        facecolor=theme.corridor,
        edgecolor='none'
    )
    ax.add_patch(horizontal_corridor)
    
    # Draw walls (thick lines)
    wall_props = dict(color=theme.wall, linewidth=2, solid_capstyle='round')
    
    # Vertical corridor walls
    ax.plot([1, 1], [1, 3], **wall_props)  # Left vertical wall
    ax.plot([2, 2], [1, 3], **wall_props)  # Right vertical wall
    
    # Horizontal corridor walls
    ax.plot([0, 3], [4, 4], **wall_props)  # Top wall
    ax.plot([0, 1], [3, 3], **wall_props)  # Bottom left horizontal
    ax.plot([2, 3], [3, 3], **wall_props)  # Bottom right horizontal
    
    # Bottom wall of vertical corridor
    ax.plot([1, 2], [1, 1], **wall_props)
    
    # Left and right walls of horizontal corridor
    ax.plot([0, 0], [3, 4], **wall_props)  # Left wall
    ax.plot([3, 3], [3, 4], **wall_props)  # Right wall
    
    # Draw grid lines (subtle)
    grid_props = dict(color=theme.wall, linewidth=0.5, alpha=0.4)
    ax.plot([1, 2], [2, 2], **grid_props)  # Horizontal grid in vertical corridor
    ax.plot([1, 2], [3, 3], **grid_props)  # Horizontal grid between middle and top
    ax.plot([1, 1], [3, 4], **grid_props)  # Left vertical grid in horizontal corridor
    ax.plot([2, 2], [3, 4], **grid_props)  # Right vertical grid in horizontal corridor
    
    # Draw reward locations
    marker_size = 500
    stroke_width = 2
    
    # Left reward (state 2)
    left_color = theme.reward_positive if reward_location == 'left' else theme.reward_negative
    ax.scatter([0.5], [3.5], s=marker_size, c=left_color, alpha=0.7,
               edgecolors=theme.wall, linewidths=stroke_width, zorder=2)
    
    # Right reward (state 4)
    right_color = theme.reward_positive if reward_location == 'right' else theme.reward_negative
    ax.scatter([2.5], [3.5], s=marker_size, c=right_color, alpha=0.7,
               edgecolors=theme.wall, linewidths=stroke_width, zorder=2)
    
    # Draw cue location (state 0)
    ax.scatter([1.5], [1.5], s=marker_size, c=theme.cue, alpha=0.7,
               edgecolors=theme.wall, linewidths=stroke_width, zorder=2)
    
    # Add "Cue" label if agent is not at cue location
    if agent_state != 0:
        ax.text(1.5, 1.5, "Cue", ha='center', va='center',
                fontsize=10, fontweight='bold', color=theme.wall)
    
    agent_x, agent_y = STATE_COORDS[agent_state]
    
    # Draw agent
    agent_size = 250
    ax.scatter([agent_x], [agent_y], s=agent_size, c=theme.agent, 
               edgecolors=theme.wall, linewidths=stroke_width, zorder=3)
    
    # Draw plan arrows if plan data is provided
    if plan_data is not None:
        _draw_plan_arrows_mpl(ax, plan_data, theme, current_state=agent_state)
    
    # Add step/action info if provided
    title_parts = []
    if step is not None:
        title_parts.append(f"Step {step}")
    if action is not None:
        action_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        title_parts.append(f"Action: {action_names.get(action, '?')}")
    
    if title_parts:
        ax.set_title(" | ".join(title_parts), fontsize=12, fontweight='bold',
                     color=theme.wall, pad=10)
    
    # Add legend for reward location knowledge
    if has_seen_cue:
        info_text = f"Reward: {reward_location.upper()}"
    else:
        info_text = "Reward: ???"
    
    ax.text(1.5, 0.95, info_text, ha='center', va='top',
            fontsize=10, color=theme.wall, fontstyle='italic')
    
    plt.tight_layout()
    return fig


def create_episode_video(
    trajectory: List[int],
    actions: List[int],
    reward_location: str,
    output_path: Path,
    fps: int = 2,
    theme: Optional[MazeTheme] = None,
    planning_history: Optional[List[PlanData]] = None,
) -> None:
    """
    Create a video from an episode's trajectory.
    
    Args:
        trajectory: List of states visited (including initial state).
        actions: List of actions taken.
        reward_location: 'left' or 'right'.
        output_path: Path to save the video (e.g., 'episode.mp4').
        fps: Frames per second.
        theme: Color theme for visualization.
        planning_history: Optional list of PlanData for each step (shows plan arrows).
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("imageio is required for video creation. Install with: pip install imageio[ffmpeg]")
    
    frames = []
    
    # First frame: initial state (no action yet)
    # Planning for step 0 is available if planning_history is provided
    has_seen_cue = trajectory[0] == 0
    plan_data = planning_history[0] if planning_history else None
    fig = plot_tmaze_frame(
        agent_state=trajectory[0],
        reward_location=reward_location,
        has_seen_cue=has_seen_cue,
        theme=theme,
        step=0,
        action=None,
        plan_data=plan_data,
    )
    frames.append(_fig_to_array(fig))
    plt.close(fig)
    
    # Subsequent frames: after each action
    for step, (state, action) in enumerate(zip(trajectory[1:], actions), start=1):
        # Update has_seen_cue based on trajectory
        has_seen_cue = any(s == 0 for s in trajectory[:step+1])
        
        # Planning for this step (if available)
        plan_data = planning_history[step] if planning_history and step < len(planning_history) else None
        
        fig = plot_tmaze_frame(
            agent_state=state,
            reward_location=reward_location,
            has_seen_cue=has_seen_cue,
            theme=theme,
            step=step,
            action=action,
            plan_data=plan_data,
        )
        frames.append(_fig_to_array(fig))
        plt.close(fig)
    
    # Write video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use imageio to write video
    iio.imwrite(output_path, frames, fps=fps, codec='libx264')
    print(f"Video saved to: {output_path}")


def _fig_to_array(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    
    try:
        import imageio.v3 as iio
        img = iio.imread(buf)
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(buf))
    
    buf.close()
    return img


def _generate_tikz_plan_arrows(
    plan_data: PlanData,
    current_state: int,
    arrow_color: str = "paireddarkorange",
) -> List[str]:
    """
    Generate TikZ code for plan arrows.
    
    The mapping is:
    - t=0: Arrows at current_state using all_action_probs[0]
    - t=1: Arrows at expected states all_state_probs[0] using all_action_probs[1]
    - etc.
    
    Returns a list of TikZ lines to draw the arrows.
    """
    all_state_probs = plan_data.all_state_probs
    all_action_probs = plan_data.all_action_probs
    horizon = len(all_state_probs)
    
    lines = ["", "  % Plan arrows"]
    
    for t in range(horizon):
        action_probs = all_action_probs[t]
        
        if t == 0:
            # First action is from current state (deterministic)
            state_probs = [0.0] * 5
            state_probs[current_state] = 1.0
        else:
            # Subsequent actions are from expected states at previous timestep
            state_probs = all_state_probs[t - 1]
        
        for state_idx, state_prob in enumerate(state_probs):
            if state_prob < 0.01:
                continue
            
            x, y = STATE_COORDS[state_idx]
            valid_actions = VALID_ACTIONS[state_idx]
            
            for action_idx in valid_actions:
                action_prob = action_probs[action_idx]
                if action_prob < 0.01:
                    continue
                
                # Combined probability
                combined_prob = state_prob * action_prob
                
                # Sigmoid alpha: low probs -> invisible, high probs -> very visible
                alpha = combined_prob
                
                dx, dy = ACTION_DIRECTIONS[action_idx]
                end_x, end_y = x + dx, y + dy
                
                # TikZ arrow with opacity
                lines.append(
                    f"  \\draw[->, {arrow_color}, line width=1.5pt, opacity={alpha:.2f}] "
                    f"({x}, {y}) -- ({end_x}, {end_y});"
                )
    
    return lines


def generate_tikz_frame(
    agent_state: int,
    reward_location: str,
    has_seen_cue: bool = False,
    theme: Optional[TikzTheme] = None,
    step: Optional[int] = None,
    action: Optional[int] = None,
    standalone: bool = False,
    show_legend: bool = False,
    show_title: bool = False,
    plan_data: Optional[PlanData] = None,
) -> str:
    """
    Generate TikZ code for a T-maze visualization frame.
    
    Args:
        agent_state: Current state of the agent (0-4).
            0 = BOTTOM (cue), 1 = MIDDLE, 2 = TOP_LEFT, 
            3 = TOP_MIDDLE, 4 = TOP_RIGHT
        reward_location: 'left' or 'right' indicating reward arm.
        has_seen_cue: Whether agent has visited the cue location.
        theme: Color theme for visualization.
        step: Optional step number to display.
        action: Optional action taken (0=N, 1=E, 2=S, 3=W).
        standalone: If True, wrap in standalone document class (default: False).
        show_legend: If True, include legend at bottom (default: False).
        show_title: If True, show "Step N | Action" title (default: False).
        plan_data: Optional planning data for showing plan arrows.
        
    Returns:
        TikZ code as a string.
    """
    if theme is None:
        theme = TikzTheme()
    
    agent_x, agent_y = STATE_COORDS[agent_state]
    
    # Determine reward colors
    left_color = theme.reward_positive if reward_location == 'left' else theme.reward_negative
    right_color = theme.reward_positive if reward_location == 'right' else theme.reward_negative
    
    # Build title
    title_parts = []
    if step is not None:
        title_parts.append(f"Step {step}")
    if action is not None:
        action_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        title_parts.append(f"Action: {action_names.get(action, '?')}")
    title_text = " | ".join(title_parts) if title_parts else ""
    
    # Build TikZ code
    tikz_lines = []
    
    if standalone:
        tikz_lines.extend([
            r"\documentclass[tikz,border=5pt]{standalone}",
            r"\usepackage{tikz}",
            r"\usepackage[dvipsnames]{xcolor}",
            r"\usetikzlibrary{shapes,positioning,arrows.meta}",
            "",
            "% ColorBrewer Paired_9 palette",
            get_tikz_color_definitions(),
            "",
            r"\begin{document}",
        ])
    
    tikz_lines.extend([
        r"\begin{tikzpicture}[scale=1.5]",
        f"  % Background",
        f"  \\fill[{theme.background}] (-0.2, 0.5) rectangle (3.2, 4.5);",
        "",
        f"  % Corridors",
        f"  \\fill[{theme.corridor}] (1, 1) rectangle (2, 3);  % Vertical corridor",
        f"  \\fill[{theme.corridor}] (0, 3) rectangle (3, 4);  % Horizontal corridor",
        "",
        f"  % Walls (thick lines)",
        f"  \\draw[{theme.wall}, line width=1.5pt, line cap=round]",
        f"    % Vertical corridor walls",
        f"    (1, 1) -- (1, 3)",
        f"    (2, 1) -- (2, 3)",
        f"    % Horizontal corridor walls",
        f"    (0, 4) -- (3, 4)",
        f"    (0, 3) -- (1, 3)",
        f"    (2, 3) -- (3, 3)",
        f"    % Bottom wall",
        f"    (1, 1) -- (2, 1)",
        f"    % Side walls",
        f"    (0, 3) -- (0, 4)",
        f"    (3, 3) -- (3, 4);",
        "",
        f"  % Grid lines (subtle)",
        f"  \\draw[{theme.wall}, line width=0.3pt, opacity=0.4]",
        f"    (1, 2) -- (2, 2)",
        f"    (1, 3) -- (2, 3)",
        f"    (1, 3) -- (1, 4)",
        f"    (2, 3) -- (2, 4);",
        "",
        f"  % Reward locations",
        f"  \\fill[{left_color}, opacity=0.7] (0.5, 3.5) circle (0.2);",
        f"  \\draw[{theme.wall}, line width=1pt] (0.5, 3.5) circle (0.2);",
        f"  \\fill[{right_color}, opacity=0.7] (2.5, 3.5) circle (0.2);",
        f"  \\draw[{theme.wall}, line width=1pt] (2.5, 3.5) circle (0.2);",
        "",
        f"  % Cue location",
        f"  \\fill[{theme.cue}, opacity=0.7] (1.5, 1.5) circle (0.2);",
        f"  \\draw[{theme.wall}, line width=1pt] (1.5, 1.5) circle (0.2);",
        "",
        f"  % Agent",
        f"  \\fill[{theme.agent}] ({agent_x}, {agent_y}) circle (0.13);",
        f"  \\draw[{theme.wall}, line width=1pt] ({agent_x}, {agent_y}) circle (0.13);",
    ])
    
    # Add plan arrows if plan data is provided
    if plan_data is not None:
        tikz_lines.extend(_generate_tikz_plan_arrows(plan_data, current_state=agent_state))
    
    # Add title if requested and present
    if show_title and title_text:
        tikz_lines.extend([
            "",
            f"  % Title",
            f"  \\node[font=\\bfseries, anchor=south] at (1.5, 4.2) {{{title_text}}};",
        ])
    
    # Add legend if requested
    if show_legend:
        tikz_lines.extend([
            "",
            f"  % Legend",
            f"  \\node[anchor=west, font=\\footnotesize] at (-0.1, 0.75) {{",
            f"    \\tikz{{\\fill[{theme.agent}] (0,0) circle (0.15); \\draw[{theme.wall}, line width=0.5pt] (0,0) circle (0.15);}} Agent",
            f"    \\quad",
            f"    \\tikz{{\\fill[{theme.cue}, opacity=0.7] (0,0) circle (0.15); \\draw[{theme.wall}, line width=0.5pt] (0,0) circle (0.15);}} Cue",
            f"    \\quad",
            f"    \\tikz{{\\fill[{theme.reward_positive}, opacity=0.7] (0,0) circle (0.15); \\draw[{theme.wall}, line width=0.5pt] (0,0) circle (0.15);}} Reward",
            f"    \\quad",
            f"    \\tikz{{\\fill[{theme.reward_negative}, opacity=0.7] (0,0) circle (0.15); \\draw[{theme.wall}, line width=0.5pt] (0,0) circle (0.15);}} Loss",
            f"  }};",
        ])
    
    tikz_lines.append(r"\end{tikzpicture}")
    
    if standalone:
        tikz_lines.append(r"\end{document}")
    
    return "\n".join(tikz_lines)


def generate_tmaze_reference(
    theme: Optional[TikzTheme] = None,
    standalone: bool = False,
) -> str:
    """
    Generate a reference T-maze figure with legend showing all elements.
    
    This creates a figure at the starting position (MIDDLE) with both
    reward locations shown as unknown (both positive color) and includes
    a legend explaining all symbols.
    
    Args:
        theme: Color theme for visualization.
        standalone: If True, wrap in standalone document class.
        
    Returns:
        TikZ code as a string.
    """
    return generate_tikz_frame(
        agent_state=1,  # MIDDLE (starting position)
        reward_location='left',  # Arbitrary, both shown same for reference
        has_seen_cue=False,
        theme=theme,
        step=None,
        action=None,
        standalone=standalone,
        show_legend=True,
    )


def save_tmaze_reference(
    output_path: Path,
    theme: Optional[TikzTheme] = None,
    standalone: bool = False,
) -> Path:
    """
    Save a reference T-maze figure with legend.
    
    Also saves colors.tex in the same directory for preamble inclusion.
    
    Args:
        output_path: Path to save the .tex file (e.g., 'data/tmaze.tex').
        theme: Color theme for visualization.
        standalone: If True, wrap in standalone document class.
        
    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tikz_code = generate_tmaze_reference(theme=theme, standalone=standalone)
    
    with open(output_path, 'w') as f:
        f.write(tikz_code)
    
    # Also save colors.tex in the same directory
    colors_path = output_path.parent / "colors.tex"
    if not colors_path.exists():
        with open(colors_path, 'w') as f:
            f.write("% ColorBrewer Paired_9 palette for T-maze figures\n")
            f.write("% Include this in your paper preamble with: \\input{data/colors.tex}\n")
            f.write(get_tikz_color_definitions())
            f.write("\n")
        print(f"Color definitions saved to: {colors_path}")
    
    print(f"T-maze reference figure saved to: {output_path}")
    return output_path


def save_episode_tikz_frames(
    trajectory: List[int],
    actions: List[int],
    reward_location: str,
    output_dir: Path,
    theme: Optional[TikzTheme] = None,
    standalone: bool = False,
    planning_history: Optional[List[PlanData]] = None,
) -> List[Path]:
    """
    Save TikZ frames for an episode's trajectory.
    
    When planning_history is provided, saves BOTH versions for each frame:
    - frame_XX.tex: without planning arrows
    - frame_XX_arrows.tex: with planning arrows
    
    Args:
        trajectory: List of states visited (including initial state).
        actions: List of actions taken.
        reward_location: 'left' or 'right'.
        output_dir: Directory to save the .tex files.
        theme: Color theme for visualization.
        standalone: If False (default), outputs just tikzpicture for inclusion in papers.
        planning_history: Optional list of PlanData for each step (shows plan arrows).
        
    Returns:
        List of paths to saved .tex files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Save color definitions for paper preamble
    colors_path = output_dir / "colors.tex"
    with open(colors_path, 'w') as f:
        f.write("% ColorBrewer Paired_9 palette for T-maze figures\n")
        f.write("% Include this in your paper preamble with: \\input{frames/colors.tex}\n")
        f.write(get_tikz_color_definitions())
        f.write("\n")
    saved_paths.append(colors_path)
    
    def save_frame(step: int, state: int, action: Optional[int], has_seen_cue: bool, plan_data: Optional[PlanData]):
        """Helper to save a single frame (with and without arrows if plan_data provided)."""
        paths = []
        
        # Always save version WITHOUT arrows
        tikz_code_no_arrows = generate_tikz_frame(
            agent_state=state,
            reward_location=reward_location,
            has_seen_cue=has_seen_cue,
            theme=theme,
            step=step,
            action=action,
            standalone=standalone,
            plan_data=None,  # No arrows
        )
        
        frame_path = output_dir / f"frame_{step:02d}.tex"
        with open(frame_path, 'w') as f:
            f.write(tikz_code_no_arrows)
        paths.append(frame_path)
        
        # If planning history available, also save version WITH arrows
        if plan_data is not None:
            tikz_code_with_arrows = generate_tikz_frame(
                agent_state=state,
                reward_location=reward_location,
                has_seen_cue=has_seen_cue,
                theme=theme,
                step=step,
                action=action,
                standalone=standalone,
                plan_data=plan_data,  # With arrows
            )
            
            frame_path_arrows = output_dir / f"frame_{step:02d}_arrows.tex"
            with open(frame_path_arrows, 'w') as f:
                f.write(tikz_code_with_arrows)
            paths.append(frame_path_arrows)
        
        return paths
    
    # First frame: initial state (no action yet)
    has_seen_cue = trajectory[0] == 0
    plan_data = planning_history[0] if planning_history else None
    saved_paths.extend(save_frame(0, trajectory[0], None, has_seen_cue, plan_data))
    
    # Subsequent frames: after each action
    for step, (state, action) in enumerate(zip(trajectory[1:], actions), start=1):
        has_seen_cue = any(s == 0 for s in trajectory[:step+1])
        plan_data = planning_history[step] if planning_history and step < len(planning_history) else None
        saved_paths.extend(save_frame(step, state, action, has_seen_cue, plan_data))
    
    print(f"TikZ frames saved to: {output_dir}")
    return saved_paths
