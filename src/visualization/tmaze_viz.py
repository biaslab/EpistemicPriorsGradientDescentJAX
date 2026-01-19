"""
T-Maze visualization utilities.

Creates matplotlib visualizations of the T-maze environment,
similar to the Julia Plots.jl implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import numpy as np


@dataclass
class MazeTheme:
    """Color theme for T-maze visualization."""
    background: str = "#f5f5f5"
    corridor: str = "#e8e8e8"
    wall: str = "#2d2d2d"
    agent: str = "#4a90d9"
    reward_positive: str = "#4caf50"
    reward_negative: str = "#f44336"
    cue: str = "#ff9800"


def plot_tmaze_frame(
    agent_state: int,
    reward_location: str,
    has_seen_cue: bool = False,
    theme: Optional[MazeTheme] = None,
    step: Optional[int] = None,
    action: Optional[int] = None,
    figsize: tuple = (6, 6),
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
    ax.plot([1, 1], [3, 4], **grid_props)  # Left vertical grid in horizontal corridor
    ax.plot([2, 2], [3, 4], **grid_props)  # Right vertical grid in horizontal corridor
    
    # Draw reward locations
    marker_size = 800
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
    
    # Map agent state to coordinates
    state_to_coords = {
        0: (1.5, 1.5),   # BOTTOM (cue)
        1: (1.5, 2.5),   # MIDDLE
        2: (0.5, 3.5),   # TOP_LEFT
        3: (1.5, 3.5),   # TOP_MIDDLE
        4: (2.5, 3.5),   # TOP_RIGHT
    }
    
    agent_x, agent_y = state_to_coords[agent_state]
    
    # Draw agent
    agent_size = 500
    ax.scatter([agent_x], [agent_y], s=agent_size, c=theme.agent, 
               edgecolors=theme.wall, linewidths=stroke_width, zorder=3)
    
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
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("imageio is required for video creation. Install with: pip install imageio[ffmpeg]")
    
    frames = []
    
    # First frame: initial state (no action yet)
    has_seen_cue = trajectory[0] == 0
    fig = plot_tmaze_frame(
        agent_state=trajectory[0],
        reward_location=reward_location,
        has_seen_cue=has_seen_cue,
        theme=theme,
        step=0,
        action=None,
    )
    frames.append(_fig_to_array(fig))
    plt.close(fig)
    
    # Subsequent frames: after each action
    for step, (state, action) in enumerate(zip(trajectory[1:], actions), start=1):
        # Update has_seen_cue based on trajectory
        has_seen_cue = any(s == 0 for s in trajectory[:step+1])
        
        fig = plot_tmaze_frame(
            agent_state=state,
            reward_location=reward_location,
            has_seen_cue=has_seen_cue,
            theme=theme,
            step=step,
            action=action,
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
