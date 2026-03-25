"""Environment implementations for Active Inference experiments."""

from .tmaze import TMaze, create_tmaze_tensors
from .epistemic_maze import (
    EpistemicMaze,
    create_epistemic_maze_tensors,
    get_initial_state_distribution,
    state_to_components,
    components_to_state,
    get_state_name,
    get_action_name,
    N_STATES,
    N_ACTIONS,
    N_LOCATIONS,
    N_KNOB_STATES,
    N_NAV_STATES,
    Location,
    EpistemicAction,
)

__all__ = [
    "TMaze",
    "create_tmaze_tensors",
    "EpistemicMaze",
    "create_epistemic_maze_tensors",
    "get_initial_state_distribution",
    "state_to_components",
    "components_to_state",
    "get_state_name",
    "get_action_name",
    "N_STATES",
    "N_ACTIONS",
    "N_LOCATIONS",
    "N_KNOB_STATES",
    "N_NAV_STATES",
    "Location",
    "EpistemicAction",
]
