"""Environment tensors dataclass for general environment abstraction."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from jax import Array

from .observation_modality import ObservationModality


@dataclass
class EnvironmentTensors:
    """All tensors needed to run temporal VFE planning for any environment.

    Attributes:
        n_states: Number of hidden states
        n_actions: Number of actions
        n_theta: Number of latent context values
        transition_tensor: p(s'|s,a) shape (s,s,a) or p(s'|s,θ,a) shape (s,s,θ,a).
            Can be None when transition_index is provided.
        theta_dependent_transitions: Whether transitions depend on θ
        observation_modalities: List of observation modalities
        goal_mapping: p(goal|s,θ) shape (s, θ)
        action_prior: p(a) shape (a,)
        theta_prior: p(θ) shape (θ,)
        metadata: Optional extra data (e.g. fov_pattern_map)
        transition_index: Optional deterministic transition index.
            shape (n_states, n_actions, n_theta) -> int next_state
    """
    n_states: int
    n_actions: int
    n_theta: int
    transition_tensor: Optional[Array]
    theta_dependent_transitions: bool
    observation_modalities: List[ObservationModality]
    goal_mapping: Array
    initial_state: Array
    action_prior: Array
    theta_prior: Array
    metadata: Dict[str, Any] = field(default_factory=dict)
    transition_index: Optional[Array] = None

    @property
    def planning_modalities(self) -> List[ObservationModality]:
        return [m for m in self.observation_modalities if m.theta_dependent]
