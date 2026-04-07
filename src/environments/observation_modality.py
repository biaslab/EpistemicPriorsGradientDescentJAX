"""Observation modality dataclass for multi-modality VFE."""

from dataclasses import dataclass
from typing import Optional
from jax import Array


@dataclass(frozen=True)
class ObservationModality:
    """A single observation modality in the generative model.

    Attributes:
        name: Human-readable identifier (e.g. "theta", "location", "fov_3_2", "reward")
        generative_tensor: p(o|s,θ) shape (n_obs, n_states, n_theta) if theta_dependent,
                          or p(o|s) shape (n_obs, n_states) if not
        theta_dependent: Whether observations depend on the latent context θ
        n_obs: Number of observation outcomes
        observation_index: Optional deterministic obs index.
            theta_dependent: shape (n_states, n_theta) -> int obs outcome
            theta_independent: shape (n_states,) -> int obs outcome
    """
    name: str
    generative_tensor: Array
    theta_dependent: bool
    n_obs: int
    observation_index: Optional[Array] = None
