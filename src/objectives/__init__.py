"""Objective functions for planning."""

from .factorized_vfe import (
    factorized_vfe,
    extract_marginals_from_factorized,
    reconstruct_full_joint,
    enumerate_state_sequences,
    enumerate_action_sequences,
    enumerate_obs_sequences,
)

from .temporal_vfe import (
    temporal_vfe,
    temporal_vfe_jit,
    extract_marginals_temporal,
    compute_forward_marginals,
    group_modalities_for_jit,
    ModalityGroup,
)

__all__ = [
    # Factorized VFE (T-Maze)
    "factorized_vfe",
    "extract_marginals_from_factorized",
    "reconstruct_full_joint",
    "enumerate_state_sequences",
    "enumerate_action_sequences",
    "enumerate_obs_sequences",
    # Temporal VFE (unified: θ-independent policy, generic modalities, Bethe + planning correction)
    "temporal_vfe",
    "temporal_vfe_jit",
    "extract_marginals_temporal",
    "compute_forward_marginals",
    "group_modalities_for_jit",
    "ModalityGroup",
]
