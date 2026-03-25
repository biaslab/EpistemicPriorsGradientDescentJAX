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

from .temporal_vfe_epistemic import (
    temporal_vfe as temporal_vfe_epistemic,
    extract_marginals_temporal as extract_marginals_temporal_epistemic,
    compute_forward_marginals as compute_forward_marginals_epistemic,
)

__all__ = [
    # Factorized VFE (T-Maze)
    "factorized_vfe",
    "extract_marginals_from_factorized",
    "reconstruct_full_joint",
    "enumerate_state_sequences",
    "enumerate_action_sequences",
    "enumerate_obs_sequences",
    # Temporal VFE (MiniGrid — canonical version with ModalityGroup, JIT, scan)
    "temporal_vfe",
    "temporal_vfe_jit",
    "extract_marginals_temporal",
    "compute_forward_marginals",
    "group_modalities_for_jit",
    "ModalityGroup",
    # Temporal VFE (Epistemic Maze — theta-dependent policy variant)
    "temporal_vfe_epistemic",
    "extract_marginals_temporal_epistemic",
    "compute_forward_marginals_epistemic",
]
