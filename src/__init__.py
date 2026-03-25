"""Epistemic Priors for Active Inference: Planning via Variational Optimization in JAX."""

from .distributions import (
    categorical_entropy,
    categorical_kl,
    conditional_entropy,
    joint_entropy,
)
from .objectives import (
    factorized_vfe,
    extract_marginals_from_factorized,
    temporal_vfe,
    temporal_vfe_jit,
)
from .planning import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    FactorizedPlanningResult,
    plan_actions_temporal,
    select_action_temporal,
    TemporalPlanningConfig,
    TemporalPlanningResult,
)
from .environments import TMaze, create_tmaze_tensors

__all__ = [
    "categorical_entropy",
    "categorical_kl",
    "conditional_entropy",
    "joint_entropy",
    "factorized_vfe",
    "extract_marginals_from_factorized",
    "temporal_vfe",
    "temporal_vfe_jit",
    "plan_actions_factorized",
    "select_action_factorized",
    "FactorizedPlanningConfig",
    "FactorizedPlanningResult",
    "plan_actions_temporal",
    "select_action_temporal",
    "TemporalPlanningConfig",
    "TemporalPlanningResult",
    "TMaze",
    "create_tmaze_tensors",
]
