"""JAX Active Inference: Planning via Variational Optimization."""

from .distributions import (
    categorical_entropy, 
    categorical_kl, 
    conditional_entropy,
    joint_entropy,
)
from .objectives import (
    factorized_vfe,
    extract_marginals_from_factorized,
)
from .planning import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    FactorizedPlanningResult,
)
from .environments import TMaze, create_tmaze_tensors

__all__ = [
    "categorical_entropy",
    "categorical_kl", 
    "conditional_entropy",
    "joint_entropy",
    "factorized_vfe",
    "extract_marginals_from_factorized",
    "plan_actions_factorized",
    "select_action_factorized",
    "FactorizedPlanningConfig",
    "FactorizedPlanningResult",
    "TMaze",
    "create_tmaze_tensors",
]
