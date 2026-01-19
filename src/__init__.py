"""JAX Active Inference: Planning via Variational Optimization."""

from .distributions import (
    categorical_entropy, 
    categorical_kl, 
    conditional_entropy,
    joint_entropy,
)
from .objectives import (
    full_joint_vfe,
    extract_first_action_marginal,
)
from .planning import (
    plan_actions,
    select_action,
    PlanningConfig,
    PlanningResult,
)
from .environments import TMaze, create_tmaze_tensors

__all__ = [
    "categorical_entropy",
    "categorical_kl", 
    "conditional_entropy",
    "joint_entropy",
    "full_joint_vfe",
    "extract_first_action_marginal",
    "plan_actions",
    "select_action",
    "PlanningConfig",
    "PlanningResult",
    "TMaze",
    "create_tmaze_tensors",
]
