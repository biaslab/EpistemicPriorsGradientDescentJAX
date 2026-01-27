"""Planning via variational optimization with full joint factorization."""

from .full_joint_optimizer import (
    plan_actions,
    select_action,
    PlanningConfig,
    PlanningResult,
)

from .factorized_optimizer import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    FactorizedPlanningResult,
)

__all__ = [
    # Full joint
    "plan_actions",
    "select_action",
    "PlanningConfig",
    "PlanningResult",
    # Factorized
    "plan_actions_factorized",
    "select_action_factorized",
    "FactorizedPlanningConfig",
    "FactorizedPlanningResult",
]
