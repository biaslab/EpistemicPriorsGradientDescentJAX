"""Planning via variational optimization."""

from .factorized_optimizer import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    FactorizedPlanningResult,
)

from .temporal_optimizer import (
    plan_actions_temporal,
    select_action_temporal,
    TemporalPlanningConfig,
    TemporalPlanningResult,
)

from .sophisticated_planner import (
    SophisticatedPlanningConfig,
    SophisticatedPlanningResult,
    SophisticatedPlanner,
)

__all__ = [
    # Factorized VFE planner (T-Maze)
    "plan_actions_factorized",
    "select_action_factorized",
    "FactorizedPlanningConfig",
    "FactorizedPlanningResult",
    # Temporal VFE planner (Epistemic Maze, MiniGrid)
    "plan_actions_temporal",
    "select_action_temporal",
    "TemporalPlanningConfig",
    "TemporalPlanningResult",
    # Sophisticated (pymdp) planner
    "SophisticatedPlanningConfig",
    "SophisticatedPlanningResult",
    "SophisticatedPlanner",
]
