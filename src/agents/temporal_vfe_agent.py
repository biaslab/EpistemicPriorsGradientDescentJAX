"""Temporal VFE agent for MiniGrid environments."""

from dataclasses import dataclass, replace
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ..environments.environment_protocol import EnvironmentTensors
from ..planning.temporal_optimizer_minigrid import (
    TemporalPlanningConfig,
    plan_actions_temporal,
)

EPS = 1e-8


@dataclass
class TemporalVFEAgent:
    """Agent that uses temporal VFE planning for action selection.

    Implements the interface expected by gym_wrapper.run_episode:
      - reset() -> TemporalVFEAgent
      - step(vision_obs, orientation_obs, time_remaining) -> (int, TemporalVFEAgent)
      - planning_horizon: int
    """

    env_tensors: EnvironmentTensors
    config: TemporalPlanningConfig
    planning_horizon: int
    fov_size: int
    fov_gen: Array               # (n_patterns, n_states, n_theta) collapsed FOV
    orient_gen: Array            # (4, n_states)
    fov_pattern_map: dict        # tuple(int,...) → pattern index
    state_belief: Array          # (n_states,)
    theta_logits: Array          # (n_theta,)
    transition_index: Array = None  # (n_states, n_actions, n_theta) int32 or None
    step_count: int = 0

    def reset(self) -> "TemporalVFEAgent":
        n_theta = self.env_tensors.n_theta
        return replace(
            self,
            state_belief=self.env_tensors.initial_state,
            theta_logits=jnp.zeros(n_theta),
            step_count=0,
        )

    def step(
        self,
        vision_obs: Array,
        orientation_obs: Array,
        time_remaining: int,
    ) -> Tuple[int, "TemporalVFEAgent"]:
        # --- Belief update ---
        # Extract observation indices
        fov_pattern = tuple(int(x) for x in jnp.argmax(vision_obs, axis=-1).reshape(-1))
        orient_obs = jnp.argmax(orientation_obs)  # scalar

        # FOV log-likelihood: look up collapsed pattern
        pattern_idx = self.fov_pattern_map.get(fov_pattern)
        if pattern_idx is not None:
            log_fov = jnp.log(jnp.clip(self.fov_gen[pattern_idx, :, :], EPS, 1.0))  # (n_states, n_theta)
        else:
            # Unknown pattern — uniform likelihood (no information)
            log_fov = jnp.zeros((self.env_tensors.n_states, self.env_tensors.n_theta))

        # Orientation log-likelihood
        log_orient = jnp.log(jnp.clip(self.orient_gen[orient_obs, :], EPS, 1.0))  # (n_states,)

        # Joint posterior
        log_joint = (
            jnp.log(jnp.clip(self.state_belief, EPS, 1.0))[:, None]
            + self.theta_logits[None, :]
            + log_fov
            + log_orient[:, None]
        )  # (n_states, n_theta)

        # Normalize joint
        log_joint = log_joint - jax.scipy.special.logsumexp(log_joint)
        joint = jnp.exp(log_joint)

        # Marginalize
        new_state_belief = jnp.sum(joint, axis=1)  # (n_states,)
        new_state_belief = new_state_belief / jnp.sum(new_state_belief)

        new_theta_probs = jnp.sum(joint, axis=0)  # (n_theta,)
        new_theta_probs = new_theta_probs / jnp.sum(new_theta_probs)
        new_theta_logits = jnp.log(jnp.clip(new_theta_probs, EPS, 1.0))

        # --- Planning ---
        effective_horizon = min(int(time_remaining), self.planning_horizon)
        effective_horizon = max(effective_horizon, 1)

        step_config = replace(
            self.config,
            planning_horizon=effective_horizon,
            init_seed=self.config.init_seed + self.step_count,
        )
        result = plan_actions_temporal(
            new_state_belief, self.env_tensors, step_config, new_theta_logits
        )
        action = int(jnp.argmax(result.q_first_action))

        # --- State prediction after action ---
        new_theta = jax.nn.softmax(new_theta_logits)

        if self.transition_index is not None:
            # Index-based scatter-add: no dense transition tensor needed
            # transition_index: (n_states, n_actions, n_theta)
            idx = self.transition_index[:, action, :]  # (n_states, n_theta)
            weights = new_state_belief[:, None] * new_theta[None, :]  # (n_states, n_theta)
            n_states = self.env_tensors.n_states
            predicted_state = jnp.zeros(n_states).at[idx.ravel()].add(weights.ravel())
        elif self.env_tensors.theta_dependent_transitions:
            T = self.env_tensors.transition_tensor
            # T shape: (s_next, s_prev, theta, action)
            T_action = T[:, :, :, action]  # (s_next, s_prev, theta)
            predicted_state = jnp.einsum(
                "ijk,j,k->i", T_action, new_state_belief, new_theta
            )
        else:
            T = self.env_tensors.transition_tensor
            # T shape: (s_next, s_prev, action)
            T_action = T[:, :, action]  # (s_next, s_prev)
            predicted_state = T_action @ new_state_belief

        predicted_state = predicted_state / jnp.sum(predicted_state)

        new_agent = replace(
            self,
            state_belief=predicted_state,
            theta_logits=new_theta_logits,
            step_count=self.step_count + 1,
        )
        return action, new_agent


def create_temporal_vfe_agent(
    env_tensors: EnvironmentTensors,
    planning_horizon: int = 15,
    n_optimization_steps: int = 2000,
    learning_rate: float = 0.01,
    inference_mode: str = "active",
    init_seed: int = 42,
    fov_size: int = 7,
    fov_pattern_map: dict | None = None,
    freeze_obs_and_transitions: bool = False,
    policy_init_scale: float = 1.0,
    goal_scale: float = 1.0,
    optimizer_type: str = "adam",
) -> TemporalVFEAgent:
    """Create a TemporalVFEAgent from environment tensors.

    Finds the collapsed FOV modality and orientation modality for belief updates.
    """
    n_states = env_tensors.n_states
    n_theta = env_tensors.n_theta

    fov_gen = None
    orient_gen = None

    for mod in env_tensors.observation_modalities:
        if mod.name == "fov":
            fov_gen = mod.generative_tensor
        elif mod.name == "orientation":
            orient_gen = mod.generative_tensor

    if fov_gen is None:
        raise ValueError("No 'fov' modality found in env_tensors")
    if orient_gen is None:
        raise ValueError("No 'orientation' modality found in env_tensors")
    if fov_pattern_map is None:
        raise ValueError("fov_pattern_map is required for collapsed FOV belief updates")

    config = TemporalPlanningConfig(
        planning_horizon=planning_horizon,
        n_states=n_states,
        n_actions=env_tensors.n_actions,
        n_theta=n_theta,
        n_optimization_steps=n_optimization_steps,
        learning_rate=learning_rate,
        inference_mode=inference_mode,
        init_seed=init_seed,
        freeze_obs_and_transitions=freeze_obs_and_transitions,
        policy_init_scale=policy_init_scale,
        goal_scale=goal_scale,
        optimizer_type=optimizer_type,
    )

    return TemporalVFEAgent(
        env_tensors=env_tensors,
        config=config,
        planning_horizon=planning_horizon,
        fov_size=fov_size,
        fov_gen=fov_gen,
        orient_gen=orient_gen,
        fov_pattern_map=fov_pattern_map,
        state_belief=env_tensors.initial_state,
        theta_logits=jnp.zeros(n_theta),
        transition_index=env_tensors.transition_index,
    )
