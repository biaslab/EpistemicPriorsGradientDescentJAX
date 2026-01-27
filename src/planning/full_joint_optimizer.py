"""Planning via VFE minimization with full joint q(y_{1:T}, x_{1:T}, u_{1:T}, θ)."""

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax
import optax

from ..objectives.full_joint_vfe import (
    full_joint_vfe,
    extract_first_action_marginal,
    extract_reward_location_marginal,
    extract_all_action_marginals,
    extract_all_state_marginals,
    extract_all_obs_marginals,
    enumerate_state_sequences,
)


@dataclass
class PlanningConfig:
    """Configuration for planning optimization."""
    planning_horizon: int = 4
    n_obs: int = 4
    n_states: int = 5
    n_actions: int = 4
    n_theta: int = 2  # renamed from n_reward_locs for clarity
    n_optimization_steps: int = 200
    learning_rate: float = 0.1
    verbose: bool = False
    inference_mode: str = "marginal"


@dataclass 
class PlanningResult:
    """Result of planning optimization."""
    q_yxu_theta: Array           # q(y, x, u, θ)
    first_action_probs: Array    # q(u_1)
    reward_location_probs: Array # q(θ)
    all_action_probs: Array      # q(u_t) for all t, shape (horizon, n_actions)
    all_state_probs: Array       # q(x_t) for all t, shape (horizon, n_states)
    all_obs_probs: Array         # q(y_t) for all t, shape (horizon, n_obs)
    final_loss: float
    loss_history: List[float]


def plan_actions(
    prior_state: Array,
    prior_reward_location: Array,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    config: PlanningConfig,
) -> PlanningResult:
    """
    Plan actions by minimizing VFE over q(y_{1:T}, x_{1:T}, u_{1:T}, θ).
    """
    n_obs_seqs = config.n_obs ** config.planning_horizon
    n_state_seqs = config.n_states ** config.planning_horizon
    n_action_seqs = config.n_actions ** config.planning_horizon
    
    # Initialize logits to zeros (uniform distribution)
    initial_logits = jnp.zeros((n_obs_seqs, n_state_seqs, n_action_seqs, config.n_theta))
    
    params = {'q_logits': initial_logits}
    
    # Uniform action prior
    action_prior = jnp.ones(config.n_actions) / config.n_actions
    
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    def loss_fn(params):
        return full_joint_vfe(
            q_logits=params['q_logits'],
            initial_state=prior_state,
            transition_tensor=transition_tensor,
            observation_tensor=observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=prior_reward_location,
            horizon=config.planning_horizon,
            inference_mode=config.inference_mode,
        )
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    loss_history = []
    for i in range(config.n_optimization_steps):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(float(loss))
        
        if config.verbose and (i + 1) % 50 == 0:
            print(f"Step {i+1}/{config.n_optimization_steps}, Loss: {loss:.4f}")
    
    # Extract results
    q_yxu_theta = softmax(params['q_logits'].flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, config.n_theta)
    )
    
    first_action_probs = extract_first_action_marginal(
        params['q_logits'], config.n_obs, config.n_states, config.n_actions, 
        config.n_theta, config.planning_horizon
    )
    reward_location_probs = extract_reward_location_marginal(
        params['q_logits'], config.n_obs, config.n_states, config.n_actions, 
        config.n_theta, config.planning_horizon
    )
    all_action_probs = extract_all_action_marginals(
        params['q_logits'], config.n_obs, config.n_states, config.n_actions, 
        config.n_theta, config.planning_horizon
    )
    all_state_probs = extract_all_state_marginals(
        params['q_logits'], config.n_obs, config.n_states, config.n_actions, 
        config.n_theta, config.planning_horizon
    )
    all_obs_probs = extract_all_obs_marginals(
        params['q_logits'], config.n_obs, config.n_states, config.n_actions, 
        config.n_theta, config.planning_horizon
    )
    
    return PlanningResult(
        q_yxu_theta=q_yxu_theta,
        first_action_probs=first_action_probs,
        reward_location_probs=reward_location_probs,
        all_action_probs=all_action_probs,
        all_state_probs=all_state_probs,
        all_obs_probs=all_obs_probs,
        final_loss=loss_history[-1],
        loss_history=loss_history,
    )


def select_action(result: PlanningResult) -> int:
    """Select action with highest marginal probability q(u_1)."""
    return int(jnp.argmax(result.first_action_probs))
