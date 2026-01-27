"""Planning via VFE minimization with factorized q(x|u) q(y,θ|x) q(u)."""

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax
import optax

from ..objectives.factorized_vfe import (
    factorized_vfe,
    extract_marginals_from_factorized,
    reconstruct_full_joint,
    enumerate_action_sequences,
    enumerate_state_sequences,
)


@dataclass
class FactorizedPlanningConfig:
    """Configuration for factorized planning optimization."""
    planning_horizon: int = 4
    n_obs: int = 4
    n_states: int = 5
    n_actions: int = 4
    n_theta: int = 2
    n_optimization_steps: int = 200
    learning_rate: float = 0.1
    verbose: bool = False
    inference_mode: str = "marginal"


@dataclass 
class FactorizedPlanningResult:
    """Result of factorized planning optimization."""
    q_yxu_theta: Array           # Reconstructed q(y, x, u, θ)
    first_action_probs: Array    # q(u_1)
    reward_location_probs: Array # q(θ)
    all_action_probs: Array      # q(u_t) for all t, shape (horizon, n_actions)
    all_state_probs: Array       # q(x_t) for all t, shape (horizon, n_states)
    all_obs_probs: Array         # q(y_t) for all t, shape (horizon, n_obs)
    final_loss: float
    loss_history: List[float]
    # Factorized parameters (for inspection)
    q_u_logits: Array
    q_x_given_u_logits: Array
    q_y_theta_given_x_logits: Array


def plan_actions_factorized(
    prior_state: Array,
    prior_reward_location: Array,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    config: FactorizedPlanningConfig,
) -> FactorizedPlanningResult:
    """
    Plan actions by minimizing VFE over factorized q(x|u) q(y,θ|x) q(u).
    
    Starts from fully uniform distributions for all factors.
    """
    n_obs_seqs = config.n_obs ** config.planning_horizon
    n_state_seqs = config.n_states ** config.planning_horizon
    n_action_seqs = config.n_actions ** config.planning_horizon
    
    # Initialize q(u) to uniform
    q_u_logits = jnp.zeros(n_action_seqs)
    
    # Initialize q(x|u) to match transition dynamics p(x|u, x_0)
    # This breaks the softmax gradient vanishing problem by making q(x) depend on q(u)
    from ..objectives.factorized_vfe import compute_transition_log_probs
    state_sequences = enumerate_state_sequences(config.n_states, config.planning_horizon)
    action_sequences = enumerate_action_sequences(config.n_actions, config.planning_horizon)
    initial_state_idx = jnp.argmax(prior_state)
    
    log_transition_probs = compute_transition_log_probs(
        initial_state_idx, state_sequences, action_sequences, transition_tensor
    )  # (n_state_seqs, n_action_seqs)
    
    # Use log transition probs as initial logits for q(x|u)
    # This makes q(x|u) ≈ p(x|u, x_0) initially
    q_x_given_u_logits = log_transition_probs
    
    # Initialize q(y,θ|x) to uniform
    q_y_theta_given_x_logits = jnp.zeros((n_obs_seqs, config.n_theta, n_state_seqs))
    
    params = {
        'q_u_logits': q_u_logits,
        'q_x_given_u_logits': q_x_given_u_logits,
        'q_y_theta_given_x_logits': q_y_theta_given_x_logits,
    }
    
    # Uniform action prior
    action_prior = jnp.ones(config.n_actions) / config.n_actions
    
    # Use standard optimizer - the gradient scaling issue needs to be fixed
    # in the VFE computation, not in the optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    def loss_fn(params):
        return factorized_vfe(
            q_u_logits=params['q_u_logits'],
            q_x_given_u_logits=params['q_x_given_u_logits'],
            q_y_theta_given_x_logits=params['q_y_theta_given_x_logits'],
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
    marginals = extract_marginals_from_factorized(
        params['q_u_logits'],
        params['q_x_given_u_logits'],
        params['q_y_theta_given_x_logits'],
        config.n_obs,
        config.n_states,
        config.n_actions,
        config.n_theta,
        config.planning_horizon,
    )
    
    q_yxu_theta = reconstruct_full_joint(
        params['q_u_logits'],
        params['q_x_given_u_logits'],
        params['q_y_theta_given_x_logits'],
        config.n_obs,
        config.n_states,
        config.n_actions,
        config.n_theta,
        config.planning_horizon,
    )
    
    return FactorizedPlanningResult(
        q_yxu_theta=q_yxu_theta,
        first_action_probs=marginals['q_first_action'],
        reward_location_probs=marginals['q_theta'],
        all_action_probs=marginals['q_all_actions'],
        all_state_probs=marginals['q_all_states'],
        all_obs_probs=marginals['q_all_obs'],
        final_loss=loss_history[-1],
        loss_history=loss_history,
        q_u_logits=params['q_u_logits'],
        q_x_given_u_logits=params['q_x_given_u_logits'],
        q_y_theta_given_x_logits=params['q_y_theta_given_x_logits'],
    )


def select_action_factorized(result: FactorizedPlanningResult) -> int:
    """Select action with highest marginal probability q(u_1)."""
    return int(jnp.argmax(result.first_action_probs))
