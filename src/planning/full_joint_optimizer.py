"""Planning via VFE minimization with full joint q(x_{1:T}, u_{1:T}, r)."""

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
    enumerate_state_sequences,
)


@dataclass
class PlanningConfig:
    """Configuration for planning optimization."""
    planning_horizon: int = 4
    n_states: int = 5
    n_actions: int = 4
    n_reward_locs: int = 2
    n_optimization_steps: int = 200
    learning_rate: float = 0.1
    verbose: bool = False
    marginal_inference: bool = False


@dataclass 
class PlanningResult:
    """Result of planning optimization."""
    q_xur: Array                 # q(x, u, r)
    first_action_probs: Array    # q(u_1)
    reward_location_probs: Array # q(r)
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
    Plan actions by minimizing VFE over q(x_{1:T}, u_{1:T}, r).
    """
    n_state_seqs = config.n_states ** config.planning_horizon
    n_action_seqs = config.n_actions ** config.planning_horizon
    
    # Initialize logits with reward prior
    log_prior_r = jnp.log(prior_reward_location + 1e-10)
    initial_logits = jnp.zeros((n_state_seqs, n_action_seqs, config.n_reward_locs))
    initial_logits = initial_logits + log_prior_r[None, None, :]
    
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
            horizon=config.planning_horizon,
            marginal_inference=config.marginal_inference,
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
    q_xur = softmax(params['q_logits'].flatten()).reshape(
        (n_state_seqs, n_action_seqs, config.n_reward_locs)
    )
    
    first_action_probs = extract_first_action_marginal(
        params['q_logits'], config.n_states, config.n_actions, config.planning_horizon
    )
    reward_location_probs = extract_reward_location_marginal(
        params['q_logits'], config.n_states, config.n_actions, config.planning_horizon
    )
    
    return PlanningResult(
        q_xur=q_xur,
        first_action_probs=first_action_probs,
        reward_location_probs=reward_location_probs,
        final_loss=loss_history[-1],
        loss_history=loss_history,
    )


def select_action(result: PlanningResult) -> int:
    """Select action with highest marginal probability q(u_1)."""
    return int(jnp.argmax(result.first_action_probs))
