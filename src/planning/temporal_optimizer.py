"""
Temporal planning optimizer for Active Inference.

Uses Adam optimization to minimize temporal VFE and plan actions.
"""

from dataclasses import dataclass
from typing import List, Optional
import jax
import jax.numpy as jnp
from jax import Array
import optax

from ..objectives.temporal_vfe_epistemic import temporal_vfe, extract_marginals_temporal


@dataclass
class TemporalPlanningConfig:
    """Configuration for temporal factorization planning."""
    planning_horizon: int = 7
    n_states: int = 32
    n_actions: int = 4
    n_theta: int = 2
    n_obs: int = 3
    n_optimization_steps: int = 2000
    learning_rate: float = 0.01
    inference_mode: str = "marginal"  # "marginal", "active", or "planning"
    init_seed: int = 42


@dataclass
class TemporalPlanningResult:
    """Result of temporal planning with full variational distributions."""
    q_theta: Array                          # (n_theta,)
    q_u_given_x_theta: Array               # (horizon, n_states, n_theta, n_actions)
    q_x_given_xu_theta: Array              # (horizon, n_states, n_states, n_actions, n_theta)
    q_y_theta_given_x_theta: Array         # (horizon, n_obs_theta, n_states, n_theta)
    q_y_location_given_x: Array            # (horizon, n_locations, n_states)
    q_first_action: Array                   # (n_actions,) marginalized for t=1
    final_loss: float
    loss_history: List[float]
    all_losses: List[float]                  # Loss at every optimization step
    # Computed marginals for analysis
    q_x_theta: Array                        # (horizon+1, n_states, n_theta)
    q_u_theta: Array                        # (horizon, n_actions, n_theta)


def plan_actions_temporal(
    initial_state: Array,                    # (n_states,) one-hot or distribution
    transition_tensor: Array,                # (n_states, n_states, n_actions)
    theta_observation_tensor: Array,         # (n_obs_theta, n_states, n_theta)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    config: TemporalPlanningConfig,
    prior_theta_logits: Optional[Array] = None,  # Optional prior belief about θ
    location_observation_tensor: Array = None,  # (n_locations, n_states)
) -> TemporalPlanningResult:
    """
    Plan actions using temporal VFE minimization.

    Args:
        initial_state: Initial state distribution (n_states,)
        transition_tensor: p(s'|s,a) (n_states, n_states, n_actions)
        theta_observation_tensor: p(o|s,θ) (n_obs_theta, n_states, n_theta)
        goal_mapping: p(goal|s,θ) (n_states, n_theta)
        action_prior: p(a) (n_actions,)
        theta_prior: p(θ) (n_theta,)
        config: Planning configuration
        prior_theta_logits: Optional prior belief about θ (from previous timesteps)
        location_observation_tensor: p(location|s) (n_locations, n_states) - required

    Returns:
        TemporalPlanningResult with optimized distributions
    """
    # Initialize parameters
    key = jax.random.PRNGKey(config.init_seed)
    
    # Pre-compute log values outside optimization loop (avoids recomputation)
    log_transition = jnp.log(jnp.clip(transition_tensor, 1e-8, 1.0))
    log_observation = jnp.log(jnp.clip(theta_observation_tensor, 1e-8, 1.0))

    # Determine effective theta prior for VFE
    # If prior_theta_logits is provided, use it instead of uniform theta_prior
    if prior_theta_logits is not None:
        q_theta_logits_init = prior_theta_logits
        # Convert logits to probability for use in VFE
        effective_theta_prior = jax.nn.softmax(prior_theta_logits)
    else:
        # Initialize to log prior
        q_theta_logits_init = jnp.log(theta_prior + 1e-8)
        effective_theta_prior = theta_prior

    # Vectorized initialization: split keys and generate all random tensors at once
    keys = jax.random.split(key, 4)
    
    # Initialize q(u_t | x_{t-1}, θ) logits with small random noise
    # Shape: (horizon, n_states, n_theta, n_actions)
    q_u_given_x_theta_logits_init = jax.random.normal(
        keys[0],
        shape=(config.planning_horizon, config.n_states, config.n_theta, config.n_actions)
    ) * 0.01  # Small noise around zero (near-uniform policy)

    # Initialize q(x_t | x_{t-1}, u_t, θ) logits with small random noise
    # Shape: (horizon, n_states_next, n_states_prev, n_actions, n_theta)
    # Initialize close to generative model by using log of transition_tensor + noise
    q_x_given_xu_theta_logits_init = jax.random.normal(
        keys[1],
        shape=(config.planning_horizon, config.n_states, config.n_states, config.n_actions, config.n_theta)
    ) * 0.01 + log_transition[:, :, :, None]

    # Initialize q(y_theta_t | x_t, θ) logits with small random noise
    # Shape: (horizon, n_obs_theta, n_states, n_theta)
    # Initialize close to generative model by using log of observation_tensor + noise
    q_y_theta_given_x_theta_logits_init = jax.random.normal(
        keys[2],
        shape=(config.planning_horizon, config.n_obs, config.n_states, config.n_theta)
    ) * 0.01 + log_observation[None, :, :, :]

    # Initialize q(y_location_t | x_t) logits with small random noise
    # Shape: (horizon, n_locations, n_states)
    # Infer n_locations from tensor shape
    n_locations = location_observation_tensor.shape[0]
    log_location_obs = jnp.log(jnp.clip(location_observation_tensor, 1e-8, 1.0))
    q_y_location_given_x_logits_init = jax.random.normal(
        keys[3],
        shape=(config.planning_horizon, n_locations, config.n_states)
    ) * 0.01 + log_location_obs[None, :, :]

    # Create parameter pytree
    params = {
        'q_theta_logits': q_theta_logits_init,
        'q_u_given_x_theta_logits': q_u_given_x_theta_logits_init,
        'q_x_given_xu_theta_logits': q_x_given_xu_theta_logits_init,
        'q_y_theta_given_x_theta_logits': q_y_theta_given_x_theta_logits_init,
        'q_y_location_given_x_logits': q_y_location_given_x_logits_init,
    }

    # Define loss function
    def loss_fn(params):
        return temporal_vfe(
            q_theta_logits=params['q_theta_logits'],
            q_u_given_x_theta_logits=params['q_u_given_x_theta_logits'],
            q_x_given_xu_theta_logits=params['q_x_given_xu_theta_logits'],
            q_y_theta_given_x_theta_logits=params['q_y_theta_given_x_theta_logits'],
            q_y_location_given_x_logits=params['q_y_location_given_x_logits'],
            initial_state=initial_state,
            transition_tensor=transition_tensor,
            theta_observation_tensor=theta_observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=effective_theta_prior,  # Use actual prior, not uniform
            horizon=config.planning_horizon,
            inference_mode=config.inference_mode,
        )

    # Setup optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(params)

    # Pre-allocate loss history array (stores every 100 steps)
    num_samples = (config.n_optimization_steps - 1) // 100 + 1
    loss_history_array = jnp.zeros(num_samples)

    # Define scan-compatible optimization step
    def step_fn(carry, step_idx):
        params, opt_state, loss_hist, hist_idx = carry
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Record loss every 100 steps
        should_record = (step_idx % 100) == 0
        new_hist_idx = hist_idx + jnp.where(should_record, 1, 0)
        loss_hist = jax.lax.cond(
            should_record,
            lambda: loss_hist.at[hist_idx].set(loss),
            lambda: loss_hist
        )
        
        return (params, opt_state, loss_hist, new_hist_idx), loss

    # Run optimization using jax.lax.scan (keeps entire loop compiled)
    initial_carry = (params, opt_state, loss_history_array, 0)
    (params, opt_state, loss_history_array, _), all_losses = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(config.n_optimization_steps)
    )

    # Convert loss history to list, filter out zeros (unused slots)
    loss_history = [float(x) for x in loss_history_array if x != 0.0]

    # Convert all losses from scan to list
    all_losses_list = [float(x) for x in all_losses]

    # Extract final loss from all_losses
    final_loss = float(all_losses[-1])

    # Extract final parameters
    q_theta_logits_final = params['q_theta_logits']
    q_u_given_x_theta_logits_final = params['q_u_given_x_theta_logits']
    q_x_given_xu_theta_logits_final = params['q_x_given_xu_theta_logits']
    q_y_theta_given_x_theta_logits_final = params['q_y_theta_given_x_theta_logits']
    q_y_location_given_x_logits_final = params['q_y_location_given_x_logits']

    # Extract marginals
    marginals = extract_marginals_temporal(
        q_theta_logits=q_theta_logits_final,
        q_u_given_x_theta_logits=q_u_given_x_theta_logits_final,
        q_x_given_xu_theta_logits=q_x_given_xu_theta_logits_final,
        q_y_theta_given_x_theta_logits=q_y_theta_given_x_theta_logits_final,
        q_y_location_given_x_logits=q_y_location_given_x_logits_final,
        initial_state=initial_state,
        horizon=config.planning_horizon,
    )

    # Compute first action distribution
    # q(u_1) marginalized over current state and θ
    # We have q(u_1 | x_0, θ), q(θ), and x_0 (initial_state)
    # q(u_1) = ∑_θ q(u_1 | x_0, θ) p(x_0) q(θ)
    #        = ∑_{x_0, θ} q(u_1 | x_0, θ) p(x_0) q(θ)

    policy_t0 = marginals['q_u_given_x_theta'][0]  # (n_states, n_theta, n_actions)
    q_theta = marginals['q_theta']  # (n_theta,)

    # Weight by initial state and θ
    # Broadcasting: (n_states, n_theta, n_actions) * (n_states, 1, 1) * (1, n_theta, 1)
    q_first_action = jnp.sum(
        policy_t0 * initial_state[:, None, None] * q_theta[None, :, None],
        axis=(0, 1)  # Sum over states and θ
    )  # (n_actions,)

    # Normalize
    q_first_action = q_first_action / jnp.sum(q_first_action)

    return TemporalPlanningResult(
        q_theta=marginals['q_theta'],
        q_u_given_x_theta=marginals['q_u_given_x_theta'],
        q_x_given_xu_theta=marginals['q_x_given_xu_theta'],
        q_y_theta_given_x_theta=marginals['q_y_theta_given_x_theta'],
        q_y_location_given_x=marginals['q_y_location_given_x'],
        q_first_action=q_first_action,
        final_loss=final_loss,
        loss_history=loss_history,
        all_losses=all_losses_list,
        q_x_theta=marginals['q_x_theta'],
        q_u_theta=marginals['q_u_theta'],
    )


def select_action_temporal(
    result: TemporalPlanningResult,
    current_state_idx: int,
    timestep: int = 0,
) -> int:
    """
    Select action based on planning result and current state.

    Args:
        result: Planning result from plan_actions_temporal()
        current_state_idx: Current state index (0-31)
        timestep: Current timestep (0-based, default 0 for first action)

    Returns:
        Selected action index (argmax)
    """
    if timestep == 0:
        # Use pre-computed first action distribution
        return int(jnp.argmax(result.q_first_action))
    else:
        # For future timesteps, marginalize policy over θ
        # q(u_t | x_{t-1} = current_state) = ∑_θ q(u_t | x_{t-1}, θ) q(θ)
        policy_t = result.q_u_given_x_theta[timestep]  # (n_states, n_theta, n_actions)
        q_theta = result.q_theta  # (n_theta,)

        # Get policy for current state
        policy_at_state = policy_t[current_state_idx]  # (n_theta, n_actions)

        # Marginalize over θ
        q_action = jnp.sum(policy_at_state * q_theta[:, None], axis=0)  # (n_actions,)

        return int(jnp.argmax(q_action))


def get_belief_summary(result: TemporalPlanningResult) -> dict:
    """
    Extract human-readable summary of belief state.

    Returns:
        Dictionary with key beliefs (θ distribution, policy summary, etc.)
    """
    return {
        'theta_belief': {
            'left': float(result.q_theta[0]),
            'right': float(result.q_theta[1]),
        },
        'first_action_probs': {
            'learn': float(result.q_first_action[0]),
            'navigate': float(result.q_first_action[1]),
            'increase_knob': float(result.q_first_action[2]),
            'decrease_knob': float(result.q_first_action[3]),
        },
        'final_loss': result.final_loss,
        'converged': len(result.loss_history) > 1 and abs(result.loss_history[-1] - result.loss_history[-2]) < 0.01,
    }
