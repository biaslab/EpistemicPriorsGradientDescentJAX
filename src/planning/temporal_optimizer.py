"""
Temporal planning optimizer for Active Inference.

Uses Adam optimization to minimize temporal VFE and plan actions.
Uses the unified temporal_vfe.py with θ-independent policy q(u|x).
"""

from dataclasses import dataclass
from typing import List, Optional
import jax
import jax.numpy as jnp
from jax import Array
import optax

from ..objectives.temporal_vfe import (
    temporal_vfe_jit, group_modalities_for_jit, extract_marginals_temporal,
)
from ..environments.observation_modality import ObservationModality


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
    q_u_given_x: Array                     # (horizon, n_states, n_actions)
    q_x_given_xu_theta: Array              # (horizon, n_states, n_states, n_actions, n_theta)
    q_obs: List[Array]                      # one per modality (after softmax)
    observation_modality_names: List[str]   # for identification
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
    Plan actions using temporal VFE minimization with θ-independent policy.

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

    # Pre-compute log values outside optimization loop
    log_transition = jnp.log(jnp.clip(transition_tensor, 1e-8, 1.0))
    log_observation = jnp.log(jnp.clip(theta_observation_tensor, 1e-8, 1.0))

    # Wrap observation tensors as ObservationModality instances
    observation_modalities = [
        ObservationModality(
            name="theta",
            generative_tensor=theta_observation_tensor,
            theta_dependent=True,
            n_obs=theta_observation_tensor.shape[0],
        ),
    ]
    if location_observation_tensor is not None:
        # Location modality needs theta dimension for group_modalities_for_jit
        # Expand (n_locations, n_states) -> (n_locations, n_states, n_theta) by broadcasting
        n_locations = location_observation_tensor.shape[0]
        loc_gen_expanded = jnp.broadcast_to(
            location_observation_tensor[:, :, None],
            (n_locations, config.n_states, config.n_theta),
        )
        observation_modalities.append(
            ObservationModality(
                name="location",
                generative_tensor=loc_gen_expanded,
                theta_dependent=True,  # treat as theta-dependent for unified handling
                n_obs=n_locations,
            ),
        )

    # Determine effective theta prior for VFE
    if prior_theta_logits is not None:
        q_theta_logits_init = prior_theta_logits
        effective_theta_prior = jax.nn.softmax(prior_theta_logits)
    else:
        q_theta_logits_init = jnp.log(theta_prior + 1e-8)
        effective_theta_prior = theta_prior

    # Vectorized initialization
    n_mods = len(observation_modalities)
    keys = jax.random.split(key, 3 + n_mods)

    # Initialize q(u_t | x_{t-1}) logits — θ-independent policy
    q_u_given_x_logits_init = jax.random.normal(
        keys[0],
        shape=(config.planning_horizon, config.n_states, config.n_actions)
    ) * 0.01

    # Initialize q(x_t | x_{t-1}, u_t, θ) logits close to generative model
    q_x_given_xu_theta_logits_init = jax.random.normal(
        keys[1],
        shape=(config.planning_horizon, config.n_states, config.n_states, config.n_actions, config.n_theta)
    ) * 0.01 + log_transition[:, :, :, None]

    # Initialize observation logits per modality
    q_obs_logits_init = []
    for i, mod in enumerate(observation_modalities):
        log_gen = jnp.log(jnp.clip(mod.generative_tensor, 1e-8, 1.0))
        noise = jax.random.normal(
            keys[3 + i],
            shape=(config.planning_horizon, mod.n_obs, config.n_states, config.n_theta)
        ) * 0.01
        q_obs_logits_init.append(noise + log_gen[None, :, :, :])

    # Pre-group modalities for JIT
    _, gen_tensor_groups, log_gen_tensor_groups, \
        modality_index_groups = group_modalities_for_jit(
            q_obs_logits_init, observation_modalities)

    # Stack initial logits by group
    q_obs_logits_groups_init = []
    for mod_indices in modality_index_groups:
        q_obs_logits_groups_init.append(
            jnp.stack([q_obs_logits_init[i] for i in mod_indices]))

    # Create parameter pytree
    params = {
        'q_theta_logits': q_theta_logits_init,
        'q_u_given_x_logits': q_u_given_x_logits_init,
        'q_x_given_xu_theta_logits': q_x_given_xu_theta_logits_init,
        'q_obs_logits_groups': q_obs_logits_groups_init,
    }

    # Define loss function
    def loss_fn(params):
        return temporal_vfe_jit(
            q_theta_logits=params['q_theta_logits'],
            q_u_given_x_logits=params['q_u_given_x_logits'],
            q_x_given_xu_theta_logits=params['q_x_given_xu_theta_logits'],
            q_obs_logits_groups=params['q_obs_logits_groups'],
            initial_state=initial_state,
            transition_tensor=transition_tensor,
            gen_tensor_groups=gen_tensor_groups,
            log_gen_tensor_groups=log_gen_tensor_groups,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=effective_theta_prior,
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

    # Run optimization using jax.lax.scan
    initial_carry = (params, opt_state, loss_history_array, 0)
    (params, opt_state, loss_history_array, _), all_losses = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(config.n_optimization_steps)
    )

    # Convert loss history
    loss_history = [float(x) for x in loss_history_array if x != 0.0]
    all_losses_list = [float(x) for x in all_losses]
    final_loss = float(all_losses[-1])

    # Ungroup obs logits back to per-modality list
    q_obs_logits_list_final = [None] * n_mods
    for group_idx, mod_indices in enumerate(modality_index_groups):
        group_logits = params['q_obs_logits_groups'][group_idx]
        for j, orig_idx in enumerate(mod_indices):
            q_obs_logits_list_final[orig_idx] = group_logits[j]

    # Extract marginals
    marginals = extract_marginals_temporal(
        q_theta_logits=params['q_theta_logits'],
        q_u_given_x_logits=params['q_u_given_x_logits'],
        q_x_given_xu_theta_logits=params['q_x_given_xu_theta_logits'],
        q_obs_logits_list=q_obs_logits_list_final,
        observation_modalities=observation_modalities,
        initial_state=initial_state,
        horizon=config.planning_horizon,
    )

    # Compute first action distribution — θ-independent policy
    policy_t0 = marginals['q_u_given_x'][0]  # (n_states, n_actions)
    q_first_action = jnp.sum(
        policy_t0 * initial_state[:, None],
        axis=0,
    )  # (n_actions,)
    q_first_action = q_first_action / jnp.sum(q_first_action)

    # Collect observation marginals
    q_obs_result = []
    obs_names = []
    for mod in observation_modalities:
        q_obs_result.append(marginals[f'q_obs_{mod.name}'])
        obs_names.append(mod.name)

    return TemporalPlanningResult(
        q_theta=marginals['q_theta'],
        q_u_given_x=marginals['q_u_given_x'],
        q_x_given_xu_theta=marginals['q_x_given_xu_theta'],
        q_obs=q_obs_result,
        observation_modality_names=obs_names,
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
        current_state_idx: Current state index
        timestep: Current timestep (0-based, default 0 for first action)

    Returns:
        Selected action index (argmax)
    """
    if timestep == 0:
        return int(jnp.argmax(result.q_first_action))
    else:
        # θ-independent policy: q(u_t | x_{t-1})
        policy_t = result.q_u_given_x[timestep]  # (n_states, n_actions)
        q_action = policy_t[current_state_idx]  # (n_actions,)
        return int(jnp.argmax(q_action))


def get_belief_summary(result: TemporalPlanningResult) -> dict:
    """
    Extract human-readable summary of belief state.
    """
    return {
        'theta_belief': {
            'left': float(result.q_theta[0]),
            'right': float(result.q_theta[1]),
        },
        'first_action_probs': {
            f'action_{i}': float(result.q_first_action[i])
            for i in range(len(result.q_first_action))
        },
        'final_loss': result.final_loss,
        'converged': len(result.loss_history) > 1 and abs(result.loss_history[-1] - result.loss_history[-2]) < 0.01,
    }
