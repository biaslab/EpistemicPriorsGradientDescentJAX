"""
Temporal planning optimizer for Active Inference.

Uses gradient-based optimization to minimize temporal VFE and plan actions.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import optax

from ..objectives.temporal_vfe import (
    temporal_vfe_jit, group_modalities_for_jit, extract_marginals_temporal,
)
from ..environments.observation_modality import ObservationModality
from ..environments.environment_protocol import EnvironmentTensors


@dataclass
class TemporalPlanningConfig:
    """Configuration for temporal factorization planning."""
    planning_horizon: int = 7
    n_states: int = 32
    n_actions: int = 4
    n_theta: int = 2
    n_optimization_steps: int = 2000
    learning_rate: float = 0.01
    inference_mode: str = "marginal"  # "marginal", "active", or "planning"
    init_seed: int = 42
    gradient_scale_factor: float = 1.0
    freeze_obs_and_transitions: bool = False
    policy_init_scale: float = 1.0
    goal_scale: float = 1.0
    optimizer_type: str = "adam"  # "adam" or "adafactor"


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
    # Computed marginals for analysis
    q_x_theta: Array                        # (horizon+1, n_states, n_theta)
    q_u_theta: Array                        # (horizon, n_actions, n_theta)


def _make_optimizer(optimizer_type: str, learning_rate: float):
    """Create an optax optimizer by name."""
    if optimizer_type == "adam":
        return optax.adam(learning_rate=learning_rate)
    elif optimizer_type == "adafactor":
        return optax.adafactor(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


# static_argnums: horizon(10), n_opt_steps(11), inference_mode(12),
#                 freeze_obs_and_transitions(15),
#                 use_transition_index(17), optimizer_type(18)
# Dynamic scalar args: learning_rate(13), gradient_scale_factor(14), goal_scale(16)
@partial(jax.jit, static_argnums=(10, 11, 12, 15, 17, 18))
def _run_optimization(
    params, opt_state,
    initial_state, transition_tensor, transition_index,
    gen_tensor_groups, log_gen_tensor_groups,
    goal_mapping, action_prior, theta_prior,
    horizon, n_opt_steps, inference_mode,
    learning_rate,
    gradient_scale_factor,
    freeze_obs_and_transitions, goal_scale,
    use_transition_index,
    optimizer_type,
):
    """JIT-compiled optimization loop. Compiles once per unique set of static args."""
    optimizer = _make_optimizer(optimizer_type, learning_rate)

    # Dummy values for frozen params (not used in computation, just pytree placeholders)
    _dummy_x_logits = jnp.zeros(())
    _dummy_obs_groups = tuple(jnp.zeros(()) for _ in gen_tensor_groups)

    def loss_fn(params):
        if freeze_obs_and_transitions:
            q_x_logits = _dummy_x_logits
            q_obs_groups = _dummy_obs_groups
        else:
            q_x_logits = params['q_x_given_xu_theta_logits']
            q_obs_groups = params['q_obs_logits_groups']
        return temporal_vfe_jit(
            q_theta_logits=params['q_theta_logits'],
            q_u_given_x_logits=params['q_u_given_x_logits'],
            q_x_given_xu_theta_logits=q_x_logits,
            q_obs_logits_groups=q_obs_groups,
            initial_state=initial_state,
            transition_tensor=transition_tensor,
            gen_tensor_groups=gen_tensor_groups,
            log_gen_tensor_groups=log_gen_tensor_groups,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=theta_prior,
            transition_index=transition_index,
            horizon=horizon,
            inference_mode=inference_mode,
            goal_scale=goal_scale,
            freeze_obs_and_transitions=freeze_obs_and_transitions,
            use_transition_index=use_transition_index,
        )

    def step_fn(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # Scale policy gradients: earlier timesteps get larger multiplier
        policy_grads = grads['q_u_given_x_logits']
        scales = gradient_scale_factor ** jnp.arange(horizon - 1, -1, -1)
        policy_grads = policy_grads * scales[:, None, None]
        grads = {**grads, 'q_u_given_x_logits': policy_grads}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), all_losses = jax.lax.scan(
        step_fn, (params, opt_state), jnp.arange(n_opt_steps)
    )
    return params, all_losses


def plan_actions_temporal(
    initial_state: Array,                    # (n_states,) one-hot or distribution
    env_tensors: EnvironmentTensors,
    config: TemporalPlanningConfig,
    prior_theta_logits: Optional[Array] = None,
) -> TemporalPlanningResult:
    """
    Plan actions using temporal VFE minimization.

    Args:
        initial_state: Initial state distribution (n_states,)
        env_tensors: Environment tensors (transitions, observations, goals, priors)
        config: Planning configuration
        prior_theta_logits: Optional prior belief about θ (from previous timesteps)

    Returns:
        TemporalPlanningResult with optimized distributions
    """
    # Initialize parameters
    key = jax.random.PRNGKey(config.init_seed)

    # Determine if we use index-based transitions
    use_transition_index = env_tensors.transition_index is not None and config.freeze_obs_and_transitions
    transition_index_jax = env_tensors.transition_index if use_transition_index else jnp.zeros((1,), dtype=jnp.int32)

    # Pre-compute log values outside optimization loop (only needed for non-frozen path)
    if env_tensors.transition_tensor is not None:
        log_transition = jnp.log(jnp.clip(env_tensors.transition_tensor, 1e-8, 1.0))
    else:
        log_transition = None

    # Determine effective theta prior for VFE
    if prior_theta_logits is not None:
        q_theta_logits_init = prior_theta_logits
        effective_theta_prior = jax.nn.softmax(prior_theta_logits)
    else:
        q_theta_logits_init = jnp.log(env_tensors.theta_prior + 1e-8)
        effective_theta_prior = env_tensors.theta_prior

    # Split keys: 3 base + 1 per planning modality
    n_planning_modalities = len(env_tensors.planning_modalities)
    keys = jax.random.split(key, 3 + n_planning_modalities)

    # Initialize q(u_t | x_{t-1}) logits — θ-independent policy
    q_u_given_x_logits_init = jax.random.normal(
        keys[0],
        shape=(config.planning_horizon, config.n_states, config.n_actions)
    ) * config.policy_init_scale

    if config.freeze_obs_and_transitions:
        # Skip creating the massive transition/obs variational params.
        # Use generative tensors directly inside temporal_vfe_jit.

        # Still need dummy obs logits for group_modalities_for_jit to determine grouping
        q_obs_logits_init = []
        for mod in env_tensors.planning_modalities:
            q_obs_logits_init.append(jnp.zeros((1, mod.n_obs, config.n_states, config.n_theta)))

        _, gen_tensor_groups, log_gen_tensor_groups, \
            modality_index_groups = group_modalities_for_jit(
                q_obs_logits_init, env_tensors.planning_modalities)

        # Only theta and policy params — no 7.3M transition logits, no obs logits
        params = {
            'q_theta_logits': q_theta_logits_init,
            'q_u_given_x_logits': q_u_given_x_logits_init,
        }
    else:
        # Initialize q(x_t | x_{t-1}, u_t, θ) logits close to generative model
        # Target shape: (horizon, n_states, n_states, n_actions, n_theta)
        assert log_transition is not None, "Non-frozen path requires transition_tensor"
        if env_tensors.theta_dependent_transitions:
            # log_transition: (s, s, theta, a) -> transpose to (s, s, a, theta) for broadcasting
            log_trans_init = jnp.transpose(log_transition, (0, 1, 3, 2))  # (s, s, a, theta)
        else:
            # log_transition: (s, s, a) -> add theta dim
            log_trans_init = log_transition[:, :, :, None]  # (s, s, a, 1)
        q_x_given_xu_theta_logits_init = jax.random.normal(
            keys[1],
            shape=(config.planning_horizon, config.n_states, config.n_states, config.n_actions, config.n_theta)
        ) * 0.01 + log_trans_init

        # Initialize observation logits per modality (planning modalities only)
        q_obs_logits_init = []
        for i, mod in enumerate(env_tensors.planning_modalities):
            log_gen = jnp.log(jnp.clip(mod.generative_tensor, 1e-8, 1.0))
            # Shape: (horizon, n_obs, n_states, n_theta)
            noise = jax.random.normal(
                keys[3 + i],
                shape=(config.planning_horizon, mod.n_obs, config.n_states, config.n_theta)
            ) * 0.01
            init = noise + log_gen[None, :, :, :]
            q_obs_logits_init.append(init)

        # Pre-group modalities for JIT (once, outside JIT boundary)
        _, gen_tensor_groups, log_gen_tensor_groups, \
            modality_index_groups = group_modalities_for_jit(
                q_obs_logits_init, env_tensors.planning_modalities)

        # Stack initial logits by group (matching the grouped structure)
        q_obs_logits_groups_init = []
        for mod_indices in modality_index_groups:
            q_obs_logits_groups_init.append(
                jnp.stack([q_obs_logits_init[i] for i in mod_indices]))

        # Full parameter pytree
        params = {
            'q_theta_logits': q_theta_logits_init,
            'q_u_given_x_logits': q_u_given_x_logits_init,
            'q_x_given_xu_theta_logits': q_x_given_xu_theta_logits_init,
            'q_obs_logits_groups': q_obs_logits_groups_init,
        }

    if config.n_optimization_steps > 0:
        # Setup optimizer
        optimizer = _make_optimizer(config.optimizer_type, config.learning_rate)
        opt_state = optimizer.init(params)

        # Transition tensor: use dummy when index-based
        transition_tensor_jax = (
            jnp.zeros((1,)) if use_transition_index
            else env_tensors.transition_tensor
        )

        gen_tensor_groups_opt = gen_tensor_groups
        log_gen_tensor_groups_opt = log_gen_tensor_groups
        initial_state_opt = initial_state
        goal_mapping_opt = env_tensors.goal_mapping
        action_prior_opt = env_tensors.action_prior
        theta_prior_opt = effective_theta_prior

        # Run JIT-compiled optimization
        params, all_losses = _run_optimization(
            params, opt_state,
            initial_state_opt, transition_tensor_jax, transition_index_jax,
            gen_tensor_groups_opt, log_gen_tensor_groups_opt,
            goal_mapping_opt, action_prior_opt,
            theta_prior_opt,
            config.planning_horizon, config.n_optimization_steps, config.inference_mode,
            jnp.float32(config.learning_rate),
            jnp.float32(config.gradient_scale_factor),
            config.freeze_obs_and_transitions, jnp.float32(config.goal_scale),
            use_transition_index,
            config.optimizer_type,
        )

        # Subsample loss history (every 100 steps) from full trace
        loss_history = [float(all_losses[i]) for i in range(0, config.n_optimization_steps, 100)]
        final_loss = float(all_losses[-1])
    else:
        # No optimization: evaluate loss at initial params
        from ..objectives.temporal_vfe import temporal_vfe_jit as _vfe_jit
        _q_x_logits = params.get('q_x_given_xu_theta_logits', jnp.zeros(()))
        _q_obs_groups = params.get('q_obs_logits_groups',
                                    tuple(jnp.zeros(()) for _ in gen_tensor_groups))
        transition_tensor_jax = (
            jnp.zeros((1,)) if use_transition_index
            else env_tensors.transition_tensor
        )
        final_loss = float(_vfe_jit(
            q_theta_logits=params['q_theta_logits'],
            q_u_given_x_logits=params['q_u_given_x_logits'],
            q_x_given_xu_theta_logits=_q_x_logits,
            q_obs_logits_groups=_q_obs_groups,
            initial_state=initial_state,
            transition_tensor=transition_tensor_jax,
            gen_tensor_groups=gen_tensor_groups,
            log_gen_tensor_groups=log_gen_tensor_groups,
            goal_mapping=env_tensors.goal_mapping,
            action_prior=env_tensors.action_prior,
            theta_prior=effective_theta_prior,
            transition_index=transition_index_jax,
            horizon=config.planning_horizon,
            inference_mode=config.inference_mode,
            goal_scale=config.goal_scale,
            freeze_obs_and_transitions=config.freeze_obs_and_transitions,
            use_transition_index=use_transition_index,
        ))
        loss_history = [final_loss]

    planning_mods = env_tensors.planning_modalities
    n_planning_mods = len(planning_mods)

    if config.freeze_obs_and_transitions:
        if use_transition_index:
            # Index path: no dense transition logits needed for marginals
            q_x_given_xu_theta_logits_final = jnp.zeros(())  # dummy
        else:
            # Build transition from generative model for marginal extraction.
            # NO broadcast over horizon — pass raw tensors and use constant_transitions=True.
            transition_tensor = env_tensors.transition_tensor
            if transition_tensor.ndim == 4:
                q_x_from_gen = jnp.transpose(transition_tensor, (0, 1, 3, 2))
            else:
                q_x_from_gen = transition_tensor[..., None]
            q_x_given_xu_theta_logits_final = jnp.log(
                jnp.clip(q_x_from_gen, 1e-8, 1.0)
            )  # (n_states, n_states, n_actions, n_theta) — no horizon broadcast

        q_obs_logits_list_final = []
        for mod in planning_mods:
            log_gen = jnp.log(jnp.clip(mod.generative_tensor, 1e-8, 1.0))
            q_obs_logits_list_final.append(log_gen)  # no horizon broadcast
    else:
        q_x_given_xu_theta_logits_final = params['q_x_given_xu_theta_logits']
        # Ungroup obs logits back to per-modality list
        q_obs_logits_list_final = [None] * n_planning_mods
        for group_idx, mod_indices in enumerate(modality_index_groups):
            group_logits = params['q_obs_logits_groups'][group_idx]
            for j, orig_idx in enumerate(mod_indices):
                q_obs_logits_list_final[orig_idx] = group_logits[j]

    # Extract marginals
    marginals = extract_marginals_temporal(
        q_theta_logits=params['q_theta_logits'],
        q_u_given_x_logits=params['q_u_given_x_logits'],
        q_x_given_xu_theta_logits=q_x_given_xu_theta_logits_final,
        q_obs_logits_list=q_obs_logits_list_final,
        observation_modalities=planning_mods,
        initial_state=initial_state,
        horizon=config.planning_horizon,
        constant_transitions=config.freeze_obs_and_transitions,
        transition_index=transition_index_jax if use_transition_index else None,
        use_transition_index=use_transition_index,
    )

    # Compute first action distribution — policy is θ-independent
    policy_t0 = marginals['q_u_given_x'][0]  # (n_states, n_actions)

    q_first_action = jnp.sum(
        policy_t0 * initial_state[:, None],
        axis=0,
    )  # (n_actions,)

    # Normalize
    q_first_action = q_first_action / jnp.sum(q_first_action)

    # Collect observation marginals (planning modalities only)
    q_obs_result = []
    obs_names = []
    for mod in planning_mods:
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
        # Policy is θ-independent: q(u_t | x_{t-1}) directly
        policy_t = result.q_u_given_x[timestep]  # (n_states, n_actions)
        q_action = policy_t[current_state_idx]  # (n_actions,)
        return int(jnp.argmax(q_action))


def get_belief_summary(result: TemporalPlanningResult) -> dict:
    """
    Extract human-readable summary of belief state.

    Returns:
        Dictionary with key beliefs (θ distribution, policy summary, etc.)
    """
    return {
        'theta_belief': {
            f'theta_{i}': float(result.q_theta[i])
            for i in range(len(result.q_theta))
        },
        'first_action_probs': {
            f'action_{i}': float(result.q_first_action[i])
            for i in range(len(result.q_first_action))
        },
        'final_loss': result.final_loss,
        'converged': len(result.loss_history) > 1 and abs(result.loss_history[-1] - result.loss_history[-2]) < 0.01,
    }
