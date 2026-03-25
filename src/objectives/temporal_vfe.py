"""
Temporal factorization of variational free energy for Active Inference planning.

This module implements VFE minimization with full temporal (Markovian) factorization:
    q(x_{0:T}, u_{1:T}, y_{1:T}, θ) = q(θ) · ∏_{t=1}^T q(u_t|x_{t-1}) · q(x_t|x_{t-1},u_t,θ) · q(y_t|x_t,θ)

The policy q(u_t|x_{t-1}) is independent of θ (factorization: q(u,θ) = q(u)q(θ)).
All four factors are variational (optimized):
1. q(θ) - parameter belief
2. q(u_t|x_{t-1}) - policy (θ-independent)
3. q(x_t|x_{t-1},u_t,θ) - variational transitions
4. q(y_t|x_t,θ) - variational observations

VFE Decomposition:
    VFE = -H[q] + E_q[-log p(...)]

Where:
    H[q] = H[q(θ)]
         + Σ_t E_{x_{t-1}}[H[q(u_t|x_{t-1})]]               # Policy entropy
         + Σ_t E_{x_{t-1},u_t,θ}[H[q(x_t|x_{t-1},u_t,θ)]]   # Transition entropy
         + Σ_t E_{x_t,θ}[H[q(y_t|x_t,θ)]]                   # Observation entropy

Key advantages over sequence-based factorization:
- Linear scaling with horizon (vs exponential)
- Natural reactive policy representation
- Full variational inference over all factors
"""

from typing import Tuple, Dict, List, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax

from ..environments.observation_modality import ObservationModality

# Numerical stability constant
EPS = 1e-8


class ModalityGroup(NamedTuple):
    """A batch of observation modalities sharing the same n_obs signature."""
    n_obs: int
    q_obs_batch: Array     # (n_mods, horizon, n_obs, n_states, n_theta)
    gen_batch: Array       # (n_mods, n_obs, n_states, n_theta)
    log_gen_batch: Array   # (n_mods, n_obs, n_states, n_theta)


def group_modalities(
    q_obs_list: List[Array],
    observation_modalities: List[ObservationModality],
) -> List[ModalityGroup]:
    """Group observation modalities by n_obs and stack into batch tensors.

    Collapses e.g. 49 same-shape FOV modalities into a single batched group,
    reducing Python loop iterations from O(n_modalities) to O(n_groups).
    """
    buckets: Dict[int, Dict[str, list]] = {}
    for q_obs, mod in zip(q_obs_list, observation_modalities):
        key = mod.n_obs
        if key not in buckets:
            buckets[key] = {'q_obs': [], 'gen': []}
        buckets[key]['q_obs'].append(q_obs)
        buckets[key]['gen'].append(mod.generative_tensor)

    groups = []
    for n_obs, data in buckets.items():
        q_obs_batch = jnp.stack(data['q_obs'])  # (M, horizon, n_obs, n_states, n_theta)
        gen_batch = jnp.stack(data['gen'])       # (M, n_obs, n_states, n_theta)
        log_gen_batch = jnp.log(jnp.clip(gen_batch, EPS, 1.0))
        groups.append(ModalityGroup(
            n_obs=n_obs,
            q_obs_batch=q_obs_batch,
            gen_batch=gen_batch,
            log_gen_batch=log_gen_batch,
        ))
    return groups


def group_modalities_for_jit(
    q_obs_logits_list: List[Array],
    observation_modalities: List[ObservationModality],
) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...], Tuple[Tuple[int, ...], ...]]:
    """Pre-group modality data for JIT-compatible computation.

    Call once outside the JIT boundary. Returns grouped arrays and static config
    that can be passed to temporal_vfe_jit.

    Returns:
        q_obs_logits_groups: tuple of stacked logit arrays per group
        gen_tensor_groups: tuple of stacked gen tensor arrays per group
        log_gen_tensor_groups: tuple of stacked log gen tensor arrays per group
        modality_index_groups: tuple of index tuples (for ungrouping results)
    """
    buckets: Dict[int, Dict[str, list]] = {}
    for i, (q_obs_logits, mod) in enumerate(zip(q_obs_logits_list, observation_modalities)):
        key = mod.n_obs
        if key not in buckets:
            buckets[key] = {'logits': [], 'gen': [], 'indices': []}
        buckets[key]['logits'].append(q_obs_logits)
        buckets[key]['gen'].append(mod.generative_tensor)
        buckets[key]['indices'].append(i)

    q_obs_logits_groups = []
    gen_tensor_groups = []
    log_gen_tensor_groups = []
    modality_index_groups = []

    for _n_obs, data in buckets.items():
        q_obs_logits_groups.append(jnp.stack(data['logits']))
        gen = jnp.stack(data['gen'])
        gen_tensor_groups.append(gen)
        log_gen_tensor_groups.append(jnp.log(jnp.clip(gen, EPS, 1.0)))
        modality_index_groups.append(tuple(data['indices']))

    return (tuple(q_obs_logits_groups), tuple(gen_tensor_groups),
            tuple(log_gen_tensor_groups),
            tuple(modality_index_groups))


def compute_forward_marginals(
    q_theta: Array,                          # (n_theta,)
    q_u_given_x: Array,                     # (horizon, n_states, n_actions)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta) OR (n_states, n_states, n_actions, n_theta) when constant_transitions=True
    initial_state: Array,                    # (n_states,) one-hot or distribution
    horizon: int,
    constant_transitions: bool = False,
    transition_index: Array = None,          # (n_states, n_actions, n_theta) int32 when use_transition_index=True
    use_transition_index: bool = False,
    frozen: bool = False,
) -> Tuple[Array, Array, Array]:
    """
    Compute forward marginals via message passing using variational transitions.

    The policy q(u_t|x_{t-1}) is θ-independent and broadcast over θ in the forward pass:
        q(x_t, θ) = Σ_{x_{t-1}, u_t} q(x_t|x_{t-1},u_t,θ) · q(u_t|x_{t-1}) · q(x_{t-1}, θ)

    Uses jax.lax.scan for the sequential forward pass instead of a Python for loop,
    producing a compact XLA graph regardless of horizon length.

    When constant_transitions=True, q_x_given_xu_theta is (n_states, n_states, n_actions, n_theta)
    and is captured via closure instead of being passed as a scan input, avoiding
    materialization of a (horizon, n_states, n_states, n_actions, n_theta) broadcast.

    When use_transition_index=True, transition_index (n_states, n_actions, n_theta) -> int
    is used with scatter-add instead of a dense einsum, avoiding the dense transition tensor entirely.

    When frozen=True, q_xu_theta is not materialized (returns scalar dummy). Instead,
    q_u_theta is computed via einsum, avoiding a large (H, S, A, T) intermediate.

    Returns:
        q_x_theta: State-parameter marginals (horizon+1, n_states, n_theta)
        q_xu_theta: State-action-parameter marginals (horizon, n_states, n_actions, n_theta),
                    or scalar dummy when frozen=True
        q_u_theta: Action-parameter marginals (horizon, n_actions, n_theta)
    """
    # Clip and normalize inputs for numerical stability
    q_theta = jnp.clip(q_theta, EPS, 1.0)
    q_theta = q_theta / jnp.sum(q_theta)

    initial_state = jnp.clip(initial_state, EPS, 1.0)
    initial_state = initial_state / jnp.sum(initial_state)

    # Initial condition: q(x_0, θ) = p(x_0) · q(θ)
    q_x_theta_init = initial_state[:, None] * q_theta[None, :]  # (n_states, n_theta)

    if constant_transitions and use_transition_index:
        # Index-based scatter-add: no dense transition tensor needed.
        # transition_index: (n_states, n_actions, n_theta) -> int next_state
        n_states, n_theta_local = q_x_theta_init.shape
        theta_idx = jnp.broadcast_to(
            jnp.arange(n_theta_local)[None, None, :], transition_index.shape
        )

        def scan_body(q_x_theta_t, policy_t):
            # policy_t: (n_states, n_actions) — broadcast over θ
            q_xu_theta_t = policy_t[:, :, None] * q_x_theta_t[:, None, :]  # (n_states, n_actions, n_theta)
            q_x_next_theta = jnp.zeros((n_states, n_theta_local), dtype=q_x_theta_t.dtype)
            q_x_next_theta = q_x_next_theta.at[
                transition_index.ravel(), theta_idx.ravel()
            ].add(q_xu_theta_t.ravel())
            q_x_next_theta = jnp.clip(q_x_next_theta, EPS, None)
            normalizer = jnp.sum(q_x_next_theta, dtype=jnp.float32)
            q_x_next_theta = (q_x_next_theta / jnp.maximum(normalizer, EPS)).astype(q_x_theta_t.dtype)
            return q_x_next_theta, q_x_next_theta

        _, q_x_theta_rest = jax.lax.scan(
            scan_body, q_x_theta_init, q_u_given_x
        )
    elif constant_transitions:
        # Transition tensor is constant across time — capture via closure to avoid
        # materializing a (horizon, s, s, a, theta) broadcast tensor.
        var_trans_const = q_x_given_xu_theta  # (n_states, n_states, n_actions, n_theta)

        def scan_body(q_x_theta_t, policy_t):
            # policy_t: (n_states, n_actions) — broadcast over θ
            q_xu_theta_t = policy_t[:, None, :] * q_x_theta_t[:, :, None]  # (n_states, n_theta, n_actions)
            q_xu_theta_t = jnp.transpose(q_xu_theta_t, (0, 2, 1))  # (n_states, n_actions, n_theta)
            q_x_next_theta = jnp.einsum('xpau,pau->xu', var_trans_const, q_xu_theta_t)
            q_x_next_theta = jnp.clip(q_x_next_theta, EPS, None)
            normalizer = jnp.sum(q_x_next_theta, dtype=jnp.float32)
            q_x_next_theta = (q_x_next_theta / jnp.maximum(normalizer, EPS)).astype(q_x_theta_t.dtype)
            return q_x_next_theta, q_x_next_theta

        _, q_x_theta_rest = jax.lax.scan(
            scan_body, q_x_theta_init, q_u_given_x
        )
    else:
        def scan_body(q_x_theta_t, inputs):
            policy_t, var_trans_t = inputs
            # q(x_{t-1}, u_t, θ) = q(u_t|x_{t-1}) · q(x_{t-1}, θ) — policy broadcast over θ
            q_xu_theta_t = policy_t[:, None, :] * q_x_theta_t[:, :, None]  # (n_states, n_theta, n_actions)
            q_xu_theta_t = jnp.transpose(q_xu_theta_t, (0, 2, 1))  # (n_states_prev, n_actions, n_theta)

            # q(x_t, θ) = Σ_{x_{t-1}, u_t} q(x_t|x_{t-1},u_t,θ) · q(x_{t-1}, u_t, θ)
            q_x_next_theta = jnp.einsum('xpau,pau->xu', var_trans_t, q_xu_theta_t)

            # Normalize over full joint q(x_t, θ) so it sums to 1
            q_x_next_theta = jnp.clip(q_x_next_theta, EPS, None)
            normalizer = jnp.sum(q_x_next_theta, dtype=jnp.float32)
            q_x_next_theta = (q_x_next_theta / jnp.maximum(normalizer, EPS)).astype(q_x_theta_t.dtype)

            return q_x_next_theta, q_x_next_theta

        _, q_x_theta_rest = jax.lax.scan(
            scan_body, q_x_theta_init, (q_u_given_x, q_x_given_xu_theta)
        )

    # Concatenate initial state with scanned results: (horizon+1, n_states, n_theta)
    q_x_theta = jnp.concatenate([q_x_theta_init[None], q_x_theta_rest], axis=0)

    if frozen:
        # Frozen path: compute q_u_theta via einsum, avoid materializing (H, S, A, T) intermediate.
        # This is a batched (A, S) @ (S, T) matmul — benefits from Tensor Cores.
        q_u_theta = jnp.einsum('hsa,hst->hat', q_u_given_x, q_x_theta[:-1])  # (H, A, T)
        q_xu_theta = jnp.zeros(())  # dummy, not used in frozen path
    else:
        # Compute q(x_{t-1}, u_t, θ) and q(u_t, θ) from q_x_theta and policy — all timesteps at once
        # policy: (H, S, A), q_x_theta[:-1]: (H, S, T) -> q_xu_theta: (H, S, A, T)
        q_xu_theta = q_u_given_x[:, :, None, :] * q_x_theta[:-1, :, :, None]  # (H, S, T, A)
        q_xu_theta = jnp.transpose(q_xu_theta, (0, 1, 3, 2))  # (horizon, n_states, n_actions, n_theta)
        q_u_theta = jnp.sum(q_xu_theta, axis=1)  # (horizon, n_actions, n_theta)

    return q_x_theta, q_xu_theta, q_u_theta


def compute_entropy_terms(
    q_theta: Array,                          # (n_theta,)
    q_u_given_x: Array,                     # (horizon, n_states, n_actions)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    modality_groups: List[ModalityGroup],
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_xu_theta: Array,                       # (horizon, n_states, n_actions, n_theta)
    horizon: int = None,
) -> Array:
    """
    Compute total entropy over all variational factors — vectorized over time and modalities.
    """
    total_entropy = 0.0

    # 1. Parameter entropy: H[q(θ)]
    q_theta_safe = jnp.clip(q_theta, EPS, 1.0)
    h_theta = -jnp.sum(q_theta_safe * jnp.log(q_theta_safe))
    total_entropy += h_theta

    # 2a. Policy entropy — all timesteps at once (θ-independent)
    policy_safe = jnp.clip(q_u_given_x, EPS, 1.0)
    h_u_given_x = -jnp.sum(policy_safe * jnp.log(policy_safe), axis=2)  # (horizon, n_states)
    q_x_prev = jnp.sum(q_x_theta[:-1], axis=2)  # (horizon, n_states)
    policy_entropy = jnp.sum(q_x_prev * h_u_given_x)
    total_entropy += policy_entropy

    # 2b. Transition entropy — all timesteps at once
    var_trans_safe = jnp.clip(q_x_given_xu_theta, EPS, 1.0)
    h_x_given_xu_theta = -jnp.sum(var_trans_safe * jnp.log(var_trans_safe), axis=1)  # (horizon, n_states_prev, n_actions, n_theta)
    transition_entropy = jnp.sum(q_xu_theta * h_x_given_xu_theta)
    total_entropy += transition_entropy

    # 2c. Observation entropy — per modality group, all timesteps and modalities at once
    for grp in modality_groups:
        # grp.q_obs_batch: (M, horizon, n_obs, n_states, n_theta)
        var_obs_safe = jnp.clip(grp.q_obs_batch, EPS, 1.0)
        # Sum over obs outcomes (axis=2)
        h_obs = -jnp.sum(var_obs_safe * jnp.log(var_obs_safe), axis=2)
        # h_obs: (M, horizon, n_states, n_theta), weight by q(x_t, θ)
        # q_x_theta[1:]: (horizon, n_states, n_theta) -> broadcast over M
        total_entropy += jnp.sum(q_x_theta[1:][None, :, :, :] * h_obs)

    return total_entropy


def compute_epistemic_priors(
    q_theta: Array,                          # (n_theta,)
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_u_given_x: Array,                     # (horizon, n_states, n_actions)
    q_x_given_xu_theta: Array,              # (S,S,A,Θ) when constant_obs, (H,S,S,A,Θ) otherwise, or dummy
    modality_groups: List[ModalityGroup],
    horizon: int = None,
    constant_obs: bool = False,
    include_bethe: bool = False,
) -> Array:
    """
    Compute epistemic prior energies for Active Inference.

    Includes:
    - Bethe control prior: ũ(u_t) ∝ exp(H[q(x_t,x_{t-1}|u_t)] - H[q(x_{t-1}|u_t)])
    - State info prior: ũ(x_t) ∝ exp(-E_{q(θ|x)}[H[q(y|x,θ)]])
    - Observation prior: ũ(y,x) ∝ exp(KL[q(θ|y,x) || q(θ|x)])
    """
    total = 0.0

    # ============ Bethe control prior ============
    if include_bethe:
        # Stop gradient on inputs used to compute prior targets
        q_x_theta_prev_sg = jax.lax.stop_gradient(q_x_theta[:-1])  # (H, S, Θ)
        var_trans_sg = jax.lax.stop_gradient(q_x_given_xu_theta)

        # Differentiable expectation weights
        q_x_prev_all = jnp.sum(q_x_theta[:-1], axis=2)  # (H, S)
        q_u_all = jnp.sum(q_u_given_x * q_x_prev_all[:, :, None], axis=1)  # (H, A)

        # Stopped q(u) for computing prior targets
        q_u_all_sg = jnp.sum(
            q_u_given_x * jnp.sum(q_x_theta_prev_sg, axis=2)[:, :, None], axis=1)  # (H, A)

        if constant_obs:
            # var_trans_sg: (S_next, S_prev, A, Θ) — captured via closure
            def _bethe_step_const(carry, inputs):
                q_x_theta_t_sg, q_u_given_x_t, q_u_t_sg, q_u_t = inputs
                # q(x,θ|u) = q(u|x) q(x,θ) / q(u)
                q_x_theta_given_u = (
                    q_u_given_x_t[:, :, None] * q_x_theta_t_sg[:, None, :]
                ) / (q_u_t_sg[None, :, None] + EPS)  # (S, A, Θ)
                # q(x_t, x_{t-1} | u) = Σ_θ q(x_t|x_{t-1},u,θ) q(x_{t-1},θ|u)
                q_joint = jnp.einsum('xpat,pat->xpa', var_trans_sg, q_x_theta_given_u)
                q_joint_safe = jnp.clip(q_joint, EPS, 1.0)
                h_joint = -jnp.sum(q_joint_safe * jnp.log(q_joint_safe), axis=(0, 1))  # (A,)
                q_x_prev_given_u = jnp.sum(q_x_theta_given_u, axis=2)  # (S, A)
                q_x_prev_safe = jnp.clip(q_x_prev_given_u, EPS, 1.0)
                h_marg = -jnp.sum(q_x_prev_safe * jnp.log(q_x_prev_safe), axis=0)  # (A,)
                log_control_prior = jnp.log(softmax(h_joint - h_marg) + EPS)
                return carry + (-jnp.sum(q_u_t * log_control_prior)), None

            bethe_total, _ = jax.lax.scan(
                _bethe_step_const, 0.0,
                (q_x_theta_prev_sg, q_u_given_x, q_u_all_sg, q_u_all))
            total += bethe_total
        else:
            # var_trans_sg: (H, S_next, S_prev, A, Θ) — passed as scan input
            def _bethe_step_tv(carry, inputs):
                q_x_theta_t_sg, q_u_given_x_t, q_u_t_sg, q_u_t, var_trans_t_sg = inputs
                q_x_theta_given_u = (
                    q_u_given_x_t[:, :, None] * q_x_theta_t_sg[:, None, :]
                ) / (q_u_t_sg[None, :, None] + EPS)
                q_joint = jnp.einsum('xpat,pat->xpa', var_trans_t_sg, q_x_theta_given_u)
                q_joint_safe = jnp.clip(q_joint, EPS, 1.0)
                h_joint = -jnp.sum(q_joint_safe * jnp.log(q_joint_safe), axis=(0, 1))
                q_x_prev_given_u = jnp.sum(q_x_theta_given_u, axis=2)
                q_x_prev_safe = jnp.clip(q_x_prev_given_u, EPS, 1.0)
                h_marg = -jnp.sum(q_x_prev_safe * jnp.log(q_x_prev_safe), axis=0)
                log_control_prior = jnp.log(softmax(h_joint - h_marg) + EPS)
                return carry + (-jnp.sum(q_u_t * log_control_prior)), None

            bethe_total, _ = jax.lax.scan(
                _bethe_step_tv, 0.0,
                (q_x_theta_prev_sg, q_u_given_x, q_u_all_sg, q_u_all, var_trans_sg))
            total += bethe_total

    # ============ State info prior + Obs prior — scan over timesteps ============
    # Prior values: fixed targets (stop gradient so prior doesn't reshape theta/obs beliefs)
    q_x_theta_sg = jax.lax.stop_gradient(q_x_theta[1:])
    q_x_all_sg = jnp.sum(q_x_theta_sg, axis=2)
    q_theta_given_x_all = q_x_theta_sg / (q_x_all_sg[:, :, None] + EPS)  # (H, n_states, n_theta)
    log_q_theta_given_x_all = jnp.log(q_theta_given_x_all + EPS)

    # Expectation weight: differentiable (gradient flows to q_u and q_x_given_xu_theta)
    q_x_all = jnp.sum(q_x_theta[1:], axis=2)  # (H, n_states)

    if constant_obs:
        # Pre-compute time-independent quantities per group (captured via closure)
        precomputed_obs = []
        for grp in modality_groups:
            obs_batch_sg = jax.lax.stop_gradient(grp.q_obs_batch[:, 0])  # (M, n_obs, n_states, n_theta)
            obs_safe = jnp.clip(obs_batch_sg, EPS, 1.0)
            h_y_given_x_theta = -jnp.sum(obs_safe * jnp.log(obs_safe), axis=1)  # (M, n_states, n_theta)
            precomputed_obs.append((obs_batch_sg, h_y_given_x_theta))

        def _epistemic_obs_step(carry, inputs):
            q_theta_given_x_t, log_q_theta_given_x_t, q_x_t = inputs
            energy = 0.0
            for obs_batch_sg, h_y_given_x_theta in precomputed_obs:
                # State info prior
                h_y_given_x = jnp.sum(h_y_given_x_theta * q_theta_given_x_t[None, :, :], axis=2)  # (M, n_states)
                log_sip = jnp.log(softmax(-h_y_given_x, axis=1) + EPS)  # (M, n_states)
                energy += -jnp.sum(q_x_t[None, :] * log_sip)

                # Observation prior — per-timestep: (M, n_obs, n_states, n_theta) instead of (H, M, ...)
                q_theta_yx_unnorm = obs_batch_sg * q_theta_given_x_t[None, None, :, :]
                q_theta_given_yx = q_theta_yx_unnorm / (jnp.sum(q_theta_yx_unnorm, axis=3, keepdims=True) + EPS)
                log_ratio = jnp.log(q_theta_given_yx + EPS) - log_q_theta_given_x_t[None, None, :, :]
                kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=3)  # (M, n_obs, n_states)
                M_g, n_obs_g, n_states_g = kl_yx.shape
                yx_prior = softmax(kl_yx.reshape(M_g, -1), axis=1).reshape(M_g, n_obs_g, n_states_g)
                log_yx_prior = jnp.log(yx_prior + EPS)
                q_y_given_x = jnp.sum(obs_batch_sg * q_theta_given_x_t[None, None, :, :], axis=3)
                energy += -jnp.sum(q_y_given_x * q_x_t[None, None, :] * log_yx_prior)
            return carry + energy, None

        total_obs, _ = jax.lax.scan(
            _epistemic_obs_step, 0.0,
            (q_theta_given_x_all, log_q_theta_given_x_all, q_x_all))
        total += total_obs
    else:
        # Time-varying observations — scan per group
        for grp in modality_groups:
            obs_batch_sg = jax.lax.stop_gradient(grp.q_obs_batch)  # (M, H, n_obs, n_states, n_theta)
            obs_batch_sg_t = jnp.transpose(obs_batch_sg, (1, 0, 2, 3, 4))  # (H, M, n_obs, n_states, n_theta)

            def _epistemic_obs_step_tv(carry, inputs):
                q_theta_given_x_t, log_q_theta_given_x_t, q_x_t, obs_sg_t = inputs
                energy = 0.0

                # State info prior
                obs_safe = jnp.clip(obs_sg_t, EPS, 1.0)
                h_y_given_x_theta = -jnp.sum(obs_safe * jnp.log(obs_safe), axis=1)  # (M, n_states, n_theta)
                h_y_given_x = jnp.sum(h_y_given_x_theta * q_theta_given_x_t[None, :, :], axis=2)  # (M, n_states)
                log_sip = jnp.log(softmax(-h_y_given_x, axis=1) + EPS)
                energy += -jnp.sum(q_x_t[None, :] * log_sip)

                # Observation prior — per-timestep: (M, n_obs, n_states, n_theta)
                q_theta_yx_unnorm = obs_sg_t * q_theta_given_x_t[None, None, :, :]
                q_theta_given_yx = q_theta_yx_unnorm / (jnp.sum(q_theta_yx_unnorm, axis=3, keepdims=True) + EPS)
                log_ratio = jnp.log(q_theta_given_yx + EPS) - log_q_theta_given_x_t[None, None, :, :]
                kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=3)  # (M, n_obs, n_states)
                M_g, n_obs_g, n_states_g = kl_yx.shape
                yx_prior = softmax(kl_yx.reshape(M_g, -1), axis=1).reshape(M_g, n_obs_g, n_states_g)
                log_yx_prior = jnp.log(yx_prior + EPS)
                q_y_given_x = jnp.sum(obs_sg_t * q_theta_given_x_t[None, None, :, :], axis=3)
                energy += -jnp.sum(q_y_given_x * q_x_t[None, None, :] * log_yx_prior)

                return carry + energy, None

            total_obs_grp, _ = jax.lax.scan(
                _epistemic_obs_step_tv, 0.0,
                (q_theta_given_x_all, log_q_theta_given_x_all, q_x_all, obs_batch_sg_t))
            total += total_obs_grp

    return total


def compute_planning_correction(
    q_u_given_x: Array,                    # (horizon, n_states, n_actions)
    q_x_theta: Array,                       # (horizon+1, n_states, n_theta)
    horizon: int = None,
) -> Array:
    """
    Compute planning correction — vectorized over all timesteps at once.

    With θ-independent policy, q(u|x) is the policy directly (no θ marginalization needed).
    """
    q_x_prev = jnp.sum(q_x_theta[:-1], axis=2)  # (horizon, n_states)
    q_u_x = q_u_given_x * q_x_prev[:, :, None]  # (horizon, n_states, n_actions)

    q_u_given_x_safe = jnp.clip(q_u_given_x, EPS, 1.0)

    return -jnp.sum(q_u_x * jnp.log(q_u_given_x_safe))


def compute_energy_terms(
    q_theta: Array,                          # (n_theta,)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    modality_groups: List[ModalityGroup],
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_xu_theta: Array,                       # (horizon, n_states, n_actions, n_theta)
    q_u_theta: Array,                        # (horizon, n_actions, n_theta)
    transition_tensor: Array,                # (n_states, n_states, n_actions) or (n_states, n_states, n_theta, n_actions)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    horizon: int,
    goal_scale: float = 1.0,
) -> Dict[str, Array]:
    """
    Compute all energy terms — vectorized over time and modalities.
    """
    energies = {}

    # 1. Parameter prior energy: E_q[-log p(θ)]
    theta_prior_safe = jnp.clip(theta_prior, EPS, 1.0)
    theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior_safe))
    energies['theta'] = theta_energy

    # 2a. Action prior energy — all timesteps at once
    action_prior_safe = jnp.clip(action_prior, EPS, 1.0)
    q_u_all = jnp.sum(q_u_theta, axis=2)  # (horizon, n_actions)
    action_energy = -jnp.sum(q_u_all * jnp.log(action_prior_safe)[None, :])
    energies['action'] = action_energy

    # 2b. Transition energy — all timesteps at once
    trans_safe = jnp.clip(transition_tensor, EPS, 1.0)
    log_trans = jnp.log(trans_safe)

    q_joint_xpxu_theta = q_x_given_xu_theta * q_xu_theta[:, None, :, :, :]

    if transition_tensor.ndim == 3:
        q_joint_xpxu = jnp.sum(q_joint_xpxu_theta, axis=4)
        transition_energy = -jnp.sum(q_joint_xpxu * log_trans[None, :, :, :])
    else:
        log_trans_reordered = jnp.transpose(log_trans, (0, 1, 3, 2))
        transition_energy = -jnp.sum(q_joint_xpxu_theta * log_trans_reordered[None, :, :, :, :])

    energies['transition'] = transition_energy

    # 2c. Observation energy — per modality group, batched
    obs_energy = 0.0
    for grp in modality_groups:
        # grp.q_obs_batch: (M, horizon, n_obs, n_states, n_theta)
        # grp.log_gen_batch: (M, n_obs, n_states, n_theta)
        # q_joint = q(y|x,θ) · q(x,θ): (M, horizon, n_obs, n_states, n_theta)
        q_joint = grp.q_obs_batch * q_x_theta[1:][None, :, None, :, :]
        # log_gen: (M, n_obs, n_states, n_theta) -> broadcast over horizon
        obs_energy += -jnp.sum(q_joint * grp.log_gen_batch[:, None, :, :, :])
    energies['observation'] = obs_energy

    # 3. Goal energy: E_q[-log p(goal|x_T,θ)]
    goal_mapping_safe = jnp.clip(goal_mapping, EPS, 1.0)
    log_goal = jnp.log(goal_mapping_safe)
    q_x_final_theta = q_x_theta[horizon]
    goal_energy = -jnp.sum(q_x_final_theta * log_goal)
    energies['goal'] = goal_energy * goal_scale

    return energies


def temporal_vfe_jit(
    q_theta_logits: Array,                   # (n_theta,)
    q_u_given_x_logits: Array,              # (horizon, n_states, n_actions)
    q_x_given_xu_theta_logits: Array,       # (horizon, n_states, n_states, n_actions, n_theta) or dummy
    q_obs_logits_groups: Tuple[Array, ...],  # tuple of stacked logit arrays per group, or dummy
    initial_state: Array,                    # (n_states,)
    transition_tensor: Array,                # (n_states, n_states, n_actions) or (n_states, n_states, n_theta, n_actions) or dummy
    gen_tensor_groups: Tuple[Array, ...],    # tuple of stacked gen tensors per group
    log_gen_tensor_groups: Tuple[Array, ...], # tuple of stacked log gen tensors per group
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    transition_index: Array = None,          # (n_states, n_actions, n_theta) int32 or dummy
    horizon: int = 1,
    inference_mode: str = "marginal",
    goal_scale: float = 1.0,
    freeze_obs_and_transitions: bool = False,
    use_transition_index: bool = False,
) -> Array:
    """
    JIT-compatible temporal VFE. Takes pre-grouped modality data instead of
    ObservationModality objects, so all arguments are JAX arrays or static values.

    The policy q(u|x) is θ-independent with shape (horizon, n_states, n_actions).

    When freeze_obs_and_transitions=True, uses generative model tensors directly
    for transitions and observations (no variational factors allocated).
    Transition/observation entropy and energy cancel (-H[p] + E_p[-log p] = 0)
    and are skipped entirely.
    """
    # Convert logits to probabilities
    q_theta = softmax(q_theta_logits)
    q_u_given_x = softmax(q_u_given_x_logits, axis=-1)  # (horizon, n_states, n_actions)

    if freeze_obs_and_transitions:
        # Set up transitions
        if use_transition_index:
            # Index-based path: no dense transition tensor needed.
            q_x_given_xu_theta = None
        else:
            # Use generative transition tensor directly — NO broadcast over horizon.
            if transition_tensor.ndim == 4:
                q_x_from_gen = jnp.transpose(transition_tensor, (0, 1, 3, 2))
            else:
                q_x_from_gen = transition_tensor[..., None]
            q_x_given_xu_theta = q_x_from_gen  # (n_states, n_states, n_actions, n_theta)

        # Use generative obs tensors directly — NO broadcast over horizon.
        # Store with horizon=1 dim for ModalityGroup shape contract.
        modality_groups = []
        for i in range(len(gen_tensor_groups)):
            gen = gen_tensor_groups[i]
            q_obs_batch = gen[:, None]  # (M, 1, n_obs, n_states, n_theta)
            modality_groups.append(ModalityGroup(
                n_obs=gen.shape[1],
                q_obs_batch=q_obs_batch,
                gen_batch=gen,
                log_gen_batch=log_gen_tensor_groups[i],
            ))
    else:
        q_x_given_xu_theta = softmax(q_x_given_xu_theta_logits, axis=1)

        modality_groups = []
        for i in range(len(gen_tensor_groups)):
            q_obs_batch = softmax(q_obs_logits_groups[i], axis=2)
            modality_groups.append(ModalityGroup(
                n_obs=q_obs_batch.shape[2],
                q_obs_batch=q_obs_batch,
                gen_batch=gen_tensor_groups[i],
                log_gen_batch=log_gen_tensor_groups[i],
            ))

    # Compute forward marginals (checkpointed: recompute during backprop instead of storing)
    # static_argnums: horizon(4), constant_transitions(5), use_transition_index(7), frozen(8)
    q_x_theta, q_xu_theta, q_u_theta = jax.checkpoint(
        compute_forward_marginals, static_argnums=(4, 5, 7, 8)
    )(
        q_theta, q_u_given_x, q_x_given_xu_theta, initial_state,
        horizon, freeze_obs_and_transitions, transition_index, use_transition_index,
        freeze_obs_and_transitions,  # frozen: avoid materializing q_xu_theta when frozen
    )

    if freeze_obs_and_transitions:
        # When q_x = p and q_y = p, transition/obs entropy and energy cancel.
        # Only compute: -H[q(θ)] - H[q(u|x)] + E[-log p(θ)] + E[-log p(a)] + E[-log goal]
        q_theta_safe = jnp.clip(q_theta, EPS, 1.0)
        h_theta = -jnp.sum(q_theta_safe * jnp.log(q_theta_safe))

        policy_safe = jnp.clip(q_u_given_x, EPS, 1.0)
        h_u = -jnp.sum(policy_safe * jnp.log(policy_safe), axis=2)  # (horizon, n_states)
        q_x_prev = jnp.sum(q_x_theta[:-1], axis=2)  # (horizon, n_states)
        policy_entropy = jnp.sum(q_x_prev * h_u)

        entropy = h_theta + policy_entropy

        # Energy: only θ prior, action prior, goal
        theta_prior_safe = jnp.clip(theta_prior, EPS, 1.0)
        theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior_safe))

        action_prior_safe = jnp.clip(action_prior, EPS, 1.0)
        q_u_all = jnp.sum(q_u_theta, axis=2)
        action_energy = -jnp.sum(q_u_all * jnp.log(action_prior_safe)[None, :])

        goal_mapping_safe = jnp.clip(goal_mapping, EPS, 1.0)
        goal_energy = -jnp.sum(q_x_theta[horizon] * jnp.log(goal_mapping_safe)) * goal_scale

        vfe = -entropy + theta_energy + action_energy + goal_energy
    else:
        # Full VFE with all variational factors (checkpointed for memory efficiency)
        # static_argnums: horizon(6)
        entropy = jax.checkpoint(compute_entropy_terms, static_argnums=(6,))(
            q_theta, q_u_given_x, q_x_given_xu_theta,
            modality_groups, q_x_theta, q_xu_theta, horizon,
        )

        # static_argnums: horizon(10)
        energies = jax.checkpoint(compute_energy_terms, static_argnums=(10,))(
            q_theta, q_x_given_xu_theta, modality_groups, q_x_theta,
            q_xu_theta, q_u_theta, transition_tensor, goal_mapping,
            action_prior, theta_prior, horizon, goal_scale,
        )

        vfe = -entropy + energies['action'] + energies['transition'] + energies['observation'] + energies['goal'] + energies['theta']

    # Add epistemic priors for active inference
    if inference_mode == "active":
        # Planning correction: -Σ q(u,x) log q(u|x)
        planning_correction = compute_planning_correction(
            q_u_given_x=q_u_given_x,
            q_x_theta=q_x_theta,
            horizon=horizon,
        )
        # Recompute forward marginals with stopped q_theta —
        # epistemic priors should not produce gradients on q_theta_logits
        # static_argnums: horizon(4), constant_transitions(5), use_transition_index(7), frozen(8)
        q_x_theta_ep, _, _ = jax.checkpoint(
            compute_forward_marginals, static_argnums=(4, 5, 7, 8)
        )(
            jax.lax.stop_gradient(q_theta), q_u_given_x, q_x_given_xu_theta,
            initial_state, horizon, freeze_obs_and_transitions,
            transition_index, use_transition_index,
            freeze_obs_and_transitions,  # frozen
        )
        # Bethe requires dense transition tensor (not available with index path)
        include_bethe = q_x_given_xu_theta is not None
        bethe_trans = q_x_given_xu_theta if include_bethe else jnp.zeros(())
        # static_argnums: horizon(5), constant_obs(6), include_bethe(7)
        epistemic_energy = jax.checkpoint(
            compute_epistemic_priors, static_argnums=(5, 6, 7)
        )(
            q_theta, q_x_theta_ep, q_u_given_x, bethe_trans,
            modality_groups, horizon, freeze_obs_and_transitions, include_bethe,
        )
        vfe = vfe + planning_correction + epistemic_energy
    elif inference_mode == "planning":
        planning_correction = compute_planning_correction(
            q_u_given_x=q_u_given_x,
            q_x_theta=q_x_theta,
            horizon=horizon,
        )
        vfe = vfe + planning_correction

    return vfe


def temporal_vfe(
    q_theta_logits: Array,                   # (n_theta,)
    q_u_given_x_logits: Array,              # (horizon, n_states, n_actions)
    q_x_given_xu_theta_logits: Array,       # (horizon, n_states, n_states, n_actions, n_theta)
    q_obs_logits_list: List[Array],          # list of logit arrays, one per modality
    initial_state: Array,                    # (n_states,)
    transition_tensor: Array,                # (n_states, n_states, n_actions) or (n_states, n_states, n_theta, n_actions)
    observation_modalities: List[ObservationModality],
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    horizon: int,
    inference_mode: str = "marginal",
    goal_scale: float = 1.0,
) -> Array:
    """
    Compute variational free energy with full temporal factorization.
    Convenience wrapper that groups modalities and delegates to temporal_vfe_jit.
    """
    q_obs_logits_groups, gen_tensor_groups, log_gen_tensor_groups, \
        _ = group_modalities_for_jit(
            q_obs_logits_list, observation_modalities)

    return temporal_vfe_jit(
        q_theta_logits=q_theta_logits,
        q_u_given_x_logits=q_u_given_x_logits,
        q_x_given_xu_theta_logits=q_x_given_xu_theta_logits,
        q_obs_logits_groups=q_obs_logits_groups,
        initial_state=initial_state,
        transition_tensor=transition_tensor,
        gen_tensor_groups=gen_tensor_groups,
        log_gen_tensor_groups=log_gen_tensor_groups,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        transition_index=None,
        horizon=horizon,
        inference_mode=inference_mode,
        goal_scale=goal_scale,
    )


def extract_marginals_temporal(
    q_theta_logits: Array,
    q_u_given_x_logits: Array,
    q_x_given_xu_theta_logits: Array,
    q_obs_logits_list: List[Array],
    observation_modalities: List[ObservationModality],
    initial_state: Array,
    horizon: int,
    constant_transitions: bool = False,
    transition_index: Array = None,
    use_transition_index: bool = False,
) -> Dict[str, Array]:
    """Extract all marginal distributions for analysis.

    When constant_transitions=True, q_x_given_xu_theta_logits is (n_states, n_states, n_actions, n_theta)
    and obs logits have no horizon dimension — avoids materializing large broadcast tensors.

    When use_transition_index=True, uses scatter-add for forward marginals instead of dense einsum.
    """
    q_theta = softmax(q_theta_logits)
    q_u_given_x = softmax(q_u_given_x_logits, axis=-1)  # (horizon, n_states, n_actions)

    if constant_transitions and not use_transition_index:
        # Transition logits are (n_states, n_states, n_actions, n_theta) — softmax over axis 0 (next state)
        q_x_given_xu_theta = softmax(q_x_given_xu_theta_logits, axis=0)
    elif not constant_transitions:
        q_x_given_xu_theta = softmax(q_x_given_xu_theta_logits, axis=1)
    else:
        q_x_given_xu_theta = None  # Not needed for index path

    if constant_transitions:
        # Obs logits have no horizon dim — softmax over obs outcomes (axis 0)
        q_obs_list = []
        for q_obs_logits in q_obs_logits_list:
            q_obs_list.append(softmax(q_obs_logits, axis=0))
    else:
        q_obs_list = []
        for q_obs_logits in q_obs_logits_list:
            q_obs_list.append(softmax(q_obs_logits, axis=1))

    q_x_theta, q_xu_theta, q_u_theta = compute_forward_marginals(
        q_theta=q_theta,
        q_u_given_x=q_u_given_x,
        q_x_given_xu_theta=q_x_given_xu_theta,
        initial_state=initial_state,
        horizon=horizon,
        constant_transitions=constant_transitions,
        transition_index=transition_index,
        use_transition_index=use_transition_index,
    )

    result = {
        'q_theta': q_theta,
        'q_u_given_x': q_u_given_x,
        'q_x_given_xu_theta': q_x_given_xu_theta,
        'q_x_theta': q_x_theta,
        'q_xu_theta': q_xu_theta,
        'q_u_theta': q_u_theta,
    }

    if constant_transitions:
        # Obs have no horizon dim — broadcast for API consistency
        for q_obs, mod in zip(q_obs_list, observation_modalities):
            result[f'q_obs_{mod.name}'] = jnp.broadcast_to(q_obs[None], (horizon, *q_obs.shape))
    else:
        for q_obs, mod in zip(q_obs_list, observation_modalities):
            result[f'q_obs_{mod.name}'] = q_obs

    return result
