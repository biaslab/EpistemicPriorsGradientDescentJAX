"""
Temporal factorization of variational free energy for Active Inference planning.

This module implements VFE minimization with full temporal (Markovian) factorization:
    q(x_{0:T}, u_{1:T}, y_{1:T}, θ) = q(θ) · ∏_{t=1}^T q(u_t|x_{t-1},θ) · q(x_t|x_{t-1},u_t,θ) · q(y_t|x_t,θ)

All four factors are variational (optimized):
1. q(θ) - parameter belief
2. q(u_t|x_{t-1},θ) - policy
3. q(x_t|x_{t-1},u_t,θ) - variational transitions
4. q(y_t|x_t,θ) - variational observations

VFE Decomposition:
    VFE = -H[q] + E_q[-log p(...)]

Where:
    H[q] = H[q(θ)]
         + Σ_t E_{x_{t-1},θ}[H[q(u_t|x_{t-1},θ)]]           # Policy entropy
         + Σ_t E_{x_{t-1},u_t,θ}[H[q(x_t|x_{t-1},u_t,θ)]]   # Transition entropy
         + Σ_t E_{x_t,θ}[H[q(y_t|x_t,θ)]]                   # Observation entropy

Key advantages over sequence-based factorization:
- Linear scaling with horizon (vs exponential)
- Natural reactive policy representation
- Full variational inference over all factors
"""

from typing import Tuple, Dict, Optional
import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax
from jax.scipy.special import logsumexp

# Numerical stability constant
EPS = 1e-8


def compute_forward_marginals(
    q_theta: Array,                          # (n_theta,)
    q_u_given_x_theta: Array,               # (horizon, n_states, n_theta, n_actions)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    initial_state: Array,                    # (n_states,) one-hot or distribution
    horizon: int,
) -> Tuple[Array, Array, Array]:
    """
    Compute forward marginals via message passing using variational transitions.

    Recursively computes:
        q(x_0, θ) = p(x_0) · q(θ)
        q(x_t, θ) = ∑_{x_{t-1}, u_t} q(x_t|x_{t-1},u_t,θ) · q(u_t|x_{t-1},θ) · q(x_{t-1},θ)

    Note: Uses variational distribution q(x_t|x_{t-1},u_t,θ) for message passing,
    not the generative model p(x_t|...). Energy terms still use p(x_t|...).

    Args:
        q_theta: Parameter belief distribution (n_theta,)
        q_u_given_x_theta: Policy per timestep (horizon, n_states, n_theta, n_actions)
        q_x_given_xu_theta: Variational transitions (horizon, n_states_next, n_states_prev, n_actions, n_theta)
        initial_state: Initial state distribution (n_states,)
        horizon: Planning horizon

    Returns:
        q_x_theta: State-parameter marginals (horizon+1, n_states, n_theta)
        q_xu_theta: State-action-parameter marginals (horizon, n_states, n_actions, n_theta)
        q_u_theta: Action-parameter marginals (horizon, n_actions, n_theta)
    """
    n_states = q_x_given_xu_theta.shape[1]
    n_actions = q_x_given_xu_theta.shape[3]
    n_theta = q_theta.shape[0]

    # Clip and normalize inputs for numerical stability
    q_theta = jnp.clip(q_theta, EPS, 1.0)
    q_theta = q_theta / jnp.sum(q_theta)

    initial_state = jnp.clip(initial_state, EPS, 1.0)
    initial_state = initial_state / jnp.sum(initial_state)

    # Initialize storage
    q_x_theta = jnp.zeros((horizon + 1, n_states, n_theta))

    # Initial condition: q(x_0, θ) = p(x_0) · q(θ)
    # Broadcasting: (n_states, 1) * (1, n_theta) -> (n_states, n_theta)
    q_x_theta = q_x_theta.at[0].set(initial_state[:, None] * q_theta[None, :])

    # Forward pass: compute marginals for each timestep
    for t in range(horizon):
        # Get policy for this timestep: q(u_t | x_{t-1}, θ)
        # Shape: (n_states, n_theta, n_actions)
        policy_t = q_u_given_x_theta[t]  # (n_states, n_theta, n_actions)

        # Compute q(x_{t-1}, u_t, θ) = q(u_t | x_{t-1}, θ) · q(x_{t-1}, θ)
        # Broadcasting: (n_states, n_theta, n_actions) * (n_states, n_theta, 1)
        q_x_prev_theta_t = q_x_theta[t]  # (n_states, n_theta)
        q_xu_theta_t = policy_t * q_x_prev_theta_t[:, :, None]  # (n_states, n_theta, n_actions)
        q_xu_theta_t = jnp.transpose(q_xu_theta_t, (0, 2, 1))  # (n_states_prev, n_actions, n_theta)

        # Get variational transition for this timestep: q(x_t | x_{t-1}, u_t, θ)
        # Shape: (n_states_next, n_states_prev, n_actions, n_theta)
        var_trans_t = q_x_given_xu_theta[t]  # (n_states_next, n_states_prev, n_actions, n_theta)

        # Compute q(x_t, θ) = ∑_{x_{t-1}, u_t} q(x_t|x_{t-1},u_t,θ) · q(x_{t-1}, u_t, θ)
        # var_trans_t: (n_states_next, n_states_prev, n_actions, n_theta)
        # q_xu_theta_t: (n_states_prev, n_actions, n_theta)
        # Multiply and marginalize over x_{t-1} and u_t
        q_x_next_theta = jnp.einsum('xpau,pau->xu', var_trans_t, q_xu_theta_t)  # (n_states_next, n_theta)

        # Normalize and clip for numerical stability
        q_x_next_theta = jnp.clip(q_x_next_theta, EPS, None)
        normalizer = jnp.sum(q_x_next_theta, axis=0, keepdims=True)  # (1, n_theta)
        q_x_next_theta = q_x_next_theta / jnp.maximum(normalizer, EPS)

        q_x_theta = q_x_theta.at[t + 1].set(q_x_next_theta)

    # Compute q(x_{t-1}, u_t, θ) and q(u_t, θ) from q_x_theta and policy
    q_xu_theta = q_u_given_x_theta * q_x_theta[:-1, :, :, None]  # (horizon, n_states, n_theta, n_actions)
    q_xu_theta = jnp.transpose(q_xu_theta, (0, 1, 3, 2))  # (horizon, n_states, n_actions, n_theta)
    q_u_theta = jnp.sum(q_xu_theta, axis=1)  # (horizon, n_actions, n_theta)

    return q_x_theta, q_xu_theta, q_u_theta


def compute_entropy_terms(
    q_theta: Array,                          # (n_theta,)
    q_u_given_x_theta: Array,               # (horizon, n_states, n_theta, n_actions)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    q_y_given_x_theta: Array,               # (horizon, n_obs, n_states, n_theta)
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_xu_theta: Array,                       # (horizon, n_states, n_actions, n_theta)
    q_y_location_given_x: Array,            # (horizon, n_locations, n_states)
    horizon: int = None,
) -> Array:
    """
    Compute total entropy over all variational factors.

    H[q] = H[q(θ)] + ∑_t [ E_{x_{t-1},θ}[H[q(u_t|x_{t-1},θ)]]
                          + E_{x_{t-1},u_t,θ}[H[q(x_t|x_{t-1},u_t,θ)]]
                          + E_{x_t,θ}[H[q(y_t|x_t,θ)]] ]

    All four entropy components are computed over variational distributions.
    """
    total_entropy = 0.0

    # 1. Parameter entropy: H[q(θ)]
    q_theta_safe = jnp.clip(q_theta, EPS, 1.0)
    h_theta = -jnp.sum(q_theta_safe * jnp.log(q_theta_safe))
    total_entropy += h_theta

    # 2. Per-timestep entropies
    for t in range(horizon):
        # 2a. Policy entropy: E_{x_{t-1},θ}[H[q(u_t|x_{t-1},θ)]]
        policy_t = q_u_given_x_theta[t]  # (n_states, n_theta, n_actions)
        policy_t_safe = jnp.clip(policy_t, EPS, 1.0)

        # H[q(u_t|x_{t-1},θ)] for each (x_{t-1}, θ)
        h_u_given_x_theta = -jnp.sum(
            policy_t_safe * jnp.log(policy_t_safe),
            axis=2  # Sum over actions
        )  # (n_states, n_theta)

        # Weight by q(x_{t-1}, θ)
        q_x_prev_theta = q_x_theta[t]  # (n_states, n_theta)
        policy_entropy = jnp.sum(q_x_prev_theta * h_u_given_x_theta)
        total_entropy += policy_entropy

        # 2b. Transition entropy: E_{x_{t-1},u_t,θ}[H[q(x_t|x_{t-1},u_t,θ)]]
        # Variational transitions: q_x_given_xu_theta[t] has shape (n_states_next, n_states_prev, n_actions, n_theta)
        var_trans_t = q_x_given_xu_theta[t]  # (n_states_next, n_states_prev, n_actions, n_theta)
        var_trans_t_safe = jnp.clip(var_trans_t, EPS, 1.0)

        # H[q(x_t|x_{t-1},u_t,θ)] for each (x_{t-1}, u_t, θ)
        h_x_given_xu_theta = -jnp.sum(
            var_trans_t_safe * jnp.log(var_trans_t_safe),
            axis=0  # Sum over x_next
        )  # (n_states_prev, n_actions, n_theta)

        # Weight by q(x_{t-1}, u_t, θ) = q(u_t|x_{t-1},θ) * q(x_{t-1},θ)
        # q_xu_theta has shape (horizon, n_states, n_actions, n_theta)
        q_xu_theta_t = q_xu_theta[t]  # (n_states_prev, n_actions, n_theta)
        transition_entropy = jnp.sum(q_xu_theta_t * h_x_given_xu_theta)
        total_entropy += transition_entropy

        # 2c. Observation entropy: E_{x_t,θ}[H[q(y_t|x_t,θ)]]
        # Variational observations: q_y_given_x_theta[t] has shape (n_obs, n_states, n_theta)
        var_obs_t = q_y_given_x_theta[t]  # (n_obs, n_states, n_theta)
        var_obs_t_safe = jnp.clip(var_obs_t, EPS, 1.0)

        # H[q(y_t|x_t,θ)] for each (x_t, θ)
        h_y_given_x_theta = -jnp.sum(
            var_obs_t_safe * jnp.log(var_obs_t_safe),
            axis=0  # Sum over observations
        )  # (n_states, n_theta)

        # Weight by q(x_t, θ)
        q_x_next_theta = q_x_theta[t + 1]  # (n_states, n_theta)
        obs_entropy = jnp.sum(q_x_next_theta * h_y_given_x_theta)
        total_entropy += obs_entropy

    # 2d. Location observation entropy: E_{x_t}[H[q(y_location|x_t)]]
    for t in range(horizon):
        q_loc_t = q_y_location_given_x[t]  # (n_locations, n_states)
        q_loc_t_safe = jnp.clip(q_loc_t, EPS, 1.0)
        
        # H[q(y_location|x_t)] for each x_t
        h_loc_given_x = -jnp.sum(
            q_loc_t_safe * jnp.log(q_loc_t_safe),
            axis=0  # Sum over locations
        )  # (n_states,)
        
        # Weight by q(x_t)
        q_x_t = jnp.sum(q_x_theta[t + 1], axis=1)  # (n_states,)
        loc_obs_entropy = jnp.sum(q_x_t * h_loc_given_x)
        total_entropy += loc_obs_entropy

    return total_entropy


def compute_epistemic_priors(
    q_theta: Array,                          # (n_theta,)
    q_u_given_x_theta: Array,               # (horizon, n_states, n_theta, n_actions)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_y_theta_given_x_theta: Array,          # (horizon, n_obs_theta, n_states, n_theta) - variational theta observations
    q_y_location_given_x: Array,  # (horizon, n_locations, n_states) - variational location observations
    horizon: int = None,
) -> Array:
    """
    Compute epistemic prior energies for Active Inference (Bethe factorization).

    Four epistemic priors (all computed from variational q, not generative p):
    1. Control prior: ũ(u_t) ∝ exp(H[q(x_t,x_{t-1}|u_t)] - H[q(x_{t-1}|u_t)])
       - Prefer actions with uncertain next-state outcomes (exploration)
       - Uses variational transition q(x_t|x_{t-1},u_t,θ)

    2. State prior (theta obs): ũ(x_t) ∝ exp(-H[q(y_theta|x_t)])
       - Prefer states with informative theta observations (low entropy)
       - Uses variational observation q(y_theta|x_t,θ) marginalized over θ

    3. Observation prior (theta): ũ(y_t, x_t) ∝ exp(KL[q(θ|y_t,x_t) || q(θ)])
       - Prefer (observation, state) pairs that update θ beliefs
       - Uses variational observation q(y_theta|x_t,θ) for Bayesian update

    4. State prior (location obs): ũ(x_t) ∝ exp(-H[q(y_location|x_t)])
       - Prefer states with informative location observations
       - Uses variational observation q(y_location|x_t)

    Returns:
        Total epistemic energy (scalar)
    """
    n_actions = q_x_given_xu_theta.shape[3]

    total_epistemic_energy = 0.0

    for t in range(horizon):
        # ============ 1. Control prior (Bethe): ũ(u_t) ∝ exp(H[q(x_t,x_{t-1}|u_t)] - H[q(x_{t-1}|u_t)]) ============
        # Uses variational transition q(x_t|x_{t-1},u_t,θ) instead of generative p(x_t|...)

        # Compute q(x_{t-1}, u_t, θ) = q(u_t|x_{t-1},θ) q(x_{t-1},θ)
        q_x_prev_theta = q_x_theta[t]  # (n_states, n_theta)
        policy_t = q_u_given_x_theta[t]  # (n_states, n_theta, n_actions)
        q_x_prev_u_theta = policy_t * q_x_prev_theta[:, :, None]  # (n_states, n_theta, n_actions)

        # q(u_t) = Σ_{x,θ} q(x_{t-1}, u_t, θ)
        q_u_t = jnp.sum(q_x_prev_u_theta, axis=(0, 1))  # (n_actions,)

        # Get variational transition for this timestep
        var_trans_t = q_x_given_xu_theta[t]  # (n_states_next, n_states_prev, n_actions, n_theta)
        var_trans_t_safe = jnp.clip(var_trans_t, EPS, 1.0)

        # For each action, compute H[q(x_t,x_{t-1}|u_t)] - H[q(x_{t-1}|u_t)]
        bethe_delta_h = jnp.zeros(n_actions)

        for u in range(n_actions):
            denom = q_u_t[u] + EPS
            q_x_prev_given_u = q_x_prev_u_theta[:, :, u] / denom  # (n_states, n_theta)

            # q(x_{t-1}|u_t) = Σ_θ q(x_{t-1},θ|u_t)
            q_x_prev_given_u_marg = jnp.sum(q_x_prev_given_u, axis=1)  # (n_states,)

            # q(x_t, x_{t-1} | u_t) using variational transition
            # = Σ_θ q(x_t|x_{t-1},u_t,θ) q(x_{t-1},θ|u_t)
            # var_trans_t_safe[:, :, u, :]: (n_states_next, n_states_prev, n_theta)
            # q_x_prev_given_u: (n_states_prev, n_theta)
            q_x_t_x_prev = jnp.einsum('xpu,pu->xp', var_trans_t_safe[:, :, u, :], q_x_prev_given_u)

            q_x_t_x_prev_safe = jnp.clip(q_x_t_x_prev, EPS, 1.0)
            h_joint = -jnp.sum(q_x_t_x_prev_safe * jnp.log(q_x_t_x_prev_safe))

            q_x_prev_safe = jnp.clip(q_x_prev_given_u_marg, EPS, 1.0)
            h_marg = -jnp.sum(q_x_prev_safe * jnp.log(q_x_prev_safe))

            bethe_delta_h = bethe_delta_h.at[u].set(h_joint - h_marg)

        # Control prior: softmax over actions based on entropy
        control_prior = softmax(bethe_delta_h)
        log_control_prior = jnp.log(control_prior + EPS)

        # Energy: E_q[-log ũ(u_t)]
        control_energy = -jnp.sum(q_u_t * log_control_prior)
        total_epistemic_energy += control_energy

        # ============ 2. State prior (THETA): ũ(x_t) ∝ exp(-E_{q(θ|x_t)}[H[q(y_theta|x_t,θ)]]) ============
        # Expected conditional entropy: prefer states where observations are
        # informative about theta, weighted by current beliefs about theta given state.

        # H[q(y_theta|x_t,θ)] for each (x_t, θ)
        q_y_theta_given_x_theta_t = q_y_theta_given_x_theta[t]  # (n_obs_theta, n_states, n_theta)
        q_y_safe = jnp.clip(q_y_theta_given_x_theta_t, EPS, 1.0)
        h_y_given_x_theta = -jnp.sum(
            q_y_safe * jnp.log(q_y_safe),
            axis=0  # Sum over observations
        )  # (n_states, n_theta)

        # q(θ|x_t) = q(x_t,θ) / q(x_t) from forward marginals
        q_x_theta_t = q_x_theta[t + 1]  # (n_states, n_theta)
        q_x_t = jnp.sum(q_x_theta_t, axis=1)  # (n_states,)
        q_theta_given_x = q_x_theta_t / jnp.clip(q_x_t[:, None], EPS, None)  # (n_states, n_theta)

        # E_{q(θ|x_t)}[H[q(y_theta|x_t,θ)]]
        expected_cond_h = jnp.sum(q_theta_given_x * h_y_given_x_theta, axis=1)  # (n_states,)

        # State prior: ũ(x_t) ∝ exp(-E[H[...]]) - prefer LOW expected conditional entropy
        state_theta_prior = softmax(-expected_cond_h)
        log_state_theta_prior = jnp.log(state_theta_prior + EPS)

        # Energy: E_q[-log ũ(x_t)]
        state_theta_energy = -jnp.sum(q_x_t * log_state_theta_prior)
        total_epistemic_energy += state_theta_energy

         # ============ 3. State prior (LOCATION): ũ(x_t) ∝ exp(-H[q(y_location | x_t)]) ============
        # Uses VARIATIONAL location observation model q(y_location|x_t)
        # Location observations are independent of θ
       
        # q(y_location | x_t) from variational distribution
        # q_y_location_given_x: (horizon, n_locations, n_states)
        q_loc_given_x_t = q_y_location_given_x[t]  # (n_locations, n_states)

        q_loc_given_x_safe = jnp.clip(q_loc_given_x_t, EPS, 1.0)

        # H[q(y_location | x_t)] for each x_t
        h_loc_given_x = -jnp.sum(
            q_loc_given_x_safe * jnp.log(q_loc_given_x_safe),
            axis=0  # Sum over locations
        )  # (n_states,)

        # Location informativeness prior: ũ(x_t) ∝ exp(-H[y_location | x_t])
        # Prefer states with certain location observations
        loc_info_prior = softmax(-h_loc_given_x)
        log_loc_info_prior = jnp.log(loc_info_prior + EPS)

        # Energy: E_q[-log ũ(x_t)] from location informativeness
        loc_info_energy = -jnp.sum(q_x_t * log_loc_info_prior)
        total_epistemic_energy += loc_info_energy

        # ============ 4. Observation prior (THETA): ũ(y_t, x_t) ∝ exp(KL[q(θ|y_t,x_t) || q(θ)]) ============
        # Uses VARIATIONAL observation model q(y_theta|x_t,θ) for Bayesian update
        # Information gain: how much would observing y at state x update θ beliefs

        # q(θ | y_t, x_t) ∝ q(y_t | x_t, θ) q(θ)
        # q_y_theta_given_x_theta_t: (n_obs_theta, n_states, n_theta)
        # q_theta: (n_theta,) -> (1, 1, n_theta)

        # Unnormalized posterior: q(y|x,θ) q(θ)
        q_theta_yx_unnorm = q_y_theta_given_x_theta_t * q_theta[None, None, :]  # (n_obs_theta, n_states, n_theta)

        # Normalize over θ to get q(θ | y, x)
        q_theta_yx_norm = jnp.sum(q_theta_yx_unnorm, axis=2, keepdims=True)  # (n_obs_theta, n_states, 1)
        q_theta_given_yx = q_theta_yx_unnorm / (q_theta_yx_norm + EPS)  # (n_obs_theta, n_states, n_theta)

        # KL[q(θ|y,x) || q(θ)] for each (y, x)
        log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(q_theta[None, None, :] + EPS)
        kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=2)  # (n_obs_theta, n_states)

        # Prior: ũ(y_t, x_t) ∝ exp(KL) - prefer (y,x) pairs with high information gain
        yx_prior = softmax(kl_yx.flatten()).reshape(kl_yx.shape)  # (n_obs_theta, n_states)
        log_yx_prior = jnp.log(yx_prior + EPS)

        # q(y_t, x_t) = Σ_θ q(y|x,θ) q(x,θ)
        q_yx = jnp.sum(
            q_y_theta_given_x_theta_t * q_x_theta_t[None, :, :],  # (n_obs_theta, n_states, n_theta)
            axis=2  # Sum over θ
        )  # (n_obs_theta, n_states)

        # Energy: E_q[-log ũ(y_t, x_t)]
        obs_theta_energy = -jnp.sum(q_yx * log_yx_prior)
        total_epistemic_energy += obs_theta_energy

       

    return total_epistemic_energy


def compute_planning_correction(
    q_u_given_x_theta: Array,               # (horizon, n_states, n_theta, n_actions
    q_x_theta: Array,                       # (horizon+1, n_states, n_theta)
    horizon: int = None,
) -> Array:
    """
    Compute planning correction: expected conditional entropy of q(u_t | x_{t-1}).

    Constructs q(u_t, x_{t-1}) by marginalizing θ:
        q(u_t, x_{t-1}) = Σ_θ q(u_t | x_{t-1}, θ) q(x_{t-1}, θ)

    Then computes the conditional entropy as:
        Σ_t E_{q(u_t, x_{t-1})}[-log q(u_t | x_{t-1})]
    """
    total_entropy = 0.0

    for t in range(horizon):
        # q(x_{t-1}, θ)
        q_x_prev_theta = q_x_theta[t]  # (n_states, n_theta)

        # q(u_t, x_{t-1}, θ) = q(u_t | x_{t-1}, θ) q(x_{t-1}, θ)
        q_u_x_theta = q_u_given_x_theta[t] * q_x_prev_theta[:, :, None]  # (n_states, n_theta, n_actions)

        # q(u_t, x_{t-1}) = Σ_θ q(u_t, x_{t-1}, θ)
        q_u_x = jnp.sum(q_u_x_theta, axis=1)  # (n_states, n_actions)

        # q(x_{t-1}) for normalization
        q_x_prev = jnp.sum(q_x_prev_theta, axis=1, keepdims=True)  # (n_states, 1)

        # q(u_t | x_{t-1}) = q(u_t, x_{t-1}) / q(x_{t-1})
        q_u_given_x = q_u_x / (q_x_prev + EPS)  # (n_states, n_actions)

        q_u_given_x_safe = jnp.clip(q_u_given_x, EPS, 1.0)
        q_u_x_safe = jnp.clip(q_u_x, EPS, 1.0)

        # E_{q(u_t, x_{t-1})}[-log q(u_t | x_{t-1})]
        total_entropy += -jnp.sum(q_u_x_safe * jnp.log(q_u_given_x_safe))

    return total_entropy


def compute_energy_terms(
    q_theta: Array,                          # (n_theta,)
    q_x_given_xu_theta: Array,              # (horizon, n_states, n_states, n_actions, n_theta)
    q_y_given_x_theta: Array,               # (horizon, n_obs, n_states, n_theta)
    q_y_location_given_x: Array,            # (horizon, n_locations, n_states)
    q_x_theta: Array,                        # (horizon+1, n_states, n_theta)
    q_xu_theta: Array,                       # (horizon, n_states, n_actions, n_theta)
    q_u_theta: Array,                        # (horizon, n_actions, n_theta)
    transition_tensor: Array,                # (n_states, n_states, n_actions)
    observation_tensor: Array,               # (n_obs, n_states, n_theta)
    location_observation_tensor: Array,      # (n_locations, n_states)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    horizon: int,
) -> Dict[str, Array]:
    """
    Compute all energy terms: E_q[-log p(·)].

    Energy terms use the generative model p(...) but are computed over
    variational marginals q(...). This is the standard VFE formulation.

    Returns dictionary with individual terms for analysis.
    """
    energies = {}

    # 1. Parameter prior energy: E_q[-log p(θ)]
    theta_prior_safe = jnp.clip(theta_prior, EPS, 1.0)
    theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior_safe))
    energies['theta'] = theta_energy

    # 2. Per-timestep energies
    action_prior_safe = jnp.clip(action_prior, EPS, 1.0)
    trans_safe = jnp.clip(transition_tensor, EPS, 1.0)
    log_trans = jnp.log(trans_safe)  # (n_states, n_states, n_actions)
    obs_model_safe = jnp.clip(observation_tensor, EPS, 1.0)
    log_obs_model = jnp.log(obs_model_safe)  # (n_obs, n_states, n_theta)

    action_energy = 0.0
    transition_energy = 0.0
    observation_energy = 0.0

    for t in range(horizon):
        # 2a. Action prior energy: E_q[-log p(u_t)]
        q_u_t = jnp.sum(q_u_theta[t], axis=1)  # Marginalize over θ -> (n_actions,)
        action_energy += -jnp.sum(q_u_t * jnp.log(action_prior_safe))

        # 2b. Transition energy: E_q[-log p(x_t|x_{t-1},u_t,θ)]
        # We need q(x_t, x_{t-1}, u_t, θ) = q(x_t|x_{t-1},u_t,θ) · q(x_{t-1},u_t,θ)
        # Then compute: ∑_{x_t,x_{t-1},u_t,θ} q(x_t,x_{t-1},u_t,θ) · [-log p(x_t|x_{t-1},u_t)]

        # Get variational transition: q(x_t|x_{t-1},u_t,θ)
        var_trans_t = q_x_given_xu_theta[t]  # (n_states_next, n_states_prev, n_actions, n_theta)

        # Get q(x_{t-1}, u_t, θ)
        q_xu_theta_t = q_xu_theta[t]  # (n_states_prev, n_actions, n_theta)

        # Compute q(x_t, x_{t-1}, u_t, θ)
        # var_trans_t: (x_next, x_prev, u, θ)
        # q_xu_theta_t: (x_prev, u, θ) -> expand to (1, x_prev, u, θ)
        q_joint_xpxu_theta = var_trans_t * q_xu_theta_t[None, :, :, :]  # (x_next, x_prev, u, θ)

        if transition_tensor.ndim == 3:
            # p(x_t|x_{t-1},u_t) doesn't depend on θ
            # log_trans: (x_next, x_prev, u)
            # Sum over θ first to get q(x_t, x_{t-1}, u_t)
            q_joint_xpxu = jnp.sum(q_joint_xpxu_theta, axis=3)  # (x_next, x_prev, u)
            transition_energy += -jnp.sum(q_joint_xpxu * log_trans)
        else:
            # p(x_t|x_{t-1},u_t,θ) depends on θ
            # log_trans: (x_next, x_prev, u, θ)
            transition_energy += -jnp.sum(q_joint_xpxu_theta * log_trans)

        # 2c. Observation energy: E_q[-log p(y_t|x_t,θ)]
        # We need q(y_t, x_t, θ) = q(y_t|x_t,θ) · q(x_t,θ)
        # Then compute: ∑_{y_t,x_t,θ} q(y_t,x_t,θ) · [-log p(y_t|x_t,θ)]

        # Get variational observation: q(y_t|x_t,θ)
        var_obs_t = q_y_given_x_theta[t]  # (n_obs, n_states, n_theta)

        # Get q(x_t, θ)
        q_x_next_theta = q_x_theta[t + 1]  # (n_states, n_theta)

        # Compute q(y_t, x_t, θ)
        # var_obs_t: (n_obs, n_states, n_theta)
        # q_x_next_theta: (n_states, n_theta) -> expand to (1, n_states, n_theta)
        q_joint_yx_theta = var_obs_t * q_x_next_theta[None, :, :]  # (n_obs, n_states, n_theta)

        # log_obs_model: (n_obs, n_states, n_theta)
        observation_energy += -jnp.sum(q_joint_yx_theta * log_obs_model)

    energies['action'] = action_energy
    energies['transition'] = transition_energy
    energies['observation'] = observation_energy

    # 2d. Location observation energy: E_q[-log p(y_location|x_t)] (if available)
    location_energy = 0.0
    loc_obs_model_safe = jnp.clip(location_observation_tensor, EPS, 1.0)
    log_loc_obs_model = jnp.log(loc_obs_model_safe)  # (n_locations, n_states)
    
    for t in range(horizon):
        # q(y_location, x) = q(y_location|x) · q(x)
        q_loc_t = q_y_location_given_x[t]  # (n_locations, n_states)
        q_x_t = jnp.sum(q_x_theta[t + 1], axis=1)  # (n_states,)
        
        q_joint_loc_x = q_loc_t * q_x_t[None, :]  # (n_locations, n_states)
        
        # Energy: -sum q(y,x) log p(y|x)
        location_energy += -jnp.sum(q_joint_loc_x * log_loc_obs_model)
    
    energies['location_observation'] = location_energy

    # 3. Goal energy: E_q[-log p(goal|x_T,θ)]
    goal_mapping_safe = jnp.clip(goal_mapping, EPS, 1.0)
    log_goal = jnp.log(goal_mapping_safe)  # (n_states, n_theta)

    q_x_final_theta = q_x_theta[horizon]  # Final state (n_states, n_theta)
    goal_energy = -jnp.sum(q_x_final_theta * log_goal)
    energies['goal'] = goal_energy

    return energies


def temporal_vfe(
    q_theta_logits: Array,                   # (n_theta,)
    q_u_given_x_theta_logits: Array,        # (horizon, n_states, n_theta, n_actions)
    q_x_given_xu_theta_logits: Array,       # (horizon, n_states, n_states, n_actions, n_theta)
    q_y_theta_given_x_theta_logits: Array,  # (horizon, n_obs_theta, n_states, n_theta) - theta observations
    q_y_location_given_x_logits: Array,  # (horizon, n_locations, n_states) - location observations
    initial_state: Array,                    # (n_states,)
    transition_tensor: Array,                # (n_states, n_states, n_actions)
    theta_observation_tensor: Array,         # (n_obs_theta, n_states, n_theta)
    location_observation_tensor: Array,      # (n_locations, n_states)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    horizon: int,
    inference_mode: str = "marginal",
) -> Array:
    """
    Compute variational free energy with full temporal factorization.

    VFE = -H[q] + E_q[-log p(...)]

    Where H[q] includes entropy over variational factors:
    - q(θ): parameter belief
    - q(u_t|x_{t-1},θ): policy
    - q(x_t|x_{t-1},u_t,θ): variational transitions
    - q(y_theta_t|x_t,θ): variational theta observations
    - q(y_location_t|x_t): variational location observations

    Args:
        q_theta_logits: Logits for parameter belief (n_theta,)
        q_u_given_x_theta_logits: Logits for policy (horizon, n_states, n_theta, n_actions)
        q_x_given_xu_theta_logits: Logits for variational transitions (horizon, n_states, n_states, n_actions, n_theta)
        q_y_theta_given_x_theta_logits: Logits for variational theta observations (horizon, n_obs_theta, n_states, n_theta)
        q_y_location_given_x_logits: Logits for variational location observations (horizon, n_locations, n_states)
        initial_state: Initial state distribution (n_states,)
        transition_tensor: p(s'|s,a) generative model (n_states, n_states, n_actions)
        theta_observation_tensor: p(o_theta|s,θ) generative model for theta (n_obs_theta, n_states, n_theta)
        location_observation_tensor: p(o_location|s) generative model (n_locations, n_states)
        goal_mapping: p(goal|s,θ) (n_states, n_theta)
        action_prior: p(a) (n_actions,)
        theta_prior: p(θ) (n_theta,)
        horizon: Planning horizon
        inference_mode: "marginal", "active", or "planning"

    Returns:
        VFE scalar loss
    """
    # Convert logits to probabilities
    q_theta = softmax(q_theta_logits)
    q_u_given_x_theta = softmax(q_u_given_x_theta_logits, axis=-1)  # Softmax over actions
    q_x_given_xu_theta = softmax(q_x_given_xu_theta_logits, axis=1)  # Softmax over x_next (axis 1)
    q_y_theta_given_x_theta = softmax(q_y_theta_given_x_theta_logits, axis=1)  # Softmax over y_theta (axis 1)
    
    # Location observations (always provided)
    q_y_location_given_x = softmax(q_y_location_given_x_logits, axis=1)  # Softmax over y_location (axis 1)

    # Compute forward marginals using variational transitions
    q_x_theta, q_xu_theta, q_u_theta = compute_forward_marginals(
        q_theta=q_theta,
        q_u_given_x_theta=q_u_given_x_theta,
        q_x_given_xu_theta=q_x_given_xu_theta,
        initial_state=initial_state,
        horizon=horizon,
    )

    # Compute entropy over all variational factors
    entropy = compute_entropy_terms(
        q_theta=q_theta,
        q_u_given_x_theta=q_u_given_x_theta,
        q_x_given_xu_theta=q_x_given_xu_theta,
        q_y_given_x_theta=q_y_theta_given_x_theta,  # Theta observations
        q_x_theta=q_x_theta,
        q_xu_theta=q_xu_theta,
        q_y_location_given_x=q_y_location_given_x,  # Location observations
        horizon=horizon,
    )

    # Compute energies (using generative model p(...) over variational marginals)
    energies = compute_energy_terms(
        q_theta=q_theta,
        q_x_given_xu_theta=q_x_given_xu_theta,
        q_y_given_x_theta=q_y_theta_given_x_theta,  # Theta observations
        q_y_location_given_x=q_y_location_given_x,  # Location observations
        q_x_theta=q_x_theta,
        q_xu_theta=q_xu_theta,
        q_u_theta=q_u_theta,
        transition_tensor=transition_tensor,
        observation_tensor=theta_observation_tensor,  # Theta observation model
        location_observation_tensor=location_observation_tensor,  # Location observation model
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        horizon=horizon,
    )

    # Base VFE
    vfe = -entropy + energies['action'] + energies['transition'] + energies['observation'] + energies['location_observation'] + energies['goal'] + energies['theta']

    # Add epistemic priors for active inference
    if inference_mode == "active":
        epistemic_energy = compute_epistemic_priors(
            q_theta=q_theta,
            q_u_given_x_theta=q_u_given_x_theta,
            q_x_given_xu_theta=q_x_given_xu_theta,
            q_x_theta=q_x_theta,
            q_y_theta_given_x_theta=q_y_theta_given_x_theta,
            q_y_location_given_x=q_y_location_given_x,
            horizon=horizon,
        )
        vfe = vfe + epistemic_energy
    elif inference_mode == "planning":
        planning_correction = compute_planning_correction(
            q_u_given_x_theta=q_u_given_x_theta,
            q_x_theta=q_x_theta,
            horizon=horizon,
        )
        vfe = vfe + planning_correction

    return vfe


def extract_marginals_temporal(
    q_theta_logits: Array,
    q_u_given_x_theta_logits: Array,
    q_x_given_xu_theta_logits: Array,
    q_y_theta_given_x_theta_logits: Array,
    q_y_location_given_x_logits: Array,
    initial_state: Array,
    horizon: int,
) -> Dict[str, Array]:
    """
    Extract all marginal distributions for analysis.

    Returns:
        Dictionary with:
        - q_theta: (n_theta,)
        - q_u_given_x_theta: (horizon, n_states, n_theta, n_actions)
        - q_x_given_xu_theta: (horizon, n_states, n_states, n_actions, n_theta)
        - q_y_theta_given_x_theta: (horizon, n_obs_theta, n_states, n_theta)
        - q_y_location_given_x: (horizon, n_locations, n_states)
        - q_x_theta: (horizon+1, n_states, n_theta)
        - q_xu_theta: (horizon, n_states, n_actions, n_theta)
        - q_u_theta: (horizon, n_actions, n_theta)
    """
    q_theta = softmax(q_theta_logits)
    q_u_given_x_theta = softmax(q_u_given_x_theta_logits, axis=-1)
    q_x_given_xu_theta = softmax(q_x_given_xu_theta_logits, axis=1)  # Softmax over x_next
    q_y_theta_given_x_theta = softmax(q_y_theta_given_x_theta_logits, axis=1)  # Softmax over y_theta
    q_y_location_given_x = softmax(q_y_location_given_x_logits, axis=1)  # Softmax over y_location

    q_x_theta, q_xu_theta, q_u_theta = compute_forward_marginals(
        q_theta=q_theta,
        q_u_given_x_theta=q_u_given_x_theta,
        q_x_given_xu_theta=q_x_given_xu_theta,
        initial_state=initial_state,
        horizon=horizon,
    )

    return {
        'q_theta': q_theta,
        'q_u_given_x_theta': q_u_given_x_theta,
        'q_x_given_xu_theta': q_x_given_xu_theta,
        'q_y_theta_given_x_theta': q_y_theta_given_x_theta,
        'q_y_location_given_x': q_y_location_given_x,
        'q_x_theta': q_x_theta,
        'q_xu_theta': q_xu_theta,
        'q_u_theta': q_u_theta,
    }
