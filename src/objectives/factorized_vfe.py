"""
Variational Free Energy minimization with FACTORIZED distribution.

q(y_{1:T}, x_{1:T}, u_{1:T}, θ) = q(x|u) q(y,θ|x) q(u)

This factorization exploits the structure:
- q(u): distribution over action sequences
- q(x|u): distribution over state sequences given actions  
- q(y,θ|x): joint distribution over observations and parameters given states

VFE = -H[q] + E_q[-log p(u)] + E_q[-log p(x|x_prev, u)] + E_q[-log p(y|x, θ)] + E_q[-log p(goal|x_T, θ)] + E_q[-log p(θ)]

Entropy decomposes as:
H[q] = H[q(u)] + E_{q(u)}[H[q(x|u)]] + E_{q(x)}[H[q(y,θ|x)]]

Epistemic priors (active mode):
- p̃(u) ∝ exp(H[q(x|u)])              - prefer actions with uncertain state outcomes
- p̃(x) ∝ exp(-H[q(y|x)])             - prefer states with informative observations  
- p̃(y,x) ∝ exp(KL[q(θ|y,x)||q(θ|x)]) - prefer (y,x) pairs that update beliefs about θ
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax

EPS = 1e-10


def enumerate_state_sequences(n_states: int, horizon: int) -> Array:
    """Enumerate all possible state sequences of length horizon."""
    grids = jnp.meshgrid(*[jnp.arange(n_states) for _ in range(horizon)], indexing='ij')
    sequences = jnp.stack([g.flatten() for g in grids], axis=1)
    return sequences


def enumerate_action_sequences(n_actions: int, horizon: int) -> Array:
    """Enumerate all possible action sequences."""
    grids = jnp.meshgrid(*[jnp.arange(n_actions) for _ in range(horizon)], indexing='ij')
    sequences = jnp.stack([g.flatten() for g in grids], axis=1)
    return sequences


def enumerate_obs_sequences(n_obs: int, horizon: int) -> Array:
    """Enumerate all possible observation sequences of length horizon."""
    grids = jnp.meshgrid(*[jnp.arange(n_obs) for _ in range(horizon)], indexing='ij')
    sequences = jnp.stack([g.flatten() for g in grids], axis=1)
    return sequences


def compute_transition_log_probs(
    initial_state_idx: int,
    state_sequences: Array,
    action_sequences: Array,
    transition_tensor: Array,
) -> Array:
    """Compute log p(x_{1:T} | x_0, u_{1:T}) for all (state_seq, action_seq) pairs."""
    horizon = state_sequences.shape[1]
    
    def log_prob_for_pair(state_seq, action_seq):
        log_prob = 0.0
        prev_state = initial_state_idx
        for t in range(horizon):
            curr_state = state_seq[t]
            action = action_seq[t]
            p_transition = transition_tensor[curr_state, prev_state, action]
            log_prob = log_prob + jnp.log(p_transition + EPS)
            prev_state = curr_state
        return log_prob
    
    log_probs = jax.vmap(
        lambda s_seq: jax.vmap(lambda a_seq: log_prob_for_pair(s_seq, a_seq))(action_sequences)
    )(state_sequences)
    
    return log_probs  # (n_state_seqs, n_action_seqs)


def compute_observation_log_probs(
    obs_sequences: Array,
    state_sequences: Array,
    observation_tensor: Array,
) -> Array:
    """
    Compute log p(y_{1:T} | x_{1:T}, θ) for all (obs_seq, state_seq, θ) combinations.
    
    Returns:
        Shape (n_obs_seqs, n_state_seqs, n_theta)
    """
    horizon = obs_sequences.shape[1]
    n_theta = observation_tensor.shape[2]
    
    def log_prob_for_triple(obs_seq, state_seq, theta_idx):
        log_prob = 0.0
        for t in range(horizon):
            obs = obs_seq[t]
            state = state_seq[t]
            p_obs = observation_tensor[obs, state, theta_idx]
            log_prob = log_prob + jnp.log(p_obs + EPS)
        return log_prob
    
    log_probs = jax.vmap(
        lambda o_seq: jax.vmap(
            lambda s_seq: jax.vmap(
                lambda theta: log_prob_for_triple(o_seq, s_seq, theta)
            )(jnp.arange(n_theta))
        )(state_sequences)
    )(obs_sequences)
    
    return log_probs  # (n_obs_seqs, n_state_seqs, n_theta)


def factorized_vfe(
    q_u_logits: Array,
    q_x_given_u_logits: Array,
    q_y_theta_given_x_logits: Array,
    initial_state: Array,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    action_prior: Array,
    theta_prior: Array,
    horizon: int,
    inference_mode: str = "marginal",
) -> Array:
    """
    Compute VFE with factorized q(y,x,u,θ) = q(x|u) q(y,θ|x) q(u).
    
    Args:
        q_u_logits: Logits for q(u), shape (n_action_seqs,)
        q_x_given_u_logits: Logits for q(x|u), shape (n_state_seqs, n_action_seqs)
        q_y_theta_given_x_logits: Logits for q(y,θ|x), shape (n_obs_seqs, n_theta, n_state_seqs)
        initial_state: One-hot initial state, shape (n_states,)
        transition_tensor: p(x'|x,u), shape (n_states, n_states, n_actions)
        observation_tensor: p(y|x,θ), shape (n_obs, n_states, n_theta)
        goal_mapping: p(goal|x,θ), shape (n_states, n_theta)
        action_prior: Action prior, shape (n_actions,)
        theta_prior: Prior over θ (reward location), shape (n_theta,)
        horizon: Planning horizon T
        inference_mode: "marginal", "active", or "planning"
        
    Returns:
        Scalar VFE loss.
    """
    n_obs = observation_tensor.shape[0]
    n_states = transition_tensor.shape[0]
    n_actions = transition_tensor.shape[2]
    n_theta = goal_mapping.shape[1]
    
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    # Enumerate all sequences
    obs_sequences = enumerate_obs_sequences(n_obs, horizon)
    state_sequences = enumerate_state_sequences(n_states, horizon)
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    initial_state_idx = jnp.argmax(initial_state)
    
    # ========== Compute factorized distributions ==========
    # q(u): distribution over action sequences
    q_u = softmax(q_u_logits)  # (n_action_seqs,)
    
    # q(x|u): for each action sequence, distribution over state sequences
    # Apply softmax along state dimension for each action sequence
    q_x_given_u = softmax(q_x_given_u_logits, axis=0)  # (n_state_seqs, n_action_seqs)
    
    # q(y,θ|x): for each state sequence, joint distribution over (y,θ)
    # Reshape to (n_obs_seqs * n_theta, n_state_seqs) for softmax, then reshape back
    q_y_theta_given_x_flat = q_y_theta_given_x_logits.reshape(-1, n_state_seqs)
    q_y_theta_given_x = softmax(q_y_theta_given_x_flat, axis=0).reshape(
        n_obs_seqs, n_theta, n_state_seqs
    )  # (n_obs_seqs, n_theta, n_state_seqs)
    
    # ========== Compute marginals from factorization ==========
    # q(x,u) = q(x|u) q(u)
    q_xu = q_x_given_u * q_u[None, :]  # (n_state_seqs, n_action_seqs)
    
    # q(x) = sum_u q(x,u)
    q_x = jnp.sum(q_xu, axis=1)  # (n_state_seqs,)
    
    # q(y,θ,x) = q(y,θ|x) q(x)
    # q_y_theta_given_x is (n_obs_seqs, n_theta, n_state_seqs)
    q_y_theta_x = q_y_theta_given_x * q_x[None, None, :]  # (n_obs_seqs, n_theta, n_state_seqs)
    
    # q(x,θ) = sum_y q(y,θ,x)
    q_x_theta = jnp.sum(q_y_theta_x, axis=0).T  # (n_state_seqs, n_theta)
    
    # q(θ) = sum_x q(x,θ)
    q_theta = jnp.sum(q_x_theta, axis=0)  # (n_theta,)
    
    # ========== Entropy: H[q] = H[q(u)] + E_{q(u)}[H[q(x|u)]] + E_{q(x)}[H[q(y,θ|x)]] ==========
    # H[q(u)]
    h_u = -jnp.sum(q_u * jnp.log(q_u + EPS))
    
    # E_{q(u)}[H[q(x|u)]] = sum_u q(u) H[q(x|u)]
    h_x_given_u = -jnp.sum(q_x_given_u * jnp.log(q_x_given_u + EPS), axis=0)  # (n_action_seqs,)
    expected_h_x_given_u = jnp.sum(q_u * h_x_given_u)
    
    # E_{q(x)}[H[q(y,θ|x)]] = sum_x q(x) H[q(y,θ|x)]
    h_y_theta_given_x = -jnp.sum(
        q_y_theta_given_x * jnp.log(q_y_theta_given_x + EPS), axis=(0, 1)
    )  # (n_state_seqs,)
    expected_h_y_theta_given_x = jnp.sum(q_x * h_y_theta_given_x)
    
    total_entropy = h_u + expected_h_x_given_u + expected_h_y_theta_given_x
    neg_entropy = -total_entropy
    
    # ========== Action prior energy: E_q[-log p(u)] ==========
    log_prior_per_action_seq = jnp.sum(jnp.log(action_prior[action_sequences] + EPS), axis=1)
    action_energy = -jnp.sum(q_u * log_prior_per_action_seq)
    
    # ========== Transition energy: E_q[-log p(x|x_prev, u)] ==========
    log_transition_probs = compute_transition_log_probs(
        initial_state_idx, state_sequences, action_sequences, transition_tensor
    )  # (n_state_seqs, n_action_seqs)
    transition_energy = -jnp.sum(q_xu * log_transition_probs)
    
    # ========== Observation likelihood energy: E_q[-log p(y|x, θ)] ==========
    log_obs_probs = compute_observation_log_probs(
        obs_sequences, state_sequences, observation_tensor
    )  # (n_obs_seqs, n_state_seqs, n_theta)
    # Rearrange to match q_y_theta_x shape: (n_obs_seqs, n_theta, n_state_seqs)
    log_obs_probs_reordered = jnp.transpose(log_obs_probs, (0, 2, 1))
    obs_energy = -jnp.sum(q_y_theta_x * log_obs_probs_reordered)
    
    # ========== Goal energy: E_q[-log p(goal|x_T, θ)] ==========
    final_states = state_sequences[:, -1]
    log_goal = jnp.log(goal_mapping + EPS)
    log_goal_per_x_theta = log_goal[final_states, :]  # (n_state_seqs, n_theta)
    goal_energy = -jnp.sum(q_x_theta * log_goal_per_x_theta)
    
    # ========== Parameter prior energy: E_q[-log p(θ)] ==========
    theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior + EPS))
    
    # ========== Epistemic priors / planning correction ==========
    epistemic_u_energy = 0.0
    epistemic_x_energy = 0.0
    epistemic_yx_energy = 0.0
    planning_correction = 0.0
    
    if inference_mode == "active":
        # p̃(u) ∝ exp(H[q(x|u)]) - prefer actions with uncertain outcomes
        # h_x_given_u already computed above
        control_prior = softmax(h_x_given_u)
        log_control_prior = jnp.log(control_prior + EPS)
        epistemic_u_energy = -jnp.sum(q_u * log_control_prior)
        
        # p̃(x) ∝ exp(-H[q(y|x)]) - prefer states with low observation entropy
        # NOTE: This is NOT uniform! q(y|x) = sum_θ q(y,θ|x) depends on q(θ|x).
        # At the cue state, if q(θ|x) is peaked, q(y|x) is peaked (low H).
        # If q(θ|x) is uniform, q(y|x) is uniform (high H).
        # This creates an implicit dependence on θ beliefs!
        
        # q(y|x) = sum_θ q(y,θ|x) for each state sequence
        # q_y_theta_given_x is (n_obs_seqs, n_theta, n_state_seqs)
        q_y_given_x = jnp.sum(q_y_theta_given_x, axis=1)  # (n_obs_seqs, n_state_seqs)
        
        # H[q(y|x)] for each state sequence x
        h_y_given_x = -jnp.sum(
            q_y_given_x * jnp.log(q_y_given_x + EPS), axis=0
        )  # (n_state_seqs,)
        
        # Prior: p̃(x) ∝ exp(-H[q(y|x)]) - low entropy (informative) states preferred
        state_prior = softmax(-h_y_given_x)
        log_state_prior = jnp.log(state_prior + EPS)
        epistemic_x_energy = -jnp.sum(q_x * log_state_prior)
        
        # p̃(y,x) ∝ exp(KL[q(θ|y,x) || q(θ|x)]) - prefer (y,x) that update θ beliefs
        # q(y,x) = sum_θ q(y,θ,x)
        q_yx = jnp.sum(q_y_theta_x, axis=1)  # (n_obs_seqs, n_state_seqs)
        
        # q(θ|y,x) = q(y,θ,x) / q(y,x)
        # q_y_theta_x is (n_obs_seqs, n_theta, n_state_seqs)
        q_theta_given_yx = q_y_theta_x / (q_yx[:, None, :] + EPS)  # (n_obs_seqs, n_theta, n_state_seqs)
        
        # q(θ|x) = q(x,θ) / q(x)
        q_theta_given_x = q_x_theta / (q_x[:, None] + EPS)  # (n_state_seqs, n_theta)
        
        # KL[q(θ|y,x) || q(θ|x)] for each (y,x)
        # Need to broadcast q_theta_given_x to (n_obs_seqs, n_theta, n_state_seqs)
        q_theta_given_x_broadcast = q_theta_given_x.T[None, :, :]  # (1, n_theta, n_state_seqs)
        log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(q_theta_given_x_broadcast + EPS)
        kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=1)  # (n_obs_seqs, n_state_seqs)
        
        # Prior: p̃(y,x) ∝ exp(KL)
        yx_prior = softmax(kl_yx.flatten()).reshape(kl_yx.shape)
        log_yx_prior = jnp.log(yx_prior + EPS)
        epistemic_yx_energy = -jnp.sum(q_yx * log_yx_prior)
    
    elif inference_mode == "planning":
        # Planning entropy correction: ∑_{t=1}^T H[q(x_{t-1}, u_t)] - H[q(x_{t-1})]
        planning_correction = _compute_planning_entropy_correction(
            q_xu, state_sequences, action_sequences, n_states, n_actions, horizon
        )
    
    return (
        neg_entropy 
        + action_energy 
        + transition_energy 
        + obs_energy 
        + goal_energy 
        + theta_energy
        + epistemic_u_energy 
        + epistemic_x_energy 
        + epistemic_yx_energy
        + planning_correction
    )


def _compute_planning_entropy_correction(
    q_xu: Array,
    state_sequences: Array,
    action_sequences: Array,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """Compute ∑_{t=1}^T H[q(x_{t-1}, u_t)] - H[q(x_{t-1})]."""
    correction = 0.0
    
    # t=1: x_0 is deterministic (H[x_0] = 0), so contribution = H[u_1]
    first_actions = action_sequences[:, 0]
    action_one_hot = jax.nn.one_hot(first_actions, n_actions)
    q_u_seqs = jnp.sum(q_xu, axis=0)
    q_u1 = action_one_hot.T @ q_u_seqs
    h_u1 = -jnp.sum(q_u1 * jnp.log(q_u1 + EPS))
    correction = h_u1
    
    # t=2 to T
    for t in range(2, horizon + 1):
        states_tm1 = state_sequences[:, t - 2]
        actions_t = action_sequences[:, t - 1]
        
        state_one_hot = jax.nn.one_hot(states_tm1, n_states)
        action_one_hot = jax.nn.one_hot(actions_t, n_actions)
        
        q_joint = state_one_hot.T @ q_xu @ action_one_hot
        h_joint = -jnp.sum(q_joint * jnp.log(q_joint + EPS))
        
        q_marginal = jnp.sum(q_joint, axis=1)
        h_marginal = -jnp.sum(q_marginal * jnp.log(q_marginal + EPS))
        
        correction = correction + h_joint - h_marginal
    
    return correction


def extract_marginals_from_factorized(
    q_u_logits: Array,
    q_x_given_u_logits: Array,
    q_y_theta_given_x_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> dict:
    """
    Extract all marginals from the factorized distribution.
    
    Returns:
        Dictionary with keys: q_u, q_x, q_theta, q_first_action, q_all_actions, q_all_states, q_all_obs
    """
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    # Compute factorized distributions
    q_u = softmax(q_u_logits)
    q_x_given_u = softmax(q_x_given_u_logits, axis=0)
    q_y_theta_given_x_flat = q_y_theta_given_x_logits.reshape(-1, n_state_seqs)
    q_y_theta_given_x = softmax(q_y_theta_given_x_flat, axis=0).reshape(
        n_obs_seqs, n_theta, n_state_seqs
    )
    
    # Compute marginals
    q_xu = q_x_given_u * q_u[None, :]
    q_x = jnp.sum(q_xu, axis=1)
    q_y_theta_x = q_y_theta_given_x * q_x[None, None, :]
    q_x_theta = jnp.sum(q_y_theta_x, axis=0).T
    q_theta = jnp.sum(q_x_theta, axis=0)
    q_y = jnp.sum(q_y_theta_x, axis=(1, 2))
    
    # Extract time-slice marginals (vectorized)
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    state_sequences = enumerate_state_sequences(n_states, horizon)
    obs_sequences = enumerate_obs_sequences(n_obs, horizon)
    
    # q(u_t) for all t using one-hot and matmul: (horizon, n_actions)
    action_onehot = jax.nn.one_hot(action_sequences, n_actions)  # (n_action_seqs, horizon, n_actions)
    q_all_actions = jnp.einsum('s,stn->tn', q_u, action_onehot)  # (horizon, n_actions)
    q_first_action = q_all_actions[0]
    
    # q(x_t) for all t: (horizon, n_states)
    state_onehot = jax.nn.one_hot(state_sequences, n_states)  # (n_state_seqs, horizon, n_states)
    q_all_states = jnp.einsum('s,stn->tn', q_x, state_onehot)  # (horizon, n_states)
    
    # q(y_t) for all t: (horizon, n_obs)
    obs_onehot = jax.nn.one_hot(obs_sequences, n_obs)  # (n_obs_seqs, horizon, n_obs)
    q_all_obs = jnp.einsum('s,stn->tn', q_y, obs_onehot)  # (horizon, n_obs)
    
    return {
        'q_u': q_u,
        'q_x': q_x,
        'q_theta': q_theta,
        'q_first_action': q_first_action,
        'q_all_actions': q_all_actions,
        'q_all_states': q_all_states,
        'q_all_obs': q_all_obs,
    }


def reconstruct_full_joint(
    q_u_logits: Array,
    q_x_given_u_logits: Array,
    q_y_theta_given_x_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """
    Reconstruct the full joint q(y,x,u,θ) from the factorized distributions.
    
    q(y,x,u,θ) = q(x|u) q(y,θ|x) q(u)
    
    Returns:
        Array of shape (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    """
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    # Compute factorized distributions
    q_u = softmax(q_u_logits)  # (n_action_seqs,)
    q_x_given_u = softmax(q_x_given_u_logits, axis=0)  # (n_state_seqs, n_action_seqs)
    q_y_theta_given_x_flat = q_y_theta_given_x_logits.reshape(-1, n_state_seqs)
    q_y_theta_given_x = softmax(q_y_theta_given_x_flat, axis=0).reshape(
        n_obs_seqs, n_theta, n_state_seqs
    )  # (n_obs_seqs, n_theta, n_state_seqs)
    
    # q(y,x,u,θ) = q(y,θ|x) * q(x|u) * q(u)
    # Start with q(x|u) * q(u) = q(x,u)
    q_xu = q_x_given_u * q_u[None, :]  # (n_state_seqs, n_action_seqs)
    
    # Now multiply by q(y,θ|x)
    # q_y_theta_given_x: (n_obs_seqs, n_theta, n_state_seqs)
    # q_xu: (n_state_seqs, n_action_seqs)
    # Want: (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    
    # q(y,x,u,θ) = q(y,θ|x) * q(x,u)
    # For each (y,x,u,θ): q(y,θ|x) * q(x,u)
    q_yxu_theta = (
        q_y_theta_given_x[:, :, :, None] *  # (n_obs_seqs, n_theta, n_state_seqs, 1)
        q_xu[None, None, :, :]              # (1, 1, n_state_seqs, n_action_seqs)
    )  # (n_obs_seqs, n_theta, n_state_seqs, n_action_seqs)
    
    # Reorder to (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    q_yxu_theta = jnp.transpose(q_yxu_theta, (0, 2, 3, 1))
    
    return q_yxu_theta
