"""
Variational Free Energy minimization with FULL JOINT factorization.

q(y_{1:T}, x_{1:T}, u_{1:T}, θ) - joint over observations, states, actions, and parameters.

VFE = -H[q] + E_q[-log p(u)] + E_q[-log p(x|x_prev, u)] + E_q[-log p(y|x, θ)] + E_q[-log p(goal|x_T, θ)] + E_q[-log p(θ)]

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
    
    return log_probs


def compute_observation_log_probs(
    obs_sequences: Array,
    state_sequences: Array,
    observation_tensor: Array,
) -> Array:
    """
    Compute log p(y_{1:T} | x_{1:T}, θ) for all (obs_seq, state_seq, θ) combinations.
    
    Args:
        obs_sequences: Shape (n_obs_seqs, T)
        state_sequences: Shape (n_state_seqs, T)
        observation_tensor: p(y|x,θ), shape (n_obs, n_states, n_theta)
        
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
    
    # Vectorize over all combinations
    log_probs = jax.vmap(
        lambda o_seq: jax.vmap(
            lambda s_seq: jax.vmap(
                lambda theta: log_prob_for_triple(o_seq, s_seq, theta)
            )(jnp.arange(n_theta))
        )(state_sequences)
    )(obs_sequences)
    
    return log_probs  # (n_obs_seqs, n_state_seqs, n_theta)


def compute_planning_entropy_correction(
    q_xu: Array,
    state_sequences: Array,
    action_sequences: Array,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """
    Compute ∑_{t=1}^T H[q(x_{t-1}, u_t)] - H[q(x_{t-1})].
    """
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


def compute_epistemic_prior_u(q_yxu_theta: Array) -> tuple[Array, Array]:
    """
    Compute epistemic control prior: p̃(u) ∝ exp(H[q(x|u)])
    Prefer actions with high state entropy (uncertain outcomes).
    
    Returns:
        (q_u, log_prior): marginal q(u) and log epistemic prior per action sequence
    """
    # Get marginal q(x, u) by summing over y and θ
    q_xu = jnp.sum(q_yxu_theta, axis=(0, 3))  # (n_state_seqs, n_action_seqs)
    q_u = jnp.sum(q_xu, axis=0)  # (n_action_seqs,)
    
    # For each action sequence, compute H[q(x|u)]
    q_x_given_u = q_xu / (q_u[None, :] + EPS)  # (n_state_seqs, n_action_seqs)
    
    # Entropy H[q(x|u)] for each action sequence
    h_x_given_u = -jnp.sum(q_x_given_u * jnp.log(q_x_given_u + EPS), axis=0)  # (n_action_seqs,)
    
    # Prior: p̃(u) ∝ exp(H[q(x|u)])
    control_prior = softmax(h_x_given_u)
    log_control_prior = jnp.log(control_prior + EPS)
    
    return q_u, log_control_prior


def compute_epistemic_prior_x(
    q_yxu_theta: Array,
    state_sequences: Array,
    observation_tensor: Array,
    n_obs: int,
    n_states: int,
    n_theta: int,
) -> tuple[Array, Array]:
    """
    Compute epistemic state prior: p̃(x) ∝ exp(-H[q(y|x)])
    
    Uniform because when marginalizing out θ, q(y|x) = [0.5, 0.5] for all states,
    so H[q(y|x)] is constant and p̃(x) is uniform.
    """
    n_state_seqs = state_sequences.shape[0]
    q_x = jnp.sum(q_yxu_theta, axis=(0, 2, 3))  # (n_state_seqs,)
    log_prior_per_seq = jnp.zeros(n_state_seqs)
    return q_x, log_prior_per_seq


def compute_epistemic_prior_yx(q_yxu_theta: Array) -> tuple[Array, Array]:
    """
    Compute epistemic (y,x) prior: p̃(y,x) ∝ exp(KL[q(θ|y,x) || q(θ|x)])
    Prefer (y,x) pairs that maximally update beliefs about θ (information gain).
    
    Returns:
        (q_yx, log_prior): marginal q(y,x) and log epistemic prior per (y,x) pair
    """
    # Get marginals
    q_yx_theta = jnp.sum(q_yxu_theta, axis=2)  # (n_obs_seqs, n_state_seqs, n_theta)
    q_yx = jnp.sum(q_yx_theta, axis=2)  # (n_obs_seqs, n_state_seqs)
    q_x_theta = jnp.sum(q_yx_theta, axis=0)  # (n_state_seqs, n_theta)
    q_x = jnp.sum(q_x_theta, axis=1)  # (n_state_seqs,)
    
    # q(θ|y,x) = q(y,x,θ) / q(y,x)
    q_theta_given_yx = q_yx_theta / (q_yx[:, :, None] + EPS)  # (n_obs_seqs, n_state_seqs, n_theta)
    
    # q(θ|x) = q(x,θ) / q(x)
    q_theta_given_x = q_x_theta / (q_x[:, None] + EPS)  # (n_state_seqs, n_theta)
    
    # KL[q(θ|y,x) || q(θ|x)] for each (y,x) pair
    # KL = sum_θ q(θ|y,x) * log(q(θ|y,x) / q(θ|x))
    log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(q_theta_given_x[None, :, :] + EPS)
    kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=2)  # (n_obs_seqs, n_state_seqs)
    
    # Prior: p̃(y,x) ∝ exp(KL[q(θ|y,x) || q(θ|x)])
    yx_prior = softmax(kl_yx.flatten()).reshape(kl_yx.shape)
    log_yx_prior = jnp.log(yx_prior + EPS)
    
    return q_yx, log_yx_prior


def full_joint_vfe(
    q_logits: Array,
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
    Compute VFE with full joint q(y_{1:T}, x_{1:T}, u_{1:T}, θ).
    
    Args:
        q_logits: Logits for q, shape (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
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
    
    # q(y, x, u, θ) from logits
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    
    # ========== Compute marginals ==========
    q_xu = jnp.sum(q_yxu_theta, axis=(0, 3))      # (n_state_seqs, n_action_seqs)
    q_x_theta = jnp.sum(q_yxu_theta, axis=(0, 2)) # (n_state_seqs, n_theta)
    q_u = jnp.sum(q_xu, axis=0)                   # (n_action_seqs,)
    q_theta = jnp.sum(q_x_theta, axis=0)          # (n_theta,)
    
    # ========== Entropy: -H[q] ==========
    neg_entropy = jnp.sum(q_yxu_theta * jnp.log(q_yxu_theta + EPS))
    
    # ========== Action prior energy: E_q[-log p(u)] ==========
    log_prior_per_action_seq = jnp.sum(jnp.log(action_prior[action_sequences] + EPS), axis=1)
    action_energy = -jnp.sum(q_u * log_prior_per_action_seq)
    
    # ========== Transition energy: E_q[-log p(x|x_prev, u)] ==========
    log_transition_probs = compute_transition_log_probs(
        initial_state_idx, state_sequences, action_sequences, transition_tensor
    )
    transition_energy = -jnp.sum(q_xu * log_transition_probs)
    
    # ========== Observation likelihood energy: E_q[-log p(y|x, θ)] ==========
    log_obs_probs = compute_observation_log_probs(
        obs_sequences, state_sequences, observation_tensor
    )  # (n_obs_seqs, n_state_seqs, n_theta)
    q_yx_theta = jnp.sum(q_yxu_theta, axis=2)  # (n_obs_seqs, n_state_seqs, n_theta)
    obs_energy = -jnp.sum(q_yx_theta * log_obs_probs)
    
    # ========== Goal energy: E_q[-log p(goal|x_T, θ)] ==========
    final_states = state_sequences[:, -1]
    log_goal = jnp.log(goal_mapping + EPS)
    log_goal_per_x_theta = log_goal[final_states, :]  # (n_state_seqs, n_theta)
    goal_energy = -jnp.sum(q_x_theta * log_goal_per_x_theta)
    
    # ========== Parameter prior energy: E_q[-log p(θ)] ==========
    # theta_prior is passed as argument (updated based on observations)
    theta_energy = -jnp.sum(q_theta * jnp.log(theta_prior + EPS))
    
    # ========== Epistemic priors / planning correction ==========
    epistemic_u_energy = 0.0
    epistemic_x_energy = 0.0
    epistemic_yx_energy = 0.0
    planning_correction = 0.0
    
    if inference_mode == "active":
        # p̃(u) ∝ exp(H[q(x|u)]) - prefer actions with uncertain outcomes
        q_u_marg, log_prior_u = compute_epistemic_prior_u(q_yxu_theta)
        epistemic_u_energy = -jnp.sum(q_u_marg * log_prior_u)
        
        # p̃(x) ∝ exp(-H[q(y|x)])
        q_x_marg, log_prior_x = compute_epistemic_prior_x(
            q_yxu_theta, state_sequences, observation_tensor, n_obs, n_states, n_theta
        )
        epistemic_x_energy = -jnp.sum(q_x_marg * log_prior_x)
        
        # p̃(y,x) ∝ exp(KL[q(θ|y,x) || q(θ|x)]) - prefer (y,x) that update θ beliefs
        q_yx_marg, log_prior_yx = compute_epistemic_prior_yx(q_yxu_theta)
        epistemic_yx_energy = -jnp.sum(q_yx_marg * log_prior_yx)
    
    elif inference_mode == "planning":
        planning_correction = compute_planning_entropy_correction(
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


def extract_first_action_marginal(
    q_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """Extract q(u_1) from q(y_{1:T}, x_{1:T}, u_{1:T}, θ)."""
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    q_u = jnp.sum(q_yxu_theta, axis=(0, 1, 3))  # (n_action_seqs,)
    
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    first_actions = action_sequences[:, 0]
    
    q_first_action = jnp.zeros(n_actions)
    for a in range(n_actions):
        mask = (first_actions == a).astype(jnp.float32)
        q_first_action = q_first_action.at[a].set(jnp.sum(q_u * mask))
    
    return q_first_action


def extract_reward_location_marginal(
    q_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """Extract q(θ) from q(y, x, u, θ)."""
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    return jnp.sum(q_yxu_theta, axis=(0, 1, 2))


def extract_all_action_marginals(
    q_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """
    Extract q(u_t) for all t = 1, ..., T from q(y_{1:T}, x_{1:T}, u_{1:T}, θ).
    
    Returns:
        Array of shape (horizon, n_actions) where [t, a] = q(u_{t+1} = a).
    """
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    q_u = jnp.sum(q_yxu_theta, axis=(0, 1, 3))  # (n_action_seqs,)
    
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    
    # Extract marginal for each time step
    all_marginals = []
    for t in range(horizon):
        actions_t = action_sequences[:, t]
        q_action_t = jnp.zeros(n_actions)
        for a in range(n_actions):
            mask = (actions_t == a).astype(jnp.float32)
            q_action_t = q_action_t.at[a].set(jnp.sum(q_u * mask))
        all_marginals.append(q_action_t)
    
    return jnp.stack(all_marginals, axis=0)


def extract_all_state_marginals(
    q_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """
    Extract q(x_t) for all t = 1, ..., T from q(y_{1:T}, x_{1:T}, u_{1:T}, θ).
    
    Returns:
        Array of shape (horizon, n_states) where [t, s] = q(x_{t+1} = s).
    """
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    q_x = jnp.sum(q_yxu_theta, axis=(0, 2, 3))  # (n_state_seqs,)
    
    state_sequences = enumerate_state_sequences(n_states, horizon)
    
    # Extract marginal for each time step
    all_marginals = []
    for t in range(horizon):
        states_t = state_sequences[:, t]
        q_state_t = jnp.zeros(n_states)
        for s in range(n_states):
            mask = (states_t == s).astype(jnp.float32)
            q_state_t = q_state_t.at[s].set(jnp.sum(q_x * mask))
        all_marginals.append(q_state_t)
    
    return jnp.stack(all_marginals, axis=0)


def extract_all_obs_marginals(
    q_logits: Array,
    n_obs: int,
    n_states: int,
    n_actions: int,
    n_theta: int,
    horizon: int,
) -> Array:
    """
    Extract q(y_t) for all t = 1, ..., T from q(y_{1:T}, x_{1:T}, u_{1:T}, θ).
    
    Returns:
        Array of shape (horizon, n_obs) where [t, o] = q(y_{t+1} = o).
    """
    n_obs_seqs = n_obs ** horizon
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    q_yxu_theta = softmax(q_logits.flatten()).reshape(
        (n_obs_seqs, n_state_seqs, n_action_seqs, n_theta)
    )
    q_y = jnp.sum(q_yxu_theta, axis=(1, 2, 3))  # (n_obs_seqs,)
    
    obs_sequences = enumerate_obs_sequences(n_obs, horizon)
    
    # Extract marginal for each time step
    all_marginals = []
    for t in range(horizon):
        obs_t = obs_sequences[:, t]
        q_obs_t = jnp.zeros(n_obs)
        for o in range(n_obs):
            mask = (obs_t == o).astype(jnp.float32)
            q_obs_t = q_obs_t.at[o].set(jnp.sum(q_y * mask))
        all_marginals.append(q_obs_t)
    
    return jnp.stack(all_marginals, axis=0)
