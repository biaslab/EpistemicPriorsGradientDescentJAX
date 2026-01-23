"""
Variational Free Energy minimization with FULL JOINT factorization.

q(x_{1:T}, u_{1:T}, r) - joint over state trajectories, action trajectories, and reward location.

VFE = -H[q] + E_q[-log p(u)] + E_q[-log p(x|x_prev, u)] + E_q[-log p(goal|x_T, r)] + E_q[-log p(r)]
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

def conditional_entropy_3d(marginal: Array) -> Array:
    """
    Compute conditional entropy for tensor of shape (n_obs, n_states, n_reward).
    
    For a joint/marginal p(obs, state, reward), computes:
    H[obs | reward, state=i] for each state i.
    
    This is: H[obs, reward | state] - H[reward | state]
    
    Args:
        marginal: Array of shape (n_obs, n_states, n_reward_locs)
        
    Returns:
        Array of shape (n_states,) with conditional entropy per state.
    """
    # in_dim = 1 (state), out_dim = 0 (obs)
    # sum_dims = [0, 2] (all except in_dim)
    
    # q_in = p(state) by marginalizing over obs and reward
    q_in = jnp.sum(marginal, axis=(0, 2), keepdims=True)  # (1, n_states, 1)
    
    # q_given_in = p(obs, reward | state)
    q_given_in = marginal / (q_in + EPS)  # (n_obs, n_states, n_reward)
    
    # Joint entropy H[obs, reward | state=i] for each state i
    joint_entropies = -jnp.sum(
        q_given_in * jnp.log(q_given_in + EPS), axis=(0, 2)
    )  # (n_states,)
    
    # Marginalize out obs: p(reward | state)
    q_marginalized = jnp.sum(q_given_in, axis=0, keepdims=True)  # (1, n_states, n_reward)
    
    # Marginal entropy H[reward | state=i] for each state i
    marginal_entropies = -jnp.sum(
        q_marginalized * jnp.log(q_marginalized + EPS), axis=(0, 2)
    )  # (n_states,)
    
    # H[obs | reward, state] = H[obs, reward | state] - H[reward | state]
    return joint_entropies - marginal_entropies

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


def full_joint_vfe(
    q_logits: Array,
    initial_state: Array,
    transition_tensor: Array,
    observation_tensor: Array,
    goal_mapping: Array,
    action_prior: Array,
    horizon: int,
    inference_mode: str = "marginal",
) -> Array:
    """
    Compute vanilla VFE with full joint q(x_{1:T}, u_{1:T}, r).
    
    Args:
        q_logits: Logits for q(x, u, r), shape (n_state_seqs, n_action_seqs, n_reward_locs)
        initial_state: One-hot initial state, shape (n_states,)
        transition_tensor: p(x'|x,u), shape (n_states, n_states, n_actions)
        observation_tensor: p(o|x,r), shape (n_obs, n_states, n_reward_locs)
        goal_mapping: p(goal|x,r), shape (n_states, n_reward_locs)
        action_prior: Action prior, shape (n_actions,)
        horizon: Planning horizon T
        inference_mode: "marginal", "active", or "planning"
        
    Returns:
        Scalar VFE loss.
    """
    n_states = transition_tensor.shape[0]
    n_actions = transition_tensor.shape[2]
    n_reward_locs = goal_mapping.shape[1]
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    
    state_sequences = enumerate_state_sequences(n_states, horizon)
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    initial_state_idx = jnp.argmax(initial_state)
    
    # q(x, u, r) from logits
    q_xur = softmax(q_logits.flatten()).reshape((n_state_seqs, n_action_seqs, n_reward_locs))
    
    # Marginals
    q_xu = jnp.sum(q_xur, axis=2)  # (n_state_seqs, n_action_seqs)
    q_xr = jnp.sum(q_xur, axis=1)  # (n_state_seqs, n_reward_locs)
    q_u = jnp.sum(q_xu, axis=0)    # (n_action_seqs,)
    q_r = jnp.sum(q_xr, axis=0)    # (n_reward_locs,)
    
    # Entropy: -H[q]
    neg_entropy = jnp.sum(q_xur * jnp.log(q_xur + EPS))
    
    # Action prior energy: E_q[-log p(u)]
    log_prior_per_action_seq = jnp.sum(jnp.log(action_prior[action_sequences] + EPS), axis=1)
    action_energy = -jnp.sum(q_u * log_prior_per_action_seq)
    
    # Transition energy: E_q[-log p(x|x_prev, u)]
    log_transition_probs = compute_transition_log_probs(
        initial_state_idx, state_sequences, action_sequences, transition_tensor
    )
    transition_energy = -jnp.sum(q_xu * log_transition_probs)
    
    # Goal energy: E_q[-log p(goal|x_T, r)]
    final_states = state_sequences[:, -1]
    log_goal = jnp.log(goal_mapping + EPS)
    log_goal_per_xr = log_goal[final_states, :]
    goal_energy = -jnp.sum(q_xr * log_goal_per_xr)

    # Epistemic terms based on inference mode
    state_energy = 0.0
    control_energy = 0.0
    planning_correction = 0.0
    
    if inference_mode == "active":
        # State energy: E_q[-log p̃(x)] ∝ exp(-H[q(y|x)]) - prefer states with informative observations
        prior_state = softmax(-conditional_entropy_3d(observation_tensor))
        log_state_prior = jnp.log(prior_state + EPS)
        log_prior_per_state_seq = jnp.sum(log_state_prior[state_sequences], axis=1)
        q_x = jnp.sum(q_xu, axis=1)  # marginal over state sequences
        state_energy = -jnp.sum(q_x * log_prior_per_state_seq)

        # Control energy: E_q[-log p̃(u)] where p̃(u) ∝ exp(H[x_t, x_{t-1} | u] - H[x_{t-1} | u])
        # transition_tensor shape: (n_states, n_states, n_actions) = p(x_t | x_{t-1}, u)
        
        # Form joint q(x_t, x_{t-1} | u) assuming uniform prior on x_{t-1}
        # joint[x_t, x_{t-1}, u] = p(x_t | x_{t-1}, u) * (1/n_states)
        joint_given_u = transition_tensor / n_states  # (n_states, n_states, n_actions)
        
        # H[q(x_t, x_{t-1} | u)] for each u - entropy of the joint
        joint_entropy_per_u = -jnp.sum(
            joint_given_u * jnp.log(joint_given_u + EPS),
            axis=(0, 1)
        )  # (n_actions,)
        
        # Marginalize out x_t to get q(x_{t-1} | u)
        marginal_xt1_given_u = jnp.sum(joint_given_u, axis=0)  # (n_states, n_actions)
        
        # H[q(x_{t-1} | u)] for each u
        marginal_entropy_per_u = -jnp.sum(
            marginal_xt1_given_u * jnp.log(marginal_xt1_given_u + EPS),
            axis=0
        )  # (n_actions,)
        
        # Conditional entropy: H[x_t | x_{t-1}, u] = H[x_t, x_{t-1} | u] - H[x_{t-1} | u]
        control_cond_entropy = joint_entropy_per_u - marginal_entropy_per_u  # (n_actions,)
        
        # Control prior: p̃(u) ∝ exp(H[x_t | x_{t-1}, u])
        control_prior = softmax(control_cond_entropy)
        
        # Apply prior to action sequences (sum log prior over timesteps)
        log_control_prior = jnp.log(control_prior + EPS)
        log_prior_per_action_seq = jnp.sum(log_control_prior[action_sequences], axis=1)
        control_energy = -jnp.sum(q_u * log_prior_per_action_seq)
    
    elif inference_mode == "planning":
        planning_correction = compute_planning_entropy_correction(
            q_xu, state_sequences, action_sequences, n_states, n_actions, horizon
        )
    
    # Reward prior energy: E_q[-log p(r)]
    reward_prior = jnp.ones(n_reward_locs) / n_reward_locs
    reward_energy = -jnp.sum(q_r * jnp.log(reward_prior + EPS))
    
    return neg_entropy + action_energy + transition_energy + goal_energy + reward_energy + state_energy + control_energy + planning_correction


def extract_first_action_marginal(
    q_logits: Array,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """Extract q(u_1) from q(x_{1:T}, u_{1:T}, r)."""
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    n_reward_locs = q_logits.shape[2]
    
    q_xur = softmax(q_logits.flatten()).reshape((n_state_seqs, n_action_seqs, n_reward_locs))
    q_u = jnp.sum(q_xur, axis=(0, 2))
    
    action_sequences = enumerate_action_sequences(n_actions, horizon)
    first_actions = action_sequences[:, 0]
    
    q_first_action = jnp.zeros(n_actions)
    for a in range(n_actions):
        mask = (first_actions == a).astype(jnp.float32)
        q_first_action = q_first_action.at[a].set(jnp.sum(q_u * mask))
    
    return q_first_action


def extract_reward_location_marginal(
    q_logits: Array,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """Extract q(r) from q(x, u, r)."""
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    n_reward_locs = q_logits.shape[2]
    
    q_xur = softmax(q_logits.flatten()).reshape((n_state_seqs, n_action_seqs, n_reward_locs))
    return jnp.sum(q_xur, axis=(0, 1))


def extract_all_action_marginals(
    q_logits: Array,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """
    Extract q(u_t) for all t = 1, ..., T from q(x_{1:T}, u_{1:T}, r).
    
    Returns:
        Array of shape (horizon, n_actions) where [t, a] = q(u_{t+1} = a).
    """
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    n_reward_locs = q_logits.shape[2]
    
    q_xur = softmax(q_logits.flatten()).reshape((n_state_seqs, n_action_seqs, n_reward_locs))
    q_u = jnp.sum(q_xur, axis=(0, 2))  # (n_action_seqs,)
    
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
    n_states: int,
    n_actions: int,
    horizon: int,
) -> Array:
    """
    Extract q(x_t) for all t = 1, ..., T from q(x_{1:T}, u_{1:T}, r).
    
    Returns:
        Array of shape (horizon, n_states) where [t, s] = q(x_{t+1} = s).
    """
    n_state_seqs = n_states ** horizon
    n_action_seqs = n_actions ** horizon
    n_reward_locs = q_logits.shape[2]
    
    q_xur = softmax(q_logits.flatten()).reshape((n_state_seqs, n_action_seqs, n_reward_locs))
    q_x = jnp.sum(q_xur, axis=(1, 2))  # (n_state_seqs,)
    
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
