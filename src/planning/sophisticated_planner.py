"""
Sophisticated Active Inference planner using pymdp.

This module provides a wrapper around pymdp's Agent class for
"sophisticated" active inference - tree-search based planning
with Expected Free Energy (EFE).

This serves as a baseline implementation without novel entropy
corrections, using standard active inference as implemented in pymdp.

Reference:
    https://pymdp-rtd.readthedocs.io/en/stable/notebooks/cue_chaining_demo.html
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from jax import Array

from pymdp.agent import Agent
from pymdp import control as pymdp_control
from pymdp import utils as pymdp_utils


@dataclass
class SophisticatedPlanningConfig:
    """Configuration for pymdp-based planning (sophisticated or vanilla)."""
    planning_horizon: int = 7
    n_states: int = 35
    n_actions: int = 8
    n_theta: int = 2
    policy_len: int = 1  # Length of action sequences in policies
    inference_horizon: int = 7  # Tree-search depth for planning (sophisticated mode)
    use_utility: bool = True  # Include extrinsic value (goal-seeking)
    use_states_info_gain: bool = True  # Include epistemic value (state info gain on x)
    use_param_info_gain: bool = True  # Include epistemic value (parameter info gain on theta)
    action_selection: str = "deterministic"  # "deterministic" or "stochastic"
    gamma: float = 16.0  # Precision for action selection
    sophisticated: bool = True  # If True, use tree-search; if False, use vanilla EFE
    include_reward_modality: bool = True  # If True, add reward observation modality
    goal_temperature: float = 1.0  # Scaling factor for goal preferences in C matrix


@dataclass
class SophisticatedPlanningResult:
    """Result of sophisticated planning."""
    q_pi: np.ndarray  # Policy distribution (n_actions,) for first action
    q_s: List[np.ndarray]  # State beliefs per factor
    q_theta: np.ndarray  # Parameter (context) belief (n_theta,)
    efe: np.ndarray  # Expected free energy per action
    selected_action: int


def convert_tensors_to_pymdp(
    transition_tensor: Array,
    theta_observation_tensor: Array,
    location_observation_tensor: Array,
    goal_mapping: Array,
    theta_prior: Array,
    n_theta: int,
    include_reward_modality: bool = True,
    goal_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert epistemic maze tensors to pymdp format.

    pymdp conventions:
    - A matrices: obj_array of observation likelihood per modality
      A[m] has shape (num_obs[m], num_states[0], ..., num_states[F-1])
    - B matrices: obj_array of transition matrices per state factor
      B[f] has shape (num_states[f], num_states[f], num_controls[f])
    - C vectors: obj_array of preference vectors per modality
      C[m] has shape (num_obs[m],) or (num_obs[m], T)
    - D vectors: obj_array of prior state vectors per factor
      D[f] has shape (num_states[f],)

        For epistemic maze with hidden theta:
    - We treat theta as a second state factor that doesn't transition
    - State factors: [location_x_knob (35 states), theta (n_theta states)]
        - Observation modalities: [location (7 obs), theta_obs (n_theta+1 obs)]
            Optionally add reward modality if include_reward_modality=True

    The reward modality (optional) is key for goal-seeking behavior (following pymdp T-maze demo).
    It emits: 0=neutral, 1=reward (correct goal), 2=loss (wrong goal), 3=sink (safe sink)

    Args:
        transition_tensor: p(s'|s,a) shape (35, 35, 8)
        theta_observation_tensor: p(o_theta|s, theta) shape (n_theta+1, 35, n_theta)
        location_observation_tensor: p(o_loc|s) shape (7, 35)
        goal_mapping: p(goal|s, theta) shape (35, n_theta) - soft preferences
        theta_prior: p(theta) shape (n_theta,)
        n_theta: Number of context values

    Returns:
        A, B, C, D in pymdp obj_array format
    """
    n_states = transition_tensor.shape[0]  # 35
    n_actions = transition_tensor.shape[2]  # 8
    n_locations = location_observation_tensor.shape[0]  # 7
    n_theta_obs = theta_observation_tensor.shape[0]  # n_theta + 1

    # Convert JAX arrays to numpy
    transition_np = np.array(transition_tensor)
    theta_obs_np = np.array(theta_observation_tensor)
    location_obs_np = np.array(location_observation_tensor)
    goal_mapping_np = np.array(goal_mapping)
    theta_prior_np = np.array(theta_prior)

    # ==================== A MATRICES (Observation Likelihood) ====================
    # We have TWO observation modalities by default:
    # 1. Location observations: depend only on state (location x knob)
    # 2. Theta observations: depend on state AND theta
    # Optionally add reward observations as a third modality.

    n_modalities = 3 if include_reward_modality else 2
    A = pymdp_utils.obj_array(n_modalities)

    # A[0]: Location observations - shape (n_locations, n_states, n_theta)
    # Location obs are independent of theta, so replicate across theta dimension
    A[0] = np.zeros((n_locations, n_states, n_theta))
    for theta_idx in range(n_theta):
        A[0][:, :, theta_idx] = location_obs_np

    # A[1]: Theta observations - shape (n_theta_obs, n_states, n_theta)
    # Already in correct format
    A[1] = theta_obs_np.copy()

    if include_reward_modality:
        # A[2]: Reward observations - shape (4, n_states, n_theta)
        # 0 = no reward, 1 = big reward, 2 = loss, 3 = small reward (sink)
        # This encodes the goal structure: big reward at (location=theta, knob=4)
        n_reward_obs = 4
        A[2] = np.zeros((n_reward_obs, n_states, n_theta))

        for s in range(n_states):
            loc = s % 7  # location = state % N_LOCATIONS
            knob = s // 7  # knob = state // N_LOCATIONS

            for theta_idx in range(n_theta):
                if loc == 5:  # SAFE_SINK
                    # Safe sink: small reward observation
                    A[2][3, s, theta_idx] = 1.0  # small reward
                elif loc < 5 and knob == 4:  # Nav state with max knob
                    if loc == theta_idx:
                        # Correct goal: big reward observation
                        A[2][1, s, theta_idx] = 1.0  # big reward
                    else:
                        # Wrong goal: loss observation
                        A[2][2, s, theta_idx] = 1.0  # loss
                else:
                    # All other states: no reward
                    A[2][0, s, theta_idx] = 1.0  # no reward

    # ==================== B MATRICES (Transition Dynamics) ====================
    # We have two state factors:
    # 1. Location x Knob (35 states) - transitions based on action
    # 2. Theta (n_theta states) - never transitions (hidden context)

    B = pymdp_utils.obj_array(2)

    # B[0]: State transitions - shape (n_states, n_states, n_actions)
    # pymdp convention is (next, current, action) - same as ours
    B[0] = transition_np.copy()

    # B[1]: Theta transitions - identity (theta never changes)
    # Shape: (n_theta, n_theta, 1) - only ONE "control" for uncontrollable factor
    # Following pymdp convention: uncontrollable factors have num_controls=1
    B[1] = np.zeros((n_theta, n_theta, 1))
    B[1][:, :, 0] = np.eye(n_theta)

    # ==================== C VECTORS (Preferences) ====================
    # Preferences over observations - THIS IS KEY FOR GOAL-SEEKING
    # Following the pymdp T-maze demo pattern

    C = pymdp_utils.obj_array(n_modalities)

    # C[0]: Location observation preferences - neutral
    C[0] = np.zeros(n_locations)

    # C[1]: Theta observation preferences - neutral (epistemic drive handles this)
    C[1] = np.zeros(n_theta_obs)

    if include_reward_modality:
        # C[2]: Reward observation preferences - explicit log-preferences
        # Outcomes: 0=no reward, 1=big reward, 2=loss, 3=small reward (sink)
        # Scale by goal_temperature to control preference strength
        C[2] = np.zeros(n_reward_obs)
        C[2][0] = 0.0 
        C[2][1] = goal_temperature * 1.0
        C[2][2] = -goal_temperature * 1.0
        C[2][3] = goal_temperature * 0.33

    # ==================== D VECTORS (State Priors) ====================
    # Prior beliefs about initial states

    D = pymdp_utils.obj_array(2)

    # D[0]: Uniform over states (will be overridden by actual belief)
    D[0] = np.zeros(n_states)
    for loc in range(7):
        s = loc + 7 * 4
        D[0][s] = 1.0 / 7

    # D[1]: Prior over theta
    D[1] = theta_prior_np.copy()

    return A, B, C, D


def create_goal_preferences_from_mapping(
    goal_mapping: Array,
    n_theta: int,
) -> List[np.ndarray]:
    """
    Create state preferences from goal mapping for pymdp.

    pymdp uses C matrices for observation preferences, but we can also
    set state preferences via the E matrix (habit/prior over policies)
    or by biasing the agent's beliefs.

    For now, we encode goals through the EFE computation by setting
    preferences over final states.

    Args:
        goal_mapping: p(goal|s, theta) shape (n_states, n_theta)
        n_theta: Number of context values

    Returns:
        List of preference arrays that can bias action selection
    """
    goal_np = np.array(goal_mapping)

    # For each theta, the goal_mapping gives soft preferences over states
    # Higher values = more preferred states
    # We can use this to construct C matrices or bias the planning

    # Log preferences (utility in active inference)
    log_prefs = np.log(goal_np + 1e-8)

    return [log_prefs[:, theta_idx] for theta_idx in range(n_theta)]


class SophisticatedPlanner:
    """
    Wrapper for pymdp Agent that implements sophisticated active inference.

    This planner uses expected free energy (EFE) minimization via tree search,
    which is the standard "sophisticated" active inference approach.

    No additional entropy corrections are applied - this is a baseline
    implementation for comparison with novel methods.
    """

    def __init__(
        self,
        transition_tensor: Array,
        theta_observation_tensor: Array,
        location_observation_tensor: Array,
        goal_mapping: Array,
        action_prior: Array,
        theta_prior: Array,
        config: SophisticatedPlanningConfig,
    ):
        """
        Initialize the sophisticated planner.

        Args:
            transition_tensor: p(s'|s,a) shape (n_states, n_states, n_actions)
            theta_observation_tensor: p(o_theta|s, theta) shape (n_theta_obs, n_states, n_theta)
            location_observation_tensor: p(o_loc|s) shape (n_locations, n_states)
            goal_mapping: p(goal|s, theta) shape (n_states, n_theta)
            action_prior: p(a) shape (n_actions,)
            theta_prior: p(theta) shape (n_theta,)
            config: Planning configuration
        """
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.n_theta = config.n_theta

        # Store original tensors for reference
        self.goal_mapping = np.array(goal_mapping)
        self.action_prior = np.array(action_prior)

        # Convert to pymdp format
        self.A, self.B, self.C, self.D = convert_tensors_to_pymdp(
            transition_tensor=transition_tensor,
            theta_observation_tensor=theta_observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            theta_prior=theta_prior,
            n_theta=config.n_theta,
            include_reward_modality=config.include_reward_modality,
            goal_temperature=config.goal_temperature,
        )

        # Construct policies and policy prior from action_prior (no learning)
        control_fac_idx = [0]  # Only factor 0 (state) is controllable
        num_controls = [self.B[0].shape[-1], self.B[1].shape[-1]]
        policies = pymdp_control.construct_policies(
            [self.n_states, self.n_theta],
            num_controls,
            policy_len=config.policy_len,
            control_fac_idx=control_fac_idx,
        )

        # Flat (uniform) policy prior - let EFE drive action selection
        E = np.ones(len(policies)) / len(policies)

        # Create pymdp agent
        # Following pymdp cue_chaining_demo pattern:
        # - Agent infers num_controls from B matrix shapes
        # - policy_len=1 for single-step policies with re-planning each step
        # - control_fac_idx specifies controllable factors (only factor 0)
        # - sophisticated=True enables tree-search based planning
        #
        # B matrix shapes encode control structure:
        # - B[0]: (35, 35, 8) - 8 actions for state factor
        # - B[1]: (n_theta, n_theta, 1) - 1 "no-op" for theta factor
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=E,
            control_fac_idx=control_fac_idx,
            policies=policies,
            policy_len=config.policy_len,
            inference_horizon=config.inference_horizon,
            gamma=config.gamma,
            use_utility=config.use_utility,
            use_states_info_gain=config.use_states_info_gain,
            use_param_info_gain=config.use_param_info_gain,
            action_selection=config.action_selection,
            sophisticated=config.sophisticated,  # Tree-search (True) or vanilla EFE (False)
            si_horizon=config.inference_horizon if config.sophisticated else 1,
            si_policy_prune_threshold=1e-1,
            si_state_prune_threshold=1e-1,
        )

    def rebuild_policies(self, new_policy_len: int) -> None:
        """
        Rebuild policies with a new policy length (for receding horizon control).

        This regenerates the policy space and updates the agent's policy prior.

        Args:
            new_policy_len: New policy length (planning horizon)
        """
        if new_policy_len == self.agent.policy_len and self.agent.policies is not None:
            return  # No change needed

        control_fac_idx = [0]  # Only factor 0 (state) is controllable
        num_controls = [self.B[0].shape[-1], self.B[1].shape[-1]]

        # Regenerate policies with new length
        new_policies = pymdp_control.construct_policies(
            [self.n_states, self.n_theta],
            num_controls,
            policy_len=new_policy_len,
            control_fac_idx=control_fac_idx,
        )

        # Update agent's policies and policy prior
        self.agent.policies = new_policies
        self.agent.policy_len = new_policy_len
        self.agent.E = np.ones(len(new_policies)) / len(new_policies)

    def infer_states(
        self,
        observation: List[int],
    ) -> np.ndarray:
        """
        Update beliefs given observations.

        Following pymdp pattern: call this FIRST each timestep.

        Args:
            observation: List of observation indices [location_obs, theta_obs]
                If config.include_reward_modality=True, then include reward_obs as third entry.

        Returns:
            Updated state beliefs (qs)
        """
        return self.agent.infer_states(observation)

    def infer_policies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute policy distribution via Expected Free Energy.

        Following pymdp pattern: call this AFTER infer_states.

        Returns:
            Tuple of (q_pi, efe) - policy distribution and EFE values
        """
        return self.agent.infer_policies()

    def sample_action(self) -> np.ndarray:
        """
        Sample action from the policy distribution.

        Following pymdp pattern: call this AFTER infer_policies.
        This handles extracting the correct action from multi-step policies.

        Returns:
            Action array (one action per controllable factor)
        """
        return self.agent.sample_action()

    def plan_action(self) -> SophisticatedPlanningResult:
        """
        Plan and select an action using EFE-based policy inference.

        This combines infer_policies() and sample_action() and returns
        a structured result. Call this AFTER infer_states().

        Returns:
            SophisticatedPlanningResult with selected action and beliefs
        """
        # Compute policy distribution via Expected Free Energy
        q_pi, efe = self.infer_policies()

        # Sample action from policy distribution
        action_arr = self.sample_action()
        selected_action = int(action_arr[0])  # First controllable factor

        # Get current beliefs
        q_s = [self.agent.qs[i].copy() for i in range(len(self.agent.qs))]
        q_theta = self.agent.qs[1].copy()

        return SophisticatedPlanningResult(
            q_pi=q_pi,
            q_s=q_s,
            q_theta=q_theta,
            efe=efe,
            selected_action=selected_action,
        )

    def get_beliefs(self) -> SophisticatedPlanningResult:
        """
        Get current beliefs and last planning results.

        Returns:
            SophisticatedPlanningResult with current state
        """
        q_s = [self.agent.qs[i].copy() for i in range(len(self.agent.qs))]
        q_theta = self.agent.qs[1].copy()

        # Get last computed q_pi and efe if available
        q_pi = getattr(self.agent, 'q_pi', np.zeros(self.n_actions))
        efe = getattr(self.agent, 'G', np.zeros(self.n_actions))

        return SophisticatedPlanningResult(
            q_pi=q_pi,
            q_s=q_s,
            q_theta=q_theta,
            efe=efe,
            selected_action=-1,  # Not set until sample_action called
        )

