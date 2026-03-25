"""
Epistemic Maze environment for Active Inference experiments.

This implements the Minimal Reactive Epistemic Maze with multi-goal epistemic uncertainty.
The environment demonstrates that both reactive behavior and epistemic learning
are necessary and sufficient for optimal performance.

State Space:
    - Entity 1: Location (7 states)
        - States 0-4: Navigation states (goal is θ-dependent)
        - State 5: Safe Sink (guaranteed 0.33 reward)
        - State 6: Instructional Cue (reveals θ)
    - Entity 2: Reactivity Knob (5 states)
        - State 0: Minimum reactivity (no control, but safe sink guaranteed)
        - State 4: Maximum reactivity (full deterministic control)
    - Total: 7 × 5 = 35 hidden states

Action Space (8 actions):
    - Actions 0-4: Navigate (state-dependent: (location + action) mod 5)
    - Action 5: Decrease reactivity knob
    - Action 6: Increase reactivity knob
    - Action 7: Visit instructional cue (costly, reveals θ)

Context Parameter θ:
    - θ ∈ {0, 1, ..., n_theta-1} determines which navigation state is optimal
    - When θ = i, the optimal goal is at navigation state i

Reward Structure (only at final timestep):
    - +1.0: At correct goal (location = θ) with max reactivity (knob = 4)
    - +0.33: At safe sink (location = 5), regardless of reactivity
    - -1.0: At wrong navigation state with max reactivity (knob = 4)
    - -0.33: At any non-sink state with non-max reactivity (knob < 4)
    - 0.0: Otherwise
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional
import random

import jax
import jax.numpy as jnp
from jax import Array


class EpistemicAction(IntEnum):
    """Actions available in the Epistemic Maze."""
    NAV_0 = 0  # Navigate: (loc + 0) mod 5
    NAV_1 = 1  # Navigate: (loc + 1) mod 5
    NAV_2 = 2  # Navigate: (loc + 2) mod 5
    NAV_3 = 3  # Navigate: (loc + 3) mod 5
    NAV_4 = 4  # Navigate: (loc + 4) mod 5
    DECREASE_KNOB = 5  # Decrease reactivity
    INCREASE_KNOB = 6  # Increase reactivity
    VISIT_CUE = 7  # Visit instructional cue


class Location(IntEnum):
    """Location states in the Epistemic Maze."""
    NAV_0 = 0
    NAV_1 = 1
    NAV_2 = 2
    NAV_3 = 3
    NAV_4 = 4
    SAFE_SINK = 5
    CUE = 6


# Environment constants
N_LOCATIONS = 7
N_KNOB_STATES = 5
N_STATES = N_LOCATIONS * N_KNOB_STATES  # 35
N_ACTIONS = 8
N_NAV_STATES = 5  # Navigation states 0-4


def state_to_components(state_idx: int) -> Tuple[int, int]:
    """
    Convert flat state index to (location, knob) tuple.

    Encoding: state_idx = location + N_LOCATIONS * knob
    """
    location = state_idx % N_LOCATIONS
    knob = state_idx // N_LOCATIONS
    return location, knob


def components_to_state(location: int, knob: int) -> int:
    """Convert (location, knob) tuple to flat state index."""
    return location + N_LOCATIONS * knob


@dataclass
class EpistemicMaze:
    """
    Epistemic Maze environment with reactivity dynamics.

    Attributes:
        location: Current location (0-6)
        knob: Reactivity knob value (0-4)
        theta: True goal location (0 to n_theta-1)
        _has_seen_cue: Whether agent has visited the cue location
        n_theta: Number of possible theta values
        cue_accuracy: Probability of correct cue observation at the cue
        location_accuracy: Probability of correct location observation
    """
    location: int
    knob: int
    theta: int
    n_theta: int = 6
    cue_accuracy: float = 1.00
    location_accuracy: float = 0.90
    _has_seen_cue: bool = False

    @classmethod
    def create(
        cls,
        theta: Optional[int] = None,
        start_location: Optional[int] = None,
        n_theta: int = 5,
        cue_accuracy: float = 1.0,
        location_accuracy: float = 0.90,
    ) -> "EpistemicMaze":
        """
        Create a new Epistemic Maze environment.

        Args:
            theta: Goal location (0 to n_theta-1). If None, chosen randomly.
            start_location: Starting location (0-6). If None, chosen randomly from nav states (0-4).
            n_theta: Number of possible theta values (2 or 5).
            cue_accuracy: Probability of correct cue observation at the cue.
            location_accuracy: Probability of correct location observation.

        Returns:
            New EpistemicMaze instance.
        """
        if theta is None:
            theta = random.randint(0, n_theta - 1)

        if not 0 <= theta < n_theta:
            raise ValueError(f"theta must be 0 to {n_theta-1}, got {theta}")

        if start_location is None:
            start_location = random.randint(0, N_NAV_STATES - 1)

        if not 0 <= start_location < N_LOCATIONS:
            raise ValueError(f"start_location must be 0-6, got {start_location}")
        
        # If starting at CUE, mark as already seen
        has_seen_cue = (start_location == Location.CUE)

        return cls(
            location=start_location,
            knob=4,  # Always start at maximum reactivity
            theta=theta,
            n_theta=n_theta,
            cue_accuracy=cue_accuracy,
            location_accuracy=location_accuracy,
            _has_seen_cue=has_seen_cue,
        )

    def reset(
        self,
        theta: Optional[int] = None,
        start_location: Optional[int] = None,
    ) -> "EpistemicMaze":
        """Reset the environment to initial state."""
        if theta is None:
            theta = random.randint(0, self.n_theta - 1)

        if start_location is None:
            start_location = random.randint(0, N_NAV_STATES - 1)

        self.location = start_location
        self.knob = 4  # Reset to max reactivity
        self.theta = theta
        self._has_seen_cue = False
        return self

    @property
    def state_idx(self) -> int:
        """Get flat state index."""
        return components_to_state(self.location, self.knob)

    def step(self, action: int) -> Tuple[Array, Array, float, bool]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (0-7).

        Returns:
            Tuple of (location_obs, context_obs, reward, done).
            - location_obs: Location observation probabilities
            - context_obs: Context observation probabilities
            - reward: Always 0.0 (reward only at terminal)
            - done: Always False (use get_final_reward at horizon)
        """
        self._execute_transition(action)

        # Check if at cue location
        if self.location == Location.CUE:
            self._has_seen_cue = True

        # Generate location and context observations
        location_obs = self._get_location_observation()
        context_obs = self._get_context_observation()

        # Reward is 0 during episode (only delivered at final timestep)
        return location_obs, context_obs, 0.0, False

    def _get_location_observation(self) -> Array:
        """
        Get location observation.

        Reveals current location with `location_accuracy`, uniform otherwise.
        """
        # Observations over N_LOCATIONS (0-6)
        obs = jnp.zeros(N_LOCATIONS)
        obs = obs.at[self.location].set(self.location_accuracy)
        # Spread remaining probability uniformly over other locations
        other_prob = (1.0 - self.location_accuracy) / (N_LOCATIONS - 1)
        for i in range(N_LOCATIONS):
            if i != self.location:
                obs = obs.at[i].set(other_prob)
        return obs

    def _execute_transition(self, action: int) -> None:
        """Execute state transition based on action."""
        # Handle cue state - any action returns to random nav state
        if self.location == Location.CUE:
            self.location = random.randint(0, N_NAV_STATES - 1)
            return

        # Handle safe sink - absorbing state
        if self.location == Location.SAFE_SINK:
            return

        # Handle knob decrease (action 5): behave like nav action 0 (stay location)
        if action == EpistemicAction.DECREASE_KNOB:
            new_knob = max(0, self.knob - 1)
            success_prob = self.knob / 4.0
            if random.random() < success_prob:
                # Successful: stay at same location, decrease knob
                self.knob = new_knob
            else:
                # Failure: fall to safe sink
                self.knob = new_knob
                self.location = Location.SAFE_SINK
            return

        # Handle knob increase (action 6): behave like nav action 0 (stay location)
        if action == EpistemicAction.INCREASE_KNOB:
            new_knob = min(4, self.knob + 1)
            success_prob = self.knob / 4.0
            if random.random() < success_prob:
                # Successful: stay at same location, increase knob
                self.knob = new_knob
            else:
                # Failure: fall to safe sink
                self.knob = new_knob
                self.location = Location.SAFE_SINK
            return

        # Handle visit cue (action 7)
        if action == EpistemicAction.VISIT_CUE:
            self.location = Location.CUE
            return

        # Handle navigation actions (0-4)
        if 0 <= action <= 4:
            success_prob = self.knob / 4.0
            if random.random() < success_prob:
                # Successful navigation: (location + action) mod 5
                self.location = (self.location + action) % N_NAV_STATES
            else:
                # Fall to safe sink
                self.location = Location.SAFE_SINK

    def _get_context_observation(self) -> Array:
        """
        Get context observation.

        At cue location: Reveals theta with `cue_accuracy`.
        Elsewhere: Deterministic neutral observation (unambiguous not-at-cue).
        """
        n_obs = self.n_theta + 1  # Context observations + neutral

        if self.location == Location.CUE:
            # Informative observation - high probability on true theta
            obs = jnp.zeros(n_obs)
            obs = obs.at[self.theta].set(self.cue_accuracy)
            # Spread remaining probability among other context observations
            other_prob = (1.0 - self.cue_accuracy) / (self.n_theta - 1) if self.n_theta > 1 else 0.0
            for i in range(self.n_theta):
                if i != self.theta:
                    obs = obs.at[i].set(other_prob)
            return obs
        else:
            # Deterministic neutral observation (index n_theta)
            obs = jnp.zeros(n_obs)
            return obs.at[self.n_theta].set(1.0)

    def get_final_reward(self) -> float:
        """
        Get reward at terminal state.

        Returns:
            +1.0 if at correct goal (location = θ) with max reactivity (knob = 4)
            +0.33 if at safe sink (location = 5)
            -1.0 if at wrong navigation state with max reactivity
            -0.33 if at non-sink state with knob < 4
            0.0 otherwise
        """
        if self.location == Location.SAFE_SINK:
            return 0.33

        if self.location < N_NAV_STATES and self.knob == 4:
            if self.location == self.theta:
                return 1.0
            else:
                return -1.0

        if self.location != Location.SAFE_SINK and self.knob < 4:
            return -0.33

        return 0.0

    def get_outcome(self) -> str:
        """Classify terminal outcome: 'goal', 'safe_sink', 'wrong_goal', 'penalty', or 'other'."""
        if self.location == Location.SAFE_SINK:
            return "safe_sink"
        if self.location < N_NAV_STATES and self.knob == 4:
            return "goal" if self.location == self.theta else "wrong_goal"
        if self.location != Location.SAFE_SINK and self.knob < 4:
            return "penalty"
        return "other"

    @property
    def has_seen_cue(self) -> bool:
        """Whether agent has visited the cue location."""
        return self._has_seen_cue


def create_epistemic_maze_tensors(
    n_theta: int = 2,
    cue_accuracy: float = 1.0,
    location_observation_accuracy: float = 0.95,
    goal_temperature: float = 5.0,
    cue_cost_epsilon: float = 0.01,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Create the Epistemic Maze transition, observation, and goal tensors.

    Compatible with temporal_vfe.py and temporal_optimizer.py.

    Args:
        n_theta: Number of context values (2 or 5)
        cue_accuracy: Probability of correct cue observation (default 1.0)
        location_observation_accuracy: Probability of correct location observation (default 0.95)
        goal_temperature: Softmax temperature for goal mapping
        cue_cost_epsilon: Relative cost of visiting cue (1.0=uniform, <1.0=favorable, >1.0=costly)

    Returns:
        Tuple of:
        - transition_tensor: Shape (35, 35, 8) - p(s' | s, a)
        - location_observation_tensor: Shape (7, 35) - p(location_obs | s)
        - theta_observation_tensor: Shape (n_theta+1, 35, n_theta) - p(theta_obs | s, θ)
        - goal_mapping: Shape (35, n_theta) - p(goal | s, θ)
        - action_prior: Shape (8,) - p(a)
        - theta_prior: Shape (n_theta,) - p(θ)
    """
    n_states = N_STATES  # 35
    n_actions = N_ACTIONS  # 8
    n_theta_obs = n_theta + 1  # Context observations + neutral

    # ==================== TRANSITION TENSOR ====================
    # Shape: (next_state, current_state, action)
    # Encodes p(s' | s, a)
    transition = jnp.zeros((n_states, n_states, n_actions))

    for s in range(n_states):
        loc, knob = state_to_components(s)

        for a in range(n_actions):
            # === Cue state (location 6): any action → uniform over nav states ===
            if loc == Location.CUE:
                for next_loc in range(N_NAV_STATES):
                    next_s = components_to_state(next_loc, knob)
                    transition = transition.at[next_s, s, a].set(1.0 / N_NAV_STATES)
                continue

            # === Safe sink (location 5): absorbing state ===
            if loc == Location.SAFE_SINK:
                transition = transition.at[s, s, a].set(1.0)
                continue

            # === Decrease knob (action 5): behave like nav action 0 (stay location) ===
            if a == EpistemicAction.DECREASE_KNOB:
                new_knob = max(0, knob - 1)
                success_prob = knob / 4.0
                fail_prob = 1.0 - success_prob
                
                # Success: stay at same location, decrease knob
                next_s = components_to_state(loc, new_knob)
                transition = transition.at[next_s, s, a].add(success_prob)
                
                # Failure: fall to safe sink, still decrease knob
                sink_s = components_to_state(Location.SAFE_SINK, new_knob)
                transition = transition.at[sink_s, s, a].add(fail_prob)
                continue

            # === Increase knob (action 6): behave like nav action 0 (stay location) ===
            if a == EpistemicAction.INCREASE_KNOB:
                new_knob = min(4, knob + 1)
                success_prob = knob / 4.0
                fail_prob = 1.0 - success_prob
                
                # Success: stay at same location, increase knob
                next_s = components_to_state(loc, new_knob)
                transition = transition.at[next_s, s, a].add(success_prob)
                
                # Failure: fall to safe sink, still increase knob
                sink_s = components_to_state(Location.SAFE_SINK, new_knob)
                transition = transition.at[sink_s, s, a].add(fail_prob)
                continue

            # === Visit cue (action 7): deterministic move to cue state ===
            if a == EpistemicAction.VISIT_CUE:
                next_s = components_to_state(Location.CUE, knob)
                transition = transition.at[next_s, s, a].set(1.0)
                continue

            # === Navigation actions (0-4): stochastic based on knob ===
            if 0 <= a <= 4:
                success_prob = knob / 4.0
                fail_prob = 1.0 - success_prob

                # Successful navigation: (location + action) mod 5
                next_loc = (loc + a) % N_NAV_STATES
                next_s = components_to_state(next_loc, knob)
                transition = transition.at[next_s, s, a].add(success_prob)

                # Failure: fall to safe sink
                sink_s = components_to_state(Location.SAFE_SINK, knob)
                transition = transition.at[sink_s, s, a].add(fail_prob)

    # ==================== LOCATION OBSERVATION TENSOR ====================
    # Shape: (n_locations, n_states)
    # Encodes p(location_obs | s)
    # Independent of theta: just reveals where the agent is
    location_observation = jnp.zeros((N_LOCATIONS, n_states))

    for s in range(n_states):
        loc, _ = state_to_components(s)

        # At CUE: perfect observation
        if loc == Location.CUE:
            location_observation = location_observation.at[Location.CUE, s].set(1.0)
        else:
            # Elsewhere: observe location with location_observation_accuracy
            location_observation = location_observation.at[loc, s].set(location_observation_accuracy)
            # Spread remaining probability uniformly over other NON-CUE locations
            # Never observe being at CUE when not actually there
            other_prob = (1.0 - location_observation_accuracy) / (N_LOCATIONS - 2)  # Exclude true location and CUE
            for obs_loc in range(N_LOCATIONS):
                if obs_loc != loc and obs_loc != Location.CUE:
                    location_observation = location_observation.at[obs_loc, s].add(other_prob)

    # ==================== THETA OBSERVATION TENSOR ====================
    # Shape: (n_theta_obs, n_states, n_theta)
    # Encodes p(theta_obs | s, θ)
    # Only informative at CUE location
    theta_observation = jnp.zeros((n_theta_obs, n_states, n_theta))

    for s in range(n_states):
        loc, _ = state_to_components(s)

        for theta in range(n_theta):
            if loc == Location.CUE:
                # At cue: informative observation about θ
                # P(obs = θ | state at cue, θ) = cue_accuracy
                theta_observation = theta_observation.at[theta, s, theta].set(cue_accuracy)
                # Spread remaining probability among other context observations
                other_prob = (1.0 - cue_accuracy) / (n_theta - 1) if n_theta > 1 else 0.0
                for obs_idx in range(n_theta):
                    if obs_idx != theta:
                        theta_observation = theta_observation.at[obs_idx, s, theta].set(other_prob)
            else:
                # Elsewhere: deterministic neutral observation (index n_theta_obs - 1)
                theta_observation = theta_observation.at[n_theta_obs - 1, s, theta].set(1.0)

    # ==================== GOAL MAPPING ====================
    # Shape: (n_states, n_theta)
    # Encodes reward structure as soft goal preferences via softmax
    # Higher logits = more desirable states
    goal_logits = jnp.zeros((n_states, n_theta))

    for s in range(n_states):
        loc, knob = state_to_components(s)

        for theta in range(n_theta):
            if loc == Location.SAFE_SINK:
                # Safe sink: moderate reward (+0.33), independent of θ
                goal_logits = goal_logits.at[s, theta].set(goal_temperature * 0.33)
            elif loc < N_NAV_STATES and knob == 4:
                # Navigation state with max reactivity (knob = 4)
                if loc == theta:
                    # Correct goal: high reward (+1.0)
                    goal_logits = goal_logits.at[s, theta].set(goal_temperature * 1.0)
                else:
                    # Wrong goal: penalty (-1.0)
                    goal_logits = goal_logits.at[s, theta].set(goal_temperature * (-1.0))
            elif loc != Location.SAFE_SINK and knob < 4:
                # Non-sink, non-max reactivity: mild penalty (-0.33)
                goal_logits = goal_logits.at[s, theta].set(goal_temperature * (-1.0))
            else:
                # Other states (cue with knob=4): penalty to avoid
                goal_logits = goal_logits.at[s, theta].set(-1.0)

    # Convert logits to probabilities via softmax over states
    goal_mapping = jax.nn.softmax(goal_logits, axis=0)

    # ==================== ACTION PRIOR ====================
    # Semantics: cue_cost_epsilon controls the relative cost of visiting cue
    # - 1.0: uniform distribution (neutral prior)
    # - < 1.0: more favorable to visit cue (lower cost)
    # - > 1.0: actual cost (less likely to visit)
    base_prob = 1.0 / n_actions
    cue_prob = base_prob / cue_cost_epsilon
    other_prob = (1.0 - cue_prob) / (n_actions - 1)
    action_prior = jnp.ones(n_actions) * other_prob
    action_prior = action_prior.at[EpistemicAction.VISIT_CUE].set(cue_prob)

    # ==================== THETA PRIOR ====================
    # Uniform prior over context
    theta_prior = jnp.ones(n_theta) / n_theta

    return transition, location_observation, theta_observation, goal_mapping, action_prior, theta_prior


def get_initial_state_distribution(
    start_location: Optional[int] = None,
    start_knob: int = 4,
) -> Array:
    """
    Get initial state distribution.

    Args:
        start_location: Specific starting location (0-4), or None for uniform
        start_knob: Starting knob value (default 4 = max reactivity)

    Returns:
        Initial state distribution (35,)
    """
    initial = jnp.zeros(N_STATES)

    if start_location is not None:
        # Specific starting location
        s = components_to_state(start_location, start_knob)
        initial = initial.at[s].set(1.0)
    else:
        # Uniform over navigation states with given knob
        for loc in range(N_NAV_STATES):
            s = components_to_state(loc, start_knob)
            initial = initial.at[s].set(1.0 / N_NAV_STATES)

    return initial


def get_state_name(state_idx: int) -> str:
    """Get human-readable name for a state."""
    loc, knob = state_to_components(state_idx)
    loc_names = ["Nav0", "Nav1", "Nav2", "Nav3", "Nav4", "Sink", "Cue"]
    return f"{loc_names[loc]}(k={knob})"


def get_action_name(action: int) -> str:
    """Get human-readable name for an action."""
    action_names = [
        "Nav+0", "Nav+1", "Nav+2", "Nav+3", "Nav+4",
        "Knob--", "Knob++", "VisitCue"
    ]
    return action_names[action]
