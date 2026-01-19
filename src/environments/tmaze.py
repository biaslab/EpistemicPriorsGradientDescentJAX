"""
T-Maze environment for Active Inference experiments.

The T-maze has 5 valid positions arranged in a T-shape:

    State 2 (top-left) --- State 3 (top-middle) --- State 4 (top-right)
                                |
                          State 1 (middle)
                                |
                          State 0 (bottom/cue)

The reward can be at State 2 (left) or State 4 (right).
At State 0 (bottom), the agent receives a cue about reward location.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple
import random

import jax.numpy as jnp
from jax import Array


class Action(IntEnum):
    """Actions available in the T-maze."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class State(IntEnum):
    """States in the T-maze."""
    BOTTOM = 0      # Cue location
    MIDDLE = 1      # Junction
    TOP_LEFT = 2    # Left arm (possible reward)
    TOP_MIDDLE = 3  # Top middle
    TOP_RIGHT = 4   # Right arm (possible reward)


@dataclass
class TMaze:
    """
    T-Maze environment with deterministic dynamics.
    
    Attributes:
        agent_state: Current state of the agent (0-4).
        reward_location: Which arm has the reward ('left' or 'right').
        _has_seen_cue: Whether agent has visited the cue location.
    """
    agent_state: int
    reward_location: str  # 'left' or 'right'
    _has_seen_cue: bool = False
    
    @classmethod
    def create(cls, reward_location: str | None = None, start_state: int = State.MIDDLE) -> "TMaze":
        """
        Create a new T-maze environment.
        
        Args:
            reward_location: 'left' or 'right'. If None, chosen randomly.
            start_state: Starting state for the agent.
            
        Returns:
            New TMaze instance.
        """
        if reward_location is None:
            reward_location = random.choice(['left', 'right'])
        
        if reward_location not in ['left', 'right']:
            raise ValueError(f"reward_location must be 'left' or 'right', got {reward_location}")
        
        if not 0 <= start_state <= 4:
            raise ValueError(f"start_state must be 0-4, got {start_state}")
            
        env = cls(agent_state=start_state, reward_location=reward_location)
        # Check if starting at cue location
        env._has_seen_cue = (start_state == State.BOTTOM)
        return env
    
    def reset(self, reward_location: str | None = None, start_state: int = State.MIDDLE) -> "TMaze":
        """Reset the environment to initial state."""
        if reward_location is None:
            reward_location = random.choice(['left', 'right'])
        
        self.agent_state = start_state
        self.reward_location = reward_location
        self._has_seen_cue = (start_state == State.BOTTOM)
        return self
    
    def step(self, action: int) -> Tuple[Array, Array, float, bool]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=North, 1=East, 2=South, 3=West).
            
        Returns:
            Tuple of (location_obs, reward_cue, reward, done).
            - location_obs: One-hot vector of current location (5,)
            - reward_cue: Reward location cue [left_prob, right_prob] (2,)
            - reward: Reward received at this step
            - done: Whether episode is finished
        """
        # Update state based on action
        self.agent_state = self._get_next_state(self.agent_state, action)
        
        # Check if at cue location
        if self.agent_state == State.BOTTOM:
            self._has_seen_cue = True
        
        # Generate observations
        location_obs = self._get_location_observation()
        reward_cue = self._get_reward_cue()
        reward = self._get_reward()
        
        # Episode done if agent reaches reward location
        done = (reward != 0.0)
        
        return location_obs, reward_cue, reward, done
    
    def _get_next_state(self, state: int, action: int) -> int:
        """Deterministic state transition."""
        transitions = {
            # From BOTTOM (state 0)
            (State.BOTTOM, Action.NORTH): State.MIDDLE,
            (State.BOTTOM, Action.EAST): State.BOTTOM,   # Wall
            (State.BOTTOM, Action.SOUTH): State.BOTTOM,  # Wall
            (State.BOTTOM, Action.WEST): State.BOTTOM,   # Wall
            
            # From MIDDLE (state 1)
            (State.MIDDLE, Action.NORTH): State.TOP_MIDDLE,
            (State.MIDDLE, Action.EAST): State.MIDDLE,   # Wall
            (State.MIDDLE, Action.SOUTH): State.BOTTOM,
            (State.MIDDLE, Action.WEST): State.MIDDLE,   # Wall
            
            # From TOP_LEFT (state 2)
            (State.TOP_LEFT, Action.NORTH): State.TOP_LEFT,    # Wall
            (State.TOP_LEFT, Action.EAST): State.TOP_MIDDLE,
            (State.TOP_LEFT, Action.SOUTH): State.MIDDLE,
            (State.TOP_LEFT, Action.WEST): State.TOP_LEFT,     # Wall
            
            # From TOP_MIDDLE (state 3)
            (State.TOP_MIDDLE, Action.NORTH): State.TOP_MIDDLE,  # Wall
            (State.TOP_MIDDLE, Action.EAST): State.TOP_RIGHT,
            (State.TOP_MIDDLE, Action.SOUTH): State.MIDDLE,
            (State.TOP_MIDDLE, Action.WEST): State.TOP_LEFT,
            
            # From TOP_RIGHT (state 4)
            (State.TOP_RIGHT, Action.NORTH): State.TOP_RIGHT,  # Wall
            (State.TOP_RIGHT, Action.EAST): State.TOP_RIGHT,   # Wall
            (State.TOP_RIGHT, Action.SOUTH): State.MIDDLE,
            (State.TOP_RIGHT, Action.WEST): State.TOP_MIDDLE,
        }
        return transitions.get((state, action), state)
    
    def _get_location_observation(self) -> Array:
        """Return one-hot location observation."""
        obs = jnp.zeros(5)
        obs = obs.at[self.agent_state].set(1.0)
        return obs
    
    def _get_reward_cue(self) -> Array:
        """
        Return reward location cue.
        
        At the cue location (bottom), reveals true reward location.
        Elsewhere, returns uniform uncertainty.
        """
        if self._has_seen_cue:
            if self.reward_location == 'left':
                return jnp.array([1.0, 0.0])
            else:
                return jnp.array([0.0, 1.0])
        else:
            return jnp.array([0.5, 0.5])
    
    def _get_reward(self) -> float:
        """Return reward at current state."""
        if self.agent_state == State.TOP_LEFT:
            return 1.0 if self.reward_location == 'left' else -1.0
        elif self.agent_state == State.TOP_RIGHT:
            return 1.0 if self.reward_location == 'right' else -1.0
        return 0.0
    
    @property
    def has_seen_cue(self) -> bool:
        """Whether agent has visited the cue location."""
        return self._has_seen_cue


def create_tmaze_tensors() -> Tuple[Array, Array, Array]:
    """
    Create the TMaze transition and observation tensors.
    
    Returns:
        Tuple of:
        - transition_tensor: Shape (5, 5, 4) - p(s' | s, a)
        - reward_obs_tensor: Shape (2, 5, 2) - p(cue | location, reward_loc)
        - goal_mapping: Shape (5, 2) - mapping from reward_loc to goal state
    """
    # Transition tensor: (next_state, current_state, action)
    # Deterministic transitions
    transition = jnp.zeros((5, 5, 4))
    
    # From BOTTOM (state 0)
    transition = transition.at[1, 0, 0].set(1.0)  # North -> Middle
    transition = transition.at[0, 0, 1].set(1.0)  # East -> stay
    transition = transition.at[0, 0, 2].set(1.0)  # South -> stay
    transition = transition.at[0, 0, 3].set(1.0)  # West -> stay
    
    # From MIDDLE (state 1)
    transition = transition.at[3, 1, 0].set(1.0)  # North -> Top middle
    transition = transition.at[1, 1, 1].set(1.0)  # East -> stay
    transition = transition.at[0, 1, 2].set(1.0)  # South -> Bottom
    transition = transition.at[1, 1, 3].set(1.0)  # West -> stay
    
    # From TOP_LEFT (state 2)
    transition = transition.at[2, 2, 0].set(1.0)  # North -> stay
    transition = transition.at[3, 2, 1].set(1.0)  # East -> Top middle
    transition = transition.at[1, 2, 2].set(1.0)  # South -> Middle
    transition = transition.at[2, 2, 3].set(1.0)  # West -> stay
    
    # From TOP_MIDDLE (state 3)
    transition = transition.at[3, 3, 0].set(1.0)  # North -> stay
    transition = transition.at[4, 3, 1].set(1.0)  # East -> Top right
    transition = transition.at[1, 3, 2].set(1.0)  # South -> Middle
    transition = transition.at[2, 3, 3].set(1.0)  # West -> Top left
    
    # From TOP_RIGHT (state 4)
    transition = transition.at[4, 4, 0].set(1.0)  # North -> stay
    transition = transition.at[4, 4, 1].set(1.0)  # East -> stay
    transition = transition.at[1, 4, 2].set(1.0)  # South -> Middle
    transition = transition.at[3, 4, 3].set(1.0)  # West -> Top middle
    
    # Reward observation tensor: (cue_obs, location, reward_location)
    # cue_obs: 0=left_cue, 1=right_cue
    # reward_location: 0=left, 1=right
    reward_obs = jnp.ones((2, 5, 2)) * 0.5  # Default: uniform
    
    # At cue location (state 0), observation reveals reward location
    reward_obs = reward_obs.at[:, 0, 0].set(jnp.array([1.0, 0.0]))  # Left reward -> left cue
    reward_obs = reward_obs.at[:, 0, 1].set(jnp.array([0.0, 1.0]))  # Right reward -> right cue
    
    # Goal mapping: p(goal | state, reward_loc)
    # Small epsilon everywhere to avoid log(0), high value at correct goals
    eps = 1e-6
    goal_mapping = jnp.ones((5, 2)) 
    goal_mapping = goal_mapping.at[2, 0].set(50.0)  # Left reward -> goal at state 2 (top left)
    goal_mapping = goal_mapping.at[4, 1].set(50.0)  # Right reward -> goal at state 4 (top right)
    # Normalize columns so each is a valid distribution
    goal_mapping = goal_mapping / jnp.sum(goal_mapping, axis=0, keepdims=True)
    
    return transition, reward_obs, goal_mapping
