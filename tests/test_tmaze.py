"""Tests for T-maze environment and planning."""

import jax.numpy as jnp
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.environments import TMaze, create_tmaze_tensors
from src.distributions import categorical_entropy, categorical_kl
from src.planning import plan_actions, PlanningConfig


class TestTMazeEnvironment:
    """Tests for TMaze environment."""
    
    def test_create_tmaze(self):
        """Test TMaze creation."""
        env = TMaze.create(reward_location='left', start_state=1)
        assert env.agent_state == 1
        assert env.reward_location == 'left'
        assert not env.has_seen_cue
    
    def test_deterministic_transitions(self):
        """Test that transitions are deterministic."""
        env = TMaze.create(reward_location='left', start_state=1)
        
        # From middle (1), going south should go to bottom (0)
        env.step(2)  # South
        assert env.agent_state == 0
        assert env.has_seen_cue
    
    def test_reward_cue(self):
        """Test reward cue at bottom location."""
        env = TMaze.create(reward_location='left', start_state=0)
        assert env.has_seen_cue
        
        _, cue, _, _ = env.step(0)  # Any action
        assert jnp.allclose(cue, jnp.array([1.0, 0.0]))  # Left cue
    
    def test_reward_at_correct_location(self):
        """Test reward is given at correct arm."""
        # Left reward
        env = TMaze.create(reward_location='left', start_state=2)  # Top left
        _, _, reward, done = env.step(0)  # Stay
        assert reward == 1.0
        assert done
        
        # Right reward at wrong location
        env = TMaze.create(reward_location='right', start_state=2)  # Top left
        _, _, reward, done = env.step(0)
        assert reward == -1.0


class TestTMazeTensors:
    """Tests for TMaze tensors."""
    
    def test_transition_tensor_shape(self):
        """Test transition tensor has correct shape."""
        transition, _, _ = create_tmaze_tensors()
        assert transition.shape == (5, 5, 4)
    
    def test_transition_tensor_normalized(self):
        """Test transition tensor is properly normalized."""
        transition, _, _ = create_tmaze_tensors()
        # Each (state, action) pair should have exactly one next state
        for s in range(5):
            for a in range(4):
                assert jnp.sum(transition[:, s, a]) == 1.0
    
    def test_observation_tensor_shape(self):
        """Test observation tensor has correct shape."""
        _, obs, _ = create_tmaze_tensors()
        assert obs.shape == (2, 5, 2)


class TestDistributions:
    """Tests for distribution utilities."""
    
    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        p = jnp.ones(4) / 4
        h = categorical_entropy(p)
        expected = jnp.log(4.0)  # Maximum entropy for 4 categories
        assert jnp.isclose(h, expected, atol=1e-5)
    
    def test_entropy_deterministic(self):
        """Test entropy of deterministic distribution."""
        p = jnp.array([1.0, 0.0, 0.0, 0.0])
        h = categorical_entropy(p)
        assert jnp.isclose(h, 0.0, atol=1e-5)
    
    def test_kl_same_distribution(self):
        """Test KL divergence between identical distributions."""
        p = jnp.array([0.2, 0.3, 0.5])
        kl = categorical_kl(p, p)
        assert jnp.isclose(kl, 0.0, atol=1e-5)


class TestPlanning:
    """Tests for planning module."""
    
    def test_planning_returns_valid_distribution(self):
        """Test that planning returns valid probability distributions."""
        transition, obs, goal = create_tmaze_tensors()
        
        config = PlanningConfig(
            planning_horizon=2,
            n_obs=2,
            n_states=5,
            n_actions=4,
            n_theta=2,
            n_optimization_steps=10,
        )
        
        prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)  # At middle
        prior_reward = jnp.array([0.5, 0.5])
        
        result = plan_actions(
            prior_state=prior_state,
            prior_reward_location=prior_reward,
            transition_tensor=transition,
            observation_tensor=obs,
            goal_mapping=goal,
            config=config,
        )
        
        # Check probabilities sum to 1
        assert jnp.isclose(jnp.sum(result.first_action_probs), 1.0, atol=1e-5)
        
        # Check all probabilities are non-negative
        assert jnp.all(result.first_action_probs >= 0)
    
    def test_planning_with_known_reward_location(self):
        """Test that agent goes directly to goal when reward location is known."""
        transition, obs, goal = create_tmaze_tensors()
        
        config = PlanningConfig(
            planning_horizon=4,
            n_obs=2,
            n_states=5,
            n_actions=4,
            n_theta=2,
            n_optimization_steps=50,
            inference_mode="marginal",  # Use marginal inference for direct goal-seeking
        )
        
        # Agent at middle, knows reward is on left
        prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)
        prior_reward = jnp.array([1.0, 0.0])  # Certain left
        
        result = plan_actions(
            prior_state=prior_state,
            prior_reward_location=prior_reward,
            transition_tensor=transition,
            observation_tensor=obs,
            goal_mapping=goal,
            config=config,
        )
        
        # Should prefer North (0) to go toward goal
        # (North goes to top-middle, then West to top-left)
        assert result.first_action_probs[0] > 0.3  # North should have significant prob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
