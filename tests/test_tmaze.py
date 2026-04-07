"""Tests for T-maze environment and planning."""

import jax.numpy as jnp
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.environments import TMaze, create_tmaze_tensors
from src.distributions import categorical_entropy, categorical_kl
from src.planning import plan_actions_factorized, FactorizedPlanningConfig
from src.planning import (
    convert_tmaze_tensors_to_pymdp,
    SophisticatedPlanningConfig,
    SophisticatedPlanner,
)


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
        
        config = FactorizedPlanningConfig(
            planning_horizon=2,
            n_obs=2,
            n_states=5,
            n_actions=4,
            n_theta=2,
            n_optimization_steps=10,
        )
        
        prior_state = jnp.array([0, 1, 0, 0, 0], dtype=jnp.float32)  # At middle
        prior_reward = jnp.array([0.5, 0.5])
        
        result = plan_actions_factorized(
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
        
        config = FactorizedPlanningConfig(
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
        
        result = plan_actions_factorized(
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


class TestTMazePymdp:
    """Tests for pymdp integration with T-Maze."""

    def test_convert_tmaze_tensors_shapes(self):
        """Test A/B/C/D have correct shapes."""
        transition, obs, goal = create_tmaze_tensors()
        theta_prior = jnp.array([0.5, 0.5])
        A, B, C, D = convert_tmaze_tensors_to_pymdp(
            transition, obs, goal, theta_prior,
        )
        assert A[0].shape == (5, 5, 2)  # location obs
        assert A[1].shape == (2, 5, 2)  # cue obs
        assert A[2].shape == (3, 5, 2)  # reward obs
        assert B[0].shape == (5, 5, 4)  # location transitions
        assert B[1].shape == (2, 2, 1)  # theta identity
        assert C[0].shape == (5,)
        assert C[1].shape == (2,)
        assert C[2].shape == (3,)
        assert D[0].shape == (5,)
        assert D[1].shape == (2,)

    def test_a_matrix_location_identity(self):
        """Test A[0] is identity-like for locations."""
        import numpy as np
        transition, obs, goal = create_tmaze_tensors()
        theta_prior = jnp.array([0.5, 0.5])
        A, _, _, _ = convert_tmaze_tensors_to_pymdp(
            transition, obs, goal, theta_prior,
        )
        for theta_idx in range(2):
            assert np.allclose(A[0][:, :, theta_idx], np.eye(5))

    def test_a_matrix_cue_matches_reward_obs(self):
        """Test A[1] matches the original reward_obs_tensor."""
        import numpy as np
        transition, obs, goal = create_tmaze_tensors()
        theta_prior = jnp.array([0.5, 0.5])
        A, _, _, _ = convert_tmaze_tensors_to_pymdp(
            transition, obs, goal, theta_prior,
        )
        assert np.allclose(A[1], np.array(obs))

    def test_a_matrix_reward_structure(self):
        """Test A[2] encodes correct/wrong goal at goal states."""
        import numpy as np
        transition, obs, goal = create_tmaze_tensors()
        theta_prior = jnp.array([0.5, 0.5])
        A, _, _, _ = convert_tmaze_tensors_to_pymdp(
            transition, obs, goal, theta_prior,
        )
        # TOP_LEFT (s=2), theta=0 (left) -> correct (idx 1)
        assert A[2][1, 2, 0] == 1.0
        # TOP_LEFT (s=2), theta=1 (right) -> wrong (idx 2)
        assert A[2][2, 2, 1] == 1.0
        # TOP_RIGHT (s=4), theta=1 (right) -> correct (idx 1)
        assert A[2][1, 4, 1] == 1.0
        # TOP_RIGHT (s=4), theta=0 (left) -> wrong (idx 2)
        assert A[2][2, 4, 0] == 1.0
        # MIDDLE (s=1) -> always no_reward (idx 0)
        assert A[2][0, 1, 0] == 1.0
        assert A[2][0, 1, 1] == 1.0

    def test_pymdp_episode_smoke(self):
        """Smoke test: run a full pymdp episode without errors."""
        import random
        random.seed(42)

        transition, obs, goal = create_tmaze_tensors()
        theta_prior = jnp.array([0.5, 0.5])
        A, B, C, D = convert_tmaze_tensors_to_pymdp(
            transition, obs, goal, theta_prior,
        )

        config = SophisticatedPlanningConfig(
            planning_horizon=4,
            n_states=5,
            n_actions=4,
            n_theta=2,
            policy_len=1,
            inference_horizon=4,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            action_selection="deterministic",
            gamma=16.0,
            sophisticated=True,
        )

        planner = SophisticatedPlanner.from_pymdp_arrays(A, B, C, D, config)
        planner.agent.reset()

        # Simulate one step: observe location=1 (MIDDLE), ambiguous cue, no reward
        obs_indices = [1, 0, None]
        planner.infer_states(obs_indices)
        q_pi, efe = planner.infer_policies()
        action_arr = planner.sample_action()

        assert 0 <= int(action_arr[0]) < 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
