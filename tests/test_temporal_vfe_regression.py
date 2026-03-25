"""Regression tests for temporal_vfe vectorization.

Captures VFE output + gradients for all 3 inference modes with fixed random inputs,
ensuring vectorized implementation matches the original loop-based version.
"""

import pytest
import jax
import jax.numpy as jnp
from jax.nn import softmax

from src.objectives.temporal_vfe import temporal_vfe
from src.environments.observation_modality import ObservationModality


def make_test_inputs(horizon=3, n_states=7, n_theta=2, n_actions=4, seed=42):
    """Create deterministic test inputs for regression testing."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    q_theta_logits = jax.random.normal(keys[0], (n_theta,))
    q_u_logits = jax.random.normal(keys[1], (horizon, n_states, n_actions))
    q_x_logits = jax.random.normal(keys[2], (horizon, n_states, n_states, n_actions, n_theta))

    # Two observation modalities: both theta-dependent
    n_obs_theta = n_theta + 1
    n_obs_loc = 7
    q_obs_theta_logits = jax.random.normal(keys[3], (horizon, n_obs_theta, n_states, n_theta))
    q_obs_loc_logits = jax.random.normal(keys[4], (horizon, n_obs_loc, n_states, n_theta))

    initial_state = jnp.zeros(n_states).at[0].set(1.0)

    # Generative model tensors
    gen_theta = jax.random.dirichlet(keys[5], jnp.ones(n_obs_theta), shape=(n_states, n_theta))
    gen_theta = jnp.transpose(gen_theta, (2, 0, 1))  # (n_obs, n_states, n_theta)
    gen_loc = jax.random.dirichlet(keys[6], jnp.ones(n_obs_loc), shape=(n_states, n_theta))
    gen_loc = jnp.transpose(gen_loc, (2, 0, 1))  # (n_obs, n_states, n_theta)

    obs_modalities = [
        ObservationModality(name="theta", generative_tensor=gen_theta, theta_dependent=True, n_obs=n_obs_theta),
        ObservationModality(name="location", generative_tensor=gen_loc, theta_dependent=True, n_obs=n_obs_loc),
    ]

    transition_tensor = jax.random.dirichlet(keys[7], jnp.ones(n_states), shape=(n_states, n_actions))
    transition_tensor = jnp.transpose(transition_tensor, (2, 0, 1))  # (n_states_next, n_states_prev, n_actions)

    goal_mapping = jax.random.dirichlet(keys[8], jnp.ones(2), shape=(n_states, n_theta))[:, :, 0]
    goal_mapping = jnp.clip(goal_mapping, 0.01, 0.99)

    action_prior = jnp.ones(n_actions) / n_actions
    theta_prior = jnp.ones(n_theta) / n_theta

    return dict(
        q_theta_logits=q_theta_logits,
        q_u_given_x_logits=q_u_logits,
        q_x_given_xu_theta_logits=q_x_logits,
        q_obs_logits_list=[q_obs_theta_logits, q_obs_loc_logits],
        initial_state=initial_state,
        transition_tensor=transition_tensor,
        observation_modalities=obs_modalities,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        horizon=horizon,
    )


@pytest.mark.parametrize("inference_mode", ["marginal", "active", "planning"])
def test_temporal_vfe_value_and_grad(inference_mode):
    """Test that VFE value and gradients match reference values."""
    inputs = make_test_inputs()

    def vfe_fn(q_theta_logits, q_u_logits, q_x_logits):
        return temporal_vfe(
            q_theta_logits=q_theta_logits,
            q_u_given_x_logits=q_u_logits,
            q_x_given_xu_theta_logits=q_x_logits,
            q_obs_logits_list=inputs['q_obs_logits_list'],
            initial_state=inputs['initial_state'],
            transition_tensor=inputs['transition_tensor'],
            observation_modalities=inputs['observation_modalities'],
            goal_mapping=inputs['goal_mapping'],
            action_prior=inputs['action_prior'],
            theta_prior=inputs['theta_prior'],
            horizon=inputs['horizon'],
            inference_mode=inference_mode,
        )

    value, grads = jax.value_and_grad(vfe_fn, argnums=(0, 1, 2))(
        inputs['q_theta_logits'],
        inputs['q_u_given_x_logits'],
        inputs['q_x_given_xu_theta_logits'],
    )

    # Check value is finite
    assert jnp.isfinite(value), f"VFE value is not finite: {value}"

    # Check gradients are finite
    for i, g in enumerate(grads):
        assert jnp.all(jnp.isfinite(g)), f"Gradient {i} has non-finite values"

    # Regression check against reference values
    ref = REFERENCE_VALUES[inference_mode]
    assert jnp.allclose(value, ref["value"], atol=1e-4), \
        f"{inference_mode}: VFE {float(value):.8f} != ref {ref['value']:.8f}"
    for i, g in enumerate(grads):
        gnorm = float(jnp.linalg.norm(g))
        assert abs(gnorm - ref["grad_norms"][i]) < 1e-4, \
            f"{inference_mode}: grad[{i}] norm {gnorm:.8f} != ref {ref['grad_norms'][i]:.8f}"


# Reference values from original (loop-based) implementation
# These were captured by running the test before vectorization
REFERENCE_VALUES = {
    "marginal": {"value": 8.4810810089, "grad_norms": [0.6133067607879639, 0.27373576164245605, 0.31200340390205383]},
    "active": {"value": 45.2728500366, "grad_norms": [0.6133067607879639, 0.2503315210342407, 0.296514093875885]},
    "planning": {"value": 11.9153862000, "grad_norms": [0.6219660043716431, 0.233661949634552, 0.3072192966938019]},
}


def _compute_reference_values():
    """Helper to compute and print reference values for all modes."""
    inputs = make_test_inputs()
    for mode in ["marginal", "active", "planning"]:
        def vfe_fn(q_theta_logits, q_u_logits, q_x_logits):
            return temporal_vfe(
                q_theta_logits=q_theta_logits,
                q_u_given_x_logits=q_u_logits,
                q_x_given_xu_theta_logits=q_x_logits,
                q_obs_logits_list=inputs['q_obs_logits_list'],
                initial_state=inputs['initial_state'],
                transition_tensor=inputs['transition_tensor'],
                observation_modalities=inputs['observation_modalities'],
                goal_mapping=inputs['goal_mapping'],
                action_prior=inputs['action_prior'],
                theta_prior=inputs['theta_prior'],
                horizon=inputs['horizon'],
                inference_mode=mode,
            )

        value, grads = jax.value_and_grad(vfe_fn, argnums=(0, 1, 2))(
            inputs['q_theta_logits'],
            inputs['q_u_given_x_logits'],
            inputs['q_x_given_xu_theta_logits'],
        )
        grad_norms = [float(jnp.linalg.norm(g)) for g in grads]
        print(f"{mode}: value={float(value):.10f}, grad_norms={grad_norms}")


if __name__ == "__main__":
    _compute_reference_values()
