"""Entropy and information-theoretic utilities for categorical distributions."""

import jax.numpy as jnp
from jax import Array

# Small constant to avoid log(0)
EPS = 1e-10


def categorical_entropy(probs: Array) -> Array:
    """
    Compute entropy of a categorical distribution.
    
    H[p] = -sum_i p_i log p_i
    
    Args:
        probs: Probability vector of shape (n_categories,)
        
    Returns:
        Scalar entropy value.
    """
    # Clip to avoid log(0)
    probs_safe = jnp.clip(probs, EPS, 1.0)
    return -jnp.sum(probs * jnp.log(probs_safe))


def categorical_kl(p: Array, q: Array) -> Array:
    """
    Compute KL divergence between two categorical distributions.
    
    KL[p || q] = sum_i p_i log(p_i / q_i)
    
    Args:
        p: First probability vector (the "true" distribution)
        q: Second probability vector (the "approximate" distribution)
        
    Returns:
        Scalar KL divergence value.
    """
    p_safe = jnp.clip(p, EPS, 1.0)
    q_safe = jnp.clip(q, EPS, 1.0)
    return jnp.sum(p * (jnp.log(p_safe) - jnp.log(q_safe)))


def joint_entropy(joint_probs: Array) -> Array:
    """
    Compute entropy of a joint distribution.
    
    H[p(x,y)] = -sum_{x,y} p(x,y) log p(x,y)
    
    Args:
        joint_probs: Joint probability array of arbitrary shape.
        
    Returns:
        Scalar entropy value.
    """
    probs_safe = jnp.clip(joint_probs, EPS, 1.0)
    return -jnp.sum(joint_probs * jnp.log(probs_safe))


def conditional_entropy(joint_probs: Array, condition_axis: int) -> Array:
    """
    Compute conditional entropy H(X|Y) from joint distribution p(X,Y).
    
    H(X|Y) = H(X,Y) - H(Y)
           = -sum_{x,y} p(x,y) log p(x|y)
           = sum_y p(y) H(X|Y=y)
    
    Args:
        joint_probs: Joint probability array p(X,Y) where X is the variable
                     we're conditioning and Y is the conditioning variable.
        condition_axis: Axis corresponding to the conditioning variable Y.
        
    Returns:
        Scalar conditional entropy H(X|Y).
    """
    # H(X,Y)
    h_joint = joint_entropy(joint_probs)
    
    # p(Y) = sum_x p(x,y)
    p_y = jnp.sum(joint_probs, axis=tuple(
        i for i in range(joint_probs.ndim) if i != condition_axis
    ))
    
    # H(Y)
    h_y = categorical_entropy(p_y)
    
    # H(X|Y) = H(X,Y) - H(Y)
    return h_joint - h_y


def conditional_entropy_from_conditionals(
    p_x_given_y: Array, 
    p_y: Array,
    x_axis: int = 0
) -> Array:
    """
    Compute conditional entropy H(X|Y) from conditional p(X|Y) and marginal p(Y).
    
    H(X|Y) = sum_y p(y) H(X|Y=y)
           = sum_y p(y) * (-sum_x p(x|y) log p(x|y))
    
    Args:
        p_x_given_y: Conditional distribution p(X|Y), shape depends on x_axis.
        p_y: Marginal distribution p(Y).
        x_axis: Axis corresponding to X in p_x_given_y.
        
    Returns:
        Scalar conditional entropy H(X|Y).
    """
    # Compute H(X|Y=y) for each y
    p_safe = jnp.clip(p_x_given_y, EPS, 1.0)
    h_x_given_y = -jnp.sum(p_x_given_y * jnp.log(p_safe), axis=x_axis)
    
    # E_y[H(X|Y=y)] = sum_y p(y) H(X|Y=y)
    return jnp.sum(p_y * h_x_given_y)
