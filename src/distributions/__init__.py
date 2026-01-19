"""Discrete distribution utilities for Active Inference."""

from .entropy import (
    categorical_entropy,
    categorical_kl,
    conditional_entropy,
    joint_entropy,
    conditional_entropy_from_conditionals,
)

__all__ = [
    "categorical_entropy",
    "categorical_kl",
    "conditional_entropy",
    "joint_entropy",
    "conditional_entropy_from_conditionals",
]
