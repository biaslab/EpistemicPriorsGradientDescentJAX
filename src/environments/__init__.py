"""Environment implementations for Active Inference experiments."""

from .tmaze import TMaze, create_tmaze_tensors

__all__ = ["TMaze", "create_tmaze_tensors"]
