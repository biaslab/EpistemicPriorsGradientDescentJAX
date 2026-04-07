"""Direct MiniGrid environment wrapper for JAX agent."""

import re
import time
import gymnasium as gym
from gymnasium.envs.registration import register
import minigrid  # Register standard MiniGrid environments
import numpy as np
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from .minigrid import N_CELL_TYPES, N_ORIENTATIONS, N_ACTIONS


def register_doorkey_env(size: int) -> str:
    """
    Register a DoorKey environment with the given size if not already registered.
    
    MiniGrid only pre-registers certain sizes (5x5, 6x6, 8x8, 16x16).
    This function registers custom sizes dynamically.
    
    Args:
        size: Grid size (e.g., 7 for 7x7)
        
    Returns:
        env_id: The registered environment ID
    """
    env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
    
    try:
        register(
            id=env_id,
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": size},
        )
    except Exception:
        # Environment may already be registered, which is fine
        pass
    
    return env_id


def ensure_env_registered(env_name: str) -> str:
    """
    Ensure the environment is registered, registering it if necessary.
    
    Args:
        env_name: Environment name (e.g., "MiniGrid-DoorKey-7x7-v0")
        
    Returns:
        env_name: The (possibly registered) environment name
    """
    # Parse DoorKey environment pattern
    match = re.match(r"MiniGrid-DoorKey-(\d+)x(\d+)-v0", env_name)
    if match:
        size = int(match.group(1))
        return register_doorkey_env(size)
    
    # For other environments, assume they're already registered
    return env_name


def save_video(frames: list, video_path: str, fps: int = 5):
    """Save frames to video using imageio."""
    import imageio.v3 as iio
    
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_path, frames, fps=fps, codec="libx264")
    print(f"Video saved to {video_path}")


def save_frames(frames: list, frames_dir: str, episode_id: int):
    """Save individual frames as PNG files."""
    import imageio.v3 as iio
    
    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = frames_path / f"episode_{episode_id:03d}_frame_{i:03d}.png"
        iio.imwrite(str(frame_path), frame)


@dataclass
class StepResult:
    vision_obs: jnp.ndarray  # (fov_size, fov_size, 11) one-hot
    orientation_obs: jnp.ndarray  # (4,) one-hot
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class MiniGridWrapper:
    def __init__(
        self,
        env_name: str = "MiniGrid-DoorKey-5x5-v0",
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        fov_size: int = 7,
    ):
        # Ensure the environment is registered (needed for non-standard sizes)
        env_name = ensure_env_registered(env_name)
        self.env = gym.make(env_name, render_mode=render_mode)
        if fov_size != 7:
            from minigrid.wrappers import ViewSizeWrapper
            self.env = ViewSizeWrapper(self.env, agent_view_size=fov_size)
            self.env.unwrapped.agent_view_size = fov_size
        if max_steps is not None:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=max_steps)
        self.env_name = env_name
        self.fov_size = fov_size

    def reset(self, seed: Optional[int] = None) -> StepResult:
        obs, info = self.env.reset(seed=seed)
        return self._convert_observation(obs, 0.0, False, False, info)

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs, reward, terminated, truncated, info)

    def _convert_observation(
        self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> StepResult:
        image = obs["image"][:, :, 0]  # Just the object type channel
        direction = obs["direction"]

        vision_obs = self._image_to_onehot(image)
        orientation_obs = self._direction_to_onehot(direction)

        return StepResult(
            vision_obs=jnp.array(vision_obs),
            orientation_obs=jnp.array(orientation_obs),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _image_to_onehot(self, image: np.ndarray) -> np.ndarray:
        fov_w, fov_h = image.shape[0], image.shape[1]
        onehot = np.zeros((fov_w, fov_h, N_CELL_TYPES), dtype=np.float32)
        for i in range(fov_w):
            for j in range(fov_h):
                onehot[i, j, int(image[i, j])] = 1.0
        return onehot

    def _direction_to_onehot(self, direction: int) -> np.ndarray:
        onehot = np.zeros(N_ORIENTATIONS, dtype=np.float32)
        onehot[direction] = 1.0
        return onehot

    def close(self):
        self.env.close()

    @property
    def max_steps(self) -> int:
        if hasattr(self.env, "_max_episode_steps"):
            return self.env._max_episode_steps
        return self.env.unwrapped.max_steps

    def render(self):
        return self.env.render()


def run_episode(
    agent,
    env: MiniGridWrapper,
    seed: Optional[int] = None,
    receding_horizon: bool = False,
    verbose: bool = False,
    record: bool = False,
    no_orientation: bool = False,
) -> dict:
    """
    Run a single episode with the agent.

    Args:
        agent: Agent instance
        env: MiniGrid environment wrapper
        seed: Random seed for episode
        receding_horizon: If True, decrease planning horizon as time runs out (like Julia).
                         If False, use fixed horizon (standard MPC).
        verbose: Print debug info
        record: Whether to record frames

    Returns dict with episode statistics and optional frames.
    """
    episode_start = time.perf_counter()
    uniform_orientation = jnp.ones(N_ORIENTATIONS) / N_ORIENTATIONS

    result = env.reset(seed=seed)
    agent = agent.reset()

    if no_orientation:
        result = StepResult(
            vision_obs=result.vision_obs,
            orientation_obs=uniform_orientation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    total_reward = 0.0
    steps = 0
    max_steps = env.max_steps
    frames = []

    if record:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while True:
        if receding_horizon:
            time_remaining = max_steps - steps
        else:
            time_remaining = agent.planning_horizon

        action, agent = agent.step(
            result.vision_obs, result.orientation_obs, time_remaining
        )

        if verbose:
            print(f"Step {steps}: action={action}, time_remaining={time_remaining}")

        result = env.step(action)
        if no_orientation:
            result = StepResult(
                vision_obs=result.vision_obs,
                orientation_obs=uniform_orientation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=result.info,
            )
        total_reward += result.reward
        steps += 1

        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if result.terminated or result.truncated:
            break

    success = result.reward > 0
    wall_clock_s = time.perf_counter() - episode_start

    episode_result = {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "wall_clock_s": wall_clock_s,
    }
    
    if record:
        episode_result["frames"] = frames
    
    return episode_result


def run_experiment(
    agent,
    env_name: str = "MiniGrid-DoorKey-5x5-v0",
    n_episodes: int = 100,
    max_steps: int = 100,
    receding_horizon: bool = False,
    seed_start: int = 0,
    verbose: bool = False,
    render_mode: Optional[str] = None,
    show_progress: bool = True,
    record_episodes: Optional[list[int]] = None,
    video_dir: Optional[str] = None,
    fov_size: int = 7,
    no_orientation: bool = False,
) -> dict:
    """
    Run multiple episodes and collect statistics.

    Args:
        agent: Agent instance
        env_name: MiniGrid environment name
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        receding_horizon: If True, decrease planning horizon as time runs out
        seed_start: Starting seed for reproducibility
        verbose: Print debug info
        render_mode: Render mode for environment (auto-set to rgb_array if recording)
        show_progress: Show progress bar
        record_episodes: List of episode indices to record (e.g., [0, 9, 99] for first, 10th, 100th)
        video_dir: Directory to save videos (required if record_episodes is set)

    Returns dict with aggregated statistics.
    """
    experiment_start = time.perf_counter()

    if record_episodes is None:
        record_episodes = []

    actual_render_mode = "rgb_array" if record_episodes else render_mode
    env = MiniGridWrapper(env_name=env_name, render_mode=actual_render_mode, max_steps=max_steps, fov_size=fov_size)

    results = []
    successes = 0
    
    pbar = tqdm(range(n_episodes), disable=not show_progress, desc="Episodes")
    for i in pbar:
        seed = seed_start + i
        should_record = i in record_episodes
        
        episode_result = run_episode(
            agent, env, seed=seed, receding_horizon=receding_horizon,
            verbose=verbose, record=should_record, no_orientation=no_orientation
        )
        
        if should_record and "frames" in episode_result and video_dir:
            frames = episode_result.pop("frames")
            frames_dir = str(Path(video_dir) / f"frames_episode_{i:03d}")
            save_frames(frames, frames_dir, i)
            video_path = str(Path(video_dir) / f"episode_{i:03d}.mp4")
            save_video(frames, video_path)
        
        results.append(episode_result)
        
        if episode_result["success"]:
            successes += 1
        
        success_rate = successes / (i + 1)
        pbar.set_postfix({"success": f"{success_rate:.1%}", "steps": episode_result["steps"]})

    env.close()

    total_steps = sum(r["steps"] for r in results)
    total_wall_clock_s = time.perf_counter() - experiment_start

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": sum(r["total_reward"] for r in results) / n_episodes,
        "total_wall_clock_s": total_wall_clock_s,
        "avg_wall_clock_s_per_episode": total_wall_clock_s / n_episodes,
        "episode_results": results,
    }
