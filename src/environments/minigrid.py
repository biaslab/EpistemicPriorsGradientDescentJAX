from enum import IntEnum
import numpy as np
import jax
import jax.numpy as jnp

from .observation_modality import ObservationModality
from .environment_protocol import EnvironmentTensors


class ActionType(IntEnum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


class CellType(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10


class Orientation(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


N_CELL_TYPES = 11
N_ORIENTATIONS = 4
N_ACTIONS = 7
N_DOOR_KEY_STATES = 3


def state_to_coords(s: int, n: int) -> tuple[int, int]:
    return (s // n, s % n)


def coords_to_state(x: int, y: int, n: int) -> int:
    return x * n + y


def flatten_state_index(
    state: int,
    orientation: int,
    door_key_state: int,
    n_states: int,
    n_orientations: int,
    n_door_key_states: int,
) -> int:
    return (
        state * (n_orientations * n_door_key_states)
        + orientation * n_door_key_states
        + door_key_state
    )


def unflatten_state_index(
    flat_idx: int, n_states: int, n_orientations: int, n_door_key_states: int
) -> tuple[int, int, int]:
    door_key_state = flat_idx % n_door_key_states
    flat_idx = flat_idx // n_door_key_states
    orientation = flat_idx % n_orientations
    state = flat_idx // n_orientations
    return (state, orientation, door_key_state)


def key_position(key_pos: int, n: int) -> tuple[int, int]:
    return (key_pos // n, key_pos % n)


def door_position(door_pos: int, n: int) -> tuple[int, int]:
    return (door_pos // n + 1, door_pos % n)


def get_valid_static_configs(n: int) -> list[tuple[int, int]]:
    """Return list of (key_pos, door_pos) where key_x < door_x."""
    n_positions = n * n - 2 * n
    configs = []
    for key_pos in range(n_positions):
        key_x, _ = key_position(key_pos, n)
        for door_pos in range(n_positions):
            door_x, _ = door_position(door_pos, n)
            if key_x < door_x:
                configs.append((key_pos, door_pos))
    return configs


def get_relative_coords(
    agent_x: int, agent_y: int, orientation: int, target_x: int, target_y: int
) -> tuple[int, int]:
    dx = target_x - agent_x
    dy = target_y - agent_y

    if orientation == Orientation.RIGHT:
        return (-dy, dx)
    elif orientation == Orientation.DOWN:
        return (dx, dy)
    elif orientation == Orientation.LEFT:
        return (dy, -dx)
    else:  # UP
        return (-dx, -dy)


def in_fov(rel_x: int, rel_y: int, fov_size: int = 7) -> bool:
    half = fov_size // 2
    return -half <= rel_x <= half and 0 <= rel_y <= fov_size - 1


def relative_to_fov_coords(rel_x: int, rel_y: int, fov_size: int = 7) -> tuple[int, int]:
    half = fov_size // 2
    fov_x = half - rel_x  # Agent at column half (0-indexed)
    fov_y = (fov_size - 1) - rel_y  # Agent at last row (0-indexed)
    return (fov_x, fov_y)


def relative_to_absolute_coords(
    agent_x: int, agent_y: int, orientation: int, rel_x: int, rel_y: int
) -> tuple[int, int]:
    if orientation == Orientation.RIGHT:
        dx = rel_y
        dy = -rel_x
    elif orientation == Orientation.DOWN:
        dx = rel_x
        dy = rel_y
    elif orientation == Orientation.LEFT:
        dx = -rel_y
        dy = rel_x
    else:  # UP
        dx = -rel_x
        dy = -rel_y

    return (agent_x + dx, agent_y + dy)


def generate_visibility_mask(
    agent_x: int, agent_y: int, width: int, height: int, walls: set[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros((width, height), dtype=bool)
    mask[agent_x, agent_y] = True

    for j in range(height - 1, -1, -1):
        for i in range(width - 1):
            if not mask[i, j]:
                continue
            if (i, j) in walls:
                continue
            mask[i + 1, j] = True
            if j > 0:
                mask[i + 1, j - 1] = True
                mask[i, j - 1] = True

        for i in range(width - 1, 0, -1):
            if not mask[i, j]:
                continue
            if (i, j) in walls:
                continue
            mask[i - 1, j] = True
            if j > 0:
                mask[i - 1, j - 1] = True
                mask[i, j - 1] = True

    return mask


def create_wall_set(door_x: int, door_y: int, n: int) -> set[tuple[int, int]]:
    walls = set()
    for y in range(n):
        if y != door_y:
            walls.add((door_x, y))
    for x in range(n):
        walls.add((x, -1))
        walls.add((x, n))
    for y in range(n):
        walls.add((-1, y))
        walls.add((n, y))
    walls.add((-1, -1))
    walls.add((n, -1))
    walls.add((-1, n))
    walls.add((n, n))
    return walls


def get_fov(
    agent_x: int,
    agent_y: int,
    orientation: int,
    key_x: int,
    key_y: int,
    door_x: int,
    door_y: int,
    door_key_state: int,
    n: int,
    fov_size: int = 7,
) -> np.ndarray:
    half = fov_size // 2
    fov = np.full((fov_size, fov_size), CellType.EMPTY, dtype=np.int32)
    walls = create_wall_set(door_x, door_y, n)

    for wall_x, wall_y in walls:
        rel_wall = get_relative_coords(agent_x, agent_y, orientation, wall_x, wall_y)
        if in_fov(*rel_wall, fov_size):
            fov_x, fov_y = relative_to_fov_coords(*rel_wall, fov_size)
            fov[fov_x, fov_y] = CellType.WALL

    # Place goal
    goal_x, goal_y = n - 1, n - 1
    rel_goal = get_relative_coords(agent_x, agent_y, orientation, goal_x, goal_y)
    if in_fov(*rel_goal, fov_size):
        fov_x, fov_y = relative_to_fov_coords(*rel_goal, fov_size)
        fov[fov_x, fov_y] = CellType.GOAL

    # Place key on ground (only when not carrying)
    if door_key_state == 0:
        rel_key = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if in_fov(*rel_key, fov_size):
            fov_x, fov_y = relative_to_fov_coords(*rel_key, fov_size)
            fov[fov_x, fov_y] = CellType.KEY

    # Place door
    rel_door = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if in_fov(*rel_door, fov_size):
        fov_x, fov_y = relative_to_fov_coords(*rel_door, fov_size)
        fov[fov_x, fov_y] = CellType.DOOR

    # Carried key at agent position — last so it overrides door when agent is on door cell
    if door_key_state >= 1:
        fov[half, fov_size - 1] = CellType.KEY

    if door_key_state != 2:  # Door not open - blocks visibility
        walls.add((door_x, door_y))

    relative_walls = set()
    for wall in walls:
        rel = get_relative_coords(agent_x, agent_y, orientation, wall[0], wall[1])
        if in_fov(*rel, fov_size):
            relative_walls.add(relative_to_fov_coords(*rel, fov_size))

    visibility_mask = generate_visibility_mask(half, fov_size - 1, fov_size, fov_size, relative_walls)
    for x in range(-half, half + 1):
        for y in range(fov_size):
            fov_x, fov_y = relative_to_fov_coords(x, y, fov_size)
            if not visibility_mask[fov_x, fov_y]:
                fov[fov_x, fov_y] = CellType.UNSEEN

    return fov


def get_next_orientation(orientation: int, action: int) -> int:
    if action == ActionType.TURN_LEFT:
        return (orientation + 3) % 4  # Counter-clockwise
    elif action == ActionType.TURN_RIGHT:
        return (orientation + 1) % 4  # Clockwise
    else:
        return orientation


def get_next_door_key_state(
    agent_x: int,
    agent_y: int,
    orientation: int,
    key_x: int,
    key_y: int,
    door_x: int,
    door_y: int,
    action: int,
    door_key_state: int,
) -> int:
    if action == ActionType.PICKUP:
        if door_key_state > 0:
            return door_key_state
        rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, key_x, key_y)
        if rel_x == 0 and rel_y == 1:
            return 1  # Key picked up
        return door_key_state

    if action != ActionType.TOGGLE:
        return door_key_state

    if door_key_state != 1:  # Need to have key but door not open
        return door_key_state

    rel_x, rel_y = get_relative_coords(agent_x, agent_y, orientation, door_x, door_y)
    if rel_x == 0 and rel_y == 1:
        return 2  # Door opened
    return door_key_state


def get_next_agent_position(
    agent_x: int,
    agent_y: int,
    orientation: int,
    door_x: int,
    door_y: int,
    key_x: int,
    key_y: int,
    door_key_state: int,
    action: int,
    n: int,
) -> int:
    if action == ActionType.FORWARD:
        new_x, new_y = agent_x, agent_y
        if orientation == Orientation.RIGHT:
            new_x += 1
        elif orientation == Orientation.DOWN:
            new_y += 1
        elif orientation == Orientation.LEFT:
            new_x -= 1
        else:  # UP
            new_y -= 1

        if new_x < 0 or new_x >= n or new_y < 0 or new_y >= n:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == key_x and new_y == key_y and door_key_state == 0:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == door_x and door_key_state != 2:
            return coords_to_state(agent_x, agent_y, n)
        if new_x == door_x and new_y != door_y:
            return coords_to_state(agent_x, agent_y, n)
        return coords_to_state(new_x, new_y, n)
    else:
        return coords_to_state(agent_x, agent_y, n)


def generate_observation_tensor(
    n: int, valid_configs: list[tuple[int, int]], fov_size: int = 7, dtype=np.float16
) -> np.ndarray:
    """Generate full observation tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    B = np.zeros((fov_size, fov_size, N_CELL_TYPES, n_total_states, n_static_states), dtype=dtype)

    for agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                key_x, key_y = key_position(key_pos, n)
                door_x, door_y = door_position(door_pos, n)

                for door_key_state in range(N_DOOR_KEY_STATES):
                    fov = get_fov(
                        agent_x,
                        agent_y,
                        orientation,
                        key_x,
                        key_y,
                        door_x,
                        door_y,
                        door_key_state,
                        n,
                        fov_size,
                    )
                    flat_state = flatten_state_index(
                        agent_state,
                        orientation,
                        door_key_state,
                        n_location_states,
                        N_ORIENTATIONS,
                        N_DOOR_KEY_STATES,
                    )
                    for i in range(fov_size):
                        for j in range(fov_size):
                            B[i, j, fov[i, j], flat_state, static_idx] = 1.0

    return B


def generate_orientation_observation_tensor(n: int, dtype=np.float16) -> np.ndarray:
    """Generate full orientation observation tensor (for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES

    B = np.zeros((N_ORIENTATIONS, n_total_states), dtype=dtype)

    for state_idx in range(n_total_states):
        _, orientation, _ = unflatten_state_index(
            state_idx, n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES
        )
        B[orientation, state_idx] = 1.0

    return B



def generate_transition_tensor(
    n: int, valid_configs: list[tuple[int, int]], dtype=np.float16
) -> np.ndarray:
    """Generate full transition tensor (memory-intensive, for reference/testing)."""
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    T = np.zeros(
        (n_total_states, n_total_states, n_static_states, N_ACTIONS), dtype=dtype
    )

    for old_agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(old_agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                    key_x, key_y = key_position(key_pos, n)
                    door_x, door_y = door_position(door_pos, n)

                    old_idx = flatten_state_index(
                        old_agent_state,
                        orientation,
                        door_key_state,
                        n_location_states,
                        N_ORIENTATIONS,
                        N_DOOR_KEY_STATES,
                    )

                    if agent_x == door_x and agent_y != door_y:
                        T[old_idx, old_idx, static_idx, :] = 1.0
                        continue

                    for action in range(N_ACTIONS):
                        new_agent_state = get_next_agent_position(
                            agent_x,
                            agent_y,
                            orientation,
                            door_x,
                            door_y,
                            key_x,
                            key_y,
                            door_key_state,
                            action,
                            n,
                        )
                        new_door_key_state = get_next_door_key_state(
                            agent_x,
                            agent_y,
                            orientation,
                            key_x,
                            key_y,
                            door_x,
                            door_y,
                            action,
                            door_key_state,
                        )
                        new_orientation = get_next_orientation(orientation, action)
                        new_idx = flatten_state_index(
                            new_agent_state,
                            new_orientation,
                            new_door_key_state,
                            n_location_states,
                            N_ORIENTATIONS,
                            N_DOOR_KEY_STATES,
                        )
                        T[new_idx, old_idx, static_idx, action] = 1.0

    return T



def observation_to_onehot(image: np.ndarray) -> np.ndarray:
    fov_w, fov_h = image.shape[0], image.shape[1]
    onehot = np.zeros((fov_w, fov_h, N_CELL_TYPES), dtype=np.float16)
    for i in range(fov_w):
        for j in range(fov_h):
            onehot[i, j, image[i, j]] = 1.0
    return onehot


def direction_to_onehot(direction: int) -> np.ndarray:
    onehot = np.zeros(N_ORIENTATIONS, dtype=np.float16)
    onehot[direction] = 1.0
    return onehot


def action_to_onehot(action: int) -> np.ndarray:
    onehot = np.zeros(N_ACTIONS, dtype=np.float16)
    onehot[action] = 1.0
    return onehot


def convert_action(action: int) -> int:
    return action


def contains_key(image: np.ndarray) -> bool:
    return CellType.KEY in image[:, :, 0]


def contains_door(image: np.ndarray) -> bool:
    return CellType.DOOR in image[:, :, 0]


def create_reward_observation_tensor_minigrid(
    n: int,
    n_static_states: int,
) -> np.ndarray:
    """
    Create reward observation tensor for minigrid: p(reward_obs | s, θ).

    Reward outcomes: {none=0, goal=1} → 2 outcomes.
    The goal is at position (n-1, n-1). The agent gets reward when at goal.
    This is θ-dependent since the static config determines layout.

    Shape: (2, n_total_states, n_static_states)
    """
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES

    goal_state = coords_to_state(n - 1, n - 1, n)

    R = np.zeros((2, n_total_states, n_static_states), dtype=np.float32)

    for state_idx in range(n_total_states):
        agent_state, orientation, door_key_state = unflatten_state_index(
            state_idx, n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES
        )
        if agent_state == goal_state:
            R[1, state_idx, :] = 1.0  # goal reached
        else:
            R[0, state_idx, :] = 1.0  # no reward

    return R


def collapse_fov_to_flat_modality(
    B: np.ndarray,
    fov_size: int,
) -> tuple[np.ndarray, list, dict]:
    """Collapse per-pixel FOV observations into a single flat modality.

    Enumerates unique full-FOV patterns (argmax per pixel) and builds
    a one-hot generative tensor over pattern indices.

    Args:
        B: One-hot observation tensor, shape (fov, fov, n_cell_types, n_states, n_static).
        fov_size: FOV grid dimension.

    Returns:
        B_flat: Generative tensor shape (n_patterns, n_states, n_static).
        pattern_list: Sorted list of unique pattern tuples.
        pattern_map: Dict mapping pattern tuple → pattern index.
    """
    _, _, _, n_states, n_static = B.shape
    n_pixels = fov_size * fov_size

    # For each (state, static), argmax cell type per pixel → flat tuple
    patterns = np.argmax(B, axis=2)  # (fov, fov, n_states, n_static)
    patterns_flat = patterns.reshape(n_pixels, n_states, n_static)

    # Collect unique patterns
    unique_patterns = set()
    for s in range(n_states):
        for th in range(n_static):
            key = tuple(int(x) for x in patterns_flat[:, s, th])
            unique_patterns.add(key)

    # Sort for deterministic ordering
    pattern_list = sorted(unique_patterns)
    pattern_map = {pat: idx for idx, pat in enumerate(pattern_list)}
    n_patterns = len(pattern_list)

    # Build one-hot flat generative tensor
    B_flat = np.zeros((n_patterns, n_states, n_static), dtype=np.float32)
    for s in range(n_states):
        for th in range(n_static):
            key = tuple(int(x) for x in patterns_flat[:, s, th])
            B_flat[pattern_map[key], s, th] = 1.0

    return B_flat, pattern_list, pattern_map


def generate_transition_index(
    n: int, valid_configs: list[tuple[int, int]],
) -> np.ndarray:
    """Generate transition index array (compact form of transition tensor).

    Returns:
        T_idx: shape (n_total_states, N_ACTIONS, n_static_states) dtype int32
               T_idx[old_state, action, theta] = new_state
               Layout puts n_static_states (multiple of 8) as innermost dim
               for GPU memory coalescing.
    """
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    T_idx = np.zeros((n_total_states, N_ACTIONS, n_static_states), dtype=np.int32)

    for old_agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(old_agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                old_idx = flatten_state_index(
                    old_agent_state, orientation, door_key_state,
                    n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES,
                )

                for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                    key_x, key_y = key_position(key_pos, n)
                    door_x, door_y = door_position(door_pos, n)

                    if agent_x == door_x and agent_y != door_y:
                        T_idx[old_idx, :, static_idx] = old_idx
                        continue

                    for action in range(N_ACTIONS):
                        new_agent_state = get_next_agent_position(
                            agent_x, agent_y, orientation,
                            door_x, door_y, key_x, key_y,
                            door_key_state, action, n,
                        )
                        new_door_key_state = get_next_door_key_state(
                            agent_x, agent_y, orientation,
                            key_x, key_y, door_x, door_y,
                            action, door_key_state,
                        )
                        new_orientation = get_next_orientation(orientation, action)
                        new_idx = flatten_state_index(
                            new_agent_state, new_orientation, new_door_key_state,
                            n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES,
                        )
                        T_idx[old_idx, action, static_idx] = new_idx

    return T_idx


def generate_fov_observation_index(
    n: int, valid_configs: list[tuple[int, int]], fov_size: int = 7,
) -> tuple[np.ndarray, list, dict]:
    """Generate FOV observation index directly from get_fov(), without intermediate tensor.

    For each (state, theta), computes the FOV pattern and maps it to a unique index.

    Returns:
        fov_idx: shape (n_total_states, n_static_states) dtype int32
        pattern_list: sorted list of unique FOV pattern tuples
        pattern_map: dict mapping pattern tuple -> index
    """
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    n_static_states = len(valid_configs)

    unique_patterns = set()
    pattern_grid = [[None] * n_static_states for _ in range(n_total_states)]

    for agent_state in range(n_location_states):
        agent_x, agent_y = state_to_coords(agent_state, n)

        for orientation in range(N_ORIENTATIONS):
            for door_key_state in range(N_DOOR_KEY_STATES):
                flat_state = flatten_state_index(
                    agent_state, orientation, door_key_state,
                    n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES,
                )

                for static_idx, (key_pos, door_pos) in enumerate(valid_configs):
                    key_x, key_y = key_position(key_pos, n)
                    door_x, door_y = door_position(door_pos, n)

                    fov = get_fov(
                        agent_x, agent_y, orientation,
                        key_x, key_y, door_x, door_y,
                        door_key_state, n, fov_size,
                    )
                    pattern = tuple(fov.ravel().tolist())
                    unique_patterns.add(pattern)
                    pattern_grid[flat_state][static_idx] = pattern

    pattern_list = sorted(unique_patterns)
    pattern_map = {pat: idx for idx, pat in enumerate(pattern_list)}

    fov_idx = np.zeros((n_total_states, n_static_states), dtype=np.int32)
    for s in range(n_total_states):
        for th in range(n_static_states):
            fov_idx[s, th] = pattern_map[pattern_grid[s][th]]

    return fov_idx, pattern_list, pattern_map


def create_minigrid_env_tensors(
    n: int,
    fov_size: int = 3,
) -> EnvironmentTensors:
    """
    Create EnvironmentTensors for the Minigrid Door-Key environment.

    Uses index arrays instead of dense tensors for transitions and FOV observations,
    avoiding massive intermediate allocations. Dense tensors are only built for
    small modalities (orientation, reward) and the collapsed FOV generative tensor.

    Args:
        n: Grid size (n x n)
        fov_size: Field-of-view size (must be odd)

    Returns:
        EnvironmentTensors with collapsed FOV modality + orientation + reward
    """
    n_location_states = n * n
    n_total_states = n_location_states * N_ORIENTATIONS * N_DOOR_KEY_STATES
    valid_configs = get_valid_static_configs(n)
    n_static_states = len(valid_configs)

    # Generate index arrays (compact — no massive dense tensors)
    T_idx = generate_transition_index(n, valid_configs)                 # (s, a, n_static) int32
    fov_idx, pattern_list, pattern_map = generate_fov_observation_index(
        n, valid_configs, fov_size
    )                                                                    # (s, n_static) int32
    n_fov_patterns = len(pattern_list)

    # Build collapsed FOV generative tensor from index (small: n_patterns × s × n_static)
    B_flat = np.zeros((n_fov_patterns, n_total_states, n_static_states), dtype=np.float32)
    for s in range(n_total_states):
        for th in range(n_static_states):
            B_flat[fov_idx[s, th], s, th] = 1.0
    B_flat_jax = jnp.array(B_flat, dtype=jnp.float32)
    fov_idx_jax = jnp.array(fov_idx, dtype=jnp.int32)

    # Small dense tensors (orientation and reward are tiny)
    O = generate_orientation_observation_tensor(n)                      # (4, s)
    R = create_reward_observation_tensor_minigrid(n, n_static_states)   # (2, s, n_static)
    O_jax = jnp.array(O, dtype=jnp.float32)
    R_jax = jnp.array(R, dtype=jnp.float32)

    # Transition index as JAX array
    T_idx_jax = jnp.array(T_idx, dtype=jnp.int32)

    modalities = []
    modalities.append(ObservationModality(
        name="fov",
        generative_tensor=B_flat_jax,
        theta_dependent=True,
        n_obs=n_fov_patterns,
        observation_index=fov_idx_jax,
    ))

    # Orientation modality (θ-independent)
    modalities.append(ObservationModality(
        name="orientation",
        generative_tensor=O_jax,
        theta_dependent=False,
        n_obs=N_ORIENTATIONS,
    ))

    # Reward modality (θ-dependent)
    modalities.append(ObservationModality(
        name="reward",
        generative_tensor=R_jax,
        theta_dependent=True,
        n_obs=2,
    ))

    # Uniform priors
    action_prior = jnp.ones(N_ACTIONS) / N_ACTIONS
    theta_prior = jnp.ones(n_static_states) / n_static_states

    # Goal mapping: prefer goal state, shape (n_total_states, n_static_states)
    goal_state = coords_to_state(n - 1, n - 1, n)
    goal_logits = jnp.full((n_total_states, n_static_states), -1.0)
    for orient in range(N_ORIENTATIONS):
        for dks in range(N_DOOR_KEY_STATES):
            flat = flatten_state_index(
                goal_state, orient, dks,
                n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES,
            )
            goal_logits = goal_logits.at[flat, :].set(1.0)
    goal_mapping = jax.nn.softmax(goal_logits * 5.0, axis=0)

    # Initial state prior: dks=0, x < n-2 (agent starts left of wall, without key)
    initial_state = np.zeros(n_total_states, dtype=np.float32)
    for loc in range(n_location_states):
        x, y = state_to_coords(loc, n)
        if x >= n - 2:
            continue
        for orient in range(N_ORIENTATIONS):
            flat = flatten_state_index(
                loc, orient, 0,
                n_location_states, N_ORIENTATIONS, N_DOOR_KEY_STATES,
            )
            initial_state[flat] = 1.0
    initial_state = initial_state / initial_state.sum()

    return EnvironmentTensors(
        n_states=n_total_states,
        n_actions=N_ACTIONS,
        n_theta=n_static_states,
        transition_tensor=None,
        theta_dependent_transitions=True,
        observation_modalities=modalities,
        goal_mapping=goal_mapping,
        initial_state=jnp.array(initial_state),
        action_prior=action_prior,
        theta_prior=theta_prior,
        metadata={"fov_pattern_map": pattern_map, "valid_configs": valid_configs},
        transition_index=T_idx_jax,
    )
