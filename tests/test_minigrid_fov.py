"""Validate get_fov() against MiniGrid's gen_obs_grid() as ground truth.

For each test scenario, the model's get_fov() output is compared cell-by-cell
against MiniGrid's gen_obs_grid()[:,:,0] (the object-type channel).
"""

import pytest
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import ViewSizeWrapper

from src.environments.minigrid import (
    CellType,
    ActionType,
    get_fov,
)
from src.environments.gym_wrapper import register_doorkey_env


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_gym_state(env):
    """Scan grid for key/door positions, agent state."""
    uw = env.unwrapped
    agent_pos = tuple(int(x) for x in uw.agent_pos)
    agent_dir = int(uw.agent_dir)
    carrying = uw.carrying

    key_pos = None
    door_pos = None
    door_is_open = False
    door_is_locked = True

    for i in range(uw.width):
        for j in range(uw.height):
            cell = uw.grid.get(i, j)
            if cell is not None:
                if cell.type == "key":
                    key_pos = (i, j)
                elif cell.type == "door":
                    door_pos = (i, j)
                    door_is_open = cell.is_open
                    door_is_locked = cell.is_locked

    return {
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "carrying": carrying,
        "key_pos": key_pos,
        "door_pos": door_pos,
        "door_is_open": door_is_open,
        "door_is_locked": door_is_locked,
    }


def gym_to_model_coords(gym_pos, grid_size):
    """Subtract 1 for outer wall offset: gym (1,1) -> model (0,0)."""
    return (gym_pos[0] - 1, gym_pos[1] - 1)


def model_door_key_state(carrying, door_open, door_locked):
    """Map gym state to model dks: 0=key on ground, 1=carrying, 2=door open."""
    if door_open:
        return 2
    elif carrying is not None and carrying.type == "key":
        return 1
    else:
        return 0


def get_gym_fov(env, fov_size):
    """Call gen_obs_grid(), return cell-type channel."""
    uw = env.unwrapped
    grid, vis_mask = uw.gen_obs_grid(fov_size)
    image = grid.encode(vis_mask)
    return image[:, :, 0]


def setup_env(grid_size, fov_size, seed):
    """Create env with ViewSizeWrapper, reset, return (env, state)."""
    if grid_size not in (5, 6, 8, 16):
        register_doorkey_env(grid_size)
    env = gym.make(f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0")
    if fov_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=fov_size)
        env.unwrapped.agent_view_size = fov_size
    env.reset(seed=seed)
    state = extract_gym_state(env)
    return env, state


def compare_fov(env, fov_size, grid_size, last_key_pos=None):
    """Compare model get_fov() vs gym gen_obs_grid(). Returns (model, gym, match)."""
    n = grid_size - 2
    state = extract_gym_state(env)
    ap = state["agent_pos"]
    ad = state["agent_dir"]
    kp = state["key_pos"]
    dp = state["door_pos"]
    carrying_key = state["carrying"] is not None and state["carrying"].type == "key"
    dks = model_door_key_state(
        state["carrying"], state["door_is_open"], state["door_is_locked"]
    )

    m_agent = gym_to_model_coords(ap, grid_size)
    m_door = gym_to_model_coords(dp, grid_size)

    if carrying_key:
        # key_x/key_y unused when dks >= 1; use last known or dummy
        m_key = gym_to_model_coords(last_key_pos, grid_size) if last_key_pos else (0, 0)
    else:
        m_key = gym_to_model_coords(kp, grid_size)

    gym_fov = get_gym_fov(env, fov_size)
    model_fov = get_fov(
        m_agent[0], m_agent[1], ad,
        m_key[0], m_key[1],
        m_door[0], m_door[1],
        dks, n, fov_size,
    )

    return model_fov, gym_fov, np.array_equal(model_fov, gym_fov)


def diff_msg(model_fov, gym_fov, **ctx):
    """Format an assertion message showing mismatched cells."""
    positions = list(zip(*np.where(gym_fov != model_fov)))
    lines = [f"FOV mismatch ({', '.join(f'{k}={v}' for k, v in ctx.items())})"]
    for i, j in positions[:10]:
        lines.append(f"  ({i},{j}): gym={int(gym_fov[i,j])} model={int(model_fov[i,j])}")
    if len(positions) > 10:
        lines.append(f"  ... and {len(positions) - 10} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def turn_actions(current_dir, target_dir):
    """Minimal turn-action sequence from current_dir to target_dir."""
    if current_dir == target_dir:
        return []
    steps_right = (target_dir - current_dir) % 4
    steps_left = (current_dir - target_dir) % 4
    if steps_right <= steps_left:
        return [int(ActionType.TURN_RIGHT)] * steps_right
    else:
        return [int(ActionType.TURN_LEFT)] * steps_left


def face_direction(env, target_dir):
    """Turn agent to face target_dir."""
    uw = env.unwrapped
    for a in turn_actions(int(uw.agent_dir), target_dir):
        env.step(a)


def dir_from_to(from_pos, to_pos):
    """Direction to face from from_pos toward adjacent to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    return None


def navigate_greedy(env, target_pos, max_steps=50):
    """Greedy navigation to target_pos. Returns True if reached."""
    uw = env.unwrapped
    for _ in range(max_steps):
        agent_pos = tuple(int(x) for x in uw.agent_pos)
        if agent_pos == target_pos:
            return True
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        dirs = []
        if dx > 0:
            dirs.append(0)
        if dx < 0:
            dirs.append(2)
        if dy > 0:
            dirs.append(1)
        if dy < 0:
            dirs.append(3)
        moved = False
        for d in dirs:
            face_direction(env, d)
            old_pos = tuple(int(x) for x in uw.agent_pos)
            env.step(int(ActionType.FORWARD))
            new_pos = tuple(int(x) for x in uw.agent_pos)
            if new_pos != old_pos:
                moved = True
                break
        if not moved:
            return False
    return tuple(int(x) for x in uw.agent_pos) == target_pos


def approach_and_face(env, target_pos):
    """Navigate to a cell adjacent to target, facing it. Returns True on success."""
    uw = env.unwrapped
    agent_pos = tuple(int(x) for x in uw.agent_pos)

    candidates = [
        ((target_pos[0] - 1, target_pos[1]), 0),  # from left, face RIGHT
        ((target_pos[0], target_pos[1] - 1), 1),  # from above, face DOWN
        ((target_pos[0] + 1, target_pos[1]), 2),  # from right, face LEFT
        ((target_pos[0], target_pos[1] + 1), 3),  # from below, face UP
    ]
    # Sort by Manhattan distance to current agent position
    candidates.sort(
        key=lambda c: abs(c[0][0] - agent_pos[0]) + abs(c[0][1] - agent_pos[1])
    )

    for approach_pos, face_dir in candidates:
        if not (
            1 <= approach_pos[0] < uw.width - 1
            and 1 <= approach_pos[1] < uw.height - 1
        ):
            continue
        if navigate_greedy(env, approach_pos):
            face_direction(env, face_dir)
            return True
    return False


# ---------------------------------------------------------------------------
# Scenario setup helpers
# ---------------------------------------------------------------------------


def pickup_key_sequence(env):
    """Navigate to key and pick it up. Returns (success, key_gym_pos)."""
    state = extract_gym_state(env)
    key_pos = state["key_pos"]
    if key_pos is None:
        return False, None

    if not approach_and_face(env, key_pos):
        return False, None

    env.step(int(ActionType.PICKUP))

    new_state = extract_gym_state(env)
    carrying = new_state["carrying"] is not None and new_state["carrying"].type == "key"
    return carrying, key_pos


def open_door_sequence(env):
    """Navigate to door and toggle it open. Returns success."""
    state = extract_gym_state(env)
    door_pos = state["door_pos"]
    if door_pos is None:
        return False

    if not approach_and_face(env, door_pos):
        return False

    env.step(int(ActionType.TOGGLE))

    new_state = extract_gym_state(env)
    return new_state["door_is_open"]


# ===========================================================================
# A. Initial state (broad coverage)
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 5, 7])
@pytest.mark.parametrize("seed", range(50))
def test_fov_initial_state(grid_size, fov_size, seed):
    """Reset env, compare FOV. Covers all 4 orientations and diverse layouts."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        assert match, diff_msg(
            model_fov, gym_fov, seed=seed, grid=grid_size, fov=fov_size
        )
    finally:
        env.close()


# ===========================================================================
# B. Wall visibility from all angles
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("orientation", [0, 1, 2, 3])
def test_fov_facing_wall(grid_size, fov_size, seed, orientation):
    """Turn agent to each orientation, compare FOV."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        face_direction(env, orientation)
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        orient_name = ["RIGHT", "DOWN", "LEFT", "UP"][orientation]
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, facing=orient_name,
        )
    finally:
        env.close()


# ===========================================================================
# C. Key on ground — visible / not visible
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_key_visible(grid_size, fov_size, seed):
    """Agent facing key (1 cell ahead). Key should appear in FOV."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        if not approach_and_face(env, state["key_pos"]):
            pytest.skip("Cannot navigate to face key")
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        assert CellType.KEY in gym_fov, "Key not visible in gym FOV"
        assert match, diff_msg(
            model_fov, gym_fov, seed=seed, grid=grid_size, fov=fov_size, test="key_visible"
        )
    finally:
        env.close()


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_key_behind(grid_size, fov_size, seed):
    """Agent facing away from key. Key should be UNSEEN or absent."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        if not approach_and_face(env, state["key_pos"]):
            pytest.skip("Cannot navigate to face key")
        # Turn 180 degrees
        uw = env.unwrapped
        away_dir = (int(uw.agent_dir) + 2) % 4
        face_direction(env, away_dir)
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        assert match, diff_msg(
            model_fov, gym_fov, seed=seed, grid=grid_size, fov=fov_size, test="key_behind"
        )
    finally:
        env.close()


# ===========================================================================
# D. Carried key (dks=1)
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_carrying_key(grid_size, fov_size, seed):
    """After pickup, agent in open area. KEY should appear at agent position."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert match, diff_msg(
            model_fov, gym_fov, seed=seed, grid=grid_size, fov=fov_size, test="carrying_key"
        )
    finally:
        env.close()


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_carrying_key_facing_door(grid_size, fov_size, seed):
    """After pickup, facing locked door. Door visible + KEY at agent pos."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not approach_and_face(env, state["door_pos"]):
            pytest.skip("Cannot navigate to face door")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert CellType.DOOR in gym_fov, "Door not visible in gym FOV"
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="carrying_key_facing_door",
        )
    finally:
        env.close()


# ===========================================================================
# E. Door states
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_facing_locked_door(grid_size, fov_size, seed):
    """Agent facing locked door (dks=0), no key. Door should block visibility."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        if not approach_and_face(env, state["door_pos"]):
            pytest.skip("Cannot navigate to face door")
        new_state = extract_gym_state(env)
        if new_state["carrying"] is not None:
            pytest.skip("Accidentally picked up key during navigation")
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        assert CellType.DOOR in gym_fov, "Door not visible"
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="facing_locked_door",
        )
    finally:
        env.close()


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_facing_closed_door_with_key(grid_size, fov_size, seed):
    """Agent facing door (dks=1). Door visible, cells behind UNSEEN."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not approach_and_face(env, state["door_pos"]):
            pytest.skip("Cannot navigate to face door")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert CellType.DOOR in gym_fov, "Door not visible"
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="closed_door_with_key",
        )
    finally:
        env.close()


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_after_toggle(grid_size, fov_size, seed):
    """Door just opened (dks=2), agent in front. Cells behind door now VISIBLE."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not open_door_sequence(env):
            pytest.skip("Cannot open door")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="after_toggle",
        )
    finally:
        env.close()


# ===========================================================================
# F. Agent on open door cell
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_on_open_door(grid_size, fov_size, seed):
    """Agent standing ON the open door cell (dks=2).

    Gym shows KEY at agent pos (carried object overrides door).
    Tests the render-order fix.
    """
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not open_door_sequence(env):
            pytest.skip("Cannot open door")
        # Walk forward onto the door cell
        env.step(int(ActionType.FORWARD))
        uw = env.unwrapped
        agent_pos = tuple(int(x) for x in uw.agent_pos)
        if agent_pos != state["door_pos"]:
            pytest.skip("Agent not on door cell after forward")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="on_open_door",
        )
    finally:
        env.close()


# ===========================================================================
# G. Goal visibility
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_goal_visible(grid_size, fov_size, seed):
    """Agent facing goal at (n-1, n-1). FOV should contain GOAL cell type."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        goal_pos = (grid_size - 2, grid_size - 2)
        # Must go through door to reach goal
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not open_door_sequence(env):
            pytest.skip("Cannot open door")
        if not approach_and_face(env, goal_pos):
            pytest.skip("Cannot navigate to face goal")
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert CellType.GOAL in gym_fov, "Goal not visible in gym FOV"
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="goal_visible",
        )
    finally:
        env.close()


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(10))
def test_fov_goal_behind_wall(grid_size, fov_size, seed):
    """Agent on left side, goal on right behind wall. Goal should be UNSEEN."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        # Face RIGHT toward the wall (goal is behind it)
        face_direction(env, 0)
        model_fov, gym_fov, match = compare_fov(env, fov_size, grid_size)
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="goal_behind_wall",
        )
    finally:
        env.close()


# ===========================================================================
# H. Through door — right side of grid
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(20))
def test_fov_past_door(grid_size, fov_size, seed):
    """Full sequence: pickup -> toggle -> walk through -> compare from right side."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")
        success, key_pos = pickup_key_sequence(env)
        if not success:
            pytest.skip("Cannot pick up key")
        if not open_door_sequence(env):
            pytest.skip("Cannot open door")
        # Walk through door
        env.step(int(ActionType.FORWARD))
        # One more step to be past it
        env.step(int(ActionType.FORWARD))
        model_fov, gym_fov, match = compare_fov(
            env, fov_size, grid_size, last_key_pos=key_pos
        )
        assert match, diff_msg(
            model_fov, gym_fov,
            seed=seed, grid=grid_size, fov=fov_size, test="past_door",
        )
    finally:
        env.close()


# ===========================================================================
# I. Full episode rollout
# ===========================================================================


@pytest.mark.parametrize("grid_size", [5, 7])
@pytest.mark.parametrize("fov_size", [3, 7])
@pytest.mark.parametrize("seed", range(10))
def test_fov_full_episode(grid_size, fov_size, seed):
    """Run 20-step random episode, compare at EVERY step."""
    env, state = setup_env(grid_size, fov_size, seed)
    try:
        if state["key_pos"] is None or state["door_pos"] is None:
            pytest.skip("Invalid layout")

        rng = np.random.default_rng(seed)
        last_key_pos = state["key_pos"]

        for step in range(20):
            model_fov, gym_fov, match = compare_fov(
                env, fov_size, grid_size, last_key_pos=last_key_pos
            )
            assert match, diff_msg(
                model_fov, gym_fov,
                seed=seed, grid=grid_size, fov=fov_size, step=step,
            )

            action = int(rng.integers(0, 7))
            obs, reward, terminated, truncated, info = env.step(action)

            # Track key position (it disappears from grid when carried)
            new_state = extract_gym_state(env)
            if new_state["key_pos"] is not None:
                last_key_pos = new_state["key_pos"]

            if terminated or truncated:
                break
    finally:
        env.close()
