from utils import *
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Wall, Goal, Key, Door
import os
import sys
import heapq
import pickle
import numpy as np
from collections import defaultdict

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

ACTION_NAMES = {MF: "MF", TL: "TL", TR: "TR", PK: "PK", UD: "UD"}
DIRS = {
    0: (1, 0),    # right
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (0, -1),   # up
}
ACTIONS = (MF, TL, TR, PK, UD)

# Random-map family from create_env.py / project PDF
RANDOM_KEY_POSITIONS = ((2, 2), (2, 3), (1, 6))
RANDOM_GOAL_POSITIONS = ((6, 1), (7, 3), (6, 6))
RANDOM_DOOR_POSITIONS = ((5, 3), (5, 7))
RANDOM_SIZE = 10


class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

# Make pickle loading robust for .env files created when create_env.py was __main__.
setattr(sys.modules["__main__"], "DoorKey10x10Env", DoorKey10x10Env)
try:
    register(
        id="MiniGrid-DoorKey-10x10-v0",
        entry_point="__main__:DoorKey10x10Env",
    )
except Exception:
    pass


def action_cost(action):
    if action in (TL, TR):
        return 1
    if action == MF:
        return 3
    if action == PK:
        return 2
    if action == UD:
        return 5
    raise ValueError(f"Unknown action: {action}")


def sequence_cost(seq):
    return sum(action_cost(a) for a in seq)


def seq_to_names(seq):
    return [ACTION_NAMES[a] for a in seq]


def dir_to_int(env):
    if hasattr(env.unwrapped, "agent_dir"):
        return int(env.unwrapped.agent_dir)
    v = tuple(np.asarray(env.unwrapped.dir_vec).tolist())
    return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[v]


def scan_env(env):
    height, width = env.unwrapped.height, env.unwrapped.width
    walls, keys, goals, doors = set(), set(), set(), []

    for y in range(height):
        for x in range(width):
            obj = env.unwrapped.grid.get(x, y)
            if isinstance(obj, Wall):
                walls.add((x, y))
            elif isinstance(obj, Key):
                keys.add((x, y))
            elif isinstance(obj, Goal):
                goals.add((x, y))
            elif isinstance(obj, Door):
                doors.append({
                    "pos": (x, y),
                    "initial_open": bool(obj.is_open),
                    "initial_locked": bool(obj.is_locked),
                })

    doors.sort(key=lambda item: item["pos"])
    door_index = {item["pos"]: i for i, item in enumerate(doors)}
    return width, height, walls, keys, goals, doors, door_index


def transition_general(state, action, maps):
    """Transition for arbitrary loaded maps. Used for Part (A)."""
    width, height, walls, keys, goals, doors, door_index = maps
    x, y, d, has_key, door_open_tuple = state
    door_open = list(door_open_tuple)

    if action == TL:
        return (x, y, (d - 1) % 4, has_key, tuple(door_open))
    if action == TR:
        return (x, y, (d + 1) % 4, has_key, tuple(door_open))

    dx, dy = DIRS[d]
    front = (x + dx, y + dy)
    fx, fy = front
    if fx < 0 or fx >= width or fy < 0 or fy >= height:
        return state

    if front in walls:
        kind = "wall"
    elif front in keys and not has_key:
        kind = "key"
    elif front in door_index:
        kind = "door"
    elif front in goals:
        kind = "goal"
    else:
        kind = "empty"

    if action == PK:
        if (not has_key) and kind == "key":
            return (x, y, d, True, tuple(door_open))
        return state

    if action == UD:
        if has_key and kind == "door":
            door_open[door_index[front]] = True
            return (x, y, d, has_key, tuple(door_open))
        return state

    if action == MF:
        if kind in ("wall", "key"):
            return state
        if kind == "door" and not door_open[door_index[front]]:
            return state
        return (fx, fy, d, has_key, tuple(door_open))

    raise ValueError(f"Unknown action: {action}")


def is_goal_general(state, goals):
    return (state[0], state[1]) in goals


def known_map(env):
    """Part (A): map-specific DP shortest-path solver for known maps."""
    maps = scan_env(env)
    width, height, walls, keys, goals, doors, door_index = maps
    init_open = tuple(item["initial_open"] for item in doors)
    init_state = (
        int(env.unwrapped.agent_pos[0]),
        int(env.unwrapped.agent_pos[1]),
        dir_to_int(env),
        env.unwrapped.carrying is not None,
        init_open,
    )

    pq = [(0, init_state)]
    dist = {init_state: 0}
    parent = {init_state: (None, None)}

    while pq:
        cost, state = heapq.heappop(pq)
        if cost != dist[state]:
            continue
        if is_goal_general(state, goals):
            seq = []
            cur = state
            while parent[cur][0] is not None:
                prev, act = parent[cur]
                seq.append(act)
                cur = prev
            return list(reversed(seq))

        for action in ACTIONS:
            nxt = transition_general(state, action, maps)
            new_cost = cost + action_cost(action)
            if new_cost < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                parent[nxt] = (state, action)
                heapq.heappush(pq, (new_cost, nxt))

    raise RuntimeError("No feasible path found for known map.")


# ----------------------- True single-policy Part (B) -----------------------
# The policy state explicitly includes the random-map configuration. Therefore
# one unified table pi(state) works for all 36 random environments.
#
# policy key:
#   (agent_x, agent_y, direction, has_key, key_pos, goal_pos, door_open_tuple)
# value:
#   best action in {MF, TL, TR, PK, UD}

_RANDOM_POLICY = None
_RANDOM_VALUE = None


def random_walls():
    # create_env.py uses env.grid.vert_wall(5, 0), then replaces (5,3),(5,7) by doors.
    return {(5, y) for y in range(RANDOM_SIZE) if (5, y) not in RANDOM_DOOR_POSITIONS}


def random_transition(state, action):
    x, y, d, has_key, key_pos, goal_pos, door_open_tuple = state
    door_open = list(door_open_tuple)
    walls = random_walls()
    door_index = {pos: i for i, pos in enumerate(RANDOM_DOOR_POSITIONS)}

    if action == TL:
        return (x, y, (d - 1) % 4, has_key, key_pos, goal_pos, tuple(door_open))
    if action == TR:
        return (x, y, (d + 1) % 4, has_key, key_pos, goal_pos, tuple(door_open))

    dx, dy = DIRS[d]
    front = (x + dx, y + dy)
    fx, fy = front
    if fx < 0 or fx >= RANDOM_SIZE or fy < 0 or fy >= RANDOM_SIZE:
        return state

    front_is_key = (front == key_pos and not has_key)
    front_is_door = front in door_index
    front_is_wall = front in walls

    if action == PK:
        if front_is_key and not has_key:
            return (x, y, d, True, key_pos, goal_pos, tuple(door_open))
        return state

    if action == UD:
        if has_key and front_is_door:
            door_open[door_index[front]] = True
            return (x, y, d, has_key, key_pos, goal_pos, tuple(door_open))
        return state

    if action == MF:
        if front_is_wall or front_is_key:
            return state
        if front_is_door and not door_open[door_index[front]]:
            return state
        return (fx, fy, d, has_key, key_pos, goal_pos, tuple(door_open))

    raise ValueError(f"Unknown action: {action}")


def random_is_goal(state):
    x, y, *_rest = state
    goal_pos = state[5]
    return (x, y) == goal_pos


def enumerate_random_states_for_config(key_pos, goal_pos):
    walls = random_walls()
    traversable = [
        (x, y)
        for x in range(RANDOM_SIZE)
        for y in range(RANDOM_SIZE)
        if (x, y) not in walls
    ]
    for x, y in traversable:
        for d in range(4):
            for has_key in (False, True):
                for o1 in (False, True):
                    for o2 in (False, True):
                        yield (x, y, d, has_key, key_pos, goal_pos, (o1, o2))


def build_single_random_policy():
    """
    Build one policy table for the entire Part (B) random-map family.

    This is backward dynamic programming on the reverse graph.
    It is a single policy because the policy key includes the map parameters
    (key position, goal position, and door-open states), not a file name.
    """
    global _RANDOM_POLICY, _RANDOM_VALUE
    if _RANDOM_POLICY is not None:
        return _RANDOM_POLICY, _RANDOM_VALUE

    policy = {}
    value = {}

    for key_pos in RANDOM_KEY_POSITIONS:
        for goal_pos in RANDOM_GOAL_POSITIONS:
            states = list(enumerate_random_states_for_config(key_pos, goal_pos))

            reverse_edges = defaultdict(list)
            for s in states:
                for a in ACTIONS:
                    sp = random_transition(s, a)
                    reverse_edges[sp].append((s, a, action_cost(a)))

            pq = []
            dist = {}
            for s in states:
                if random_is_goal(s):
                    dist[s] = 0
                    heapq.heappush(pq, (0, s))

            while pq:
                cur_cost, s = heapq.heappop(pq)
                if cur_cost != dist[s]:
                    continue
                for prev, act, c in reverse_edges[s]:
                    new_cost = cur_cost + c
                    if new_cost < dist.get(prev, float("inf")):
                        dist[prev] = new_cost
                        policy[prev] = act
                        heapq.heappush(pq, (new_cost, prev))

            value.update(dist)

    _RANDOM_POLICY = policy
    _RANDOM_VALUE = value
    return policy, value


def is_random_family_env(env):
    width, height, walls, keys, goals, doors, door_index = scan_env(env)
    if width != RANDOM_SIZE or height != RANDOM_SIZE:
        return False
    door_positions = tuple(sorted([item["pos"] for item in doors]))
    if door_positions != RANDOM_DOOR_POSITIONS:
        return False
    if len(keys) != 1 or len(goals) != 1:
        return False
    if next(iter(keys)) not in RANDOM_KEY_POSITIONS:
        return False
    if next(iter(goals)) not in RANDOM_GOAL_POSITIONS:
        return False
    return True


def random_state_from_env(env):
    _width, _height, _walls, keys, goals, doors, _door_index = scan_env(env)
    key_pos = next(iter(keys))
    goal_pos = next(iter(goals))
    doors = sorted(doors, key=lambda item: item["pos"])
    door_open = tuple(bool(item["initial_open"]) for item in doors)
    return (
        int(env.unwrapped.agent_pos[0]),
        int(env.unwrapped.agent_pos[1]),
        dir_to_int(env),
        env.unwrapped.carrying is not None,
        key_pos,
        goal_pos,
        door_open,
    )


def rollout_single_policy(env, max_steps=200):
    """Execute the precomputed single Part (B) policy in the abstract MDP."""
    policy, value = build_single_random_policy()
    state = random_state_from_env(env)
    seq = []

    for _ in range(max_steps):
        if random_is_goal(state):
            return seq
        if state not in policy:
            raise RuntimeError(f"Single policy has no action for state: {state}")
        act = policy[state]
        seq.append(act)
        state = random_transition(state, act)

    raise RuntimeError("Policy rollout exceeded max_steps; possible loop.")


def doorkey_problem(env):
    """
    Main entry point.

    Part (B): If the loaded environment is from the 10x10 random-map family,
    use the true single control policy pi(x) precomputed over the entire family.

    Part (A): Otherwise, solve the known map using DP.
    """
    if is_random_family_env(env):
        return rollout_single_policy(env)
    return known_map(env)


def partA():
    env_paths = [
        "./envs/known_envs/doorkey-5x5-normal.env",
        "./envs/known_envs/doorkey-6x6-normal.env",
        "./envs/known_envs/doorkey-8x8-normal.env",
        "./envs/known_envs/doorkey-6x6-direct.env",
        "./envs/known_envs/doorkey-8x8-direct.env",
        "./envs/known_envs/doorkey-6x6-shortcut.env",
        "./envs/known_envs/doorkey-8x8-shortcut.env",
    ]
    os.makedirs("./gif", exist_ok=True)
    for env_path in env_paths:
        env, _ = load_env(env_path)
        seq = doorkey_problem(env)
        name = os.path.splitext(os.path.basename(env_path))[0]
        print(f"{name}: cost={sequence_cost(seq)}, steps={len(seq)}, seq={seq_to_names(seq)}")
        draw_gif_from_seq(seq, load_env(env_path)[0], path=f"./gif/partA_{name}.gif")


def _numeric_env_id(filename):
    return int(filename.split("-")[-1].split(".")[0])


def partB():
    # Build the policy once. This is the single policy used for all files.
    policy, value = build_single_random_policy()
    print(f"Single Part(B) policy size: {len(policy)} states")

    env_folder = "./envs/random_envs"
    env_files = sorted(
        [f for f in os.listdir(env_folder) if f.endswith(".env")],
        key=_numeric_env_id,
    )
    os.makedirs("./gif", exist_ok=True)

    for env_file in env_files:
        env_path = os.path.join(env_folder, env_file)
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        seq = doorkey_problem(env)
        idx = _numeric_env_id(env_file)
        print(f"{env_file}: cost={sequence_cost(seq)}, steps={len(seq)}, seq={seq_to_names(seq)}")
        # 36 GIFs.
        draw_gif_from_seq(seq, pickle.load(open(env_path, "rb")), path=f"./gif/partB_{idx:02d}.gif")


if __name__ == "__main__":
    partA()
    partB()
