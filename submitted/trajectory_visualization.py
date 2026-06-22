"""
Trajectory visualization for ECE276B Project 1 Door-Key.

Put this file in the same folder as doorkey.py, utils.py, envs/, and gif/.
It produces static PNG figures for the report.

Usage:
    python trajectory_visualization.py

Outputs:
    figures/partA_*.png
    figures/partB_DoorKey-10x10-*.png
"""

import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.world_object import Wall, Key, Door, Goal

from utils import load_env, step
from doorkey import doorkey_problem, sequence_cost, seq_to_names

from minigrid.envs.doorkey import DoorKeyEnv
from gymnasium.envs.registration import register

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)




MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pick Up Key
UD = 4  # Unlock Door

ACTION_NAMES = {
    MF: "MF",
    TL: "TL",
    TR: "TR",
    PK: "PK",
    UD: "UD",
}

DIR_SYMBOLS = {
    0: ">",  # right
    1: "v",  # down
    2: "<",  # left
    3: "^",  # up
}


def numeric_id_from_filename(filename: str) -> int:
    """Extract the final number from DoorKey-10x10-36.env."""
    match = re.search(r"-(\d+)\.env$", filename)
    if match is None:
        return 10**9
    return int(match.group(1))


def scan_static_objects(env):
    """Return map objects for plotting."""
    width, height = env.unwrapped.width, env.unwrapped.height
    walls, keys, doors, goals = [], [], [], []

    for y in range(height):
        for x in range(width):
            obj = env.unwrapped.grid.get(x, y)
            if isinstance(obj, Wall):
                walls.append((x, y))
            elif isinstance(obj, Key):
                keys.append((x, y))
            elif isinstance(obj, Door):
                doors.append((x, y, obj.is_open, obj.is_locked))
            elif isinstance(obj, Goal):
                goals.append((x, y))

    return width, height, walls, keys, doors, goals


def rollout_positions(env, seq):
    """Execute action sequence and record positions, directions, and actions."""
    positions = [tuple(env.unwrapped.agent_pos)]
    directions = [int(env.unwrapped.agent_dir)]
    actions_taken = []

    for action in seq:
        actions_taken.append(action)
        step(env, action)
        positions.append(tuple(env.unwrapped.agent_pos))
        directions.append(int(env.unwrapped.agent_dir))

    return positions, directions, actions_taken


def plot_trajectory(env, seq, title, out_path, show_step_numbers=True):
    """
    Plot static trajectory overlay:
    - gray cells: walls
    - yellow marker: key
    - orange marker: door
    - green marker: goal
    - red star: start
    - black line/arrows: trajectory
    """
    # Use a copy so plotting does not modify the original environment.
    env_for_rollout = copy.deepcopy(env)
    width, height, walls, keys, doors, goals = scan_static_objects(env)
    positions, directions, actions_taken = rollout_positions(env_for_rollout, seq)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw grid background.
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_aspect("equal")

    # Draw walls.
    for x, y in walls:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, alpha=0.35))
        ax.text(x, y, "W", ha="center", va="center", fontsize=9)

    # Draw keys.
    for x, y in keys:
        ax.scatter(x, y, s=260, marker="*", label="Key" if "Key" not in ax.get_legend_handles_labels()[1] else None)
        ax.text(x, y + 0.28, "K", ha="center", va="center", fontsize=10, fontweight="bold")

    # Draw doors.
    for x, y, is_open, is_locked in doors:
        label = "Door"
        ax.scatter(x, y, s=220, marker="s", label=label if label not in ax.get_legend_handles_labels()[1] else None)
        door_text = "D(open)" if is_open else "D"
        ax.text(x, y, door_text, ha="center", va="center", fontsize=8, fontweight="bold")

    # Draw goals.
    for x, y in goals:
        ax.scatter(x, y, s=260, marker="P", label="Goal" if "Goal" not in ax.get_legend_handles_labels()[1] else None)
        ax.text(x, y + 0.28, "G", ha="center", va="center", fontsize=10, fontweight="bold")

    # Draw trajectory line and arrows.
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, linewidth=2.5, marker="o", label="Trajectory")

    # Draw arrows only when the agent changes cell.
    for i in range(len(positions) - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        dx, dy = x1 - x0, y1 - y0
        if dx != 0 or dy != 0:
            ax.arrow(x0, y0, dx * 0.72, dy * 0.72,
                     head_width=0.13, length_includes_head=True)

    # Start and final agent direction.
    sx, sy = positions[0]
    gx, gy = positions[-1]
    ax.scatter(sx, sy, s=300, marker="*", label="Start")
    ax.text(sx, sy - 0.30, f"S{DIR_SYMBOLS[directions[0]]}",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.scatter(gx, gy, s=220, marker="X", label="End")
    ax.text(gx, gy - 0.30, f"E{DIR_SYMBOLS[directions[-1]]}",
            ha="center", va="center", fontsize=10, fontweight="bold")

    if show_step_numbers:
        # Number only real movement positions; repeated cells from turn/pick/unlock are skipped.
        last = None
        for idx, (x, y) in enumerate(positions):
            if (x, y) != last:
                ax.text(x - 0.28, y - 0.28, str(idx), fontsize=8)
                last = (x, y)

    cost = sequence_cost(seq)
    action_text = " -> ".join(seq_to_names(seq))
    if len(action_text) > 90:
        action_text = action_text[:90] + " ..."

    ax.set_title(f"{title}\nCost = {cost}, Steps = {len(seq)}")
    ax.set_xlabel(f"Actions: {action_text}")
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def visualize_one_env(env_path, out_path):
    env, info = load_env(env_path)
    seq = doorkey_problem(env)
    # Reload clean env for plotting because doorkey_problem should not mutate env,
    # but this keeps the workflow safe.
    env_clean, _ = load_env(env_path)
    title = os.path.splitext(os.path.basename(env_path))[0]
    plot_trajectory(env_clean, seq, title, out_path)
    return seq


def visualize_partA():
    env_paths = [
        "./envs/known_envs/doorkey-5x5-normal.env",
        "./envs/known_envs/doorkey-6x6-normal.env",
        "./envs/known_envs/doorkey-8x8-normal.env",
        "./envs/known_envs/doorkey-6x6-direct.env",
        "./envs/known_envs/doorkey-8x8-direct.env",
        "./envs/known_envs/doorkey-6x6-shortcut.env",
        "./envs/known_envs/doorkey-8x8-shortcut.env",
    ]

    for env_path in env_paths:
        name = os.path.splitext(os.path.basename(env_path))[0]
        visualize_one_env(env_path, f"./figures/partA_{name}.png")


def visualize_partB(selected_ids=None):
    """
    Visualize Part B random envs in filename-number order.

    selected_ids=None means visualize all 36.
    selected_ids=[1, 4, 18, 27, 36] means visualize only those cases.
    """
    folder = "./envs/random_envs"
    env_files = [f for f in os.listdir(folder) if f.endswith(".env")]
    env_files = sorted(env_files, key=numeric_id_from_filename)

    selected_set = None if selected_ids is None else set(selected_ids)

    for fname in env_files:
        idx = numeric_id_from_filename(fname)
        if selected_set is not None and idx not in selected_set:
            continue
        env_path = os.path.join(folder, fname)
        out_path = f"./figures/partB_{os.path.splitext(fname)[0]}.png"
        visualize_one_env(env_path, out_path)


def main():
    visualize_partA()

    # Option 1: visualize all 36 random environments.
    visualize_partB(selected_ids=None)

    # Option 2: for a shorter report, comment the line above and use representative cases:
    # visualize_partB(selected_ids=[1, 4, 18, 27, 36])


if __name__ == "__main__":
    main()
