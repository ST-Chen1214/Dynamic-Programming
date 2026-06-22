# ECE276B Project 1 DoorKey DP

## Main files
- `doorkey.py`: main entry point. It implements a dynamic programming planner for DoorKey.
- `utils.py`: utility functions.

## How to run

1. Run all known-map and random-map tests:

```bash
python doorkey.py
```

2. Create visualizations:

```bash
python trajectory_visualization.py
```

The program prints the optimal policy sequence, total cost, and number of policies for each environment. GIFs for the known and random maps are saved in the `gif/` folder.

## Action encoding

| Code | Policy |
|---|---|
| 0 | MF, move forward |
| 1 | TL, turn left |
| 2 | TR, turn right |
| 3 | PK, pick up key |
| 4 | UD, unlock door |

## Cost function

| Action | Cost |
|---|---:|
| TL / TR | 1 |
| MF | 3 |
| PK | 2 |
| UD | 5 |

## Planner summary

The planner uses dynamic programming because every policy has a positive but non-uniform cost. This is equivalent to dynamic programming on the deterministic shortest-path MDP. The returned sequence minimizes the accumulated stage cost until the agent reaches the goal.
