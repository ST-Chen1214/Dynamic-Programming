# Cost-Optimal Door-Key Navigation

A discrete-state planning system that computes minimum-cost MiniGrid trajectories for seven known layouts and one reusable state-feedback policy for an exhaustive family of $36$ parameterized $10 \times 10$ environments.

## Animated results

### Fixed layouts: knowing when interaction matters

| Key and door required | Irrelevant subgoals ignored |
| :---: | :---: |
| <img src="submitted/gif/partA_doorkey-8x8-normal.gif" width="100%" alt="Cost-optimal trajectory on the 8-by-8 normal Door-Key map"/> | <img src="submitted/gif/partA_doorkey-8x8-direct.gif" width="100%" alt="Direct cost-optimal trajectory on the 8-by-8 Door-Key map"/> |
| Cost $54$; $21$ actions.<br>The agent makes a long detour to collect the key, returns to the partition, unlocks the door, and then reaches the goal. | Cost $17$; $7$ actions.<br>Because the goal is already reachable, the planner correctly avoids the key and locked door instead of following a hard-coded interaction sequence. |

### One policy, different door states

| Lower passage open | Both passages locked |
| :---: | :---: |
| <img src="submitted/gif/partB_03.gif" width="100%" alt="Unified policy using the open lower door in random environment 3"/> | <img src="submitted/gif/partB_04.gif" width="100%" alt="Unified policy collecting a key and unlocking a door in random environment 4"/> |
| Cost $29$; $11$ actions.<br>With the upper door locked and lower door open, the policy bypasses the key and crosses through the available passage. | Cost $50$; $19$ actions.<br>For the same key and goal locations, closing the lower door changes the optimal strategy: retrieve the key, unlock the upper door, then cross. |

> **Visualization note:** MiniGrid renders only the agent's current view, so black cells are outside the visible region in that frame. The planner itself uses the complete loaded map state; this is a planning system, not a learned perception policy. All repository GIFs loop continuously at the intended $0.8\,\mathrm{s}$ per frame.

| Verified goal-reaching rollouts | Random-family policy entries | Scenario-specific result sets |
| :---: | :---: | :---: |
| $43/43$ | $26{,}208$ | $43\ \text{GIFs} + 43\ \text{trajectory plots}$ |

## Overview

The task is to move a simulated robot to a goal while accounting for walls, orientation, keys, and locked doors. The objective is not simply to minimize the number of actions: each action has a different positive cost, so the planner must trade cheap rotations against forward motion and object interactions.

The implementation models the environment as a deterministic weighted shortest-path Markov decision process. It uses an augmented state to remember task progress, an exact cost-aware planner for known maps, and a precomputed feedback table that covers every supplied key, goal, and two-door configuration in the random-map family.

**[Read the full technical report](ece276b_hw1_report.pdf)** - MDP formulation, Bellman optimality equations, all $43$ annotated trajectories, and a discussion of limitations. The original [project specification](ECE276_PR1.pdf) is also included.

## What I built

- A MiniGrid transition model for turning, collision-aware forward motion, front-cell key pickup, and locked-door interaction.
- A forward Dijkstra planner that returns a minimum-cost action sequence for each fully known layout.
- A unified random-family policy built with multi-source reverse Dijkstra from every terminal state, then cached for direct dictionary lookup during rollout.
- An evaluation and visualization pipeline that replays plans in MiniGrid and produces both looping GIFs and annotated $1400 \times 1400$ trajectory figures.
- Deterministic environment assets for seven assessed known maps and all $36$ combinations of three key positions, three goal positions, and four two-door states.

## Technical approach

### State and objective

For a known map, the planner uses the augmented state

$$
s_{\mathrm{known}}
=
\left(
x_{\mathrm{agent}},
y_{\mathrm{agent}},
\theta,
k,
\mathbf{o}
\right).
$$

Here, $\theta \in \{0,1,2,3\}$ is the heading, $k \in \{0,1\}$ records key possession, and $\mathbf{o} \in \{0,1\}^{m}$ stores the open/closed state of the $m$ doors in the scanned map.

For the $10 \times 10$ random-map family, the policy key also includes the configuration variables:

$$
s_{\mathrm{family}}
=
\left(
x_{\mathrm{agent}},
y_{\mathrm{agent}},
\theta,
k,
p_{\mathrm{key}},
p_{\mathrm{goal}},
\mathbf{o}
\right).
$$

This distinction is what makes a single feedback table valid across all $36$ files: key position, goal position, and both door states are part of the state on which the action depends.

The five controls use the non-uniform stage costs defined by the project:

| Control | Meaning | Cost |
|---|---|---:|
| $\mathrm{TL}$, $\mathrm{TR}$ | Turn left or right | $1$ |
| $\mathrm{PK}$ | Pick up the key in the front cell | $2$ |
| $\mathrm{MF}$ | Move forward one traversable cell | $3$ |
| $\mathrm{UD}$ | Unlock the door in the front cell | $5$ |

The cost-to-go obeys the Bellman relation

$$
V^*(s)
=
\min_{a}
\left[
c(a) + V^*\!\left(f(s,a)\right)
\right].
$$

The boundary condition is $V^*(s)=0$ for every goal state $s \in \mathcal{G}$. Because $c(a)>0$ for every action, the implementation solves this relation exactly with priority-queue shortest-path algorithms rather than synchronous value-iteration sweeps.

### Known-map planning

The planner scans the loaded grid, records walls, keys, goals, and an arbitrary tuple of door states, and initializes the robot pose from the environment. Forward Dijkstra then expands the five actions from the supplied start state. Parent pointers reconstruct the minimum-cost action sequence as soon as a goal state is settled.

This formulation captures why the two fixed-map demos above differ: object interaction is selected only when it lowers the cost of reaching the goal under that map's geometry.

### Unified random-family policy

The random family fixes a vertical barrier at $x=5$, with doors at $(5,3)$ and $(5,7)$, and the start pose at $(4,8)$ facing upward. The remaining variables form the complete test family:

$$
3\ \text{key positions}
\times
3\ \text{goal positions}
\times
4\ \text{door-state combinations}
=
36\ \text{maps}.
$$

For each of the $9$ key/goal configurations, the implementation enumerates:

$$
\begin{aligned}
92 \times 4 \times 2 \times 4
&= 2{,}944
&& \text{states per configuration}, \\
9 \times 2{,}944
&= 26{,}496
&& \text{value states across the family}.
\end{aligned}
$$

It constructs the reverse transition graph and runs multi-source Dijkstra from all goal states. The resulting table contains $26{,}208$ nonterminal state-action decisions; the other $288$ states are terminal. Evaluation performs no per-file replanning - it rolls out this cached table with a $200$-action guard against unexpected loops.

### System pipeline

<p align="center">
  <img src="assets/system_pipeline.svg" alt="Planning pipeline from a serialized MiniGrid map through augmented-state construction, branching into forward Dijkstra for known layouts or a cached reverse-Dijkstra policy for the random family, then producing minimum-cost actions, a MiniGrid replay, a GIF, and an annotated trajectory." width="620"/>
</p>

## Results

I replayed every returned sequence through the repository's MiniGrid action adapter. All $43$ cases terminated at the goal, and the planner's computed cost matched the simulator replay cost in every case.

| Evaluation set | Goal reached | Cost range | Action range | Mean cost | Mean actions |
|---|---:|---:|---:|---:|---:|
| $7$ known maps | $7/7$ | $13\ \text{to}\ 54$ | $5\ \text{to}\ 21$ | $25.57$ | $10.57$ |
| $36$ random maps, one policy | $36/36$ | $14\ \text{to}\ 53$ | $6\ \text{to}\ 20$ | $30.17$ | $11.47$ |

Across matched key/goal configurations in the random family, the mean optimal cost rises from $22.67$ when both doors are open to $46.33$ when both are locked. The increase reflects the required key-retrieval detour and unlock action rather than a change in the optimization objective.

## Engineering highlights

- **Exact weighted planning:** positive, non-uniform costs are handled directly rather than approximated by action count.
- **Task-aware state design:** pose alone is insufficient; key possession and individual door states preserve the Markov property needed for correct decisions.
- **Reusable policy construction:** reverse dynamic programming moves work offline so random-map execution becomes a dictionary lookup at each state.
- **Reproducible evaluation:** all assessed environments and complete visual result artifacts are versioned with the implementation.
- **Failure containment:** unsupported states raise explicit errors, and policy rollout has a bounded-step guard instead of failing silently.

## Technology stack

**Core stack:** Python, NumPy, Gymnasium, MiniGrid, Matplotlib, ImageIO, and standard-library priority queues.

## Limitations and next steps

- The current system is simulation-only and plans from the full serialized grid; it does not solve perception, localization, or partial observability.
- Dynamics are deterministic and the unified policy assumes the repository's fixed $10 \times 10$ barrier and door geometry.
- Explicit enumeration grows quickly as maps, objects, and door states are added. Reachability pruning, symbolic state compression, or heuristic search would extend the approach to larger task spaces.
- A natural next step is to add stochastic transitions and online state estimation, then evaluate the planner in a partially observed or physical robot setting.

---

Developed for UC San Diego ECE 276B: Planning and Learning in Robotics. The implementation and results are presented here as a discrete planning and robotics engineering portfolio project.
