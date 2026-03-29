import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_config import generate_scenario

STUDENT_ID = 23013383

FREE = 0
WALL = 1
PIT = 2
GOAL = 3
START = 4

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
}
PERPENDICULAR = {
    "UP":    ["LEFT", "RIGHT"],
    "DOWN":  ["LEFT", "RIGHT"],
    "LEFT":  ["UP",   "DOWN"],
    "RIGHT": ["UP",   "DOWN"],
}


class WarehouseGrid:

    def __init__(self, size, rewards):
        self.size = size
        self.rewards = rewards
        self.grid = np.full((size, size), FREE, dtype=int)

        walls = [
            (0, 3), (1, 3),
            (3, 0), (3, 1),
            (2, 7), (3, 7),
            (5, 3), (6, 3),
            (6, 8), (7, 8),
        ]
        pits = [
            (1, 5),
            (3, 4),
            (5, 1),
            (5, 6),
            (7, 5),
        ]
        self.start = (0, 0)
        self.goal = (9, 9)

        for r, c in walls:
            self.grid[r, c] = WALL
        for r, c in pits:
            self.grid[r, c] = PIT
        self.grid[self.goal] = GOAL
        self.grid[self.start] = START

        self._states = [
            (r, c) for r in range(size) for c in range(size)
            if self.grid[r, c] != WALL
        ]

    def get_states(self):
        return self._states

    def get_actions(self):
        return ACTIONS

    def is_terminal(self, s):
        return self.grid[s] in (GOAL, PIT)

    def reward(self, s):
        cell = self.grid[s]
        if cell == GOAL:
            return self.rewards["goal"]
        elif cell == PIT:
            return self.rewards["pit"]
        return self.rewards["step"]

    def _move(self, state, action):
        dr, dc = ACTION_DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != WALL:
            return (nr, nc)
        return state

    def transition(self, state, action, p_slip):
        if self.is_terminal(state):
            return [(1.0, state)]

        outcomes = {}
        intended = self._move(state, action)
        outcomes[intended] = outcomes.get(intended, 0.0) + (1.0 - p_slip)

        for perp in PERPENDICULAR[action]:
            slipped = self._move(state, perp)
            outcomes[slipped] = outcomes.get(slipped, 0.0) + (p_slip / 2.0)

        return [(p, s) for s, p in outcomes.items()]


def value_iteration(grid, gamma, p_slip, theta=1e-6):
    states = grid.get_states()
    actions = grid.get_actions()

    V = {}
    for s in states:
        V[s] = grid.reward(s) if grid.is_terminal(s) else 0.0
    pi = {s: None for s in states}
    deltas = []

    while True:
        delta = 0.0
        for s in states:
            if grid.is_terminal(s):
                continue
            old_v = V[s]
            best_val = -np.inf
            best_act = None
            for a in actions:
                q = sum(
                    p * (grid.reward(s_next) + gamma * V[s_next])
                    for p, s_next in grid.transition(s, a, p_slip)
                )
                if q > best_val:
                    best_val = q
                    best_act = a
            V[s] = best_val
            pi[s] = best_act
            delta = max(delta, abs(V[s] - old_v))
        deltas.append(delta)
        if delta < theta:
            break

    return V, pi, deltas


def _policy_evaluate(grid, pi, gamma, p_slip, theta=1e-6):
    states = grid.get_states()
    V = {}
    for s in states:
        V[s] = grid.reward(s) if grid.is_terminal(s) else 0.0

    while True:
        delta = 0.0
        for s in states:
            if grid.is_terminal(s):
                continue
            old_v = V[s]
            a = pi[s]
            V[s] = sum(
                p * (grid.reward(s_next) + gamma * V[s_next])
                for p, s_next in grid.transition(s, a, p_slip)
            )
            delta = max(delta, abs(V[s] - old_v))
        if delta < theta:
            break
    return V


def policy_iteration(grid, gamma, p_slip, theta=1e-6):
    states = grid.get_states()
    actions = grid.get_actions()

    pi = {s: (None if grid.is_terminal(s) else "UP") for s in states}
    deltas = []
    iters = 0

    while True:
        V = _policy_evaluate(grid, pi, gamma, p_slip, theta)

        policy_stable = True
        max_change = 0.0
        for s in states:
            if grid.is_terminal(s):
                continue
            old_action = pi[s]
            best_val = -np.inf
            best_act = None
            for a in actions:
                q = sum(
                    p * (grid.reward(s_next) + gamma * V[s_next])
                    for p, s_next in grid.transition(s, a, p_slip)
                )
                if q > best_val:
                    best_val = q
                    best_act = a
            pi[s] = best_act
            max_change = max(max_change, abs(V[s] - best_val))
            if old_action != pi[s]:
                policy_stable = False

        deltas.append(max_change)
        iters += 1
        if policy_stable:
            break

    return V, pi, iters, deltas


ARROW_MAP = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
CELL_COLOURS = {
    WALL: "#2c3e50",
    PIT:  "#c0392b",
    GOAL: "#27ae60",
    START: "#3498db",
    FREE: "#ecf0f1",
}


def visualise_policy(grid, V, pi, title, save_path=None):
    size = grid.size
    value_array = np.full((size, size), np.nan)
    for (r, c) in grid.get_states():
        value_array[r, c] = V[(r, c)]

    fig, ax = plt.subplots(figsize=(9, 9))

    masked = np.ma.masked_where(np.isnan(value_array), value_array)
    vmin, vmax = np.nanmin(value_array), np.nanmax(value_array)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    ax.imshow(masked, cmap="RdYlGn", norm=norm, origin="upper")

    for r in range(size):
        for c in range(size):
            cell = grid.grid[r, c]
            if cell == WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color=CELL_COLOURS[WALL]))
                ax.text(c, r, "W", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
            elif cell == PIT:
                ax.text(c, r - 0.22, "PIT", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
                ax.text(c, r + 0.18, f"{V[(r,c)]:.0f}", ha="center",
                        va="center", fontsize=7, color="white")
            elif cell == GOAL:
                ax.text(c, r - 0.22, "GOAL", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
                ax.text(c, r + 0.18, f"{V[(r,c)]:.0f}", ha="center",
                        va="center", fontsize=7, color="white")
            else:
                arrow = ARROW_MAP.get(pi.get((r, c), ""), "")
                ax.text(c, r - 0.2, arrow, ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#2c3e50")
                ax.text(c, r + 0.2, f"{V[(r,c)]:.1f}", ha="center",
                        va="center", fontsize=6, color="#2c3e50")

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=CELL_COLOURS[WALL], label="Wall"),
        mpatches.Patch(color=CELL_COLOURS[PIT],  label="Pit (-100)"),
        mpatches.Patch(color=CELL_COLOURS[GOAL], label="Goal (+100)"),
        mpatches.Patch(color=CELL_COLOURS[START], label="Start"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_convergence(vi_deltas, pi_deltas, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(range(1, len(vi_deltas) + 1), vi_deltas,
                "o-", color="#e74c3c", label=f"Value Iteration ({len(vi_deltas)} sweeps)",
                markersize=3, linewidth=1.5)
    ax.semilogy(range(1, len(pi_deltas) + 1), pi_deltas,
                "s-", color="#3498db", label=f"Policy Iteration ({len(pi_deltas)} improvements)",
                markersize=6, linewidth=2)
    ax.axhline(1e-6, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(1, 1.5e-6, "θ = 1×10⁻⁶", fontsize=8, color="grey")
    ax.annotate(f"{len(vi_deltas)} sweeps",
                xy=(len(vi_deltas), vi_deltas[-1]),
                xytext=(len(vi_deltas) - 8, vi_deltas[-1] * 100),
                fontsize=8, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
    ax.annotate(f"{len(pi_deltas)} sweeps",
                xy=(len(pi_deltas), pi_deltas[-1]),
                xytext=(len(pi_deltas) + 2, pi_deltas[-1] * 100),
                fontsize=8, color="#3498db",
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.2))
    ax.set_xlabel("Iteration / Improvement Sweep", fontsize=11)
    ax.set_ylabel("Max Bellman Residual  (log scale)", fontsize=11)
    ax.set_title("Convergence Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    labels = ["Value Iteration", "Policy Iteration"]
    counts = [len(vi_deltas), len(pi_deltas)]
    colors = ["#e74c3c", "#3498db"]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8,
                   width=0.4)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")
    speedup = len(vi_deltas) / len(pi_deltas)
    ax2.set_ylabel("Iterations to Convergence", fontsize=11)
    ax2.set_title(f"Iterations Required\n(PI is {speedup:.1f}× fewer sweeps)", fontsize=12,
                  fontweight="bold")
    ax2.set_ylim(0, max(counts) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Value Iteration vs Policy Iteration — Convergence Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def slip_sensitivity_analysis(grid, gamma, slip_values, save_path=None):
    n = len(slip_values)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten()

    for idx, p_slip in enumerate(slip_values):
        V, pi, _ = value_iteration(grid, gamma, p_slip)
        ax = axes[idx]
        size = grid.size

        value_array = np.full((size, size), np.nan)
        for (r, c) in grid.get_states():
            value_array[r, c] = V[(r, c)]

        masked = np.ma.masked_where(np.isnan(value_array), value_array)
        vmin, vmax = np.nanmin(value_array), np.nanmax(value_array)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None
        ax.imshow(masked, cmap="RdYlGn", norm=norm, origin="upper")

        for r in range(size):
            for c in range(size):
                cell = grid.grid[r, c]
                if cell == WALL:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color=CELL_COLOURS[WALL]))
                elif cell == PIT:
                    ax.text(c, r, "P", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")
                elif cell == GOAL:
                    ax.text(c, r, "G", ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")
                else:
                    arrow = ARROW_MAP.get(pi.get((r, c), ""), "")
                    ax.text(c, r, arrow, ha="center", va="center",
                            fontsize=11, fontweight="bold", color="#2c3e50")

        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.3)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_title(f"p_slip = {p_slip:.2f}", fontsize=11, fontweight="bold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Slip Sensitivity: Policy Changes with Increasing Uncertainty",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_value_surface(grid, V, title="Value Function Surface", save_path=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    size = grid.size
    Z = np.full((size, size), np.nan)
    for (r, c) in grid.get_states():
        Z[r, c] = V[(r, c)]

    for r in range(size):
        for c in range(size):
            if np.isnan(Z[r, c]):
                nbrs = [Z[r+dr, c+dc]
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= r+dr < size and 0 <= c+dc < size
                        and not np.isnan(Z[r+dr, c+dc])]
                Z[r, c] = np.mean(nbrs) if nbrs else 0.0

    X = np.arange(size)
    Y = np.arange(size)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn", edgecolor="none", alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="V(s)")

    gr, gc = grid.goal
    ax.scatter([gc], [gr], [V[grid.goal]], color="#27ae60", s=120, zorder=5, label="Goal")
    for (r, c) in grid.get_states():
        if grid.grid[r, c] == PIT:
            ax.scatter([c], [r], [V[(r, c)]], color="#c0392b", s=80, zorder=5)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("V(s)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def simulate_path(grid, pi, p_slip, start=None, max_steps=200, seed=None):
    rng = np.random.default_rng(seed)
    state = start if start is not None else grid.start
    path = [state]
    for _ in range(max_steps):
        if grid.is_terminal(state):
            break
        action = pi.get(state)
        if action is None:
            break
        outcomes = grid.transition(state, action, p_slip)
        probs, nexts = zip(*[(p, s) for p, s in outcomes])
        idx = rng.choice(len(nexts), p=list(probs))
        state = nexts[idx]
        path.append(state)
    return path


def visualise_simulated_paths(grid, pi, p_slip, n_paths=15, save_path=None):
    size = grid.size
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for panel_idx, (ax, p_s, label) in enumerate(zip(
            axes,
            [0.0, p_slip],
            [f"Deterministic  (p_slip=0.00)", f"Stochastic  (p_slip={p_slip:.2f})"])):

        bg = np.where(grid.grid == WALL, 0.25,
              np.where(grid.grid == PIT,  0.6,
              np.where(grid.grid == GOAL, 0.9, 1.0)))
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper")

        for r in range(size):
            for c in range(size):
                cell = grid.grid[r, c]
                if cell == PIT:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                               color="#c0392b", alpha=0.6))
                    ax.text(c, r, "PIT", ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
                elif cell == GOAL:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                               color="#27ae60", alpha=0.7))
                    ax.text(c, r, "G", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
                elif cell == WALL:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                               color="#2c3e50"))

        success, fail = 0, 0
        cmap_paths = plt.cm.cool
        for i in range(n_paths):
            path = simulate_path(grid, pi, p_s, seed=i * 7)
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            color = cmap_paths(i / n_paths)
            reached_goal = grid.grid[path[-1]] == GOAL
            if reached_goal:
                success += 1
            else:
                fail += 1
            ax.plot(cols, rows, "-", color=color, alpha=0.55, linewidth=1.3)
            ax.plot(cols[0], rows[0], "s", color=color, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.plot(cols[-1], rows[-1], "*", color=color, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5)

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.4)
        ax.set_xlabel("Column", fontsize=10)
        ax.set_ylabel("Row", fontsize=10)
        ax.set_title(f"{label}\n{n_paths} rollouts  —  "
                     f"reached goal: {success}/{n_paths}",
                     fontsize=11, fontweight="bold")

    fig.suptitle("Stochastic Rollouts: Optimal Policy Under Slip Uncertainty",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_pit_avoidance(grid, gamma, slip_values, save_path=None):
    pit_cells = [(r, c) for (r, c) in grid.get_states() if grid.grid[r, c] == PIT]

    policy_escape_dists = []
    for p_slip in slip_values:
        V, pi, _ = value_iteration(grid, gamma, p_slip)
        escape_dists = []
        for (r, c) in grid.get_states():
            if grid.is_terminal((r, c)) or grid.grid[r, c] == WALL:
                continue
            a = pi[(r, c)]
            if a is None:
                continue
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.size and 0 <= nc < grid.size and grid.grid[nr, nc] != WALL:
                min_d = min(abs(nr - pr) + abs(nc - pc) for pr, pc in pit_cells)
            else:
                min_d = min(abs(r - pr) + abs(c - pc) for pr, pc in pit_cells)
            escape_dists.append(min_d)
        policy_escape_dists.append(np.mean(escape_dists))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(slip_values, policy_escape_dists, "o-", color="#8e44ad",
            linewidth=2, markersize=8)
    ax.axvline(slip_values[slip_values.index(max(slip_values, key=lambda x: x == 0.08))
                           if 0.08 in slip_values else 0],
               color="grey", linestyle=":", alpha=0.0)
    ax.set_xlabel("Slip Probability  p_slip", fontsize=11)
    ax.set_ylabel("Mean Distance to Nearest Pit\n(policy next-step)", fontsize=10)
    ax.set_title("Policy Pit-Avoidance vs Slip Probability", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    away_counts = []
    for p_slip in slip_values:
        V, pi, _ = value_iteration(grid, gamma, p_slip)
        away = 0
        total = 0
        for (r, c) in grid.get_states():
            if grid.is_terminal((r, c)) or grid.grid[r, c] == WALL:
                continue
            a = pi[(r, c)]
            if a is None:
                continue
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.size and 0 <= nc < grid.size):
                nr, nc = r, c
            nearest_pit = min(pit_cells, key=lambda p: abs(r-p[0])+abs(c-p[1]))
            dist_before = abs(r - nearest_pit[0]) + abs(c - nearest_pit[1])
            dist_after  = abs(nr - nearest_pit[0]) + abs(nc - nearest_pit[1])
            if dist_after >= dist_before:
                away += 1
            total += 1
        away_counts.append(100 * away / total if total > 0 else 0)

    ax2.bar([f"{p:.2f}" for p in slip_values], away_counts,
            color=["#c0392b" if p < 0.08 else "#27ae60" if p > 0.08 else "#f39c12"
                   for p in slip_values],
            edgecolor="black", linewidth=0.8)
    ax2.set_xlabel("Slip Probability  p_slip", fontsize=11)
    ax2.set_ylabel("States where policy moves\naway from nearest pit  (%)", fontsize=10)
    ax2.set_title("% States Avoiding Pits vs Slip Probability", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(away_counts):
        ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Quantitative Pit-Avoidance Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def print_policy_grid(grid, pi):
    size = grid.size
    print("\nOptimal Policy Grid:")
    print("+" + "----+" * size)
    for r in range(size):
        row_str = "|"
        for c in range(size):
            cell = grid.grid[r, c]
            if cell == WALL:
                row_str += " WW |"
            elif cell == PIT:
                row_str += " XX |"
            elif cell == GOAL:
                row_str += " GG |"
            else:
                arrow = ARROW_MAP.get(pi.get((r, c), ""), "  ")
                row_str += f" {arrow}  |"

        print(row_str)
        print("+" + "----+" * size)


def main():
    scenario = generate_scenario(STUDENT_ID)
    p_slip = scenario["p_slip"]
    size   = scenario["size"]
    gamma  = scenario["gamma"]
    rewards = scenario["rewards"]

    print("=" * 60)
    print("COMP0224 — Task A: MDP Navigation")
    print(f"Student ID : {STUDENT_ID}")
    print(f"Grid Size  : {size}x{size}")
    print(f"p_slip     : {p_slip}")
    print(f"gamma      : {gamma}")
    print(f"Rewards    : {rewards}")
    print("=" * 60)

    grid = WarehouseGrid(size=size, rewards=rewards)

    print("\n[1] Running Value Iteration ...")
    V_vi, pi_vi, deltas_vi = value_iteration(grid, gamma, p_slip)
    print(f"    Converged in {len(deltas_vi)} sweeps  "
          f"(final residual: {deltas_vi[-1]:.2e})")
    print_policy_grid(grid, pi_vi)

    print("\n[2] Running Policy Iteration ...")
    V_pi, pi_pi, iters_pi, deltas_pi = policy_iteration(grid, gamma, p_slip)
    print(f"    Converged in {iters_pi} improvement sweeps")

    mismatches = sum(
        1 for s in grid.get_states()
        if not grid.is_terminal(s) and pi_vi[s] != pi_pi[s]
    )
    print(f"\n[Check] Policy mismatches between VI and PI: {mismatches}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    visualise_policy(grid, V_vi, pi_vi,
                     f"Value Iteration — Optimal Policy  (p_slip={p_slip})",
                     save_path=os.path.join(output_dir, "task_a_vi_policy.png"))

    visualise_policy(grid, V_pi, pi_pi,
                     f"Policy Iteration — Optimal Policy  (p_slip={p_slip})",
                     save_path=os.path.join(output_dir, "task_a_pi_policy.png"))

    visualise_convergence(deltas_vi, deltas_pi,
                          save_path=os.path.join(output_dir, "task_a_convergence.png"))

    slip_values_list = [0.00, 0.05, 0.08, 0.15, 0.20, 0.30]
    slip_sensitivity_analysis(
        grid, gamma,
        slip_values_list,
        save_path=os.path.join(output_dir, "task_a_slip_sensitivity.png"))

    visualise_value_surface(
        grid, V_vi,
        title=f"Value Function Surface — Value Iteration  (p_slip={p_slip})",
        save_path=os.path.join(output_dir, "task_a_value_surface.png"))

    visualise_simulated_paths(
        grid, pi_vi, p_slip, n_paths=20,
        save_path=os.path.join(output_dir, "task_a_stochastic_rollouts.png"))

    visualise_pit_avoidance(
        grid, gamma, slip_values_list,
        save_path=os.path.join(output_dir, "task_a_pit_avoidance.png"))

    plt.show()
    print("\n[Done] All figures saved and displayed.")


if __name__ == "__main__":
    main()
