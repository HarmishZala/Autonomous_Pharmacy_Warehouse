import heapq
import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_config import generate_scenario

STUDENT_ID = 23013383


class WarehouseGridCBS:

    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

        walls = [
            (0, 3), (1, 3),
            (3, 0), (3, 1),
            (2, 7), (3, 7),
            (5, 3), (6, 3),
            (6, 8), (7, 8),
        ]
        for r, c in walls:
            self.grid[r, c] = 1

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == 0

    def get_neighbours(self, pos):
        r, c = pos
        neighbours = [(r, c)]  # wait in place
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_free((nr, nc)):
                neighbours.append((nr, nc))
        return neighbours


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def low_level_astar(grid, start, goal, constraints):
    # Separate vertex (3-tuple) and edge (5-tuple) constraints
    vertex_constraints = set()
    edge_constraints = set()
    for c in constraints:
        if len(c) == 3:
            vertex_constraints.add(c)
        elif len(c) == 5:
            edge_constraints.add(c)

    open_list = []
    initial = (heuristic(start, goal), 0, start, 0)
    heapq.heappush(open_list, initial)

    closed = set()
    came_from = {}
    g_score = {(start, 0): 0}

    max_time = grid.size * 4

    while open_list:
        f, g, pos, t = heapq.heappop(open_list)

        if pos == goal:
            path = []
            state = (pos, t)
            while state in came_from:
                path.append(state[0])
                state = came_from[state]
            path.append(state[0])
            path.reverse()
            return path

        if (pos, t) in closed:
            continue
        closed.add((pos, t))

        if t >= max_time:
            continue

        for npos in grid.get_neighbours(pos):
            nt = t + 1
            if (npos[0], npos[1], nt) in vertex_constraints:
                continue
            if (pos[0], pos[1], npos[0], npos[1], t) in edge_constraints:
                continue
            new_g = g + 1
            if (npos, nt) not in g_score or new_g < g_score[(npos, nt)]:
                g_score[(npos, nt)] = new_g
                new_f = new_g + heuristic(npos, goal)
                heapq.heappush(open_list, (new_f, new_g, npos, nt))
                came_from[(npos, nt)] = (pos, t)

    return None


class CBSNode:

    _counter = 0

    def __init__(self):
        self.constraints = {}
        self.paths = {}
        self.cost = 0

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        return CBSNode._counter < other._counter


def find_conflict(paths):
    robot_ids = list(paths.keys())
    max_t = max(len(p) for p in paths.values())

    for t in range(max_t):
        positions = {}
        for rid in robot_ids:
            pos = paths[rid][t] if t < len(paths[rid]) else paths[rid][-1]
            if pos in positions:
                return ('vertex', positions[pos], rid, pos, t)
            positions[pos] = rid

        # Edge conflicts: robots swap positions
        if t + 1 < max_t:
            for i in range(len(robot_ids)):
                for j in range(i + 1, len(robot_ids)):
                    ra, rb = robot_ids[i], robot_ids[j]
                    pos_a_t = paths[ra][t] if t < len(paths[ra]) else paths[ra][-1]
                    pos_b_t = paths[rb][t] if t < len(paths[rb]) else paths[rb][-1]
                    pos_a_t1 = paths[ra][t+1] if t+1 < len(paths[ra]) else paths[ra][-1]
                    pos_b_t1 = paths[rb][t+1] if t+1 < len(paths[rb]) else paths[rb][-1]
                    if pos_a_t == pos_b_t1 and pos_b_t == pos_a_t1:
                        return ('edge', ra, rb, pos_a_t, pos_b_t, t)

    return None


def high_level_search(grid, starts, goals, max_nodes=1000):
    robot_ids = list(starts.keys())

    root = CBSNode()
    root.constraints = {rid: set() for rid in robot_ids}
    for rid in robot_ids:
        path = low_level_astar(grid, starts[rid], goals[rid], set())
        if path is None:
            raise ValueError(f"No path for robot {rid} from {starts[rid]} to {goals[rid]}")
        root.paths[rid] = path
    root.cost = sum(len(p) - 1 for p in root.paths.values())

    open_list = []
    CBSNode._counter = 0
    heapq.heappush(open_list, (root.cost, CBSNode._counter, root))
    CBSNode._counter += 1

    nodes_expanded = 0
    best_node = root

    while open_list:
        _, _, node = heapq.heappop(open_list)
        nodes_expanded += 1

        if nodes_expanded > max_nodes:
            print(f"  [CBS] Node limit ({max_nodes}) reached, returning best solution found.")
            return best_node.paths, best_node.cost, nodes_expanded

        conflict = find_conflict(node.paths)
        if conflict is None:
            return node.paths, node.cost, nodes_expanded

        if node.cost <= best_node.cost:
            best_node = node

        conflict_type = conflict[0]

        if conflict_type == 'vertex':
            _, robot_a, robot_b, loc, t = conflict
            for rid in [robot_a, robot_b]:
                child = CBSNode()
                child.constraints = {r: set(c) for r, c in node.constraints.items()}
                child.constraints[rid].add((loc[0], loc[1], t))

                child.paths = dict(node.paths)
                new_path = low_level_astar(
                    grid, starts[rid], goals[rid], child.constraints[rid]
                )
                if new_path is None:
                    continue

                child.paths[rid] = new_path
                child.cost = sum(len(p) - 1 for p in child.paths.values())

                heapq.heappush(open_list, (child.cost, CBSNode._counter, child))
                CBSNode._counter += 1

        elif conflict_type == 'edge':
            _, robot_a, robot_b, pos_a, pos_b, t = conflict
            edge_constraints_to_add = {
                robot_a: (pos_a[0], pos_a[1], pos_b[0], pos_b[1], t),
                robot_b: (pos_b[0], pos_b[1], pos_a[0], pos_a[1], t),
            }
            for rid in [robot_a, robot_b]:
                child = CBSNode()
                child.constraints = {r: set(c) for r, c in node.constraints.items()}
                child.constraints[rid].add(edge_constraints_to_add[rid])

                child.paths = dict(node.paths)
                new_path = low_level_astar(
                    grid, starts[rid], goals[rid], child.constraints[rid]
                )
                if new_path is None:
                    continue

                child.paths[rid] = new_path
                child.cost = sum(len(p) - 1 for p in child.paths.values())

                heapq.heappush(open_list, (child.cost, CBSNode._counter, child))
                CBSNode._counter += 1

    raise ValueError("CBS failed — no solution exists")


ROBOT_COLOURS = {
    "transporter1": "#e74c3c",
    "transporter2": "#3498db",
}


def visualise_paths(grid, paths, title="CBS Optimal Paths", save_path=None):
    size = grid.size
    max_t = max(len(p) for p in paths.values())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    bg = np.where(grid.grid == 1, 0.3, 1.0)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper")

    for rid, path in paths.items():
        colour = ROBOT_COLOURS.get(rid, "#2ecc71")
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, "o-", color=colour, markersize=5,
                linewidth=2, label=f"{rid} (cost={len(path)-1})", alpha=0.8)
        ax.plot(cols[0], rows[0], "s", color=colour, markersize=12, markeredgecolor="black")
        ax.plot(cols[-1], rows[-1], "*", color=colour, markersize=16, markeredgecolor="black")

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.5)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Trajectory Traces")
    ax.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]
    snapshot_times = list(range(0, max_t, max(1, max_t // 6))) + [max_t - 1]
    snapshot_times = sorted(set(snapshot_times))

    ax2.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper")
    for t in snapshot_times:
        for rid, path in paths.items():
            pos = path[t] if t < len(path) else path[-1]
            colour = ROBOT_COLOURS.get(rid, "#2ecc71")
            alpha = 0.3 + 0.7 * (t / (max_t - 1)) if max_t > 1 else 1.0
            ax2.plot(pos[1], pos[0], "o", color=colour, markersize=8,
                     alpha=alpha, markeredgecolor="black", markeredgewidth=0.5)
            ax2.annotate(f"t={t}", (pos[1] + 0.15, pos[0] - 0.15),
                         fontsize=5, color=colour, alpha=alpha)

    ax2.set_xticks(range(size))
    ax2.set_yticks(range(size))
    ax2.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax2.grid(which="minor", color="grey", linewidth=0.5)
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_title("Position Snapshots Over Time")

    legend_handles = [
        mpatches.Patch(color=ROBOT_COLOURS[rid], label=rid)
        for rid in paths
    ]
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_timesteps(grid, paths, save_path=None):
    size = grid.size
    max_t = max(len(p) for p in paths.values())
    steps = list(range(min(max_t, 12)))
    cols = 4
    rows = (len(steps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, t in enumerate(steps):
        ax = axes[idx]
        bg = np.where(grid.grid == 1, 0.3, 1.0)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper")

        for rid, path in paths.items():
            pos = path[t] if t < len(path) else path[-1]
            colour = ROBOT_COLOURS.get(rid, "#2ecc71")
            ax.plot(pos[1], pos[0], "o", color=colour, markersize=14,
                    markeredgecolor="black", markeredgewidth=1.5)
            ax.text(pos[1], pos[0], rid[-1], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.3)
        ax.set_title(f"t = {t}", fontsize=10, fontweight="bold")

    for idx in range(len(steps), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("CBS Solution: Timestep-by-Timestep",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_metrics(scenario_names, nodes_list, costs_list, path_lengths,
                      save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(scenario_names))
    width = 0.5

    ax = axes[0]
    bars = ax.bar(x, nodes_list, width, color=["#3498db", "#e74c3c", "#2ecc71"])
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.set_ylabel("Nodes Expanded")
    ax.set_title("CBS Nodes Expanded")
    for bar, val in zip(bars, nodes_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, costs_list, width, color=["#3498db", "#e74c3c", "#2ecc71"])
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.set_ylabel("Sum-of-Costs")
    ax.set_title("CBS Solution Cost")
    for bar, val in zip(bars, costs_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[2]
    width2 = 0.3
    for i, rid in enumerate(["transporter1", "transporter2"]):
        lengths = [pl[rid] for pl in path_lengths]
        offset = (i - 0.5) * width2
        bars = ax.bar(x + offset, lengths, width2,
                       color=ROBOT_COLOURS[rid], label=rid)
        for bar, val in zip(bars, lengths):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.set_ylabel("Path Length (steps)")
    ax.set_title("Individual Path Lengths")
    ax.legend(fontsize=9)

    fig.suptitle("CBS Metrics Comparison Across Scenarios",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def verify_collision_free(paths):
    robot_ids = list(paths.keys())
    max_t = max(len(p) for p in paths.values())
    collision_free = True
    for t in range(max_t):
        positions = []
        for rid in robot_ids:
            pos = paths[rid][t] if t < len(paths[rid]) else paths[rid][-1]
            if pos in positions:
                print(f"  [COLLISION] at t={t}, pos={pos}")
                collision_free = False
            positions.append(pos)
        if t + 1 < max_t:
            for i in range(len(robot_ids)):
                for j in range(i + 1, len(robot_ids)):
                    ra, rb = robot_ids[i], robot_ids[j]
                    pa_t = paths[ra][t] if t < len(paths[ra]) else paths[ra][-1]
                    pb_t = paths[rb][t] if t < len(paths[rb]) else paths[rb][-1]
                    pa_t1 = paths[ra][t+1] if t+1 < len(paths[ra]) else paths[ra][-1]
                    pb_t1 = paths[rb][t+1] if t+1 < len(paths[rb]) else paths[rb][-1]
                    if pa_t == pb_t1 and pb_t == pa_t1:
                        print(f"  [EDGE COLLISION] at t={t}, {ra}:{pa_t}->{pa_t1}, {rb}:{pb_t}->{pb_t1}")
                        collision_free = False
    return collision_free


def main():
    scenario = generate_scenario(STUDENT_ID)
    size = scenario["size"]

    print("=" * 60)
    print("COMP0224 — Task C (Part 1): Conflict-Based Search")
    print(f"Student ID : {STUDENT_ID}")
    print(f"Grid Size  : {size}x{size}")
    print("=" * 60)

    grid = WarehouseGridCBS(size)

    scenario_names = []
    nodes_list = []
    costs_list = []
    path_lengths_list = []

    # Scenario 1: Opposing starts
    print("\n[CBS] Scenario 1: Opposing Starts")
    starts1 = {"transporter1": (9, 0), "transporter2": (9, 9)}
    goals1 = {"transporter1": (0, 0), "transporter2": (0, 9)}
    print(f"Robot starts : {starts1}")
    print(f"Robot goals  : {goals1}")

    paths1, cost1, nodes1 = high_level_search(grid, starts1, goals1)
    print(f"[CBS] Solution found!")
    print(f"  Sum-of-Costs : {cost1}")
    print(f"  Nodes expanded: {nodes1}")
    for rid, path in paths1.items():
        print(f"  {rid}: cost={len(path)-1}, path={path}")
    print(f"  Collision-free: {verify_collision_free(paths1)}")

    scenario_names.append("S1: Opposing")
    nodes_list.append(nodes1)
    costs_list.append(cost1)
    path_lengths_list.append({rid: len(p)-1 for rid, p in paths1.items()})

    # Scenario 2: Crossing paths
    print("\n" + "=" * 60)
    print("[CBS] Scenario 2: Crossing paths (conflict-heavy)")
    starts2 = {"transporter1": (0, 2), "transporter2": (0, 7)}
    goals2 = {"transporter1": (9, 7), "transporter2": (9, 2)}
    print(f"Robot starts : {starts2}")
    print(f"Robot goals  : {goals2}")

    paths2, cost2, nodes2 = high_level_search(grid, starts2, goals2)
    print(f"[CBS] Solution found!")
    print(f"  Sum-of-Costs : {cost2}")
    print(f"  Nodes expanded: {nodes2}")
    for rid, path in paths2.items():
        print(f"  {rid}: cost={len(path)-1}")
    print(f"  Collision-free: {verify_collision_free(paths2)}")

    scenario_names.append("S2: Crossing")
    nodes_list.append(nodes2)
    costs_list.append(cost2)
    path_lengths_list.append({rid: len(p)-1 for rid, p in paths2.items()})

    # Scenario 3: Head-on swap
    print("\n" + "=" * 60)
    print("[CBS] Scenario 3: Head-on collision (guaranteed conflict resolution)")
    starts3 = {"transporter1": (4, 5), "transporter2": (4, 9)}
    goals3 = {"transporter1": (4, 9), "transporter2": (4, 5)}
    print(f"Robot starts : {starts3}")
    print(f"Robot goals  : {goals3}")

    paths3, cost3, nodes3 = high_level_search(grid, starts3, goals3)
    print(f"[CBS] Solution found!")
    print(f"  Sum-of-Costs  : {cost3}")
    print(f"  Nodes expanded: {nodes3}  (>1 confirms conflict was resolved)")
    for rid, path in paths3.items():
        print(f"  {rid}: cost={len(path)-1}, path={path}")
    print(f"  Collision-free: {verify_collision_free(paths3)}")

    scenario_names.append("S3: Head-on")
    nodes_list.append(nodes3)
    costs_list.append(cost3)
    path_lengths_list.append({rid: len(p)-1 for rid, p in paths3.items()})

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    visualise_paths(grid, paths1,
                    title=f"CBS Scenario 1: Opposing Starts (Cost={cost1})",
                    save_path=os.path.join(output_dir, "task_c_cbs_scenario1.png"))

    visualise_paths(grid, paths2,
                    title=f"CBS Scenario 2: Crossing Paths (Cost={cost2})",
                    save_path=os.path.join(output_dir, "task_c_cbs_scenario2.png"))

    visualise_paths(grid, paths3,
                    title=f"CBS Scenario 3: Head-On Collision  "
                          f"(Nodes expanded={nodes3}, Cost={cost3})",
                    save_path=os.path.join(output_dir, "task_c_cbs_scenario3.png"))

    visualise_timesteps(grid, paths3,
                        save_path=os.path.join(output_dir, "task_c_cbs_timesteps.png"))

    visualise_metrics(scenario_names, nodes_list, costs_list, path_lengths_list,
                      save_path=os.path.join(output_dir, "task_c_cbs_metrics.png"))

    # Optimality proof: compare CBS cost vs unconstrained lower bound
    print("\n" + "=" * 60)
    print("[CBS] Sum-of-Costs Optimality Verification")
    all_scenarios = [
        ("S1", starts1, goals1, cost1),
        ("S2", starts2, goals2, cost2),
        ("S3", starts3, goals3, cost3),
    ]
    for label, s, g, cbs_cost in all_scenarios:
        lb = sum(len(low_level_astar(grid, s[r], g[r], set())) - 1 for r in s)
        status = "OPTIMAL (=LB)" if cbs_cost == lb else f"CBS={cbs_cost} >= LB={lb} (conflict overhead={cbs_cost-lb})"
        print(f"  {label}: {status}")

    plt.show()
    print("\n[Done] CBS complete.")


if __name__ == "__main__":
    main()
