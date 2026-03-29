import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCATIONS = ["shelf_A", "shelf_B", "shelf_C",
             "picking_station", "aisle", "staging_area", "dock"]

CONNECTIONS = [
    ("shelf_A", "shelf_B"),
    ("shelf_B", "shelf_C"),
    ("shelf_A", "picking_station"),
    ("shelf_B", "aisle"),
    ("shelf_C", "staging_area"),
    ("picking_station", "dock"),
    ("dock", "aisle"),
    ("aisle", "staging_area"),
]

POSITIONS = {
    "shelf_A":         (0.0, 2.0),
    "shelf_B":         (1.5, 2.0),
    "shelf_C":         (3.0, 2.0),
    "picking_station": (0.0, 1.0),
    "aisle":           (1.5, 1.0),
    "staging_area":    (3.0, 1.0),
    "dock":            (0.75, 0.0),
}

ITEM_LOCATIONS = {
    "insulin":     "shelf_A",
    "amoxicillin": "shelf_B",
    "ibuprofen":   "shelf_C",
}

ROBOT_STARTS = {
    "picker1":      "picking_station",
    "transporter1": "dock",
    "transporter2": "dock",
}

PLAN = [
    ("move",    "transporter2", "dock",             "aisle"),
    ("move",    "transporter2", "aisle",            "shelf_B"),
    ("move",    "transporter2", "shelf_B",          "shelf_C"),
    ("move",    "transporter1", "dock",             "aisle"),
    ("move",    "transporter1", "aisle",            "shelf_B"),
    ("move",    "picker1",      "picking_station",  "shelf_A"),
    ("grasp",   "transporter2", "ibuprofen",        "shelf_C"),
    ("move",    "transporter2", "shelf_C",          "shelf_B"),
    ("move",    "transporter2", "shelf_B",          "aisle"),
    ("move",    "transporter2", "aisle",            "dock"),
    ("grasp",   "transporter1", "amoxicillin",      "shelf_B"),
    ("move",    "transporter1", "shelf_B",          "aisle"),
    ("move",    "transporter1", "aisle",            "dock"),
    ("grasp",   "picker1",      "insulin",          "shelf_A"),
    ("move",    "picker1",      "shelf_A",          "picking_station"),
    ("move",    "picker1",      "picking_station",  "dock"),
    ("deliver", "transporter2", "ibuprofen",        "dock"),
    ("deliver", "transporter1", "amoxicillin",      "dock"),
    ("deliver", "picker1",      "insulin",          "dock"),
]

ROBOT_COLOURS = {
    "picker1":      "#e67e22",
    "transporter1": "#e74c3c",
    "transporter2": "#3498db",
}

ACTION_COLOURS = {
    "move":    "#7f8c8d",
    "grasp":   "#27ae60",
    "deliver": "#8e44ad",
}


def fig_warehouse_graph(save_path=None):
    G = nx.Graph()
    G.add_nodes_from(LOCATIONS)
    G.add_edges_from(CONNECTIONS)

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = POSITIONS

    node_colours = []
    node_sizes = []
    for loc in LOCATIONS:
        if loc.startswith("shelf"):
            node_colours.append("#3498db")
            node_sizes.append(1800)
        elif loc == "dock":
            node_colours.append("#8e44ad")
            node_sizes.append(2200)
        elif loc == "picking_station":
            node_colours.append("#e67e22")
            node_sizes.append(1800)
        else:
            node_colours.append("#27ae60")
            node_sizes.append(1500)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bdc3c7",
                           width=2.5, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5,
                            font_weight="bold", font_color="white")

    for item, loc in ITEM_LOCATIONS.items():
        x, y = pos[loc]
        ax.annotate(f"[{item}]", (x, y + 0.18), ha="center",
                    fontsize=7.5, color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1",
                              edgecolor="#95a5a6", alpha=0.9))

    for robot, loc in ROBOT_STARTS.items():
        x, y = pos[loc]
        offset = {"picker1": (-0.25, -0.18),
                  "transporter1": (-0.1, -0.22),
                  "transporter2": (0.15, -0.22)}[robot]
        ax.annotate(f"▸{robot}", (x + offset[0], y + offset[1]),
                    fontsize=7, color=ROBOT_COLOURS[robot],
                    fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#3498db", label="Shelf (bottle storage)"),
        mpatches.Patch(color="#8e44ad", label="Delivery Dock"),
        mpatches.Patch(color="#e67e22", label="Picking Station"),
        mpatches.Patch(color="#27ae60", label="Navigation Area"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9)

    ax.text(0.01, 0.98, "All edges are bidirectional (connected ?l1 ?l2) ∧ (connected ?l2 ?l1)",
            transform=ax.transAxes, fontsize=8, va="top",
            color="#7f8c8d", style="italic")

    ax.set_title("Task B — Pharmacy Warehouse Topology\n"
                 "(PDDL 'connected' predicate encodes this graph)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def fig_gantt(save_path=None):
    robots = ["picker1", "transporter1", "transporter2"]
    n_steps = len(PLAN)

    fig, ax = plt.subplots(figsize=(14, 5))

    y_pos = {r: i for i, r in enumerate(robots)}
    bar_height = 0.55

    for step, action in enumerate(PLAN):
        act_type = action[0]
        robot    = action[1]
        arg2     = action[2]
        arg3     = action[3]

        y = y_pos[robot]
        colour = ACTION_COLOURS[act_type]

        ax.barh(y, 1, left=step, height=bar_height,
                color=colour, edgecolor="white", linewidth=0.8, alpha=0.88)

        if act_type == "move":
            lbl = f"→ {arg3.replace('_', ' ')}"
        elif act_type == "grasp":
            lbl = f"GRASP\n{arg2}"
        else:
            lbl = f"DELIVER\n{arg2}"

        ax.text(step + 0.5, y, lbl, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white")

    ax.set_yticks(range(len(robots)))
    ax.set_yticklabels(robots, fontsize=11, fontweight="bold")
    ax.set_xticks(range(n_steps + 1))
    ax.set_xticklabels([str(i) for i in range(n_steps + 1)], fontsize=8)
    ax.set_xlabel("Plan Step", fontsize=11)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(-0.5, len(robots) - 0.5)
    ax.set_title("Task B — Plan Gantt Chart: Robot × Action × Step\n"
                 "(19 total actions; 3 robots execute partly in parallel)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="grey")

    for tick, robot in zip(ax.get_yticklabels(), robots):
        tick.set_color(ROBOT_COLOURS[robot])

    legend_handles = [
        mpatches.Patch(color=ACTION_COLOURS["move"],    label="move"),
        mpatches.Patch(color=ACTION_COLOURS["grasp"],   label="grasp (PickUp)"),
        mpatches.Patch(color=ACTION_COLOURS["deliver"], label="deliver"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def _execute_plan(plan):
    state = {
        "at":              {("picker1", "picking_station"),
                            ("transporter1", "dock"),
                            ("transporter2", "dock")},
        "gripper-empty":   {"picker1", "transporter1", "transporter2"},
        "holding":         set(),
        "bottle-at":       {("insulin", "shelf_A"),
                            ("amoxicillin", "shelf_B"),
                            ("ibuprofen", "shelf_C")},
        "order-fulfilled": set(),
    }
    snapshots = [_snapshot(state)]

    for step in plan:
        act = step[0]
        if act == "move":
            _, robot, frm, to = step
            state["at"].discard((robot, frm))
            state["at"].add((robot, to))
        elif act == "grasp":
            _, robot, item, loc = step
            state["holding"].add((robot, item))
            state["bottle-at"].discard((item, loc))
            state["gripper-empty"].discard(robot)
        elif act == "deliver":
            _, robot, item, loc = step
            state["order-fulfilled"].add(item)
            state["holding"].discard((robot, item))
            state["gripper-empty"].add(robot)
        snapshots.append(_snapshot(state))

    return snapshots


def _execute_defective_plan(plan):
    state = {
        "at":              {("picker1", "picking_station"),
                            ("transporter1", "dock"),
                            ("transporter2", "dock")},
        "gripper-empty":   {"picker1", "transporter1", "transporter2"},
        "holding":         set(),
        "bottle-at":       {("insulin", "shelf_A"),
                            ("amoxicillin", "shelf_B"),
                            ("ibuprofen", "shelf_C")},
        "order-fulfilled": set(),
    }
    snapshots = [_snapshot(state)]
    blocked_at = None

    for step_idx, step in enumerate(plan):
        act = step[0]
        if act == "move":
            _, robot, frm, to = step
            state["at"].discard((robot, frm))
            state["at"].add((robot, to))
        elif act == "grasp":
            _, robot, item, loc = step
            # DEFECT: holding is NOT added
            state["bottle-at"].discard((item, loc))
            state["gripper-empty"].discard(robot)
        elif act == "deliver":
            _, robot, item, loc = step
            if blocked_at is None:
                blocked_at = step_idx
        snapshots.append(_snapshot(state))

    return snapshots, blocked_at


def _snapshot(state):
    return {k: frozenset(v) for k, v in state.items()}


def fig_state_trace(save_path=None):
    robots = ["picker1", "transporter1", "transporter2"]
    items  = ["insulin", "amoxicillin", "ibuprofen"]

    snapshots = _execute_plan(PLAN)
    n_steps = len(snapshots)

    tracked = []
    labels  = []

    for robot in robots:
        for item in items:
            tracked.append(("holding", robot, item))
            labels.append(f"holding({robot[:6]}, {item[:6]})")

    for item in items:
        tracked.append(("order-fulfilled", item))
        labels.append(f"order-fulfilled({item[:6]})")

    for robot in robots:
        tracked.append(("gripper-empty", robot))
        labels.append(f"gripper-empty({robot[:6]})")

    matrix = np.zeros((len(tracked), n_steps))
    for col, snap in enumerate(snapshots):
        for row, pred in enumerate(tracked):
            if pred[0] == "holding":
                matrix[row, col] = 1 if (pred[1], pred[2]) in snap["holding"] else 0
            elif pred[0] == "order-fulfilled":
                matrix[row, col] = 1 if pred[1] in snap["order-fulfilled"] else 0
            elif pred[0] == "gripper-empty":
                matrix[row, col] = 1 if pred[1] in snap["gripper-empty"] else 0

    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"s{i}" for i in range(n_steps)], fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    key_steps = [step for step, act in enumerate(PLAN) if act[0] in ("grasp", "deliver")]
    for ks in key_steps:
        ax.axvline(ks + 0.5, color="#2c3e50", linewidth=1.2, linestyle="--", alpha=0.6)

    for step, act in enumerate(PLAN):
        if act[0] == "grasp":
            ax.text(step + 1, -0.8, f"grasp\n{act[2][:4]}", ha="center",
                    fontsize=6.5, color=ACTION_COLOURS["grasp"], fontweight="bold",
                    va="top")
        elif act[0] == "deliver":
            ax.text(step + 1, -0.8, f"deliver\n{act[2][:4]}", ha="center",
                    fontsize=6.5, color=ACTION_COLOURS["deliver"], fontweight="bold",
                    va="top")

    ax.set_xlabel("Plan Step (state after action)", fontsize=11)
    ax.set_title("Task B — STRIPS Predicate State Trace\n"
                 "(Green = True, Red = False; dashed lines = grasp/deliver steps)",
                 fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Predicate truth value")

    ax.axhline(len(robots) * len(items) - 0.5, color="white", linewidth=2)
    ax.axhline(len(robots) * len(items) + len(items) - 0.5, color="white", linewidth=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def fig_defect_analysis(save_path=None):
    robots = ["picker1", "transporter1", "transporter2"]
    items  = ["insulin", "amoxicillin", "ibuprofen"]

    snaps_correct = _execute_plan(PLAN)
    snaps_defect, blocked_at = _execute_defective_plan(PLAN)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)

    def draw_panel(ax, snapshots, title, blocked_at=None):
        tracked, labels = [], []
        for robot in robots:
            for item in items:
                tracked.append(("holding", robot, item))
                labels.append(f"holding({robot[:5]}, {item[:5]})")
        for item in items:
            tracked.append(("order-fulfilled", item))
            labels.append(f"fulfilled({item[:5]})")

        n_steps = len(snapshots)
        matrix = np.zeros((len(tracked), n_steps))
        for col, snap in enumerate(snapshots):
            for row, pred in enumerate(tracked):
                if pred[0] == "holding":
                    matrix[row, col] = 1 if (pred[1], pred[2]) in snap["holding"] else 0
                elif pred[0] == "order-fulfilled":
                    matrix[row, col] = 1 if pred[1] in snap["order-fulfilled"] else 0

        ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels([f"s{i}" for i in range(n_steps)], fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)

        for step, act in enumerate(PLAN):
            if act[0] == "grasp":
                ax.axvline(step + 0.5, color="#27ae60", linewidth=1.5,
                           linestyle="--", alpha=0.7)
                ax.text(step + 1, -0.6, "grasp", ha="center",
                        fontsize=6, color="#27ae60", fontweight="bold")
            elif act[0] == "deliver":
                ax.axvline(step + 0.5, color="#8e44ad", linewidth=1.5,
                           linestyle="--", alpha=0.7)
                ax.text(step + 1, -0.6, "deliver", ha="center",
                        fontsize=6, color="#8e44ad", fontweight="bold")

        if blocked_at is not None:
            ax.axvspan(blocked_at + 0.5, n_steps - 0.5,
                       color="#e74c3c", alpha=0.12)
            ax.text((blocked_at + n_steps) / 2, len(tracked) / 2,
                    "DEADLOCK\n(holding never set;\ndeliver blocked)",
                    ha="center", va="center", fontsize=9,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="#c0392b",
                              alpha=0.85))

        ax.set_xlabel("Plan Step", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(len(robots) * len(items) - 0.5, color="white", linewidth=2)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    draw_panel(ax0, snaps_correct,
               "Correct Domain\n(holding ?r ?i) added by grasp ✓")
    draw_panel(ax1, snaps_defect, blocked_at=blocked_at,
               title="Defective Domain\n(holding ?r ?i) missing from grasp ✗")

    fig.suptitle("Task B — Critical Thinking: Effect of Missing Grasp ADD Effect\n"
                 "Without (holding ?r ?i) in grasp effects, deliver's precondition "
                 "is never satisfied → plan is UNSOLVABLE",
                 fontsize=12, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="True (green)"),
        mpatches.Patch(color="#c0392b", label="False (red)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def main():
    print("=" * 60)
    print("COMP0224 — Task B: PDDL Visualisations")
    print("Student ID: 23013383")
    print("=" * 60)

    fig_warehouse_graph(
        save_path=os.path.join(OUTPUT_DIR, "task_b_warehouse_graph.png"))

    fig_gantt(
        save_path=os.path.join(OUTPUT_DIR, "task_b_gantt.png"))

    fig_state_trace(
        save_path=os.path.join(OUTPUT_DIR, "task_b_state_trace.png"))

    fig_defect_analysis(
        save_path=os.path.join(OUTPUT_DIR, "task_b_defect_analysis.png"))

    print("\n[Done] All Task B figures saved.")
    plt.show()


if __name__ == "__main__":
    main()
