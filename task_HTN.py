import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_config import generate_scenario

STUDENT_ID = 23013383


class WorldState:

    def __init__(self, floor_friction=0.92):
        self.data = {
            "bottle_found":  False,
            "bottle_coords": None,
            "floor_slippery": floor_friction,
            "robot_at_dock": False,
            "holding_bottle": False,
        }

    def __str__(self):
        return (f"  bottle_found={self.data['bottle_found']}, "
                f"holding={self.data['holding_bottle']}, "
                f"at_dock={self.data['robot_at_dock']}, "
                f"friction={self.data['floor_slippery']:.2f}")


class TaskNode:

    def __init__(self, name):
        self.name = name

    def execute(self, state):
        raise NotImplementedError


class MotionPrimitive(TaskNode):

    def __init__(self, name, primitive_type, cost=10):
        super().__init__(name)
        self.primitive_type = primitive_type
        self.cost = cost

    def check_preconditions(self, state):
        if self.primitive_type == "Grasp_DMP":
            if not state.data["bottle_found"]:
                return False
        elif self.primitive_type == "Place_DMP":
            if not state.data["robot_at_dock"]:
                return False
            if not state.data["holding_bottle"]:
                return False
        return True

    def apply_effects(self, state):
        if self.primitive_type == "Vision_Scan":
            state.data["bottle_found"] = True
            state.data["bottle_coords"] = (2.5, 3.1)
            print(f"    [Effect] Bottle located at {state.data['bottle_coords']}")
        elif self.primitive_type == "Grasp_DMP":
            state.data["holding_bottle"] = True
            state.data["bottle_found"] = False
            print(f"    [Effect] Bottle grasped — gripper occupied")
        elif self.primitive_type == "Standard_Controller":
            state.data["robot_at_dock"] = True
            print(f"    [Effect] Arrived at dock (standard controller)")
        elif self.primitive_type == "Traction_Controller":
            state.data["robot_at_dock"] = True
            print(f"    [Effect] Arrived at dock (traction controller)")
        elif self.primitive_type == "Place_DMP":
            state.data["holding_bottle"] = False
            print(f"    [Effect] Bottle placed at dock — order fulfilled")

    def execute_dmp(self, state):
        print(f"    [DMP] Executing {self.primitive_type} ...")

        if "Controller" in self.primitive_type:
            slip_chance = 1.0 - state.data["floor_slippery"]
            if "Traction" not in self.primitive_type:
                if np.random.random() < slip_chance:
                    print(f"    [!] SLIP DETECTED: {self.name} failed "
                          f"(slip_prob={slip_chance:.2f})")
                    return False
            else:
                residual_slip = slip_chance * 0.1
                if np.random.random() < residual_slip:
                    print(f"    [!] SLIP DETECTED even with traction: "
                          f"{self.name} failed (residual={residual_slip:.3f})")
                    return False

        time.sleep(0.3)
        return True

    def execute(self, state):
        print(f"  [Primitive] {self.name} ({self.primitive_type})")
        if not self.check_preconditions(state):
            print(f"    [Precondition FAILED] for {self.name}")
            return False
        if not self.execute_dmp(state):
            return False
        self.apply_effects(state)
        return True


class AndNode(TaskNode):

    def __init__(self, name, sub_tasks):
        super().__init__(name)
        self.sub_tasks = sub_tasks

    def execute(self, state):
        print(f"\n[HTN] AND-decompose: {self.name}  "
              f"({len(self.sub_tasks)} sub-tasks)")
        for i, task in enumerate(self.sub_tasks, 1):
            print(f"  [{i}/{len(self.sub_tasks)}] → {task.name}")
            if not task.execute(state):
                print(f"[FAIL] {task.name} failed — aborting {self.name}")
                return False
        print(f"[OK] {self.name} completed successfully")
        return True


class OrNode(TaskNode):

    def __init__(self, name, strategies):
        super().__init__(name)
        self.strategies = strategies

    def execute(self, state):
        print(f"\n[HTN] OR-decompose: {self.name}  "
              f"({len(self.strategies)} strategies)")
        for i, strategy in enumerate(self.strategies, 1):
            print(f"  Strategy {i}/{len(self.strategies)}: {strategy.name}")
            if strategy.execute(state):
                print(f"[OK] {self.name} succeeded via: {strategy.name}")
                return True
            print(f"  Strategy {strategy.name} failed — trying next ...")
        print(f"[FAIL] {self.name} — all strategies exhausted")
        return False


def build_mission():
    find_bottle = MotionPrimitive("FindBottle", "Vision_Scan", cost=5)
    pick_bottle = MotionPrimitive("PickBottle", "Grasp_DMP", cost=15)
    deliver_bottle = MotionPrimitive("DeliverBottle", "Place_DMP", cost=10)

    fast_nav = MotionPrimitive("Fast_Navigation", "Standard_Controller", cost=10)
    safe_nav = MotionPrimitive("Safe_Navigation", "Traction_Controller", cost=20)
    navigate_to_dock = OrNode("Navigate_to_Dock", [fast_nav, safe_nav])

    mission = AndNode("FulfilOrder", [
        find_bottle,
        pick_bottle,
        navigate_to_dock,
        deliver_bottle,
    ])

    return mission


def print_htn_tree(node, indent=0):
    prefix = "  " * indent
    if isinstance(node, AndNode):
        print(f"{prefix}AND: {node.name}")
        for child in node.sub_tasks:
            print_htn_tree(child, indent + 1)
    elif isinstance(node, OrNode):
        print(f"{prefix}OR: {node.name}")
        for child in node.strategies:
            print_htn_tree(child, indent + 1)
    elif isinstance(node, MotionPrimitive):
        print(f"{prefix}LEAF: {node.name} [{node.primitive_type}] (cost={node.cost})")


def visualise_htn_tree(save_path=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    nodes = {
        "FulfilOrder":      (5.0, 5.0),
        "FindBottle":       (1.5, 3.0),
        "PickBottle":       (4.0, 3.0),
        "Navigate_to_Dock": (6.5, 3.0),
        "DeliverBottle":    (9.0, 3.0),
        "Fast_Navigation":  (5.5, 1.0),
        "Safe_Navigation":  (7.5, 1.0),
    }

    meta = {
        "FulfilOrder":      ("AND", "#2c3e50", "FulfilOrder\n(AND)"),
        "FindBottle":       ("LEAF", "#3498db", "FindBottle\n[Vision_Scan]"),
        "PickBottle":       ("LEAF", "#27ae60", "PickBottle\n[Grasp_DMP]"),
        "Navigate_to_Dock": ("OR",   "#e67e22", "Navigate_to_Dock\n(OR)"),
        "DeliverBottle":    ("LEAF", "#8e44ad", "DeliverBottle\n[Place_DMP]"),
        "Fast_Navigation":  ("LEAF", "#e74c3c", "Fast_Navigation\n[Standard_Ctrl]"),
        "Safe_Navigation":  ("LEAF", "#2ecc71", "Safe_Navigation\n[Traction_Ctrl]"),
    }

    edges = [
        ("FulfilOrder", "FindBottle"),
        ("FulfilOrder", "PickBottle"),
        ("FulfilOrder", "Navigate_to_Dock"),
        ("FulfilOrder", "DeliverBottle"),
        ("Navigate_to_Dock", "Fast_Navigation"),
        ("Navigate_to_Dock", "Safe_Navigation"),
    ]
    for parent, child in edges:
        px, py = nodes[parent]
        cx, cy = nodes[child]
        ax.annotate("", xy=(cx, cy + 0.35), xytext=(px, py - 0.35),
                     arrowprops=dict(arrowstyle="-|>", color="#7f8c8d",
                                     lw=1.8, connectionstyle="arc3,rad=0.0"))

    for name, (x, y) in nodes.items():
        ntype, colour, label = meta[name]
        if ntype in ("AND", "OR"):
            box_style = "round,pad=0.3"
            fontsize = 10
        else:
            box_style = "round,pad=0.25"
            fontsize = 9
        bbox = FancyBboxPatch((x - 0.9, y - 0.3), 1.8, 0.6,
                               boxstyle=box_style, facecolor=colour,
                               edgecolor="black", linewidth=1.5, alpha=0.9)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    for i, child in enumerate(["FindBottle", "PickBottle", "Navigate_to_Dock", "DeliverBottle"]):
        cx, cy = nodes[child]
        ax.text(cx, cy + 0.5, f"({i+1})", ha="center", fontsize=8,
                color="#7f8c8d", fontweight="bold")

    ax.annotate("try first", xy=(5.5, 1.4), fontsize=8, color="#e74c3c",
                ha="center", fontweight="bold")
    ax.annotate("fallback", xy=(7.5, 1.4), fontsize=8, color="#2ecc71",
                ha="center", fontweight="bold")

    ax.text(5.0, 4.2, "AND: all children must succeed sequentially",
            ha="center", fontsize=9, color="#2c3e50", style="italic")
    ax.text(6.5, 2.2, "OR: first success wins\n(handles slippery floor)",
            ha="center", fontsize=9, color="#e67e22", style="italic")

    legend_handles = [
        mpatches.Patch(color="#2c3e50", label="AND node (sequential)"),
        mpatches.Patch(color="#e67e22", label="OR node (alternative strategies)"),
        mpatches.Patch(color="#3498db", label="Leaf: Vision primitive"),
        mpatches.Patch(color="#27ae60", label="Leaf: Grasp DMP"),
        mpatches.Patch(color="#8e44ad", label="Leaf: Place DMP"),
        mpatches.Patch(color="#e74c3c", label="Leaf: Standard Controller"),
        mpatches.Patch(color="#2ecc71", label="Leaf: Traction Controller"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
              framealpha=0.9, ncol=2)

    ax.set_title("Task D — HTN Decomposition: FulfilOrder AND/OR Graph\n"
                 "High-level mission decomposes into primitives linked to DMP controllers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_trial_results(trial_data, p_slip, save_path=None):
    n = len(trial_data)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    colours = ["#27ae60" if t["success"] else "#e74c3c" for t in trial_data]
    ax.bar(range(1, n + 1), [1] * n, color=colours, edgecolor="black",
           linewidth=0.6, alpha=0.85)
    success_count = sum(1 for t in trial_data if t["success"])
    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xticks(range(1, n + 1))
    ax.set_title(f"Trial Outcomes ({success_count}/{n} succeeded)",
                 fontsize=11, fontweight="bold")
    legend = [mpatches.Patch(color="#27ae60", label="Success"),
              mpatches.Patch(color="#e74c3c", label="Failure")]
    ax.legend(handles=legend, fontsize=9)

    ax2 = axes[1]
    strat_counts = {"Standard\n(fast)": 0, "Traction\n(safe)": 0, "Both failed": 0}
    for t in trial_data:
        s = t.get("strategy_used", "unknown")
        if "Standard" in s or "Fast" in s:
            strat_counts["Standard\n(fast)"] += 1
        elif "Traction" in s or "Safe" in s:
            strat_counts["Traction\n(safe)"] += 1
        else:
            strat_counts["Both failed"] += 1
    strat_labels = list(strat_counts.keys())
    strat_vals = list(strat_counts.values())
    strat_colours = ["#e74c3c", "#2ecc71", "#95a5a6"]
    bars = ax2.bar(strat_labels, strat_vals, color=strat_colours,
                   edgecolor="black", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, strat_vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     str(val), ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Navigation Strategy Selected by OR Node",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(strat_vals) * 1.3 + 1)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    ax3 = axes[2]
    steps_data = [t.get("steps", 0) for t in trial_data]
    bar_colours = ["#27ae60" if t["success"] else "#e74c3c" for t in trial_data]
    ax3.bar(range(1, n + 1), steps_data, color=bar_colours,
            edgecolor="black", linewidth=0.6, alpha=0.85)
    ax3.set_xlabel("Trial", fontsize=11)
    ax3.set_ylabel("Primitives Executed", fontsize=11)
    ax3.set_xticks(range(1, n + 1))
    ax3.set_title("Primitives Executed per Trial", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Task D — HTN Trial Analysis (p_slip = {p_slip:.2f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_dmp_timeline(trial_data, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    dmp_colours = {
        "Vision_Scan":          "#3498db",
        "Grasp_DMP":            "#27ae60",
        "Standard_Controller":  "#e74c3c",
        "Traction_Controller":  "#2ecc71",
        "Place_DMP":            "#8e44ad",
    }

    y_spacing = 1.0
    for trial_idx, trial in enumerate(trial_data):
        y = (len(trial_data) - trial_idx - 1) * y_spacing
        x = 0
        for dmp_name, dmp_ok in trial.get("dmp_log", []):
            colour = dmp_colours.get(dmp_name, "#95a5a6")
            alpha = 0.85 if dmp_ok else 0.4
            hatch = "" if dmp_ok else "///"
            ax.barh(y, 1, left=x, height=0.6, color=colour, alpha=alpha,
                    edgecolor="black", linewidth=0.6, hatch=hatch)
            label = dmp_name.replace("_Controller", "\nCtrl").replace("_", "\n")
            ax.text(x + 0.5, y, label, ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white" if dmp_ok else "black")
            x += 1

    ax.set_yticks([i * y_spacing for i in range(len(trial_data))])
    ax.set_yticklabels([f"Trial {len(trial_data) - i}"
                        for i in range(len(trial_data))], fontsize=9)
    ax.set_xlabel("DMP Execution Step", fontsize=11)
    ax.set_xlim(-0.2, 7)
    ax.set_title("Task D — DMP Execution Timeline per Trial\n"
                 "(hatched = failed DMP; OR node retries with fallback)",
                 fontsize=12, fontweight="bold")

    legend_handles = [mpatches.Patch(color=c, label=n) for n, c in dmp_colours.items()]
    legend_handles.append(mpatches.Patch(facecolor="#cccccc", hatch="///",
                                          edgecolor="black", label="Failed"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def run_pharmacy_sim():
    scenario = generate_scenario(STUDENT_ID)
    p_slip = scenario["p_slip"]
    floor_friction = 1.0 - p_slip

    print("=" * 60)
    print("COMP0224 — Task D: HTN Mission Execution")
    print(f"Student ID     : {STUDENT_ID}")
    print(f"Slip Probability: {p_slip}")
    print(f"Floor Friction  : {floor_friction:.2f}")
    print("=" * 60)

    mission = build_mission()
    print("\nHTN Decomposition Tree:")
    print("-" * 40)
    print_htn_tree(mission)
    print("-" * 40)

    n_trials = 20
    successes = 0
    trial_data = []

    print(f"\nRunning {n_trials} trials (seed varies per trial) ...\n")

    for trial in range(1, n_trials + 1):
        np.random.seed(trial * 42)
        state = WorldState(floor_friction=floor_friction)

        print(f"\n{'='*60}")
        print(f"TRIAL {trial}/{n_trials}")
        print(f"State: {state}")
        print(f"{'='*60}")

        dmp_log = []
        mission = build_mission()

        original_execute_dmp = MotionPrimitive.execute_dmp

        def logged_execute_dmp(self, state, _log=dmp_log):
            result = original_execute_dmp(self, state)
            _log.append((self.primitive_type, result))
            return result

        MotionPrimitive.execute_dmp = logged_execute_dmp
        result = mission.execute(state)
        MotionPrimitive.execute_dmp = original_execute_dmp

        if result:
            successes += 1
            print(f"\n>>> TRIAL {trial}: ORDER FULFILLED <<<")
        else:
            print(f"\n>>> TRIAL {trial}: ORDER FAILED <<<")

        print(f"Final state: {state}")

        strategy = "unknown"
        for dmp_name, dmp_ok in dmp_log:
            if "Standard" in dmp_name and dmp_ok:
                strategy = "Standard (fast)"
                break
            elif "Traction" in dmp_name and dmp_ok:
                strategy = "Traction (safe)"
                break

        trial_data.append({
            "trial": trial,
            "success": result,
            "strategy_used": strategy,
            "steps": len([d for d in dmp_log if d[1]]),
            "dmp_log": dmp_log,
        })

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Trials     : {n_trials}")
    print(f"Successes  : {successes}/{n_trials} ({100*successes/n_trials:.0f}%)")
    print(f"Slip prob  : {p_slip:.2f}")
    print(f"Friction   : {floor_friction:.2f}")
    print(f"\nThe OR node (Navigate_to_Dock) handles uncertainty by")
    print(f"falling back from Standard_Controller to Traction_Controller")
    print(f"when a slip is detected on the warehouse floor.")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    visualise_htn_tree(
        save_path=os.path.join(output_dir, "task_d_htn_tree.png"))

    visualise_trial_results(
        trial_data, p_slip,
        save_path=os.path.join(output_dir, "task_d_htn_trials.png"))

    visualise_dmp_timeline(
        trial_data,
        save_path=os.path.join(output_dir, "task_d_htn_dmp_timeline.png"))

    plt.show()
    print("\n[Done] HTN complete — all figures saved.")


if __name__ == "__main__":
    run_pharmacy_sim()
