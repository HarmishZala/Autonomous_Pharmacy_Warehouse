import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_config import generate_scenario

STUDENT_ID = 23013383


class ORCARobot:

    def __init__(self, robot_id, position, goal, radius=0.4, max_speed=1.0):
        self.robot_id = robot_id
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.velocity = np.zeros(2)
        self.radius = radius
        self.max_speed = max_speed
        self.trajectory = [self.position.copy()]
        self.reached_goal = False

    @property
    def preferred_velocity(self):
        diff = self.goal - self.position
        dist = np.linalg.norm(diff)
        if dist < 0.1:
            self.reached_goal = True
            return np.zeros(2)
        return (diff / dist) * min(self.max_speed, dist)


def compute_orca_half_planes(robot, neighbours, tau=5.0):
    planes = []

    for other in neighbours:
        rel_pos = other.position - robot.position
        rel_vel = robot.velocity - other.velocity
        dist = np.linalg.norm(rel_pos)
        combined_radius = robot.radius + other.radius

        if dist < 1e-6:
            normal = np.array([1.0, 0.0])
            point = robot.velocity + 0.5 * robot.max_speed * normal
            planes.append((point, normal))
            continue

        if dist > combined_radius:
            direction = rel_pos / dist
            sin_angle = min(combined_radius / dist, 1.0)
            cos_angle = np.sqrt(max(0, 1.0 - sin_angle ** 2))

            cutoff_centre = rel_pos / tau
            w = rel_vel - cutoff_centre
            w_len = np.linalg.norm(w)

            if w_len < 1e-6:
                normal = direction.copy()
            else:
                dot_product = np.dot(w, direction)
                if dot_product < 0 and dot_product * dot_product > \
                        combined_radius * combined_radius * w_len * w_len / (dist * dist):
                    normal = w / w_len
                else:
                    cross = rel_pos[0] * w[1] - rel_pos[1] * w[0]
                    if cross > 0:
                        normal = np.array([
                            direction[0] * cos_angle - direction[1] * sin_angle,
                            direction[0] * sin_angle + direction[1] * cos_angle
                        ])
                    else:
                        normal = np.array([
                            direction[0] * cos_angle + direction[1] * sin_angle,
                            -direction[0] * sin_angle + direction[1] * cos_angle
                        ])

            u_mag = combined_radius / tau - np.dot(w, normal)
            u = u_mag * normal
            point = robot.velocity + 0.5 * u
        else:
            normal = rel_pos / dist if dist > 1e-6 else np.array([1.0, 0.0])
            penetration = combined_radius - dist
            point = robot.velocity - 0.5 * penetration * normal

        planes.append((point, normal))

    return planes


def solve_velocity(preferred_vel, planes, max_speed):
    result = preferred_vel.copy()

    for i, (point, normal) in enumerate(planes):
        if np.dot(result - point, normal) >= 0:
            continue
        tangent = np.array([-normal[1], normal[0]])
        t = np.dot(result - point, tangent)
        result = point + t * tangent

    speed = np.linalg.norm(result)
    if speed > max_speed:
        result = result / speed * max_speed

    return result


def simulate_orca(robots, walls, dt=0.25, max_steps=200, tau=5.0,
                  sensing_radius=5.0, grid_size=10):
    # Wall agents with larger radius for safety buffer
    wall_agents = []
    for wr, wc in walls:
        wa = ORCARobot("wall", [wc, wr], [wc, wr], radius=0.7, max_speed=0.0)
        wa.velocity = np.zeros(2)
        wall_agents.append(wa)

    def enforce_wall_boundary(pos, robot_radius):
        for wr, wc in walls:
            wx, wy = float(wc), float(wr)
            dx = pos[0] - wx
            dy = pos[1] - wy
            dist = np.sqrt(dx * dx + dy * dy)
            min_sep = robot_radius + 0.5
            if dist < min_sep and dist > 1e-6:
                push = (min_sep - dist) / dist
                pos[0] += dx * push
                pos[1] += dy * push
        pos[0] = np.clip(pos[0], 0.0, grid_size - 1.0)
        pos[1] = np.clip(pos[1], 0.0, grid_size - 1.0)
        return pos

    for step in range(max_steps):
        all_done = all(r.reached_goal for r in robots)
        if all_done:
            return {r.robot_id: r.trajectory for r in robots}, step

        for robot in robots:
            if robot.reached_goal:
                continue

            neighbours = []
            for other in robots:
                if other.robot_id == robot.robot_id:
                    continue
                if np.linalg.norm(other.position - robot.position) < sensing_radius:
                    neighbours.append(other)

            for wa in wall_agents:
                if np.linalg.norm(wa.position - robot.position) < sensing_radius:
                    neighbours.append(wa)

            planes = compute_orca_half_planes(robot, neighbours, tau)
            v_pref = robot.preferred_velocity
            new_vel = solve_velocity(v_pref, planes, robot.max_speed)
            robot.velocity = new_vel

        for robot in robots:
            if not robot.reached_goal:
                robot.position = robot.position + robot.velocity * dt
                robot.position = enforce_wall_boundary(robot.position, robot.radius)
                robot.trajectory.append(robot.position.copy())
            else:
                robot.trajectory.append(robot.position.copy())

    return {r.robot_id: r.trajectory for r in robots}, max_steps


ROBOT_COLOURS = {
    "transporter1": "#e74c3c",
    "transporter2": "#3498db",
}


def visualise_orca(grid_size, walls, trajectories, starts, goals, title,
                   save_path=None):
    fig, ax = plt.subplots(figsize=(9, 9))

    bg = np.ones((grid_size, grid_size))
    for wr, wc in walls:
        bg[wr, wc] = 0.3
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper",
              extent=(-0.5, grid_size - 0.5, grid_size - 0.5, -0.5))

    for rid, traj in trajectories.items():
        colour = ROBOT_COLOURS.get(rid, "#2ecc71")
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, "-", color=colour, linewidth=2, alpha=0.7,
                label=f"{rid}")

        start = starts[rid]
        ax.plot(start[0], start[1], "s", color=colour, markersize=14,
                markeredgecolor="black", markeredgewidth=1.5)
        ax.text(start[0], start[1], "S", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

        goal = goals[rid]
        ax.plot(goal[0], goal[1], "*", color=colour, markersize=18,
                markeredgecolor="black", markeredgewidth=1)
        ax.text(goal[0] + 0.3, goal[1] - 0.3, "G", fontsize=8,
                fontweight="bold", color=colour)

        ax.plot(xs[-1], ys[-1], "o", color=colour, markersize=10,
                markeredgecolor="black")

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def visualise_orca_snapshots(grid_size, walls, trajectories, snapshot_times,
                             title, save_path=None):
    n = len(snapshot_times)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, t in enumerate(snapshot_times):
        ax = axes[idx]
        bg = np.ones((grid_size, grid_size))
        for wr, wc in walls:
            bg[wr, wc] = 0.3
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper",
                  extent=(-0.5, grid_size - 0.5, grid_size - 0.5, -0.5))

        for rid, traj in trajectories.items():
            colour = ROBOT_COLOURS.get(rid, "#2ecc71")
            tidx = min(t, len(traj) - 1)
            pos = traj[tidx]
            circle = Circle((pos[0], pos[1]), 0.4, color=colour, alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], rid[-1], ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(which="minor", color="grey", linewidth=0.3)
        ax.set_title(f"Step {t}", fontsize=10, fontweight="bold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    return fig


def main():
    scenario = generate_scenario(STUDENT_ID)
    size = scenario["size"]
    p_slip = scenario["p_slip"]

    print("=" * 60)
    print("COMP0224 — Task C (Part 2): ORCA (Decentralised Avoidance)")
    print(f"Student ID : {STUDENT_ID}")
    print(f"Grid Size  : {size}x{size}")
    print("Scenario   : Communication failure — robots act independently")
    print("=" * 60)

    walls = [
        (0, 3), (1, 3),
        (3, 0), (3, 1),
        (2, 7), (3, 7),
        (5, 3), (6, 3),
        (6, 8), (7, 8),
    ]

    starts = {
        "transporter1": np.array([0.0, 0.0]),
        "transporter2": np.array([9.0, 0.0]),
    }
    goals = {
        "transporter1": np.array([9.0, 9.0]),
        "transporter2": np.array([0.0, 9.0]),
    }

    print(f"\nRobot starts : { {k: v.tolist() for k, v in starts.items()} }")
    print(f"Robot goals  : { {k: v.tolist() for k, v in goals.items()} }")

    robots = [
        ORCARobot("transporter1", starts["transporter1"], goals["transporter1"],
                  radius=0.4, max_speed=1.0),
        ORCARobot("transporter2", starts["transporter2"], goals["transporter2"],
                  radius=0.4, max_speed=1.0),
    ]

    print("\n[ORCA] Simulating (no communication between robots) ...")
    trajectories, steps = simulate_orca(robots, walls, dt=0.25, max_steps=300,
                                        tau=5.0, sensing_radius=5.0,
                                        grid_size=size)

    print(f"[ORCA] Simulation complete in {steps} steps")
    for rid, traj in trajectories.items():
        final = traj[-1]
        goal = goals[rid]
        dist = np.linalg.norm(final - goal)
        print(f"  {rid}: final_pos=({final[0]:.2f}, {final[1]:.2f}), "
              f"goal_dist={dist:.3f}, reached={dist < 0.5}")

    min_dist = float("inf")
    for t in range(min(len(trajectories["transporter1"]),
                       len(trajectories["transporter2"]))):
        d = np.linalg.norm(trajectories["transporter1"][t] -
                           trajectories["transporter2"][t])
        min_dist = min(min_dist, d)
    print(f"  Minimum inter-robot distance: {min_dist:.3f} "
          f"(collision threshold: {0.8:.1f})")
    print(f"  Collision-free: {min_dist >= 0.8 - 0.01}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    visualise_orca(size, walls, trajectories, starts, goals,
                   title="ORCA: Decentralised Collision Avoidance (No Communication)",
                   save_path=os.path.join(output_dir, "task_c_orca_trajectories.png"))

    max_t = max(len(t) for t in trajectories.values())
    snapshot_times = [0, max_t // 4, max_t // 2, 3 * max_t // 4, max_t - 1]
    visualise_orca_snapshots(size, walls, trajectories, snapshot_times,
                             title="ORCA: Position Snapshots",
                             save_path=os.path.join(output_dir, "task_c_orca_snapshots.png"))

    plt.show()
    print("\n[Done] ORCA complete.")


if __name__ == "__main__":
    main()
