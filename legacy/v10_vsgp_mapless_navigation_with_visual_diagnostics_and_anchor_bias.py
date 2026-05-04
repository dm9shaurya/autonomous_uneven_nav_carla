
import argparse
import colorsys
import math
import random
import time
from collections import deque

import carla
import numpy as np
import pygame
from pygame.locals import K_a, K_d, K_ESCAPE, K_m, K_p, K_q, K_r, K_s, K_w

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FRONT_WARN_DIST = 8.0
FRONT_STOP_DIST = 4.8
FRONT_EMERG_DIST = 2.0
OBSTACLE_Z_MIN = -1.8
OBSTACLE_Z_MAX = 2.6
FALL_Z_THRESH = -10.0

SPAWN_SETTLE_S = 3.0
STUCK_WINDOW_FRAMES = 90
STUCK_MIN_TRAVEL_M = 1.5
RECOVERY_FRAMES = 18
REVERSE_FRAMES = 16

LIDAR_CHANNELS = "32"
LIDAR_PTS_PER_SEC = "56000"
LIDAR_ROT_FREQ = "10"
LIDAR_RANGE = 50.0
LIDAR_FRAME_SKIP = 2

RGB_WIDTH = 320
RGB_HEIGHT = 180

INVERT_DRIVE = True
_R_FWD = INVERT_DRIVE
_R_REV = not INVERT_DRIVE

ROC = 7.0
VARIANCE_THRESHOLD = 0.7
GP_MAX_POINTS = 50
GP_UPDATE_FREQ = 7

SUBGOAL_DISTANCE_MIN = 3.0
SUBGOAL_DISTANCE_MAX = 14.0
ROBOT_WIDTH = 2.0
ROBOT_MARGIN = 0.5

K_DIR = 0.2
K_DST = 0.3
K_STP = 0.5
K_EXP = 0.12

MAX_ROLL = 0.524
MAX_PITCH = 0.785

STEER_EMA_ALPHA = 0.30
THROTTLE_EMA_ALPHA = 0.36
STEER_RATE_LIMIT = 0.11
THROTTLE_RATE_LIMIT = 0.10
MIN_THROTTLE = 0.22
MAX_THROTTLE = 0.78
CRUISE_SPEED = 7.0
SLOW_SPEED = 3.5

GRID_ALPHA = np.linspace(-math.pi / 2, math.pi / 2, 48)
GRID_BETA = np.linspace(-math.pi / 6, math.pi / 6, 24)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class CustomTimer:
    def __init__(self):
        self.timer = time.perf_counter

    def time(self):
        return self.timer()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    alpha = np.arctan2(y, x)
    beta = np.arcsin(np.clip(z / r, -1.0, 1.0))
    return alpha, beta, r


def local_to_world(vehicle_transform, local_xyz):
    loc = vehicle_transform.location
    yaw = math.radians(vehicle_transform.rotation.yaw)
    lx, ly, lz = local_xyz
    wx = loc.x + math.cos(yaw) * lx - math.sin(yaw) * ly
    wy = loc.y + math.sin(yaw) * lx + math.cos(yaw) * ly
    wz = loc.z + lz
    return carla.Location(wx, wy, wz)


def world_to_local(vehicle_transform, world_loc):
    loc = vehicle_transform.location
    yaw = math.radians(vehicle_transform.rotation.yaw)
    dx = world_loc.x - loc.x
    dy = world_loc.y - loc.y
    lx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
    ly = math.sin(-yaw) * dx + math.cos(-yaw) * dy
    return lx, ly


def hsv_to_rgb_vectorised(hue_arr):
    h6 = hue_arr * 6.0
    hi = np.floor(h6).astype(np.int32) % 6
    f = (h6 - np.floor(h6)).astype(np.float32)
    q = (1.0 - f)
    rgb = np.zeros((len(hue_arr), 3), dtype=np.float32)
    lut = [(1.0, f, 0.0), (q, 1.0, 0.0), (0.0, 1.0, f),
           (0.0, q, 1.0), (f, 0.0, 1.0), (1.0, 0.0, q)]
    for s, (rv, gv, bv) in enumerate(lut):
        m = hi == s
        if not m.any():
            continue
        rgb[m, 0] = rv if np.isscalar(rv) else rv[m]
        rgb[m, 1] = gv if np.isscalar(gv) else gv[m]
        rgb[m, 2] = bv if np.isscalar(bv) else bv[m]
    return (rgb * 255).astype(np.uint8)


def precompute_colour_bar(height, bar_w=12):
    bar = np.zeros((bar_w, height, 3), dtype=np.uint8)
    for py in range(height):
        norm = 1.0 - py / max(1, height)
        hue = (1.0 - norm) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bar[:, py] = (int(r * 255), int(g * 255), int(b * 255))
    return bar


# -----------------------------------------------------------------------------
# Lightweight Variational Sparse GP perception
# -----------------------------------------------------------------------------
class VSGPPerception:
    """
    Lightweight sparse GP-style local perception:
    - fixed inducing points on (alpha, beta)
    - Bayesian linear model in kernel feature space
    - predicts mean and variance over occupancy surface
    """

    def __init__(self):
        alpha_i = np.linspace(-math.pi / 2, math.pi / 2, 8)
        beta_i = np.linspace(-math.pi / 6, math.pi / 6, 6)
        Z = np.array([[a, b] for b in beta_i for a in alpha_i], dtype=np.float64)
        self.Z = Z
        self.M = Z.shape[0]
        self.length_scale = 0.55
        self.signal_var = 1.0
        self.noise_var = 0.12
        self.rq_alpha = 1.0
        self.frame_count = 0
        self.mean_map = None
        self.variance_map = None
        self.last_free_segments = []

        self.Kzz = self.kernel(self.Z, self.Z) + 1e-6 * np.eye(self.M)
        self.Lzz = np.linalg.cholesky(self.Kzz)

    def kernel(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        diff = X[:, None, :] - Y[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        return self.signal_var * np.power(
            1.0 + sq / (2.0 * self.rq_alpha * (self.length_scale ** 2)),
            -self.rq_alpha
        )

    def _features(self, X):
        Kxz = self.kernel(X, self.Z)
        # Phi = Kxz * Kzz^{-1/2}
        Phi = np.linalg.solve(self.Lzz, Kxz.T).T
        return Phi

    def update(self, lidar_points):
        self.frame_count += 1
        if lidar_points is None or len(lidar_points) < 10:
            return [], None, None

        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]
        alpha, beta, r = cartesian_to_spherical(x, y, z)

        valid = (r > 0.45) & (r < ROC * 2.0) & (np.abs(beta) < math.pi / 3)
        alpha = alpha[valid]
        beta = beta[valid]
        r = r[valid]

        if len(alpha) < 20:
            return [], None, None

        fz = ROC - r

        if len(alpha) > GP_MAX_POINTS:
            idx = np.random.choice(len(alpha), GP_MAX_POINTS, replace=False)
            alpha = alpha[idx]
            beta = beta[idx]
            fz = fz[idx]

        X = np.column_stack([alpha, beta])
        y = fz.astype(np.float64)

        Phi = self._features(X)
        A = np.eye(self.M) + (Phi.T @ Phi) / self.noise_var
        Sigma = np.linalg.inv(A)
        mu = (Sigma @ Phi.T @ y) / self.noise_var

        a_grid = GRID_ALPHA
        b_grid = GRID_BETA
        Alpha, Beta = np.meshgrid(a_grid, b_grid)
        Xg = np.column_stack([Alpha.ravel(), Beta.ravel()])
        Phig = self._features(Xg)

        mean = Phig @ mu
        var_latent = np.einsum("ij,jk,ik->i", Phig, Sigma, Phig)
        var = np.maximum(1e-5, self.signal_var - var_latent)

        self.mean_map = mean.reshape(Alpha.shape)
        self.variance_map = var.reshape(Alpha.shape)
        segments = self._segment_free_space(Alpha, Beta, self.variance_map)
        self.last_free_segments = segments
        return segments, self.variance_map, self.mean_map

    def _segment_free_space(self, Alpha, Beta, var_map):
        segments = []
        free_mask = var_map > VARIANCE_THRESHOLD  # high variance = free space
        rows, cols = free_mask.shape

        for row in range(rows):
            start = None
            for col in range(cols):
                if free_mask[row, col] and start is None:
                    start = col
                elif (not free_mask[row, col]) and start is not None:
                    end = col - 1
                    if end >= start:
                        alpha_start = Alpha[row, start]
                        alpha_end = Alpha[row, end]
                        alpha_center = 0.5 * (alpha_start + alpha_end)
                        beta_center = Beta[row, start]
                        width_m = ROC * abs(alpha_end - alpha_start)
                        variance = float(np.mean(var_map[row, start:end + 1]))
                        if width_m >= (ROBOT_WIDTH + ROBOT_MARGIN):
                            segments.append({
                                "alpha_start": float(alpha_start),
                                "alpha_end": float(alpha_end),
                                "alpha": float(alpha_center),
                                "beta": float(beta_center),
                                "width": float(width_m),
                                "variance": variance,
                            })
                    start = None
            if start is not None:
                end = cols - 1
                alpha_start = Alpha[row, start]
                alpha_end = Alpha[row, end]
                alpha_center = 0.5 * (alpha_start + alpha_end)
                beta_center = Beta[row, start]
                width_m = ROC * abs(alpha_end - alpha_start)
                variance = float(np.mean(var_map[row, start:end + 1]))
                if width_m >= (ROBOT_WIDTH + ROBOT_MARGIN):
                    segments.append({
                        "alpha_start": float(alpha_start),
                        "alpha_end": float(alpha_end),
                        "alpha": float(alpha_center),
                        "beta": float(beta_center),
                        "width": float(width_m),
                        "variance": variance,
                    })

        segments.sort(key=lambda s: (abs(s["beta"]), -s["width"], -s["variance"]))
        return segments


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------
class SubgoalPlanner:
    def __init__(self):
        self.subgoals = []
        self.selected_subgoal = None
        self.cost_history = deque(maxlen=180)
        self.trajectory_history = deque(maxlen=500)
        self.predicted_history = deque(maxlen=120)
        self.control_history = deque(maxlen=400)
        self.selected_history = deque(maxlen=220)
        self.visited_cells = deque(maxlen=1200)
        self.cost_candidates = deque(maxlen=32)
        self.spawn_transform = None
        self.anchor_world = None

    def configure_anchor(self, vehicle_transform, world=None):
        self.spawn_transform = vehicle_transform
        loc = vehicle_transform.location
        yaw = math.radians(vehicle_transform.rotation.yaw)
        candidate = carla.Location(
            loc.x + math.cos(yaw) * 200.0,
            loc.y + math.sin(yaw) * 200.0,
            loc.z,
        )
        if world is not None:
            try:
                wp = world.get_map().get_waypoint(
                    candidate,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if wp is not None:
                    candidate = wp.transform.location
            except Exception:
                pass
        self.anchor_world = candidate

    def add_trajectory_point(self, location):
        self.trajectory_history.append((float(location.x), float(location.y)))
        self.visited_cells.append((round(location.x / 4.0), round(location.y / 4.0)))

    def add_control_point(self, speed, steer, throttle, omega, state):
        self.control_history.append({
            "speed": float(speed),
            "steer": float(steer),
            "throttle": float(throttle),
            "omega": float(omega),
            "state": state,
        })

    def _repeat_penalty(self, subgoal):
        wx = subgoal.get("wx", subgoal.get("x", 0.0))
        wy = subgoal.get("wy", subgoal.get("y", 0.0))
        cell = (round(wx / 4.0), round(wy / 4.0))
        return 1.0 if cell in self.visited_cells else 0.0

    def _is_safe(self, sg):
        return bool(
            sg["safe"] and
            abs(sg["roll"]) < MAX_ROLL and
            abs(sg["pitch"]) < MAX_PITCH and
            sg["segment_width"] >= (ROBOT_WIDTH + ROBOT_MARGIN)
        )

    def generate_subgoals(self, free_segments, vehicle_transform, perception):
        self.subgoals = []
        if not free_segments:
            return self.subgoals

        for seg in free_segments:
            span = abs(seg["alpha_end"] - seg["alpha_start"])
            if span <= 1e-6:
                continue

            n_pts = 1
            if seg["width"] > 2.5 * (ROBOT_WIDTH + ROBOT_MARGIN):
                n_pts = 3
            if seg["width"] > 4.5 * (ROBOT_WIDTH + ROBOT_MARGIN):
                n_pts = 4

            alphas = np.linspace(seg["alpha_start"], seg["alpha_end"], n_pts)
            beta = seg["beta"]
            dist = clamp(ROC + 0.5 + 0.1 * seg["width"], SUBGOAL_DISTANCE_MIN, SUBGOAL_DISTANCE_MAX)

            for alpha in alphas:
                lx = dist * math.cos(beta) * math.cos(alpha)
                ly = dist * math.cos(beta) * math.sin(alpha)
                lz = dist * math.sin(beta)

                wloc = local_to_world(vehicle_transform, (lx, ly, lz))
                wx, wy, wz = wloc.x, wloc.y, wloc.z
                terrain_pitch = math.atan2(abs(lz), max(0.25, math.hypot(lx, ly)))

                sg = {
                    "x": float(lx), "y": float(ly), "z": float(lz),
                    "wx": float(wx), "wy": float(wy), "wz": float(wz),
                    "alpha": float(alpha), "beta": float(beta),
                    "distance": float(dist),
                    "segment_width": float(seg["width"]),
                    "variance": float(seg["variance"]),
                    "pitch": float(terrain_pitch),
                    "roll": 0.0,
                    "safe": True,
                    "cost": float("inf"),
                    "cost_breakdown": {},
                }

                if abs(terrain_pitch) >= MAX_PITCH or seg["variance"] < 0.0:
                    sg["safe"] = False

                self.subgoals.append(sg)

        return self.subgoals

    def evaluate_costs(self, vehicle_transform):
        self.cost_candidates.clear()
        best = None
        anchor = self.anchor_world
        anchor_l = None
        if anchor is not None:
            anchor_l = world_to_local(vehicle_transform, anchor)

        for sg in self.subgoals:
            if not self._is_safe(sg):
                sg["cost"] = float("inf")
                continue

            direction_cost = abs(sg["alpha"]) / (math.pi / 2)
            distance_cost = 1.0 - clamp(sg["distance"] / SUBGOAL_DISTANCE_MAX, 0.0, 1.0)

            dz = sg["z"]
            pitch = sg["pitch"]
            steepness_cost = dz * dz + math.exp(math.copysign(1.0, pitch) * dz) * abs(pitch)
            steepness_cost = min(steepness_cost, 10.0) / 10.0

            anchor_cost = 0.0
            if anchor_l is not None:
                anchor_cost = clamp(math.hypot(sg["x"] - anchor_l[0], sg["y"] - anchor_l[1]) / 200.0, 0.0, 1.0)

            repeat_penalty = self._repeat_penalty(sg)
            total = (
                K_DIR * direction_cost +
                K_DST * distance_cost +
                K_STP * steepness_cost +
                K_EXP * anchor_cost +
                0.07 * repeat_penalty
            )
            total = max(0.0, float(total))

            sg["cost"] = total
            sg["cost_breakdown"] = {
                "direction": K_DIR * direction_cost,
                "distance": K_DST * distance_cost,
                "steepness": K_STP * steepness_cost,
                "anchor": K_EXP * anchor_cost,
                "repeat": 0.07 * repeat_penalty,
                "total": total,
            }
            self.cost_candidates.append({
                "wx": sg["wx"],
                "wy": sg["wy"],
                "alpha": sg["alpha"],
                "dist": sg["distance"],
                "cost": total,
                "safe": True,
            })
            if best is None or total < best["cost"]:
                best = sg

        self.selected_subgoal = best
        if best is not None:
            self.selected_history.append((best["wx"], best["wy"]))
            self.cost_history.append(best["cost_breakdown"])
        return best

    def predict_rollout(self, vehicle_transform, horizon=14):
        self.predicted_history.clear()
        if self.selected_subgoal is None:
            return list(self.predicted_history)
        loc = vehicle_transform.location
        sx, sy = self.selected_subgoal["wx"], self.selected_subgoal["wy"]
        for i in range(1, horizon + 1):
            t = i / horizon
            self.predicted_history.append((
                loc.x * (1 - t) + sx * t,
                loc.y * (1 - t) + sy * t,
            ))
        return list(self.predicted_history)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
class VisualizationManager:
    def __init__(self, display_manager):
        self.dm = display_manager
        self.font = pygame.font.SysFont("monospace", 11)
        self.bold_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.cost_surface = None
        self.traj_surface = None
        self.phase_surface = None
        self._init_figs()

    def _init_figs(self):
        self.fig_cost = plt.figure(figsize=(4, 3), dpi=80)
        self.ax_cost = self.fig_cost.add_subplot(111)
        self.canvas_cost = FigureCanvas(self.fig_cost)

        self.fig_traj = plt.figure(figsize=(4, 3), dpi=80)
        self.ax_traj = self.fig_traj.add_subplot(111)
        self.canvas_traj = FigureCanvas(self.fig_traj)

        self.fig_phase = plt.figure(figsize=(4, 3), dpi=80)
        self.ax_phase = self.fig_phase.add_subplot(111)
        self.canvas_phase = FigureCanvas(self.fig_phase)

    def _canvas_to_surface(self, canvas):
        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba())
        arr = np.flipud(arr[:, :, :3])
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def _placeholder(self, ax, title, msg):
        ax.clear()
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    def update(self, planner):
        self._update_cost_plot(planner)
        self._update_traj_plot(planner)
        self._update_phase_plot(planner)

    def _update_cost_plot(self, planner):
        ax = self.ax_cost
        ax.clear()
        if len(planner.cost_history) < 1:
            self._placeholder(ax, "Cost Function", "No received cost data yet")
        else:
            data = list(planner.cost_history)
            x = np.arange(len(data))
            ax.plot(x, [d["direction"] for d in data], "b--", lw=1.2, label="Direction")
            ax.plot(x, [d["distance"] for d in data], "g-.", lw=1.2, label="Distance")
            ax.plot(x, [d["steepness"] for d in data], "r:", lw=1.5, label="Steepness")
            ax.plot(x, [d["total"] for d in data], "k-", lw=2.0, label="Total")
            ax.set_title("Cost Function")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Cost")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_ylim(0, max(1.2, max(d["total"] for d in data) * 1.2))
        self.cost_surface = self._canvas_to_surface(self.canvas_cost)

    def _update_traj_plot(self, planner):
        ax = self.ax_traj
        ax.clear()

        traj = list(planner.trajectory_history)
        pred = list(planner.predicted_history)
        sel = list(planner.selected_history)

        if len(traj) < 1:
            self._placeholder(ax, "Trajectory / Subgoals", "No received trajectory data yet")
        else:
            tx = [p[0] for p in traj]
            ty = [p[1] for p in traj]
            ax.plot(tx, ty, "g-", lw=2.0, label="Actual trajectory")

            if len(pred) > 1:
                px = [p[0] for p in pred]
                py = [p[1] for p in pred]
                ax.plot(px, py, "c--", lw=2.0, label="Predicted rollout")

            if len(sel) > 0:
                sx = [p[0] for p in sel]
                sy = [p[1] for p in sel]
                ax.plot(sx, sy, "y-.", lw=1.3, label="Selected route")

            for sg in planner.subgoals:
                if sg == planner.selected_subgoal:
                    color, marker, size = "yellow", "*", 130
                elif not sg["safe"]:
                    color, marker, size = "red", "x", 45
                elif sg["variance"] > VARIANCE_THRESHOLD * 1.1:
                    color, marker, size = "orange", "o", 35
                else:
                    color, marker, size = "dodgerblue", ".", 24
                ax.scatter(sg["wx"], sg["wy"], c=color, marker=marker, s=size, alpha=0.85)

            if planner.anchor_world is not None:
                ax.scatter(planner.anchor_world.x, planner.anchor_world.y, c="magenta", marker="^", s=100, label="200m anchor")

            ax.scatter(tx[-1], ty[-1], c="white", edgecolors="black", marker="o", s=70, label="Vehicle")
            ax.set_title("Trajectory / Subgoal Selection")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.25)
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(loc="upper right", fontsize=7)

            xs = tx + [p[0] for p in pred] if pred else tx
            ys = ty + [p[1] for p in pred] if pred else ty
            pad = 8.0
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)

            top = sorted(list(planner.cost_candidates), key=lambda c: c["cost"])[:5]
            lines = ["Top subgoals:"]
            for i, c in enumerate(top, 1):
                lines.append(f"{i}. a={math.degrees(c['alpha']):+.0f}°  d={c['dist']:.1f}  c={c['cost']:.2f}")
            ax.text(
                0.02, 0.02, "\n".join(lines),
                transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="black", alpha=0.35, ec="none")
            )

        self.traj_surface = self._canvas_to_surface(self.canvas_traj)

    def _update_phase_plot(self, planner):
        ax = self.ax_phase
        ax.clear()
        ctrls = list(planner.control_history)

        if len(ctrls) < 1:
            self._placeholder(ax, "Velocity Phase Portrait", "No received control data yet")
        else:
            v = [c["speed"] for c in ctrls]
            w = [c["omega"] for c in ctrls]
            ax.plot(v, w, "k-", lw=1.2)
            ax.scatter(v[-1], w[-1], c="red", s=35)
            ax.set_title("Velocity Phase Portrait")
            ax.set_xlabel("Linear velocity (m/s)")
            ax.set_ylabel("Angular velocity (rad/s)")
            ax.grid(True, alpha=0.25)

        self.phase_surface = self._canvas_to_surface(self.canvas_phase)

    def render(self):
        ds = self.dm.get_display_size()
        for col, surf, label in [
            (0, self.cost_surface, "COST FUNCTION"),
            (1, self.traj_surface, "TRAJECTORY / SUBGOALS"),
            (2, self.phase_surface, "VELOCITY PHASE PORTRAIT"),
        ]:
            off = self.dm.get_display_offset([1, col])
            if surf is None:
                panel = pygame.Surface((ds[0], ds[1]))
                panel.fill((18, 18, 28))
                panel.blit(self.bold_font.render(label, True, (220, 220, 220)), (8, 8))
                surf = panel
            scaled = pygame.transform.scale(surf, (ds[0], ds[1]))
            self.dm.display.blit(scaled, off)
            pygame.draw.rect(self.dm.display, (60, 60, 80), (off[0], off[1], ds[0], ds[1]), 1)
            self.dm.display.blit(self.bold_font.render(label, True, (220, 220, 220)), (off[0] + 4, off[1] + 4))


# -----------------------------------------------------------------------------
# Display and sensors
# -----------------------------------------------------------------------------
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA VSGP Mapless Navigation")
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self._font = pygame.font.SysFont("monospace", 13)
        self._hud_lines = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [
            int(self.window_size[0] / self.grid_size[1]),
            int(self.window_size[1] / self.grid_size[0]),
        ]

    def get_display_offset(self, grid_pos):
        ds = self.get_display_size()
        return [int(grid_pos[1] * ds[0]), int(grid_pos[0] * ds[1])]

    def add_sensor(self, s):
        self.sensor_list.append(s)

    def set_hud(self, lines):
        self._hud_lines = lines

    def render(self):
        self.display.fill((10, 10, 20))
        for s in self.sensor_list:
            s.render()
        self._draw_labels()
        self._draw_hud()

    def _draw_labels(self):
        ds = self.get_display_size()
        for s in self.sensor_list:
            off = self.get_display_offset(s.display_pos)
            lbl = self._font.render(s.label, True, (200, 200, 200))
            self.display.blit(lbl, (off[0] + 4, off[1] + 4))
            pygame.draw.rect(self.display, (60, 60, 80), (off[0], off[1], ds[0], ds[1]), 1)

    def _draw_hud(self):
        x = 10
        y = self.window_size[1] - 20 * len(self._hud_lines) - 6
        for line in self._hud_lines:
            surf = self._font.render(line, True, (0, 255, 120))
            self.display.blit(surf, (x, y))
            y += 18

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()


class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, label=""):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.label = label
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.time_proc = 0.0
        self.ticks = 0
        self.lidar_points = None
        self.depth_array = None
        self._lidar_frame = 0
        self._colour_bar = None
        self.sensor = self._init_sensor(sensor_type, transform, attached, sensor_options)
        self.display_man.add_sensor(self)

    def _init_sensor(self, sensor_type, transform, attached, opts):
        bp_lib = self.world.get_blueprint_library()
        ds = self.display_man.get_display_size()

        if sensor_type == "RGBCamera":
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rgb)
            return actor

        if sensor_type == "RainbowLiDAR":
            bp = bp_lib.find("sensor.lidar.ray_cast")
            bp.set_attribute("range", str(LIDAR_RANGE))
            bp.set_attribute("channels", LIDAR_CHANNELS)
            bp.set_attribute("points_per_second", LIDAR_PTS_PER_SEC)
            bp.set_attribute("rotation_frequency", LIDAR_ROT_FREQ)
            bp.set_attribute("dropoff_general_rate", "0.0")
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rainbow_lidar)
            self._colour_bar = precompute_colour_bar(ds[1])
            return actor

        return None

    def _save_rgb(self, image):
        t0 = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        arr = arr[:, :, :3][:, :, ::-1]
        ds = self.display_man.get_display_size()
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.surface = pygame.transform.scale(surf, (ds[0], ds[1]))
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def _save_rainbow_lidar(self, image):
        t0 = self.timer.time()
        self._lidar_frame += 1
        pts = np.frombuffer(image.raw_data, dtype=np.float32).reshape((-1, 4))
        xyz = pts[:, :3].copy()
        self.lidar_points = xyz

        if self._lidar_frame % LIDAR_FRAME_SKIP != 0:
            self.time_proc += self.timer.time() - t0
            self.ticks += 1
            return

        ds = self.display_man.get_display_size()
        lidar_range = 2.0 * float(LIDAR_RANGE)
        xy = xyz[:, :2].copy()
        xy *= min(ds) / lidar_range
        xy += (0.5 * ds[0], 0.5 * ds[1])
        xy = xy.astype(np.int32)

        z_norm = np.clip((xyz[:, 2] - (-2.5)) / 5.5, 0.0, 1.0)
        hue = ((1.0 - z_norm) * 0.75).astype(np.float32)
        rgb_u8 = hsv_to_rgb_vectorised(hue)
        img = np.zeros((ds[0], ds[1], 3), dtype=np.uint8)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < ds[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < ds[1]))
        img[xy[valid, 0], xy[valid, 1]] = rgb_u8[valid]
        cx, cy = ds[0] // 2, ds[1] // 2
        r = 7
        img[cx - 1:cx + 2, max(0, cy - r):min(ds[1], cy + r)] = 255
        img[max(0, cx - r):min(ds[0], cx + r), cy - 1:cy + 2] = 255
        bar_w = self._colour_bar.shape[0]
        img[ds[0] - bar_w:ds[0], :] = self._colour_bar
        self.surface = pygame.surfarray.make_surface(img)
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


# -----------------------------------------------------------------------------
# Navigation controller
# -----------------------------------------------------------------------------
class NavigationController:
    def __init__(self):
        self.vsgp = VSGPPerception()
        self.planner = SubgoalPlanner()
        self.steer_ema = 0.0
        self.throttle_ema = 0.0
        self.pos_history = deque(maxlen=STUCK_WINDOW_FRAMES)
        self.reverse_count = 0
        self.recovery_count = 0
        self.emergency_count = 0
        self.spawn_z = None
        self.frame_count = 0
        self.diag = {
            "state": "INIT",
            "speed": 0.0,
            "steer": 0.0,
            "throttle": 0.0,
            "subgoals": 0,
            "best_cost": float("inf"),
            "front": LIDAR_RANGE,
        }

    def configure_spawn(self, world, vehicle):
        self.spawn_z = vehicle.get_location().z
        self.planner.configure_anchor(vehicle.get_transform(), world)

    def _filter_lidar_for_vsgp(self, pts):
        if pts is None:
            return np.empty((0, 3), dtype=np.float64)
        mask = (
            (pts[:, 0] > 0) & (pts[:, 0] < 26.0) &
            (pts[:, 1] > -12.0) & (pts[:, 1] < 12.0) &
            (pts[:, 2] > -2.5) & (pts[:, 2] < 3.2)
        )
        filtered = pts[mask]
        if len(filtered) > GP_MAX_POINTS * 2:
            idx = np.random.choice(len(filtered), GP_MAX_POINTS * 2, replace=False)
            filtered = filtered[idx]
        return filtered.astype(np.float64)

    def _analyse_lidar_legacy(self, pts):
        if pts is None or len(pts) == 0:
            return None
        mask = (pts[:, 2] > OBSTACLE_Z_MIN) & (pts[:, 2] < OBSTACLE_Z_MAX)
        obs = pts[mask]
        cap = float(LIDAR_RANGE)
        if len(obs) == 0:
            return {"front": cap, "fl": cap, "fr": cap}

        ang = np.arctan2(obs[:, 1], obs[:, 0])
        dist = np.hypot(obs[:, 0], obs[:, 1])
        sw = math.pi / 8

        def sector_min(lo, hi):
            m = (ang >= lo) & (ang < hi)
            return float(np.min(dist[m])) if m.any() else cap

        return {
            "front": sector_min(-sw, sw),
            "fl": sector_min(sw, 3 * sw),
            "fr": sector_min(-3 * sw, -sw),
        }

    def _check_stuck(self):
        if len(self.pos_history) < STUCK_WINDOW_FRAMES:
            return False
        old = self.pos_history[0]
        new = self.pos_history[-1]
        dist = math.hypot(new[0] - old[0], new[1] - old[1])
        return dist < STUCK_MIN_TRAVEL_M and self.diag["speed"] < 0.9

    def record_manual(self, vehicle, control):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        ang = vehicle.get_angular_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.diag["state"] = "MANUAL"
        self.diag["speed"] = speed
        self.diag["steer"] = control.steer
        self.diag["throttle"] = control.throttle
        self.planner.add_trajectory_point(loc)
        self.planner.add_control_point(speed, control.steer, control.throttle, ang.z, "MANUAL")
        self.pos_history.append((loc.x, loc.y))

    def _record_auto(self, vehicle, steer, throttle, state):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        ang = vehicle.get_angular_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.diag["speed"] = speed
        self.diag["steer"] = steer
        self.diag["throttle"] = throttle
        self.diag["state"] = state
        self.planner.add_trajectory_point(loc)
        self.planner.add_control_point(speed, steer, throttle, ang.z, state)
        self.pos_history.append((loc.x, loc.y))

    def compute_control(self, lidar_pts, vehicle):
        self.frame_count += 1
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        rot = vehicle.get_transform().rotation
        ang = vehicle.get_angular_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.diag["speed"] = speed
        self.pos_history.append((loc.x, loc.y))
        self.planner.add_trajectory_point(loc)

        if self.spawn_z is not None and loc.z < self.spawn_z - 7.5:
            self.emergency_count = 30

        if self.emergency_count > 0:
            self.emergency_count -= 1
            self._record_auto(vehicle, 0.0, 0.0, "EMRG-STOP")
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.95, reverse=_R_FWD)

        analysis = self._analyse_lidar_legacy(lidar_pts)
        front = analysis["front"] if analysis is not None else LIDAR_RANGE
        self.diag["front"] = front

        if self.reverse_count > 0:
            self.reverse_count -= 1
            steer_cmd = -0.35 if (self.reverse_count % 2 == 0) else 0.35
            throttle = 0.32
            self._record_auto(vehicle, steer_cmd, throttle, "REVERSING")
            return carla.VehicleControl(throttle=throttle, steer=steer_cmd, brake=0.0, reverse=_R_REV)

        if self.recovery_count > 0:
            self.recovery_count -= 1
            steer_cmd = 0.42 if (self.recovery_count % 2 == 0) else -0.42
            throttle = 0.28
            if front < FRONT_STOP_DIST:
                throttle = 0.18
            self._record_auto(vehicle, steer_cmd, throttle, "RECOVERING")
            return carla.VehicleControl(throttle=throttle, steer=steer_cmd, brake=0.0, reverse=_R_FWD)

        nav_pts = self._filter_lidar_for_vsgp(lidar_pts)
        free_segments, _, _ = self.vsgp.update(nav_pts)
        self.planner.generate_subgoals(free_segments, vehicle.get_transform(), self.vsgp)
        best = self.planner.evaluate_costs(vehicle.get_transform())
        self.planner.predict_rollout(vehicle.get_transform())

        self.diag["subgoals"] = len(self.planner.subgoals)
        self.diag["best_cost"] = best["cost"] if best is not None else float("inf")

        if best is None:
            self.diag["state"] = "SEARCHING"
            steer_target = 0.0
            if analysis is not None:
                steer_target = -0.28 if analysis["fl"] > analysis["fr"] else 0.28
                if front < FRONT_WARN_DIST:
                    steer_target *= 1.2
            throttle_target = 0.34 if speed < 2.5 else 0.26
            if self._check_stuck():
                self.recovery_count = RECOVERY_FRAMES
            self.steer_ema = clamp(self.steer_ema + clamp(steer_target - self.steer_ema, -STEER_RATE_LIMIT, STEER_RATE_LIMIT), -1.0, 1.0)
            self.throttle_ema = clamp(self.throttle_ema + clamp(throttle_target - self.throttle_ema, -THROTTLE_RATE_LIMIT, THROTTLE_RATE_LIMIT), 0.0, 1.0)
            self._record_auto(vehicle, self.steer_ema, self.throttle_ema, "SEARCHING")
            return carla.VehicleControl(throttle=float(self.throttle_ema), steer=float(self.steer_ema), brake=0.0, reverse=_R_FWD)

        steer_target = clamp(best["alpha"] / (math.pi / 3), -1.0, 1.0)
        distance_bias = clamp(best["distance"] / SUBGOAL_DISTANCE_MAX, 0.0, 1.0)
        target_speed = SLOW_SPEED + (CRUISE_SPEED - SLOW_SPEED) * distance_bias

        if abs(steer_target) > 0.55:
            target_speed *= 0.82
        if best["variance"] > VARIANCE_THRESHOLD * 1.2:
            target_speed *= 0.88

        if front < FRONT_EMERG_DIST:
            self.emergency_count = 12
            self._record_auto(vehicle, 0.0, 0.0, "EMRG-STOP")
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.95, reverse=_R_FWD)

        if front < FRONT_STOP_DIST:
            self.diag["state"] = "AVOIDING"
            target_speed = min(target_speed, 2.6)
            steer_target += -0.18 if analysis and analysis["fl"] < analysis["fr"] else 0.18
        elif speed < target_speed - 0.45:
            self.diag["state"] = "ACCELERATING"
        elif abs(steer_target) > 0.35:
            self.diag["state"] = "TURNING"
        else:
            self.diag["state"] = "CRUISING"

        pitch_abs = abs(rot.pitch)
        roll_abs = abs(rot.roll)
        if pitch_abs > 24.0 or roll_abs > 18.0:
            target_speed = min(target_speed, 2.4)
            self.diag["state"] = "AVOIDING"

        if self._check_stuck():
            self.recovery_count = RECOVERY_FRAMES

        throttle_target = clamp(0.32 + 0.14 * (target_speed - speed), MIN_THROTTLE, MAX_THROTTLE)
        if speed < 0.7:
            throttle_target = max(throttle_target, 0.50)

        steer_prev = self.steer_ema
        throttle_prev = self.throttle_ema
        steer_limited = steer_prev + clamp(steer_target - steer_prev, -STEER_RATE_LIMIT, STEER_RATE_LIMIT)
        throttle_limited = throttle_prev + clamp(throttle_target - throttle_prev, -THROTTLE_RATE_LIMIT, THROTTLE_RATE_LIMIT)
        self.steer_ema = clamp((1.0 - STEER_EMA_ALPHA) * steer_prev + STEER_EMA_ALPHA * steer_limited, -1.0, 1.0)
        self.throttle_ema = clamp((1.0 - THROTTLE_EMA_ALPHA) * throttle_prev + THROTTLE_EMA_ALPHA * throttle_limited, 0.0, 1.0)

        self._record_auto(vehicle, self.steer_ema, self.throttle_ema, self.diag["state"])
        return carla.VehicleControl(
            throttle=float(self.throttle_ema),
            steer=float(self.steer_ema),
            brake=0.0,
            reverse=_R_FWD,
        )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def run_simulation(args, client):
    display_manager = None
    visualizer = None
    world = None
    original_settings = None
    actors = []
    nav = NavigationController()

    try:
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            tm = client.get_trafficmanager(8000)
            settings = world.get_settings()
            tm.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.cybertruck")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle = None
        for sp in spawn_points:
            sp.location.z += 5.0
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"Spawned at ({sp.location.x:.1f}, {sp.location.y:.1f}, {sp.location.z:.1f})")
                break

        if vehicle is None:
            raise RuntimeError("No valid spawn point found")
        actors.append(vehicle)

        print(f"Settling {SPAWN_SETTLE_S}s ...")
        t0 = time.time()
        while time.time() - t0 < SPAWN_SETTLE_S:
            world.tick() if args.sync else world.wait_for_tick()

        nav.configure_spawn(world, vehicle)
        print(f"VSGP running. No fake plot data. Torch={('available' if 'torch' in globals() else 'unused')}")

        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])
        visualizer = VisualizationManager(display_manager)

        front_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[0, 0], label="FRONT VIEW"
        )
        actors.append(front_cam.sensor)

        rainbow_lidar = SensorManager(
            world, display_manager, "RainbowLiDAR",
            carla.Transform(carla.Location(z=2.5)),
            vehicle, {}, display_pos=[0, 1], label="RAINBOW LiDAR"
        )
        actors.append(rainbow_lidar.sensor)

        rear_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.5, z=2.0), carla.Rotation(yaw=180, pitch=-10)),
            vehicle, {}, display_pos=[0, 2], label="REAR VIEW"
        )
        actors.append(rear_cam.sensor)

        clock = pygame.time.Clock()
        running = True
        frame = 0
        manual_mode = False
        manual_steer = 0.0

        print("ESC/Q: quit   R: reset   P/M: toggle manual/auto")
        print("Manual: W forward   S brake/reverse   A/D steer")

        while running:
            clock.tick(20)
            frame += 1

            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_r:
                        nav.pos_history.clear()
                        nav.planner.trajectory_history.clear()
                        nav.planner.predicted_history.clear()
                        nav.planner.control_history.clear()
                        nav.planner.selected_history.clear()
                        nav.planner.cost_history.clear()
                        nav.planner.cost_candidates.clear()
                        nav.reverse_count = 0
                        nav.recovery_count = 0
                        nav.emergency_count = 0
                        print("Navigation state reset")
                    elif event.key in (K_p, K_m):
                        manual_mode = not manual_mode
                        manual_steer = 0.0
                        print("Mode:", "MANUAL" if manual_mode else "AUTO")

            if manual_mode:
                keys = pygame.key.get_pressed()
                throttle = 0.0
                brake = 0.10
                reverse = _R_FWD
                steer_tgt = 0.0

                if keys[K_w]:
                    throttle, brake, reverse = 0.60, 0.0, _R_FWD
                if keys[K_s]:
                    vel = vehicle.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    if spd > 0.6:
                        brake, throttle, reverse = 0.80, 0.0, _R_FWD
                    else:
                        throttle, brake, reverse = 0.35, 0.0, _R_REV
                if keys[K_a]:
                    steer_tgt = -0.55
                if keys[K_d]:
                    steer_tgt = 0.55

                manual_steer = 0.45 * steer_tgt + 0.55 * manual_steer
                control = carla.VehicleControl(
                    throttle=float(clamp(throttle, 0, 1)),
                    steer=float(clamp(manual_steer, -1, 1)),
                    brake=float(clamp(brake, 0, 1)),
                    reverse=reverse,
                )
                vehicle.apply_control(control)
                nav.record_manual(vehicle, control)
                mode_tag = "MANUAL"
            else:
                lidar_pts = rainbow_lidar.lidar_points
                control = nav.compute_control(lidar_pts, vehicle)
                vehicle.apply_control(control)
                mode_tag = "AUTO"

            visualizer.update(nav.planner)

            display_manager.set_hud([
                f"MODE  : {mode_tag:<6}  FRAME: {frame}",
                f"STATE : {nav.diag['state']:<12}  SUBGOALS: {nav.diag['subgoals']:3d}",
                f"SPEED : {nav.diag['speed'] * 3.6:5.1f} km/h  STEER: {nav.diag['steer']:+.3f}  THR: {nav.diag['throttle']:.2f}",
                f"FRONT : {nav.diag['front']:4.1f} m  BEST COST: {nav.diag['best_cost']:.3f}",
                f"RECOV : {nav.recovery_count:3d}  REV : {nav.reverse_count:3d}  EMRG: {nav.emergency_count:3d}",
            ])

            display_manager.render()
            visualizer.render()
            pygame.display.flip()

    finally:
        print("\nCleaning up ...")
        try:
            if display_manager is not None:
                display_manager.destroy()
        except Exception:
            pass
        try:
            if world is not None and original_settings is not None:
                world.apply_settings(original_settings)
        except Exception:
            pass
        try:
            if actors:
                client.apply_batch([carla.command.DestroyActor(a) for a in actors if a is not None])
        except Exception:
            for a in actors:
                try:
                    if a is not None:
                        a.destroy()
                except Exception:
                    pass
        plt.close("all")
        pygame.quit()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="CARLA VSGP Mapless Navigation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--sync", action="store_true", default=True)
    parser.add_argument("--async", dest="sync", action="store_false")
    parser.add_argument("--res", default="1280x720", metavar="WxH")
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    run_simulation(args, client)


if __name__ == "__main__":
    main()