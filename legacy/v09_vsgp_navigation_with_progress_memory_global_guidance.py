"""
CARLA VSGP Mapless Navigation v6
Based on: "Autonomous Mapless Navigation on Uneven Terrains" (ICRA 2024)

UI Layout (2×3 grid):
  Row 0: [FRONT CAM] | [LiDAR + SUBGOAL WINDOW] | [REAR CAM]
  Row 1: [TRAJECTORY MAP (GP+Actual)] | [COST FUNCTION PLOT] | [v-ω PHASE PORTRAIT]

Navigation: SGP-based exploration (no fixed goal), manual toggle via P key.
Controls: ESC/Q=quit, R=reset, P=toggle manual/auto, W/S/A/D=manual drive
"""

import carla
import argparse
import random
import time
import math
import colorsys
import numpy as np
from collections import deque

# ─────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ─────────────────────────────────────────────────────────────
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not found – GP uses fallback variance estimation.")

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q, K_r, K_w, K_s, K_a, K_d, K_p
except ImportError:
    raise RuntimeError("pygame not installed – run: pip install pygame")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not found – plots will be blank panels.")


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Sensors
LIDAR_CHANNELS       = "32"
LIDAR_PTS_PER_SEC    = "72000"
LIDAR_ROT_FREQ       = "10"
LIDAR_RANGE          = "50"
LIDAR_FRAME_SKIP     = 2
RGB_WIDTH            = 320
RGB_HEIGHT           = 180
RAINBOW_Z_MIN        = -2.5
RAINBOW_Z_MAX        =  3.0

# Vehicle
INVERT_DRIVE    = True
_R_FWD          = INVERT_DRIVE
_R_REV          = not INVERT_DRIVE
MAX_SPEED_MS    = 7.0
THROTTLE_CRUISE = 0.72
THROTTLE_SLOW   = 0.38
WHEELBASE       = 2.8        # metres  (approx Cybertruck)
MAX_STEER_RAD   = 0.52       # physical max steer angle

# Safety
FRONT_WARN_DIST     = 8.0
FRONT_STOP_DIST     = 4.5
FRONT_EMRG_DIST     = 2.8
OBSTACLE_Z_MIN      = -1.8
OBSTACLE_Z_MAX      =  2.5
MAX_ROLL            = 0.524  # 30 deg – Clearpath spec
MAX_PITCH           = 0.785  # 45 deg – Clearpath spec

# Stuck recovery
STUCK_WINDOW_FRAMES = 90
STUCK_MIN_TRAVEL_M  = 1.5
REVERSE_FRAMES      = 45

# Spawn
SPAWN_SETTLE_S = 3.0
FALL_Z_THRESH  = -10.0

# ── SGP / VSGP paper parameters ──────────────────────────────
ROC                = 7.0    # occupancy surface radius (m)
GP_LENGTH_SCALE    = 0.5    # rad
GP_NOISE           = 0.1
GP_MAX_POINTS      = 300
GP_UPDATE_FREQ     = 3
VARIANCE_THRESHOLD = 0.70   # Vth from paper

# Subgoal geometry
SUBGOAL_DIST_MIN  = 3.0
SUBGOAL_DIST_MAX  = 12.0
ROBOT_WIDTH       = 2.0
ROBOT_MARGIN      = 0.5

# Cost weights (paper: kdir=0.2, kdst=0.3, kstp=0.5)
K_DIR     = 0.20
K_DST     = 0.30
K_STP     = 0.45
K_EXPLORE = 0.05   # novelty bonus for unexplored headings

# EMA smoothing
STEER_EMA_ALPHA    = 0.30
THROTTLE_EMA_ALPHA = 0.40


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

class CustomTimer:
    def __init__(self):
        self.timer = time.perf_counter
    def time(self):
        return self.timer()


def _hsv_to_rgb_vectorised(hue_arr):
    """Pure-numpy HSV→RGB for array of hues (S=1, V=1)."""
    h6 = hue_arr * 6.0
    hi = np.floor(h6).astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    q  = (1.0 - f)
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


def _precompute_colour_bar(height, bar_w=12):
    bar = np.zeros((bar_w, height, 3), dtype=np.uint8)
    for py in range(height):
        norm = 1.0 - py / height
        hue  = (1.0 - norm) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bar[:, py] = (int(r * 255), int(g * 255), int(b * 255))
    return bar


def cartesian_to_spherical(x, y, z):
    r     = np.sqrt(x**2 + y**2 + z**2)
    r_s   = np.where(r < 1e-6, 1e-6, r)
    alpha = np.arctan2(y, x)
    beta  = np.arcsin(np.clip(z / r_s, -1, 1))
    return alpha, beta, r


def world_transform(x_veh, y_veh, vehicle_transform):
    """Transform vehicle-relative (x,y) → world (x,y)."""
    loc = vehicle_transform.location
    yaw = math.radians(vehicle_transform.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    xw = loc.x + x_veh * c - y_veh * s
    yw = loc.y + x_veh * s + y_veh * c
    return xw, yw


# ─────────────────────────────────────────────────────────────
# VSGP PERCEPTION  (Section V.A of the paper)
# ─────────────────────────────────────────────────────────────

class VSGPPerception:
    """
    Sparse Gaussian Process Local Perception Model.

    Pipeline:
      LiDAR → spherical (α, β, r)
            → occupancy f(z) = r_oc − r
            → VSGP (mean + variance surface)
            → variance threshold → free-space segments
    """

    def __init__(self):
        self.gp          = None
        self.X_train     = None
        self.y_train     = None
        self.variance_map = None
        self.mean_map     = None
        self.frame_count  = 0
        self.use_gp       = SKLEARN_AVAILABLE

        # Full 360° azimuth grid for exploration
        self._alpha_grid = np.linspace(-math.pi, math.pi, 72)
        self._beta_grid  = np.linspace(-math.pi / 6, math.pi / 6, 24)
        self._Alpha, self._Beta = np.meshgrid(self._alpha_grid, self._beta_grid)
        self._X_test = np.column_stack([self._Alpha.ravel(), self._Beta.ravel()])

        if self.use_gp:
            kernel = (C(1.0, (1e-3, 1e3))
                      * RBF(GP_LENGTH_SCALE, (1e-2, 1e2))
                      + WhiteKernel(GP_NOISE))
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=0,
                normalize_y=True
            )

    def update(self, lidar_pts):
        """Process LiDAR → (free_segments, variance_map, mean_map)."""
        self.frame_count += 1

        if lidar_pts is None or len(lidar_pts) < 10:
            return [], None, None

        x, y, z = lidar_pts[:, 0], lidar_pts[:, 1], lidar_pts[:, 2]
        alpha, beta, r = cartesian_to_spherical(x, y, z)

        # Valid points: reasonable range, exclude ground bounce and sky
        valid = (r > 0.5) & (r < ROC * 2.0) & (np.abs(beta) < math.pi / 3)
        alpha, beta, r = alpha[valid], beta[valid], r[valid]

        if len(alpha) < 15:
            return [], None, None

        f_z = ROC - r  # occupancy function from paper

        # Sparse: downsample
        if len(alpha) > GP_MAX_POINTS:
            idx = np.random.choice(len(alpha), GP_MAX_POINTS, replace=False)
            alpha, beta, f_z = alpha[idx], beta[idx], f_z[idx]

        X = np.column_stack([alpha, beta])

        # Train GP every GP_UPDATE_FREQ frames
        if self.frame_count % GP_UPDATE_FREQ == 0 and self.use_gp:
            try:
                self.X_train = X
                self.y_train = f_z
                self.gp.fit(X, f_z)
            except Exception:
                pass

        # Predict mean + variance over full grid
        if self.use_gp and self.gp is not None and self.X_train is not None:
            try:
                mean, std = self.gp.predict(self._X_test, return_std=True)
                self.mean_map     = mean.reshape(self._Alpha.shape)
                self.variance_map = (std ** 2).reshape(self._Alpha.shape)
            except Exception:
                self._fallback_maps(f_z)
        else:
            self._fallback_maps(f_z)

        # Segment free space from variance map
        free_segments = self._segment_free_space()
        return free_segments, self.variance_map, self.mean_map

    def _fallback_maps(self, f_z):
        h, w = len(self._beta_grid), len(self._alpha_grid)
        self.variance_map = np.ones((h, w)) * float(np.var(f_z))
        self.mean_map     = np.ones((h, w)) * float(np.mean(f_z))

    def _segment_free_space(self):
        """
        Extract free-space segments from variance surface.
        Free = variance > Vth (no LiDAR hits → high uncertainty = open space).
        """
        segments = []
        if self.variance_map is None:
            return segments

        # High variance → free space (paper Section V.B)
        free_mask = self.variance_map > VARIANCE_THRESHOLD

        for col in range(self.variance_map.shape[1]):
            col_free = free_mask[:, col]
            start    = None
            for row in range(len(col_free)):
                if col_free[row] and start is None:
                    start = row
                elif not col_free[row] and start is not None:
                    self._add_segment(segments, col, start, row)
                    start = None
            if start is not None:
                self._add_segment(segments, col, start, len(col_free))

        return segments

    def _add_segment(self, segments, col, r0, r1):
        if r1 - r0 < 2:
            return
        beta_c  = float(np.mean(self._Beta[r0:r1, col]))
        alpha_c = float(self._Alpha[r0, col])
        var_c   = float(np.mean(self.variance_map[r0:r1, col]))
        segments.append({
            "alpha":    alpha_c,
            "beta":     beta_c,
            "width":    r1 - r0,
            "variance": var_c
        })


# ─────────────────────────────────────────────────────────────
# SUBGOAL PLANNER  (Section V.B–C of the paper)
# ─────────────────────────────────────────────────────────────

class SubgoalPlanner:
    """
    Generate candidate subgoals from free-space segments,
    evaluate cost J(g) = kdir·α + kdst·D + kstp·Cstp,
    and select g* = argmin J.
    Exploration mode: no fixed goal – prefer novelty & forward progress.
    """

    def __init__(self):
        self.subgoals          = []
        self.selected_subgoal  = None
        self.cost_history      = deque(maxlen=120)
        self.trajectory_history    = deque(maxlen=300)  # world-frame actual
        self.gp_trajectory_history = deque(maxlen=300)  # world-frame GP-predicted
        self.control_history   = deque(maxlen=150)
        # Exploration memory: track recently selected headings
        self.visited_headings  = deque(maxlen=20)
        self.no_path_count     = 0
        self.explore_turn_dir  = 1   # ±1

    # ── Subgoal generation ───────────────────────────────────

    def generate_subgoals(self, free_segments, vehicle_transform, perception):
        self.subgoals = []
        if not free_segments:
            return self.subgoals

        yaw = math.radians(vehicle_transform.rotation.yaw)

        for seg in free_segments:
            # Width check: segment must be wider than robot + safety margin
            for dist in np.linspace(SUBGOAL_DIST_MIN, SUBGOAL_DIST_MAX, 4):
                seg_w = seg["width"] * (math.pi / 24.0) * dist  # approx metric width
                if seg_w < (ROBOT_WIDTH + ROBOT_MARGIN):
                    continue

                alpha = seg["alpha"]
                beta  = seg["beta"]

                # Vehicle-relative Cartesian
                xv = dist * math.cos(beta) * math.cos(alpha)
                yv = dist * math.cos(beta) * math.sin(alpha)
                zv = dist * math.sin(beta)

                # GP surface lookup for height
                mean_z = 0.0
                if perception.mean_map is not None:
                    try:
                        ai = int(np.clip(
                            (alpha + math.pi) / (2 * math.pi) * 72, 0, 71))
                        bi = int(np.clip(
                            (beta + math.pi / 6) / (math.pi / 3) * 24, 0, 23))
                        mean_z = float(perception.mean_map[bi, ai])
                    except Exception:
                        pass

                # Pitch/roll estimate from GP gradient
                pitch, roll = beta, 0.0
                if perception.mean_map is not None:
                    try:
                        mm = perception.mean_map
                        ai = int(np.clip(
                            (alpha + math.pi) / (2 * math.pi) * 72, 0, 71))
                        bi = int(np.clip(
                            (beta + math.pi / 6) / (math.pi / 3) * 24, 0, 23))
                        if bi + 1 < mm.shape[0]:
                            pitch = float(mm[bi + 1, ai] - mm[bi, ai])
                        if ai + 1 < mm.shape[1]:
                            roll  = float(mm[bi, ai + 1] - mm[bi, ai])
                    except Exception:
                        pass

                # World-frame position (for trajectory plot)
                xw, yw = world_transform(xv, yv, vehicle_transform)

                safe = (abs(roll) < MAX_ROLL) and (abs(pitch) < MAX_PITCH)

                self.subgoals.append({
                    "x": xv, "y": yv, "z": zv + mean_z,   # vehicle-relative
                    "x_world": xw, "y_world": yw,           # world-frame
                    "alpha": alpha, "beta": beta,
                    "distance": dist,
                    "variance": seg.get("variance", 1.0),
                    "pitch": pitch, "roll": roll,
                    "safe": safe,
                    "cost": float("inf"),
                    "cost_breakdown": {}
                })

        return self.subgoals

    # ── Cost evaluation (Eq. 6 of the paper + exploration term) ─

    def evaluate_costs(self, vehicle_transform):
        for sg in self.subgoals:
            if not sg["safe"]:
                sg["cost"] = float("inf")
                sg["cost_breakdown"] = {
                    "direction": 0.0, "distance": 0.0,
                    "steepness": 0.0, "explore": 0.0, "total": float("inf")
                }
                continue

            # ① Direction cost – prefer roughly straight ahead
            dir_cost = abs(sg["alpha"]) / (math.pi / 2)
            dir_cost = min(dir_cost, 1.0)

            # ② Distance/progress cost – prefer farther (max exploration)
            dst_cost = 1.0 - np.clip(sg["distance"] / SUBGOAL_DIST_MAX, 0, 1)

            # ③ Steepness cost  Cstp = dz² + exp(sign(ψ)·dz)·|ψ|  (paper Eq.6)
            dz    = sg["z"]
            psi   = sg["pitch"]
            s_psi = float(np.sign(psi)) if psi != 0.0 else 0.0
            stp_cost = dz ** 2 + math.exp(np.clip(s_psi * dz, -5, 5)) * abs(psi)
            stp_cost = min(stp_cost, 10.0) / 10.0   # normalise

            # ④ Exploration novelty bonus – penalise recently-visited headings
            exp_cost = 0.0
            if len(self.visited_headings) > 0:
                diffs = [abs(sg["alpha"] - h) for h in self.visited_headings]
                min_diff = min(diffs)
                exp_cost = max(0.0, 1.0 - min_diff / (math.pi / 3))

            total = (K_DIR     * dir_cost
                     + K_DST   * dst_cost
                     + K_STP   * stp_cost
                     + K_EXPLORE * exp_cost)

            sg["cost"] = total
            sg["cost_breakdown"] = {
                "direction": K_DIR * dir_cost,
                "distance":  K_DST * dst_cost,
                "steepness": K_STP * stp_cost,
                "explore":   K_EXPLORE * exp_cost,
                "total":     total
            }

        # Record best for history
        safe_sgs = [s for s in self.subgoals
                    if s["safe"] and s["cost"] < float("inf")]
        if safe_sgs:
            best = min(safe_sgs, key=lambda s: s["cost"])
            self.cost_history.append(best["cost_breakdown"])

    def select_best_subgoal(self):
        safe = [s for s in self.subgoals
                if s["safe"] and s["cost"] < float("inf")]
        if not safe:
            self.selected_subgoal = None
            self.no_path_count += 1
            return None

        self.no_path_count = 0
        self.selected_subgoal = min(safe, key=lambda s: s["cost"])

        # Track heading for diversity
        self.visited_headings.append(self.selected_subgoal["alpha"])

        # GP-predicted trajectory (world frame)
        self.gp_trajectory_history.append(
            (self.selected_subgoal["x_world"],
             self.selected_subgoal["y_world"])
        )
        return self.selected_subgoal

    def add_trajectory_point(self, location):
        self.trajectory_history.append((location.x, location.y))

    def add_control_point(self, speed, steer, angular_v):
        self.control_history.append({
            "speed": speed,
            "steer": steer,
            "angular_v": angular_v
        })


# ─────────────────────────────────────────────────────────────
# VISUALIZATION MANAGER  – 6-panel UI
# ─────────────────────────────────────────────────────────────

class VisualizationManager:
    """
    Panels:
      [0,0] Front cam      [0,1] LiDAR + Subgoal Window  [0,2] Rear cam
      [1,0] Trajectory Map [1,1] Cost Function Plot       [1,2] v-ω Phase Portrait
    """

    # Colour palette
    C_BEST       = (0,   255,   0)
    C_REJECT     = (255,  50,  50)
    C_UNCERTAIN  = (255, 200,   0)
    C_CANDIDATE  = (100, 100, 255)
    C_VEHICLE    = (255, 255, 255)
    C_GRID       = ( 40,  40,  60)

    def __init__(self, display_manager):
        self.dm    = display_manager
        self.font  = pygame.font.SysFont("monospace", 11)
        self.bfont = pygame.font.SysFont("monospace", 12, bold=True)

        self.traj_surface  = None
        self.cost_surface  = None
        self.phase_surface = None
        self._frame        = 0
        self._plot_interval = 3

        if MATPLOTLIB_AVAILABLE:
            self._init_plots()

    def _init_plots(self):
        plt.style.use("dark_background")
        BG = "#1a1a2e"

        # Trajectory map
        self.fig_traj, self.ax_traj = plt.subplots(figsize=(3.5, 2.2), dpi=80)
        self.fig_traj.patch.set_facecolor(BG)
        self.canvas_traj = FigureCanvas(self.fig_traj)

        # Cost function
        self.fig_cost, self.ax_cost = plt.subplots(figsize=(3.5, 2.2), dpi=80)
        self.fig_cost.patch.set_facecolor(BG)
        self.canvas_cost = FigureCanvas(self.fig_cost)

        # v-ω phase portrait
        self.fig_phase, self.ax_phase = plt.subplots(figsize=(3.5, 2.2), dpi=80)
        self.fig_phase.patch.set_facecolor(BG)
        self.canvas_phase = FigureCanvas(self.fig_phase)

    # ── Update all three matplotlib panels ───────────────────

    def update_plots(self, planner, vehicle):
        if not MATPLOTLIB_AVAILABLE:
            return
        self._frame += 1
        if self._frame % self._plot_interval != 0:
            return

        self._render_trajectory(planner)
        self._render_cost(planner)
        self._render_phase_portrait(planner)

    # ── Trajectory map ───────────────────────────────────────

    def _render_trajectory(self, planner):
        ax = self.ax_traj
        ax.clear()
        ax.set_facecolor("#16213e")

        # Actual trajectory (green)
        if len(planner.trajectory_history) > 1:
            pts = list(planner.trajectory_history)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#2ecc71", lw=2.0, label="Actual", solid_capstyle="round")

        # GP-predicted trajectory (cyan dashed)
        if len(planner.gp_trajectory_history) > 1:
            gpts = list(planner.gp_trajectory_history)
            ax.plot([p[0] for p in gpts], [p[1] for p in gpts],
                    color="#00d4ff", lw=1.5, ls=":", label="GP Predicted", alpha=0.75)

        # Current subgoals (world frame)
        all_x, all_y = [], []
        for sg in planner.subgoals:
            wx, wy = sg.get("x_world", sg["x"]), sg.get("y_world", sg["y"])
            all_x.append(wx); all_y.append(wy)
            if sg is planner.selected_subgoal:
                ax.scatter(wx, wy, c="#ffff00", marker="*", s=160, zorder=10,
                           edgecolors="white", lw=0.5)
            elif not sg["safe"]:
                ax.scatter(wx, wy, c="#ff3333", marker="x", s=50, zorder=5)
            elif sg["variance"] > VARIANCE_THRESHOLD * 0.8:
                ax.scatter(wx, wy, c="#ffaa00", marker="s", s=55, zorder=6,
                           edgecolors="white", lw=0.4)
            else:
                ax.scatter(wx, wy, c="#6688ff", marker=".", s=28, zorder=4,
                           edgecolors="none")

        # Vehicle marker
        if planner.trajectory_history:
            vx, vy = planner.trajectory_history[-1]
            ax.scatter(vx, vy, c="#2ecc71", marker="^", s=120, zorder=12,
                       edgecolors="white", lw=1.0)

        ax.set_xlabel("X (m)", fontsize=8, color="#aaa")
        ax.set_ylabel("Y (m)", fontsize=8, color="#aaa")
        ax.set_title("Trajectory & Subgoals", fontsize=9, color="#eee", pad=3)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.4, facecolor="#1a1a2e")
        ax.grid(True, alpha=0.12, color="#555")
        ax.tick_params(colors="#888", labelsize=7)

        # Auto-zoom around vehicle
        traj_pts = list(planner.trajectory_history)
        if len(traj_pts) > 1:
            xs = [p[0] for p in traj_pts] + all_x
            ys = [p[1] for p in traj_pts] + all_y
            mg = 8.0
            ax.set_xlim(min(xs) - mg, max(xs) + mg)
            ax.set_ylim(min(ys) - mg, max(ys) + mg)
        ax.set_aspect("equal", adjustable="datalim")

        self.fig_traj.tight_layout(pad=0.4)
        self.canvas_traj.draw()
        self.traj_surface = self._to_pygame(self.canvas_traj)

    # ── Cost function plot ───────────────────────────────────

    def _render_cost(self, planner):
        ax = self.ax_cost
        ax.clear()
        ax.set_facecolor("#16213e")

        if len(planner.cost_history) > 1:
            costs  = list(planner.cost_history)
            xs     = list(range(len(costs)))
            tot    = [c.get("total", 0)     for c in costs]
            dir_   = [c.get("direction", 0) for c in costs]
            dst_   = [c.get("distance", 0)  for c in costs]
            stp_   = [c.get("steepness", 0) for c in costs]

            ax.plot(xs, tot,  color="#e94560", lw=2.0, label="Total")
            ax.plot(xs, dir_, color="#7fffff", lw=1.3, ls="--", label="Direction")
            ax.plot(xs, dst_, color="#00b4d8", lw=1.3, ls="-.", label="Distance")
            ax.plot(xs, stp_, color="#f77f00", lw=1.3, ls=":",  label="Steepness")

            ax.set_ylim(0, max(0.5, max(tot) * 1.35) if tot else 1.0)

        ax.set_xlabel("Frame", fontsize=8, color="#aaa")
        ax.set_ylabel("Cost",  fontsize=8, color="#aaa")
        ax.set_title("Cost Function J(g)", fontsize=9, color="#eee", pad=3)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.4, facecolor="#1a1a2e",
                  ncol=2)
        ax.grid(True, alpha=0.12, color="#555")
        ax.tick_params(colors="#888", labelsize=7)

        self.fig_cost.tight_layout(pad=0.4)
        self.canvas_cost.draw()
        self.cost_surface = self._to_pygame(self.canvas_cost)

    # ── v-ω Phase Portrait ───────────────────────────────────

    def _render_phase_portrait(self, planner):
        """
        Phase portrait: linear velocity v (m/s) vs angular velocity ω (rad/s).
        ω is estimated from steering using bicycle model: ω ≈ v·tan(δ)/L.
        Trail is coloured from pale (old) → bright (recent).
        """
        ax = self.ax_phase
        ax.clear()
        ax.set_facecolor("#16213e")

        ctrls = list(planner.control_history)
        if len(ctrls) > 3:
            v_arr  = np.array([c["speed"]     for c in ctrls])
            om_arr = np.array([c["angular_v"]  for c in ctrls])
            n      = len(v_arr)
            # Colour gradient: dim → bright
            alphas = np.linspace(0.15, 1.0, n)
            cmap   = plt.get_cmap("plasma")
            # Scatter trail
            for i in range(n - 1):
                ax.plot(v_arr[i:i+2], om_arr[i:i+2],
                        color=cmap(i / n), alpha=float(alphas[i]), lw=1.5)
            # Current point
            ax.scatter(v_arr[-1], om_arr[-1], c="#ffff00", s=70, zorder=10,
                       edgecolors="white", lw=0.7, label="Current")

        # Reference lines
        ax.axhline(0, color="#555", lw=0.8, ls="--")
        ax.axvline(0, color="#555", lw=0.8, ls="--")

        ax.set_xlabel("v  (m/s)",   fontsize=8, color="#aaa")
        ax.set_ylabel("ω  (rad/s)", fontsize=8, color="#aaa")
        ax.set_title("v – ω Phase Portrait", fontsize=9, color="#eee", pad=3)
        ax.grid(True, alpha=0.12, color="#555")
        ax.tick_params(colors="#888", labelsize=7)
        if len(ctrls) > 3:
            ax.legend(fontsize=6, framealpha=0.4, facecolor="#1a1a2e")

        self.fig_phase.tight_layout(pad=0.4)
        self.canvas_phase.draw()
        self.phase_surface = self._to_pygame(self.canvas_phase)

    # ── Subgoal Selection Window (overlay on LiDAR panel) ────

    def render_subgoal_overlay(self, surface, planner, slot_size):
        """
        Polar subgoal selection window drawn in the bottom third of the LiDAR panel.
        Shows: all candidate subgoals, selected (green★), unsafe (red✕),
               uncertain (orange□), candidates (blue●), and route line.
        """
        w, h = slot_size
        panel_h = h // 3
        panel_y = h - panel_h

        # Semi-transparent panel background
        bg = pygame.Surface((w, panel_h), pygame.SRCALPHA)
        bg.fill((8, 12, 28, 195))
        surface.blit(bg, (0, panel_y))

        cx = w // 2
        cy = panel_y + panel_h // 2
        scale = min(w, panel_h) / (SUBGOAL_DIST_MAX * 2.2)

        # Polar grid rings
        for d in [3.0, 6.0, 9.0, 12.0]:
            r_px = int(d * scale)
            pygame.draw.circle(surface, self.C_GRID, (cx, cy), r_px, 1)
        # Heading lines
        for ang_deg in range(0, 360, 30):
            a = math.radians(ang_deg)
            ex = cx + int(math.cos(a) * 12 * scale)
            ey = cy - int(math.sin(a) * 12 * scale)
            pygame.draw.line(surface, self.C_GRID, (cx, cy), (ex, ey), 1)

        # Draw subgoals
        for sg in planner.subgoals:
            ang  = sg["alpha"]
            dist = sg["distance"]
            # In LiDAR frame: x=forward, y=left;  screen: right=x-axis, up=y-axis
            px = cx + int(dist * scale * math.sin(ang))   # left-right
            py = cy - int(dist * scale * math.cos(ang))   # forward = up

            px = max(0, min(w - 1, px))
            py = max(panel_y, min(h - 1, py))

            if sg is planner.selected_subgoal:
                # Draw route line to best subgoal
                pygame.draw.line(surface, self.C_BEST, (cx, cy), (px, py), 2)
                pygame.draw.circle(surface, self.C_BEST, (px, py), 7)
                pygame.draw.circle(surface, (0, 0, 0),  (px, py), 7, 1)
                # Star overlay
                lbl = self.bfont.render("★", True, (255, 255, 0))
                surface.blit(lbl, (px - 6, py - 8))
            elif not sg["safe"]:
                pygame.draw.line(surface, self.C_REJECT,
                                 (px - 4, py - 4), (px + 4, py + 4), 2)
                pygame.draw.line(surface, self.C_REJECT,
                                 (px + 4, py - 4), (px - 4, py + 4), 2)
            elif sg["variance"] > VARIANCE_THRESHOLD * 0.8:
                pygame.draw.rect(surface, self.C_UNCERTAIN,
                                 (px - 4, py - 4, 8, 8), 1)
            else:
                pygame.draw.circle(surface, self.C_CANDIDATE, (px, py), 3)

        # Vehicle dot
        pygame.draw.circle(surface, self.C_VEHICLE, (cx, cy), 5)
        pygame.draw.circle(surface, (0, 0, 0),      (cx, cy), 5, 1)
        # Forward arrow
        pygame.draw.line(surface, (0, 255, 200),
                         (cx, cy), (cx, cy - int(4 * scale)), 2)

        # Panel title + legend
        title = self.bfont.render("SUBGOAL SELECTION  (polar view)", True, (180, 220, 255))
        surface.blit(title, (5, panel_y + 2))

        legend = [("★ Best", self.C_BEST),
                  ("● Safe", self.C_CANDIDATE),
                  ("□ Uncert", self.C_UNCERTAIN),
                  ("✕ Unsafe", self.C_REJECT)]
        lx = 5
        for txt, col in legend:
            s = self.font.render(txt, True, col)
            surface.blit(s, (lx, h - 15))
            lx += s.get_width() + 8

    # ── Blit matplotlib panels to display ────────────────────

    def render(self, planner, vehicle, display_manager):
        ds = display_manager.get_display_size()
        sw, sh = ds[0], ds[1]

        def blit_at(surface, row, col):
            if surface is None:
                return
            scaled = pygame.transform.scale(surface, (sw, sh))
            off    = display_manager.get_display_offset([row, col])
            display_manager.display.blit(scaled, off)
            pygame.draw.rect(display_manager.display, (60, 60, 80),
                             (off[0], off[1], sw, sh), 1)

        blit_at(self.traj_surface,  1, 0)   # Trajectory map
        blit_at(self.cost_surface,  1, 1)   # Cost function
        blit_at(self.phase_surface, 1, 2)   # Phase portrait

        # Subgoal overlay on LiDAR panel
        ov = pygame.Surface((sw, sh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 0))
        self.render_subgoal_overlay(ov, planner, (sw, sh))
        off = display_manager.get_display_offset([0, 1])
        display_manager.display.blit(ov, off)

    @staticmethod
    def _to_pygame(canvas):
        buf = canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3].copy()
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


# ─────────────────────────────────────────────────────────────
# DISPLAY MANAGER
# ─────────────────────────────────────────────────────────────

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display     = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA VSGP Exploration v6")
        self.grid_size   = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self._font       = pygame.font.SysFont("monospace", 13)
        self._hud_lines  = []

    def get_window_size(self):
        return list(self.window_size)

    def get_display_size(self):
        return [self.window_size[0] // self.grid_size[1],
                self.window_size[1] // self.grid_size[0]]

    def get_display_offset(self, grid_pos):
        ds = self.get_display_size()
        return [int(grid_pos[1] * ds[0]), int(grid_pos[0] * ds[1])]

    def add_sensor(self, s):
        self.sensor_list.append(s)

    def set_hud(self, lines):
        self._hud_lines = lines

    def render(self):
        self.display.fill((8, 10, 18))
        for s in self.sensor_list:
            s.render()
        self._draw_labels()
        self._draw_hud()
        pygame.display.flip()

    def _draw_labels(self):
        ds = self.get_display_size()
        for s in self.sensor_list:
            off = self.get_display_offset(s.display_pos)
            lbl = self._font.render(s.label, True, (180, 200, 255))
            self.display.blit(lbl, (off[0] + 4, off[1] + 4))
            pygame.draw.rect(self.display, (50, 55, 80),
                             (off[0], off[1], ds[0], ds[1]), 1)

    def _draw_hud(self):
        x = 10
        y = self.window_size[1] - 19 * len(self._hud_lines) - 4
        for line in self._hud_lines:
            surf = self._font.render(line, True, (0, 240, 100))
            self.display.blit(surf, (x, y))
            y += 18

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()


# ─────────────────────────────────────────────────────────────
# SENSOR MANAGER
# ─────────────────────────────────────────────────────────────

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform,
                 attached, sensor_options, display_pos, label=""):
        self.surface      = None
        self.world        = world
        self.display_man  = display_man
        self.display_pos  = display_pos
        self.label        = label
        self.timer        = CustomTimer()
        self.ticks        = 0
        self.time_proc    = 0.0
        self.lidar_points = None
        self._lidar_frame = 0
        self._colour_bar  = None
        self.sensor = self._init_sensor(sensor_type, transform, attached,
                                        sensor_options)
        self.display_man.add_sensor(self)

    def _init_sensor(self, sensor_type, transform, attached, opts):
        bp_lib = self.world.get_blueprint_library()
        ds     = self.display_man.get_display_size()

        if sensor_type == "RGBCamera":
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rgb)
            return actor

        elif sensor_type == "RainbowLiDAR":
            bp = bp_lib.find("sensor.lidar.ray_cast")
            bp.set_attribute("range",             LIDAR_RANGE)
            bp.set_attribute("channels",          LIDAR_CHANNELS)
            bp.set_attribute("points_per_second", LIDAR_PTS_PER_SEC)
            bp.set_attribute("rotation_frequency",LIDAR_ROT_FREQ)
            bp.set_attribute("dropoff_general_rate", "0.0")
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rainbow_lidar)
            self._colour_bar = _precompute_colour_bar(ds[1])
            return actor

        return None

    def _save_rgb(self, image):
        t0 = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        arr  = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr  = np.reshape(arr, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        ds   = self.display_man.get_display_size()
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.surface = pygame.transform.scale(surf, (ds[0], ds[1]))
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def _save_rainbow_lidar(self, image):
        t0 = self.timer.time()
        self._lidar_frame += 1
        pts = np.frombuffer(image.raw_data, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3].copy()
        self.lidar_points = xyz

        if self._lidar_frame % LIDAR_FRAME_SKIP != 0:
            self.time_proc += self.timer.time() - t0
            self.ticks += 1
            return

        ds          = self.display_man.get_display_size()
        lidar_range = 2.0 * float(LIDAR_RANGE)
        xy = xyz[:, :2].copy()
        xy *= min(ds) / lidar_range
        xy += (0.5 * ds[0], 0.5 * ds[1])
        xy  = xy.astype(np.int32)

        z_norm = np.clip(
            (xyz[:, 2] - RAINBOW_Z_MIN) / (RAINBOW_Z_MAX - RAINBOW_Z_MIN), 0, 1)
        hue    = ((1.0 - z_norm) * 0.75).astype(np.float32)
        rgb_u8 = _hsv_to_rgb_vectorised(hue)

        img   = np.zeros((ds[0], ds[1], 3), dtype=np.uint8)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < ds[0])
                 & (xy[:, 1] >= 0) & (xy[:, 1] < ds[1]))
        img[xy[valid, 0], xy[valid, 1]] = rgb_u8[valid]

        # Crosshair at vehicle centre
        cx, cy = ds[0] // 2, ds[1] // 2
        r = 7
        img[cx-1:cx+2, max(0, cy-r):min(ds[1], cy+r)] = 255
        img[max(0, cx-r):min(ds[0], cx+r), cy-1:cy+2] = 255

        # Colour-height legend bar
        bw = self._colour_bar.shape[0]
        img[ds[0]-bw:ds[0], :] = self._colour_bar

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


# ─────────────────────────────────────────────────────────────
# NAVIGATION CONTROLLER
# ─────────────────────────────────────────────────────────────

class NavigationController:
    """
    VSGP mapless navigation controller.
    Steps: filter LiDAR → SGP perception → subgoal generation
           → cost evaluation → select g* → PID steer + throttle.
    Exploration mode: no fixed goal.
    """

    def __init__(self):
        # EMA filters
        self.steer_ema    = 0.0
        self.throttle_ema = 0.0

        # History
        self.pos_history   = deque(maxlen=STUCK_WINDOW_FRAMES)
        self.reverse_count = 0
        self.spawn_z       = None
        self.emergency_count = 0

        # Navigation systems
        self.vsgp    = VSGPPerception()
        self.planner = SubgoalPlanner()

        # No-path recovery state
        self._nopath_turn_dir = 1
        self._nopath_frames   = 0
        self._recent_steers   = deque(maxlen=20)   # oscillation detection

        self.diag = {
            "state": "INIT", "speed": 0.0, "steer": 0.0,
            "throttle": 0.0, "subgoals": 0, "best_cost": float("inf")
        }

    # ── LiDAR filters ────────────────────────────────────────

    def _filter_for_vsgp(self, pts):
        """Full 360° ROI filter for VSGP."""
        if pts is None:
            return np.empty((0, 3))
        r_xy = np.hypot(pts[:, 0], pts[:, 1])
        mask = (r_xy > 0.5) & (r_xy < 22.0) & (pts[:, 2] > -2.0) & (pts[:, 2] < 3.5)
        filt = pts[mask]
        if len(filt) > GP_MAX_POINTS * 2:
            idx  = np.random.choice(len(filt), GP_MAX_POINTS * 2, replace=False)
            filt = filt[idx]
        return filt

    def _analyse_obstacle_sectors(self, pts):
        """Quick obstacle sector analysis for safety override."""
        if pts is None or len(pts) == 0:
            return None
        mask = (pts[:, 2] > OBSTACLE_Z_MIN) & (pts[:, 2] < OBSTACLE_Z_MAX)
        obs  = pts[mask]
        cap  = float(LIDAR_RANGE)
        if len(obs) == 0:
            return {"front": cap, "fl": cap, "fr": cap}
        ang  = np.arctan2(obs[:, 1], obs[:, 0])
        dist = np.hypot(obs[:, 0], obs[:, 1])
        sw   = math.pi / 8

        def sec(lo, hi):
            m = (ang >= lo) & (ang < hi)
            return float(np.min(dist[m])) if m.any() else cap

        return {"front": sec(-sw, sw), "fl": sec(sw, 3*sw), "fr": sec(-3*sw, -sw)}

    # ── Stuck detection ──────────────────────────────────────

    def _is_stuck(self):
        if len(self.pos_history) < STUCK_WINDOW_FRAMES:
            return False
        old, new = self.pos_history[0], self.pos_history[-1]
        moved = math.hypot(new[0]-old[0], new[1]-old[1])
        return moved < STUCK_MIN_TRAVEL_M and self.diag["speed"] < 0.4

    def _is_oscillating(self):
        """Detect rapid left-right oscillation in steering."""
        if len(self._recent_steers) < 10:
            return False
        s = list(self._recent_steers)
        sign_changes = sum(1 for i in range(1, len(s))
                           if s[i] * s[i-1] < -0.1)
        return sign_changes > 6

    # ── Main control loop ────────────────────────────────────

    def compute_control(self, lidar_pts, vehicle):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        spd = math.hypot(vel.x, vel.y)
        self.diag["speed"] = spd
        self.pos_history.append((loc.x, loc.y))
        self.planner.add_trajectory_point(loc)

        # ── Fall recovery ─────────────────────────────────────
        if self.spawn_z is not None and loc.z < self.spawn_z - 8.0:
            self.emergency_count = 40
        if self.emergency_count > 0:
            self.emergency_count -= 1
            self.diag["state"] = "FALL-RECOV"
            return carla.VehicleControl(throttle=0.6, reverse=_R_REV)

        # ── Stuck → reverse ──────────────────────────────────
        if self._is_stuck():
            self.reverse_count = REVERSE_FRAMES
            self.pos_history.clear()
            self._nopath_turn_dir *= -1
            self.diag["state"] = "STUCK→REV"
            return carla.VehicleControl(
                throttle=0.5, steer=float(self._nopath_turn_dir) * -0.6,
                brake=0.0, reverse=_R_REV)

        if self.reverse_count > 0:
            self.reverse_count -= 1
            self.diag["state"] = "REVERSING"
            return carla.VehicleControl(
                throttle=0.5, steer=float(self._nopath_turn_dir) * -0.6,
                brake=0.0, reverse=_R_REV)

        # ── VSGP navigation pipeline ─────────────────────────
        nav_pts = self._filter_for_vsgp(lidar_pts)
        free_segs, var_map, mean_map = self.vsgp.update(nav_pts)

        self.planner.generate_subgoals(free_segs, vehicle.get_transform(), self.vsgp)
        self.planner.evaluate_costs(vehicle.get_transform())
        best_sg = self.planner.select_best_subgoal()

        self.diag["subgoals"] = len(self.planner.subgoals)
        self.diag["best_cost"] = best_sg["cost"] if best_sg else float("inf")

        # ── Control generation ───────────────────────────────
        target_steer = 0.0
        throttle     = THROTTLE_CRUISE
        brake        = 0.0

        if best_sg:
            self.diag["state"]  = "SGP-NAV"
            self._nopath_frames = 0

            # Steer proportional to azimuth angle
            raw_steer    = np.clip(best_sg["alpha"] / (math.pi / 3), -1.0, 1.0)
            target_steer = raw_steer

            # Slow for close or uncertain subgoals
            if best_sg["distance"] < 5.0:
                throttle = THROTTLE_SLOW
            elif best_sg["variance"] > VARIANCE_THRESHOLD * 0.6:
                throttle = THROTTLE_SLOW

            # Hard obstacle safety override
            obs = self._analyse_obstacle_sectors(lidar_pts)
            if obs:
                if obs["front"] < FRONT_EMRG_DIST:
                    self.diag["state"] = "EMRG-STOP"
                    brake, throttle, target_steer = 0.7, 0.0, 0.0
                elif obs["front"] < FRONT_STOP_DIST:
                    self.diag["state"] = "SLOW"
                    throttle = 0.18
                    # Steer away from nearest obstacle
                    if obs["fl"] < obs["fr"]:
                        target_steer = -0.5
                    else:
                        target_steer =  0.5

        else:
            # ── No-path: in-place rotation to find open space ─
            self._nopath_frames += 1
            self.diag["state"]   = "SEARCHING"

            if self._nopath_frames > 35:
                # Switch rotation direction and try reverse
                self._nopath_turn_dir *= -1
                self._nopath_frames    = 0
                self.diag["state"]     = "REORIENT"
                return carla.VehicleControl(
                    throttle=0.35, steer=float(self._nopath_turn_dir) * 0.75,
                    brake=0.0, reverse=_R_REV)

            target_steer = float(self._nopath_turn_dir) * 0.80
            throttle     = 0.28

        # Anti-oscillation: if oscillating, commit harder to current direction
        self._recent_steers.append(target_steer)
        if self._is_oscillating():
            target_steer = np.sign(target_steer) * min(abs(target_steer) + 0.2, 1.0)

        # Low-speed boost
        if spd < 0.7 and brake == 0.0:
            throttle = max(throttle, 0.42)

        # EMA smoothing
        self.steer_ema    = (STEER_EMA_ALPHA    * target_steer
                             + (1 - STEER_EMA_ALPHA)    * self.steer_ema)
        self.throttle_ema = (THROTTLE_EMA_ALPHA * throttle
                             + (1 - THROTTLE_EMA_ALPHA) * self.throttle_ema)

        self.diag["steer"]    = self.steer_ema
        self.diag["throttle"] = self.throttle_ema

        # Estimate angular velocity (bicycle model: ω = v·tan(δ)/L)
        steer_rad  = self.steer_ema * MAX_STEER_RAD
        angular_v  = (spd * math.tan(np.clip(steer_rad, -1.5, 1.5))) / WHEELBASE
        self.planner.add_control_point(spd, self.steer_ema, angular_v)

        return carla.VehicleControl(
            throttle = float(np.clip(self.throttle_ema, 0, 1)),
            steer    = float(np.clip(self.steer_ema, -1, 1)),
            brake    = float(np.clip(brake, 0, 1)),
            reverse  = _R_FWD
        )


# ─────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────

def run_simulation(args, client):
    display_manager = None
    actors          = []
    nav             = NavigationController()
    visualizer      = None

    try:
        world             = client.get_world()
        original_settings = world.get_settings()

        # Sync mode
        if args.sync:
            tm = client.get_trafficmanager(8000)
            settings = world.get_settings()
            tm.set_synchronous_mode(True)
            settings.synchronous_mode    = True
            settings.fixed_delta_seconds = 0.05
            settings.max_substep_delta_time = 0.01
            settings.max_substeps           = 10
            world.apply_settings(settings)

        # Spawn vehicle
        bp_lib      = world.get_blueprint_library()
        vehicle_bp  = bp_lib.find("vehicle.tesla.cybertruck")
        spawn_pts   = world.get_map().get_spawn_points()
        random.shuffle(spawn_pts)

        vehicle = None
        for sp in spawn_pts:
            sp.location.z += 5.0
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"Spawned at ({sp.location.x:.1f}, "
                      f"{sp.location.y:.1f}, {sp.location.z:.1f})")
                break

        if vehicle is None:
            raise RuntimeError("No valid spawn point found")
        actors.append(vehicle)

        # Settle
        t0 = time.time()
        while time.time() - t0 < SPAWN_SETTLE_S:
            world.tick() if args.sync else world.wait_for_tick()
        nav.spawn_z = vehicle.get_location().z
        print(f"Settled Z={nav.spawn_z:.2f}m | GP={SKLEARN_AVAILABLE} | "
              f"Plots={MATPLOTLIB_AVAILABLE}")

        # ── Display Manager (2×3 grid) ───────────────────────
        display_manager = DisplayManager(
            grid_size=[2, 3],
            window_size=[args.width, args.height]
        )

        if MATPLOTLIB_AVAILABLE:
            visualizer = VisualizationManager(display_manager)

        # ── ROW 0: Front | LiDAR | Rear ──────────────────────
        front_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.5, z=1.8),
                            carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[0, 0], label="FRONT CAM"
        )
        actors.append(front_cam.sensor)

        rainbow_lidar = SensorManager(
            world, display_manager, "RainbowLiDAR",
            carla.Transform(carla.Location(z=2.5)),
            vehicle, {}, display_pos=[0, 1], label="LiDAR  +  SUBGOAL WINDOW"
        )
        actors.append(rainbow_lidar.sensor)

        rear_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.8, z=2.2),
                            carla.Rotation(yaw=180, pitch=-12)),
            vehicle, {}, display_pos=[0, 2], label="REAR CAM"
        )
        actors.append(rear_cam.sensor)

        # ── ROW 1: Matplotlib panels (sensors are placeholders) ─
        # We spawn minimal cameras just to show the panel label;
        # their surfaces are immediately overwritten by matplotlib.
        ph_traj = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.5, z=1.8),
                            carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[1, 0], label="TRAJECTORY MAP  (GP + Actual)"
        )
        actors.append(ph_traj.sensor)

        ph_cost = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.5, z=1.8),
                            carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[1, 1], label="COST FUNCTION  J(g)"
        )
        actors.append(ph_cost.sensor)

        ph_phase = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.8, z=2.2),
                            carla.Rotation(yaw=180, pitch=-12)),
            vehicle, {}, display_pos=[1, 2], label="v – ω  PHASE PORTRAIT"
        )
        actors.append(ph_phase.sensor)

        # ── Main loop ─────────────────────────────────────────
        clock         = pygame.time.Clock()
        running       = True
        frame         = 0
        manual_mode   = False
        manual_steer  = 0.0

        print("=" * 62)
        print(" VSGP EXPLORATION  |  ESC/Q: quit  R: reset  P: manual/auto")
        print(" Manual: W=fwd  S=brake/rev  A/D=steer")
        print("=" * 62)

        while running:
            clock.tick(20)
            frame += 1
            world.tick() if args.sync else world.wait_for_tick()

            # ── Event handling ────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_r:
                        nav.planner.trajectory_history.clear()
                        nav.planner.gp_trajectory_history.clear()
                        nav.planner.cost_history.clear()
                        nav.planner.control_history.clear()
                        nav.planner.visited_headings.clear()
                        nav.pos_history.clear()
                        nav.reverse_count = 0
                        nav.emergency_count = 0
                        nav._nopath_frames = 0
                        print("[R] State reset")
                    elif event.key == K_p:
                        manual_mode  = not manual_mode
                        manual_steer = 0.0
                        tag = "MANUAL" if manual_mode else "SGP-AUTO"
                        print(f"[P] Mode → {tag}")

            # ── Control ───────────────────────────────────────
            if manual_mode:
                keys = pygame.key.get_pressed()
                mt, mb, mr, ms_tgt = 0.0, 0.08, _R_FWD, 0.0
                if keys[K_w]:
                    mt, mb, mr = 0.65, 0.0, _R_FWD
                if keys[K_s]:
                    spd = math.hypot(
                        *[v for v in [vehicle.get_velocity().x,
                                      vehicle.get_velocity().y]])
                    if spd > 0.5:
                        mb, mt, mr = 0.8, 0.0, _R_FWD
                    else:
                        mt, mb, mr = 0.55, 0.0, _R_REV
                if keys[K_a]:
                    ms_tgt = -0.6
                if keys[K_d]:
                    ms_tgt =  0.6
                manual_steer = 0.45 * ms_tgt + 0.55 * manual_steer
                vehicle.apply_control(carla.VehicleControl(
                    throttle=float(np.clip(mt, 0, 1)),
                    steer   =float(np.clip(manual_steer, -1, 1)),
                    brake   =float(np.clip(mb, 0, 1)),
                    reverse =mr
                ))
                mode_tag = "MANUAL"
            else:
                lidar_pts = rainbow_lidar.lidar_points
                ctrl      = nav.compute_control(lidar_pts, vehicle)
                vehicle.apply_control(ctrl)
                mode_tag = "SGP-AUTO"

                # Update and render matplotlib panels
                if visualizer and MATPLOTLIB_AVAILABLE:
                    try:
                        visualizer.update_plots(nav.planner, vehicle)
                        visualizer.render(nav.planner, vehicle, display_manager)
                    except Exception as e:
                        pass  # Never crash on visualization error

            # ── HUD ───────────────────────────────────────────
            d = nav.diag
            bc_str = f"{d['best_cost']:.3f}" if d['best_cost'] < 1e6 else "∞"
            display_manager.set_hud([
                f"MODE  : {mode_tag:<12}  FRAME : {frame:5d}",
                f"STATE : {d['state']:<16}  GP    : {SKLEARN_AVAILABLE}",
                f"SPEED : {d['speed']*3.6:5.1f} km/h  STEER : {d['steer']:+.3f}",
                f"THR   : {d['throttle']:.2f}        SUBGOALS: {d['subgoals']:3d}",
                f"COST  : {bc_str:<10}  Vth   : {VARIANCE_THRESHOLD}",
                f"STUCK : {'YES' if nav._is_stuck() else 'no':<4}"
                f"  EMRG: {nav.emergency_count:3d}",
            ])

            try:
                display_manager.render()
            except Exception:
                pass

    finally:
        print("\nCleaning up ...")
        if display_manager:
            display_manager.destroy()
        client.apply_batch(
            [carla.command.DestroyActor(a) for a in actors if a is not None])
        world.apply_settings(original_settings)
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")
        pygame.quit()
        print("Done.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CARLA VSGP Mapless Exploration v6")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--sync",  action="store_true",  default=True)
    parser.add_argument("--async", dest="sync", action="store_false")
    parser.add_argument("--res",   default="1280x720", metavar="WxH")
    args = parser.parse_args()
    args.width, args.height = (int(x) for x in args.res.split("x"))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(12.0)
        run_simulation(args, client)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()