

from __future__ import annotations

import math
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import beta
import pygame
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# Optional scipy / sklearn
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found – inducing points fall back to random selection")

try:
    import carla
except ImportError:
    print("[FATAL] carla Python package not found. Put CARLA's egg on PYTHONPATH.")
    sys.exit(1)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA connection ──────────────────────────────────────────────────────
    carla_host: str            = "127.0.0.1"
    carla_port: int            = 2000
    carla_timeout: float       = 20.0
    synchronous: bool          = True
    fixed_delta_seconds: float = 0.05          # 20 Hz physics

    # ── Vehicle ───────────────────────────────────────────────────────────────
    vehicle_blueprint: str     = "vehicle.tesla.cybertruck"
    spawn_z_offset: float      = 5.0
    post_spawn_wait_s: float   = 3.0

    # ── Vehicle Geometry (New for Unified Planner) ───────────────────────────
    vehicle_length: float      = 4.5
    vehicle_width: float       = 2.0
    vehicle_height: float      = 1.8
    wheel_base: float          = 2.8
    track_width: float         = 1.7
    wheel_radius_real: float   = 0.35

    # ── Sensors ───────────────────────────────────────────────────────────────
    lidar_range: float         = 20.0
    lidar_points_per_sec: int  = 100_000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int        = 64
    lidar_upper_fov: float     = 15.0
    lidar_lower_fov: float     = -25.0
    camera_width: int          = 320
    camera_height: int         = 240
    camera_fov: int            = 90

    # ── Mecanum geometry (Legacy Kinematics) ─────────────────────────────────
    wheel_radius: float        = 0.15           # rw  [m]
    lx: float                  = 0.25           # half wheelbase [m]
    ly: float                  = 0.20           # half track width [m]
    max_wheel_omega: float     = 40.0           # rad/s

    # ── Planner ───────────────────────────────────────────────────────────────
    robot_width: float         = 0.8            # [m] (Legacy)
    safety_margin: float       = 0.4            # [m]
    max_slope_deg: float       = 20.0           # degrees
    subgoal_distance: float    = 4.0            # [m] preferred subgoal radius
    n_subgoals_max: int        = 8

    # ── Cost weights ──────────────────────────────────────────────────────────
    w_direction: float         = 1.2
    w_distance: float          = 0.6
    w_steepness: float         = 1.5
    w_collision: float         = 3.0
    w_oscillation: float       = 0.8
    w_flatness: float          = 0.5

    # ── Control ───────────────────────────────────────────────────────────────
    max_throttle: float        = 0.55
    max_steer: float           = 0.6
    max_brake: float           = 0.8
    ema_alpha: float           = 0.35           # smoothing factor
    max_steer_rate: float      = 0.12           # per step
    max_throttle_rate: float   = 0.08           # per step
    base_speed: float          = 4.0            # m/s target forward

    # ── Stability (roll / pitch) ──────────────────────────────────────────────
    max_safe_roll_deg: float   = 15.0           # roll threshold → throttle cut
    max_safe_pitch_deg: float  = 22.0           # pitch threshold → throttle cut
    stability_throttle_scale: float = 0.4       # multiplier when unstable
    roll_corrective_steer: float    = 0.25      # steer gain towards level

    # ── Stuck detection ───────────────────────────────────────────────────────
    stuck_window_s: float      = 4.0
    stuck_disp_thresh: float   = 0.6            # [m]
    recovery_duration_s: float = 2.5

    # ── VSGP ─────────────────────────────────────────────────────────────────
    vsgp_n_inducing: int       = 64
    vsgp_alpha: float          = 1.0
    vsgp_length_scale: float   = 0.3
    vsgp_noise_var: float      = 0.05
    vsgp_lr: float             = 0.02

    # ── Visualization ─────────────────────────────────────────────────────────
    viz_update_hz: float       = 5.0
    traj_history: int          = 400


CFG = Config()


# ╔════════════════════════════════════════════════════════════════════════════╗
# §1  MECANUM KINEMATIC MODEL
# ╚════════════════════════════════════════════════════════════════════════════╝

class MecanumKinematics:
    """Standard four-mecanum-wheel kinematic model."""

    def __init__(self, cfg: Config = CFG):
        rw = cfg.wheel_radius
        L  = cfg.lx + cfg.ly

        self.Jfwd = (rw / 4.0) * np.array([
            [ 1,  1,  1,  1],
            [-1,  1,  1, -1],
            [-1/L, 1/L, -1/L, 1/L],
        ], dtype=np.float64)

        self.Jinv = (1.0 / rw) * np.array([
            [ 1, -1, -L],
            [ 1,  1,  L],
            [ 1,  1, -L],
            [ 1, -1,  L],
        ], dtype=np.float64)

        self.cfg = cfg

    def forward(self, wheel_omegas: np.ndarray) -> Tuple[float, float, float]:
        twist = self.Jfwd @ wheel_omegas
        return float(twist[0]), float(twist[1]), float(twist[2])

    def inverse(self, vx: float, vy: float, omega: float) -> np.ndarray:
        twist = np.array([vx, vy, omega], dtype=np.float64)
        ws = self.Jinv @ twist
        return np.clip(ws, -self.cfg.max_wheel_omega, self.cfg.max_wheel_omega)

    @staticmethod
    def to_carla_control(
        vx: float, vy: float, omega: float, cfg: Config = CFG
    ) -> Tuple[float, float, float]:
        speed    = math.sqrt(vx ** 2 + vy ** 2)
        throttle = float(np.clip(speed / cfg.base_speed, 0.0, cfg.max_throttle))
        brake    = 0.0

        if vx < -0.1:
            throttle = float(np.clip(-vx / cfg.base_speed, 0.0, cfg.max_throttle))

        steer_omega   = omega / (cfg.base_speed + 1e-6)
        steer_lateral = -vy / (cfg.base_speed + 1e-6) * 0.4
        steer = float(np.clip(steer_omega + steer_lateral,
                               -cfg.max_steer, cfg.max_steer))

        return throttle, steer, brake


# ╔════════════════════════════════════════════════════════════════════════════╗
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚════════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha: float = 1.0,
                 length_scale: float = 0.3,
                 variance: float = 1.0):
        self.alpha        = alpha
        self.length_scale = length_scale
        self.variance     = variance

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        diff  = X[:, None, :] - Y[None, :, :]
        sq_d  = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.variance * np.ones(len(X))


class VSGP:
    def __init__(self, cfg: Config = CFG):
        self.cfg       = cfg
        self.kernel    = RationalQuadraticKernel(
            alpha        = cfg.vsgp_alpha,
            length_scale = cfg.vsgp_length_scale,
            variance     = 1.0,
        )
        self.noise_var = cfg.vsgp_noise_var
        self.n_ind     = cfg.vsgp_n_inducing
        self.lr        = cfg.vsgp_lr

        side  = int(math.sqrt(self.n_ind))
        a_v   = np.linspace(-np.pi / 2,  np.pi / 2,  side)
        b_v   = np.linspace(-np.pi / 4,  np.pi / 4,  side)
        A, B  = np.meshgrid(a_v, b_v)
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

        self.mu = np.zeros(len(self.Z))
        self.Su = np.eye(len(self.Z)) * 0.1

        self._Kuu         : Optional[np.ndarray] = None
        self._Kuu_inv     : Optional[np.ndarray] = None
        self._trained     : bool = False
        self._update_count: int  = 0

    def _select_inducing_points(self, X: np.ndarray) -> None:
        actual_n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3,
                        max_iter=50, random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            idx    = np.random.choice(len(X), actual_n, replace=False)
            self.Z = X[idx].copy()

        n = len(self.Z)
        self.mu = np.zeros(n)
        self.Su = np.eye(n) * 0.1

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return
        self._update_count += 1
        if self._update_count % 50 == 0:
            self._select_inducing_points(X)

        m         = len(self.Z)
        Kuu       = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6
        Kfu       = self.kernel(X, self.Z)
        Kuu_inv   = np.linalg.inv(Kuu)
        A         = Kfu @ Kuu_inv

        noise_inv = 1.0 / self.noise_var
        Lambda    = noise_inv * (A.T @ A) + Kuu_inv
        rhs       = noise_inv * (A.T @ y)

        Su_new    = np.linalg.inv(Lambda + np.eye(m) * 1e-6)
        mu_new    = Su_new @ rhs

        lr        = self.lr
        self.Su   = (1 - lr) * self.Su + lr * Su_new
        self.mu   = (1 - lr) * self.mu + lr * mu_new

        self._Kuu     = Kuu
        self._Kuu_inv = Kuu_inv
        self._trained = True

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained or self._Kuu_inv is None:
            n = len(Xs)
            return np.zeros(n), np.ones(n)

        Ksu            = self.kernel(Xs, self.Z)
        A              = Ksu @ self._Kuu_inv
        mean           = A @ self.mu

        Kss_diag       = self.kernel.diag(Xs)
        var_explained  = np.sum(A * (A @ self._Kuu), axis=1)
        var_variational = np.sum(A @ self.Su * A, axis=1)
        var = Kss_diag - var_explained + var_variational + self.noise_var
        var = np.clip(var, 1e-6, None)

        return mean, var


# ╔════════════════════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class PerceptionOutput:
    alpha_grid     : np.ndarray
    beta_grid      : np.ndarray
    occupancy_mean : np.ndarray
    occupancy_var  : np.ndarray
    slope_map      : np.ndarray
    traversability : np.ndarray
    raw_points     : np.ndarray
    free_mask      : Optional[np.ndarray] = None
    mean_surface   : Optional[np.ndarray] = None


class PerceptionModule:
    ALPHA_RES       = 60
    BETA_RES        = 30
    ROC             = 10.0
    VAR_FREE_WEIGHT = 1.0
    SLOPE_WEIGHT    = 1.2
    OCC_WEIGHT      = 1.0

    def __init__(self, cfg: Config = CFG):
        self.cfg          = cfg
        self.vsgp         = VSGP(cfg)
        self._last_output : Optional[PerceptionOutput] = None

        self._a_lin = np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES)
        self._b_lin = np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES)
        AG, BG      = np.meshgrid(self._a_lin, self._b_lin)
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG       = AG
        self._BG       = BG

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw  = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(raw) < 20:
            return self._last_output

        dist = np.linalg.norm(raw, axis=1)
        mask = (dist > 0.5) & (dist < self.cfg.lidar_range)
        pts  = raw[mask]
        if len(pts) < 10:
            return self._last_output

        r     = np.linalg.norm(pts, axis=1)
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta  = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1.0, 1.0))

        X = np.column_stack([alpha, beta])
        y = self.ROC - r

        if len(X) > 2000:
            idx  = np.random.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
            pts  = pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)

        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var  = var.reshape(self.BETA_RES, self.ALPHA_RES)

        d_alpha   = np.gradient(mean, axis=1)
        d_beta    = np.gradient(mean, axis=0)
        slope     = np.sqrt(d_alpha ** 2 + d_beta ** 2)
        slope     = slope / (np.max(slope) + 1e-6)

        mean_range = mean.max() - mean.min()
        occ_norm   = (mean - mean.min()) / (mean_range + 1e-6)
        var_range  = var.max() - var.min()
        var_norm   = (var  - var.min())  / (var_range  + 1e-6)

        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT   * occ_norm
            - self.SLOPE_WEIGHT * slope
            + self.VAR_FREE_WEIGHT * (1.0 - var_norm),
            0.0, 1.0
        )
        free_mask = traversability > 0.5

        out = PerceptionOutput(
            alpha_grid     = self._AG,
            beta_grid      = self._BG,
            occupancy_mean = mean,
            occupancy_var  = var,
            slope_map      = slope,
            traversability = traversability,
            raw_points     = pts,
            free_mask      = free_mask,
            mean_surface   = mean,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════════╗
# §4  SUBGOAL PLANNER  (Unified Terrain-Aware + Recovery-Aware)
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Subgoal:
    alpha     : float
    beta      : float
    distance  : float
    local_pos : np.ndarray
    slope_deg : float
    safe      : bool
    width_m   : float
    cost      : float = 0.0
    world_pos : Optional[np.ndarray] = None


class SubgoalPlanner:
    """
    Unified Variance-Based Subgoal Planner.
    Supports forward and reverse driving modes for VSGP-guided recovery.
    Implements Vehicle Footprint & Wheel Stability Checks.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg               = cfg
        self._alpha_history    : deque = deque(maxlen=8)
        self._exploration_bias : float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface used by the navigation loop
    # ─────────────────────────────────────────────────────────────────────────

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        mode: str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        """
        Thin wrapper so old call-sites still work.
        Returns (best_subgoal, all_subgoals).
        """
        best = self.plan_from_variance(perc, vehicle_pitch_rad, mode=mode)
        return best, []

    def plan_from_variance(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        mode: str = "forward",
    ) -> Optional[Subgoal]:
        """
        Unified Terrain-Aware + Recovery-Aware planner.
        mode = "forward"  → candidates in front half (|α| < π/2)
        mode = "reverse"  → candidates in rear half  (|α| > π/2)
        """
        if perc.occupancy_var is None:
            return None

        var       = perc.occupancy_var
        alpha     = perc.alpha_grid
        beta      = perc.beta_grid
        slope_map = perc.slope_map
        occ_mean  = perc.occupancy_mean

        # ✅ FULL TERRAIN AWARE (VERY IMPORTANT)
        var_row   = np.mean(var, axis=0)
        slope_row = np.mean(slope_map, axis=0)
        occ_row   = np.mean(occ_mean, axis=0)

        alpha_row = alpha[0]   # alpha same across rows
        beta_row  = beta[:, 0] # optional

        # ── Directional constraint ────────────────────────────────────────────
        if mode == "forward":
            dir_mask = (alpha_row > -np.pi / 2) & (alpha_row < np.pi / 2)
        else:
            dir_mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        # ── Build 1D terrain profile ──────────────────────────────────────────
        profile = []
        for i in range(len(alpha_row)):
            if dir_mask[i]:
                profile.append({
                    "alpha": alpha_row[i],
                    "var":   var_row[i],
                    "slope": slope_row[i],
                    "occ":   occ_row[i],
                    "idx":   i
                })

        if len(profile) < 3:
            return None

        # ── Segment Extraction ────────────────────────────────────────────────
        segments = []
        current = []
        threshold = np.percentile(var_row, 60)

        for p in profile:
            if p["var"] < np.percentile(var_row, 50) and p["slope"] < 0.4:
                current.append(p)
            else:
                if len(current) > 3:
                    segments.append(current)
                current = []
        if len(current) > 3:
            segments.append(current)

        best_subgoal : Optional[Subgoal] = None
        best_cost    : float = float("inf")

        # ── Evaluate Segments ─────────────────────────────────────────────────
        for seg in segments:
            α_start = seg[0]["alpha"]
            α_end   = seg[-1]["alpha"]

            width_alpha = α_end - α_start
            width_m     = self.cfg.subgoal_distance * abs(width_alpha)

            α_center = 0.5 * (α_start + α_end)

            # Orientation-aware shrink
            effective_width = width_m * np.cos(α_center)
            req_w = self.cfg.vehicle_width + self.cfg.safety_margin

            if effective_width < req_w:
                continue

            # ── Wheel Stability Check ─────────────────────────────────────────
            safe_ratio = 0
            samples = seg[::2] # Check every other point in segment

            for p in samples:
                lx = self.cfg.subgoal_distance * np.cos(p["alpha"])
                ly = self.cfg.subgoal_distance * np.sin(p["alpha"])

                if self._are_wheels_stable(perc, lx, ly, p["alpha"]):
                    safe_ratio += 1

            safe_ratio /= max(len(samples), 1)

            if safe_ratio < 0.6:
                continue

            # ── Subgoal Generation ────────────────────────────────────────────
            a = α_center
            b = b = float(np.mean([p["slope"] for p in seg])) * 0.2
            d = self.cfg.subgoal_distance

            lx = d * np.cos(a)
            ly = d * np.sin(a)
            lz = 0.0

            # ── Trajectory Rollout Cost ───────────────────────────────────────
            C_rollout = self._rollout_cost(perc, a, d)
            if C_rollout > 900: # Obstacle in path
                continue

            # ── Final Cost Function ───────────────────────────────────────────
            C_dir   = abs(a) / (np.pi / 2)
            C_clear = 1.0 / (width_m - req_w + 1e-3)

            # Add pitch penalty
            pitch_deg = abs(math.degrees(vehicle_pitch_rad))
            C_pitch   = pitch_deg / (self.cfg.max_slope_deg + 1e-6)

            # ✅ NEW: segment slope cost
            C_slope = np.mean([p["slope"] for p in seg])
         # ✅ NEW: uphill penalty (prevents climbing)
            C_uphill = np.mean([p["slope"] for p in seg]) * max(0.0, np.cos(a))

            J = (
                1.2 * C_dir +
                2.0 * C_rollout +
                2.0 * C_slope +
                1.5 * C_clear +
                1.0 * (1.0 - safe_ratio) +
                1.5 * C_pitch +
                1.5 * C_uphill 
            )
           

          
            # Apply exploration bias (set during stuck recovery)
            if self._exploration_bias > 0:
                J *= (1.0 - self._exploration_bias * 0.5)

            sg = Subgoal(
                alpha     = a,
                beta      = b,
                distance  = d,
                local_pos = np.array([lx, ly, lz]),
                slope_deg = 0.0, # Derived from rollout/slope_map if needed
                safe      = True,
                width_m   = width_m,
                cost      = J,
            )

            if J < best_cost:
                best_cost    = J
                best_subgoal = sg

        # ── Fallback: highest-variance point ─────────────────────────────────
        if best_subgoal is None:
            # Restrict fallback to the correct half
            valid_profile = [p for p in profile]
            if len(valid_profile) == 0:
                return None

            best_p = max(valid_profile, key=lambda k: k["var"])
            a  = float(best_p["alpha"])
            b  = 0.0
            d  = self.cfg.subgoal_distance
            lx = d * np.cos(a)
            ly = d * np.sin(a)
            lz = 0.0

            best_subgoal = Subgoal(
                alpha     = a,
                beta      = b,
                distance  = d,
                local_pos = np.array([lx, ly, lz]),
                slope_deg = 0.0,
                safe      = False,
                width_m   = 0.5,
                cost      = 999.0,
            )

        # Update oscillation history
        self._alpha_history.append(best_subgoal.alpha)
        return best_subgoal

    def set_exploration_bias(self, v: float) -> None:
        self._exploration_bias = float(np.clip(v, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # NEW HELPER FUNCTIONS (Unified Algorithm)
    # ─────────────────────────────────────────────────────────────────────────

    def _rollout_cost(self, perc: PerceptionOutput, a: float, d: float) -> float:
        """Evaluates cost along the trajectory to the subgoal."""
        steps = 10
        total_cost = 0.0

        for i in range(1, steps + 1):
            frac = i / steps
            x = frac * d * np.cos(a)
            y = frac * d * np.sin(a)

            if self._is_occupied(perc, x, y):
                return 999.0

            alpha = np.arctan2(y, x)
            # Map alpha to grid index
            # alpha_grid is (Beta, Alpha). We look at row 0 (or mid) for alpha values
            ai = int(np.argmin(np.abs(perc.alpha_grid[0] - alpha)))
            bi = perc.beta_grid.shape[0] // 2

            # Bounds check
            if bi < 0 or bi >= perc.slope_map.shape[0] or \
               ai < 0 or ai >= perc.slope_map.shape[1]:
                continue

            slope = perc.slope_map[bi, ai]
            var   = perc.occupancy_var[bi, ai]

            total_cost += slope * 2.0 + var * 1.5

        return total_cost / steps

    def _are_wheels_stable(self, perc: PerceptionOutput, cx: float, cy: float, yaw: float) -> bool:
        """Checks if vehicle wheels land on stable ground at target pose."""
        wb = self.cfg.wheel_base / 2.0
        tw = self.cfg.track_width / 2.0

        # Wheel positions relative to center (x, y)
        wheels = np.array([
            [ wb,  tw],
            [ wb, -tw],
            [-wb,  tw],
            [-wb, -tw],
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)],
        ])

        # Rotate and translate to target position
        wheels = wheels @ R.T
        wheels[:, 0] += cx
        wheels[:, 1] += cy

        stable_count = 0
        for wx, wy in wheels:
            if not self._is_occupied(perc, wx, wy):
                stable_count += 1

        # Require at least 3 wheels on stable ground
        return stable_count >= 3

    def _is_occupied(self, perc: PerceptionOutput, x: float, y: float) -> bool:
        """Checks occupancy at a specific local coordinate."""
        r = np.sqrt(x**2 + y**2)
        if r < 1e-3:
            return True # Center is always occupied (vehicle body)

        alpha = np.arctan2(y, x)
        # Map to grid
        # alpha_grid is (Beta, Alpha). We assume flat ground check (beta ~ 0)
        ai = int(np.argmin(np.abs(perc.alpha_grid[0] - alpha)))
        bi = perc.beta_grid.shape[0] // 2

        # Bounds check
        if bi < 0 or bi >= perc.occupancy_mean.shape[0] or \
           ai < 0 or ai >= perc.occupancy_mean.shape[1]:
            return True # Out of bounds considered occupied

        occ = perc.occupancy_mean[bi, ai]
        return occ < 0.3


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5  CONTROLLER MODULE  (with Roll / Pitch Stabilisation)
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle : float = 0.0
    steer    : float = 0.0
    brake    : float = 0.0
    reverse  : bool  = False


class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg     = cfg
        self.mec     = MecanumKinematics(cfg)
        self._state  = ControlState()
        self._vx_ema = 0.0
        self._vy_ema = 0.0
        self._om_ema = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Main compute paths
    # ─────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        subgoal           : Optional[Subgoal],
        vehicle_speed_ms  : float,
        terrain_slope_deg : float = 0.0,
        roll_deg          : float = 0.0,
        pitch_deg         : float = 0.0,
    ) -> ControlState:
        if subgoal is None:
            return self._apply_smooth(0.0, 0.0, 0.0, roll_deg=roll_deg,
                                      pitch_deg=pitch_deg)

        alpha = subgoal.alpha
        beta  = subgoal.beta
        slope = subgoal.slope_deg

        speed_factor = max(
            0.3,
            1.0
            - slope  / self.cfg.max_slope_deg * 0.5
            - abs(alpha) / (math.pi / 2)      * 0.3,
        )
        vx_desired    = self.cfg.base_speed * speed_factor
        vy_desired    = -math.sin(alpha) * self.cfg.base_speed * 0.2
        omega_desired = alpha * 1.8

        ws           = self.mec.inverse(vx_desired, vy_desired, omega_desired)
        vx_a, vy_a, om_a = self.mec.forward(ws)

        return self._apply_smooth(vx_a, vy_a, om_a,
                                   roll_deg=roll_deg, pitch_deg=pitch_deg)

    def compute_manual(
        self,
        vx: float, vy: float, omega: float,
    ) -> ControlState:
        ws           = self.mec.inverse(vx, vy, omega)
        vx_a, vy_a, om_a = self.mec.forward(ws)
        return self._apply_smooth(vx_a, vy_a, om_a)

    def compute_recovery(self, phase: int) -> ControlState:
        """Fallback hard-coded recovery (used only when VSGP recovery fails)."""
        if phase == 0:
            return ControlState(throttle=0.3, steer=0.0,  brake=0.0, reverse=True)
        elif phase == 1:
            return ControlState(throttle=0.3, steer=0.4,  brake=0.0, reverse=True)
        else:
            return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Stabilisation logic
    # ─────────────────────────────────────────────────────────────────────────

    def _stability_modifiers(
        self, roll_deg: float, pitch_deg: float
    ) -> Tuple[float, float]:
        """
        Returns (throttle_scale, corrective_steer).
        - throttle_scale in [0, 1]
        - corrective_steer opposes the roll direction
        """
        throttle_scale    = 1.0
        corrective_steer  = 0.0

        # Roll correction
        abs_roll = abs(roll_deg)
        if abs_roll > self.cfg.max_safe_roll_deg:
            excess         = abs_roll - self.cfg.max_safe_roll_deg
            roll_factor    = 1.0 - min(excess / 15.0, 1.0)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale +
                                 (1.0 - self.cfg.stability_throttle_scale) * roll_factor)
            # Steer towards the downhill side to level the vehicle
            corrective_steer = -math.copysign(
                min(self.cfg.roll_corrective_steer * (excess / 10.0),
                    self.cfg.roll_corrective_steer),
                roll_deg,
            )

        # Pitch correction (steep uphill → limit throttle to avoid flip)
        abs_pitch = abs(pitch_deg)
        if abs_pitch > self.cfg.max_safe_pitch_deg:
            excess_p       = abs_pitch - self.cfg.max_safe_pitch_deg
            pitch_factor   = 1.0 - min(excess_p / 10.0, 0.7)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale * pitch_factor
                                 + 0.15)

        return float(throttle_scale), float(corrective_steer)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal smooth application
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_smooth(
        self,
        vx: float,
        vy: float,
        omega: float,
        roll_deg: float  = 0.0,
        pitch_deg: float = 0.0,
    ) -> ControlState:
        a            = self.cfg.ema_alpha
        self._vx_ema = a * vx    + (1 - a) * self._vx_ema
        self._vy_ema = a * vy    + (1 - a) * self._vy_ema
        self._om_ema = a * omega + (1 - a) * self._om_ema

        throttle, steer, brake = MecanumKinematics.to_carla_control(
            self._vx_ema, self._vy_ema, self._om_ema, self.cfg
        )

        # Rate limiting
        max_dr   = self.cfg.max_steer_rate
        max_dt   = self.cfg.max_throttle_rate
        steer    = float(np.clip(steer,
                                  self._state.steer    - max_dr,
                                  self._state.steer    + max_dr))
        throttle = float(np.clip(throttle,
                                  self._state.throttle - max_dt,
                                  self._state.throttle + max_dt))

        # ── Stability modifiers ───────────────────────────────────────────────
        thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
        throttle *= thr_scale
        steer     = float(np.clip(steer + cor_steer,
                                   -self.cfg.max_steer, self.cfg.max_steer))

        self._state = ControlState(throttle=throttle, steer=steer, brake=brake)
        return self._state

    def reset_ema(self) -> None:
        self._vx_ema = self._vy_ema = self._om_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6  STUCK DETECTION & RECOVERY
# ╚════════════════════════════════════════════════════════════════════════════╝

class StuckDetector:
    IDLE       = "IDLE"
    RECOVERING = "RECOVERING"

    def __init__(self, cfg: Config = CFG):
        self.cfg             = cfg
        self._pos_history    : deque = deque(maxlen=200)
        self._steer_history  : deque = deque(maxlen=40)
        self._state          = self.IDLE
        self._recovery_start : float = 0.0
        self._recovery_phase : int   = 0

    def update(
        self, pos: np.ndarray, steer: float, t: float
    ) -> Tuple[bool, int]:
        self._pos_history.append((t, pos.copy()))
        self._steer_history.append(steer)

        if self._state == self.RECOVERING:
            elapsed   = t - self._recovery_start
            phase_dur = self.cfg.recovery_duration_s / 3.0
            if elapsed > self.cfg.recovery_duration_s:
                self._state = self.IDLE
                return False, 0
            phase = min(int(elapsed / phase_dur), 2)
            return True, phase

        # Displacement check
        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    self._trigger_recovery(t)
                    return True, 0

        # Steer oscillation check
        if len(self._steer_history) == self._steer_history.maxlen:
            arr   = np.array(self._steer_history)
            signs = np.sign(arr)
            flips = int(np.sum(np.diff(signs) != 0))
            if flips > len(signs) * 0.7:
                self._trigger_recovery(t)
                return True, 0

        return False, 0

    def _trigger_recovery(self, t: float) -> None:
        if self._state == self.IDLE:
            print("[STUCK] Recovery triggered!")
            self._state          = self.RECOVERING
            self._recovery_start = t
            self._recovery_phase = 0
            self._pos_history.clear()

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING


# ╔════════════════════════════════════════════════════════════════════════════╗
# §7  VISUALIZATION MODULE
# ╚════════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        plt.ion()
        self.fig = plt.figure("VSGP Navigator", figsize=(16, 9))
        self.fig.patch.set_facecolor("#111")
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        self.ax_front = self.fig.add_subplot(gs[0, 0])
        self.ax_lidar = self.fig.add_subplot(gs[0, 1])
        self.ax_rear  = self.fig.add_subplot(gs[0, 2])
        self.ax_cost  = self.fig.add_subplot(gs[1, 0])
        self.ax_traj  = self.fig.add_subplot(gs[1, 1])
        self.ax_vel   = self.fig.add_subplot(gs[1, 2])

        for ax in self.fig.get_axes():
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#ddd")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")

        maxlen = cfg.traj_history
        self._cost_total : deque = deque(maxlen=maxlen)
        self._cost_dir   : deque = deque(maxlen=maxlen)
        self._cost_dist  : deque = deque(maxlen=maxlen)
        self._cost_steep : deque = deque(maxlen=maxlen)
        self._vel_lin    : deque = deque(maxlen=maxlen)
        self._vel_ang    : deque = deque(maxlen=maxlen)
        self._traj_x     : deque = deque(maxlen=maxlen)
        self._traj_y     : deque = deque(maxlen=maxlen)
        self._subgoal_x  : deque = deque(maxlen=20)
        self._subgoal_y  : deque = deque(maxlen=20)

        self._front_img   : Optional[np.ndarray] = None
        self._rear_img    : Optional[np.ndarray] = None
        self._lidar_pts   : Optional[np.ndarray] = None
        self._mode_text   : str   = "AUTO"
        self._last_update : float = 0.0
        self._interval    : float = 1.0 / cfg.viz_update_hz

    def push_data(
        self,
        *,
        subgoal     : Optional[Subgoal],
        vehicle_pos : np.ndarray,
        speed_ms    : float,
        steer       : float,
        perc        : Optional[PerceptionOutput],
        mode        : str = "AUTO",
    ) -> None:
        self._traj_x.append(float(vehicle_pos[0]))
        self._traj_y.append(float(vehicle_pos[1]))

        if subgoal is not None:
            self._cost_total.append(subgoal.cost)
            self._cost_dir.append(
                abs(subgoal.alpha) / (math.pi / 2) * CFG.w_direction)
            self._cost_dist.append(
                abs(subgoal.distance - CFG.subgoal_distance)
                / CFG.subgoal_distance * CFG.w_distance)
            self._cost_steep.append(
                subgoal.slope_deg / CFG.max_slope_deg * CFG.w_steepness)
            self._subgoal_x.append(vehicle_pos[0] + subgoal.local_pos[0])
            self._subgoal_y.append(vehicle_pos[1] + subgoal.local_pos[1])
        else:
            self._cost_total.append(0)
            self._cost_dir.append(0)
            self._cost_dist.append(0)
            self._cost_steep.append(0)

        self._vel_lin.append(speed_ms)
        self._vel_ang.append(steer * CFG.base_speed)

        if perc is not None:
            self._lidar_pts = perc.raw_points

        self._mode_text = mode

    def set_front_image(self, arr: np.ndarray) -> None:
        self._front_img = arr

    def set_rear_image(self, arr: np.ndarray) -> None:
        self._rear_img = arr

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_update) < self._interval:
            return
        self._last_update = now
        try:
            self._draw_camera(self.ax_front, self._front_img, "Front Camera")
            self._draw_camera(self.ax_rear,  self._rear_img,  "Rear Camera")
            self._draw_lidar()
            self._draw_cost()
            self._draw_trajectory()
            self._draw_velocity()
            self.fig.suptitle(
                f"VSGP Mapless Navigator  |  Mode: {self._mode_text}",
                color="#eee", fontsize=11, y=0.99,
            )
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _draw_camera(
        self, ax: plt.Axes, img: Optional[np.ndarray], title: str
    ) -> None:
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if img is not None:
            ax.imshow(img)

    def _draw_lidar(self) -> None:
        ax = self.ax_lidar
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("LiDAR (top view)", fontsize=8)
        if self._lidar_pts is not None and len(self._lidar_pts) > 0:
            pts   = self._lidar_pts
            z     = pts[:, 2]
            z_n   = (z - z.min()) / (z.max() - z.min() + 1e-9)
            ax.scatter(pts[:, 1], pts[:, 0], c=z_n,
                       cmap="plasma", s=0.5, alpha=0.6)
            ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
            ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)

    def _draw_cost(self) -> None:
        ax = self.ax_cost
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Cost Function", fontsize=8)
        n = len(self._cost_total)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._cost_total), c="#ff6b6b", lw=1.2, label="Total")
        ax.plot(xs, list(self._cost_dir),   c="#ffa36c", lw=0.8, label="Direction")
        ax.plot(xs, list(self._cost_dist),  c="#c3f584", lw=0.8, label="Distance")
        ax.plot(xs, list(self._cost_steep), c="#74b9ff", lw=0.8, label="Steepness")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Cost",  fontsize=7)

    def _draw_trajectory(self) -> None:
        ax = self.ax_traj
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Trajectory", fontsize=8)
        tx, ty = list(self._traj_x), list(self._traj_y)
        if len(tx) > 1:
            ax.plot(ty, tx, c="#00cec9", lw=1.2, label="Actual")
        if len(tx) > 0:
            ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)
        sx, sy = list(self._subgoal_x), list(self._subgoal_y)
        if sx:
            ax.scatter(sy, sx, c="#e17055", s=12, marker="x",
                       zorder=4, label="Subgoals")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    def _draw_velocity(self) -> None:
        ax = self.ax_vel
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Velocities", fontsize=8)
        n = len(self._vel_lin)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._vel_lin), c="#55efc4", lw=1.2, label="Linear (m/s)")
        ax.plot(xs, list(self._vel_ang), c="#a29bfe", lw=1.0, label="ω·v (m/s)")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("m/s",  fontsize=7)

    def close(self) -> None:
        plt.close(self.fig)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §8  CARLA SENSOR WRAPPERS
# ╚════════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self.vehicle = vehicle
        self.world   = world
        self._actors : list = []

        self._lidar_data  = None
        self._front_image : Optional[np.ndarray] = None
        self._rear_image  : Optional[np.ndarray] = None

        bp_lib = world.get_blueprint_library()

        # ── LiDAR ─────────────────────────────────────────────────────────────
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",              str(cfg.lidar_range))
        lidar_bp.set_attribute("points_per_second",  str(cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("channels",           str(cfg.lidar_channels))
        lidar_bp.set_attribute("upper_fov",          str(cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov",          str(cfg.lidar_lower_fov))
        lidar = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.5, z=2.0)),
            attach_to=vehicle,
        )
        lidar.listen(self._on_lidar)
        self._actors.append(lidar)

        # ── Cameras ───────────────────────────────────────────────────────────
        self._actors.append(self._spawn_camera(
            bp_lib, vehicle,
            carla.Transform(carla.Location(x=1.5, z=1.8)),
            "front",
        ))
        self._actors.append(self._spawn_camera(
            bp_lib, vehicle,
            carla.Transform(carla.Location(x=-1.5, z=1.8),
                            carla.Rotation(yaw=180)),
            "rear",
        ))

    def _spawn_camera(self, bp_lib, vehicle, transform, tag: str):
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov",          str(self.cfg.camera_fov))
        cam = self.world.spawn_actor(cam_bp, transform, attach_to=vehicle)
        if tag == "front":
            cam.listen(self._on_front_image)
        else:
            cam.listen(self._on_rear_image)
        return cam

    def _on_lidar(self, data) -> None:
        self._lidar_data = data

    def _on_front_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._front_image = arr[:, :, ::-1].copy()

    def _on_rear_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._rear_image = arr[:, :, ::-1].copy()

    @property
    def lidar_data(self):
        return self._lidar_data

    @property
    def front_image(self) -> Optional[np.ndarray]:
        return self._front_image

    @property
    def rear_image(self) -> Optional[np.ndarray]:
        return self._rear_image

    def destroy(self) -> None:
        for a in self._actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# §9  MANUAL KEYBOARD CONTROLLER  (Holonomic)
# ╚════════════════════════════════════════════════════════════════════════════╝

class ManualHolonomicController:
    VX_SPEED = 3.0
    VY_SPEED = 2.0
    OM_SPEED = 1.2

    def __init__(self, cfg: Config = CFG):
        self.ctrl = Controller(cfg)

    def tick(self, keys) -> ControlState:
        vx = vy = om = 0.0
        if keys[pygame.K_w]: vx =  self.VX_SPEED
        if keys[pygame.K_s]: vx = -self.VX_SPEED
        if keys[pygame.K_a]: vy =  self.VY_SPEED
        if keys[pygame.K_d]: vy = -self.VY_SPEED
        if keys[pygame.K_q]: om =  self.OM_SPEED
        if keys[pygame.K_e]: om = -self.OM_SPEED
        return self.ctrl.compute_manual(vx, vy, om)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §10  VEHICLE SPAWNER
# ╚════════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle = None
    for sp in spawn_points:
        sp.location.z += cfg.spawn_z_offset
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("No valid spawn point found")

    print(f"[SPAWN] {cfg.vehicle_blueprint} at {vehicle.get_location()}")

    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    print(f"[SPAWN] Settled Z = {vehicle.get_location().z:.2f} m")
    return vehicle, vehicle.get_location().z


# ╔════════════════════════════════════════════════════════════════════════════╗
# §11  MAIN NAVIGATION LOOP
# ╚════════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._mode    = "AUTO"

        self._client  : Optional[carla.Client] = None
        self._world   = None
        self._vehicle = None
        self._sensors : Optional[SensorManager] = None

        self._perception  = PerceptionModule(cfg)
        self._planner     = SubgoalPlanner(cfg)
        self._controller  = Controller(cfg)
        self._manual_ctrl = ManualHolonomicController(cfg)
        self._stuck       = StuckDetector(cfg)
        self._viz         = VisualizationModule(cfg)

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit")
        self._screen = pygame.display.set_mode((220, 88))
        self._font   = pygame.font.SysFont("monospace", 13)

        self._step   = 0
        self._t0     = 0.0

    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port} …")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world  = self._client.get_world()
        if self.cfg.synchronous:
            settings = self._world.get_settings()
            settings.synchronous_mode    = True
            settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self._world.apply_settings(settings)
            print("[CARLA] Synchronous mode ON")

    def setup(self) -> None:
        self._vehicle, self._spawn_z = spawn_vehicle(self._world, self.cfg)
        self._sensors = SensorManager(self._world, self._vehicle, self.cfg)
        self._t0      = time.time()
        print("[SETUP] All sensors attached. Starting navigation loop.")

    def run(self) -> None:
        self._running = True
        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        print("[CLEANUP] Stopping actors …")
        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.apply_control(carla.VehicleControl(brake=1.0))
        if self._sensors:
            self._sensors.destroy()
        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.destroy()
        if self._world and self.cfg.synchronous:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)
        self._viz.close()
        pygame.quit()
        print("[CLEANUP] Done.")

    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self.cfg.synchronous:
            self._world.tick()

        t_now      = time.time() - self._t0
        self._step += 1

        # ── Event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return
                if event.key == pygame.K_TAB:
                    self._mode = "MANUAL" if self._mode == "AUTO" else "AUTO"
                    self._controller.reset_ema()
                    print(f"[MODE] Switched to {self._mode}")

        keys = pygame.key.get_pressed()

        # ── Vehicle state ─────────────────────────────────────────────────────
        loc       = self._vehicle.get_location()
        vel       = self._vehicle.get_velocity()
        speed     = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos       = np.array([loc.x, loc.y, loc.z])

        transform    = self._vehicle.get_transform()
        pitch_rad    = math.radians(transform.rotation.pitch)
        roll_rad     = math.radians(transform.rotation.roll)
        pitch_deg    = math.degrees(pitch_rad)
        roll_deg     = math.degrees(roll_rad)

        # ── Perception ────────────────────────────────────────────────────────
        perc : Optional[PerceptionOutput] = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        best_goal  : Optional[Subgoal] = None
        ctrl_state : ControlState

        # ── Control decision ──────────────────────────────────────────────────
        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)

        else:  # AUTO
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                # ── VSGP-guided recovery ──────────────────────────────────────
                self._planner.set_exploration_bias(0.8)

                if perc is not None:
                    best_goal = self._planner.plan_from_variance(
                        perc, pitch_rad, mode="reverse"
                    )

                if best_goal is not None:
                    # Drive backward towards the chosen free direction
                    best_goal.local_pos[0] = -abs(best_goal.local_pos[0])

                    # Compute steering from the subgoal's lateral offset
                    raw_steer = float(np.clip(
                        -best_goal.alpha * 1.5,
                        -self.cfg.max_steer, self.cfg.max_steer
                    ))
                    throttle  = min(0.35, self.cfg.max_throttle)

                    # Apply stability modifiers even during recovery
                    thr_scale, cor_steer = self._controller._stability_modifiers(
                        roll_deg, pitch_deg
                    )
                    ctrl_state = ControlState(
                        throttle = throttle * thr_scale,
                        steer    = float(np.clip(raw_steer + cor_steer,
                                                 -self.cfg.max_steer,
                                                  self.cfg.max_steer)),
                        brake    = 0.0,
                        reverse  = True,
                    )
                    print(f"[RECOVERY] VSGP reverse  α={best_goal.alpha:.2f}  "
                          f"steer={ctrl_state.steer:.2f}")
                else:
                    # Fallback hard-coded recovery
                    ctrl_state = self._controller.compute_recovery(rec_phase)

            else:
                self._planner.set_exploration_bias(0.0)
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc, pitch_rad,
                                                       mode="forward")

                slope      = best_goal.slope_deg if best_goal else 0.0
                ctrl_state = self._controller.compute(
                    best_goal, speed, slope,
                    roll_deg=roll_deg, pitch_deg=pitch_deg,
                )

            # Update steer history for oscillation detection
            if self._stuck._steer_history:
                self._stuck._steer_history.pop()
            self._stuck._steer_history.append(ctrl_state.steer)

        # ── Apply control ─────────────────────────────────────────────────────
        self._vehicle.apply_control(carla.VehicleControl(
            throttle = float(ctrl_state.throttle),
            steer    = float(ctrl_state.steer),
            brake    = float(ctrl_state.brake),
            reverse  = ctrl_state.reverse,
        ))

        # ── HUD ───────────────────────────────────────────────────────────────
        self._screen.fill((20, 20, 40))
        lines = [
            f"Mode: {self._mode}",
            f"Speed: {speed:.1f} m/s",
            f"Thr:{ctrl_state.throttle:.2f}  Str:{ctrl_state.steer:.2f}",
            f"Roll:{roll_deg:+.1f}°  Pitch:{pitch_deg:+.1f}°",
            f"Step: {self._step}   t={t_now:.1f}s",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (200, 220, 255))
            self._screen.blit(surf, (5, 4 + i * 16))
        pygame.display.flip()

        # ── Visualization push ────────────────────────────────────────────────
        self._viz.push_data(
            subgoal     = best_goal if self._mode == "AUTO" else None,
            vehicle_pos = pos,
            speed_ms    = speed,
            steer       = ctrl_state.steer,
            perc        = perc,
            mode        = self._mode,
        )
        if self._sensors.front_image is not None:
            self._viz.set_front_image(self._sensors.front_image)
        if self._sensors.rear_image is not None:
            self._viz.set_rear_image(self._sensors.rear_image)
        self._viz.render()

        if not self.cfg.synchronous:
            time.sleep(self.cfg.fixed_delta_seconds)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §12  ENTRY POINT
# ╚════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    nav = NavigationSystem(CFG)
    try:
        nav.connect()
        nav.setup()
        nav.run()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()