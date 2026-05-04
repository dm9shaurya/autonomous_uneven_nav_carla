from __future__ import annotations

import math
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
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
# §0  CONFIGURATION (Code 1 Vehicle + Paper Parameters)
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA connection ──────────────────────────────────────────────────────
    carla_host: str            = "127.0.0.1"
    carla_port: int            = 2000
    carla_timeout: float       = 20.0
    synchronous: bool          = True
    fixed_delta_seconds: float = 0.05          # 20 Hz physics

    # ── Vehicle (FROM CODE 1) ─────────────────────────────────────────────────
    vehicle_blueprint: str     = "vehicle.tesla.cybertruck"
    spawn_z_offset: float      = 5.0
    post_spawn_wait_s: float   = 3.0

    # ── Vehicle Geometry ──────────────────────────────────────────────────────
    vehicle_length: float      = 4.5
    vehicle_width: float       = 2.0
    vehicle_height: float      = 1.8
    wheel_base: float          = 2.8
    track_width: float         = 1.7
    wheel_radius_real: float   = 0.35

    # ── Sensors (Paper: 16-beam LiDAR, but we use 64 for better resolution) ──
    lidar_range: float         = 20.0
    lidar_points_per_sec: int  = 100_000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int        = 64
    lidar_upper_fov: float     = 15.0
    lidar_lower_fov: float     = -25.0
    camera_width: int          = 320
    camera_height: int         = 240
    camera_fov: int            = 90

    # ── Planner (Paper Parameters from Section VI.A) ─────────────────────────
    robot_width: float         = 0.8
    safety_margin: float       = 0.4
    max_slope_deg: float       = 20.0
    subgoal_distance: float    = 4.0
    n_subgoals_max: int        = 8

    # ── Variance Threshold (Paper Section VI.A: Vth = 0.7) ───────────────────
    variance_threshold: float  = 0.7
    occupancy_surface_radius: float = 7.0  # Paper: roc = 7 meters

    # ── Cost Weights (Paper Eq. 6) ───────────────────────────────────────────
    k_direction: float         = 0.2  # kdir from paper
    k_distance: float          = 0.3  # kdst from paper
    k_steepness: float         = 0.5  # kstp from paper

    # ── Control ───────────────────────────────────────────────────────────────
    max_throttle: float        = 0.55
    max_steer: float           = 0.6
    max_brake: float           = 0.8
    ema_alpha: float           = 0.35
    max_steer_rate: float      = 0.12
    max_throttle_rate: float   = 0.08
    base_speed: float          = 4.0

    # ── Stability (Paper: roll < 0.524 rad, pitch < 0.785 rad) ───────────────
    max_safe_roll_deg: float   = 30.0   # 0.524 rad
    max_safe_pitch_deg: float  = 45.0   # 0.785 rad
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float    = 0.25

    # ── Stuck detection ───────────────────────────────────────────────────────
    stuck_window_s: float      = 4.0
    stuck_disp_thresh: float   = 0.6
    recovery_duration_s: float = 3.0

    # ── VSGP (Paper Parameters) ───────────────────────────────────────────────
    vsgp_n_inducing: int       = 64
    vsgp_alpha: float          = 1.0
    vsgp_length_scale: float   = 0.3
    vsgp_noise_var: float      = 0.05
    vsgp_lr: float             = 0.02

    # ── Global Goal ───────────────────────────────────────────────────────────
    global_goal_x: float       = 0.0
    global_goal_y: float       = 100.0
    global_goal_z: float       = 0.0

    # ── Visualization ─────────────────────────────────────────────────────────
    viz_update_hz: float       = 5.0
    traj_history: int          = 400


CFG = Config()


# ╔════════════════════════════════════════════════════════════════════════════╗
# §1  CELL CLASSIFICATION (Code 2's robust approach)
# ╚════════════════════════════════════════════════════════════════════════════╝

class CellKind(Enum):
    CERTAIN_OBSTACLE = "certain_obstacle"
    CERTAIN_FREE     = "certain_free"
    UNCERTAIN        = "uncertain"


def classify_cells(
    occ_mean: np.ndarray,
    occ_var: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """Classify cells based on occupancy mean and variance."""
    var_min = occ_var.min()
    var_max = occ_var.max()
    var_range = (var_max - var_min)

    if var_range < 1e-9:
        var_norm = np.zeros_like(occ_var)
    else:
        var_norm = (occ_var - var_min) / var_range

    certainty = 1.0 - var_norm
    kinds = np.full(occ_mean.shape, CellKind.UNCERTAIN, dtype=object)

    certain_mask = certainty > 0.55  # certain_threshold

    # High mean = High Occupancy = Obstacle
    occ_obstacle_thresh = 0.45
    occ_free_thresh = 0.35

    kinds[certain_mask & (occ_mean > occ_obstacle_thresh)] = CellKind.CERTAIN_OBSTACLE
    kinds[certain_mask & (occ_mean <= occ_free_thresh)] = CellKind.CERTAIN_FREE

    return kinds


# ╔════════════════════════════════════════════════════════════════════════════╗
# §2  UNICYCLE KINEMATIC MODEL (Paper: Differential Drive)
# ╚════════════════════════════════════════════════════════════════════════════╝

class UnicycleKinematics:
    """
    Unicycle kinematic model for planning.
    Matches the paper's differential drive model (Section IV).
    CARLA receives standard throttle/steer/brake (Ackermann).
    """

    def __init__(self, cfg: Config = CFG):
        self.wheel_base = cfg.wheel_base
        self.cfg = cfg

    def forward(self, v: float, omega: float) -> Tuple[float, float, float]:
        """Returns (vx, vy, omega) - vy is always 0 for unicycle."""
        return float(v), 0.0, float(omega)

    @staticmethod
    def to_carla_control(
        v: float,
        omega: float,
        cfg: Config = CFG
    ) -> Tuple[float, float, float]:
        """
        Convert unicycle commands to CARLA Ackermann controls.
        steer = wheel_base * omega / v (bicycle model approximation)
        """
        speed_fwd = max(0.0, v)

        # Throttle based on forward velocity
        throttle = float(np.clip(speed_fwd / cfg.base_speed, 0.0, cfg.max_throttle))
        brake = 0.0

        if v < -0.1:
            throttle = float(np.clip(-v / cfg.base_speed, 0.0, cfg.max_throttle))

        # Steering from yaw rate (bicycle model)
        if speed_fwd > 0.5:
            steer_omega = (cfg.wheel_base * omega) / speed_fwd
        else:
            steer_omega = 0.0

        steer = float(np.clip(steer_omega, -cfg.max_steer, cfg.max_steer))

        return throttle, steer, brake


# ╔════════════════════════════════════════════════════════════════════════════╗
# §3  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚════════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha: float = 1.0,
                 length_scale: float = 0.3,
                 variance: float = 1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - Y[None, :, :]
        sq_d = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.variance * np.ones(len(X))


class VSGP:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.kernel = RationalQuadraticKernel(
            alpha=cfg.vsgp_alpha,
            length_scale=cfg.vsgp_length_scale,
            variance=1.0,
        )
        self.noise_var = cfg.vsgp_noise_var
        self.n_ind = cfg.vsgp_n_inducing
        self.lr = cfg.vsgp_lr

        # Initialize inducing points on spherical grid (alpha, beta)
        side = int(math.sqrt(self.n_ind))
        a_v = np.linspace(-np.pi / 2, np.pi / 2, side)
        b_v = np.linspace(-np.pi / 6, np.pi / 6, side)
        A, B = np.meshgrid(a_v, b_v)
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

        self.mu = np.zeros(len(self.Z))
        self.Su = np.eye(len(self.Z)) * 0.1

        self._Kuu: Optional[np.ndarray] = None
        self._Kuu_inv: Optional[np.ndarray] = None
        self._trained: bool = False
        self._update_count: int = 0

    def _select_inducing_points(self, X: np.ndarray) -> None:
        actual_n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3,
                        max_iter=50, random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            idx = np.random.choice(len(X), actual_n, replace=False)
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

        m = len(self.Z)
        Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-4
        Kfu = self.kernel(X, self.Z)

        try:
            Kuu_inv = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            return

        A = Kfu @ Kuu_inv
        noise_inv = 1.0 / self.noise_var
        Lambda = noise_inv * (A.T @ A) + Kuu_inv
        rhs = noise_inv * (A.T @ y)

        try:
            Su_new = np.linalg.inv(Lambda + np.eye(m) * 1e-4)
        except np.linalg.LinAlgError:
            return

        mu_new = Su_new @ rhs
        lr = self.lr

        self.Su = (1 - lr) * self.Su + lr * Su_new
        self.mu = (1 - lr) * self.mu + lr * mu_new

        self._Kuu = Kuu
        self._Kuu_inv = Kuu_inv
        self._trained = True

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained or self._Kuu_inv is None:
            n = len(Xs)
            return np.zeros(n), np.ones(n)

        Ksu = self.kernel(Xs, self.Z)
        A = Ksu @ self._Kuu_inv

        mean = A @ self.mu
        Kss_diag = self.kernel.diag(Xs)
        var_explained = np.sum(A * (A @ self._Kuu), axis=1)
        var_var = np.sum(A @ self.Su * A, axis=1)
        var = Kss_diag - var_explained + var_var + self.noise_var
        var = np.clip(var, 1e-6, None)

        return mean, var


# ╔════════════════════════════════════════════════════════════════════════════╗
# §4  PERCEPTION MODULE (Paper's Variance Surface Approach)
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class PerceptionOutput:
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    occupancy_mean: np.ndarray
    occupancy_var: np.ndarray
    slope_map: np.ndarray
    traversability: np.ndarray
    raw_points: np.ndarray
    cell_kinds: np.ndarray
    free_mask: Optional[np.ndarray] = None
    mean_surface: Optional[np.ndarray] = None
    variance_profile_1d: Optional[np.ndarray] = None  # For subgoal extraction


class PerceptionModule:
    ALPHA_RES = 60
    BETA_RES = 30
    ROC = 10.0  # Occupancy surface radius
    VAR_FREE_WEIGHT = 1.0
    SLOPE_WEIGHT = 1.2
    OCC_WEIGHT = 1.0

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.vsgp = VSGP(cfg)
        self._last_output: Optional[PerceptionOutput] = None

        self._a_lin = np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES)
        self._b_lin = np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES)
        AG, BG = np.meshgrid(self._a_lin, self._b_lin)
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG = AG
        self._BG = BG

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
        if len(raw) < 20:
            return self._last_output

        dist = np.linalg.norm(raw, axis=1)
        mask = (dist > 0.5) & (dist < self.cfg.lidar_range)
        pts = raw[mask]
        if len(pts) < 10:
            return self._last_output

        r = np.linalg.norm(pts, axis=1)
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1.0, 1.0))

        X = np.column_stack([alpha, beta])
        y = self.ROC - r

        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
            pts = pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)

        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var = var.reshape(self.BETA_RES, self.ALPHA_RES)

        # Calculate slope from gradients
        d_alpha = np.gradient(mean, axis=1)
        d_beta = np.gradient(mean, axis=0)
        slope = np.sqrt(d_alpha ** 2 + d_beta ** 2)
        slope = slope / (np.max(slope) + 1e-6)

        # Normalize occupancy for classification
        mean_range = mean.max() - mean.min()
        occ_norm = (mean - mean.min()) / (mean_range + 1e-6)

        var_range = var.max() - var.min()
        var_norm = (var - var.min()) / (var_range + 1e-6)

        # Traversability calculation
        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT * occ_norm
            - self.SLOPE_WEIGHT * slope
            + self.VAR_FREE_WEIGHT * (1.0 - var_norm),
            0.0, 1.0,
        )

        # Paper's variance thresholding for free space (Section V.B)
        free_mask = var > self.cfg.variance_threshold

        cell_kinds = classify_cells(occ_norm, var, self.cfg)

        # Extract 1D variance profile for subgoal extraction (Paper Fig. 2)
        variance_profile_1d = np.mean(var, axis=0)

        out = PerceptionOutput(
            alpha_grid=self._AG,
            beta_grid=self._BG,
            occupancy_mean=occ_norm,
            occupancy_var=var,
            slope_map=slope,
            traversability=traversability,
            raw_points=pts,
            cell_kinds=cell_kinds,
            free_mask=free_mask,
            mean_surface=mean,
            variance_profile_1d=variance_profile_1d,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5  SUBGOAL DATACLASS
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Subgoal:
    alpha: float
    beta: float
    distance: float
    local_pos: np.ndarray
    slope_deg: float
    safe: bool
    width_m: float
    cost: float = 0.0
    world_pos: Optional[np.ndarray] = None
    certainty: float = 0.0
    is_free_certain: bool = False
    is_obstacle: bool = False
    traversability: float = 0.5
    elevation_change: float = 0.0  # dz for cost function


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6  SUBGOAL PLANNER (Paper Eq. 6 Cost Function)
# ╚════════════════════════════════════════════════════════════════════════════╝

class SubgoalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._alpha_history: deque = deque(maxlen=8)
        self._exploration_bias: float = 0.0
        self._global_goal = np.array([cfg.global_goal_x, cfg.global_goal_y, cfg.global_goal_z])

    def set_global_goal(self, x: float, y: float, z: float = 0.0) -> None:
        """Set the global navigation goal."""
        self._global_goal = np.array([x, y, z])

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        vehicle_pose: np.ndarray,
        mode: str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        best = self.plan_from_variance(perc, vehicle_pitch_rad, vehicle_pose, mode=mode)
        return best, []

    def plan_from_variance(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        vehicle_pose: np.ndarray,
        mode: str = "forward",
    ) -> Optional[Subgoal]:
        """
        Extract subgoals from variance surface (Paper Section V.B).
        Uses Paper Eq. 6 for cost function.
        """
        if perc.occupancy_var is None or perc.variance_profile_1d is None:
            return None

        var_profile = perc.variance_profile_1d
        alpha_row = perc.alpha_grid[0]
        slope_map = perc.slope_map
        occ_mean = perc.occupancy_mean
        kinds = perc.cell_kinds

        # Direction constraint
        if mode == "forward":
            dir_mask = (alpha_row > -np.pi / 2) & (alpha_row < np.pi / 2)
        else:
            dir_mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        # Extract segments where variance > threshold (Paper's free space definition)
        segments = []
        current = []

        for i in range(len(alpha_row)):
            if not dir_mask[i]:
                if len(current) > 3:
                    segments.append(current)
                current = []
                continue

            # Paper: Free space = high variance areas
            if var_profile[i] > self.cfg.variance_threshold:
                current.append(i)
            else:
                if len(current) > 3:
                    segments.append(current)
                current = []

        if len(current) > 3:
            segments.append(current)

        if len(segments) == 0:
            return self._fallback_subgoal(perc, mode, vehicle_pitch_rad, vehicle_pose)

        best_subgoal: Optional[Subgoal] = None
        best_cost: float = float("inf")

        robot_z = vehicle_pose[2]
        pitch = vehicle_pitch_rad

        for seg in segments:
            alpha_vals = alpha_row[seg]
            α_center = float(np.mean(alpha_vals))
            α_span = float(alpha_vals[-1] - alpha_vals[0])

            # Calculate segment width in meters
            width_m = self.cfg.subgoal_distance * abs(α_span)
            eff_width = width_m * max(np.cos(α_center), 0.1)
            req_w = self.cfg.vehicle_width + self.cfg.safety_margin

            if eff_width < req_w:
                continue

            # Check wheel stability
            d = self.cfg.subgoal_distance
            lx_c = d * np.cos(α_center)
            ly_c = d * np.sin(α_center)

            if not self._are_wheels_stable(perc, lx_c, ly_c, α_center):
                continue

            # Check rollout cost (obstacles in path)
            C_rollout = self._rollout_cost(perc, α_center, d)
            if C_rollout > 900:
                continue

            # Get elevation at subgoal (approximate from occupancy mean)
            mid_beta_idx = perc.beta_grid.shape[0] // 2
            alpha_idx = int(np.argmin(np.abs(alpha_row - α_center)))
            alpha_idx = np.clip(alpha_idx, 0, len(alpha_row) - 1)

            # Elevation change (dz) - from occupancy surface
            dz = float(perc.mean_surface[mid_beta_idx, alpha_idx]) if perc.mean_surface is not None else 0.0
            subgoal_z = robot_z + dz

            # ─────────────────────────────────────────────────────────────────
            # PAPER EQ. 6 COST FUNCTION
            # J(gi) = kdir*α + kdst*D + kstp*Cstp
            # Cstp = (dz)^2 + exp(sign(ψ)*dz)*|ψ|
            # ─────────────────────────────────────────────────────────────────

            # Direction cost (α)
            C_dir = abs(α_center) / (np.pi / 2)

            # Distance cost to global goal (D)
            subgoal_world_x = vehicle_pose[0] + lx_c
            subgoal_world_y = vehicle_pose[1] + ly_c
            D_cost = np.sqrt((self._global_goal[0] - subgoal_world_x)**2 +
                            (self._global_goal[1] - subgoal_world_y)**2)
            D_cost = D_cost / 100.0  # Normalize (assume 100m max distance)

            # Steepness cost (Paper Eq. 6)
            C_stp = (dz ** 2) + np.exp(np.sign(pitch) * dz) * abs(pitch)

            # Total cost
            J = (self.cfg.k_direction * C_dir +
                 self.cfg.k_distance * D_cost +
                 self.cfg.k_steepness * C_stp)

            # Add exploration bias for recovery
            if self._exploration_bias > 0:
                J *= (1.0 - self._exploration_bias * 0.4)

            # Check oscillation
            C_osc = 0.0
            if len(self._alpha_history) >= 4:
                hist = np.array(list(self._alpha_history))
                if np.std(hist) > 0.3:
                    C_osc = 0.5 * abs(α_center - float(np.mean(hist)))
                    J += C_osc

            lx = d * np.cos(α_center)
            ly = d * np.sin(α_center)

            # Get slope at subgoal
            slope_val = float(slope_map[mid_beta_idx, alpha_idx]) if mid_beta_idx < slope_map.shape[0] else 0.0
            slope_deg = float(np.degrees(np.arctan(slope_val)))

            # Check cell kind
            cell_kind = kinds[mid_beta_idx, alpha_idx] if mid_beta_idx < kinds.shape[0] else CellKind.UNCERTAIN
            is_free_certain = (cell_kind == CellKind.CERTAIN_FREE)

            sg = Subgoal(
                alpha=α_center,
                beta=0.0,
                distance=d,
                local_pos=np.array([lx, ly, 0.0]),
                slope_deg=slope_deg,
                safe=True,
                width_m=width_m,
                cost=J,
                certainty=1.0 - float(var_profile[alpha_idx]),
                is_free_certain=is_free_certain,
                is_obstacle=(cell_kind == CellKind.CERTAIN_OBSTACLE),
                traversability=float(perc.traversability[mid_beta_idx, alpha_idx]) if mid_beta_idx < perc.traversability.shape[0] else 0.5,
                elevation_change=dz,
            )

            if J < best_cost:
                best_cost = J
                best_subgoal = sg

        if best_subgoal is None:
            return self._fallback_subgoal(perc, mode, vehicle_pitch_rad, vehicle_pose)

        self._alpha_history.append(best_subgoal.alpha)
        return best_subgoal

    def find_recovery_direction(self, perc: PerceptionOutput) -> Optional[float]:
        """Find best direction for reverse recovery."""
        if perc.variance_profile_1d is None:
            return None

        alpha_row = perc.alpha_grid[0]
        var_profile = perc.variance_profile_1d

        rear_mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        valid_indices = []
        for i in range(len(alpha_row)):
            if rear_mask[i] and var_profile[i] > self.cfg.variance_threshold:
                valid_indices.append((i, float(alpha_row[i]), float(var_profile[i])))

        if not valid_indices:
            return None

        # Choose direction with highest variance (most free space)
        best = max(valid_indices, key=lambda x: x[2])
        return best[1]

    def set_exploration_bias(self, v: float) -> None:
        self._exploration_bias = float(np.clip(v, 0.0, 1.0))

    def _fallback_subgoal(
        self,
        perc: PerceptionOutput,
        mode: str,
        vehicle_pitch_rad: float,
        vehicle_pose: np.ndarray,
    ) -> Optional[Subgoal]:
        """Fallback when no valid segments found."""
        alpha_row = perc.alpha_grid[0]
        var_profile = perc.variance_profile_1d

        if var_profile is None:
            return None

        if mode == "forward":
            mask = (alpha_row > -np.pi / 2) & (alpha_row < np.pi / 2)
        else:
            mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        candidates = [
            (i, float(alpha_row[i]), float(var_profile[i]))
            for i in range(len(alpha_row))
            if mask[i]
        ]

        if not candidates:
            return None

        # Choose highest variance (most free)
        best = max(candidates, key=lambda t: t[2])
        a = best[1]
        d = self.cfg.subgoal_distance
        lx, ly = d * np.cos(a), d * np.sin(a)

        return Subgoal(
            alpha=a,
            beta=0.0,
            distance=d,
            local_pos=np.array([lx, ly, 0.0]),
            slope_deg=0.0,
            safe=False,
            width_m=0.5,
            cost=999.0,
            certainty=0.0,
            is_free_certain=False,
            traversability=best[2],
            elevation_change=0.0,
        )

    def _rollout_cost(self, perc: PerceptionOutput, a: float, d: float) -> float:
        """Evaluate cost along trajectory to subgoal."""
        steps = 12
        total_cost = 0.0
        alpha_row = perc.alpha_grid[0]
        n_beta = perc.beta_grid.shape[0]
        bi_mid = n_beta // 2

        for i in range(1, steps + 1):
            frac = i / steps
            x = frac * d * np.cos(a)
            y = frac * d * np.sin(a)

            alpha_pt = np.arctan2(y, x)
            ai = int(np.argmin(np.abs(alpha_row - alpha_pt)))
            ai = np.clip(ai, 0, len(alpha_row) - 1)
            bi = np.clip(bi_mid, 0, perc.slope_map.shape[0] - 1)

            ck = perc.cell_kinds[bi, ai]
            if ck == CellKind.CERTAIN_OBSTACLE:
                return 999.0

            slope = float(perc.slope_map[bi, ai])
            var = float(perc.occupancy_var[bi, ai])

            if ck == CellKind.CERTAIN_FREE:
                total_cost += slope * 1.5 - 0.3
            elif ck == CellKind.UNCERTAIN:
                total_cost += slope * 2.0 + var * 1.0
            else:
                total_cost += slope * 2.0 + var * 1.5

        return max(total_cost / steps, 0.0)

    def _are_wheels_stable(
        self,
        perc: PerceptionOutput,
        cx: float,
        cy: float,
        yaw: float,
    ) -> bool:
        """Check if vehicle wheels land on stable ground."""
        wb = self.cfg.wheel_base / 2.0
        tw = self.cfg.track_width / 2.0

        wheels = np.array([[wb, tw], [wb, -tw], [-wb, tw], [-wb, -tw]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        wheels = wheels @ R.T
        wheels[:, 0] += cx
        wheels[:, 1] += cy

        stable = 0
        for wx, wy in wheels:
            if not self._is_certain_obstacle(perc, wx, wy):
                stable += 1
        return stable >= 3

    def _is_certain_obstacle(
        self,
        perc: PerceptionOutput,
        x: float,
        y: float,
    ) -> bool:
        """Check if position is certain obstacle."""
        r = math.sqrt(x * x + y * y)
        if r < 1e-3:
            return True

        alpha = math.atan2(y, x)
        ai = int(np.argmin(np.abs(perc.alpha_grid[0] - alpha)))
        ai = np.clip(ai, 0, perc.alpha_grid.shape[1] - 1)
        bi = perc.beta_grid.shape[0] // 2
        bi = np.clip(bi, 0, perc.cell_kinds.shape[0] - 1)

        return perc.cell_kinds[bi, ai] == CellKind.CERTAIN_OBSTACLE


# ╔════════════════════════════════════════════════════════════════════════════╗
# §7  CONTROLLER MODULE (Smooth Ackermann Control)
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    reverse: bool = False


class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.kin = UnicycleKinematics(cfg)
        self._state = ControlState()
        self._v_ema = 0.0
        self._omega_ema = 0.0

    def compute(
        self,
        subgoal: Optional[Subgoal],
        vehicle_speed_ms: float,
        terrain_slope_deg: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> ControlState:
        if subgoal is None:
            return self._apply_smooth(0.0, 0.0, roll_deg=roll_deg, pitch_deg=pitch_deg)

        alpha = subgoal.alpha
        slope = subgoal.slope_deg

        # Speed reduction based on slope and direction
        trav_factor = float(np.clip(
            subgoal.traversability * (1.2 if subgoal.is_free_certain else 1.0),
            0.3, 1.0,
        ))

        speed_factor = max(
            0.3,
            trav_factor
            - slope / self.cfg.max_slope_deg * 0.4
            - abs(alpha) / (math.pi / 2) * 0.25,
        )

        v_desired = self.cfg.base_speed * speed_factor
        omega_desired = alpha * 1.6

        return self._apply_smooth(v_desired, omega_desired,
                                   roll_deg=roll_deg, pitch_deg=pitch_deg)

    def compute_recovery(self, phase: int, roll_deg: float, pitch_deg: float) -> ControlState:
        """Recovery control with smoothing."""
        steers = [0.0, 0.5, -0.5]
        steer = steers[min(phase, 2)]

        # Use negative velocity for reverse
        return self._apply_smooth(
            -1.5,  # Reverse speed
            steer * 2.0,
            roll_deg=roll_deg, pitch_deg=pitch_deg,
        )

    def _stability_modifiers(
        self, roll_deg: float, pitch_deg: float
    ) -> Tuple[float, float]:
        """Apply stability corrections based on roll/pitch."""
        throttle_scale = 1.0
        corrective_steer = 0.0

        abs_roll = abs(roll_deg)
        if abs_roll > self.cfg.max_safe_roll_deg:
            excess = abs_roll - self.cfg.max_safe_roll_deg
            roll_factor = 1.0 - min(excess / 15.0, 1.0)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale
                                 + (1.0 - self.cfg.stability_throttle_scale) * roll_factor)
            corrective_steer = -math.copysign(
                min(self.cfg.roll_corrective_steer * (excess / 10.0),
                    self.cfg.roll_corrective_steer),
                roll_deg,
            )

        abs_pitch = abs(pitch_deg)
        if abs_pitch > self.cfg.max_safe_pitch_deg:
            excess_p = abs_pitch - self.cfg.max_safe_pitch_deg
            pitch_factor = 1.0 - min(excess_p / 10.0, 0.7)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale * pitch_factor + 0.15)

        return float(throttle_scale), float(corrective_steer)

    def _apply_smooth(
        self,
        v: float,
        omega: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> ControlState:
        """Apply EMA smoothing and rate limiting."""
        a = self.cfg.ema_alpha
        self._v_ema = a * v + (1 - a) * self._v_ema
        self._omega_ema = a * omega + (1 - a) * self._omega_ema

        throttle, steer, brake = UnicycleKinematics.to_carla_control(
            self._v_ema, self._omega_ema, self.cfg,
        )

        # Rate limiting
        steer = float(np.clip(steer,
                              self._state.steer - self.cfg.max_steer_rate,
                              self._state.steer + self.cfg.max_steer_rate))
        throttle = float(np.clip(throttle,
                                 self._state.throttle - self.cfg.max_throttle_rate,
                                 self._state.throttle + self.cfg.max_throttle_rate))

        # Stability modifiers
        thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
        throttle *= thr_scale
        steer = float(np.clip(steer + cor_steer,
                              -self.cfg.max_steer, self.cfg.max_steer))

        # Determine reverse flag
        reverse = (self._v_ema < -0.1)

        self._state = ControlState(throttle=throttle, steer=steer, brake=brake, reverse=reverse)
        return self._state

    def reset_ema(self) -> None:
        self._v_ema = 0.0
        self._omega_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §8  STUCK DETECTION & RECOVERY
# ╚════════════════════════════════════════════════════════════════════════════╝

class StuckDetector:
    IDLE = "IDLE"
    RECOVERING = "RECOVERING"

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._pos_history: deque = deque(maxlen=200)
        self._steer_history: deque = deque(maxlen=40)
        self._state = self.IDLE
        self._recovery_start: float = 0.0
        self._recovery_phase: int = 0

    def update(
        self, pos: np.ndarray, steer: float, t: float
    ) -> Tuple[bool, int]:
        self._pos_history.append((t, pos.copy()))
        self._steer_history.append(steer)

        if self._state == self.RECOVERING:
            elapsed = t - self._recovery_start
            phase_dur = self.cfg.recovery_duration_s / 3.0
            if elapsed > self.cfg.recovery_duration_s:
                self._state = self.IDLE
                return False, 0
            phase = min(int(elapsed / phase_dur), 2)
            return True, phase

        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    self._trigger_recovery(t)
                    return True, 0

        if len(self._steer_history) == self._steer_history.maxlen:
            arr = np.array(self._steer_history)
            signs = np.sign(arr)
            flips = int(np.sum(np.diff(signs) != 0))
            if flips > len(signs) * 0.70:
                self._trigger_recovery(t)
                return True, 0

        return False, 0

    def _trigger_recovery(self, t: float) -> None:
        if self._state == self.IDLE:
            print("[STUCK] Recovery triggered!")
            self._state = self.RECOVERING
            self._recovery_start = t
            self._recovery_phase = 0
            self._pos_history.clear()

    def push_steer(self, steer: float) -> None:
        self._steer_history.append(steer)

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING


# ╔════════════════════════════════════════════════════════════════════════════╗
# §9  VISUALIZATION MODULE
# ╚════════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        try:
            plt.ion()
            self.fig = plt.figure("VSGP Navigator", figsize=(16, 9))
            self.fig.patch.set_facecolor("#111")
            gs = GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.3)

            self.ax_front = self.fig.add_subplot(gs[0, 0])
            self.ax_lidar = self.fig.add_subplot(gs[0, 1])
            self.ax_rear = self.fig.add_subplot(gs[0, 2])
            self.ax_cost = self.fig.add_subplot(gs[1, 0])
            self.ax_traj = self.fig.add_subplot(gs[1, 1])
            self.ax_vel = self.fig.add_subplot(gs[1, 2])

            for ax in self.fig.get_axes():
                ax.set_facecolor("#1a1a2e")
                for sp in ax.spines.values():
                    sp.set_color("#444")
                ax.tick_params(colors="#ccc", labelsize=7)
                ax.title.set_color("#ddd")
                ax.xaxis.label.set_color("#aaa")
                ax.yaxis.label.set_color("#aaa")
        except Exception as e:
            print(f"[WARN] Visualization init failed: {e}")
            self.fig = None

        maxlen = cfg.traj_history
        self._cost_total: deque = deque(maxlen=maxlen)
        self._cost_steep: deque = deque(maxlen=maxlen)
        self._vel_lin: deque = deque(maxlen=maxlen)
        self._vel_ang: deque = deque(maxlen=maxlen)
        self._traj_x: deque = deque(maxlen=maxlen)
        self._traj_y: deque = deque(maxlen=maxlen)
        self._subgoal_x: deque = deque(maxlen=20)
        self._subgoal_y: deque = deque(maxlen=20)
        self._subgoal_certain: deque = deque(maxlen=20)

        self._front_img: Optional[np.ndarray] = None
        self._rear_img: Optional[np.ndarray] = None
        self._lidar_pts: Optional[np.ndarray] = None
        self._mode_text: str = "AUTO"
        self._last_update: float = 0.0
        self._interval: float = 1.0 / cfg.viz_update_hz

    def push_data(
        self,
        *,
        subgoal: Optional[Subgoal],
        vehicle_pos: np.ndarray,
        speed_ms: float,
        steer: float,
        perc: Optional[PerceptionOutput],
        mode: str = "AUTO",
    ) -> None:
        self._traj_x.append(float(vehicle_pos[0]))
        self._traj_y.append(float(vehicle_pos[1]))

        if subgoal is not None:
            self._cost_total.append(subgoal.cost)
            self._cost_steep.append(subgoal.elevation_change)
            self._subgoal_x.append(vehicle_pos[0] + subgoal.local_pos[0])
            self._subgoal_y.append(vehicle_pos[1] + subgoal.local_pos[1])
            self._subgoal_certain.append(subgoal.is_free_certain)
        else:
            self._cost_total.append(0)
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
        if self.fig is None:
            return
        now = time.time()
        if not force and (now - self._last_update) < self._interval:
            return
        self._last_update = now
        try:
            self._draw_camera(self.ax_front, self._front_img, "Front Camera")
            self._draw_camera(self.ax_rear, self._rear_img, "Rear Camera")
            self._draw_lidar()
            self._draw_cost()
            self._draw_trajectory()
            self._draw_velocity()
            self.fig.suptitle(
                f"VSGP Mapless Navigator (Paper Eq.6) | Mode: {self._mode_text}",
                color="#eee", fontsize=11, y=0.99,
            )
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def _draw_camera(self, ax, img, title):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if img is not None:
            ax.imshow(img)

    def _draw_lidar(self):
        ax = self.ax_lidar
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("LiDAR (top view)", fontsize=8)
        if self._lidar_pts is not None and len(self._lidar_pts) > 0:
            pts = self._lidar_pts
            z = pts[:, 2]
            z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)
            ax.scatter(pts[:, 1], pts[:, 0], c=z_n, cmap="plasma", s=0.5, alpha=0.6)
            ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
            ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)

    def _draw_cost(self):
        ax = self.ax_cost
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Cost Function (Eq.6)", fontsize=8)
        n = len(self._cost_total)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._cost_total), c="#ff6b6b", lw=1.2, label="Total Cost")
        ax.plot(xs, list(self._cost_steep), c="#74b9ff", lw=0.9, label="Elevation (dz)")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Value", fontsize=7)

    def _draw_trajectory(self):
        ax = self.ax_traj
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Trajectory + Subgoals", fontsize=8)
        tx, ty = list(self._traj_x), list(self._traj_y)
        if len(tx) > 1:
            ax.plot(ty, tx, c="#00cec9", lw=1.2, label="Actual")
        if len(tx) > 0:
            ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)
        sx = list(self._subgoal_x)
        sy = list(self._subgoal_y)
        sc = list(self._subgoal_certain)
        if sx:
            colors = ["#00b894" if c else "#e17055" for c in sc]
            ax.scatter(sy, sx, c=colors, s=14, marker="x", zorder=4,
                       label="Subgoal (green=certain)")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    def _draw_velocity(self):
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
        ax.set_ylabel("m/s", fontsize=7)

    def close(self) -> None:
        if self.fig:
            plt.close(self.fig)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §10  CARLA SENSOR WRAPPERS
# ╚════════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg = cfg
        self.vehicle = vehicle
        self.world = world
        self._actors: list = []
        self._active: bool = True

        self._lidar_data = None
        self._front_image: Optional[np.ndarray] = None
        self._rear_image: Optional[np.ndarray] = None

        bp_lib = world.get_blueprint_library()

        # ── LiDAR ─────────────────────────────────────────────────────────────
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        if lidar_bp is None:
            # Fallback if specific LiDAR not found
            lidar_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")

        if lidar_bp is None:
            raise RuntimeError("LiDAR blueprint not found")

        lidar_bp.set_attribute("range", str(cfg.lidar_range))
        lidar_bp.set_attribute("points_per_second", str(cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("channels", str(cfg.lidar_channels))
        lidar_bp.set_attribute("upper_fov", str(cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov", str(cfg.lidar_lower_fov))

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
            carla.Transform(carla.Location(x=1.5, z=1.8)), "front",
        ))
        self._actors.append(self._spawn_camera(
            bp_lib, vehicle,
            carla.Transform(carla.Location(x=-1.5, z=1.8), carla.Rotation(yaw=180)), "rear",
        ))

    def _spawn_camera(self, bp_lib, vehicle, transform, tag):
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov", str(self.cfg.camera_fov))
        cam = self.world.spawn_actor(cam_bp, transform, attach_to=vehicle)
        cam.listen(self._on_front_image if tag == "front" else self._on_rear_image)
        return cam

    def _on_lidar(self, data) -> None:
        if self._active:
            self._lidar_data = data

    def _on_front_image(self, image) -> None:
        if not self._active:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._front_image = arr[:, :, ::-1].copy()

    def _on_rear_image(self, image) -> None:
        if not self._active:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._rear_image = arr[:, :, ::-1].copy()

    @property
    def lidar_data(self):
        return self._lidar_data

    @property
    def front_image(self):
        return self._front_image

    @property
    def rear_image(self):
        return self._rear_image

    def destroy(self) -> None:
        self._active = False
        for a in self._actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# §11  MANUAL CONTROLLER (Keyboard Fallback)
# ╚════════════════════════════════════════════════════════════════════════════╝

class ManualController:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._rev = False
        self._prev_r = False
        self._thr_ema = 0.0
        self._str_ema = 0.0
        self._EMA = 0.30

    def tick(self, keys) -> ControlState:
        boost = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        if keys[pygame.K_r] and not self._prev_r:
            self._rev = not self._rev
            self._prev_r = True
        elif not keys[pygame.K_r]:
            self._prev_r = False

        thr_max = self.cfg.max_throttle * (1.4 if boost else 1.0)
        rev_max = self.cfg.max_throttle * 0.8

        throttle = 0.0
        brake = 0.0
        reverse = self._rev

        if keys[pygame.K_SPACE]:
            brake = self.cfg.max_brake
            throttle = 0.0
        elif keys[pygame.K_w]:
            throttle = thr_max if not reverse else 0.0
            brake = 0.0
        elif keys[pygame.K_s]:
            if reverse:
                throttle = rev_max
            else:
                brake = self.cfg.max_brake * 0.6

        if reverse and keys[pygame.K_w]:
            self._rev = False
            reverse = False
            throttle = thr_max

        str_max = self.cfg.max_steer * (1.2 if boost else 1.0)
        steer = 0.0
        if keys[pygame.K_a]:
            steer = -str_max
        elif keys[pygame.K_d]:
            steer = str_max

        if reverse:
            steer = -steer

        a = self._EMA
        self._thr_ema = a * throttle + (1 - a) * self._thr_ema
        self._str_ema = a * steer + (1 - a) * self._str_ema

        final_brake = brake if brake > 0 else 0.0

        return ControlState(
            throttle=float(np.clip(self._thr_ema, 0.0, self.cfg.max_throttle)),
            steer=float(np.clip(self._str_ema, -self.cfg.max_steer, self.cfg.max_steer)),
            brake=float(np.clip(final_brake, 0.0, self.cfg.max_brake)),
            reverse=reverse,
        )

    def reset(self) -> None:
        self._rev = False
        self._thr_ema = 0.0
        self._str_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §12  VEHICLE SPAWNER (Code 1 Logic)
# ╚════════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    # Fallback if Cybertruck not available in specific CARLA version
    if vehicle_bp is None:
        print(f"[WARN] {cfg.vehicle_blueprint} not found. Fallback to vehicle.tesla.model3")
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

    if vehicle_bp is None:
        vehicle_bp = bp_lib.find("vehicle.lincoln.mkz2017")

    if vehicle_bp is None:
        raise RuntimeError("No valid vehicle blueprint found")

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

    print(f"[SPAWN] {vehicle_bp.id} at {vehicle.get_location()}")

    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    print(f"[SPAWN] Settled Z = {vehicle.get_location().z:.2f} m")
    return vehicle, vehicle.get_location().z


# ╔════════════════════════════════════════════════════════════════════════════╗
# §13  MAIN NAVIGATION LOOP
# ╚════════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._running = False
        self._mode = "AUTO"

        self._client = None
        self._world = None
        self._vehicle = None
        self._sensors: Optional[SensorManager] = None

        self._perception = PerceptionModule(cfg)
        self._planner = SubgoalPlanner(cfg)
        self._controller = Controller(cfg)
        self._manual = ManualController(cfg)
        self._stuck = StuckDetector(cfg)
        self._viz = VisualizationModule(cfg)

        # Set Global Goal for Planner (Paper Eq. 6 Distance Cost)
        self._planner.set_global_goal(cfg.global_goal_x, cfg.global_goal_y, cfg.global_goal_z)

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle AUTO/MANUAL, R=reverse, ESC=quit")
        self._screen = pygame.display.set_mode((260, 112))
        self._font = pygame.font.SysFont("monospace", 12)

        self._step = 0
        self._t0 = 0.0

    def connect(self) -> None:
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port} …")
        try:
            self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
            self._client.set_timeout(self.cfg.carla_timeout)
            self._world = self._client.get_world()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA: {e}")

        if self.cfg.synchronous:
            settings = self._world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self._world.apply_settings(settings)
            print("[CARLA] Synchronous mode ON")

    def setup(self) -> None:
        self._vehicle, self._spawn_z = spawn_vehicle(self._world, self.cfg)
        self._sensors = SensorManager(self._world, self._vehicle, self.cfg)
        self._t0 = time.time()
        print("[SETUP] All sensors attached. Starting navigation loop.")
        print(f"[GOAL] Global Goal set to ({self.cfg.global_goal_x}, {self.cfg.global_goal_y})")

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

    def _tick(self) -> None:
        if self.cfg.synchronous:
            self._world.tick()

        t_now = time.time() - self._t0
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
                    self._manual.reset()
                    print(f"[MODE] Switched to {self._mode}")

        keys = pygame.key.get_pressed()

        # ── Vehicle state ─────────────────────────────────────────────────────
        loc = self._vehicle.get_location()
        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos = np.array([loc.x, loc.y, loc.z])

        transform = self._vehicle.get_transform()
        pitch_rad = math.radians(transform.rotation.pitch)
        roll_rad = math.radians(transform.rotation.roll)
        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)

        # ── Perception ────────────────────────────────────────────────────────
        perc: Optional[PerceptionOutput] = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        best_goal: Optional[Subgoal] = None
        ctrl_state: ControlState

        # ── Control decision ──────────────────────────────────────────────────
        if self._mode == "MANUAL":
            ctrl_state = self._manual.tick(keys)

        else:  # AUTO
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                self._planner.set_exploration_bias(0.9)

                recovery_alpha: Optional[float] = None
                if perc is not None:
                    recovery_alpha = self._planner.find_recovery_direction(perc)

                if recovery_alpha is not None:
                    d = self.cfg.subgoal_distance
                    lx = -abs(d * math.cos(recovery_alpha))
                    ly = d * math.sin(recovery_alpha)

                    best_goal = Subgoal(
                        alpha=recovery_alpha,
                        beta=0.0,
                        distance=d,
                        local_pos=np.array([lx, ly, 0.0]),
                        slope_deg=0.0,
                        safe=True,
                        width_m=1.0,
                        cost=0.0,
                        certainty=0.5,
                        is_free_certain=False,
                        traversability=0.5,
                        elevation_change=0.0,
                    )

                    # Use Controller for smoothing even in recovery
                    ctrl_state = self._controller.compute_recovery(rec_phase, roll_deg, pitch_deg)
                    ctrl_state.reverse = True  # Force reverse flag

                    print(f"[RECOVERY] VSGP α={recovery_alpha:.2f} steer={ctrl_state.steer:.2f}")
                else:
                    ctrl_state = self._controller.compute_recovery(rec_phase, roll_deg, pitch_deg)
                    ctrl_state.reverse = True

            else:
                self._planner.set_exploration_bias(0.0)

                if perc is not None:
                    best_goal, _ = self._planner.plan(
                        perc, pitch_rad, pos, mode="forward"
                    )

                slope = best_goal.slope_deg if best_goal else 0.0
                ctrl_state = self._controller.compute(
                    best_goal, speed, slope,
                    roll_deg=roll_deg, pitch_deg=pitch_deg,
                )

            self._stuck.push_steer(ctrl_state.steer)

        # ── Apply control ─────────────────────────────────────────────────────
        self._vehicle.apply_control(carla.VehicleControl(
            throttle=float(ctrl_state.throttle),
            steer=float(ctrl_state.steer),
            brake=float(ctrl_state.brake),
            reverse=ctrl_state.reverse,
        ))

        # ── HUD ───────────────────────────────────────────────────────────────
        self._screen.fill((20, 20, 40))
        rev_flag = "REV" if ctrl_state.reverse else "FWD"
        cert_str = ""
        if best_goal is not None and self._mode == "AUTO":
            kind = "FREE✓" if best_goal.is_free_certain else "UNC"
            cert_str = f"  [{kind} {best_goal.certainty:.2f}]"

        lines = [
            f"Mode: {self._mode}  {rev_flag}",
            f"Speed: {speed:.1f} m/s",
            f"Thr:{ctrl_state.throttle:.2f}  Str:{ctrl_state.steer:.2f}",
            f"Roll:{roll_deg:+.1f}°  Pitch:{pitch_deg:+.1f}°",
            f"Step:{self._step}  t={t_now:.1f}s",
            f"Subgoal{cert_str}",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (200, 220, 255))
            self._screen.blit(surf, (5, 4 + i * 17))
        pygame.display.flip()

        # ── Visualization push ────────────────────────────────────────────────
        self._viz.push_data(
            subgoal=best_goal if self._mode == "AUTO" else None,
            vehicle_pos=pos,
            speed_ms=speed,
            steer=ctrl_state.steer,
            perc=perc,
            mode=self._mode,
        )
        if self._sensors.front_image is not None:
            self._viz.set_front_image(self._sensors.front_image)
        if self._sensors.rear_image is not None:
            self._viz.set_rear_image(self._sensors.rear_image)
        self._viz.render()

        if not self.cfg.synchronous:
            time.sleep(self.cfg.fixed_delta_seconds)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §14  ENTRY POINT
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