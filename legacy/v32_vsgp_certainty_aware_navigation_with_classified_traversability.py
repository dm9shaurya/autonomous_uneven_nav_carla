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
# Removed unused: from numpy.random import beta
import pygame
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Optional scipy / sklearn
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
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


# ╔══════════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA connection ──────────────────────────────────────────────────────
    carla_host: str            = "127.0.0.1"
    carla_port: int            = 2000
    carla_timeout: float       = 20.0
    synchronous: bool          = True
    fixed_delta_seconds: float = 0.05          # 20 Hz physics

    # ── Vehicle ───────────────────────────────────────────────────────────────
    # FIX: Cybertruck often not available in standard libs, using Model 3
    vehicle_blueprint: str     = "vehicle.tesla.model3"
    spawn_z_offset: float      = 5.0
    post_spawn_wait_s: float   = 3.0

    # ── Vehicle Geometry ──────────────────────────────────────────────────────
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

    # ── Mecanum geometry (abstract planning model) ────────────────────────────
    wheel_radius: float        = 0.15
    lx: float                  = 0.25
    ly: float                  = 0.20
    max_wheel_omega: float     = 40.0

    # ── Planner ───────────────────────────────────────────────────────────────
    robot_width: float         = 0.8
    safety_margin: float       = 0.4
    max_slope_deg: float       = 20.0
    subgoal_distance: float    = 4.0
    n_subgoals_max: int        = 8

    # ── Certainty thresholds (NEW) ────────────────────────────────────────────
    certain_threshold: float   = 0.55
    occ_obstacle_thresh: float = 0.45
    occ_free_thresh: float     = 0.35
    uncertain_explore_bonus: float = 0.25

    # ── Cost weights ──────────────────────────────────────────────────────────
    w_direction: float         = 1.2
    w_distance: float          = 0.6
    w_steepness: float         = 1.5
    w_flatness: float          = 2.0
    w_collision: float         = 5.0
    w_oscillation: float       = 0.8
    w_certain_free: float      = 1.8
    w_uncertain_free: float    = 1.0

    # ── Control ───────────────────────────────────────────────────────────────
    max_throttle: float        = 0.55
    max_steer: float           = 0.6
    max_brake: float           = 0.8
    ema_alpha: float           = 0.35
    max_steer_rate: float      = 0.12
    max_throttle_rate: float   = 0.08
    base_speed: float          = 4.0

    # ── Stability (roll / pitch) ──────────────────────────────────────────────
    max_safe_roll_deg: float       = 15.0
    max_safe_pitch_deg: float      = 22.0
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float    = 0.25

    # ── Stuck detection ───────────────────────────────────────────────────────
    stuck_window_s: float      = 4.0
    stuck_disp_thresh: float   = 0.6
    recovery_duration_s: float = 3.0

    # ── VSGP ──────────────────────────────────────────────────────────────────
    vsgp_n_inducing: int       = 64
    vsgp_alpha: float          = 1.0
    vsgp_length_scale: float   = 0.3
    vsgp_noise_var: float      = 0.05
    vsgp_lr: float             = 0.02

    # ── Manual control ────────────────────────────────────────────────────────
    manual_throttle: float     = 0.55
    manual_steer: float        = 0.50
    manual_brake: float        = 0.80
    manual_reverse_throttle: float = 0.45

    # ── Visualization ─────────────────────────────────────────────────────────
    viz_update_hz: float       = 5.0
    traj_history: int          = 400


CFG = Config()


# ╔══════════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §1  CERTAINTY CELL CLASSIFICATION  (NEW)
# ╚══════════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class CellKind(Enum):
    CERTAIN_OBSTACLE = "certain_obstacle"
    CERTAIN_FREE     = "certain_free"
    UNCERTAIN        = "uncertain"


def classify_cells(
    occ_mean : np.ndarray,   # (H, W) - Expected Normalized [0, 1]
    occ_var  : np.ndarray,   # (H, W)
    cfg      : Config = CFG,
) -> np.ndarray:
    """
    Returns an object array of CellKind with shape (H, W).
    FIX: Assumes occ_mean is normalized occupancy probability (0=free, 1=occ).
    """
    var_min  = occ_var.min()
    var_max  = occ_var.max()
    # Avoid division by zero
    var_range = (var_max - var_min)
    if var_range < 1e-9:
        var_norm = np.zeros_like(occ_var)
    else:
        var_norm = (occ_var - var_min) / var_range

    certainty = 1.0 - var_norm
    kinds = np.full(occ_mean.shape, CellKind.UNCERTAIN, dtype=object)

    certain_mask = certainty > cfg.certain_threshold
    # High mean = High Occupancy = Obstacle
    kinds[certain_mask & (occ_mean > cfg.occ_obstacle_thresh)] = CellKind.CERTAIN_OBSTACLE
    kinds[certain_mask & (occ_mean <= cfg.occ_free_thresh)] = CellKind.CERTAIN_FREE

    return kinds


# ╔══════════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §2  MECANUM KINEMATIC MODEL  (abstract planning, not real wheel control)
# ╚══════════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class MecanumKinematics:
    """
    Four-mecanum-wheel kinematic model used as a *planning* abstraction.
    CARLA always receives standard throttle / steer / brake.
    """

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
        """
        FIX: Ackermann vehicles cannot move laterally (vy).
        We ignore vy for steering calculation to prevent erratic behavior.
        """
        speed = math.sqrt(vx ** 2 + vy ** 2)  # Keep speed magnitude for throttle
        # For throttle, we primarily care about forward velocity vx
        speed_fwd = max(0.0, vx)

        throttle = float(np.clip(speed_fwd / cfg.base_speed, 0.0, cfg.max_throttle))
        brake    = 0.0

        if vx < -0.1:
            # Reverse logic handled by ControlState flag, but throttle calc here
            throttle = float(np.clip(-vx / cfg.base_speed, 0.0, cfg.max_throttle))

        # FIX: Steer based on Omega (Yaw Rate), ignore Vy (Lateral)
        # Approximate steering angle from yaw rate: delta ≈ L * omega / v
        if speed_fwd > 0.5:
            steer_omega = (cfg.wheel_base * omega) / speed_fwd
        else:
            steer_omega = 0.0

        steer = float(np.clip(steer_omega, -cfg.max_steer, cfg.max_steer))
        return throttle, steer, brake


# ╔══════════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §3  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚══════════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

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

        # FIX: Align inducing point initialization ranges with PerceptionModule grid
        side = int(math.sqrt(self.n_ind))
        # Perception uses alpha: [-pi/2, pi/2], beta: [-pi/6, pi/6]
        a_v  = np.linspace(-np.pi / 2, np.pi / 2, side)
        b_v  = np.linspace(-np.pi / 6, np.pi / 6, side) 
        A, B = np.meshgrid(a_v, b_v)
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

        n      = len(self.Z)
        self.mu = np.zeros(n)
        self.Su = np.eye(n) * 0.1

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return
        self._update_count += 1
        if self._update_count % 50 == 0:
            self._select_inducing_points(X)

        m         = len(self.Z)
        # FIX: Increased jitter for numerical stability
        Kuu       = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-4 
        Kfu       = self.kernel(X, self.Z)
        
        try:
            Kuu_inv   = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            return # Skip update if singular

        A         = Kfu @ Kuu_inv

        noise_inv = 1.0 / self.noise_var
        Lambda    = noise_inv * (A.T @ A) + Kuu_inv
        rhs       = noise_inv * (A.T @ y)

        try:
            Su_new    = np.linalg.inv(Lambda + np.eye(m) * 1e-4)
        except np.linalg.LinAlgError:
            return

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

        Ksu = self.kernel(Xs, self.Z)
        A   = Ksu @ self._Kuu_inv

        mean          = A @ self.mu
        Kss_diag      = self.kernel.diag(Xs)
        var_explained = np.sum(A * (A @ self._Kuu), axis=1)
        var_var       = np.sum(A @ self.Su * A, axis=1)
        var  = Kss_diag - var_explained + var_var + self.noise_var
        var  = np.clip(var, 1e-6, None)

        return mean, var


# ╔════════════════════════════════════════════════════════════════════════════╗
# §4  PERCEPTION MODULE
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class PerceptionOutput:
    alpha_grid     : np.ndarray
    beta_grid      : np.ndarray
    occupancy_mean : np.ndarray # Normalized [0, 1] for classification
    occupancy_var  : np.ndarray
    slope_map      : np.ndarray
    traversability : np.ndarray
    raw_points     : np.ndarray
    cell_kinds     : np.ndarray
    free_mask      : Optional[np.ndarray] = None
    mean_surface   : Optional[np.ndarray] = None # Raw GP mean


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
        # FIX: Copy data to avoid race conditions with CARLA thread
        raw  = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
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

        d_alpha = np.gradient(mean, axis=1)
        d_beta  = np.gradient(mean, axis=0)
        slope   = np.sqrt(d_alpha ** 2 + d_beta ** 2)
        slope   = slope / (np.max(slope) + 1e-6)

        # FIX: Normalize Mean for Classification (0=Free, 1=Obstacle)
        mean_range = mean.max() - mean.min()
        occ_norm   = (mean - mean.min()) / (mean_range + 1e-6)
        
        var_range  = var.max() - var.min()
        var_norm   = (var - var.min()) / (var_range + 1e-6)

        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT   * occ_norm
            - self.SLOPE_WEIGHT * slope
            + self.VAR_FREE_WEIGHT * (1.0 - var_norm),
            0.0, 1.0,
        )
        free_mask  = traversability > 0.5

        cell_kinds = classify_cells(occ_norm, var, self.cfg)

        out = PerceptionOutput(
            alpha_grid     = self._AG,
            beta_grid      = self._BG,
            occupancy_mean = occ_norm, # Pass normalized for classifier
            occupancy_var  = var,
            slope_map      = slope,
            traversability = traversability,
            raw_points     = pts,
            cell_kinds     = cell_kinds,
            free_mask      = free_mask,
            mean_surface   = mean,     # Keep raw for other logic if needed
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
    alpha     : float
    beta      : float
    distance  : float
    local_pos : np.ndarray
    slope_deg : float
    safe      : bool
    width_m   : float
    cost      : float = 0.0
    world_pos : Optional[np.ndarray] = None
    certainty       : float = 0.0
    is_free_certain : bool  = False
    is_obstacle     : bool  = False
    traversability  : float = 0.5


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6  SUBGOAL PLANNER  — Certainty-Aware Flattest-Minima
# ╚════════════════════════════════════════════════════════════════════════════╝

class SubgoalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg               = cfg
        self._alpha_history    : deque = deque(maxlen=8)
        self._exploration_bias : float = 0.0

    def plan(
        self,
        perc              : PerceptionOutput,
        vehicle_pitch_rad : float,
        mode              : str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        best = self.plan_from_variance(perc, vehicle_pitch_rad, mode=mode)
        return best, []

    def plan_from_variance(
        self,
        perc              : PerceptionOutput,
        vehicle_pitch_rad : float,
        mode              : str = "forward",
    ) -> Optional[Subgoal]:
        if perc.occupancy_var is None:
            return None

        var       = perc.occupancy_var
        occ       = perc.occupancy_mean
        slope_map = perc.slope_map
        trav      = perc.traversability
        kinds     = perc.cell_kinds
        alpha_row = perc.alpha_grid[0]

        var_col   = np.mean(var,       axis=0)
        occ_col   = np.mean(occ,       axis=0)
        slope_col = np.mean(slope_map, axis=0)
        trav_col  = np.mean(trav,      axis=0)

        vmin, vmax  = var_col.min(), var_col.max()
        v_range = vmax - vmin
        cert_col    = 1.0 - (var_col - vmin) / (v_range + 1e-9)

        def _col_kind(ci: int) -> CellKind:
            col_kinds = list(kinds[:, ci].ravel())
            counts    = {
                CellKind.CERTAIN_OBSTACLE: col_kinds.count(CellKind.CERTAIN_OBSTACLE),
                CellKind.CERTAIN_FREE    : col_kinds.count(CellKind.CERTAIN_FREE),
                CellKind.UNCERTAIN       : col_kinds.count(CellKind.UNCERTAIN),
            }
            return max(counts, key=counts.get)

        if mode == "forward":
            dir_mask = (alpha_row > -np.pi / 2) & (alpha_row < np.pi / 2)
        else:
            dir_mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        columns = []
        for ci, a in enumerate(alpha_row):
            if not dir_mask[ci]:
                continue
            ck = _col_kind(ci)
            if ck == CellKind.CERTAIN_OBSTACLE:
                continue
            columns.append({
                "ci"       : ci,
                "alpha"    : float(a),
                "var"      : float(var_col[ci]),
                "occ"      : float(occ_col[ci]),
                "slope"    : float(slope_col[ci]),
                "trav"     : float(trav_col[ci]),
                "cert"     : float(cert_col[ci]),
                "kind"     : ck,
            })

        if len(columns) < 3:
            return self._fallback_subgoal(perc, mode, vehicle_pitch_rad)

        segments : List[List[dict]] = []
        current  : List[dict]       = []
        trav_thresh  = 0.35
        slope_thresh = 0.55

        for col in columns:
            if col["trav"] >= trav_thresh and col["slope"] <= slope_thresh:
                current.append(col)
            else:
                if len(current) >= 3:
                    segments.append(current)
                current = []
        if len(current) >= 3:
            segments.append(current)

        if not segments:
            for col in columns:
                if col["trav"] >= 0.20 and col["slope"] <= 0.75:
                    current.append(col)
                else:
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
            if len(current) >= 2:
                segments.append(current)

        if not segments:
            return self._fallback_subgoal(perc, mode, vehicle_pitch_rad)

        best_subgoal : Optional[Subgoal] = None
        best_cost    : float             = float("inf")
        pitch_deg    = abs(math.degrees(vehicle_pitch_rad))

        for seg in segments:
            n_seg       = len(seg)
            α_vals      = np.array([c["alpha"] for c in seg])
            slope_vals  = np.array([c["slope"] for c in seg])
            trav_vals   = np.array([c["trav"]  for c in seg])
            cert_vals   = np.array([c["cert"]  for c in seg])
            kind_vals   = [c["kind"] for c in seg]

            α_center     = float(np.mean(α_vals))
            α_span       = float(α_vals[-1] - α_vals[0])
            width_m = self.cfg.subgoal_distance * abs(α_span)
            eff_width = width_m * max(np.cos(α_center), 0.1)
            req_w = self.cfg.vehicle_width + self.cfg.safety_margin
            if eff_width < req_w:
                continue

            d = self.cfg.subgoal_distance
            lx_c = d * np.cos(α_center)
            ly_c = d * np.sin(α_center)
            if not self._are_wheels_stable(perc, lx_c, ly_c, α_center):
                continue

            C_rollout = self._rollout_cost(perc, α_center, d)
            if C_rollout > 900:
                continue

            C_flatness = float(np.mean(slope_vals) + 0.5 * np.std(slope_vals))

            n_free = sum(1 for k in kind_vals if k == CellKind.CERTAIN_FREE)
            n_unc = sum(1 for k in kind_vals if k == CellKind.UNCERTAIN)
            cert_reward = (
                self.cfg.w_certain_free  * (n_free / n_seg)
                + self.cfg.w_uncertain_free * (n_unc  / n_seg)
            )

            C_dir       = abs(α_center) / (np.pi / 2)
            C_clearance = 1.0 / (eff_width - req_w + 0.5)
            C_pitch = pitch_deg / (self.cfg.max_slope_deg + 1e-6)
            C_trav      = 1.0 - float(np.mean(trav_vals))

            C_osc = 0.0
            if len(self._alpha_history) >= 4:
                hist = np.array(list(self._alpha_history))
                if np.std(hist) > 0.3:
                    C_osc = 0.5 * abs(α_center - float(np.mean(hist)))

            J = (
                self.cfg.w_flatness   * C_flatness
                + self.cfg.w_steepness * float(np.mean(slope_vals))
                + 1.2                  * C_dir
                + 1.5                  * C_clearance
                + 1.0                  * C_trav
                + 1.5                  * C_pitch
                + 0.8                  * C_osc
                + 2.0                  * C_rollout
                - cert_reward
            )

            if self._exploration_bias > 0:
                J *= (1.0 - self._exploration_bias * 0.4)

            lx  = d * np.cos(α_center)
            ly  = d * np.sin(α_center)
            avg_cert = float(np.mean(cert_vals))

            sg = Subgoal(
                alpha           = α_center,
                beta            = 0.0,
                distance        = d,
                local_pos       = np.array([lx, ly, 0.0]),
                slope_deg       = float(np.degrees(np.arctan(np.mean(slope_vals)))),
                safe            = True,
                width_m         = width_m,
                cost            = J,
                certainty       = avg_cert,
                is_free_certain = n_free > n_seg * 0.5,
                is_obstacle     = False,
                traversability  = float(np.mean(trav_vals)),
            )

            if J < best_cost:
                best_cost    = J
                best_subgoal = sg

        if best_subgoal is None:
            return self._fallback_subgoal(perc, mode, vehicle_pitch_rad)

        self._alpha_history.append(best_subgoal.alpha)
        return best_subgoal

    def find_recovery_direction(self, perc: PerceptionOutput) -> Optional[float]:
        alpha_row = perc.alpha_grid[0]
        occ_col   = np.mean(perc.occupancy_mean, axis=0)
        var_col   = np.mean(perc.occupancy_var,  axis=0)
        # kinds     = perc.cell_kinds # Unused in this simplified version

        rear_mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        def _col_kind(ci: int) -> CellKind:
            col_kinds = list(perc.cell_kinds[:, ci].ravel())
            counts = {k: col_kinds.count(k) for k in CellKind}
            return max(counts, key=counts.get)

        cert_free_cols   = []
        uncertain_cols   = []
        all_rear_cols    = []

        for ci, a in enumerate(alpha_row):
            if not rear_mask[ci]:
                continue
            ck = _col_kind(ci)
            if ck == CellKind.CERTAIN_OBSTACLE:
                continue
            entry = (float(a), float(occ_col[ci]), float(var_col[ci]))
            all_rear_cols.append(entry)
            if ck == CellKind.CERTAIN_FREE:
                cert_free_cols.append(entry)
            elif ck == CellKind.UNCERTAIN:
                uncertain_cols.append(entry)

        if cert_free_cols:
            return min(cert_free_cols, key=lambda e: e[1])[0]
        if uncertain_cols:
            return max(uncertain_cols, key=lambda e: e[2])[0]
        if all_rear_cols:
            return min(all_rear_cols, key=lambda e: e[1])[0]

        return None

    def set_exploration_bias(self, v: float) -> None:
        self._exploration_bias = float(np.clip(v, 0.0, 1.0))

    def _fallback_subgoal(
        self,
        perc: PerceptionOutput,
        mode: str,
        vehicle_pitch_rad: float,
    ) -> Optional[Subgoal]:
        alpha_row = perc.alpha_grid[0]
        trav_col  = np.mean(perc.traversability, axis=0)

        if mode == "forward":
            mask = (alpha_row > -np.pi / 2) & (alpha_row < np.pi / 2)
        else:
            mask = (alpha_row <= -np.pi / 2) | (alpha_row >= np.pi / 2)

        def _col_kind(ci: int) -> CellKind:
            col_kinds = list(perc.cell_kinds[:, ci].ravel())
            counts = {k: col_kinds.count(k) for k in CellKind}
            return max(counts, key=counts.get)

        candidates = [
            (ci, float(alpha_row[ci]), float(trav_col[ci]))
            for ci in range(len(alpha_row))
            if mask[ci] and _col_kind(ci) != CellKind.CERTAIN_OBSTACLE
        ]

        if not candidates:
            return None

        best    = max(candidates, key=lambda t: t[2])
        a       = best[1]
        d       = self.cfg.subgoal_distance
        lx, ly  = d * np.cos(a), d * np.sin(a)

        return Subgoal(
            alpha          = a,
            beta           = 0.0,
            distance       = d,
            local_pos      = np.array([lx, ly, 0.0]),
            slope_deg      = 0.0,
            safe           = False,
            width_m        = 0.5,
            cost           = 999.0,
            certainty      = 0.0,
            is_free_certain= False,
            traversability = best[2],
        )

    def _rollout_cost(self, perc: PerceptionOutput, a: float, d: float) -> float:
        steps      = 12
        total_cost = 0.0
        alpha_row  = perc.alpha_grid[0]
        n_beta     = perc.beta_grid.shape[0]
        bi_mid     = n_beta // 2

        for i in range(1, steps + 1):
            frac = i / steps
            x    = frac * d * np.cos(a)
            y    = frac * d * np.sin(a)

            alpha_pt = np.arctan2(y, x)
            # FIX: Safe index calculation
            ai = int(np.argmin(np.abs(alpha_row - alpha_pt)))
            ai       = np.clip(ai, 0, len(alpha_row) - 1)
            bi       = bi_mid
            bi = np.clip(bi, 0, perc.slope_map.shape[0] - 1)

            ck = perc.cell_kinds[bi, ai]
            if ck == CellKind.CERTAIN_OBSTACLE:
                return 999.0

            slope = float(perc.slope_map[bi, ai])
            var   = float(perc.occupancy_var[bi, ai])

            if ck == CellKind.CERTAIN_FREE:
                total_cost += slope * 1.5 - 0.3
            elif ck == CellKind.UNCERTAIN:
                total_cost += slope * 2.0 + var * 1.0
            else:
                total_cost += slope * 2.0 + var * 1.5

        return max(total_cost / steps, 0.0)

    def _are_wheels_stable(
        self,
        perc  : PerceptionOutput,
        cx    : float,
        cy    : float,
        yaw   : float,
    ) -> bool:
        wb = self.cfg.wheel_base  / 2.0
        tw = self.cfg.track_width / 2.0

        wheels = np.array([[wb, tw], [wb, -tw], [-wb, tw], [-wb, -tw]])
        R      = np.array([[np.cos(yaw), -np.sin(yaw)],
                           [np.sin(yaw),  np.cos(yaw)]])
        wheels  = wheels @ R.T
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
        x   : float,
        y   : float,
    ) -> bool:
        r = math.sqrt(x * x + y * y)
        if r < 1e-3:
            return True

        alpha = math.atan2(y, x)
        # FIX: Safe index calculation
        ai = int(np.argmin(np.abs(perc.alpha_grid[0] - alpha)))
        ai    = np.clip(ai, 0, perc.alpha_grid.shape[1] - 1)
        bi    = perc.beta_grid.shape[0] // 2
        bi    = np.clip(bi, 0, perc.cell_kinds.shape[0] - 1)

        return perc.cell_kinds[bi, ai] == CellKind.CERTAIN_OBSTACLE


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §7  CONTROLLER MODULE
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

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

    def compute(
        self,
        subgoal           : Optional[Subgoal],
        vehicle_speed_ms  : float,
        terrain_slope_deg : float = 0.0,
        roll_deg          : float = 0.0,
        pitch_deg         : float = 0.0,
    ) -> ControlState:
        if subgoal is None:
            return self._apply_smooth(0.0, 0.0, 0.0, roll_deg=roll_deg, pitch_deg=pitch_deg)

        alpha = subgoal.alpha
        slope = subgoal.slope_deg

        trav_factor = float(np.clip(
            subgoal.traversability * (1.2 if subgoal.is_free_certain else 1.0),
            0.3, 1.0,
        ))

        speed_factor = max(
            0.3,
            trav_factor
            - slope  / self.cfg.max_slope_deg * 0.4
            - abs(alpha) / (math.pi / 2)      * 0.25,
        )

        vx_desired    = self.cfg.base_speed * speed_factor
        vy_desired = -math.sin(alpha) * self.cfg.base_speed * 0.15
        omega_desired = alpha * 1.6

        ws = self.mec.inverse(vx_desired, vy_desired, omega_desired)
        vx_a, vy_a, om_a = self.mec.forward(ws)

        return self._apply_smooth(vx_a, vy_a, om_a, roll_deg=roll_deg, pitch_deg=pitch_deg)

    def compute_manual(
        self, vx: float, vy: float, omega: float,
    ) -> ControlState:
        ws               = self.mec.inverse(vx, vy, omega)
        vx_a, vy_a, om_a = self.mec.forward(ws)
        return self._apply_smooth(vx_a, vy_a, om_a)

    def compute_recovery(self, phase: int, roll_deg: float, pitch_deg: float) -> ControlState:
        """FIX: Route through smoothing to prevent jerk."""
        steers = [0.0, 0.5, -0.5]
        steer  = steers[min(phase, 2)]
        # Use _apply_smooth to respect rate limits and stability modifiers
        return self._apply_smooth(
            0.0, 0.0, steer * 2.0,  # Omega approx
            roll_deg=roll_deg, pitch_deg=pitch_deg,
        )

    def _stability_modifiers(
        self, roll_deg: float, pitch_deg: float
    ) -> Tuple[float, float]:
        throttle_scale   = 1.0
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
        vx: float, vy: float, omega: float,
        roll_deg: float = 0.0, pitch_deg: float = 0.0,
    ) -> ControlState:
        a            = self.cfg.ema_alpha
        self._vx_ema = a * vx    + (1 - a) * self._vx_ema
        self._vy_ema = a * vy    + (1 - a) * self._vy_ema
        self._om_ema = a * omega + (1 - a) * self._om_ema

        throttle, steer, brake = MecanumKinematics.to_carla_control(
            self._vx_ema, self._vy_ema, self._om_ema, self.cfg,
        )

        steer    = float(np.clip(steer,
                                  self._state.steer - self.cfg.max_steer_rate,
                                  self._state.steer + self.cfg.max_steer_rate))
        throttle = float(np.clip(throttle,
                                  self._state.throttle - self.cfg.max_throttle_rate,
                                  self._state.throttle + self.cfg.max_throttle_rate))

        thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
        throttle *= thr_scale
        steer     = float(np.clip(steer + cor_steer,
                                   -self.cfg.max_steer, self.cfg.max_steer))

        # FIX: Determine reverse flag based on vx direction
        reverse = (self._vx_ema < -0.1)

        self._state = ControlState(throttle=throttle, steer=steer, brake=brake, reverse=reverse)
        return self._state

    def reset_ema(self) -> None:
        self._vx_ema = self._vy_ema = self._om_ema = 0.0


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §8  STUCK DETECTION & VSGP-GUIDED RECOVERY
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

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

        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    self._trigger_recovery(t)
                    return True, 0

        if len(self._steer_history) == self._steer_history.maxlen:
            arr   = np.array(self._steer_history)
            signs = np.sign(arr)
            flips = int(np.sum(np.diff(signs) != 0))
            if flips > len(signs) * 0.70:
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

    def push_steer(self, steer: float) -> None:
        self._steer_history.append(steer)

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §9  VISUALIZATION MODULE
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        # FIX: Check for backend availability
        try:
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
        except Exception as e:
            print(f"[WARN] Visualization init failed: {e}")
            self.fig = None

        maxlen = cfg.traj_history
        self._cost_total : deque = deque(maxlen=maxlen)
        self._cost_flat  : deque = deque(maxlen=maxlen)
        self._cost_cert  : deque = deque(maxlen=maxlen)
        self._vel_lin    : deque = deque(maxlen=maxlen)
        self._vel_ang    : deque = deque(maxlen=maxlen)
        self._traj_x     : deque = deque(maxlen=maxlen)
        self._traj_y     : deque = deque(maxlen=maxlen)
        self._subgoal_x  : deque = deque(maxlen=20)
        self._subgoal_y  : deque = deque(maxlen=20)
        self._subgoal_certain : deque = deque(maxlen=20)

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
            self._cost_flat.append(
                subgoal.slope_deg / (CFG.max_slope_deg + 1e-6) 
* CFG.w_flatness)
            self._cost_cert.append(subgoal.certainty)
            self._subgoal_x.append(vehicle_pos[0] + subgoal.local_pos[0])
            self._subgoal_y.append(vehicle_pos[1] + subgoal.local_pos[1])
            self._subgoal_certain.append(subgoal.is_free_certain)
        else:
            self._cost_total.append(0)
            self._cost_flat.append(0)
            self._cost_cert.append(0)

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
                f"VSGP Certainty-Aware Navigator  |  Mode: {self._mode_text}",
                color="#eee", fontsize=11, y=0.99,
            )
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            # FIX: Allow GUI events to process
            plt.pause(0.001) 
        except Exception:
            pass

    def _draw_camera(self, ax, img, title):
        ax.clear(); ax.set_facecolor("#1a1a2e")
        ax.set_title(title, fontsize=8); ax.axis("off")
        if img is not None:
            ax.imshow(img)

    def _draw_lidar(self):
        ax = self.ax_lidar
        ax.clear(); ax.set_facecolor("#1a1a2e")
        ax.set_title("LiDAR (top view)", fontsize=8)
        if self._lidar_pts is not None and len(self._lidar_pts) > 0:
            pts  = self._lidar_pts
            z    = pts[:, 2]
            z_n  = (z - z.min()) / (z.max() - z.min() + 1e-9)
            ax.scatter(pts[:, 1], pts[:, 0], c=z_n, cmap="plasma", s=0.5, alpha=0.6)
            ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
            ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
        ax.set_xlabel("Y [m]", fontsize=7); ax.set_ylabel("X [m]", fontsize=7)

    def _draw_cost(self):
        ax = self.ax_cost
        ax.clear(); ax.set_facecolor("#1a1a2e")
        ax.set_title("Cost / Certainty", fontsize=8)
        n = len(self._cost_total)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._cost_total), c="#ff6b6b", lw=1.2, label="Total cost")
        ax.plot(xs, list(self._cost_flat), c="#74b9ff", lw=0.9, label="Flatness cost")
        ax.plot(xs, list(self._cost_cert), c="#55efc4", lw=0.9, label="Certainty")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7); ax.set_ylabel("Value", fontsize=7)

    def _draw_trajectory(self):
        ax = self.ax_traj
        ax.clear(); ax.set_facecolor("#1a1a2e")
        ax.set_title("Trajectory + Subgoals", fontsize=8)
        tx, ty = list(self._traj_x), list(self._traj_y)
        if len(tx) > 1:
            ax.plot(ty, tx, c="#00cec9", lw=1.2, label="Actual")
        if len(tx) > 0:
            ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)
        sx  = list(self._subgoal_x)
        sy  = list(self._subgoal_y)
        sc  = list(self._subgoal_certain)
        if sx:
            colors = ["#00b894" if c else "#e17055" for c in sc]
            ax.scatter(sy, sx, c=colors, s=14, marker="x", zorder=4,
                       label="Subgoal (green=certain)")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Y [m]", fontsize=7); ax.set_ylabel("X [m]", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    def _draw_velocity(self):
        ax = self.ax_vel
        ax.clear(); ax.set_facecolor("#1a1a2e")
        ax.set_title("Velocities", fontsize=8)
        n = len(self._vel_lin)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._vel_lin), c="#55efc4", lw=1.2, label="Linear (m/s)")
        ax.plot(xs, list(self._vel_ang), c="#a29bfe", lw=1.0, label="ω·v (m/s)")
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7); ax.set_ylabel("m/s", fontsize=7)

    def close(self) -> None:
        if self.fig:
            plt.close(self.fig)


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §10  CARLA SENSOR WRAPPERS
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self.vehicle = vehicle
        self.world   = world
        self._actors : list = []
        self._active : bool = True

        self._lidar_data  = None
        self._front_image : Optional[np.ndarray] = None
        self._rear_image  : Optional[np.ndarray] = None

        bp_lib = world.get_blueprint_library()

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
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
        if self._active: self._lidar_data  = data
    def _on_front_image(self, image) -> None:
        if not self._active: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._front_image = arr[:, :, ::-1].copy()

    def _on_rear_image(self, image)  -> None:
        if not self._active: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._rear_image = arr[:, :, ::-1].copy()

    @property
    def lidar_data(self):              return self._lidar_data
    @property
    def front_image(self):             return self._front_image
    @property
    def rear_image(self):              return self._rear_image

    def destroy(self) -> None:
        self._active = False
        for a in self._actors:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §11  MANUAL CONTROLLER
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class ManualController:
    def __init__(self, cfg: Config = CFG):
        self.cfg     = cfg
        self._rev    = False   
        self._prev_r = False   # FIX: Renamed from _prev_s for clarity

        self._thr_ema = 0.0
        self._str_ema = 0.0
        self._EMA     = 0.30

    def tick(self, keys) -> ControlState:
        boost = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        if keys[pygame.K_r] and not self._prev_r:
            self._rev = not self._rev
            self._prev_r = True
        elif not keys[pygame.K_r]:
            self._prev_r = False

        thr_max = self.cfg.manual_throttle * (1.4 if boost else 1.0)
        rev_max = self.cfg.manual_reverse_throttle

        throttle = 0.0
        brake    = 0.0
        reverse  = self._rev

        if keys[pygame.K_SPACE]:
            brake    = self.cfg.manual_brake
            throttle = 0.0
        elif keys[pygame.K_w]:
            throttle = thr_max if not reverse else 0.0
            brake    = 0.0
        elif keys[pygame.K_s]:
            if reverse:
                throttle = rev_max
            else:
                brake = self.cfg.manual_brake * 0.6

        if reverse and keys[pygame.K_w]:
            self._rev = False
            reverse   = False
            throttle  = thr_max

        str_max = self.cfg.manual_steer * (1.2 if boost else 1.0)
        steer   = 0.0
        if keys[pygame.K_a]:
            steer = -str_max
        elif keys[pygame.K_d]:
            steer =  str_max

        if reverse:
            steer = -steer

        a             = self._EMA
        self._thr_ema = a * throttle + (1 - a) * self._thr_ema
        self._str_ema = a * steer    + (1 - a) * self._str_ema

        final_brake = brake if brake > 0 else 0.0

        return ControlState(
            throttle = float(np.clip(self._thr_ema, 0.0, self.cfg.max_throttle)),
            steer    = float(np.clip(self._str_ema, -self.cfg.max_steer, self.cfg.max_steer)),
            brake    = float(np.clip(final_brake, 0.0, self.cfg.max_brake)),
            reverse  = reverse,
        )

    def reset(self) -> None:
        self._rev     = False
        self._thr_ema = 0.0
        self._str_ema = 0.0


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §12  VEHICLE SPAWNER
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)
    
    if vehicle_bp is None:
        # Fallback
        vehicle_bp = bp_lib.find("vehicle.lincoln.mkz2017")

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


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §13  MAIN NAVIGATION LOOP
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._mode    = "AUTO"

        self._client  = None
        self._world   = None
        self._vehicle = None
        self._sensors : Optional[SensorManager] = None

        self._perception  = PerceptionModule(cfg)
        self._planner     = SubgoalPlanner(cfg)
        self._controller  = Controller(cfg)
        self._manual      = ManualController(cfg)
        self._stuck       = StuckDetector(cfg)
        self._viz         = VisualizationModule(cfg)

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle AUTO/MANUAL, R=reverse, ESC=quit")
        self._screen = pygame.display.set_mode((260, 112))
        self._font   = pygame.font.SysFont("monospace", 12)

        self._step = 0
        self._t0   = 0.0

    def connect(self) -> None:
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port} …")
        try:
            self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
            self._client.set_timeout(self.cfg.carla_timeout)
            self._world  = self._client.get_world()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA: {e}")
            
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
        print("[SETUP] All sensors attached.  Starting navigation loop.")

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

        t_now      = time.time() - self._t0
        self._step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False; return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False; return
                if event.key == pygame.K_TAB:
                    self._mode = "MANUAL" if self._mode == "AUTO" else "AUTO"
                    self._controller.reset_ema()
                    self._manual.reset()
                    print(f"[MODE] Switched to {self._mode}")

        keys = pygame.key.get_pressed()

        loc       = self._vehicle.get_location()
        vel       = self._vehicle.get_velocity()
        speed     = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos       = np.array([loc.x, loc.y, loc.z])
        transform = self._vehicle.get_transform()
        pitch_rad = math.radians(transform.rotation.pitch)
        roll_rad  = math.radians(transform.rotation.roll)
        pitch_deg = math.degrees(pitch_rad)
        roll_deg  = math.degrees(roll_rad)

        perc : Optional[PerceptionOutput] = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        best_goal  : Optional[Subgoal] = None
        ctrl_state : ControlState

        if self._mode == "MANUAL":
            ctrl_state = self._manual.tick(keys)

        else:  # AUTO
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                self._planner.set_exploration_bias(0.9)

                recovery_alpha : Optional[float] = None
                if perc is not None:
                    recovery_alpha = self._planner.find_recovery_direction(perc)

                if recovery_alpha is not None:
                    d   = self.cfg.subgoal_distance
                    lx  = -abs(d * math.cos(recovery_alpha))
                    ly  = d * math.sin(recovery_alpha)

                    best_goal = Subgoal(
                        alpha           = recovery_alpha,
                        beta            = 0.0,
                        distance        = d,
                        local_pos       = np.array([lx, ly, 0.0]),
                        slope_deg       = 0.0,
                        safe            = True,
                        width_m         = 1.0,
                        cost            = 0.0,
                        certainty       = 0.5,
                        is_free_certain = False,
                        traversability  = 0.5,
                    )
                    
                    # FIX: Use Controller for smoothing even in recovery
                    # We simulate a subgoal that implies reverse motion
                    # But since Controller.compute assumes forward, we use compute_recovery
                    ctrl_state = self._controller.compute_recovery(rec_phase, roll_deg, pitch_deg)
                    ctrl_state.reverse = True  # Force reverse flag

                    print(f"[RECOVERY] VSGP α={recovery_alpha:.2f} "
                          f"steer={ctrl_state.steer:.2f}")
                else:
                    ctrl_state = self._controller.compute_recovery(rec_phase, roll_deg, pitch_deg)
                    ctrl_state.reverse = True

            else:
                self._planner.set_exploration_bias(0.0)

                if perc is not None:
                    best_goal, _ = self._planner.plan(
                        perc, pitch_rad, mode="forward"
                    )

                slope = best_goal.slope_deg if best_goal else 0.0
                ctrl_state = self._controller.compute(
                    best_goal, speed, slope,
                    roll_deg=roll_deg, pitch_deg=pitch_deg,
                )

            self._stuck.push_steer(ctrl_state.steer)

        self._vehicle.apply_control(carla.VehicleControl(
            throttle = float(ctrl_state.throttle),
            steer    = float(ctrl_state.steer),
            brake    = float(ctrl_state.brake),
            reverse  = ctrl_state.reverse,
        ))

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


# ╔════════════════════════════════════════════════════════════╔════════════════════════════════════════════════════════════════════════════╗
# §14  ENTRY POINT
# ╚════════════════════════════════════════════════════════════╚════════════════════════════════════════════════════════════════════════════╝

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
