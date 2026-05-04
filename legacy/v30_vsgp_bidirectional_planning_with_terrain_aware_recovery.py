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
import pygame
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# Optional scipy / sklearn imports
# ─────────────────────────────────────────────────────────────
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
    print("[FATAL] carla Python package not found. Make sure CARLA's egg is on PYTHONPATH.")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    carla_timeout: float = 20.0
    synchronous: bool = True
    fixed_delta_seconds: float = 0.05

    vehicle_blueprint: str = "vehicle.tesla.cybertruck"
    spawn_z_offset: float = 5.0
    post_spawn_wait_s: float = 3.0

    lidar_range: float = 20.0
    lidar_points_per_sec: int = 100_000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int = 64
    lidar_upper_fov: float = 15.0
    lidar_lower_fov: float = -25.0

    camera_width: int = 320
    camera_height: int = 240
    camera_fov: int = 90

    wheel_radius: float = 0.15
    lx: float = 0.25
    ly: float = 0.20
    max_wheel_omega: float = 40.0

    robot_width: float = 0.8
    safety_margin: float = 0.4
    max_slope_deg: float = 20.0
    subgoal_distance: float = 4.0
    n_subgoals_max: int = 8

    w_direction: float = 1.2
    w_distance: float = 0.6
    w_steepness: float = 1.5
    w_collision: float = 3.0
    w_oscillation: float = 0.8
    w_flatness: float = 0.5

    max_throttle: float = 0.55
    max_steer: float = 0.6
    max_brake: float = 0.8
    ema_alpha: float = 0.35
    max_steer_rate: float = 0.12
    max_throttle_rate: float = 0.08
    base_speed: float = 4.0

    stuck_window_s: float = 4.0
    stuck_disp_thresh: float = 0.6
    recovery_duration_s: float = 2.5

    vsgp_n_inducing: int = 64
    vsgp_alpha: float = 1.0
    vsgp_length_scale: float = 0.3
    vsgp_noise_var: float = 0.05
    vsgp_lr: float = 0.02

    viz_update_hz: float = 5.0
    traj_history: int = 400


CFG = Config()


# ╔════════════════════════════════════════════════════════════════════════╗
# §1  MECANUM KINEMATIC MODEL
# ╚════════════════════════════════════════════════════════════════════════╝

class MecanumKinematics:
    """
    Standard four-mecanum-wheel kinematic model.
    """

    def __init__(self, cfg: Config = CFG):
        rw = cfg.wheel_radius
        L  = cfg.lx + cfg.ly          # geometric constant

        # ── Forward kinematics matrix (3×4)
        self.Jfwd = (rw / 4.0) * np.array([
            [ 1,  1,  1,  1],
            [-1,  1,  1, -1],
            [-1/L, 1/L, -1/L, 1/L],
        ], dtype=np.float64)

        # ── Inverse kinematics matrix (4×3)
        self.Jinv = (1.0 / rw) * np.array([
            [ 1, -1, -L],
            [ 1,  1,  L],
            [ 1,  1, -L],
            [ 1, -1,  L],
        ], dtype=np.float64)

        self.cfg = cfg

    def forward(self, wheel_omegas: np.ndarray) -> Tuple[float, float, float]:
        """wheel speeds → body twist (vx, vy, ω)"""
        twist = self.Jfwd @ wheel_omegas
        return float(twist[0]), float(twist[1]), float(twist[2])

    def inverse(self, vx: float, vy: float, omega: float) -> np.ndarray:
        """body twist → wheel speeds (4,)"""
        twist = np.array([vx, vy, omega], dtype=np.float64)
        ws = self.Jinv @ twist
        ws = np.clip(ws, -self.cfg.max_wheel_omega, self.cfg.max_wheel_omega)
        return ws

    @staticmethod
    def to_carla_control(
        vx: float,
        vy: float,
        omega: float,
        cfg: Config = CFG,
    ) -> Tuple[float, float, float]:
        """
        Map mecanum body twist to CARLA (throttle, steer, brake).
        """
        speed    = math.sqrt(vx ** 2 + vy ** 2)
        throttle = np.clip(speed / cfg.base_speed, 0.0, cfg.max_throttle)
        brake    = 0.0

        if vx < -0.1:                   # reversing
            throttle = np.clip(-vx / cfg.base_speed, 0.0, cfg.max_throttle)
            brake    = 0.0

        steer_omega   = omega / (cfg.base_speed + 1e-6)
        steer_lateral = -vy / (cfg.base_speed + 1e-6) * 0.4
        steer = np.clip(steer_omega + steer_lateral,
                        -cfg.max_steer, cfg.max_steer)

        return float(throttle), float(steer), float(brake)


# ╔════════════════════════════════════════════════════════════════════════╗
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚════════════════════════════════════════════════════════════════════════╝

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
        self.cfg    = cfg
        self.kernel = RationalQuadraticKernel(
            alpha        = cfg.vsgp_alpha,
            length_scale = cfg.vsgp_length_scale,
            variance     = 1.0,
        )
        self.noise_var = cfg.vsgp_noise_var
        self.n_ind     = cfg.vsgp_n_inducing
        self.lr        = cfg.vsgp_lr

        a_vals = np.linspace(-np.pi / 2, np.pi / 2, int(math.sqrt(self.n_ind)))
        b_vals = np.linspace(-np.pi / 4, np.pi / 4, int(math.sqrt(self.n_ind)))
        A, B   = np.meshgrid(a_vals, b_vals)
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

        self.mu = np.zeros(self.n_ind)
        self.Su = np.eye(self.n_ind) * 0.1

        self._Kuu       : Optional[np.ndarray] = None
        self._Kuu_inv   : Optional[np.ndarray] = None
        self._trained   : bool = False
        self._update_count: int = 0
        self._n_ind_actual: int = self.n_ind

    def _select_inducing_points(self, X: np.ndarray) -> None:
        actual_n = min(self.n_ind, len(X))

        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50, random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            idx    = np.random.choice(len(X), actual_n, replace=False)
            self.Z = X[idx].copy()

        n = len(self.Z)
        self.mu = np.zeros(n)
        self.Su = np.eye(n) * 0.1
        self._n_ind_actual = n

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return

        self._update_count += 1

        if self._update_count % 50 == 0:
            self._select_inducing_points(X)

        m = len(self.Z)
        Kuu     = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6
        Kfu     = self.kernel(X, self.Z)
        Kuu_inv = np.linalg.inv(Kuu)
        A       = Kfu @ Kuu_inv

        noise_inv = 1.0 / self.noise_var
        Lambda    = noise_inv * (A.T @ A) + Kuu_inv
        rhs       = noise_inv * (A.T @ y)

        Su_new = np.linalg.inv(Lambda + np.eye(m) * 1e-6)
        mu_new = Su_new @ rhs

        lr = self.lr
        self.Su = (1 - lr) * self.Su + lr * Su_new
        self.mu = (1 - lr) * self.mu + lr * mu_new

        self._Kuu     = Kuu
        self._Kuu_inv = Kuu_inv
        self._trained = True

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained or self._Kuu_inv is None:
            n = len(Xs)
            return np.zeros(n), np.ones(n)

        Ksu = self.kernel(Xs, self.Z)
        A   = Ksu @ self._Kuu_inv

        mean = A @ self.mu

        Kss_diag       = self.kernel.diag(Xs)
        var_explained  = np.sum(A * (A @ self._Kuu), axis=1)
        var_variational = np.sum(A @ self.Su * A, axis=1)
        var = Kss_diag - var_explained + var_variational + self.noise_var
        var = np.clip(var, 1e-6, None)

        return mean, var


# ╔════════════════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class PerceptionOutput:
    alpha_grid      : np.ndarray
    beta_grid       : np.ndarray
    occupancy_mean  : np.ndarray
    occupancy_var   : np.ndarray
    slope_map       : np.ndarray
    traversability  : np.ndarray
    raw_points      : np.ndarray
    free_mask       : Optional[np.ndarray] = None
    mean_surface    : Optional[np.ndarray] = None


class PerceptionModule:
    ALPHA_RES = 60
    BETA_RES  = 30
    ROC = 10.0
    VAR_FREE_WEIGHT = 1.0
    SLOPE_WEIGHT    = 1.2
    OCC_WEIGHT      = 1.0

    def __init__(self, cfg: Config = CFG):
        self.cfg  = cfg
        self.vsgp = VSGP(cfg)
        self._last_output = None

        self._a_lin = np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES)
        self._b_lin = np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES)

        AG, BG = np.meshgrid(self._a_lin, self._b_lin)
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG = AG
        self._BG = BG

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        raw = raw.reshape(-1, 4)[:, :3]

        if len(raw) < 20:
            return self._last_output

        dist = np.linalg.norm(raw, axis=1)
        mask = (dist > 0.5) & (dist < self.cfg.lidar_range)
        pts  = raw[mask]

        if len(pts) < 10:
            return self._last_output

        r     = np.linalg.norm(pts, axis=1)
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta  = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1, 1))

        X = np.column_stack([alpha, beta])
        y = self.ROC - r

        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
            pts  = pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)

        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var  = var.reshape(self.BETA_RES, self.ALPHA_RES)

        d_alpha = np.gradient(mean, axis=1)
        d_beta  = np.gradient(mean, axis=0)
        slope = np.sqrt(d_alpha ** 2 + d_beta ** 2)
        slope = slope / (np.max(slope) + 1e-6)

        mean_range = mean.max() - mean.min()
        occ_norm   = (mean - mean.min()) / (mean_range + 1e-6)

        var_range = var.max() - var.min()
        var_norm  = (var - var.min()) / (var_range + 1e-6)

        traversability = (
            1.0
            - self.OCC_WEIGHT   * occ_norm
            - self.SLOPE_WEIGHT * slope
            + self.VAR_FREE_WEIGHT * var_norm
        )
        traversability = np.clip(traversability, 0.0, 1.0)
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
    def last_output(self):
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════╗
# §4  SUBGOAL GENERATION (UNIFIED PIPELINE)
# ╚════════════════════════════════════════════════════════════════════════╝

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
    Implements Terrain-Aware + Recovery-Aware logic.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg               = cfg
        self._prev_alpha       : float = 0.0
        self._alpha_history    : deque = deque(maxlen=8)
        self._exploration_bias : float = 0.0

    def plan(self, perc: PerceptionOutput, vehicle_pitch_rad: float, mode: str = "forward") -> Tuple[Optional[Subgoal], List[Subgoal]]:
        """
        Unified planning pipeline:
        variance → segments → (generate + score locally) → best subgoal
        Supports 'forward' and 'reverse' modes for recovery.
        """
        if perc.occupancy_var is None:
            return None, []

        var       = perc.occupancy_var
        alpha     = perc.alpha_grid
        beta      = perc.beta_grid
        slope_map = perc.slope_map
        occ_mean  = perc.occupancy_mean

        # Focus on the middle elevation row (horizon)
        mid = var.shape[0] // 2
        
        # Safety check for array dimensions
        if mid >= var.shape[0]:
            mid = var.shape[0] - 1
            
        var_row   = var[mid]
        alpha_row = alpha[mid]
        beta_row  = beta[mid]
        slope_row = slope_map[mid]
        occ_row   = occ_mean[mid]

        # ── directional constraint (NEW) ─────────────────────────
        if mode == "forward":
            dir_mask = (alpha_row > -np.pi/2) & (alpha_row < np.pi/2)
        else:  # reverse mode
            dir_mask = (alpha_row <= -np.pi/2) | (alpha_row >= np.pi/2)

        # ── free space from variance ─────────────────────────────
        threshold = np.percentile(var_row, 60)
        mask = (var_row > threshold) & dir_mask

        best_subgoal = None
        best_cost    = float("inf")

        in_seg = False
        start = 0

        # Scan for contiguous segments
        for i in range(len(mask)):
            if mask[i] and not in_seg:
                start = i
                in_seg = True

            elif (not mask[i] or i == len(mask)-1) and in_seg:
                end = i
                in_seg = False

                # ── segment properties ───────────────────────────
                α_sb = alpha_row[start]
                α_se = alpha_row[end]

                width_alpha = α_se - α_sb
                width_m = self.cfg.subgoal_distance * abs(width_alpha)

                req_w = self.cfg.robot_width + self.cfg.safety_margin
                if width_m <= req_w:
                    continue

                # segment center (for corridor centering)
                α_center = 0.5 * (α_sb + α_se)

                # ── number of subgoals ───────────────────────────
                n = int(width_m // req_w)
                n = max(1, min(n, 5))

                alphas = np.linspace(α_sb, α_se, n)

                # ── evaluate each candidate ──────────────────────
                for a in alphas:
                    # Find closest index in row for slope/occ lookup
                    idx = np.argmin(np.abs(alpha_row - a))
                    
                    # Ensure idx is within bounds
                    if idx >= len(beta_row):
                        idx = len(beta_row) - 1

                    d = self.cfg.subgoal_distance
                    b = beta_row[idx]

                    lx = d * np.cos(b) * np.cos(a)
                    ly = d * np.cos(b) * np.sin(a)
                    lz = d * np.sin(b)

                    # ── TRUE slope (FIXED) ───────────────────────
                    slope_val = slope_row[idx]
                    slope_deg = slope_val * self.cfg.max_slope_deg * 1.5

                    # ── HARD terrain constraints ─────────────────
                    if slope_deg > (self.cfg.max_slope_deg - 5.0):
                        continue

                    if abs(lz) > 0.4:   # max step height
                        continue

                    # ── COSTS ───────────────────────────────────

                    # direction
                    C_dir = abs(a) / (np.pi / 2)

                    # distance
                    C_dst = 1.0 - (d / self.cfg.subgoal_distance)

                    # steepness (nonlinear)
                    s_norm = slope_deg / self.cfg.max_slope_deg
                    C_stp = np.exp(3.0 * s_norm) - 1.0

                    # obstacle proximity (wall avoidance)
                    occ_range = occ_row.max() - occ_row.min()
                    if occ_range > 1e-6:
                        occ_norm = (occ_row[idx] - occ_row.min()) / occ_range
                    else:
                        occ_norm = 0.0
                    C_obs = occ_norm

                    # corridor centering (stay away from walls)
                    C_center = abs(a - α_center) / (abs(width_alpha)/2 + 1e-6)

                    # uncertainty penalty
                    var_range = var_row.max() - var_row.min()
                    if var_range > 1e-6:
                        var_norm = (var_row[idx] - var_row.min()) / var_range
                    else:
                        var_norm = 0.0
                    C_uncertain = var_norm * s_norm

                    # vehicle pitch penalty
                    pitch_deg = abs(np.degrees(vehicle_pitch_rad))
                    C_pitch = pitch_deg / self.cfg.max_slope_deg

                    # ── FINAL COST ───────────────────────────────
                    J = (
                        self.cfg.w_direction * C_dir +
                        self.cfg.w_distance  * C_dst +
                        2.0 * C_stp +
                        2.0 * C_obs +
                        1.5 * C_center +
                        1.2 * C_uncertain +
                        1.5 * C_pitch
                    )
                    
                    # Apply exploration bias if stuck
                    if self._exploration_bias > 0:
                        J *= (1.0 - self._exploration_bias)

                    sg = Subgoal(
                        alpha=a,
                        beta=b,
                        distance=d,
                        local_pos=np.array([lx, ly, lz]),
                        slope_deg=slope_deg,
                        safe=True,
                        width_m=width_m,
                        cost=J,
                    )

                    if J < best_cost:
                        best_cost = J
                        best_subgoal = sg

        # ── Fallback (CRITICAL) ─────────────────────────────────
        if best_subgoal is None:
            # Pick highest variance point even if not in a 'segment'
            idx = np.argmax(var_row)
            a = alpha_row[idx]
            b = beta_row[idx]
            d = self.cfg.subgoal_distance

            lx = d * np.cos(b) * np.cos(a)
            ly = d * np.cos(b) * np.sin(a)
            lz = d * np.sin(b)

            best_subgoal = Subgoal(
                alpha=a,
                beta=b,
                distance=d,
                local_pos=np.array([lx, ly, lz]),
                slope_deg=0.0,
                safe=False,
                width_m=0.5,
                cost=999,
            )

        # Update history for oscillation check
        if best_subgoal:
            self._alpha_history.append(best_subgoal.alpha)

        return best_subgoal, [best_subgoal] if best_subgoal else []

    def set_exploration_bias(self, v: float) -> None:
        self._exploration_bias = np.clip(v, 0.0, 1.0)


# ╔════════════════════════════════════════════════════════════════════════╗
# §5  CONTROLLER MODULE
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle : float = 0.0
    steer    : float = 0.0
    brake    : float = 0.0
    reverse  : bool  = False


class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg    = cfg
        self.mec    = MecanumKinematics(cfg)
        self._state = ControlState()
        self._vx_ema = 0.0
        self._vy_ema = 0.0
        self._om_ema = 0.0

    def compute(
        self,
        subgoal          : Optional[Subgoal],
        vehicle_speed_ms : float,
        terrain_slope_deg: float = 0.0,
    ) -> ControlState:
        if subgoal is None:
            return self._apply_smooth(0.0, 0.0, 0.0)

        alpha = subgoal.alpha
        beta  = subgoal.beta
        slope = subgoal.slope_deg

        speed_factor = max(0.3,
                           1.0
                           - slope / self.cfg.max_slope_deg * 0.5
                           - abs(alpha) / (math.pi / 2) * 0.3)
        vx_desired = self.cfg.base_speed * speed_factor
        vy_desired = -math.sin(alpha) * self.cfg.base_speed * 0.2
        omega_desired = alpha * 1.8

        ws = self.mec.inverse(vx_desired, vy_desired, omega_desired)
        vx_ach, vy_ach, om_ach = self.mec.forward(ws)

        return self._apply_smooth(vx_ach, vy_ach, om_ach)

    def compute_manual(self, vx: float, vy: float, omega: float) -> ControlState:
        ws = self.mec.inverse(vx, vy, omega)
        vx_a, vy_a, om_a = self.mec.forward(ws)
        return self._apply_smooth(vx_a, vy_a, om_a)

    def compute_recovery(self, phase: int) -> ControlState:
        if phase == 0:
            return ControlState(throttle=0.3, steer=0.0,  brake=0.0, reverse=True)
        elif phase == 1:
            return ControlState(throttle=0.3, steer=0.4,  brake=0.0, reverse=True)
        else:
            return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

    def _apply_smooth(self, vx: float, vy: float, omega: float) -> ControlState:
        a = self.cfg.ema_alpha
        self._vx_ema = a * vx    + (1 - a) * self._vx_ema
        self._vy_ema = a * vy    + (1 - a) * self._vy_ema
        self._om_ema = a * omega + (1 - a) * self._om_ema

        throttle, steer, brake = MecanumKinematics.to_carla_control(
            self._vx_ema, self._vy_ema, self._om_ema, self.cfg
        )

        max_dr   = self.cfg.max_steer_rate
        max_dt   = self.cfg.max_throttle_rate
        steer    = np.clip(steer,
                           self._state.steer    - max_dr,
                           self._state.steer    + max_dr)
        throttle = np.clip(throttle,
                           self._state.throttle - max_dt,
                           self._state.throttle + max_dt)

        self._state = ControlState(throttle=throttle, steer=steer, brake=brake)
        return self._state

    def reset_ema(self) -> None:
        self._vx_ema = self._vy_ema = self._om_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════╗
# §6  STUCK DETECTION & RECOVERY
# ╚════════════════════════════════════════════════════════════════════════╝

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

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
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
            flips = np.sum(np.diff(signs) != 0)
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


# ╔════════════════════════════════════════════════════════════════════════╗
# §7  VISUALIZATION MODULE
# ╚════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        plt.ion()
        self.fig = plt.figure("VSGP Navigator", figsize=(16, 9))
        self.fig.patch.set_facecolor("#111")
        gs       = GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.3)

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
        self._cost_total  : deque = deque(maxlen=maxlen)
        self._cost_dir    : deque = deque(maxlen=maxlen)
        self._cost_dist   : deque = deque(maxlen=maxlen)
        self._cost_steep  : deque = deque(maxlen=maxlen)
        self._vel_lin     : deque = deque(maxlen=maxlen)
        self._vel_ang     : deque = deque(maxlen=maxlen)
        self._traj_x      : deque = deque(maxlen=maxlen)
        self._traj_y      : deque = deque(maxlen=maxlen)
        self._subgoal_x   : deque = deque(maxlen=20)
        self._subgoal_y   : deque = deque(maxlen=20)

        self._front_img  = None
        self._rear_img   = None
        self._lidar_pts  = None
        self._mode_text  = "AUTO"
        self._last_update = 0.0
        self._interval    = 1.0 / cfg.viz_update_hz

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
        self._traj_x.append(vehicle_pos[0])
        self._traj_y.append(vehicle_pos[1])

        if subgoal is not None:
            self._cost_total.append(subgoal.cost)
            dir_c  = abs(subgoal.alpha) / (math.pi / 2) * CFG.w_direction
            dist_c = (abs(subgoal.distance - CFG.subgoal_distance)
                      / CFG.subgoal_distance * CFG.w_distance)
            st_c   = subgoal.slope_deg / CFG.max_slope_deg * CFG.w_steepness
            self._cost_dir.append(dir_c)
            self._cost_dist.append(dist_c)
            self._cost_steep.append(st_c)

            sg_x = vehicle_pos[0] + subgoal.local_pos[0]
            sg_y = vehicle_pos[1] + subgoal.local_pos[1]
            self._subgoal_x.append(sg_x)
            self._subgoal_y.append(sg_y)
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

    def _draw_camera(self, ax: plt.Axes, img: Optional[np.ndarray], title: str) -> None:
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
            pts = self._lidar_pts
            z   = pts[:, 2]
            # Fix: handle ptp safely
            z_range = z.max() - z.min()
            z_n = (z - z.min()) / (z_range + 1e-9)
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
        ax.set_ylabel("Cost", fontsize=7)

    def _draw_trajectory(self) -> None:
        ax = self.ax_traj
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Trajectory", fontsize=8)
        tx = list(self._traj_x)
        ty = list(self._traj_y)
        if len(tx) > 1:
            ax.plot(ty, tx, c="#00cec9", lw=1.2, label="Actual")
        if len(tx) > 0:
            ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)

        sx = list(self._subgoal_x)
        sy = list(self._subgoal_y)
        if sx:
            ax.scatter(sy, sx, c="#e17055", s=12, marker="x", zorder=4, label="Subgoals")

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
        ax.set_ylabel("m/s", fontsize=7)

    def close(self) -> None:
        plt.close(self.fig)


# ╔════════════════════════════════════════════════════════════════════════╗
# §8  CARLA SENSOR WRAPPERS
# ╚════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self.vehicle = vehicle
        self.world   = world
        self._actors = []

        self._lidar_data  = None
        self._front_image = None
        self._rear_image  = None

        bp_lib = world.get_blueprint_library()

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",              str(cfg.lidar_range))
        lidar_bp.set_attribute("points_per_second",  str(cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("channels",           str(cfg.lidar_channels))
        lidar_bp.set_attribute("upper_fov",          str(cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov",          str(cfg.lidar_lower_fov))
        lidar_tf = carla.Transform(carla.Location(x=0.5, z=2.0))
        lidar    = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        lidar.listen(self._on_lidar)
        self._actors.append(lidar)

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
        self._front_image = arr[:, :, ::-1]

    def _on_rear_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._rear_image = arr[:, :, ::-1]

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
            if a.is_alive:
                a.destroy()


# ╔════════════════════════════════════════════════════════════════════════╗
# §9  MANUAL KEYBOARD CONTROLLER (Holonomic)
# ╚════════════════════════════════════════════════════════════════════════╝

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


# ╔════════════════════════════════════════════════════════════════════════╗
# §10  VEHICLE SPAWNER
# ╚════════════════════════════════════════════════════════════════════════╝

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

    print(f"[SPAWN] {cfg.vehicle_blueprint} spawned at {vehicle.get_location()}")

    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    spawn_z = vehicle.get_location().z
    print(f"[SPAWN] Settled Z = {spawn_z:.2f} m")
    return vehicle, spawn_z


# ╔════════════════════════════════════════════════════════════════════════╗
# §11  MAIN NAVIGATION LOOP
# ╚════════════════════════════════════════════════════════════════════════╝

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
        self._screen = pygame.display.set_mode((200, 80))
        self._font   = pygame.font.SysFont("monospace", 13)

        self._step = 0
        self._t0   = 0.0

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
        print("[CLEANUP] Stopping and destroying actors …")
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

        loc   = self._vehicle.get_location()
        vel   = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos   = np.array([loc.x, loc.y, loc.z])
        
        # Calculate vehicle pitch for the new planner
        transform = self._vehicle.get_transform()
        vehicle_pitch_rad = math.radians(transform.rotation.pitch)

        perc = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        best_goal: Optional[Subgoal] = None

        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)

        else:  # AUTO
            # Stuck detection logic (VSGP-based Recovery Upgrade)
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                # Attempt VSGP-based reverse planning first
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc, vehicle_pitch_rad, mode="reverse")
                    
                    if best_goal:
                        # Invert X for reverse motion
                        best_goal.local_pos[0] *= -1 
                        ctrl_state = self._controller.compute(best_goal, speed, best_goal.slope_deg)
                        ctrl_state.reverse = True
                    else:
                        # Fallback to hardcoded recovery if planning fails
                        ctrl_state = self._controller.compute_recovery(rec_phase)
                else:
                    # No perception data, fallback to hardcoded recovery
                    ctrl_state = self._controller.compute_recovery(rec_phase)
                
                self._planner.set_exploration_bias(0.8)
            else:
                self._planner.set_exploration_bias(0.0)
                if perc is not None:
                    # Call new unified planner with pitch
                    best_goal, _ = self._planner.plan(perc, vehicle_pitch_rad, mode="forward")

                slope      = best_goal.slope_deg if best_goal else 0.0
                ctrl_state = self._controller.compute(best_goal, speed, slope)

            # Fix steer history
            if self._stuck._steer_history:
                self._stuck._steer_history.pop()
            self._stuck._steer_history.append(ctrl_state.steer)

        carla_ctrl = carla.VehicleControl(
            throttle = float(ctrl_state.throttle),
            steer    = float(ctrl_state.steer),
            brake    = float(ctrl_state.brake),
            reverse  = ctrl_state.reverse,
        )
        self._vehicle.apply_control(carla_ctrl)

        self._screen.fill((20, 20, 40))
        lines = [
            f"Mode: {self._mode}",
            f"Speed: {speed:.1f} m/s",
            f"Thr: {ctrl_state.throttle:.2f}  Str: {ctrl_state.steer:.2f}",
            f"Step: {self._step}   t={t_now:.1f}s",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (200, 220, 255))
            self._screen.blit(surf, (5, 4 + i * 16))
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


# ╔════════════════════════════════════════════════════════════════════════╗
# §12  ENTRY POINT
# ╚════════════════════════════════════════════════════════════════════════╝

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
