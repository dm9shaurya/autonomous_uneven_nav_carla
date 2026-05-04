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

# ──────────────────────────────────────────────────────────────────────────────
# Optional scipy / sklearn
# ──────────────────────────────────────────────────────────────────────────────
try:
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found. inducing points fall back to random selection")

try:
    import carla
except ImportError:
    print("[FATAL] carla Python package not found. Put CARLA's egg on PYTHONPATH.")
    sys.exit(1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA connection
    carla_host: str            = "127.0.0.1"
    carla_port: int            = 2000
    carla_timeout: float       = 20.0
    synchronous: bool          = True
    fixed_delta_seconds: float = 0.05          # 20 Hz physics

    # ── Vehicle
    vehicle_blueprint: str     = "vehicle.tesla.cybertruck"
    spawn_z_offset: float      = 5.0
    post_spawn_wait_s: float   = 3.0

    # ── Vehicle Geometry (New for Unified Planner)
    vehicle_length: float      = 4.5
    vehicle_width: float       = 2.0
    vehicle_height: float      = 1.8
    wheel_base: float          = 2.8
    track_width: float         = 1.7
    wheel_radius_real: float   = 0.43

    # ── Sensors
    lidar_range: float         = 10.0
    lidar_points_per_sec: int  = 80000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int        = 64
    lidar_upper_fov: float     = 15.0
    lidar_lower_fov: float     = -25.0
    camera_width: int          = 320
    camera_height: int         = 240
    camera_fov: int            = 90

    # ── Mecanum geometry (Legacy Kinematics)
    wheel_radius: float        = 0.15           # rw  [m]
    lx: float                  = 0.25           # half wheelbase [m]
    ly: float                  = 0.20           # half track width [m]
    max_wheel_omega: float     = 40.0           # rad/s

    # ── Planner
    robot_width: float         = 0.8            # [m] (Legacy)
    safety_margin: float       = 2.80            # [m]
    max_slope_deg: float       = 15.0           # degrees
    subgoal_distance: float    = 5.0            # [m]
    n_subgoals_max: int        = 5

    # ── Cost weights
    w_direction: float         = 4.0
    w_distance: float          = 2.5
    w_steepness: float         = 50.0
    w_collision: float         = 500.0
    w_oscillation: float       = 50.0
    w_flatness: float          = 5.0

    # ── Control
    max_throttle: float        = 0.55
    max_steer: float           = 0.9
    max_brake: float           = 0.8
    ema_alpha: float           = 0.5           # smoothing factor
    max_steer_rate: float      = 0.2           # per step
    max_throttle_rate: float   = 0.1           # per step
    base_speed: float          = 3.0            # m/s target forward

    # ── Stability (roll / pitch)
    max_safe_roll_deg: float   = 10.0           # roll threshold → throttle cut
    max_safe_pitch_deg: float  = 15.0           # pitch threshold → throttle cut
    stability_throttle_scale: float = 0.4       # multiplier when unstable
    roll_corrective_steer: float    = 0.5      # steer gain towards level

    # ── Stuck detection
    stuck_window_s: float      = 2.0
    stuck_disp_thresh: float   = 0.6            # [m]
    recovery_duration_s: float = 2.5

    # ── VSGP
    vsgp_n_inducing: int       = 50
    vsgp_alpha: float          = 1.0
    vsgp_length_scale: float   = 0.3
    vsgp_noise_var: float      = 0.05
    vsgp_lr: float             = 0.02

    # ── Visualization
    viz_update_hz: float       = 5.0
    traj_history: int          = 500


CFG = Config()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §1  MECANUM KINEMATIC MODEL (FIXED)
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MecanumKinematics:
    """Improved mecanum model with decoupled control (speed vs steering)."""

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
    ) -> Tuple[float, float, float, bool]:
        """
        Converts velocity commands to CARLA control.
        Returns: (throttle, steer, brake, reverse)
        """
        speed = math.sqrt(vx**2 + vy**2)
        reverse = False

        # Throttle Control
        target_speed = cfg.base_speed
        speed_error = target_speed - speed

        if vx >= -0.1:
            # Forward motion
            throttle = 0.5 * speed_error
            throttle = float(np.clip(throttle, 0.0, cfg.max_throttle))
        else:
            # Reverse motion
            reverse = True
            throttle = float(np.clip(-vx / target_speed, 0.0, cfg.max_throttle))

        brake = 0.0

        # Steering Control
        # Use curvature = omega / vx. If still, use omega directly.
        if abs(vx) > 0.2:
            curvature = omega / vx
        else:
            # If moving sideways or still, map omega to steer directly
            curvature = omega * 2.0

        # Lateral correction (vy)
        steer = curvature * 2.5 - vy * 0.3

        # If reversing, steering logic is inverted in CARLA typically
        # (steer left to go right when backing up), but CARLA's VehicleControl
        # handles 'steer' as 'steer the wheels'. So if we want to turn left while
        # reversing, we still steer left. However, the error sign flips.
        # We keep steering command consistent with desired path curvature.

        steer = float(np.clip(steer, -cfg.max_steer, cfg.max_steer))
        return throttle, steer, brake, reverse


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚══════════════════════════════════════════════════════════════════════════════╝

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
    ROC             = 5.0
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
            - self.VAR_FREE_WEIGHT * var_norm,
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §4  SUBGOAL PLANNER (NEW: VSGP-BASED)
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


class VSGPSubgoalPlanner:
    """
    Generates subgoals using VSGP perception data.
    Logic:
    1. Samples candidate points in the VSGP grid.
    2. Validates candidates: must be FREE, STABLE (low var), and SAFE (traversable).
    3. Checks path to candidate.
    4. If no valid candidate found, returns None (triggers recovery/alternative route).
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        # Thresholds
        self.max_occ_thresh   = 0.35   # Occupancy must be below this (Free)
        self.max_var_thresh   = 0.4    # Variance must be below this (Stable)
        self.max_slope_thresh = 0.2    # Normalized slope below this (Safe)
        self.min_dist         = 3.0    # Minimum look-ahead
        self.max_dist         = cfg.subgoal_distance

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        mode: str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        """
        Main planning entry point.
        """
        if perc.occupancy_mean is None:
            return None, []

        # Generate Candidates — scan a fan of angles
        # NOTE: The current VSGP setup in PerceptionModule limits alpha to
        # [-pi/2, pi/2]. Reverse mode uses the same forward sensing grid but
        # commands negative speed via the controller's reverse flag.
        alpha_range = np.linspace(-np.pi / 2.5, np.pi / 2.5, 15)

        candidates = []

        # Extract grid data for faster access
        occ   = perc.occupancy_mean
        var   = perc.occupancy_var
        slope = perc.slope_map

        # Grid mapping
        alphas = perc.alpha_grid[0]          # alpha values for columns
        betas  = perc.beta_grid[:, 0]        # beta values for rows
        mid_beta_idx = len(betas) // 2

        for a in alpha_range:
            # Find closest grid index for alpha
            a_idx = np.argmin(np.abs(alphas - a))

            # Ray casting along this angle
            best_point_cost = float('inf')
            best_point_dist = 0.0
            s_val = 0.0

            dist_samples = np.linspace(self.min_dist, self.max_dist, 5)

            # for d in dist_samples:
            #     # Check if index is valid
            #     if not (0 <= a_idx < len(alphas)):
            #         break

            #     o_val = occ[mid_beta_idx, a_idx]
            #     v_val = var[mid_beta_idx, a_idx]
            #     s_val = slope[mid_beta_idx, a_idx]

            for d in dist_samples:

                # map distance → beta index (forward direction)
                step_ratio = d / self.max_dist
                b_idx = int(mid_beta_idx - step_ratio * (mid_beta_idx - 1))

                if not (0 <= a_idx < occ.shape[1] and 0 <= b_idx < occ.shape[0]):
                    break

                o_val = occ[b_idx, a_idx]
                v_val = var[b_idx, a_idx]
                s_val = slope[b_idx, a_idx]

                is_free   = o_val < self.max_occ_thresh
                is_stable = v_val < self.max_var_thresh
                is_safe   = s_val < self.max_slope_thresh

                if is_free and is_stable and is_safe:
                    # Cost: prefer straight, far, low slope
                    cost = (
                        2.0 * abs(a) / (np.pi / 2) +   # Direction penalty
                        1.0 * s_val +                   # Slope penalty
                        0.5 * v_val                     # Variance penalty
                    )
                    cost -= d * 0.5  # Prefer further points

                    if cost < best_point_cost:
                        best_point_cost = cost
                        best_point_dist = d
                else:
                    # Obstacle hit — stop ray for this angle
                    break

            if best_point_dist > 0:
                lx = best_point_dist * np.cos(a)
                ly = best_point_dist * np.sin(a)
                lz = 0.0

                sg = Subgoal(
                    alpha=a,
                    beta=0.0,
                    distance=best_point_dist,
                    local_pos=np.array([lx, ly, lz]),
                    slope_deg=s_val * 45.0,  # Approx: normalised slope 0-1 → degrees
                    safe=True,
                    width_m=self.cfg.vehicle_width,
                    cost=best_point_cost,
                )
                candidates.append(sg)

        if not candidates:
            return None, []

        # Sort candidates by cost (ascending)
        candidates.sort(key=lambda s: s.cost)
        best = candidates[0]

        return best, candidates


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §5  CONTROLLER MODULE
# ╚══════════════════════════════════════════════════════════════════════════════╝

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
            return self._apply_smooth(0.0, 0.0, 0.0, roll_deg=roll_deg,
                                      pitch_deg=pitch_deg, brake=1.0)

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
        self, vx: float, vy: float, omega: float,
    ) -> ControlState:
        ws           = self.mec.inverse(vx, vy, omega)
        vx_a, vy_a, om_a = self.mec.forward(ws)
        return self._apply_smooth(vx_a, vy_a, om_a)

    def compute_recovery(self, phase: int) -> ControlState:
        """Fallback hard-coded recovery."""
        if phase == 0:
            return ControlState(throttle=0.3, steer=0.0,  brake=0.0, reverse=True)
        elif phase == 1:
            return ControlState(throttle=0.3, steer=0.4,  brake=0.0, reverse=True)
        else:
            return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

    def _stability_modifiers(
        self, roll_deg: float, pitch_deg: float
    ) -> Tuple[float, float]:
        throttle_scale    = 1.0
        corrective_steer  = 0.0
        abs_roll = abs(roll_deg)
        if abs_roll > self.cfg.max_safe_roll_deg:
            excess         = abs_roll - self.cfg.max_safe_roll_deg
            roll_factor    = 1.0 - min(excess / 15.0, 1.0)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale +
                                 (1.0 - self.cfg.stability_throttle_scale) * roll_factor)
            corrective_steer = -math.copysign(
                min(self.cfg.roll_corrective_steer * (excess / 10.0),
                    self.cfg.roll_corrective_steer),
                roll_deg,
            )
        abs_pitch = abs(pitch_deg)
        if abs_pitch > self.cfg.max_safe_pitch_deg:
            excess_p       = abs_pitch - self.cfg.max_safe_pitch_deg
            pitch_factor   = 1.0 - min(excess_p / 10.0, 0.7)
            throttle_scale = min(throttle_scale,
                                 self.cfg.stability_throttle_scale * pitch_factor + 0.15)
        return float(throttle_scale), float(corrective_steer)

    def _apply_smooth(
        self,
        vx: float,
        vy: float,
        omega: float,
        roll_deg: float  = 0.0,
        pitch_deg: float = 0.0,
        brake: float = 0.0
    ) -> ControlState:
        a            = self.cfg.ema_alpha
        self._vx_ema = a * vx    + (1 - a) * self._vx_ema
        self._vy_ema = a * vy    + (1 - a) * self._vy_ema
        self._om_ema = a * omega + (1 - a) * self._om_ema

        throttle, steer, out_brake, reverse = MecanumKinematics.to_carla_control(
            self._vx_ema, self._vy_ema, self._om_ema, self.cfg
        )

        if brake > 0.1:
            throttle  = 0.0
            out_brake = brake

        # Rate limiting
        max_dr   = self.cfg.max_steer_rate
        max_dt   = self.cfg.max_throttle_rate
        steer    = float(np.clip(steer,
                                  self._state.steer    - max_dr,
                                  self._state.steer    + max_dr))
        throttle = float(np.clip(throttle,
                                  self._state.throttle - max_dt,
                                  self._state.throttle + max_dt))

        # Stability
        thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
        throttle *= thr_scale
        steer     = float(np.clip(steer + cor_steer,
                                   -self.cfg.max_steer,
                                   self.cfg.max_steer))

        self._state = ControlState(throttle=throttle, steer=steer,
                                   brake=out_brake, reverse=reverse)
        return self._state

    def reset_ema(self) -> None:
        self._vx_ema = self._vy_ema = self._om_ema = 0.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §6  STUCK DETECTION & RECOVERY
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §7  VISUALIZATION MODULE
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §8  CARLA SENSOR WRAPPERS
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §9  MANUAL KEYBOARD CONTROLLER
# ╚══════════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §10  VEHICLE SPAWNER
# ╚══════════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    # Manual spawn transform for testing
    spawn_transform = carla.Transform(
        carla.Location(x=906.66, y=-875.78, z=4.7620),
        carla.Rotation(pitch=-4.50, yaw=88.0, roll=0.0)
    )

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

    if vehicle is None:
        raise RuntimeError(f"Spawn failed at {spawn_transform.location}")

    print(f"[SPAWN] Manual spawn at {vehicle.get_location()}")

    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    settled_z = vehicle.get_location().z
    print(f"[SPAWN] Settled Z = {settled_z:.2f} m")

    return vehicle, settled_z


# ╔══════════════════════════════════════════════════════════════════════════════╗
# §11  MAIN NAVIGATION LOOP
# ╚══════════════════════════════════════════════════════════════════════════════╝

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
        # Use NEW Planner
        self._planner     = VSGPSubgoalPlanner(cfg)
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

    def _tick(self) -> None:
        if self.cfg.synchronous:
            self._world.tick()

        t_now      = time.time() - self._t0
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

        loc       = self._vehicle.get_location()
        vel       = self._vehicle.get_velocity()
        speed     = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos       = np.array([loc.x, loc.y, loc.z])

        transform    = self._vehicle.get_transform()
        pitch_rad    = math.radians(transform.rotation.pitch)
        roll_rad     = math.radians(transform.rotation.roll)
        pitch_deg    = math.degrees(pitch_rad)
        roll_deg     = math.degrees(roll_rad)

        # Perception
        perc : Optional[PerceptionOutput] = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        best_goal  : Optional[Subgoal] = None
        ctrl_state : ControlState

        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)

        else:  # AUTO
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                # Recovery Mode
                print("[NAV] Stuck detected. Initiating recovery.")
                if perc is not None:
                    # Try to find a reverse path
                    best_goal, _ = self._planner.plan(perc, pitch_rad, mode="reverse")

                if best_goal:
                    # Valid reverse path found
                    ctrl_state = self._controller.compute(
                        best_goal, speed, 0.0, roll_deg, pitch_deg)
                else:
                    # No path found — hard recovery
                    ctrl_state = self._controller.compute_recovery(rec_phase)

            else:
                # Normal Navigation
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc, pitch_rad, mode="forward")

                if best_goal:
                    slope      = best_goal.slope_deg
                    ctrl_state = self._controller.compute(
                        best_goal, speed, slope,
                        roll_deg=roll_deg, pitch_deg=pitch_deg,
                    )
                else:
                    # No valid subgoal found (Blocked) — stop and wait
                    # ctrl_state = ControlState(throttle=0.0, steer=0.0,
                    #                           brake=1.0, reverse=False)
                    # print("[NAV] No valid subgoal found. Stopping.")

                  # fallback: slow forward exploration

                    fallback_goal = Subgoal(
                        alpha=0.0,
                        beta=0.0,
                        distance=2.0,
                        local_pos=np.array([2.0, 0.0, 0.0]),
                        slope_deg=0.0,
                        safe=False,
                        width_m=self.cfg.vehicle_width,
                        cost=999.0,
                    )

                    ctrl_state = self._controller.compute(
                        fallback_goal, speed, 0.0,
                        roll_deg=roll_deg, pitch_deg=pitch_deg,
                    )

                    print("[NAV] Fallback: creeping forward")  

            # Update stuck detector with latest steer
            if self._stuck._steer_history:
                self._stuck._steer_history.pop()
            self._stuck._steer_history.append(ctrl_state.steer)

        # Apply Control
        self._vehicle.apply_control(carla.VehicleControl(
            throttle = float(ctrl_state.throttle),
            steer    = float(ctrl_state.steer),
            brake    = float(ctrl_state.brake),
            reverse  = ctrl_state.reverse,
        ))

        # HUD
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

        # Visualization
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
