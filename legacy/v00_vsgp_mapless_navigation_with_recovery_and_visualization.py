from __future__ import annotations

import math
import random
import sys
import time
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from queue import Queue

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA connection
    carla_host: str             = "127.0.0.1"
    carla_port: int             = 2000
    carla_timeout: float        = 20.0
    synchronous: bool           = True
    fixed_delta_seconds: float  = 0.05          # 20 Hz physics

    # ── Vehicle (Spawn point kept intact as requested)
    vehicle_blueprint: str      = "vehicle.tesla.cybertruck"
    spawn_z_offset: float       = 5.0
    post_spawn_wait_s: float    = 3.0

    # FIX #5 – Cybertruck physics inversion flag.
    # The Cybertruck blueprint in CARLA has inverted drive: positive throttle
    # moves the vehicle backward and steering direction is flipped.
    # Setting this True swaps reverse flag and negates steer at apply_control.
    invert_drive: bool          = False

    # ── Vehicle Geometry
    vehicle_length: float       = 4.5
    vehicle_width: float        = 2.0
    vehicle_height: float       = 1.8
    wheel_base: float           = 2.8
    track_width: float          = 1.7
    wheel_radius_real: float    = 0.43

    # ── Sensors
    lidar_range: float          = 10.0
    lidar_points_per_sec: int   = 80000
    lidar_rotation_freq: float  = 20.0
    lidar_channels: int         = 64
    lidar_upper_fov: float      = 15.0
    lidar_lower_fov: float      = -25.0
    camera_width: int           = 320
    camera_height: int          = 240
    camera_fov: int             = 90

    # ── Planner
    robot_width: float          = 0.8
    safety_margin: float        = 2.80
    max_slope_deg: float        = 15.0
    subgoal_distance: float     = 5.0
    n_subgoals_max: int         = 5

    # ── Cost weights
    w_direction: float          = 4.0
    w_distance: float           = 2.5
    w_steepness: float          = 50.0
    w_collision: float          = 500.0
    w_oscillation: float        = 50.0
    w_flatness: float           = 5.0

    # ── Control (Ackermann)
    max_throttle: float         = 0.55
    max_steer: float            = 0.9
    max_brake: float            = 0.8
    ema_alpha: float            = 0.3
    max_steer_rate: float       = 0.15
    max_throttle_rate: float    = 0.08
    base_speed: float           = 3.0
    kp_throttle: float          = 0.5
    kp_steer: float             = 1.5

    # ── Stability (roll / pitch)
    max_safe_roll_deg: float        = 10.0
    max_safe_pitch_deg: float       = 15.0
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float    = 0.5

    # ── Stuck detection
    stuck_window_s: float       = 2.0
    stuck_disp_thresh: float    = 0.6
    recovery_duration_s: float  = 2.5

    # ── VSGP
    vsgp_n_inducing: int        = 50
    vsgp_alpha: float           = 1.0
    vsgp_length_scale: float    = 0.3
    vsgp_noise_var: float       = 0.05
    vsgp_lr: float              = 0.02
    vsgp_update_freq: int       = 10

    # ── Visualization
    viz_update_hz: float        = 3.0
    traj_history: int           = 500


CFG = Config()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §1  ACKERMANN KINEMATIC MODEL
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AckermannKinematics:
    """Standard bicycle model for Ackermann steering vehicles."""

    def __init__(self, cfg: Config = CFG):
        self.wheel_base = cfg.wheel_base
        self.cfg = cfg

    def steering_angle_to_curvature(self, steer: float) -> float:
        if abs(steer) < 1e-6:
            return 0.0
        return math.tan(steer) / self.wheel_base

    def curvature_to_steering_angle(self, curvature: float) -> float:
        return math.atan(curvature * self.wheel_base)

    @staticmethod
    def to_carla_control(
        vx_desired: float,
        yaw_error: float,
        current_speed: float,
        cfg: Config = CFG,
    ) -> Tuple[float, float, float, bool]:
        """Returns: (throttle, steer, brake, reverse)"""
        reverse = False

        speed_error = vx_desired - current_speed
        throttle = cfg.kp_throttle * speed_error
        throttle = float(np.clip(throttle, 0.0, cfg.max_throttle))

        brake = 0.0
        if speed_error < -1.0:
            brake = min(0.5, abs(speed_error) * 0.3)
            throttle = 0.0

        steer = cfg.kp_steer * yaw_error
        steer = float(np.clip(steer, -cfg.max_steer, cfg.max_steer))

        return throttle, steer, brake, reverse


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha: float = 1.0,
                 length_scale: float = 0.3,
                 variance: float = 1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if len(X) == 0 or len(Y) == 0:
            return np.zeros((len(X), len(Y)))
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
        self.update_freq = cfg.vsgp_update_freq

        side = int(math.sqrt(self.n_ind))
        a_v = np.linspace(-np.pi / 2, np.pi / 2, side)
        b_v = np.linspace(-np.pi / 4, np.pi / 4, side)
        A, B = np.meshgrid(a_v, b_v)
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

        self.mu = np.zeros(len(self.Z))
        self.Su = np.eye(len(self.Z)) * 0.1

        self._Kuu_inv: Optional[np.ndarray] = None
        self._trained: bool = False
        self._update_count: int = 0
        self._last_X_hash: int = 0

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
        self._Kuu_inv = None

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return

        self._update_count += 1

        if self._update_count % (50 * self.update_freq) == 0:
            self._select_inducing_points(X)

        current_hash = hash(X.tobytes())  #before it was current_hash = hash(X.shape[0])
        if current_hash == self._last_X_hash and self._trained:
            return
        self._last_X_hash = current_hash

        m = len(self.Z)
        Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6

        try:
            self._Kuu_inv = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            self._Kuu_inv = np.linalg.pinv(Kuu)

        Kfu = self.kernel(X, self.Z)
        A = Kfu @ self._Kuu_inv

        noise_inv = 1.0 / self.noise_var
        Lambda = noise_inv * (A.T @ A) + self._Kuu_inv
        rhs = noise_inv * (A.T @ y)

        try:
            Su_new = np.linalg.inv(Lambda + np.eye(m) * 1e-6)
            mu_new = Su_new @ rhs
        except np.linalg.LinAlgError:
            return

        lr = self.lr
        self.Su = (1 - lr) * self.Su + lr * Su_new
        self.mu = (1 - lr) * self.mu + lr * mu_new
        self._trained = True

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained or self._Kuu_inv is None:
            n = len(Xs)
            return np.zeros(n), np.ones(n) * self.noise_var

        Ksu = self.kernel(Xs, self.Z)
        A = Ksu @ self._Kuu_inv
        mean = A @ self.mu

        Kss_diag = self.kernel.diag(Xs)
        var_explained = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
        var_variational = np.sum(A @ self.Su * A, axis=1)
        var = Kss_diag - var_explained + var_variational + self.noise_var
        var = np.clip(var, 1e-6, None)

        return mean, var


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class PerceptionOutput:
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    occupancy_mean: np.ndarray
    occupancy_var: np.ndarray
    slope_map: np.ndarray
    traversability: np.ndarray
    raw_points: np.ndarray
    free_mask: Optional[np.ndarray] = None
    mean_surface: Optional[np.ndarray] = None
    ground_slope_deg: float = 0.0


class PerceptionModule:
    ALPHA_RES = 60
    BETA_RES = 30
    ROC = 5.0
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

    def _estimate_ground_slope(self, pts: np.ndarray) -> float:
        if len(pts) < 10:
            return 0.0
        z_median = np.median(pts[:, 2])
        ground_mask = np.abs(pts[:, 2] - z_median) < 1.0
        ground_pts = pts[ground_mask]
        if len(ground_pts) < 10:
            return 0.0
        centroid = np.mean(ground_pts, axis=0)
        centered = ground_pts - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1, :]
        z_axis = np.array([0, 0, 1])
        cos_angle = np.abs(np.dot(normal, z_axis)) / (
            np.linalg.norm(normal) * np.linalg.norm(z_axis))
        slope_rad = np.arccos(np.clip(cos_angle, -1, 1))
        return float(np.clip(np.degrees(slope_rad), 0, 90))

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(raw) < 20:
            return self._last_output

        dist = np.linalg.norm(raw, axis=1)
        mask = (dist > 0.5) & (dist < self.cfg.lidar_range)
        pts = raw[mask]
        if len(pts) < 10:
            return self._last_output

        ground_slope = self._estimate_ground_slope(pts)

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

        d_alpha = np.gradient(mean, axis=1)
        d_beta = np.gradient(mean, axis=0)
        slope = np.sqrt(d_alpha ** 2 + d_beta ** 2)
        slope = slope / (np.max(slope) + 1e-6)

        mean_range = mean.max() - mean.min()
        occ_norm = (mean - mean.min()) / (mean_range + 1e-6) if mean_range > 0 else mean
        var_range = var.max() - var.min()
        var_norm = (var - var.min()) / (var_range + 1e-6) if var_range > 0 else var

        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT * occ_norm
            - self.SLOPE_WEIGHT * slope
            - self.VAR_FREE_WEIGHT * var_norm,
            0.0, 1.0,
        )
        free_mask = traversability > 0.5

        out = PerceptionOutput(
            alpha_grid=self._AG,
            beta_grid=self._BG,
            occupancy_mean=mean,
            occupancy_var=var,
            slope_map=slope,
            traversability=traversability,
            raw_points=pts,
            free_mask=free_mask,
            mean_surface=mean,
            ground_slope_deg=ground_slope,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §4  SUBGOAL PLANNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

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


class VSGPSubgoalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.max_occ_thresh = 0.35
        self.max_var_thresh = 0.4
        self.max_slope_thresh = 0.2
        self.min_dist = 3.0
        self.max_dist = cfg.subgoal_distance

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        mode: str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        if perc.occupancy_mean is None:
            return None, []

        alpha_range = np.linspace(-np.pi / 2.5, np.pi / 2.5, 15)
        candidates = []

        occ = perc.occupancy_mean
        var = perc.occupancy_var
        slope = perc.slope_map

        alphas = perc.alpha_grid[0]
        betas = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        for a in alpha_range:
            a_idx = np.argmin(np.abs(alphas - a))

            best_point_cost = float('inf')
            best_point_dist = 0.0
            s_val = 0.0

            dist_samples = np.linspace(self.min_dist, self.max_dist, 5)

            for d in dist_samples:
                step_ratio = d / self.max_dist
                pitch_correction = vehicle_pitch_rad * step_ratio * 5
                b_idx = int(mid_beta_idx - step_ratio * (mid_beta_idx - 1) + pitch_correction)
                b_idx = int(np.clip(b_idx, 0, occ.shape[0] - 1))

                if not (0 <= a_idx < occ.shape[1]):
                    break

                o_val = occ[b_idx, a_idx]
                v_val = var[b_idx, a_idx]
                s_val = slope[b_idx, a_idx]

                is_free = o_val < self.max_occ_thresh
                is_stable = v_val < self.max_var_thresh
                is_safe = s_val < self.max_slope_thresh

                if is_free and is_stable and is_safe:
                    cost = (
                        2.0 * abs(a) / (np.pi / 2) +
                        1.0 * s_val +
                        0.5 * v_val
                    )
                    cost -= d * 0.5

                    if cost < best_point_cost:
                        best_point_cost = cost
                        best_point_dist = d
                else:
                    break

            if best_point_dist > 0:
                lx = best_point_dist * np.cos(a)
                ly = best_point_dist * np.sin(a)

                sg = Subgoal(
                    alpha=a,
                    beta=0.0,
                    distance=best_point_dist,
                    local_pos=np.array([lx, ly, 0.0]),
                    slope_deg=s_val * 45.0,
                    safe=True,
                    width_m=self.cfg.vehicle_width,
                    cost=best_point_cost,
                )
                candidates.append(sg)

        if not candidates:
            return None, []

        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §5  CONTROLLER MODULE
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle: float = 0.0
    steer: float    = 0.0
    brake: float    = 0.0
    reverse: bool   = False


class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.ackermann = AckermannKinematics(cfg)
        self._state = ControlState()
        self._speed_ema = 0.0
        self._steer_ema = 0.0

    def compute(
        self,
        subgoal: Optional[Subgoal],
        vehicle_speed_ms: float,
        terrain_slope_deg: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> ControlState:
        if subgoal is None:
            return self._apply_smooth(0.0, 0.0, roll_deg=roll_deg,
                                      pitch_deg=pitch_deg, brake=1.0)

        alpha = subgoal.alpha
        speed_factor = max(
            0.3,
            1.0
            - terrain_slope_deg / self.cfg.max_slope_deg * 0.5
            - abs(alpha) / (math.pi / 2) * 0.3,
        )
        vx_desired = self.cfg.base_speed * speed_factor
        yaw_error = alpha

        throttle, steer, brake, reverse = AckermannKinematics.to_carla_control(
            vx_desired, yaw_error, vehicle_speed_ms, self.cfg
        )
        return self._apply_smooth(throttle, steer, brake, reverse,
                                  roll_deg=roll_deg, pitch_deg=pitch_deg)

    def compute_manual(self, vx: float, steer: float) -> ControlState:
        throttle = np.clip(vx / self.cfg.base_speed, 0.0, self.cfg.max_throttle)
        return self._apply_smooth(throttle, steer, 0.0, False)

    def compute_recovery(self, phase: int) -> ControlState:
        if phase == 0:
            return ControlState(throttle=0.3, steer=0.0,  brake=0.0, reverse=True)
        elif phase == 1:
            return ControlState(throttle=0.3, steer=0.4,  brake=0.0, reverse=True)
        else:
            return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

    def _stability_modifiers(self, roll_deg: float, pitch_deg: float) -> Tuple[float, float]:
        throttle_scale = 1.0
        corrective_steer = 0.0
        abs_roll = abs(roll_deg)

        if abs_roll > self.cfg.max_safe_roll_deg:
            excess = abs_roll - self.cfg.max_safe_roll_deg
            roll_factor = 1.0 - min(excess / 15.0, 1.0)
            throttle_scale = min(
                throttle_scale,
                self.cfg.stability_throttle_scale
                + (1.0 - self.cfg.stability_throttle_scale) * roll_factor,
            )
            corrective_steer = -math.copysign(
                min(self.cfg.roll_corrective_steer * (excess / 10.0),
                    self.cfg.roll_corrective_steer),
                roll_deg,
            )

        abs_pitch = abs(pitch_deg)
        if abs_pitch > self.cfg.max_safe_pitch_deg:
            excess_p = abs_pitch - self.cfg.max_safe_pitch_deg
            pitch_factor = 1.0 - min(excess_p / 10.0, 0.7)
            throttle_scale = min(
                throttle_scale,
                self.cfg.stability_throttle_scale * pitch_factor + 0.15,
            )

        return float(throttle_scale), float(corrective_steer)

    def _apply_smooth(
        self,
        throttle: float,
        steer: float,
        brake: float = 0.0,
        reverse: bool = False,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> ControlState:
        a = self.cfg.ema_alpha
        self._speed_ema = a * throttle + (1 - a) * self._speed_ema
        self._steer_ema = a * steer   + (1 - a) * self._steer_ema

        out_throttle = self._speed_ema
        out_steer    = self._steer_ema
        out_brake    = brake

        if brake > 0.1:
            out_throttle = 0.0
            out_brake    = brake

        max_dr = self.cfg.max_steer_rate
        max_dt = self.cfg.max_throttle_rate
        out_steer    = float(np.clip(out_steer,
                                     self._state.steer    - max_dr,
                                     self._state.steer    + max_dr))
        out_throttle = float(np.clip(out_throttle,
                                     self._state.throttle - max_dt,
                                     self._state.throttle + max_dt))

        thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
        out_throttle *= thr_scale
        out_steer = float(np.clip(out_steer + cor_steer,
                                  -self.cfg.max_steer, self.cfg.max_steer))

        self._state = ControlState(throttle=out_throttle, steer=out_steer,
                                   brake=out_brake, reverse=reverse)
        return self._state

    def reset_ema(self) -> None:
        self._speed_ema = self._steer_ema = 0.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §6  STUCK DETECTION & RECOVERY
# ╚═══════════════════════════════════════════════════════════════════════════╝

class StuckDetector:
    IDLE      = "IDLE"
    RECOVERING = "RECOVERING"

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._pos_history: deque   = deque(maxlen=200)
        self._steer_history: deque = deque(maxlen=40)
        self._state = self.IDLE
        self._recovery_start: float = 0.0
        self._recovery_phase: int   = 0

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
        self._pos_history.append((t, pos.copy()))
        # FIX #4 – steer is appended ONCE here inside update().
        # The original code also called update_steer() after this in _tick(),
        # causing every steer value to be recorded twice per frame, which
        # made the oscillation detector fire spuriously and corrupted the
        # sign-flip count that triggers stuck recovery.
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
            self._state            = self.RECOVERING
            self._recovery_start   = t
            self._recovery_phase   = 0
            self._pos_history.clear()

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §7  VISUALIZATION MODULE (Threaded)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._data_lock = threading.Lock()
        self._subgoal    = None
        self._vehicle_pos = None
        self._speed_ms    = 0.0
        self._steer       = 0.0
        self._perc        = None
        self._mode        = "AUTO"
        self._front_img   = None
        self._rear_img    = None
        self._lidar_pts   = None

        maxlen = cfg.traj_history
        self._cost_total: deque  = deque(maxlen=maxlen)
        self._cost_dir: deque    = deque(maxlen=maxlen)
        self._cost_dist: deque   = deque(maxlen=maxlen)
        self._cost_steep: deque  = deque(maxlen=maxlen)
        self._vel_lin: deque     = deque(maxlen=maxlen)
        self._vel_ang: deque     = deque(maxlen=maxlen)
        self._traj_x: deque      = deque(maxlen=maxlen)
        self._traj_y: deque      = deque(maxlen=maxlen)
        self._subgoal_x: deque   = deque(maxlen=20)
        self._subgoal_y: deque   = deque(maxlen=20)

        self._interval: float = 1.0 / cfg.viz_update_hz

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        plt.close("all")

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
        with self._data_lock:
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

            if perc is not None and perc.raw_points is not None:
                self._lidar_pts = perc.raw_points.copy()

            self._subgoal     = subgoal
            self._vehicle_pos = vehicle_pos.copy()
            self._speed_ms    = speed_ms
            self._steer       = steer
            self._mode        = mode

    def set_front_image(self, arr: np.ndarray) -> None:
        with self._data_lock:
            self._front_img = arr

    def set_rear_image(self, arr: np.ndarray) -> None:
        with self._data_lock:
            self._rear_img = arr

    def _render_loop(self) -> None:
        plt.ion()
        fig = plt.figure("VSGP Navigator", figsize=(16, 9))
        fig.patch.set_facecolor("#111")
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        ax_front = fig.add_subplot(gs[0, 0])
        ax_lidar = fig.add_subplot(gs[0, 1])
        ax_rear  = fig.add_subplot(gs[0, 2])
        ax_cost  = fig.add_subplot(gs[1, 0])
        ax_traj  = fig.add_subplot(gs[1, 1])
        ax_vel   = fig.add_subplot(gs[1, 2])

        for ax in fig.get_axes():
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#ddd")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")

        last_update = 0.0
        interval    = self._interval

        while self._running:
            now = time.time()
            if (now - last_update) < interval:
                time.sleep(0.01)
                continue
            last_update = now

            # FIX #1 – The original try/except block was de-indented to
            # method scope (4-space indent), placing it OUTSIDE the while
            # loop. That meant it ran only once after the loop exited, so
            # the render plots were never actually drawn during navigation.
            # Moved inside the while loop at correct indentation (12 spaces).
            try:
                with self._data_lock:
                    self._draw_camera(ax_front, self._front_img, "Front Camera")
                    self._draw_camera(ax_rear,  self._rear_img,  "Rear Camera")
                    self._draw_lidar(ax_lidar)
                    self._draw_cost(ax_cost)
                    self._draw_trajectory(ax_traj)
                    self._draw_velocity(ax_vel)

                fig.suptitle(
                    f"VSGP Mapless Navigator  |  Mode: {self._mode}",
                    color="#eee", fontsize=11, y=0.99,
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                pass

        plt.close(fig)

    def _draw_camera(self, ax, img, title):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if img is not None:
            ax.imshow(img)

    def _draw_lidar(self, ax):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("LiDAR (top view)", fontsize=8)
        if self._lidar_pts is not None and len(self._lidar_pts) > 0:
            pts = self._lidar_pts
            z   = pts[:, 2]
            z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)
            ax.scatter(pts[:, 1], pts[:, 0], c=z_n,
                       cmap="plasma", s=0.5, alpha=0.6)
            ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
            ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)

    def _draw_cost(self, ax):
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

    def _draw_trajectory(self, ax):
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

    def _draw_velocity(self, ax):
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §8  CARLA SENSOR WRAPPERS
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self.vehicle = vehicle
        self.world   = world
        self._actors: list = []

        self._lidar_queue: Queue = Queue(maxsize=1)
        self._front_queue: Queue = Queue(maxsize=1)
        self._rear_queue:  Queue = Queue(maxsize=1)

        bp_lib = world.get_blueprint_library()

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
        if not self._lidar_queue.full():
            self._lidar_queue.put(data)

    def _on_front_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1].copy()
        if not self._front_queue.full():
            self._front_queue.put(arr)

    def _on_rear_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1].copy()
        if not self._rear_queue.full():
            self._rear_queue.put(arr)

    @property
    def lidar_data(self):
        try:
            return self._lidar_queue.get_nowait()
        except Exception:
            return None

    @property
    def front_image(self) -> Optional[np.ndarray]:
        try:
            return self._front_queue.get_nowait()
        except Exception:
            return None

    @property
    def rear_image(self) -> Optional[np.ndarray]:
        try:
            return self._rear_queue.get_nowait()
        except Exception:
            return None

    def destroy(self) -> None:
        for a in self._actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §9  MANUAL KEYBOARD CONTROLLER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ManualController:
    VX_SPEED    = 3.0
    STEER_SPEED = 0.5

    def __init__(self, cfg: Config = CFG):
        self.ctrl   = Controller(cfg)
        self._steer = 0.0

    def tick(self, keys) -> ControlState:
        vx = 0.0
        if keys[pygame.K_w]:
            vx = self.VX_SPEED
        if keys[pygame.K_s]:
            vx = -self.VX_SPEED
        if keys[pygame.K_a]:
            self._steer = np.clip(self._steer + self.STEER_SPEED,
                                  -CFG.max_steer, CFG.max_steer)
        if keys[pygame.K_d]:
            self._steer = np.clip(self._steer - self.STEER_SPEED,
                                  -CFG.max_steer, CFG.max_steer)
        if keys[pygame.K_SPACE]:
            self._steer = 0.0
        return self.ctrl.compute_manual(vx, self._steer)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §10  VEHICLE SPAWNER  (spawn point kept identical to original)
# ╚═══════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    # ── Spawn transform kept intact as requested ─────────────────────────────
    spawn_transform = carla.Transform(
        carla.Location(x=906.66, y=-875.78, z=4.7620),
        carla.Rotation(pitch=-4.50, yaw=88.0, roll=0.0),
    )

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

    # ── Fallback to dynamic spawn points if hardcoded location is occupied ───
    if vehicle is None:
        print("[WARN] Hardcoded spawn failed, trying dynamic spawn points...")
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            spawn_transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

    if vehicle is None:
        raise RuntimeError(f"Spawn failed at {spawn_transform.location}")

    print(f"[SPAWN] Spawned at {vehicle.get_location()}")

    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    settled_z = vehicle.get_location().z
    print(f"[SPAWN] Settled Z = {settled_z:.2f} m")
    return vehicle, settled_z


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §11  MAIN NAVIGATION LOOP
# ╚═══════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._mode    = "AUTO"

        self._client: Optional[carla.Client] = None
        self._world   = None
        self._vehicle = None
        self._sensors: Optional[SensorManager] = None

        self._perception  = PerceptionModule(cfg)
        self._planner     = VSGPSubgoalPlanner(cfg)
        self._controller  = Controller(cfg)
        self._manual_ctrl = ManualController(cfg)
        self._stuck       = StuckDetector(cfg)
        self._viz         = VisualizationModule(cfg)

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit")
        self._screen = pygame.display.set_mode((220, 88))
        self._font   = pygame.font.SysFont("monospace", 13)

        self._step = 0
        self._t0   = 0.0

    def connect(self) -> None:
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port} …")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world = self._client.get_world()
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
        self._viz.start()
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
        self._viz.stop()
        pygame.quit()
        print("[CLEANUP] Done.")

    def _tick(self) -> None:
        if self.cfg.synchronous:
            self._world.tick()

        t_now = time.time() - self._t0
        self._step += 1

        # ── FIX #2 – Initialize ctrl_state and best_goal before any branch.
        # In the original code ctrl_state was never initialized at the top of
        # _tick(), so the first AUTO-branch call to
        #   self._stuck.update(pos, ctrl_state.steer if 'ctrl_state' in locals() else 0.0, ...)
        # silently fell back to 0.0 each tick (because _tick() locals reset
        # every call). More critically, best_goal was also never initialized,
        # so if perc was None inside the stuck-recovery branch, the subsequent
        # `if best_goal:` raised NameError and crashed the loop.
        ctrl_state: ControlState = ControlState()
        best_goal:  Optional[Subgoal] = None

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

        transform = self._vehicle.get_transform()
        pitch_rad = math.radians(transform.rotation.pitch)
        roll_rad  = math.radians(transform.rotation.roll)
        pitch_deg = math.degrees(pitch_rad)
        roll_deg  = math.degrees(roll_rad)

        # ── Perception
        # FIX #6 – When no new LiDAR frame arrives this tick, fall back to the
        # last valid PerceptionOutput instead of passing None to the planner.
        # Previously perc stayed None on every tick without fresh lidar, so
        # the planner always returned None and the vehicle drove only on the
        # fallback straight-ahead creep goal.
        lidar_data = self._sensors.lidar_data
        if lidar_data is not None:
            perc = self._perception.process_lidar(lidar_data)
        else:
            perc = self._perception.last_output   # stale but valid

        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)

        else:  # ── AUTO ────────────────────────────────────────────────────
            stuck, rec_phase = self._stuck.update(pos, ctrl_state.steer, t_now)

            if stuck:
                print("[NAV] Stuck detected. Initiating recovery.")
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc, pitch_rad, mode="reverse")

                if best_goal:
                    ctrl_state = self._controller.compute(
                        best_goal, speed, 0.0, roll_deg, pitch_deg)
                else:
                    ctrl_state = self._controller.compute_recovery(rec_phase)

            else:
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc, pitch_rad, mode="forward")

                if best_goal:
                    ctrl_state = self._controller.compute(
                        best_goal, speed, best_goal.slope_deg,
                        roll_deg=roll_deg, pitch_deg=pitch_deg,
                    )
                else:
                    # Fallback: creep straight forward
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

            # FIX #4 (companion) – update_steer() call removed here.
            # StuckDetector.update() already records the steer sample on every
            # call. The original code also called self._stuck.update_steer()
            # here, inserting a second copy of the same steer value into the
            # history deque every tick, inflating the oscillation flip-count
            # and causing false stuck triggers.

        # ── Apply Control
        # FIX #5 – Cybertruck blueprint inversion.
        # The Cybertruck in CARLA has its drive direction inverted: applying
        # throttle with reverse=False moves it backward, and steering is
        # mirrored. We compensate by flipping reverse and negating steer
        # whenever cfg.invert_drive is True.
        if self.cfg.invert_drive:
            applied_steer   = -ctrl_state.steer
            applied_reverse = not ctrl_state.reverse
        else:
            applied_steer   = ctrl_state.steer
            applied_reverse = ctrl_state.reverse

        self._vehicle.apply_control(carla.VehicleControl(
            throttle=float(ctrl_state.throttle),
            steer=float(applied_steer),
            brake=float(ctrl_state.brake),
            reverse=applied_reverse,
        ))

        # ── Pygame HUD
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

        # ── Push to visualization thread
        self._viz.push_data(
            subgoal=best_goal if self._mode == "AUTO" else None,
            vehicle_pos=pos,
            speed_ms=speed,
            steer=ctrl_state.steer,
            perc=perc,
            mode=self._mode,
        )
        front_img = self._sensors.front_image
        rear_img  = self._sensors.rear_image
        if front_img is not None:
            self._viz.set_front_image(front_img)
        if rear_img is not None:
            self._viz.set_rear_image(rear_img)

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




