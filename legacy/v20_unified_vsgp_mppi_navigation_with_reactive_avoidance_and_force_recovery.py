from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Deque, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Optional sklearn for VSGP clustering
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found clustering falls back to random")

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PyTorch (required for MPPI)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available – running on CPU")
except ImportError:
    print("[FATAL] PyTorch not found. pip install torch")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Pygame (required for visualization)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print("[FATAL] pygame not found. Install with: pip install pygame")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Matplotlib (required for plots)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[FATAL] matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CARLA (required)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print("[FATAL] carla package not found. Put CARLA egg on PYTHONPATH.")
    sys.exit(1)


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── Quick tuning profile switch
    # Options: "CURRENT" (keep explicit values below), "AGGRESSIVE", "BALANCED", "SAFE"
    tuning_profile: str = "AGGRESSIVE"

    # ── CARLA Connection
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    carla_timeout: float = 20.0
    synchronous: bool = True
    fixed_delta_seconds: float = 0.05

    # ── Map Handling (from nav7 - no map parsing)
    # IMPORTANT: This script does NOT call client.load_world().
    # It uses only the map already loaded in the CARLA server.
    use_fixed_spawn_first: bool = True
    auto_goal_from_spawn_points: bool = True

    # ── Vehicle
    vehicle_blueprint: str = "vehicle.tesla.cybertruck"
    post_spawn_wait_s: float = 2.0
    wheel_base: float = 3.807

    # ── Spawn (from nav7 - flexible z handling)
    spawn_x: float = 899.35
    spawn_y: float = -774.16
    spawn_z: float = 9.69
    spawn_pitch: float = -1.03
    spawn_yaw: float = 110.85
    spawn_roll: float = 0.00

    # ── Goal
    goal_x: float = 770.10
    goal_y: float = -287.96
    goal_tolerance: float = 5.0

    # ── Sensors (from nav4 + nav7)
    lidar_range: float = 55.0
    lidar_points_per_sec: int = 100000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int = 64
    lidar_upper_fov: float = 20.0
    lidar_lower_fov: float = -35.0
    camera_width: int = 480
    camera_height: int = 270
    camera_fov: float = 90.0

    # ── VSGP Subgoal Planning (from nav4)
    subgoal_distance: float = 30.0
    subgoal_min_distance: float = 3.0
    subgoal_num_angles: int = 3
    subgoal_num_depth: int = 10

    # ── MPPI (from nav4)
    mppi_horizon: int = 20
    mppi_num_samples: int = 1024
    mppi_lambda: float = 1.0
    mppi_noise_throttle: float = 0.2
    mppi_noise_steer: float = 0.3

    # ── Cost Weights (from nav4)
    w_goal: float = 15.0
    w_heading: float = 4.0
    w_terrain_risk: float = 8.0
    w_memory: float = 30.0
    w_learned: float = 0.0

    # ── VSGP (from nav4)
    vsgp_n_inducing: int = 50
    vsgp_alpha: float = 1.0
    vsgp_length_scale: float = 0.3
    vsgp_noise_var: float = 0.05
    vsgp_lr: float = 0.02
    vsgp_update_freq: int = 10

    # ── Reactive Avoidance (from nav4)
    react_warn_dist: float = 14.0
    react_danger_dist: float = 8.5
    react_emergency_dist: float = 5.0
    react_forward_cone: float = 50.0

    # ── Control Limits (from nav4 + nav7)
    max_throttle: float = 0.95
    max_steer: float = 0.70
    max_brake: float = 0.65
    min_throttle_auto: float = 0.62
    min_throttle_hill: float = 0.78
    target_speed_flat: float = 12.0
    target_speed_bad: float = 7.0
    target_speed_near_goal: float = 4.5
    kp_speed: float = 0.32
    throttle_rate: float = 0.30

    # ── Stability (from nav4)
    max_safe_roll_deg: float = 8.0
    max_safe_pitch_deg: float = 11.0
    stability_throttle_scale: float = 0.30
    roll_corrective_steer: float = 0.5

    # ── Stuck Detection (from nav7 - better than nav4)
    stuck_speed_thresh: float = 0.20
    stuck_time_s: float = 1.50
    stuck_disp_thresh: float = 0.45
    recovery_reverse_s: float = 1.10
    recovery_turn_s: float = 1.00
    recovery_forward_s: float = 3.50
    recovery_reverse_throttle: float = 0.65
    recovery_forward_throttle: float = 0.95
    recovery_push_speed: float = 7.0
    memory_radius: float = 15.0

    # ── Obstacle Detection (from nav7 - only high objects)
    obstacle_z_min: float = 0.65
    obstacle_y_half_width: float = 2.9
    emergency_clearance: float = 2.20
    danger_clearance: float = 4.20

    # ── Visualization
    viz_win_w: int = 1280
    viz_win_h: int = 720
    viz_every_n_ticks: int = 2
    traj_history: int = 1200

    # ── Output
    out_dir: str = "unified_nav_results"


CFG = Config()


def apply_tuning_profile(cfg: Config) -> None:
    """One-line profile switch for speed/safety behavior."""
    profiles = {
        "AGGRESSIVE": {
            "min_throttle_auto": 0.66,
            "min_throttle_hill": 0.82,
            "target_speed_flat": 13.0,
            "target_speed_bad": 8.0,
            "target_speed_near_goal": 5.0,
            "kp_speed": 0.36,
            "throttle_rate": 0.34,
            "react_warn_dist": 11.0,
            "react_danger_dist": 6.8,
            "react_emergency_dist": 4.2,
            "max_safe_roll_deg": 9.5,
            "max_safe_pitch_deg": 13.0,
            "stability_throttle_scale": 0.38,
            "obstacle_z_min": 0.80,
            "obstacle_y_half_width": 2.5,
            "emergency_clearance": 1.80,
            "danger_clearance": 3.20,
        },
        "BALANCED": {
            "min_throttle_auto": 0.58,
            "min_throttle_hill": 0.74,
            "target_speed_flat": 11.0,
            "target_speed_bad": 6.5,
            "target_speed_near_goal": 4.2,
            "kp_speed": 0.30,
            "throttle_rate": 0.28,
            "react_warn_dist": 12.0,
            "react_danger_dist": 7.6,
            "react_emergency_dist": 4.6,
            "max_safe_roll_deg": 8.8,
            "max_safe_pitch_deg": 12.0,
            "stability_throttle_scale": 0.34,
            "obstacle_z_min": 0.72,
            "obstacle_y_half_width": 2.7,
            "emergency_clearance": 2.00,
            "danger_clearance": 3.70,
        },
        "SAFE": {
            "min_throttle_auto": 0.52,
            "min_throttle_hill": 0.68,
            "target_speed_flat": 9.5,
            "target_speed_bad": 5.5,
            "target_speed_near_goal": 3.6,
            "kp_speed": 0.24,
            "throttle_rate": 0.22,
            "react_warn_dist": 14.5,
            "react_danger_dist": 9.0,
            "react_emergency_dist": 5.2,
            "max_safe_roll_deg": 7.8,
            "max_safe_pitch_deg": 10.5,
            "stability_throttle_scale": 0.28,
            "obstacle_z_min": 0.62,
            "obstacle_y_half_width": 3.0,
            "emergency_clearance": 2.30,
            "danger_clearance": 4.40,
        },
    }

    profile = str(cfg.tuning_profile).upper().strip()
    if profile == "CURRENT":
        print("[TUNE] Profile CURRENT (using explicit config values)")
        return
    if profile not in profiles:
        print(f"[TUNE] Unknown profile '{cfg.tuning_profile}'. Using CURRENT values.")
        return

    for key, value in profiles[profile].items():
        setattr(cfg, key, value)
    print(f"[TUNE] Applied profile: {profile}")


apply_tuning_profile(CFG)


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §1  UTILITY FUNCTIONS
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def local_to_world(lx: float, ly: float, yaw: float, wx: float, wy: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return wx + c * lx - s * ly, wy + s * lx + c * ly


def world_to_local(dx: float, dy: float, yaw: float) -> Tuple[float, 
float]:
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §2  DATA CLASSES
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    reverse: bool = False


@dataclass
class VehicleState:
    x: float
    y: float
    z: float
    yaw: float
    pitch_deg: float
    roll_deg: float
    speed: float


@dataclass
class Candidate:
    wx: float
    wy: float
    lx: float
    ly: float
    alpha: float
    distance: float
    progress: float
    goal_dist: float
    heading_error: float
    slope: float
    roughness: float
    obstacle_risk: float
    flatness: float
    memory_penalty: float
    cost: float
    reward: float


@dataclass
class PerceptionOutput:
    """From nav4 - VSGP-based perception"""
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    occupancy_mean: np.ndarray
    occupancy_var: np.ndarray
    slope_map: np.ndarray
    roughness_map: np.ndarray
    traversability: np.ndarray
    raw_points: np.ndarray
    free_mask: Optional[np.ndarray] = None
    mean_surface: Optional[np.ndarray] = None
    ground_slope_deg: float = 0.0


@dataclass
class Subgoal:
    """From nav4 - VSGP subgoal structure"""
    alpha: float
    beta: float
    distance: float
    local_pos: np.ndarray
    world_pos: np.ndarray
    slope: float
    roughness: float
    occupancy: float
    variance: float
    traversability: float
    terrain_cost: float
    goal_progress: float
    heading_error: float
    safe: bool
    width_m: float
    cost: float = 0.0


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §3  VARIATIONAL SPARSE GAUSSIAN PROCESS (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha: float = 1.0, length_scale: float = 0.3, 
variance: float = 1.0):
        self.alpha, self.length_scale, self.variance = alpha, length_scale, variance

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if not len(X) or not len(Y):
            return np.zeros((len(X), len(Y)))
        diff = X[:, None, :] - Y[None, :, :]
        sq_d = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.variance * np.ones(len(X))


class VSGP:
    """Variational Sparse Gaussian Process for terrain height 
regression."""
    
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.kernel = RationalQuadraticKernel(
            alpha=cfg.vsgp_alpha,
            length_scale=cfg.vsgp_length_scale,
        )
        self.noise_var = cfg.vsgp_noise_var
        self.n_ind = cfg.vsgp_n_inducing
        self.lr = cfg.vsgp_lr
        self.update_freq = cfg.vsgp_update_freq

        side = int(math.sqrt(self.n_ind))
        A, B = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, side),
            np.linspace(-np.pi / 4, np.pi / 4, side),
        )
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]
        m = len(self.Z)
        self.mu = np.zeros(m)
        self.Su = np.eye(m) * 0.1
        self._Kuu_inv: Optional[np.ndarray] = None
        self._trained = False
        self._update_count = 0
        self._last_X_hash = 0

    def _select_inducing(self, X: np.ndarray) -> None:
        n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50, 
random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            self.Z = X[np.random.choice(len(X), n, replace=False)].copy()
        m = len(self.Z)
        self.mu = np.zeros(m)
        self.Su = np.eye(m) * 0.1
        self._Kuu_inv = None

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return
        self._update_count += 1
        if self._update_count % (50 * self.update_freq) == 0:
            self._select_inducing(X)
        h = hash(X.tobytes())
        if h == self._last_X_hash and self._trained:
            return
        self._last_X_hash = h
        m = len(self.Z)
        Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6
        try:
            self._Kuu_inv = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            self._Kuu_inv = np.linalg.pinv(Kuu)
        Kfu = self.kernel(X, self.Z)
        A = Kfu @ self._Kuu_inv
        ni = 1.0 / self.noise_var
        Lambda = ni * (A.T @ A) + self._Kuu_inv
        rhs = ni * (A.T @ y)
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
            return np.zeros(len(Xs)), np.ones(len(Xs)) * self.noise_var
        Ksu = self.kernel(Xs, self.Z)
        A = Ksu @ self._Kuu_inv
        mean = A @ self.mu
        Kss_diag = self.kernel.diag(Xs)
        var_exp = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
        var_var = np.sum(A @ self.Su * A, axis=1)
        var = np.clip(Kss_diag - var_exp + var_var + self.noise_var, 1e-6, 
None)
        return mean, var


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §4  PERCEPTION MODULE (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class PerceptionModule:
    """VSGP-based terrain perception from nav4"""
    ALPHA_RES = 60
    BETA_RES = 30
    ROC = 5.0
    SLOPE_WEIGHT = 1.2
    OCC_WEIGHT = 1.0
    VAR_FREE_WEIGHT = 1.0
    ROUGHNESS_WEIGHT = 0.8

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.vsgp = VSGP(cfg)
        self._last_output: Optional[PerceptionOutput] = None
        AG, BG = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES),
            np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES),
        )
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG, self._BG = AG, BG

    def _estimate_ground_slope(self, pts: np.ndarray) -> float:
        if len(pts) < 10:
            return 0.0
        z_med = np.median(pts[:, 2])
        gp = pts[np.abs(pts[:, 2] - z_med) < 1.0]
        if len(gp) < 10:
            return 0.0
        centered = gp - gp.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        cos_ang = np.abs(normal[2]) / (np.linalg.norm(normal) + 1e-9)
        return float(np.clip(np.degrees(np.arccos(np.clip(cos_ang, -1, 
1))), 0, 90))

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        """
        Process LiDAR data and generate traversability + terrain maps.
        Supports both CARLA raw measurement and precomputed numpy arrays.
        """
        # --- 1. Extract raw points ---
        try:
            if isinstance(measurement, np.ndarray):
                raw = measurement[:, :3]
            else:
                raw = np.frombuffer(
                    measurement.raw_data,
                    dtype=np.float32
                ).reshape(-1, 4)[:, :3]
        except Exception as e:
            print(f"[LiDAR] Failed to parse measurement: {e}")
            return self._last_output

        # --- 2. Basic sanity check ---
        if raw.shape[0] < 20:
            return self._last_output

        # --- 3. Distance filtering ---
        dist = np.linalg.norm(raw, axis=1)
        pts = raw[(dist > 0.5) & (dist < self.cfg.lidar_range)]
        if pts.shape[0] < 10:
            return self._last_output

        # --- 4. Ground slope estimation ---
        ground_slope = self._estimate_ground_slope(pts)

        # --- 5. Convert to spherical coordinates ---
        r = np.linalg.norm(pts, axis=1) + 1e-9  # avoid divide-by-zero
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta = np.arcsin(np.clip(pts[:, 2] / r, -1.0, 1.0))
        X = np.column_stack((alpha, beta))
        y = self.ROC - r

        # --- 6. Subsample for GP efficiency ---
        if X.shape[0] > 2000:
            idx = np.random.choice(X.shape[0], 2000, replace=False)
            X = X[idx]
            y = y[idx]
            pts = pts[idx]

        # --- 7. Update GP model ---
        try:
            self.vsgp.update(X, y)
            mean, var = self.vsgp.predict(self._grid_pts)
        except Exception as e:
            print(f"[LiDAR] GP failure: {e}")
            return self._last_output

        # --- 8. Reshape outputs ---
        try:
            mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
            var = var.reshape(self.BETA_RES, self.ALPHA_RES)
        except Exception as e:
            print(f"[LiDAR] Reshape error: {e}")
            return self._last_output

        # --- 9. Terrain derivatives ---
        da = np.gradient(mean, axis=1)  # alpha direction
        db = np.gradient(mean, axis=0)  # beta direction
        slope = np.sqrt(da ** 2 + db ** 2)
        roughness = np.sqrt(
            np.gradient(da, axis=1) ** 2 +
            np.gradient(db, axis=0) ** 2
        )

        # --- 10. Normalization helper ---
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-6:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        slope_n = _norm(slope)
        rough_n = _norm(roughness)
        occ_n = _norm(mean)
        var_n = _norm(var)

        # --- 11. Traversability map ---
        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT * occ_n
            - self.SLOPE_WEIGHT * slope_n
            - self.VAR_FREE_WEIGHT * var_n
            - self.ROUGHNESS_WEIGHT * rough_n,
            0.0, 1.0,
        )

        # --- 12. Build output ---
        out = PerceptionOutput(
            alpha_grid=self._AG,
            beta_grid=self._BG,
            occupancy_mean=mean,
            occupancy_var=var,
            slope_map=slope_n,
            roughness_map=rough_n,
            traversability=traversability,
            raw_points=pts,
            free_mask=traversability > 0.5,
            mean_surface=mean,
            ground_slope_deg=ground_slope,
        )

        # --- 13. Cache + return ---
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §5  TERRAIN ANALYZER (from nav7)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class TerrainAnalyzer:
    """Terrain analysis from nav7 - complements VSGP perception"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_points: Optional[np.ndarray] = None

    def update(self, pts: Optional[np.ndarray]) -> None:
        if pts is not None and len(pts) > 50:
            self.last_points = pts

    def local_risk(self, lx: float, ly: float, radius: float = 4.5) -> Tuple[float, float, float, float]:
        """Returns (slope, roughness, obstacle_risk, flatness)"""
        pts = self.last_points
        if pts is None or len(pts) < 30:
            return 0.12, 0.10, 0.0, 0.85
        dx = pts[:, 0] - lx
        dy = pts[:, 1] - ly
        local = pts[(dx * dx + dy * dy) < radius * radius]
        if len(local) < 8:
            return 0.25, 0.20, 0.05, 0.70
        z = local[:, 2]
        z_span = float(np.percentile(z, 90) - np.percentile(z, 10))
        rough = clamp(z_span / 3.0, 0.0, 1.0)
        A = np.column_stack([local[:, 0], local[:, 1], 
np.ones(len(local))])
        try:
            coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
            slope_raw = math.sqrt(float(coeff[0] * coeff[0] + coeff[1] * 
coeff[1]))
        except Exception:
            slope_raw = 0.4
        slope = clamp(slope_raw / 0.9, 0.0, 1.0)

        ground = float(np.percentile(z, 20))
        high = local[z > ground + 1.20]
        obs = 0.0
        if len(high) > 0:
            hd = np.sqrt((high[:, 0] - lx) ** 2 + (high[:, 1] - ly) ** 2)
            obs = clamp(1.0 - float(np.min(hd)) / 6.0, 0.0, 1.0)
        flatness = clamp(1.0 - 0.65 * slope - 0.35 * rough - 0.25 * obs, 
0.0, 1.0)
        return slope, rough, obs, flatness

    def forward_clearance(self) -> float:
        """High-object clearance only. Does NOT treat hill/ground points 
as obstacle."""
        pts = self.last_points
        if pts is None or len(pts) < 20:
            return self.cfg.lidar_range
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = (
            (x > 1.5) & (x < self.cfg.lidar_range)
            & (np.abs(y) < self.cfg.obstacle_y_half_width)
            & (z > self.cfg.obstacle_z_min)
        )
        if not np.any(mask):
            return self.cfg.lidar_range
        return float(np.min(x[mask]))


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §6  VSGP SUBGOAL PLANNER (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class VSGPSubgoalPlanner:
    """Goal-aware VSGP subgoal planner from nav4"""
    
    OCC_FREE_THRESH = 0.6
    VAR_STABLE_THRESH = 0.7
    SLOPE_SAFE_THRESH = 0.7
    ROUGH_SAFE_THRESH = 0.7

    W_TERRAIN = 2.8
    W_GOAL_DIST = 3.5
    W_HEADING = 0.9
    W_SLOPE = 2.4
    W_ROUGH = 1.8
    W_OCC = 2.2

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.min_dist = cfg.subgoal_min_distance
        self.max_dist = cfg.subgoal_distance
        self.n_angles = cfg.subgoal_num_angles
        self.n_depth = cfg.subgoal_num_depth

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pos: np.ndarray,
        vehicle_yaw_rad: float,
        goal_pos: np.ndarray,
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        if perc.occupancy_mean is None:
            return None, []

        dx_world = goal_pos[0] - vehicle_pos[0]
        dy_world = goal_pos[1] - vehicle_pos[1]
        goal_yaw = math.atan2(dy_world, dx_world)
        goal_rel_yaw = self._norm_angle(goal_yaw - vehicle_yaw_rad)

        alphas = perc.alpha_grid[0]
        betas = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        occ = perc.occupancy_mean
        var = perc.occupancy_var
        slope = perc.slope_map
        rough = perc.roughness_map
        trav = perc.traversability

        half_fov = math.pi / 2.2
        if abs(goal_rel_yaw) < half_fov:
            scan_centre = goal_rel_yaw
        else:
            scan_centre = 0.0

        alpha_range = np.linspace(
            scan_centre - half_fov,
            scan_centre + half_fov,
            self.n_angles,
        )

        candidates: List[Subgoal] = []

        for a in alpha_range:
            a_idx = int(np.argmin(np.abs(alphas - a)))
            if not (0 <= a_idx < len(alphas)):
                continue

            depths = np.linspace(self.min_dist, self.max_dist, 
self.n_depth)
            valid_points: List[Tuple[float, float, float, float, float, 
float]] = []

            for d in depths:
                o_val = float(occ[mid_beta_idx, a_idx])
                v_val = float(var[mid_beta_idx, a_idx])
                s_val = float(slope[mid_beta_idx, a_idx])
                r_val = float(rough[mid_beta_idx, a_idx])
                t_val = float(trav[mid_beta_idx, a_idx])

                is_free = o_val < self.OCC_FREE_THRESH
                is_stable = v_val < self.VAR_STABLE_THRESH
                is_safe = s_val < self.SLOPE_SAFE_THRESH and r_val < self.ROUGH_SAFE_THRESH

                if not (is_free and is_stable and is_safe):
                    continue

                valid_points.append((d, o_val, v_val, s_val, r_val, 
t_val))

            if not valid_points:
                continue

            best_d, best_o, best_v, best_s, best_r, best_t = valid_points[-1]

            lx = best_d * math.cos(a)
            ly = best_d * math.sin(a)

            wx = vehicle_pos[0] + (math.cos(vehicle_yaw_rad) * lx - math.sin(vehicle_yaw_rad) * ly)
            wy = vehicle_pos[1] + (math.sin(vehicle_yaw_rad) * lx + math.cos(vehicle_yaw_rad) * ly)

            dist_to_goal_from_cand = math.hypot(goal_pos[0] - wx, goal_pos[1] - wy)
            dist_to_goal_from_veh = math.hypot(goal_pos[0] - vehicle_pos[0], goal_pos[1] - vehicle_pos[1])
            goal_progress = dist_to_goal_from_veh - dist_to_goal_from_cand

            cand_yaw = vehicle_yaw_rad + a
            cand_goal_yaw = math.atan2(goal_pos[1] - wy, goal_pos[0] - wx)
            heading_err = abs(self._norm_angle(cand_goal_yaw - vehicle_yaw_rad))

            terrain_cost = (
                self.W_SLOPE * best_s +
                self.W_ROUGH * best_r +
                self.W_OCC * best_o +
                self.W_TERRAIN * (1.0 - best_t)
            )

            cost = (
                terrain_cost
                - self.W_GOAL_DIST * max(0.0, goal_progress)
                + self.W_HEADING * heading_err
            )

            sg = Subgoal(
                alpha=a,
                beta=0.0,
                distance=best_d,
                local_pos=np.array([lx, ly, 0.0], dtype=np.float32),
                world_pos=np.array([wx, wy, 0.0], dtype=np.float32),
                slope=best_s,
                roughness=best_r,
                occupancy=best_o,
                variance=best_v,
                traversability=best_t,
                terrain_cost=terrain_cost,
                goal_progress=goal_progress,
                heading_error=heading_err,
                safe=True,
                width_m=self.cfg.wheel_base,
                cost=cost,
            )
            candidates.append(sg)

        if not candidates:
            return None, []

        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates

    @staticmethod
    def _norm_angle(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §7  STUCK MEMORY (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class StuckMemory:
    """Memory of stuck locations from nav4"""
    
    def __init__(self, cfg: Config):
        self.decay = 0.995
        self.radius = cfg.memory_radius
        self.threshold = 0.05
        self.points: list = []

    def update(self, pos: np.ndarray, stuck: bool, failure_type: str = 
"stall") -> None:
        self.points = [
            (x, y, w * self.decay, t)
            for x, y, w, t in self.points
            if w * self.decay > self.threshold
        ]

        if not stuck:
            return

        for i, (px, py, pw, pt) in enumerate(self.points):
            if math.hypot(pos[0] - px, pos[1] - py) < self.radius * 0.5:
                self.points[i] = (px, py, min(1.0, pw + 0.5), pt)
                return

        self.points.append((float(pos[0]), float(pos[1]), 1.0, 
failure_type))
        print(f"[MEMORY] Stuck at ({pos[0]:.1f},{pos[1]:.1f}) type={failure_type} total={len(self.points)}")

    def get_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.points:
            return None
        type_map = {"stall": 0.0, "slip": 1.0, "obstacle": 2.0}
        data = [[p[0], p[1], p[2], type_map.get(p[3], 0.0)] for p in 
self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §8  LEARNED COST NETWORK (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class CostNet(nn.Module):
    """Simple neural network for learned costs from nav4"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §9  MPPI PLANNER (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class MPPIPlanner:
    """Model Predictive Path Integral Control from nav4"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() 
else "cpu")
        self.horizon = cfg.mppi_horizon
        self.num_samples = cfg.mppi_num_samples
        self.lambda_ = cfg.mppi_lambda
        self.dt = cfg.fixed_delta_seconds

        self._u_nom = torch.zeros((self.horizon, 2), device=self.device)
        self._noise = torch.zeros((self.num_samples, self.horizon, 2), 
device=self.device)
        self.cost_net = CostNet().to(self.device)
        self.cost_net.eval()

    def plan(
        self,
        state: List[float],
        perc: PerceptionOutput,
        goal_xy: np.ndarray,
        memory_points: Optional[torch.Tensor],
    ) -> Tuple[ControlState, float]:

        if perc is None or perc.occupancy_mean is None:
            return ControlState(), 0.0

        device = self.device
        self.goal = torch.tensor(goal_xy, device=device).float()
        self.mem_pts = memory_points

        self._u_nom = torch.roll(self._u_nom, -1, dims=0)
        self._u_nom[-1] = self._u_nom[-2]

        self._noise[:, :, 0].normal_(0, self.cfg.mppi_noise_throttle)
        self._noise[:, :, 1].normal_(0, self.cfg.mppi_noise_steer)
        u = (self._u_nom.unsqueeze(0) + self._noise).clone()
        u[:, :, 0] = u[:, :, 0].clamp(0.0, self.cfg.max_throttle)
        u[:, :, 1] = u[:, :, 1].clamp(-self.cfg.max_steer, 
self.cfg.max_steer)

        occ = torch.tensor(perc.occupancy_mean, device=device).float()
        slope = torch.tensor(perc.slope_map, device=device).float()
        rough = torch.tensor(perc.roughness_map, device=device).float()
        var = torch.tensor(perc.occupancy_var, device=device).float()
        trav = torch.tensor(perc.traversability, device=device).float()

        costs = self._rollout(state, u, occ, slope, rough, var, trav)

        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum()

        self._u_nom = (weights[:, None, None] * u).sum(dim=0).detach()

        u0 = self._u_nom[0]
        ctrl = ControlState(
            throttle=float(u0[0].cpu()),
            steer=float(u0[1].cpu()),
            brake=0.0,
            reverse=False,
        )
        return ctrl, float(beta.cpu())

    def _rollout(self, state, u, occ, slope, rough, var, trav):
        K, T, _ = u.shape
        device = self.device
        n_a, n_b = occ.shape[1], occ.shape[0]
        a_max = np.pi / 2

        x = torch.tensor(state, 
device=device).float().unsqueeze(0).repeat(K, 1)
        veh_x, veh_y, veh_yaw = x[0, 0].item(), x[0, 1].item(), x[0, 
2].item()
        total = torch.zeros(K, device=device)

        for t in range(T):
            throttle = u[:, t, 0]
            steer = u[:, t, 1]

            dx = x[:, 0] - veh_x
            dy = x[:, 1] - veh_y
            ry = torch.atan2(
                torch.sin(torch.atan2(dy, dx) - veh_yaw),
                torch.cos(torch.atan2(dy, dx) - veh_yaw),
            )
            ai = ((ry + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 
1)
            bi = torch.full_like(ry, n_b // 2, dtype=torch.long)
            sp_scale = torch.exp(-3.0 * (slope[bi, ai] + rough[bi, 
ai])).clamp(0.15, 1.0)

            v = x[:, 3] + throttle * sp_scale * self.dt
            yr = (v / self.cfg.wheel_base) * torch.tan(steer.clamp(-0.99, 
0.99))
            yaw = x[:, 2] + yr * self.dt
            xp = x[:, 0] + v * torch.cos(yaw) * self.dt
            yp = x[:, 1] + v * torch.sin(yaw) * self.dt
            x = torch.stack([xp, yp, yaw, v], dim=1)

            dx = xp - veh_x
            dy = yp - veh_y
            ry = torch.atan2(
                torch.sin(torch.atan2(dy, dx) - veh_yaw),
                torch.cos(torch.atan2(dy, dx) - veh_yaw),
            )
            ai = ((ry + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 
1)
            bi = torch.full_like(ry, n_b // 2, dtype=torch.long)

            gdx = self.goal[0] - xp
            gdy = self.goal[1] - yp
            gdist = torch.sqrt(gdx ** 2 + gdy ** 2)
            gh = torch.atan2(gdy, gdx)
            he = torch.atan2(torch.sin(gh - yaw), torch.cos(gh - yaw))

            occ_v = occ[bi, ai]
            slope_v = slope[bi, ai]
            rough_v = rough[bi, ai]
            var_v = var[bi, ai]
            trav_v = trav[bi, ai]

            cost = (
                5.0 * occ_v
                + 3.0 * slope_v
                + 2.0 * rough_v
                + 6.0 * var_v
                + self.cfg.w_goal * gdist
                + self.cfg.w_heading * he.abs()
                + 15.0 * (1.0 - trav_v) ** 2 * (1.0 + 0.2 * t)
                + 5.0 * (slope_v + rough_v) * v
                + 0.1 * (throttle ** 2 + steer ** 2)
                - 3.0 / (gdist + 1e-3)
            )

            collision = (occ_v > 0.5).float()
            cost += collision * 500.0

            if self.cfg.w_learned > 0.0:
                feat = torch.stack([
                    xp, yp, yaw, v,
                    occ_v, slope_v, rough_v, var_v,
                ], dim=1)
                cost += self.cfg.w_learned * self.cost_net(feat).squeeze()

            if self.mem_pts is not None:
                mx = self.mem_pts[:, 0]
                my = self.mem_pts[:, 1]
                mw = self.mem_pts[:, 2]
                ex = xp.unsqueeze(1) - mx.unsqueeze(0)
                ey = yp.unsqueeze(1) - my.unsqueeze(0)
                d2 = ex ** 2 + ey ** 2
                ang = torch.atan2(ey, ex)
                dir_cost = (mw.unsqueeze(0)
                            * torch.exp(-d2 / (2 * self.cfg.memory_radius 
** 2))
                            * (1.0 + 0.5 * torch.cos(ang - 
yaw.unsqueeze(1)))).sum(dim=1)
                ed = torch.sqrt(d2 + 1e-6)
                esc = -2.0 * torch.sqrt(
                    (mw.unsqueeze(0) * ex / (ed + 1e-3)).sum(dim=1) ** 2
                    + (mw.unsqueeze(0) * ey / (ed + 1e-3)).sum(dim=1) ** 2
                    + 1e-6
                )
                cost += self.cfg.w_memory * dir_cost + esc

            total += cost

        fdx = self.goal[0] - x[:, 0]
        fdy = self.goal[1] - x[:, 1]
        total += 25.0 * torch.sqrt(fdx ** 2 + fdy ** 2)
        return total


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §10  REACTIVE OBSTACLE AVOIDANCE (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class ReactiveObstacleAvoidance:
    """Last-resort obstacle avoidance from nav4"""
    
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._obstacle_active = False

    @property
    def obstacle_active(self) -> bool:
        return self._obstacle_active

    def process(
        self,
        raw_pts: Optional[np.ndarray],
        ctrl: ControlState,
        speed: float,
    ) -> ControlState:
        self._obstacle_active = False

        if raw_pts is None or len(raw_pts) < 5:
            return ctrl

        x, y, z = raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2]
        dist_2d = np.sqrt(x ** 2 + y ** 2)
        angle = np.degrees(np.arctan2(y, x))

        fwd_mask = (
            (x > 0.4) &
            (np.abs(angle) < self.cfg.react_forward_cone) &
            (z > -0.3) &
            (dist_2d < self.cfg.react_warn_dist * 1.5)
        )

        if not np.any(fwd_mask):
            return ctrl

        fwd_pts = raw_pts[fwd_mask]
        fwd_dist = np.sqrt(fwd_pts[:, 0] ** 2 + fwd_pts[:, 1] ** 2)
        min_dist = float(np.min(fwd_dist))
        closest_idx = int(np.argmin(fwd_dist))

        if min_dist > self.cfg.react_warn_dist:
            return ctrl

        self._obstacle_active = True
        out = ControlState(
            throttle=ctrl.throttle,
            steer=ctrl.steer,
            brake=ctrl.brake,
            reverse=ctrl.reverse,
        )

        if min_dist <= self.cfg.react_emergency_dist:
            out.throttle *= 0.5
            out.brake = self.cfg.max_brake
            out.reverse = True
            out.throttle = 0.3
            print(f"[REACTIVE] EMERGENCY STOP  dist={min_dist:.2f} m")

        elif min_dist <= self.cfg.react_danger_dist:
            ratio = (min_dist - self.cfg.react_emergency_dist) / (
                self.cfg.react_danger_dist - 
self.cfg.react_emergency_dist)
            out.throttle = ctrl.throttle * ratio * 0.2
            out.brake = self.cfg.max_brake * (1.0 - ratio * 0.4)
            obs_y = float(fwd_pts[closest_idx, 1])
            steer_dir = -np.sign(obs_y) if abs(obs_y) > 0.1 else -np.sign(ctrl.steer + 1e-6)
            steer_mag = min(self.cfg.max_steer, 0.4 / (min_dist + 0.1))
            out.steer = float(np.clip(
                ctrl.steer + steer_dir * steer_mag,
                -self.cfg.max_steer, self.cfg.max_steer,
            ))
            print(f"[REACTIVE] Danger  dist={min_dist:.2f} m  steer_corr={steer_dir * steer_mag:+.3f}")

        else:
            ratio = (min_dist - self.cfg.react_danger_dist) / (
                self.cfg.react_warn_dist - self.cfg.react_danger_dist)
            out.throttle = ctrl.throttle * (0.3 + ratio * 0.7)
            out.brake = ctrl.brake + (1.0 - ratio) * 0.25

        if min_dist < 2.5 and not out.reverse:
            out.reverse = True
            out.throttle = 0.3

        return out


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §11  CONTROLLER (from nav4 + nav7 hybrid)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class Controller:
    """Low-level controller with stability filters from nav4"""
    
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._state = ControlState()
        self._spd_ema = 0.0
        self._str_ema = 0.0

    def apply_safety_filters(
        self, ctrl: ControlState, roll_deg: float, pitch_deg: float
    ) -> ControlState:
        a = 0.3
        self._spd_ema = a * ctrl.throttle + (1 - a) * self._spd_ema
        self._str_ema = a * ctrl.steer + (1 - a) * self._str_ema

        th = self._spd_ema
        st = self._str_ema
        br = ctrl.brake

        if br > 0.1:
            th = 0.0

        st = float(np.clip(
            st,
            self._state.steer - 0.15,
            self._state.steer + 0.15,
        ))
        th = float(np.clip(
            th,
            self._state.throttle - 0.08,
            self._state.throttle + 0.08,
        ))

        ts, cs = self._stability(roll_deg, pitch_deg)
        th *= ts
        st = float(np.clip(st + cs, -self.cfg.max_steer, 
self.cfg.max_steer))

        st -= 0.2 * (st - self._state.steer)
        th -= 0.1 * (th - self._state.throttle)

        th = float(np.clip(th, 0.0, self.cfg.max_throttle))
        st = float(np.clip(st, -self.cfg.max_steer, self.cfg.max_steer))

        self._state = ControlState(throttle=th, steer=st, brake=br, 
reverse=ctrl.reverse)
        return self._state

    def _stability(self, roll: float, pitch: float) -> Tuple[float, 
float]:
        ts = 1.0
        cs = 0.0
        ar = abs(roll)
        if ar > self.cfg.max_safe_roll_deg:
            ex = ar - self.cfg.max_safe_roll_deg
            rf = 1.0 - min(ex / 15.0, 1.0)
            ts = min(ts, self.cfg.stability_throttle_scale
                     + (1.0 - self.cfg.stability_throttle_scale) * rf)
            cs = -math.copysign(
                min(self.cfg.roll_corrective_steer * ex / 10.0,
                    self.cfg.roll_corrective_steer), roll)
        ap = abs(pitch)
        if ap > self.cfg.max_safe_pitch_deg:
            ep = ap - self.cfg.max_safe_pitch_deg
            ts = min(ts, self.cfg.stability_throttle_scale
                     * (1.0 - min(ep / 10.0, 0.7)) + 0.15)
        return float(ts), float(cs)

    def reset(self) -> None:
        self._spd_ema = self._str_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §12  STUCK RECOVERY STATE MACHINE (from nav7 - better than nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class StuckRecovery:
    """Improved stuck recovery from nav7 with REVERSE → TURN → 
FORWARD_PUSH"""
    
    IDLE = "AUTO"
    REVERSE = "RECOVERY_REVERSE"
    TURN = "RECOVERY_TURN"
    FORWARD = "RECOVERY_FORWARD_PUSH"

    def __init__(self, cfg: Config, planner: 'FlatGoalPlanner'):
        self.cfg = cfg
        self.planner = planner
        self.mode = self.IDLE
        self.mode_start = 0.0
        self.hist: Deque[Tuple[float, float, float, float]] = deque(maxlen=400)
        self.stuck_start: Optional[float] = None
        self.recovery_steer = 0.55

    def force(self, t: float, st: VehicleState) -> None:
        self._start(t, st)

    def update(self, t: float, st: VehicleState, cmd: ControlState) -> None:
        self.hist.append((t, st.x, st.y, st.speed))
        
        if self.mode != self.IDLE:
            elapsed = t - self.mode_start
            if self.mode == self.REVERSE and elapsed > self.cfg.recovery_reverse_s:
                self.mode = self.TURN
                self.mode_start = t
            elif self.mode == self.TURN and elapsed > self.cfg.recovery_turn_s:
                self.mode = self.FORWARD
                self.mode_start = t
            elif self.mode == self.FORWARD and elapsed > self.cfg.recovery_forward_s:
                self.mode = self.IDLE
                self.mode_start = t
            return

        if cmd.throttle < 0.35 or cmd.brake > 0.1:
            self.stuck_start = None
            return
        if st.speed > self.cfg.stuck_speed_thresh:
            self.stuck_start = None
            return
        old = None
        for h in self.hist:
            if t - h[0] >= self.cfg.stuck_time_s:
                old = h
                break
        if old is None:
            return
        disp = math.hypot(st.x - old[1], st.y - old[2])
        if disp < self.cfg.stuck_disp_thresh:
            if self.stuck_start is None:
                self.stuck_start = t
            elif t - self.stuck_start > 0.5:
                self._start(t, st)
                self.stuck_start = None

    def _start(self, t: float, st: VehicleState) -> None:
        print(f"\n[RECOVERY] Vehicle stuck at ({st.x:.1f},{st.y:.1f}); using reverse + force push.")
        self.planner.add_memory(st.x, st.y)
        goal_heading = math.atan2(self.cfg.goal_y - st.y, self.cfg.goal_x 
- st.x)
        rel = angle_wrap(goal_heading - st.yaw)
        self.recovery_steer = -math.copysign(0.62, rel if abs(rel) > 0.1 
else random.choice([-1.0, 1.0]))
        self.mode = self.REVERSE
        self.mode_start = t

    def cmd(self) -> Optional[ControlState]:
        if self.mode == self.IDLE:
            return None
        if self.mode == self.REVERSE:
            return ControlState(
                throttle=self.cfg.recovery_reverse_throttle,
                steer=-self.recovery_steer,
                brake=0.0,
                reverse=True
            )
        if self.mode == self.TURN:
            return ControlState(
                throttle=0.65,
                steer=self.recovery_steer,
                brake=0.0,
                reverse=False
            )
        if self.mode == self.FORWARD:
            return ControlState(
                throttle=self.cfg.recovery_forward_throttle,
                steer=0.50 * self.recovery_steer,
                brake=0.0,
                reverse=False
            )
        return None


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §13  FLAT GOAL PLANNER (from nav7 - used alongside VSGP)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class FlatGoalPlanner:
    """Terrain-aware goal planner from nav7"""
    
    def __init__(self, cfg: Config, terrain: TerrainAnalyzer):
        self.cfg = cfg
        self.terrain = terrain
        self.best: Optional[Candidate] = None
        self.tick = 0
        self.memory: List[Tuple[float, float, float]] = []
        self.candidate_count = 81
        self.fov_deg = 170.0
        self.candidate_distances = (8.0, 14.0, 22.0, 32.0, 45.0)
        self.replan_every_ticks = 2
        self.w_progress = 260.0
        self.w_goal = 1.5
        self.w_heading = 9.0
        self.w_slope = 28.0
        self.w_rough = 14.0
        self.w_obstacle = 70.0
        self.w_memory = 80.0

    def add_memory(self, x: float, y: float) -> None:
        for i, (mx, my, w) in enumerate(self.memory):
            if math.hypot(x - mx, y - my) < self.cfg.memory_radius * 0.6:
                self.memory[i] = (mx, my, min(4.0, w + 1.0))
                return
        self.memory.append((x, y, 1.0))
        self.memory = self.memory[-30:]

    def memory_penalty(self, wx: float, wy: float) -> float:
        p = 0.0
        for mx, my, w in self.memory:
            d = math.hypot(wx - mx, wy - my)
            p += w * math.exp(-(d * d) / (2.0 * self.cfg.memory_radius ** 
2))
        return p

    def plan(self, st: VehicleState) -> Candidate:
        self.tick += 1
        if self.best is not None and self.tick % self.replan_every_ticks != 0:
            return self.best

        current_goal_dist = math.hypot(self.cfg.goal_x - st.x, self.cfg.goal_y - st.y)
        goal_heading = math.atan2(self.cfg.goal_y - st.y, self.cfg.goal_x - st.x)
        rel_goal = angle_wrap(goal_heading - st.yaw)
        center = clamp(rel_goal, -math.radians(80), math.radians(80))
        half = math.radians(self.fov_deg) / 2.0
        angles = np.linspace(center - half, center + half, 
self.candidate_count)
        candidates: List[Candidate] = []

        for dist in self.candidate_distances:
            for alpha in angles:
                lx = dist * math.cos(alpha)
                ly = dist * math.sin(alpha)
                if lx < 1.0:
                    continue
                wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
                new_gdist = math.hypot(self.cfg.goal_x - wx, 
self.cfg.goal_y - wy)
                progress = current_goal_dist - new_gdist
                heading_error = abs(angle_wrap(math.atan2(wy - st.y, wx - 
st.x) - st.yaw))
                slope, rough, obs, flat = self.terrain.local_risk(lx, ly, 
radius=5.0)
                mem = self.memory_penalty(wx, wy)
                cost = (
                    self.w_goal * new_gdist
                    - self.w_progress * progress
                    + self.w_heading * heading_error
                    + self.w_slope * slope
                    + self.w_rough * rough
                    + self.w_obstacle * obs
                    + self.w_memory * mem
                )
                reward = 120.0 * progress + 45.0 * flat - 25.0 * obs - 12.0 * slope - 8.0 * rough - 40.0 * mem
                candidates.append(Candidate(wx, wy, lx, ly, alpha, dist, progress, new_gdist, heading_error,
                                            slope, rough, obs, flat, mem, cost, reward))

        if not candidates:
            dist = min(25.0, current_goal_dist)
            lx = dist * math.cos(rel_goal)
            ly = dist * math.sin(rel_goal)
            wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
            slope, rough, obs, flat = self.terrain.local_risk(lx, ly, radius=5.0)
            self.best = Candidate(wx, wy, lx, ly, rel_goal, dist, 1.0, 
current_goal_dist, abs(rel_goal),
                                  slope, rough, obs, flat, 0.0, 
current_goal_dist, 0.0)
            return self.best

        candidates.sort(key=lambda c: c.cost)
        best = candidates[0]
        if max(c.progress for c in candidates) < 0.2:
            candidates.sort(key=lambda c: (abs(c.heading_error), 
c.goal_dist + 25.0 * c.obstacle_risk))
            best = candidates[0]
        self.best = best
        return best


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §14  MANUAL CONTROLLER (from nav4)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class ManualController:
    """WASD manual control from nav4"""
    
    THROTTLE_STEP = 0.04
    STEER_STEP = 0.06
    STEER_DECAY = 0.80

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._throttle = 0.0
        self._steer = 0.0
        self._reverse = False

    def tick(self, keys, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> ControlState:
        brake = 0.0

        if keys[pygame.K_w]:
            self._reverse = False
            self._throttle = min(self._throttle + self.THROTTLE_STEP, 
self.cfg.max_throttle)
        elif keys[pygame.K_s]:
            self._reverse = True
            self._throttle = min(self._throttle + self.THROTTLE_STEP, 
self.cfg.max_throttle)
        else:
            self._throttle = max(self._throttle - self.THROTTLE_STEP * 2, 
0.0)

        if keys[pygame.K_a]:
            self._steer = min(self._steer + self.STEER_STEP, 
self.cfg.max_steer)
        elif keys[pygame.K_d]:
            self._steer = max(self._steer - self.STEER_STEP, 
-self.cfg.max_steer)
        else:
            self._steer *= self.STEER_DECAY

        if keys[pygame.K_SPACE]:
            self._throttle = 0.0
            self._steer *= 0.5
            self._reverse = False
            brake = self.cfg.max_brake

        if abs(roll_deg) > self.cfg.max_safe_roll_deg:
            corr = -math.copysign(
                min(0.15 * (abs(roll_deg) - self.cfg.max_safe_roll_deg) / 
10.0, 0.15),
                roll_deg,
            )
            self._steer = float(np.clip(
                self._steer + corr, -self.cfg.max_steer, 
self.cfg.max_steer))

        return ControlState(
            throttle=round(self._throttle, 4),
            steer=round(self._steer, 4),
            brake=brake,
            reverse=self._reverse,
        )


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §15  SENSOR MANAGER (from nav7 - better queue handling)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    """Improved sensor management from nav7 with 
_put_latest/_get_latest"""
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, cfg: 
Config):
        self.world = world
        self.vehicle = vehicle
        self.cfg = cfg
        self.actors: List[carla.Actor] = []
        self.lidar_q: Queue = Queue(maxsize=2)
        self.front_q: Queue = Queue(maxsize=2)
        self.rear_q: Queue = Queue(maxsize=2)
        self._setup()

    @staticmethod
    def _put_latest(q: Queue, data) -> None:
        """Put data, dropping oldest if queue is full"""
        while q.full():
            try:
                q.get_nowait()
            except Empty:
                break
        q.put(data)

    @staticmethod
    def _img_to_array(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        return arr[:, :, ::-1].copy()

    @staticmethod
    def _get_latest(q: Queue):
        """Get the most recent item, discarding older ones"""
        item = None
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                break
        return item

    def _setup(self) -> None:
        bp = self.world.get_blueprint_library()
        
        # LiDAR
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(self.cfg.lidar_range))
        lidar_bp.set_attribute("channels", str(self.cfg.lidar_channels))
        lidar_bp.set_attribute("points_per_second", 
str(self.cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", 
str(self.cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("upper_fov", str(self.cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov", str(self.cfg.lidar_lower_fov))
        lidar = self.world.spawn_actor(
            lidar_bp, 
            carla.Transform(carla.Location(x=0.7, z=2.25)), 
            attach_to=self.vehicle
        )
        lidar.listen(lambda data: self._put_latest(self.lidar_q, data))
        self.actors.append(lidar)

        # Front camera
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov", str(self.cfg.camera_fov))
        front = self.world.spawn_actor(
            cam_bp, 
            carla.Transform(carla.Location(x=1.7, z=1.9)), 
            attach_to=self.vehicle
        )
        front.listen(lambda img: self._put_latest(self.front_q, 
self._img_to_array(img)))
        self.actors.append(front)
        
        # Rear camera
        rear = self.world.spawn_actor(
            cam_bp, 
            carla.Transform(carla.Location(x=-1.7, z=1.9), 
carla.Rotation(yaw=180)), 
            attach_to=self.vehicle
        )
        rear.listen(lambda img: self._put_latest(self.rear_q, 
self._img_to_array(img)))
        self.actors.append(rear)

    def get_lidar_points(self) -> Optional[np.ndarray]:
        data = self._get_latest(self.lidar_q)
        if data is None:
            return None
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 
4)[:, :3]
        d = np.linalg.norm(pts[:, :2], axis=1)
        return pts[(d > 1.0) & (d < self.cfg.lidar_range)]

    def get_front_image(self):
        return self._get_latest(self.front_q)
    
    def get_rear_image(self):
        return self._get_latest(self.rear_q)

    def destroy(self) -> None:
        for a in self.actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §16  RESULT LOGGER (from nav7 - comprehensive CSV + SVG plots)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class ResultLogger:
    """Comprehensive logging from nav7"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.rows: List[dict] = []
        self.terrain_rows: List[dict] = []

    def log(
        self,
        t: float,
        st: VehicleState,
        cmd: ControlState,
        cand,
        goal_dist: float,
        mode: str,
        cost: float,
        reward: float,
    ) -> None:
        angular_velocity = (st.speed / max(self.cfg.wheel_base, 1e-6)) * math.tan(cmd.steer)
        # Support both flat-planner Candidate and VSGP Subgoal records.
        subgoal_x = getattr(cand, "wx", None)
        if subgoal_x is None:
            world_pos = getattr(cand, "world_pos", None)
            if world_pos is not None and len(world_pos) >= 2:
                subgoal_x = float(world_pos[0])
            else:
                subgoal_x = st.x

        subgoal_y = getattr(cand, "wy", None)
        if subgoal_y is None:
            world_pos = getattr(cand, "world_pos", None)
            if world_pos is not None and len(world_pos) >= 2:
                subgoal_y = float(world_pos[1])
            else:
                subgoal_y = st.y

        flatness = getattr(cand, "flatness", getattr(cand, "traversability", 0.0))
        slope = getattr(cand, "slope", 0.0)
        roughness = getattr(cand, "roughness", 0.0)
        progress = getattr(cand, "progress", getattr(cand, "goal_progress", 0.0))
        obstacle_risk = getattr(cand, "obstacle_risk", getattr(cand, "occupancy", 0.0))

        self.rows.append({
            "time": t, "x": st.x, "y": st.y, "z": st.z,
            "yaw": st.yaw, "pitch_deg": st.pitch_deg, "roll_deg": 
st.roll_deg,
            "speed": st.speed, "linear_velocity": st.speed,
            "angular_velocity": angular_velocity,
            "throttle": cmd.throttle, "steer": cmd.steer, "brake": 
cmd.brake,
            "reverse": int(cmd.reverse), "goal_distance": goal_dist,
            "mode": mode, "cost": cost, "reward": reward,
            "subgoal_x": subgoal_x, "subgoal_y": subgoal_y,
            "flatness": flatness, "slope": slope,
            "roughness": roughness, "progress": progress,
            "obstacle_risk": obstacle_risk,
        })

    def log_terrain_points(self, t: float, st: VehicleState, pts: 
Optional[np.ndarray]) -> None:
        if pts is None or len(pts) < 10:
            return
        step = max(1, len(pts) // 800)
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        for p in pts[::step]:
            lx, ly, lz = float(p[0]), float(p[1]), float(p[2])
            wx = st.x + c * lx - s * ly
            wy = st.y + s * lx + c * ly
            self.terrain_rows.append({"time": t, "x": wx, "y": wy, "z": 
st.z + lz})

    def save_csv(self) -> None:
        if not self.rows:
            return
        with open(os.path.join(self.cfg.out_dir, "navigation_log.csv"), 
"w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)

        def write_subset(name: str, keys: List[str]) -> None:
            with open(os.path.join(self.cfg.out_dir, name), "w", 
newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow({k: r[k] for k in keys})

        write_subset("controls.csv", ["time", "throttle", "steer", 
"brake", "reverse", "linear_velocity", "angular_velocity", "speed"])
        write_subset("trajectory.csv", ["time", "x", "y", "z", 
"goal_distance"])
        write_subset("cost_reward.csv", ["time", "cost", "reward", 
"goal_distance", "flatness", "slope", "roughness", "progress"])

        if self.terrain_rows:
            with open(os.path.join(self.cfg.out_dir, 
"terrain_samples.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, 
fieldnames=list(self.terrain_rows[0].keys()))
                w.writeheader()
                w.writerows(self.terrain_rows)

    def _line(self, x, y, title, ylabel, filename) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, filename), 
format="svg")
        plt.close()

    def save_plots(self) -> None:
        if not self.rows:
            return
        t = np.array([r["time"] for r in self.rows])
        x = np.array([r["x"] for r in self.rows])
        y = np.array([r["y"] for r in self.rows])
        speed = np.array([r["speed"] for r in self.rows])
        cost = np.array([r["cost"] for r in self.rows])
        reward = np.array([r["reward"] for r in self.rows])
        lin = np.array([r["linear_velocity"] for r in self.rows])
        ang = np.array([r["angular_velocity"] for r in self.rows])
        gdist = np.array([r["goal_distance"] for r in self.rows])

        self._line(t, cost, "NMPC-style Cost Function", "Cost", 
"nmpc_cost.svg")
        self._line(t, reward, "Reward", "Reward", "reward.svg")
        self._line(t, lin, "Linear Velocity", "m/s", 
"linear_velocity.svg")
        self._line(t, ang, "Angular Velocity", 
"rad/s", "angular_velocity.svg")
        self._line(t, speed, "Vehicle Speed", "m/s", "vehicle_speed.svg")
        self._line(t, gdist, "Distance to Goal", "m", "goal_distance.svg")

        plt.figure(figsize=(7, 7))
        plt.plot(x, y, linewidth=2, label="trajectory")
        plt.scatter([self.cfg.spawn_x], [self.cfg.spawn_y], marker="o", 
s=70, label="start")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", 
s=90, label="goal")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("World Trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "trajectory.svg"), 
format="svg")
        plt.close()

        plt.figure(figsize=(7, 7))
        sc = plt.scatter(x, y, c=speed, s=18)
        plt.colorbar(sc, label="Speed [m/s]")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", 
s=90)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Vehicle Speed Heat Map")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "speed_heatmap.svg"), 
format="svg")
        plt.close()

        if len(self.terrain_rows) > 50:
            tx = np.array([r["x"] for r in self.terrain_rows])
            ty = np.array([r["y"] for r in self.terrain_rows])
            tz = np.array([r["z"] for r in self.terrain_rows])
            plt.figure(figsize=(7, 7))
            sc = plt.scatter(tx, ty, c=tz, s=5)
            plt.colorbar(sc, label="Terrain height proxy [m]")
            plt.plot(x, y, linewidth=1.5)
            plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=90)
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title("Flatness / Terrain Height Heat Map")
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.out_dir, 
"flatness_heatmap.svg"), format="svg")
            plt.close()

    def finalize(self) -> None:
        print(f"\n[SAVE] Saving results in {os.path.abspath(self.cfg.out_dir)}")
        self.save_csv()
        self.save_plots()
        print("[SAVE] Done")


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §17  VISUALIZATION MODULE (from nav4 - 6-panel with 3D rainbow LiDAR)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    """6-panel visualization from nav4 with 3D rainbow LiDAR"""
    
    C = {
        'bg': (17, 17, 17),
        'panel': (26, 26, 46),
        'header': (35, 35, 65),
        'grid': (45, 45, 70),
        'border': (60, 60, 95),
        'traj': (0, 206, 201),
        'veh': (253, 203, 110),
        'goal': (255, 60, 60),
        'subgoal': (50, 255, 120),
        'memory': (255, 110, 50),
        'cost': (255, 107, 107),
        'speed': (85, 239, 196),
        'steer': (162, 155, 254),
        'white': (230, 230, 230),
        'dim': (110, 110, 110),
        'danger': (255, 50, 50),
        'ok': (80, 220, 80),
        'warn': (255, 200, 0),
    }

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._screen: Optional[pygame.Surface] = None
        self._font_s: Optional[pygame.font.Font] = None

        mx = cfg.traj_history
        self._traj_x: deque = deque(maxlen=mx)
        self._traj_y: deque = deque(maxlen=mx)
        self._costs: deque = deque(maxlen=400)
        self._speeds: deque = deque(maxlen=400)
        self._rewards: deque = deque(maxlen=400)

        self._front_img: Optional[np.ndarray] = None
        self._rear_img: Optional[np.ndarray] = None
        self._lidar_pts: Optional[np.ndarray] = None

        self._goal = np.array([cfg.goal_x, cfg.goal_y])
        self._subgoal: Optional[np.ndarray] = None
        self._subgoal_candidates: List[np.ndarray] = []
        self._mem_pts: list = []

        self._mode = "AUTO"
        self._speed = 0.0
        self._steer = 0.0
        self._cost = 0.0
        self._reward = 0.0
        self._dist_goal = 0.0
        self._obs_warn = False
        self._pitch = 0.0
        self._roll = 0.0
        self._flatness = 0.0
        self._progress = 0.0

    def init(self, screen: pygame.Surface) -> None:
        self._screen = screen
        pygame.font.init()
        self._font_s = pygame.font.SysFont("monospace", 11)

    def push(
        self, *,
        pos: np.ndarray,
        speed: float,
        steer: float,
        perc: Optional[PerceptionOutput],
        mode: str,
        cost: float,
        reward: float,
        subgoal: Optional[np.ndarray],
        all_candidates: Optional[List[Subgoal]],
        mem_pts: list,
        dist_goal: float,
        obs_warn: bool,
        flatness: float = 0.0,
        progress: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        self._traj_x.append(float(pos[0]))
        self._traj_y.append(float(pos[1]))
        self._costs.append(cost)
        self._speeds.append(speed)
        self._rewards.append(reward)
        self._mode = mode
        self._speed = speed
        self._steer = steer
        self._cost = cost
        self._reward = reward
        self._subgoal = subgoal.copy() if subgoal is not None else None
        self._subgoal_candidates = [
            c.world_pos for c in all_candidates
        ] if all_candidates else []
        self._mem_pts = list(mem_pts)
        self._dist_goal = dist_goal
        self._obs_warn = obs_warn
        self._flatness = flatness
        self._progress = progress
        self._pitch = pitch
        self._roll = roll
        if perc is not None and perc.raw_points is not None and len(perc.raw_points):
            self._lidar_pts = perc.raw_points.copy()

    def set_front(self, img: np.ndarray) -> None:
        self._front_img = img

    def set_rear(self, img: np.ndarray) -> None:
        self._rear_img = img

    def render(self) -> None:
        if self._screen is None:
            return

        W, H = self._screen.get_size()
        self._screen.fill(self.C['bg'])

        CAM_H = 240
        col_w = W // 3
        bot_y = CAM_H
        bot_h = H - bot_y
        map_w = W // 2

        # Top row: Front camera, Rear camera, 3D Rainbow LiDAR
        self._draw_camera(self._front_img, "Front camera", 0, 0, col_w, 
CAM_H)
        self._draw_camera(self._rear_img, "Rear camera", col_w, 0, col_w, 
CAM_H)
        self._draw_lidar_3d("3D Rainbow LiDAR", col_w * 2, 0, W - col_w * 
2, CAM_H)
        
        # Bottom row: Trajectory/Subgoal, MPPI Cost, Speed+Reward
        self._draw_trajectory(0, bot_y, map_w, bot_h)
        self._draw_cost_plot(map_w, bot_y, W - map_w, bot_h // 2)
        self._draw_speed_reward_plot(map_w, bot_y + bot_h // 2, W - map_w, 
bot_h - bot_h // 2)
        
        self._draw_hud(W, H)

        pygame.display.flip()

    def _draw_camera(self, img, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'], (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x 
+ 4, y + 3))
        if img is not None:
            try:
                img_r = self._resize(img, w, h - 18)
                surf = pygame.surfarray.make_surface(img_r.swapaxes(0, 1))
                self._screen.blit(surf, (x, y + 18))
            except Exception:
                pass
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_lidar_3d(self, title, x, y, w, h):
        """Draw 3D rainbow-colored LiDAR points"""
        C = self.C
        pygame.draw.rect(self._screen, C['panel'], (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x 
+ 4, y + 3))

        content_y = y + 18
        content_h = h - 18

        lr = self.cfg.lidar_range
        cx = x + w // 2
        cy = content_y + content_h // 2
        scale = (min(w, content_h) * 0.44) / lr

        # Draw range rings
        for ring_r in [lr * 0.33, lr * 0.67, lr]:
            pr = int(ring_r * scale)
            pygame.draw.circle(self._screen, C['grid'], (cx, cy), pr, 1)
        pygame.draw.line(self._screen, C['grid'], (cx, cy), (cx, cy - 
int(lr * scale)), 1)

        if self._lidar_pts is not None and len(self._lidar_pts):
            pts = self._lidar_pts
            # Rainbow coloring based on z-height
            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            z_range = max(z_max - z_min, 1e-6)
            zn = (pts[:, 2] - z_min) / z_range
            
            for i in range(0, len(pts), 3):
                px = int(cx - pts[i, 1] * scale)
                py = int(cy - pts[i, 0] * scale)
                if x <= px < x + w and content_y <= py < content_y + content_h:
                    t = float(zn[i])
                    # Rainbow: red -> yellow -> green -> cyan -> blue -> magenta
                    if t < 0.2:
                        r = 255
                        g = int(255 * (t / 0.2))
                        b = 0
                    elif t < 0.4:
                        r = int(255 * ((0.4 - t) / 0.2))
                        g = 255
                        b = 0
                    elif t < 0.6:
                        r = 0
                        g = 255
                        b = int(255 * ((t - 0.4) / 0.2))
                    elif t < 0.8:
                        r = 0
                        g = int(255 * ((0.8 - t) / 0.2))
                        b = 255
                    else:
                        r = int(255 * ((t - 0.8) / 0.2))
                        g = 0
                        b = 255
                    pygame.draw.circle(self._screen, (r, g, b), (px, py), 
1)

        pygame.draw.circle(self._screen, C['veh'], (cx, cy), 4)
        pygame.draw.line(self._screen, C['veh'], (cx, cy), (cx, cy - 14), 
2)
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_trajectory(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'], (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render("Trajectory / Subgoal Map", 
True, C['white']),
                          (x + 4, y + 3))

        tx = list(self._traj_x)
        ty = list(self._traj_y)
        if len(tx) < 1:
            pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)
            return

        ax = list(tx) + [self._goal[0]]
        ay = list(ty) + [self._goal[1]]
        for px2, py2, pw2, _ in self._mem_pts:
            ax.append(px2)
            ay.append(py2)
        if self._subgoal is not None:
            ax.append(self._subgoal[0])
            ay.append(self._subgoal[1])
        for c in self._subgoal_candidates:
            if c is not None:
                ax.append(c[0])
                ay.append(c[1])

        span = max(max(ax) - min(ax), max(ay) - min(ay), 20.0)
        pad = span * 0.12
        mn_x, mx_x = min(ax) - pad, max(ax) + pad
        mn_y, mx_y = min(ay) - pad, max(ay) + pad

        px0 = x + 6
        py0 = y + 22
        pw = w - 12
        ph = h - 26

        def w2s(wx, wy):
            sx = px0 + int((wx - mn_x) / (mx_x - mn_x) * pw)
            sy = py0 + int((1.0 - (wy - mn_y) / (mx_y - mn_y)) * ph)
            return sx, sy

        for gx2 in np.linspace(mn_x, mx_x, 5):
            sx, _ = w2s(gx2, mn_y)
            pygame.draw.line(self._screen, C['grid'], (sx, py0), (sx, py0 
+ ph), 1)
        for gy2 in np.linspace(mn_y, mx_y, 5):
            _, sy = w2s(mn_x, gy2)
            pygame.draw.line(self._screen, C['grid'], (px0, sy), (px0 + 
pw, sy), 1)

        for px2, py2, pw2, _ in self._mem_pts:
            sx, sy = w2s(px2, py2)
            r = max(4, int(pw2 * 14))
            blob = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(blob, (*C['memory'], 90), (r, r), r)
            self._screen.blit(blob, (sx - r, sy - r))
            pygame.draw.circle(self._screen, C['memory'], (sx, sy), 4)

        for i, c in enumerate(self._subgoal_candidates):
            if c is not None:
                sx, sy = w2s(c[0], c[1])
                colour = C['subgoal'] if i == 0 else tuple(v // 2 for v in 
C['subgoal'])
                pygame.draw.circle(self._screen, colour, (sx, sy), 4 if i 
== 0 else 2)

        if len(tx) > 1:
            pts = [w2s(tx[i], ty[i]) for i in range(len(tx))]
            pygame.draw.lines(self._screen, C['traj'], False, pts, 2)

        if tx:
            vx2, vy2 = w2s(tx[-1], ty[-1])
            pygame.draw.circle(self._screen, C['veh'], (vx2, vy2), 6)
            pygame.draw.circle(self._screen, C['white'], (vx2, vy2), 6, 1)

        gsx, gsy = w2s(self._goal[0], self._goal[1])
        for d in [(-7, -7, 7, 7), (-7, 7, 7, -7)]:
            pygame.draw.line(self._screen, C['goal'],
                             (gsx + d[0], gsy + d[1]), (gsx + d[2], gsy + 
d[3]), 2)

        if self._subgoal is not None:
            ssx, ssy = w2s(self._subgoal[0], self._subgoal[1])
            pygame.draw.circle(self._screen, C['subgoal'], (ssx, ssy), 6)
            pygame.draw.circle(self._screen, C['white'], (ssx, ssy), 6, 1)

        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_cost_plot(self, x, y, w, h):
        self._draw_graph(list(self._costs), "MPPI Cost", x, y, w, h, 
self.C['cost'])

    def _draw_speed_reward_plot(self, x, y, w, h):
        C = self.C
        half = h // 2
        pygame.draw.rect(self._screen, C['panel'], (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(
            f"Speed: {self._speed:.2f} m/s  Reward: {self._reward:.2f}", True, C['speed']), 
            (x + 4, y + 3))
        
        if len(self._speeds) > 2:
            data = list(self._speeds)
            mn, mx = min(data), max(data)
            rng = max(mx - mn, 1e-3)
            px0, py0 = x + 5, y + 22
            pw, ph = w - 10, h - 26
            n = len(data)
            pts = [
                (px0 + int(i / (n - 1) * pw),
                 py0 + int((1 - (v - mn) / rng) * ph))
                for i, v in enumerate(data)
            ]
            if len(pts) > 1:
                pygame.draw.lines(self._screen, C['speed'], False, pts, 2)
        
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_graph(self, data, title, x, y, w, h, color):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'], (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        lbl = f"{title}   {data[-1]:.2f}" if data else title
        self._screen.blit(self._font_s.render(lbl, True, color), (x + 4, y 
+ 3))

        if len(data) < 2:
            pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)
            return

        px0, py0 = x + 5, y + 22
        pw, ph = w - 10, h - 26
        mn, mx = min(data), max(data)
        rng = max(mx - mn, 1e-3)

        n = len(data)
        pts = [
            (px0 + int(i / (n - 1) * pw),
             py0 + int((1 - (v - mn) / rng) * ph))
            for i, v in enumerate(data)
        ]
        if len(pts) > 1:
            pygame.draw.lines(self._screen, color, False, pts, 2)

        self._screen.blit(self._font_s.render(f"{mx:.1f}", True, 
C['dim']), (px0 + 2, py0 + 1))
        self._screen.blit(self._font_s.render(f"{mn:.1f}", True, 
C['dim']), (px0 + 2, py0 + ph - 12))
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_hud(self, W: int, H: int) -> None:
        C = self.C
        mode_c = C['ok'] if self._mode == "AUTO" else C['warn']

        lines = [
            (f"MODE : {self._mode}", mode_c),
            (f"Speed: {self._speed:5.2f} m/s", C['speed']),
            (f"Steer: {self._steer:+6.3f}", C['steer']),
            (f"Cost : {self._cost:8.1f}", C['cost']),
            (f"Reward: {self._reward:8.1f}", C['ok']),
            (f"Goal : {self._dist_goal:6.1f} m", C['white']),
            (f"Flat : {self._flatness:6.2f}", C['dim']),
            (f"Prog : {self._progress:6.2f} m", C['dim']),
            (f"Pitch: {self._pitch:5.1f} deg", C['dim']),
            (f"Roll : {self._roll:5.1f} deg", C['dim']),
            (f"Mem  : {len(self._mem_pts)}", C['dim']),
            (f"Cands: {len(self._subgoal_candidates)}", C['dim']),
        ]
        if self._obs_warn:
            lines.append(("!!! OBSTACLE !!!", C['danger']))

        BW, BH = 185, len(lines) * 17 + 8
        bx = W - BW - 4
        by = 4

        bg = pygame.Surface((BW, BH), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        self._screen.blit(bg, (bx, by))

        for i, (txt, col) in enumerate(lines):
            surf = self._font_s.render(txt, True, col)
            self._screen.blit(surf, (bx + 5, by + 4 + i * 17))

    @staticmethod
    def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        if h <= 0 or w <= 0:
            return img
        sh, sw = img.shape[:2]
        ri = (np.arange(h) * sh / h).astype(int).clip(0, sh - 1)
        ci = (np.arange(w) * sw / w).astype(int).clip(0, sw - 1)
        return img[np.ix_(ri, ci)]


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §18  NAVIGATION SYSTEM (hybrid of nav4 + nav7)
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    """Unified navigation system combining best of nav4 + nav7"""
    
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._mode = "AUTO"
        self._running = False

        self._client = None
        self._world = None
        self._vehicle = None
        self._sensors: Optional[SensorManager] = None
        self._original_settings = None

        # ── From nav4: Perception & Planning Stack 

        self._perception = PerceptionModule(cfg)  # VSGP-based
        self._subgoal_pl = VSGPSubgoalPlanner(cfg)  # Goal-aware VSGP
        self._mppi = MPPIPlanner(cfg)  # Torch-based MPPI
        self._cost_net = CostNet().to(self._mppi.device)
        self._cost_net.eval()
        self._stuck_memory = StuckMemory(cfg)
        self._reactive = ReactiveObstacleAvoidance(cfg)
        self._controller = Controller(cfg)

        # ── From nav7: Terrain & Recovery 

        self._terrain = TerrainAnalyzer(cfg)  # Local risk analysis
        self._flat_planner = FlatGoalPlanner(cfg, self._terrain)  # Backup planner
        self._recovery = StuckRecovery(cfg, self._flat_planner)  # Better recovery
        self._logger = ResultLogger(cfg)

        # ── Manual Control 

        self._manual = ManualController(cfg)

        # ── Visualization (nav4 style with 3D rainbow LiDAR) ──────────────
        self._viz = VisualizationModule(cfg)

        self._goal = np.array([cfg.goal_x, cfg.goal_y], dtype=np.float32)
        self._step = 0
        self._t0 = 0.0
        self._start_wall = time.time()
        self._nav6_prev_throttle = 0.0

        pygame.init()
        self._screen = pygame.display.set_mode(
            (cfg.viz_win_w, cfg.viz_win_h), pygame.RESIZABLE)
        pygame.display.set_caption(
            "Unified VSGP-MPPI Navigator (nav4+nav7) | TAB=AUTO/MANUAL | ESC=quit | R=recover")
        self._viz.init(self._screen)

    def connect(self) -> None:
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port}")
        self._client = carla.Client(self.cfg.carla_host, 
self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        
        # Use ONLY the world already loaded - no map parsing
        self._world = self._client.get_world()
        print("[CARLA] Using currently loaded world without OpenDRIVE/map parsing")

        if self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode = True
            s.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            s.no_rendering_mode = False
            self._world.apply_settings(s)
            print("[CARLA] Synchronous mode ON")

        for _ in range(5):
            self._world.tick() if self.cfg.synchronous else self._world.wait_for_tick()

    def _safe_world_tick(self, n: int = 1) -> None:
        for _ in range(max(1, n)):
            if self.cfg.synchronous:
                self._world.tick()
            else:
                self._world.wait_for_tick()

    def _fixed_spawn_candidates(self) -> List[Tuple[str, 
carla.Transform]]:
        """Return user-defined transforms with multiple 
z-offsets"""
        candidates: List[Tuple[str, carla.Transform]] = []
        for dz in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            tf = carla.Transform(
                carla.Location(x=self.cfg.spawn_x, y=self.cfg.spawn_y, 
z=self.cfg.spawn_z + dz),
                carla.Rotation(pitch=self.cfg.spawn_pitch, 
yaw=self.cfg.spawn_yaw, roll=self.cfg.spawn_roll),
            )
            candidates.append((f"fixed_z+{dz:.1f}", tf))
        return candidates

    def spawn_vehicle(self) -> None:
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.find(self.cfg.vehicle_blueprint)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")

        transforms_to_try = self._fixed_spawn_candidates()

        self._vehicle = None
        spawn_source = "none"
        last_error = None
        for source, tf in transforms_to_try:
            try:
                self._vehicle = self._world.try_spawn_actor(bp, tf)
            except RuntimeError as e:
                last_error = e
                print(f"[WARN] Spawn attempt failed at {source}: {e}")
                self._vehicle = None
            if self._vehicle is not None:
                spawn_source = source
                break

        if self._vehicle is None:
            raise RuntimeError(
                "Could not spawn vehicle using the fixed Config spawn pose. "
                "Change Config.spawn_x/spawn_y/spawn_z/spawn_yaw to a free position in your loaded map."
            )

        self._vehicle.set_simulate_physics(True)
        self._vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, 
hand_brake=False, manual_gear_shift=False)
        )
        self._safe_world_tick(int(self.cfg.post_spawn_wait_s / 
self.cfg.fixed_delta_seconds))
        self._vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0, 
hand_brake=False, manual_gear_shift=False)
        )

        tf = self._vehicle.get_transform()
        print(f"[SPAWN] Cybertruck from {spawn_source}: x={tf.location.x:.2f}, y={tf.location.y:.2f}, z={tf.location.z:.2f}, yaw={tf.rotation.yaw:.2f}")
        print(f"[GOAL] x={self.cfg.goal_x:.2f}, y={self.cfg.goal_y:.2f}, tolerance={self.cfg.goal_tolerance:.2f} m")

        self._sensors = SensorManager(self._world, self._vehicle, 
self.cfg)

    def get_state(self) -> VehicleState:
        tf = self._vehicle.get_transform()
        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        return VehicleState(
            float(tf.location.x),
            float(tf.location.y),
            float(tf.location.z),
            math.radians(float(tf.rotation.yaw)),
            float(tf.rotation.pitch),
            float(tf.rotation.roll),
            float(speed)
        )

    def apply_control(self, cmd: ControlState) -> None:
        control = carla.VehicleControl(
            throttle=clamp(cmd.throttle, 0, 1),
            steer=clamp(cmd.steer, -1, 1),
            brake=clamp(cmd.brake, 0, 1),
            reverse=bool(cmd.reverse),
            hand_brake=False,
            manual_gear_shift=False
        )
        self._vehicle.apply_control(control)

    def force_velocity_push(self, st: VehicleState) -> None:
        """Last-resort push from nav7"""
        if self._recovery.mode == StuckRecovery.FORWARD and st.speed < 0.35:
            vx = self.cfg.recovery_push_speed * math.cos(st.yaw)
            vy = self.cfg.recovery_push_speed * math.sin(st.yaw)
            try:
                self._vehicle.set_target_velocity(carla.Vector3D(vx, vy, 
0.0))
            except Exception:
                pass

    def _apply_nav6_throttle_bias(
        self,
        ctrl: ControlState,
        st: VehicleState,
        goal_dist: float,
        slope: float,
        roughness: float,
        obstacle_risk: float,
    ) -> ControlState:
        """Apply nav6-style throttle floor and goal aggression."""
        out = ControlState(
            throttle=ctrl.throttle,
            steer=ctrl.steer,
            brake=ctrl.brake,
            reverse=ctrl.reverse,
        )

        # Respect braking/reverse/recovery phases first.
        if out.reverse or out.brake > 0.05 or self._recovery.mode != StuckRecovery.IDLE:
            self._nav6_prev_throttle = out.throttle
            return out

        bad = clamp(0.90 * slope + 0.55 * roughness + 0.75 * obstacle_risk, 0.0, 1.0)
        target_speed = (1.0 - bad) * self.cfg.target_speed_flat + bad * self.cfg.target_speed_bad
        if goal_dist < 30.0:
            target_speed = min(target_speed, self.cfg.target_speed_near_goal)
        target_speed *= max(0.60, 1.0 - 0.20 * abs(out.steer) / max(self.cfg.max_steer, 1e-6))

        min_thr = (
            self.cfg.min_throttle_hill
            if (abs(st.pitch_deg) > 7.0 or slope > 0.38)
            else self.cfg.min_throttle_auto
        )
        desired = clamp(self.cfg.kp_speed * (target_speed - st.speed), min_thr, self.cfg.max_throttle)

        # Keep MPPI intent if already stronger, but never below nav6 floor.
        out.throttle = max(out.throttle, desired)
        out.throttle = clamp(
            out.throttle,
            self._nav6_prev_throttle - self.cfg.throttle_rate,
            self._nav6_prev_throttle + self.cfg.throttle_rate,
        )

        # No near-zero throttle while still far from goal and basically stopped.
        if goal_dist > self.cfg.goal_tolerance and st.speed < 0.5:
            out.throttle = max(out.throttle, min_thr)

        out.throttle = clamp(out.throttle, 0.0, self.cfg.max_throttle)
        self._nav6_prev_throttle = out.throttle
        return out

    def run(self) -> None:
        self.connect()
        self.spawn_vehicle()
        self._running = True

        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        print("[CLEANUP] Stop vehicle and save results")
        try:
            if self._vehicle and self._vehicle.is_alive:
                self._vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, 
hand_brake=False)
                )
        except Exception:
            pass
        try:
            if self._sensors:
                self._sensors.destroy()
        except Exception:
            pass
        try:
            if self._vehicle and self._vehicle.is_alive:
                self._vehicle.destroy()
        except Exception:
            pass
        try:
            if self._world and self._original_settings:
                self._world.apply_settings(self._original_settings)
        except Exception:
            pass
        self._logger.finalize()
        pygame.quit()

    def _tick(self) -> None:
        # Advance world
        if self.cfg.synchronous:
            self._world.tick()
        else:
            self._world.wait_for_tick()
        self._step += 1

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_TAB:
                    self._mode = "MANUAL" if self._mode == "AUTO" else "AUTO"
                    self._controller.reset()
                    print(f"[MODE] {self._mode}")
                elif event.key == pygame.K_r:
                    self._recovery.force(time.time() - self._start_wall, 
self.get_state())
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, 
pygame.RESIZABLE)
                self._viz.init(self._screen)

        keys = pygame.key.get_pressed()

        t = time.time() - self._start_wall
        st = self.get_state()

        goal_dist = math.hypot(self.cfg.goal_x - st.x, self.cfg.goal_y - 
st.y)

        # Goal check
        if goal_dist < self.cfg.goal_tolerance:
            print("\n[NAV] Goal reached!")
            self.apply_control(ControlState(0.0, 0.0, 1.0, False))
            self._running = False
            return

        # Sensor data
        pts = self._sensors.get_lidar_points()
        self._terrain.update(pts)

        if self._step % 10 == 0:
            self._logger.log_terrain_points(t, st, pts)

        # Process LiDAR for perception
        perc = self._perception.process_lidar(pts) if pts is not None else self._perception.last_output

        front = self._sensors.get_front_image()
        rear = self._sensors.get_rear_image()
        if front is not None:
            self._viz.set_front(front)
        if rear is not None:
            self._viz.set_rear(rear)

        ctrl_state: Optional[ControlState] = None
        subgoal_for_viz: Optional[np.ndarray] = None
        all_candidates: Optional[List[Subgoal]] = None
        selected_candidate_for_log: Optional[Subgoal] = None
        current_cost = 0.0
        current_reward = 0.0
        current_flatness = 0.0
        current_progress = 0.0
        current_slope = 0.0
        current_roughness = 0.0
        current_obstacle_risk = 0.0
        mode_str = "AUTO"

        if self._mode == "MANUAL":
            ctrl_state = self._manual.tick(keys, st.roll_deg, 
st.pitch_deg)
            mode_str = "MANUAL"

        else:  # AUTO mode
            # Update recovery state machine (nav7)
            self._recovery.update(t, st, ctrl_state or ControlState())
            rec_cmd = self._recovery.cmd()
            mode_str = self._recovery.mode

            if rec_cmd is not None:
                ctrl_state = rec_cmd
                self._controller.reset()
            elif perc is not None:
                # Try VSGP subgoal planner (nav4)
                best_subgoal, all_candidates = self._subgoal_pl.plan(
                    perc,
                    vehicle_pos=np.array([st.x, st.y, st.z], 
dtype=np.float32),
                    vehicle_yaw_rad=st.yaw,
                    goal_pos=self._goal,
                )

                if best_subgoal is None:
                    # Fall back to flat planner (nav7)
                    cand = self._flat_planner.plan(st)
                    subgoal_for_viz = np.array([cand.wx, cand.wy, 0.0])
                    current_flatness = cand.flatness
                    current_progress = cand.progress
                    current_reward = cand.reward
                    current_slope = cand.slope
                    current_roughness = cand.roughness
                    current_obstacle_risk = cand.obstacle_risk

                    state_vec = [st.x, st.y, st.yaw, st.speed]
                    mem_tensor = self._stuck_memory.get_tensor(self._mppi.device)
                    ctrl_state, current_cost = self._mppi.plan(state_vec, 
perc, subgoal_for_viz, mem_tensor)
                    ctrl_state = self._controller.apply_safety_filters(ctrl_state, st.roll_deg, 
st.pitch_deg)

                    print(f"[FLAT-PLAN] progress={cand.progress:.2f}m flat={cand.flatness:.2f}")
                else:
                    state_vec = [st.x, st.y, st.yaw, st.speed]
                    mem_tensor = self._stuck_memory.get_tensor(self._mppi.device)

                    selected_subgoal = best_subgoal
                    subgoal_world = selected_subgoal.world_pos
                    ctrl_state, current_cost = self._mppi.plan(
                        state_vec, perc, subgoal_world, mem_tensor
                    )
                    ctrl_state = self._controller.apply_safety_filters(
                        ctrl_state, st.roll_deg, st.pitch_deg
                    )

                    # If primary subgoal yields near-stop behavior, try next feasible subgoals.
                    stalled_cmd = (
                        st.speed < 1.0 and
                        ctrl_state.throttle < max(0.25, 0.6 * self.cfg.min_throttle_auto)
                        and ctrl_state.brake < 0.25
                        and not ctrl_state.reverse
                    )
                    if stalled_cmd and all_candidates:
                        best_score = (
                            ctrl_state.throttle
                            - 0.7 * ctrl_state.brake
                            + 0.12 * selected_subgoal.goal_progress
                            + 0.10 * selected_subgoal.traversability
                        )
                        for alt in all_candidates[1:6]:
                            if alt.goal_progress < -0.5:
                                continue
                            alt_ctrl, alt_cost = self._mppi.plan(
                                state_vec, perc, alt.world_pos, mem_tensor
                            )
                            alt_ctrl = self._controller.apply_safety_filters(
                                alt_ctrl, st.roll_deg, st.pitch_deg
                            )
                            alt_score = (
                                alt_ctrl.throttle
                                - 0.7 * alt_ctrl.brake
                                + 0.12 * alt.goal_progress
                                + 0.10 * alt.traversability
                            )
                            if alt_score > best_score:
                                best_score = alt_score
                                selected_subgoal = alt
                                ctrl_state = alt_ctrl
                                current_cost = alt_cost

                        if selected_subgoal is not best_subgoal:
                            print(
                                "[SUBGOAL] switched to alternate candidate "
                                f"alpha={selected_subgoal.alpha:.2f} dist={selected_subgoal.distance:.1f}m "
                                f"progress={selected_subgoal.goal_progress:.2f}m"
                            )

                    subgoal_world = selected_subgoal.world_pos
                    subgoal_for_viz = subgoal_world
                    selected_candidate_for_log = selected_subgoal
                    current_flatness = selected_subgoal.traversability
                    current_progress = selected_subgoal.goal_progress
                    current_slope = selected_subgoal.slope
                    current_roughness = selected_subgoal.roughness
                    current_obstacle_risk = selected_subgoal.occupancy

                    print(
                        f"[SUBGOAL] alpha={selected_subgoal.alpha:.2f} dist={selected_subgoal.distance:.1f}m "
                        f"progress={selected_subgoal.goal_progress:.2f}m slope={selected_subgoal.slope:.2f} "
                        f"rough={selected_subgoal.roughness:.2f} cost={selected_subgoal.cost:.2f}"
                    )

        if ctrl_state is None:
            ctrl_state = ControlState(throttle=self.cfg.min_throttle_auto)
            print("[NAV] fallback: keep-moving forward")

        # Reactive obstacle avoidance (nav4)
        raw_pts = perc.raw_points if perc is not None else None
        ctrl_state = self._reactive.process(raw_pts, ctrl_state, st.speed)
        ctrl_state = self._apply_nav6_throttle_bias(
            ctrl=ctrl_state,
            st=st,
            goal_dist=goal_dist,
            slope=current_slope,
            roughness=current_roughness,
            obstacle_risk=current_obstacle_risk,
        )

        # Keep-moving guard: do not settle to a stop before reaching final goal.
        # Allow stopping only for close-range obstacle emergency handling.
        if (
            self._mode == "AUTO"
            and goal_dist > self.cfg.goal_tolerance
            and self._recovery.mode == StuckRecovery.IDLE
            and not ctrl_state.reverse
        ):
            clearance = self._terrain.forward_clearance()
            if clearance > self.cfg.react_emergency_dist and ctrl_state.brake < 0.2:
                min_keep = self.cfg.min_throttle_hill if abs(st.pitch_deg) > 7.0 else self.cfg.min_throttle_auto
                if st.speed < 0.8:
                    ctrl_state.throttle = max(ctrl_state.throttle, min_keep)
                    ctrl_state.brake = 0.0

        # Apply to CARLA
        self.apply_control(ctrl_state)

        # Force velocity push if stuck (nav7)
        self.force_velocity_push(st)

        # Cost and reward calculation
        cost_total = current_cost + 0.3 * ctrl_state.throttle * ctrl_state.throttle + 0.8 * ctrl_state.steer * ctrl_state.steer + 0.4 * ctrl_state.brake * ctrl_state.brake
        reward_total = current_reward + 8.0 * st.speed - 0.35 * goal_dist

        # Log data
        if selected_candidate_for_log is not None:
            cand = selected_candidate_for_log
        elif all_candidates and all_candidates[0] is not None:
            cand = all_candidates[0]
        else:
            cand = Candidate(st.x, st.y, 0, 0, 0, 0, current_progress, 
goal_dist, 0,
                           current_flatness, 0, 0, current_flatness, 0, 
current_cost, current_reward)
        self._logger.log(t, st, ctrl_state, cand, goal_dist, mode_str, 
cost_total, reward_total)

        # Print status
        if self._step % self.cfg.viz_every_n_ticks * 5 == 0:
            print(f"\r[NAV] t={t:6.1f}s dist={goal_dist:7.2f}m speed={st.speed:5.2f}m/s "
                  f"thr={ctrl_state.throttle:4.2f} steer={ctrl_state.steer:+5.2f} "
                  f"mode={mode_str:>22s} flat={current_flatness:4.2f} prog={current_progress:6.2f} "
                  f"cost={cost_total:8.2f}", end="", flush=True)

        # Render visualization
        self._viz.push(
            pos=np.array([st.x, st.y, st.z]),
            speed=st.speed,
            steer=ctrl_state.steer,
            perc=perc,
            mode=mode_str,
            cost=cost_total,
            reward=reward_total,
            subgoal=subgoal_for_viz,
            all_candidates=all_candidates,
            mem_pts=self._stuck_memory.points,
            dist_goal=goal_dist,
            obs_warn=self._reactive.obstacle_active,
            flatness=current_flatness,
            progress=current_progress,
            pitch=st.pitch_deg,
            roll=st.roll_deg,
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()


# ╔════════════════════════════════════════════════════════════════════════╔═══════════════════════════════════════════════════════════════════════════╗
# §19  ENTRY POINT
# ╚════════════════════════════════════════════════════════════════════════╚═══════════════════════════════════════════════════════════════════════════╝

# def main() -> None:
#     nav = NavigationSystem(CFG)
#     try:
#         nav.connect()
#         nav.spawn_vehicle()
#         nav.run()
#     except RuntimeError as exc:
#         print(f"[FATAL] {exc}")
#         traceback.print_exc()
#     except Exception:
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()
def main() -> None:
    nav = NavigationSystem(CFG)
    try:
        # connect() and spawn_vehicle() are called inside run()
        nav.run()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
