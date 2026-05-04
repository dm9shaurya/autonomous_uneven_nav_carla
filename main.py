from __future__ import annotations

import csv
import enum
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional sklearn
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found – clustering falls back to random")

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Pygame
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print("[FATAL] pygame not found. pip install pygame")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[FATAL] matplotlib not found. pip install matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CARLA
# ─────────────────────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print("[FATAL] carla package not found. Put CARLA egg on PYTHONPATH.")
    sys.exit(1)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §0  GOAL DEFINITION  ←  No intermediate waypoints. System plans its own path.
#     Only the final destination is required.
# ╚════════════════════════════════════════════════════════════════════════════╝
GOAL: Tuple[float, float] = (1062.65, -514.06)   # ← edit this


# ╔════════════════════════════════════════════════════════════════════════════╗
# §1  CONFIGURATION
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA Connection
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    carla_timeout: float = 20.0
    synchronous: bool = True
    fixed_delta_seconds: float = 0.05

    # ── Vehicle
    vehicle_blueprint: str = "vehicle.tesla.cybertruck"
    post_spawn_wait_s: float = 2.0
    wheel_base: float = 3.807

    # ── Spawn
    spawn_x: float = 891.35
    spawn_y: float = -890.16
    spawn_z: float = 11.69
    spawn_pitch: float = -40.03
    spawn_yaw: float = 93.85
    spawn_roll: float = 0.00

    # ── Goal
    goal_x: float = 1062.65
    goal_y: float = -514.06
    goal_tolerance: float = 7.0

    # ── Sensors
    lidar_range: float = 55.0
    lidar_points_per_sec: int = 200_000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int = 64
    lidar_upper_fov: float = 20.0
    lidar_lower_fov: float = -35.0
    camera_width: int = 480
    camera_height: int = 270
    camera_fov: float = 90.0
    semantic_camera_width: int = 320
    semantic_camera_height: int = 180

    # ── IMU
    imu_vibration_window: int = 20        # frames for vibration RMS
    imu_slip_threshold: float = 0.55      # lateral g threshold → slip
    imu_vibration_danger: float = 2.0    # vibration index → unsafe
    imu_vibration_warn: float = 0.40      # vibration index → warn
    imu_slip_danger: float = 0.90       # slip index → unsafe
    imu_bad_duration_s: float = 2.0       # seconds of bad IMU before safety trigger

    # ── Semantic terrain
    semantic_forward_rows_frac: float = 0.60   # use bottom 60% of image (ground)
    semantic_center_cols_frac: float = 0.40    # use centre 40% (vehicle path)
    sem_rock_slope_veto: float = 0.45          # rock + slope > this → veto
    sem_snow_slope_veto: float = 0.35          # snow + slope > this → veto

    # ── Topological memory
    topo_node_spacing: float = 4.0         # min distance between nodes (m)
    topo_max_nodes: int = 600
    topo_safe_cost_thresh: float = 0.40    # node cost below this = "safe"
    topo_backtrack_min_dist: float = 8.0   # min dist from vehicle to backtrack node
    topo_decay_per_tick: float = 0.9995    # per-tick quality decay
    topo_corridor_radius: float = 6.0      # radius to consider a node "nearby"
    topo_corridor_bonus: float = 3.0       # subgoal cost reduction near safe nodes

    # ── VSGP Subgoal Planning
    subgoal_distance: float = 10.0
    subgoal_min_distance: float = 4.0
    subgoal_num_angles: int = 12
    subgoal_num_depth: int = 12

    # ── MPPI
    mppi_horizon: int = 30
    mppi_num_samples: int = 1024
    mppi_lambda: float = 1.0
    mppi_noise_throttle: float = 0.2
    mppi_noise_steer: float = 0.3

    # ── Cost Weights
    w_goal: float = 400.0
    w_heading: float = 40.0
    w_terrain_risk: float = 8.0
    w_memory: float = 30.0
    w_learned: float = 0.0
    w_semantic: float = 10.0       # semantic terrain cost weight in MPPI
    w_imu: float = 2.0            # IMU vibration/slip cost weight in MPPI

    # ── VSGP
    vsgp_n_inducing: int = 100
    vsgp_alpha: float = 1.0
    vsgp_length_scale: float = 0.3
    vsgp_noise_var: float = 0.05
    vsgp_lr: float = 0.02
    vsgp_update_freq: int = 10

    # ── Reactive Avoidance
    react_warn_dist: float = 10.0
    react_danger_dist: float = 6.0
    react_emergency_dist: float = 3.8
    react_forward_cone: float = 50.0

    # ── Control Limits
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

    # ── Safety / Anti-tip
    max_safe_roll_deg: float = 30.0
    max_safe_pitch_deg: float = 35.0
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float = 0.5
    anti_tip_roll_warn: float = 15.0
    anti_tip_roll_danger: float = 40.0
    anti_tip_pitch_warn: float = 35.0
    anti_tip_pitch_danger: float = 60.0
    anti_tip_throttle_floor: float = 0.25
    anti_tip_speed_scale: float = 0.30

    # ── Speed-Steer Coupling
    speed_steer_coupling: float = 0.55
    min_steer_at_speed: float = 0.18

    # ── Terrain Speed Caps
    max_speed_on_slope: float = 6.0
    max_speed_on_rough: float = 6.5

    # ── Terrain feasibility / clearance
    vehicle_ground_clearance: float = 0.35
    max_step_height: float = 0.25
    cliff_curv_threshold: float = 0.70
    cliff_void_threshold: float = 0.45
    clearance_block_threshold: float = 0.85
    curvature_weight: float = 3.0
    void_weight: float = 5.0
    clearance_weight: float = 4.0

    # ── Stuck Detection
    stuck_speed_thresh: float = 0.00
    stuck_time_s: float = 1.50
    stuck_disp_thresh: float = 0.45
    recovery_reverse_s: float = 1.10
    recovery_turn_s: float = 1.00
    recovery_forward_s: float = 3.50
    recovery_reverse_throttle: float = 0.65
    recovery_forward_throttle: float = 0.95
    recovery_push_speed: float = 7.0
    memory_radius: float = 30.0

    # ── Obstacle Detection
    obstacle_z_min: float = 0.85
    obstacle_y_half_width: float = 2.4
    emergency_clearance: float = 2.0
    danger_clearance: float = 2.50

    # ── Mission stall
    mission_stall_ticks: int = 200

    # ── TerrainMemory (dense grid)
    terrain_memory_size: int = 600
    terrain_memory_resolution: float = 1.0
    terrain_memory_lidar_downsample: int = 50
    terrain_memory_cost_weight: float = 4.0
    terrain_memory_lookahead_steps: int = 3
    terrain_memory_lookahead_weight: float = 1.5

    # ── Visualization
    viz_win_w: int = 1280
    viz_win_h: int = 720
    viz_every_n_ticks: int = 2
    traj_history: int = 1200

    # ── Output
    out_dir: str = "unified_nav_results"


CFG = Config()


# ╔════════════════════════════════════════════════════════════════════════════╗
# §2  UTILITY FUNCTIONS
# ╚════════════════════════════════════════════════════════════════════════════╝

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def local_to_world(lx, ly, yaw, wx, wy):
    c, s = math.cos(yaw), math.sin(yaw)
    return wx + c * lx - s * ly, wy + s * lx + c * ly

def world_to_local(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


# ╔════════════════════════════════════════════════════════════════════════════╗
# §3  DATA CLASSES
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainClass(enum.IntEnum):
    UNKNOWN    = 0
    DIRT       = 1
    GRASS      = 2
    GRAVEL     = 3
    ROCK       = 4
    SNOW       = 5
    VEGETATION = 6
    ROAD       = 7

# Per-class traversability multiplier applied on top of geometric traversability
SEMANTIC_TRAV_MULT: Dict[TerrainClass, float] = {
    TerrainClass.UNKNOWN:    0.55,
    TerrainClass.DIRT:       0.92,
    TerrainClass.GRASS:      0.85,
    TerrainClass.GRAVEL:     0.75,
    TerrainClass.ROCK:       0.5,
    TerrainClass.SNOW:       0.28,
    TerrainClass.VEGETATION: 0.50,
    TerrainClass.ROAD:       1.00,
}

SEMANTIC_SPEED_CAP: Dict[TerrainClass, float] = {
    TerrainClass.UNKNOWN:    7.0,
    TerrainClass.DIRT:       12.0,
    TerrainClass.GRASS:      10.0,
    TerrainClass.GRAVEL:     9.0,
    TerrainClass.ROCK:       4.5,
    TerrainClass.SNOW:       3.5,
    TerrainClass.VEGETATION: 5.0,
    TerrainClass.ROAD:       15.0,
}


@dataclass
class ControlState:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    reverse: bool = False


@dataclass
class VehicleState:
    x: float; y: float; z: float
    yaw: float
    pitch_deg: float; roll_deg: float
    speed: float


@dataclass
class IMUData:
    """Processed IMU telemetry."""
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    vibration: float = 0.0      # normalised 0–1: RMS of high-freq accel deviation
    slip_index: float = 0.0     # normalised 0–1: lateral accel anomaly
    pitch_rate: float = 0.0     # deg/s
    roll_rate: float = 0.0      # deg/s
    stability_score: float = 1.0  # 0=unstable, 1=perfectly stable


@dataclass
class SemanticOutput:
    """Camera-derived semantic terrain classification."""
    dominant_class: TerrainClass = TerrainClass.UNKNOWN
    forward_class: TerrainClass = TerrainClass.UNKNOWN
    class_counts: Dict[int, int] = field(default_factory=dict)
    traversability_mult: float = 0.6
    speed_cap: float = 8.0
    semantic_risk: float = 0.35   # 0=safe, 1=very risky


@dataclass
class TopologicalNode:
    x: float; y: float
    cost: float = 0.5            # 0=excellent, 1=impassable
    visit_count: int = 0
    semantic_class: TerrainClass = TerrainClass.UNKNOWN
    geometric_cost: float = 0.5
    imu_vibration: float = 0.0
    is_safe: bool = True
    timestamp: float = 0.0


@dataclass
class Candidate:
    wx: float; wy: float; lx: float; ly: float
    alpha: float; distance: float; progress: float; goal_dist: float
    heading_error: float; slope: float; roughness: float
    obstacle_risk: float; flatness: float; clearance_risk: float
    memory_penalty: float; cost: float; reward: float


@dataclass
class PerceptionOutput:
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    occupancy_mean: np.ndarray
    occupancy_var: np.ndarray
    slope_map: np.ndarray
    roughness_map: np.ndarray
    curvature_map: np.ndarray
    void_risk_map: np.ndarray
    traversability: np.ndarray
    raw_points: np.ndarray
    free_mask: Optional[np.ndarray] = None
    mean_surface: Optional[np.ndarray] = None
    ground_slope_deg: float = 0.0
    void_risk: float = 0.0
    # New fused fields
    semantic: Optional[SemanticOutput] = None
    imu: Optional[IMUData] = None
    fused_traversability: Optional[np.ndarray] = None   # geo × semantic × imu


@dataclass
class Subgoal:
    alpha: float; beta: float; distance: float
    local_pos: np.ndarray; world_pos: np.ndarray
    slope: float; roughness: float; occupancy: float
    variance: float; traversability: float; terrain_cost: float
    goal_progress: float; heading_error: float; safe: bool
    width_m: float; cost: float = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §4  VARIATIONAL SPARSE GAUSSIAN PROCESS
# ╚════════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha=1.0, length_scale=0.3, variance=1.0):
        self.alpha, self.length_scale, self.variance = alpha, length_scale, variance

    def __call__(self, X, Y):
        if not len(X) or not len(Y):
            return np.zeros((len(X), len(Y)))
        diff = X[:, None, :] - Y[None, :, :]
        sq_d = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X):
        return self.variance * np.ones(len(X))


class VSGP:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.kernel = RationalQuadraticKernel(cfg.vsgp_alpha, cfg.vsgp_length_scale)
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

    def _select_inducing(self, X):
        n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50, random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            self.Z = X[np.random.choice(len(X), n, replace=False)].copy()
        m = len(self.Z)
        self.mu = np.zeros(m); self.Su = np.eye(m) * 0.1; self._Kuu_inv = None

    def update(self, X, y):
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

    def predict(self, Xs):
        if not self._trained or self._Kuu_inv is None:
            return np.zeros(len(Xs)), np.ones(len(Xs)) * self.noise_var
        Ksu = self.kernel(Xs, self.Z)
        A = Ksu @ self._Kuu_inv
        mean = A @ self.mu
        Kss_diag = self.kernel.diag(Xs)
        var_exp = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
        var_var = np.sum(A @ self.Su * A, axis=1)
        var = np.clip(Kss_diag - var_exp + var_var + self.noise_var, 1e-6, None)
        return mean, var


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5  PERCEPTION MODULE  (fused: geometry + semantics + IMU)
# ╚════════════════════════════════════════════════════════════════════════════╝

class PerceptionModule:
    ALPHA_RES = 60; BETA_RES = 30; ROC = 5.0
    SLOPE_WEIGHT = 1.2; OCC_WEIGHT = 1.0
    VAR_FREE_WEIGHT = 1.0; ROUGHNESS_WEIGHT = 3.0

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

    def _estimate_ground_slope(self, pts):
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
        return float(np.clip(np.degrees(np.arccos(np.clip(cos_ang, -1, 1))), 0, 90))

    def _directional_void_risk(
        self,
        pts: np.ndarray,
        alpha_centers: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate void / ditch risk per forward bearing.

        The previous implementation returned one scalar risk for every angle,
        which made ditches look equally bad in all directions.  This version
        keeps the risk local to each angular sector so the subgoal planner can
        naturally pick a safer side passage when one exists.
        """
        if pts is None or len(pts) < 10:
            void = 1.0
            return void, np.full((self.BETA_RES, len(alpha_centers)), void, dtype=np.float32)

        forward_pts = pts[pts[:, 0] > 1.5]
        if len(forward_pts) < 10:
            void = 1.0
            return void, np.full((self.BETA_RES, len(alpha_centers)), void, dtype=np.float32)

        angles = np.arctan2(forward_pts[:, 1], forward_pts[:, 0])
        ranges = np.linalg.norm(forward_pts[:, :2], axis=1)
        sector_ids = np.argmin(np.abs(angles[:, None] - alpha_centers[None, :]), axis=1)

        sector_void = np.ones(len(alpha_centers), dtype=np.float32)
        for idx in range(len(alpha_centers)):
            sector_ranges = np.sort(ranges[sector_ids == idx])
            if sector_ranges.size == 0:
                sector_void[idx] = 1.0
                continue

            density = np.clip(sector_ranges.size / 16.0, 0.0, 1.0)
            if sector_ranges.size > 1:
                gap = np.clip(np.max(np.diff(sector_ranges)) / 8.0, 0.0, 1.0)
                span = np.clip((sector_ranges[-1] - sector_ranges[0]) / self.cfg.lidar_range, 0.0, 1.0)
            else:
                gap = 0.35
                span = 0.0

            # Large empty sectors or sudden range jumps are hallmarks of a
            # ditch / cliff edge in the forward field of view.
            sector_void[idx] = np.clip(
                0.55 * (1.0 - density)
                + 0.25 * gap
                + 0.15 * span
                + 0.05 * float(sector_ranges[0] > 12.0),
                0.0, 1.0,
            )

        mid = len(alpha_centers) // 2
        lo = max(0, mid - 3)
        hi = min(len(alpha_centers), mid + 4)
        scalar_void = float(np.clip(np.mean(sector_void[lo:hi]), 0.0, 1.0))
        void_risk_map = np.repeat(sector_void[np.newaxis, :], self.BETA_RES, axis=0)
        return scalar_void, void_risk_map

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        try:
            if isinstance(measurement, np.ndarray):
                raw = measurement[:, :3]
            else:
                raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        except Exception as e:
            print(f"[LiDAR] parse error: {e}")
            return self._last_output

        if raw.shape[0] < 20:
            return self._last_output
        dist = np.linalg.norm(raw, axis=1)
        pts = raw[(dist > 0.5) & (dist < self.cfg.lidar_range)]
        if pts.shape[0] < 10:
            return self._last_output

        ground_slope = self._estimate_ground_slope(pts)
        r = np.linalg.norm(pts, axis=1) + 1e-9
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta = np.arcsin(np.clip(pts[:, 2] / r, -1.0, 1.0))
        X = np.column_stack((alpha, beta))
        y = self.ROC - r

        if X.shape[0] > 2000:
            idx = np.random.choice(X.shape[0], 2000, replace=False)
            X, y, pts = X[idx], y[idx], pts[idx]

        try:
            self.vsgp.update(X, y)
            mean, var = self.vsgp.predict(self._grid_pts)
        except Exception as e:
            print(f"[LiDAR] GP failure: {e}")
            return self._last_output

        try:
            mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
            var  = var.reshape(self.BETA_RES, self.ALPHA_RES)
        except Exception:
            return self._last_output

        da = np.gradient(mean, axis=1); db = np.gradient(mean, axis=0)
        slope = np.sqrt(da ** 2 + db ** 2)
        roughness = np.sqrt(np.gradient(da, axis=1) ** 2 + np.gradient(db, axis=0) ** 2)
        d2a = np.gradient(da, axis=1); d2b = np.gradient(db, axis=0)
        curvature = np.sqrt(d2a ** 2 + d2b ** 2)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return np.zeros_like(arr) if mx - mn < 1e-6 else (arr - mn) / (mx - mn)

        slope_n  = _norm(slope);  rough_n = _norm(roughness)
        curv_n   = _norm(curvature); occ_n = _norm(mean); var_n = _norm(var)
        void_risk, void_risk_map = self._directional_void_risk(pts, self._AG[0])

        geo_traversability = np.clip(
            1.0
            - self.OCC_WEIGHT       * occ_n
            - self.SLOPE_WEIGHT     * slope_n
            - self.VAR_FREE_WEIGHT  * var_n
            - self.ROUGHNESS_WEIGHT * rough_n
            - self.cfg.curvature_weight * curv_n
            - self.cfg.void_weight  * void_risk_map,
            0.0, 1.0,
        )

        out = PerceptionOutput(
            alpha_grid=self._AG, beta_grid=self._BG,
            occupancy_mean=mean, occupancy_var=var,
            slope_map=slope_n, roughness_map=rough_n,
            curvature_map=curv_n, void_risk_map=void_risk_map,
            traversability=geo_traversability, raw_points=pts,
            free_mask=geo_traversability > 0.5, mean_surface=mean,
            ground_slope_deg=ground_slope, void_risk=float(void_risk),
            fused_traversability=geo_traversability.copy(),
        )
        self._last_output = out
        return out

    def fuse(
        self,
        out: PerceptionOutput,
        semantic: Optional[SemanticOutput],
        imu: Optional[IMUData],
    ) -> PerceptionOutput:
        """Fuse semantic and IMU data into traversability."""
        if out is None:
            return out
        fused = out.traversability.copy()

        # Semantic multiplier: scales whole field by terrain type quality
        if semantic is not None:
            fused *= semantic.traversability_mult
            out.semantic = semantic

        # IMU stability modifier: reduce traversability when vehicle is bouncing/slipping
        if imu is not None:
            stability_pen = 1.0 - (
                0.5 * imu.vibration +
                0.3 * imu.slip_index +
                0.2 * clamp(abs(imu.pitch_rate) / 30.0, 0.0, 1.0)
            )
            fused *= clamp(stability_pen, 0.25, 1.0)
            out.imu = imu

        out.fused_traversability = np.clip(fused, 0.0, 1.0)
        return out

    @property
    def last_output(self):
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5.5  TERRAIN CLASSIFIER  (semantic segmentation + color sub-classification)
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainClassifier:
    """
    Classifies terrain type from CARLA's semantic segmentation camera and
    optional RGB image (for sub-classification within ground/terrain labels).

    CARLA semantic labels used:
      6  Road line  → ROAD
      7  Road       → ROAD
      8  Sidewalk   → ROAD  (treated as traversable hardstand)
      9  Vegetation → VEGETATION
      14 Ground     → sub-classified by color
      22 Terrain    → sub-classified by color
    """

    # CARLA label IDs
    _LABEL_ROAD       = {6, 7, 8}
    _LABEL_VEGETATION = {9}
    _LABEL_GROUND     = {14, 22}
    _LABEL_WATER      = {21}
    _LABEL_SKY        = {13}

    def classify(
        self,
        seg_img: Optional[np.ndarray],
        rgb_img: Optional[np.ndarray],
        cfg: Config,
    ) -> SemanticOutput:
        if seg_img is None:
            return SemanticOutput()

        h, w = seg_img.shape[:2]
        # CARLA semantic segmentation: raw BGRA → label in R channel (index 2)
        if seg_img.ndim == 3 and seg_img.shape[2] >= 3:
            labels = seg_img[:, :, 2].astype(np.int32)
        else:
            return SemanticOutput()

        # Focus on forward path: centre columns, lower rows (the ground ahead)
        r0 = int(h * (1.0 - cfg.semantic_forward_rows_frac))
        c0 = int(w * (1.0 - cfg.semantic_center_cols_frac) / 2.0)
        c1 = w - c0
        roi_labels = labels[r0:, c0:c1]
        roi_rgb    = rgb_img[r0:, c0:c1] if rgb_img is not None else None

        # Count label occurrences
        counts: Dict[int, int] = {}
        unique, cnts = np.unique(roi_labels, return_counts=True)
        for u, c in zip(unique.tolist(), cnts.tolist()):
            counts[u] = c

        total = max(roi_labels.size, 1)

        # Map CARLA labels to terrain classes
        road_frac   = sum(counts.get(l, 0) for l in self._LABEL_ROAD)     / total
        veg_frac    = sum(counts.get(l, 0) for l in self._LABEL_VEGETATION) / total
        ground_frac = sum(counts.get(l, 0) for l in self._LABEL_GROUND)   / total
        water_frac  = sum(counts.get(l, 0) for l in self._LABEL_WATER)    / total

        # Dominant coarse class
        if road_frac > 0.40:
            dominant = TerrainClass.ROAD
        elif veg_frac > 0.50:
            dominant = TerrainClass.VEGETATION
        elif water_frac > 0.10:
            dominant = TerrainClass.UNKNOWN   # water = bad
        elif ground_frac > 0.25:
            # Sub-classify ground using color
            dominant = self._subclassify_by_color(roi_rgb, roi_labels)
        else:
            dominant = TerrainClass.UNKNOWN

        # Forward class: just the centre strip of bottom quarter
        fwd_r0 = int(h * 0.75)
        fwd_c0 = int(w * 0.35); fwd_c1 = w - fwd_c0
        fwd_labels = labels[fwd_r0:, fwd_c0:fwd_c1]
        fwd_rgb    = rgb_img[fwd_r0:, fwd_c0:fwd_c1] if rgb_img is not None else None
        forward_class = self._subclassify_by_color(fwd_rgb, fwd_labels)

        trav_mult = SEMANTIC_TRAV_MULT.get(dominant, 0.55)
        speed_cap = SEMANTIC_SPEED_CAP.get(dominant, 7.0)
        risk = 1.0 - trav_mult

        return SemanticOutput(
            dominant_class=dominant,
            forward_class=forward_class,
            class_counts=counts,
            traversability_mult=trav_mult,
            speed_cap=speed_cap,
            semantic_risk=risk,
        )

    def _subclassify_by_color(
        self,
        rgb: Optional[np.ndarray],
        labels: np.ndarray,
    ) -> TerrainClass:
        """Use RGB mean/std to distinguish dirt/grass/gravel/rock/snow."""
        if rgb is None or rgb.size == 0:
            return TerrainClass.DIRT

        pix = rgb.reshape(-1, 3).astype(np.float32)
        if len(pix) == 0:
            return TerrainClass.DIRT

        r = float(pix[:, 0].mean())
        g = float(pix[:, 1].mean())
        b = float(pix[:, 2].mean())
        brightness = (r + g + b) / 3.0
        c_max = max(r, g, b)
        c_min = min(r, g, b)
        saturation = c_max - c_min   # 0–255

        # Snow: very bright, very low saturation
        if brightness > 210 and saturation < 25:
            return TerrainClass.SNOW
        # Grass: green dominant
        if g > r * 1.15 and g > b * 1.05 and saturation > 20:
            return TerrainClass.GRASS
        # Rock: gray (low saturation, medium brightness)
        if saturation < 30 and 50 < brightness < 180:
            return TerrainClass.ROCK
        # Gravel: light grey-brown
        if saturation < 55 and brightness > 140:
            return TerrainClass.GRAVEL
        # Dirt: reddish-brown
        if r > g * 1.05 and r > b * 1.15:
            return TerrainClass.DIRT
        return TerrainClass.DIRT


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5.6  IMU PROCESSOR
# ╚════════════════════════════════════════════════════════════════════════════╝

class IMUProcessor:
    """
    Processes CARLA IMU sensor readings into stability metrics.

    Computes:
      vibration   – RMS of high-frequency acceleration deviation (0–1)
      slip_index  – lateral acceleration anomaly (0–1)
      pitch_rate  – deg/s from gyroscope Y axis
      roll_rate   – deg/s from gyroscope X axis
      stability_score – combined scalar (0=very unstable, 1=stable)
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._buf: Deque[Tuple[float, float, float]] = deque(
            maxlen=cfg.imu_vibration_window
        )
        self._last = IMUData()

    def process(self, imu_measurement) -> IMUData:
        try:
            ax = float(imu_measurement.accelerometer.x)
            ay = float(imu_measurement.accelerometer.y)
            az = float(imu_measurement.accelerometer.z)
            gx = float(imu_measurement.gyroscope.x)
            gy = float(imu_measurement.gyroscope.y)
            gz = float(imu_measurement.gyroscope.z)
        except Exception:
            return self._last

        self._buf.append((ax, ay, az))

        # Vibration: RMS deviation from rolling mean
        if len(self._buf) >= 5:
            arr  = np.array(self._buf, dtype=np.float32)
            mean = arr.mean(axis=0)
            dev  = arr - mean
            vibration = float(np.sqrt((dev ** 2).mean()))
            vibration = clamp(vibration / 4.0, 0.0, 1.0)
        else:
            vibration = 0.0

        # Lateral slip: |ay| normalized (gravity-adjusted)
        g_lateral = abs(ay) / 9.81   # g-units
        slip_index = clamp(g_lateral / self.cfg.imu_slip_threshold, 0.0, 1.0)

        pitch_rate = math.degrees(gy)   # rad/s → deg/s
        roll_rate  = math.degrees(gx)

        stability = clamp(
            1.0
            - 0.45 * vibration
            - 0.35 * slip_index
            - 0.20 * clamp(abs(pitch_rate) / 35.0, 0.0, 1.0),
            0.0, 1.0,
        )

        self._last = IMUData(
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            vibration=vibration,
            slip_index=slip_index,
            pitch_rate=pitch_rate,
            roll_rate=roll_rate,
            stability_score=stability,
        )
        return self._last

    @property
    def last(self) -> IMUData:
        return self._last


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6  TERRAIN ANALYZER
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_points: Optional[np.ndarray] = None

    def update(self, pts: Optional[np.ndarray]) -> None:
        if pts is not None and len(pts) > 50:
            self.last_points = pts

    def local_risk(self, lx, ly, radius=4.5):
        pts = self.last_points
        if pts is None or len(pts) < 30:
            return 0.12, 0.10, 0.0, 0.85, 0.0
        dx = pts[:, 0] - lx; dy = pts[:, 1] - ly
        local = pts[(dx * dx + dy * dy) < radius * radius]
        if len(local) < 8:
            return 0.25, 0.20, 0.05, 0.70, 0.0
        z = local[:, 2]
        z_span      = float(np.percentile(z, 90) - np.percentile(z, 10))
        bump_height = float(np.percentile(z, 95) - np.percentile(z, 20))
        rough       = clamp(z_span / 3.0, 0.0, 1.0)
        A = np.column_stack([local[:, 0], local[:, 1], np.ones(len(local))])
        try:
            coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
            slope_raw = math.sqrt(float(coeff[0] ** 2 + coeff[1] ** 2))
        except Exception:
            slope_raw = 0.4
        slope  = clamp(slope_raw / 0.9, 0.0, 1.0)
        ground = float(np.percentile(z, 20))
        high   = local[z > ground + 1.20]
        obs    = 0.0
        if len(high) > 0:
            hd  = np.sqrt((high[:, 0] - lx) ** 2 + (high[:, 1] - ly) ** 2)
            obs = clamp(1.0 - float(np.min(hd)) / 6.0, 0.0, 1.0)
        if bump_height <= self.cfg.max_step_height:
            clearance_risk = 0.0
        elif bump_height >= self.cfg.vehicle_ground_clearance:
            clearance_risk = 1.0
        else:
            clearance_risk = clamp(
                (bump_height - self.cfg.max_step_height) /
                max(self.cfg.vehicle_ground_clearance - self.cfg.max_step_height, 1e-6),
                0.0, 1.0,
            )
        flatness = clamp(1.0 - 0.55*slope - 0.25*rough - 0.20*obs - 0.45*clearance_risk, 0.0, 1.0)
        return slope, rough, obs, flatness, clearance_risk

    def forward_clearance(self) -> float:
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


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6.5  TERRAIN MEMORY  (dense cost grid)
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainMemory:
    _EMA_ALPHA = 0.15
    _TICK_DECAY = 0.995

    def __init__(self, size=600, resolution=1.0):
        self.res = float(resolution); self.size = int(size)
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.conf = np.zeros((self.size, self.size), dtype=np.float32)

    def world_to_grid(self, x, y):
        gx = int(x / self.res) + self.size // 2
        gy = int(y / self.res) + self.size // 2
        return gx, gy

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.size and 0 <= gy < self.size

    def update(self, wx, wy, cost):
        gx, gy = self.world_to_grid(wx, wy)
        if self._in_bounds(gx, gy):
            self.grid[gx, gy] = (1 - self._EMA_ALPHA) * self.grid[gx, gy] + self._EMA_ALPHA * cost
            self.conf[gx, gy] += 1.0

    def mark_stuck(self, wx, wy, cost=1.0):
        gx, gy = self.world_to_grid(wx, wy)
        if self._in_bounds(gx, gy):
            self.grid[gx, gy] = float(clamp(cost, 0.0, 1.0))
            self.conf[gx, gy] += 1.0

    def get_cost(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        return float(self.grid[gx, gy]) if self._in_bounds(gx, gy) else 0.0

    def lookahead_cost(self, wx, wy, heading_rad, steps=3, step_m=1.0, weight=1.5):
        total = 0.0
        for t in range(1, steps + 1):
            fx = wx + t * step_m * math.cos(heading_rad)
            fy = wy + t * step_m * math.sin(heading_rad)
            total += weight * self.get_cost(fx, fy)
        return total

    def decay(self):
        self.grid *= self._TICK_DECAY

    def update_from_lidar(
        self,
        pts,
        vehicle_x,
        vehicle_y,
        downsample=50,
        alpha_grid: Optional[np.ndarray] = None,
        void_risk_map: Optional[np.ndarray] = None,
    ):
        if pts is None or len(pts) < downsample:
            return
        sampled = pts[::downsample]
        world_x  = vehicle_x + sampled[:, 0]
        world_y  = vehicle_y + sampled[:, 1]
        rough_cost = np.clip(np.abs(sampled[:, 2]) * 0.10, 0.0, 1.0)
        void_cost = None
        if alpha_grid is not None and void_risk_map is not None and void_risk_map.size > 0:
            try:
                alpha_centers = np.asarray(alpha_grid[0], dtype=np.float32)
                local_alpha = np.arctan2(sampled[:, 1], sampled[:, 0])
                sector_ids = np.argmin(np.abs(local_alpha[:, None] - alpha_centers[None, :]), axis=1)
                void_cost = np.asarray(void_risk_map[void_risk_map.shape[0] // 2, sector_ids], dtype=np.float32)
            except Exception:
                void_cost = None
        for i in range(len(sampled)):
            cost = float(rough_cost[i])
            if void_cost is not None:
                cost = float(np.clip(0.55 * cost + 0.45 * void_cost[i], 0.0, 1.0))
            self.update(float(world_x[i]), float(world_y[i]), cost)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6.6  TOPOLOGICAL MEMORY  (sparse node graph for corridor finding + backtrack)
# ╚════════════════════════════════════════════════════════════════════════════╝

class TopologicalMemory:
    """
    Sparse topological graph built from vehicle traversal history.

    Each node records position + accumulated traversal quality.
    Used for:
      • biasing subgoal planner toward proven-safe corridors
      • penalising known-bad zones
      • finding the best backtrack target during recovery

    Node quality is computed from speed, IMU vibration, slip, and semantic class.
    Nodes decay slowly so stale data fades without explicit clearing.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.nodes: List[TopologicalNode] = []
        self._last_node_x: float = float('nan')
        self._last_node_y: float = float('nan')

    # ── maintenance ────────────────────────────────────────────────────────

    def update(
        self,
        x: float, y: float, t: float,
        speed: float,
        imu: Optional[IMUData],
        semantic: Optional[SemanticOutput],
        geo_slope: float,
        geo_rough: float,
    ) -> None:
        """Record vehicle position and traversal quality."""
        # Decay all existing node costs slightly
        for n in self.nodes:
            n.cost = clamp(n.cost * self.cfg.topo_decay_per_tick, 0.0, 1.0)

        # Create new node if vehicle has moved far enough
        if not math.isnan(self._last_node_x):
            dist = math.hypot(x - self._last_node_x, y - self._last_node_y)
        else:
            dist = float('inf')

        if dist < self.cfg.topo_node_spacing:
            # Just update the most recent node quality
            if self.nodes:
                self._blend_node(self.nodes[-1], speed, imu, semantic, geo_slope, geo_rough)
            return

        # Compute node cost
        vib     = imu.vibration if imu else 0.3
        slip    = imu.slip_index if imu else 0.0
        sem_risk = semantic.semantic_risk if semantic else 0.35
        sem_cls  = semantic.dominant_class if semantic else TerrainClass.UNKNOWN

        quality_cost = clamp(
            0.30 * (1.0 - clamp(speed / 8.0, 0.0, 1.0))  # penalise slow progress
            + 0.25 * vib
            + 0.15 * slip
            + 0.20 * sem_risk
            + 0.10 * geo_slope,
            0.0, 1.0,
        )

        node = TopologicalNode(
            x=x, y=y,
            cost=quality_cost,
            visit_count=1,
            semantic_class=sem_cls,
            geometric_cost=clamp(0.5 * geo_slope + 0.5 * geo_rough, 0.0, 1.0),
            imu_vibration=vib,
            is_safe=quality_cost < self.cfg.topo_safe_cost_thresh,
            timestamp=t,
        )
        self.nodes.append(node)
        self._last_node_x = x
        self._last_node_y = y

        # Prune oldest nodes when capacity exceeded
        if len(self.nodes) > self.cfg.topo_max_nodes:
            self.nodes = self.nodes[-(self.cfg.topo_max_nodes):]

    def _blend_node(self, n: TopologicalNode, speed, imu, semantic, geo_slope, geo_rough):
        alpha = 0.15
        vib  = imu.vibration  if imu      else 0.3
        risk = semantic.semantic_risk if semantic else 0.35
        new_cost = clamp(
            0.30 * (1.0 - clamp(speed / 8.0, 0.0, 1.0))
            + 0.25 * vib + 0.20 * risk + 0.10 * geo_slope, 0.0, 1.0
        )
        n.cost = (1 - alpha) * n.cost + alpha * new_cost
        n.is_safe = n.cost < self.cfg.topo_safe_cost_thresh
        n.visit_count += 1

    def mark_stuck(self, x: float, y: float) -> None:
        """Hard-set nodes near (x, y) as unsafe."""
        radius = self.cfg.topo_node_spacing * 2
        for n in self.nodes:
            if math.hypot(n.x - x, n.y - y) < radius:
                n.cost = 1.0
                n.is_safe = False
        # Add a new bad node at the exact stuck point
        self.nodes.append(TopologicalNode(x=x, y=y, cost=1.0, is_safe=False))

    # ── read ───────────────────────────────────────────────────────────────

    def corridor_bias(self, wx: float, wy: float) -> float:
        """
        Return a cost adjustment for candidate position (wx, wy).
        Negative = prefer (safe corridor nearby), Positive = avoid (bad zone nearby).
        """
        r = self.cfg.topo_corridor_radius
        r2 = r * r
        best_safe = 0.0
        worst_bad = 0.0
        for n in self.nodes:
            d2 = (n.x - wx) ** 2 + (n.y - wy) ** 2
            if d2 > r2:
                continue
            proximity = 1.0 - math.sqrt(d2) / r
            if n.is_safe:
                best_safe = max(best_safe, proximity * (1.0 - n.cost))
            else:
                worst_bad = max(worst_bad, proximity * n.cost)
        return worst_bad * 4.0 - best_safe * self.cfg.topo_corridor_bonus

    def nearest_safe_backtrack(
        self, cx: float, cy: float
    ) -> Optional[TopologicalNode]:
        """
        Return the best safe node to backtrack to.
        Criteria: safe, at least topo_backtrack_min_dist away, lowest cost.
        """
        min_d = self.cfg.topo_backtrack_min_dist
        candidates = [
            n for n in self.nodes
            if n.is_safe and math.hypot(n.x - cx, n.y - cy) >= min_d
        ]
        if not candidates:
            return None
        # Prefer close safe nodes with low cost
        return min(candidates, key=lambda n: n.cost + 0.05 * math.hypot(n.x-cx, n.y-cy))

    def safe_node_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_safe)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §7  VSGP SUBGOAL PLANNER  (uses topological + semantic awareness)
# ╚════════════════════════════════════════════════════════════════════════════╝

class VSGPSubgoalPlanner:
    CURV_SAFE_THRESH  = 0.75
    VOID_SAFE_THRESH  = 0.45
    OCC_FREE_THRESH   = 0.6
    VAR_STABLE_THRESH = 0.7
    SLOPE_SAFE_THRESH = 0.7
    ROUGH_SAFE_THRESH = 0.7

    W_TERRAIN   = 3.5
    W_GOAL_DIST = 3.5
    W_HEADING   = 0.9
    W_SLOPE     = 2.8
    W_ROUGH     = 3.2
    W_OCC       = 1.2
    W_VOID      = 7.5

    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self.min_dist = cfg.subgoal_min_distance
        self.max_dist = cfg.subgoal_distance
        self.n_angles = cfg.subgoal_num_angles
        self.n_depth  = cfg.subgoal_num_depth

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pos: np.ndarray,
        vehicle_yaw_rad: float,
        goal_pos: np.ndarray,
        terrain_memory: Optional[TerrainMemory] = None,
        topo_memory: Optional[TopologicalMemory] = None,
        semantic: Optional[SemanticOutput] = None,
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        if perc.occupancy_mean is None:
            return None, []

        # curv = getattr(perc, "curvature_map", None) or np.zeros_like(perc.traversability)
        # void_map = getattr(perc, "void_risk_map", None) or np.zeros_like(perc.traversability)

        curv = getattr(perc, "curvature_map", None)
        if curv is None:
            curv = np.zeros_like(perc.traversability)
        void_map = getattr(perc, "void_risk_map", None)
        if void_map is None:
            void_map = np.zeros_like(perc.traversability)

        dx_world = goal_pos[0] - vehicle_pos[0]
        dy_world = goal_pos[1] - vehicle_pos[1]
        goal_yaw = math.atan2(dy_world, dx_world)
        goal_rel_yaw = self._norm_angle(goal_yaw - vehicle_yaw_rad)

        alphas       = perc.alpha_grid[0]
        betas        = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        # Use fused traversability if available
        trav = (perc.fused_traversability
                if perc.fused_traversability is not None
                else perc.traversability)

        occ   = perc.occupancy_mean
        var   = perc.occupancy_var
        slope = perc.slope_map
        rough = perc.roughness_map

        half_fov    = math.pi / 2.2
        scan_centre = goal_rel_yaw if abs(goal_rel_yaw) < half_fov else 0.0
        alpha_range = np.linspace(scan_centre - half_fov, scan_centre + half_fov, self.n_angles)

        candidates: List[Subgoal] = []

        # Semantic speed cap → affects which distances are sensible to target
        sem_speed_cap = semantic.speed_cap if semantic else 15.0
        # Reduce max look-ahead distance on dangerous terrain
        dyn_max_dist = min(self.max_dist, max(self.min_dist + 2, sem_speed_cap * 1.5))

        for a in alpha_range:
            a_idx = int(np.argmin(np.abs(alphas - a)))
            if not (0 <= a_idx < len(alphas)):
                continue

            depths = np.linspace(self.min_dist, dyn_max_dist, self.n_depth)
            valid_points = []

            for d in depths:
                o_val    = float(occ [mid_beta_idx, a_idx])
                v_val    = float(var [mid_beta_idx, a_idx])
                s_val    = float(slope[mid_beta_idx, a_idx])
                r_val    = float(rough[mid_beta_idx, a_idx])
                t_val    = float(trav [mid_beta_idx, a_idx])
                c_val    = float(curv [mid_beta_idx, a_idx])
                void_val = float(void_map[mid_beta_idx, a_idx])

                if not (o_val < self.OCC_FREE_THRESH
                        and v_val < self.VAR_STABLE_THRESH
                        and s_val < self.SLOPE_SAFE_THRESH
                        and r_val < self.ROUGH_SAFE_THRESH
                        and c_val < self.CURV_SAFE_THRESH
                        and void_val < self.VOID_SAFE_THRESH):
                    continue
                valid_points.append((d, o_val, v_val, s_val, r_val, t_val, c_val, void_val))

            if not valid_points:
                continue

            best_d, best_o, best_v, best_s, best_r, best_t, best_c, best_void = valid_points[-1]
            lx = best_d * math.cos(a)
            ly = best_d * math.sin(a)
            wx = vehicle_pos[0] + (math.cos(vehicle_yaw_rad) * lx - math.sin(vehicle_yaw_rad) * ly)
            wy = vehicle_pos[1] + (math.sin(vehicle_yaw_rad) * lx + math.cos(vehicle_yaw_rad) * ly)

            dist_cand = math.hypot(goal_pos[0] - wx, goal_pos[1] - wy)
            dist_veh  = math.hypot(goal_pos[0] - vehicle_pos[0], goal_pos[1] - vehicle_pos[1])
            goal_progress = dist_veh - dist_cand

            cand_goal_yaw = math.atan2(goal_pos[1] - wy, goal_pos[0] - wx)
            heading_err   = abs(self._norm_angle(cand_goal_yaw - vehicle_yaw_rad))

            terrain_cost = (
                self.W_SLOPE   * best_s
                + self.W_ROUGH * best_r
                + self.W_OCC   * best_o
                + self.W_TERRAIN * (1.0 - best_t)
                + 2.5 * best_c
                + self.W_VOID * best_void
            )

            # Semantic cost: penalise rock/snow
            sem_cost = 0.0
            if semantic is not None:
                sem_cost = self.cfg.w_semantic * semantic.semantic_risk

            # Dense terrain memory cost
            memory_cost = lookahead_cost = 0.0
            if terrain_memory is not None:
                memory_cost = terrain_memory.get_cost(wx, wy)
                cand_heading = math.atan2(wy - vehicle_pos[1], wx - vehicle_pos[0])
                lookahead_cost = terrain_memory.lookahead_cost(
                    wx, wy, cand_heading,
                    steps=self.cfg.terrain_memory_lookahead_steps,
                    step_m=1.0, weight=self.cfg.terrain_memory_lookahead_weight,
                )

            # Topological corridor bias (negative = good corridor, positive = bad zone)
            topo_cost = 0.0
            if topo_memory is not None:
                topo_cost = topo_memory.corridor_bias(wx, wy)

            cost = (
                terrain_cost
                - self.W_GOAL_DIST * max(0.0, goal_progress)
                + self.W_HEADING   * heading_err
                + self.cfg.terrain_memory_cost_weight * memory_cost
                + lookahead_cost
                + sem_cost
                + topo_cost
            )

            sg = Subgoal(
                alpha=a, beta=0.0, distance=best_d,
                local_pos=np.array([lx, ly, 0.0], dtype=np.float32),
                world_pos=np.array([wx, wy, 0.0], dtype=np.float32),
                slope=best_s, roughness=best_r, occupancy=best_o,
                variance=best_v, traversability=best_t,
                terrain_cost=terrain_cost, goal_progress=goal_progress,
                heading_error=heading_err, safe=True,
                width_m=self.cfg.wheel_base, cost=cost,
            )
            candidates.append(sg)

        if not candidates:
            return None, []
        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates

    @staticmethod
    def _norm_angle(a):
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


# ╔════════════════════════════════════════════════════════════════════════════╗
# §8  STUCK MEMORY
# ╚════════════════════════════════════════════════════════════════════════════╝

class StuckMemory:
    def __init__(self, cfg: Config):
        self.decay     = 0.995
        self.radius    = cfg.memory_radius
        self.threshold = 0.05
        self.points: list = []

    def update(self, pos: np.ndarray, stuck: bool, failure_type: str = "stall") -> None:
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
        self.points.append((float(pos[0]), float(pos[1]), 1.0, failure_type))

    def get_tensor(self, device):
        if not self.points:
            return None
        type_map = {"stall": 0.0, "slip": 1.0, "obstacle": 2.0}
        data = [[p[0], p[1], p[2], type_map.get(p[3], 0.0)] for p in self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §9  LEARNED COST NETWORK
# ╚════════════════════════════════════════════════════════════════════════════╝

class CostNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §10  MPPI PLANNER  (semantic + IMU cost terms added)
# ╚════════════════════════════════════════════════════════════════════════════╝

class MPPIPlanner:
    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon  = cfg.mppi_horizon
        self.num_samples = cfg.mppi_num_samples
        self.lambda_  = cfg.mppi_lambda
        self.dt       = cfg.fixed_delta_seconds
        self._u_nom   = torch.zeros((self.horizon, 2), device=self.device)
        self._noise   = torch.zeros((self.num_samples, self.horizon, 2), device=self.device)
        self.cost_net = CostNet().to(self.device)
        self.cost_net.eval()
        # Cached semantic / IMU scalars (updated per tick)
        self._semantic_risk: float = 0.0
        self._speed_cap: float = 15.0
        self._imu_vibration: float = 0.0
        self._imu_slip: float = 0.0

    def set_semantic_imu(
        self,
        semantic: Optional[SemanticOutput],
        imu: Optional[IMUData],
    ) -> None:
        self._semantic_risk = semantic.semantic_risk if semantic else 0.0
        self._speed_cap     = semantic.speed_cap     if semantic else 15.0
        self._imu_vibration = imu.vibration  if imu else 0.0
        self._imu_slip      = imu.slip_index if imu else 0.0

    def plan(self, state, perc, goal_xy, memory_points):
        if perc is None or perc.occupancy_mean is None:
            return ControlState(), 0.0

        device = self.device
        self.goal    = torch.tensor(goal_xy[:2], device=device).float()
        self.mem_pts = memory_points

        self._u_nom = torch.roll(self._u_nom, -1, dims=0)
        self._u_nom[-1] = self._u_nom[-2]
        self._noise[:, :, 0].normal_(0, self.cfg.mppi_noise_throttle)
        self._noise[:, :, 1].normal_(0, self.cfg.mppi_noise_steer)
        u = (self._u_nom.unsqueeze(0) + self._noise).clone()
        u[:, :, 0] = u[:, :, 0].clamp(0.0, self.cfg.max_throttle)
        u[:, :, 1] = u[:, :, 1].clamp(-self.cfg.max_steer, self.cfg.max_steer)

        # Use fused traversability if available
        trav_arr = (perc.fused_traversability
                    if perc.fused_traversability is not None
                    else perc.traversability)
        occ      = torch.tensor(perc.occupancy_mean, device=device).float()
        slope    = torch.tensor(perc.slope_map,      device=device).float()
        rough    = torch.tensor(perc.roughness_map,  device=device).float()
        var      = torch.tensor(perc.occupancy_var,  device=device).float()
        trav     = torch.tensor(trav_arr,             device=device).float()
        curv     = torch.tensor(perc.curvature_map,  device=device).float()
        void_map = torch.tensor(perc.void_risk_map,  device=device).float()

        # Semantic + IMU cost scalars as tensors
        sem_risk_t = torch.tensor(self._semantic_risk, device=device).float()
        imu_vib_t  = torch.tensor(self._imu_vibration, device=device).float()
        imu_slip_t = torch.tensor(self._imu_slip,      device=device).float()
        speed_cap_t = torch.tensor(self._speed_cap,    device=device).float()

        costs = self._rollout(state, u, occ, slope, rough, var, trav, curv, void_map,
                              sem_risk_t, imu_vib_t, imu_slip_t, speed_cap_t)

        beta    = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum()
        self._u_nom = (weights[:, None, None] * u).sum(dim=0).detach()

        u0   = self._u_nom[0]
        ctrl = ControlState(
            throttle=float(u0[0].cpu()),
            steer=float(u0[1].cpu()),
            brake=0.0, reverse=False,
        )
        return ctrl, float(beta.cpu())

    def _rollout(
            self,
            state,
            u,
            occ,
            slope,
            rough,
            var,
            trav,
            curv,
            void_map,
            sem_risk,
            imu_vib,
            imu_slip,
            speed_cap,
        ):
        K, T, _ = u.shape
        device = self.device

        n_a, n_b = occ.shape[1], occ.shape[0]
        a_max = np.pi / 2

            # state = [x, y, yaw, speed]
        x = torch.tensor(state, device=device).float().unsqueeze(0).repeat(K, 1)

        veh_x = x[0, 0].item()
        veh_y = x[0, 1].item()
        veh_yaw = x[0, 2].item()

        total = torch.zeros(K, device=device)

            # ─────────────────────────────
            # Initial goal distance
            # ─────────────────────────────
        prev_gdist = torch.sqrt(
                (self.goal[0] - x[:, 0]) ** 2 +
                (self.goal[1] - x[:, 1]) ** 2
            )

        for t in range(T):
                throttle = u[:, t, 0]
                steer = u[:, t, 1]

                # ─────────────────────────────
                # Map position → perception grid
                # ─────────────────────────────
                dx = x[:, 0] - veh_x
                dy = x[:, 1] - veh_y

                rel_yaw = torch.atan2(
                    torch.sin(torch.atan2(dy, dx) - veh_yaw),
                    torch.cos(torch.atan2(dy, dx) - veh_yaw),
                )

                ai = ((rel_yaw + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
                bi = torch.full_like(ai, n_b // 2)

                # ─────────────────────────────
                # Terrain-aware speed scaling
                # ─────────────────────────────
                slope_v = slope[bi, ai]
                rough_v = rough[bi, ai]

                speed_scale = torch.exp(-3.0 * (slope_v + rough_v)).clamp(0.15, 1.0)

                # ─────────────────────────────
                # Dynamics
                # ─────────────────────────────
                v = x[:, 3] + throttle * speed_scale * self.dt
                v = v.clamp(0.0, speed_cap)

                yaw_rate = (v / self.cfg.wheel_base) * torch.tan(steer.clamp(-0.99, 0.99))
                yaw = x[:, 2] + yaw_rate * self.dt

                xp = x[:, 0] + v * torch.cos(yaw) * self.dt
                yp = x[:, 1] + v * torch.sin(yaw) * self.dt

                x = torch.stack([xp, yp, yaw, v], dim=1)

                # ─────────────────────────────
                # Re-index terrain at new pose
                # ─────────────────────────────
                dx = xp - veh_x
                dy = yp - veh_y

                rel_yaw = torch.atan2(
                    torch.sin(torch.atan2(dy, dx) - veh_yaw),
                    torch.cos(torch.atan2(dy, dx) - veh_yaw),
                )

                ai = ((rel_yaw + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
                bi = torch.full_like(ai, n_b // 2)

                # ─────────────────────────────
                # Terrain values
                # ─────────────────────────────
                occ_v   = occ[bi, ai]
                slope_v = slope[bi, ai]
                rough_v = rough[bi, ai]
                var_v   = var[bi, ai]
                trav_v  = trav[bi, ai]
                curv_v  = curv[bi, ai]
                void_v  = void_map[bi, ai]

                # ─────────────────────────────
                # Goal metrics
                # ─────────────────────────────
                gdx = self.goal[0] - xp
                gdy = self.goal[1] - yp

                curr_gdist = torch.sqrt(gdx**2 + gdy**2)

                goal_heading = torch.atan2(gdy, gdx)
                heading_error = torch.atan2(
                    torch.sin(goal_heading - yaw),
                    torch.cos(goal_heading - yaw),
                ).abs()

                # ─────────────────────────────
                # PROGRESS (main driver)
                # ─────────────────────────────
                progress = prev_gdist - curr_gdist
                prev_gdist = curr_gdist

                C_progress = -25.0 * progress   # negative = reward

                # ─────────────────────────────
                # COST TERMS
                # ─────────────────────────────
                terrain_cost = (
                    5.0   * occ_v
                    + 3.5 * slope_v
                    + 3.0 * rough_v
                    + 8.0 * curv_v
                    + 120.0 * void_v
                    + 10.0 * var_v
                    + 12.0 * (1.0 - trav_v) ** 2
                )

                control_cost = 0.1 * (throttle**2 + steer**2)

                heading_cost = self.cfg.w_heading * heading_error

                # Semantic penalty (global)
                semantic_cost = self.cfg.w_semantic * sem_risk

                # IMU penalty (stability)
                imu_cost = self.cfg.w_imu * (
                    0.6 * imu_vib + 0.4 * imu_slip
                )

                # Speed penalty if exceeding safe terrain speed
                speed_penalty = torch.relu(v - speed_cap) * 5.0

                # ─────────────────────────────
                # TOTAL COST
                # ─────────────────────────────
                cost = (
                    terrain_cost
                    + heading_cost
                    + control_cost
                    + semantic_cost
                    + imu_cost
                    + speed_penalty
                    + C_progress
                )

                total += cost

        return total


# ╔════════════════════════════════════════════════════════════════════════════╗
# §11  REACTIVE OBSTACLE AVOIDANCE
# ╚════════════════════════════════════════════════════════════════════════════╝

class ReactiveObstacleAvoidance:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._obstacle_active = False

    @property
    def obstacle_active(self):
        return self._obstacle_active

    def process(self, raw_pts, ctrl, speed):
        self._obstacle_active = False
        if raw_pts is None or len(raw_pts) < 5:
            return ctrl
        x, y, z  = raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2]
        dist_2d  = np.sqrt(x ** 2 + y ** 2)
        angle    = np.degrees(np.arctan2(y, x))
        fwd_mask = (
            (x > 0.4) & (np.abs(angle) < self.cfg.react_forward_cone)
            & (z > -0.3) & (dist_2d < self.cfg.react_warn_dist * 1.5)
        )
        if not np.any(fwd_mask):
            return ctrl
        fwd_pts  = raw_pts[fwd_mask]
        fwd_dist = np.sqrt(fwd_pts[:, 0] ** 2 + fwd_pts[:, 1] ** 2)
        min_dist    = float(np.min(fwd_dist))
        closest_idx = int(np.argmin(fwd_dist))
        if min_dist > self.cfg.react_warn_dist:
            return ctrl
        self._obstacle_active = True
        out = ControlState(throttle=ctrl.throttle, steer=ctrl.steer,
                           brake=ctrl.brake, reverse=ctrl.reverse)
        if min_dist <= self.cfg.react_emergency_dist:
            out.throttle = 0.3; out.brake = self.cfg.max_brake; out.reverse = True
        elif min_dist <= self.cfg.react_danger_dist:
            ratio        = (min_dist - self.cfg.react_emergency_dist) / (
                self.cfg.react_danger_dist - self.cfg.react_emergency_dist)
            out.throttle = ctrl.throttle * ratio * 0.2
            out.brake    = self.cfg.max_brake * (1.0 - ratio * 0.4)
            obs_y        = float(fwd_pts[closest_idx, 1])
            steer_dir    = -np.sign(obs_y) if abs(obs_y) > 0.1 else -np.sign(ctrl.steer + 1e-6)
            steer_mag    = min(self.cfg.max_steer, 0.4 / (min_dist + 0.1))
            out.steer    = float(np.clip(ctrl.steer + steer_dir * steer_mag,
                                         -self.cfg.max_steer, self.cfg.max_steer))
        else:
            ratio        = (min_dist - self.cfg.react_danger_dist) / (
                self.cfg.react_warn_dist - self.cfg.react_danger_dist)
            out.throttle = ctrl.throttle * (0.3 + ratio * 0.7)
            out.brake    = ctrl.brake + (1.0 - ratio) * 0.25
        if min_dist < 2.5 and not out.reverse:
            out.reverse = True; out.throttle = 0.3
        return out


# ╔════════════════════════════════════════════════════════════════════════════╗
# §12  CONTROLLER
# ╚════════════════════════════════════════════════════════════════════════════╝

class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._state = ControlState()
        self._spd_ema = 0.0; self._str_ema = 0.0

    def apply_safety_filters(self, ctrl, roll_deg, pitch_deg, speed=0.0):
        a = 0.3
        self._spd_ema = a * ctrl.throttle + (1 - a) * self._spd_ema
        self._str_ema = a * ctrl.steer    + (1 - a) * self._str_ema
        th = self._spd_ema; st = self._str_ema; br = ctrl.brake
        if br > 0.1:
            th = 0.0
        speed_ratio = clamp(speed / max(self.cfg.target_speed_flat, 1e-6), 0.0, 1.0)
        max_steer_at_speed = self.cfg.max_steer * max(
            self.cfg.min_steer_at_speed / self.cfg.max_steer,
            1.0 - self.cfg.speed_steer_coupling * speed_ratio,
        )
        st = float(np.clip(st, -max_steer_at_speed, max_steer_at_speed))
        st = float(np.clip(st, self._state.steer    - 0.12, self._state.steer    + 0.12))
        th = float(np.clip(th, self._state.throttle - 0.08, self._state.throttle + 0.08))
        ts, cs = self._stability(roll_deg, pitch_deg)
        ts = max(0.25, ts)
        st  = float(np.clip(st + cs, -max_steer_at_speed, max_steer_at_speed))
        st -= 0.2 * (st - self._state.steer)
        th -= 0.1 * (th - self._state.throttle)
        th  = float(np.clip(th, 0.0, self.cfg.max_throttle))
        st  = float(np.clip(st, -self.cfg.max_steer, self.cfg.max_steer))
        self._state = ControlState(throttle=th, steer=st, brake=br, reverse=ctrl.reverse)
        return self._state

    def _stability(self, roll, pitch):
        ts = 1.0; cs = 0.0; ar = abs(roll)
        warn_r = self.cfg.anti_tip_roll_warn; danger_r = self.cfg.anti_tip_roll_danger
        if ar > warn_r:
            ratio = clamp((ar - warn_r) / max(danger_r - warn_r, 1.0), 0.0, 1.0)
            ts = min(ts, 1.0 - ratio * (1.0 - self.cfg.stability_throttle_scale))
            cs = -math.copysign(
                min(self.cfg.roll_corrective_steer * ratio, self.cfg.roll_corrective_steer), roll
            )
        ap = abs(pitch)
        warn_p = self.cfg.anti_tip_pitch_warn; danger_p = self.cfg.anti_tip_pitch_danger
        if ap > warn_p:
            ratio = clamp((ap - warn_p) / max(danger_p - warn_p, 1.0), 0.0, 1.0)
            ts = min(ts, 1.0 - ratio * (1.0 - self.cfg.stability_throttle_scale * 0.5))
        return float(ts), float(cs)

    def reset(self):
        self._spd_ema = self._str_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §13  STUCK RECOVERY  (topological backtracking added)
# ╚════════════════════════════════════════════════════════════════════════════╝

class StuckRecovery:
    IDLE    = "AUTO"
    REVERSE = "RECOVERY_REVERSE"
    TURN    = "RECOVERY_TURN"
    FORWARD = "RECOVERY_FORWARD_PUSH"
    BACKTRACK = "RECOVERY_BACKTRACK"

    _N_ESCAPE_ANGLES = 9

    def __init__(
        self,
        cfg: Config,
        planner: 'FlatGoalPlanner',
        terrain_memory: Optional[TerrainMemory] = None,
        topo_memory: Optional[TopologicalMemory] = None,
    ):
        self.cfg            = cfg
        self.planner        = planner
        self.terrain_memory = terrain_memory
        self.topo_memory    = topo_memory
        self.mode           = self.IDLE
        self.mode_start     = 0.0
        self.hist: Deque[Tuple[float, float, float, float]] = deque(maxlen=600)
        self.stuck_start: Optional[float] = None
        self.recovery_steer = 0.55
        # Backtrack target
        self._backtrack_target: Optional[Tuple[float, float]] = None

    def force(self, t, st, goal_xy):
        self._start(t, st, goal_xy)

    def update(self, t, st, cmd, goal_xy):
        self.hist.append((t, st.x, st.y, st.speed))
        if self.mode != self.IDLE:
            elapsed = t - self.mode_start
            if self.mode == self.REVERSE   and elapsed > self.cfg.recovery_reverse_s:
                self.mode = self.TURN; self.mode_start = t
            elif self.mode == self.TURN    and elapsed > self.cfg.recovery_turn_s:
                self.mode = self.FORWARD; self.mode_start = t
            elif self.mode == self.FORWARD and elapsed > self.cfg.recovery_forward_s:
                self.mode = self.IDLE; self.mode_start = t; self.stuck_start = None
            elif self.mode == self.BACKTRACK:
                # Exit backtrack when close enough to target or timeout
                if self._backtrack_target is not None:
                    bt_dist = math.hypot(st.x - self._backtrack_target[0],
                                         st.y - self._backtrack_target[1])
                    if bt_dist < 4.0 or elapsed > 8.0:
                        self.mode = self.IDLE; self.mode_start = t
                        self._backtrack_target = None
                elif elapsed > 8.0:
                    self.mode = self.IDLE; self.mode_start = t
            return

        if cmd.reverse or cmd.brake > 0.25:
            self.stuck_start = None; return

        old = None
        for h in self.hist:
            if t - h[0] >= self.cfg.stuck_time_s:
                old = h
        if old is None:
            return

        disp = math.hypot(st.x - old[1], st.y - old[2])
        if disp < self.cfg.stuck_disp_thresh:
            if self.stuck_start is None:
                self.stuck_start = t
            elif t - self.stuck_start > 0.5:
                print(f"[STUCK] ({st.x:.1f},{st.y:.1f}) disp={disp:.3f}m → recovery")
                self._start(t, st, goal_xy)
                self.stuck_start = None
        else:
            self.stuck_start = None

    def _start(self, t, st, goal_xy):
        print(f"\n[RECOVERY] Stuck at ({st.x:.1f},{st.y:.1f})")
        self.planner.add_memory(st.x, st.y)

        # Mark dense + topological memory
        if self.terrain_memory is not None:
            self.terrain_memory.mark_stuck(st.x, st.y, cost=1.0)
        if self.topo_memory is not None:
            self.topo_memory.mark_stuck(st.x, st.y)

        # ── Try topological backtracking first ─────────────────────────────
        if self.topo_memory is not None:
            safe_node = self.topo_memory.nearest_safe_backtrack(st.x, st.y)
            if safe_node is not None:
                print(
                    f"[RECOVERY] Topological backtrack → ({safe_node.x:.1f},{safe_node.y:.1f})"
                    f"  cost={safe_node.cost:.2f}  class={safe_node.semantic_class.name}"
                )
                self._backtrack_target = (safe_node.x, safe_node.y)
                # Steer toward backtrack node during reverse phase
                bt_heading = math.atan2(
                    safe_node.y - st.y, safe_node.x - st.x
                )
                rel = angle_wrap(bt_heading - st.yaw)
                self.recovery_steer = float(clamp(-rel, -1.0, 1.0))
                self.mode       = self.BACKTRACK
                self.mode_start = t
                return

        # ── Fall back: terrain-memory escape angle fan ─────────────────────
        goal_heading = math.atan2(goal_xy[1] - st.y, goal_xy[0] - st.x)
        rel = angle_wrap(goal_heading - st.yaw)

        if self.terrain_memory is not None:
            best_angle  = rel; lowest_cost = float("inf")
            for angle in np.linspace(-1.2, 1.2, self._N_ESCAPE_ANGLES):
                px = st.x + 5.0 * math.cos(st.yaw + angle)
                py = st.y + 5.0 * math.sin(st.yaw + angle)
                c  = self.terrain_memory.get_cost(px, py)
                if c < lowest_cost:
                    lowest_cost = c; best_angle = angle
            if abs(best_angle) < 0.10:
                best_angle = rel if abs(rel) > 0.1 else random.choice([-0.6, 0.6])
            self.recovery_steer = float(clamp(best_angle, -1.0, 1.0))
        else:
            self.recovery_steer = -math.copysign(
                0.62, rel if abs(rel) > 0.1 else random.choice([-1.0, 1.0])
            )

        self.mode = self.REVERSE; self.mode_start = t

    def cmd(self) -> Optional[ControlState]:
        if self.mode == self.IDLE:
            return None
        if self.mode == self.REVERSE:
            return ControlState(throttle=self.cfg.recovery_reverse_throttle,
                                steer=-self.recovery_steer, brake=0.0, reverse=True)
        if self.mode == self.TURN:
            return ControlState(throttle=0.65, steer=self.recovery_steer, brake=0.0)
        if self.mode == self.FORWARD:
            return ControlState(throttle=self.cfg.recovery_forward_throttle,
                                steer=0.50 * self.recovery_steer, brake=0.0)
        if self.mode == self.BACKTRACK:
            # Drive in reverse toward safe node
            steer = float(clamp(self.recovery_steer, -self.cfg.max_steer, self.cfg.max_steer))
            return ControlState(throttle=self.cfg.recovery_reverse_throttle,
                                steer=-steer, brake=0.0, reverse=True)
        return None


# ╔════════════════════════════════════════════════════════════════════════════╗
# §14  FLAT GOAL PLANNER
# ╚════════════════════════════════════════════════════════════════════════════╝

class FlatGoalPlanner:
    def __init__(self, cfg: Config, terrain: TerrainAnalyzer):
        self.cfg = cfg; self.terrain = terrain
        self.best: Optional[Candidate] = None
        self.tick = 0
        self.memory: List[Tuple[float, float, float]] = []
        self.candidate_count     = 81
        self.fov_deg             = 170.0
        self.candidate_distances = (8.0, 14.0, 22.0, 32.0, 45.0)
        self.replan_every_ticks  = 2
        self.w_progress  = 260.0; self.w_goal     = 1.5
        self.w_heading   = 9.0;   self.w_slope    = 18.0
        self.w_rough     = 10.0;  self.w_obstacle = 45.0
        self.w_clearance = 35.0;  self.w_memory   = 80.0

    def add_memory(self, x, y):
        for i, (mx, my, w) in enumerate(self.memory):
            if math.hypot(x - mx, y - my) < self.cfg.memory_radius * 0.6:
                self.memory[i] = (mx, my, min(4.0, w + 1.0)); return
        self.memory.append((x, y, 1.0)); self.memory = self.memory[-30:]

    def memory_penalty(self, wx, wy):
        p = 0.0
        for mx, my, w in self.memory:
            d = math.hypot(wx - mx, wy - my)
            p += w * math.exp(-(d * d) / (2.0 * self.cfg.memory_radius ** 2))
        return p

    def plan(self, st: VehicleState, goal_xy: Tuple[float, float]) -> Candidate:
        self.tick += 1
        if self.best is not None and self.tick % self.replan_every_ticks != 0:
            return self.best
        goal_x, goal_y   = goal_xy
        cur_gdist        = math.hypot(goal_x - st.x, goal_y - st.y)
        goal_heading     = math.atan2(goal_y - st.y, goal_x - st.x)
        rel_goal         = angle_wrap(goal_heading - st.yaw)
        center           = clamp(rel_goal, -math.radians(80), math.radians(80))
        half             = math.radians(self.fov_deg) / 2.0
        angles           = np.linspace(center - half, center + half, self.candidate_count)
        candidates: List[Candidate] = []
        for dist in self.candidate_distances:
            for alpha in angles:
                lx = dist * math.cos(alpha); ly = dist * math.sin(alpha)
                if lx < 1.0: continue
                wx, wy        = local_to_world(lx, ly, st.yaw, st.x, st.y)
                new_gdist     = math.hypot(goal_x - wx, goal_y - wy)
                progress      = cur_gdist - new_gdist
                heading_error = abs(angle_wrap(math.atan2(wy - st.y, wx - st.x) - st.yaw))
                slope, rough, obs, flat, clearance = self.terrain.local_risk(lx, ly, radius=5.0)
                mem = self.memory_penalty(wx, wy)
                cost = (self.w_goal * new_gdist - self.w_progress * progress
                        + self.w_heading * heading_error + self.w_slope * slope
                        + self.w_rough * rough + self.w_obstacle * obs
                        + self.w_clearance * clearance + self.w_memory * mem)
                reward = (120.0 * progress + 45.0 * flat - 25.0 * obs
                          - 12.0 * slope - 8.0 * rough - 35.0 * clearance - 40.0 * mem)
                candidates.append(Candidate(
                    wx, wy, lx, ly, alpha, dist, progress, new_gdist, heading_error,
                    slope, rough, obs, flat, clearance, mem, cost, reward,
                ))
        if not candidates:
            dist = min(25.0, cur_gdist)
            lx = dist * math.cos(rel_goal); ly = dist * math.sin(rel_goal)
            wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
            slope, rough, obs, flat, clearance = self.terrain.local_risk(lx, ly, radius=5.0)
            self.best = Candidate(wx, wy, lx, ly, rel_goal, dist, 1.0,
                                  cur_gdist, abs(rel_goal), slope, rough, obs,
                                  flat, clearance, 0.0, cur_gdist, 0.0)
            return self.best
        candidates.sort(key=lambda c: c.cost)
        best = candidates[0]
        if max(c.progress for c in candidates) < 0.2:
            candidates.sort(key=lambda c: (abs(c.heading_error), c.goal_dist + 25.0 * c.obstacle_risk))
            best = candidates[0]
        self.best = best; return best


# ╔════════════════════════════════════════════════════════════════════════════╗
# §15  MANUAL CONTROLLER
# ╚════════════════════════════════════════════════════════════════════════════╝

class ManualController:
    THROTTLE_STEP = 0.04; STEER_STEP = 0.06; STEER_DECAY = 0.80

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg; self._throttle = 0.0; self._steer = 0.0; self._reverse = False

    def tick(self, keys, roll_deg=0.0, pitch_deg=0.0):
        brake = 0.0
        if keys[pygame.K_w]:
            self._reverse = False
            self._throttle = min(self._throttle + self.THROTTLE_STEP, self.cfg.max_throttle)
        elif keys[pygame.K_s]:
            self._reverse = True
            self._throttle = min(self._throttle + self.THROTTLE_STEP, self.cfg.max_throttle)
        else:
            self._throttle = max(self._throttle - self.THROTTLE_STEP * 2, 0.0)
        if keys[pygame.K_a]:
            self._steer = min(self._steer + self.STEER_STEP, self.cfg.max_steer)
        elif keys[pygame.K_d]:
            self._steer = max(self._steer - self.STEER_STEP, -self.cfg.max_steer)
        else:
            self._steer *= self.STEER_DECAY
        if keys[pygame.K_SPACE]:
            self._throttle = 0.0; self._steer *= 0.5; self._reverse = False; brake = self.cfg.max_brake
        if abs(roll_deg) > self.cfg.max_safe_roll_deg:
            corr = -math.copysign(
                min(0.15 * (abs(roll_deg) - self.cfg.max_safe_roll_deg) / 10.0, 0.15), roll_deg
            )
            self._steer = float(np.clip(self._steer + corr, -self.cfg.max_steer, self.cfg.max_steer))
        return ControlState(round(self._throttle, 4), round(self._steer, 4), brake, self._reverse)

    def reset(self):
        self._throttle = 0.0; self._steer = 0.0; self._reverse = False


# ╔════════════════════════════════════════════════════════════════════════════╗
# §16  SENSOR MANAGER  (LiDAR + RGB + Semantic + IMU)
# ╚════════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config):
        self.world = world; self.vehicle = vehicle; self.cfg = cfg
        self.actors: List = []
        self.lidar_q:    Queue = Queue(maxsize=2)
        self.front_q:    Queue = Queue(maxsize=2)
        self.rear_q:     Queue = Queue(maxsize=2)
        self.semantic_q: Queue = Queue(maxsize=2)
        self.imu_q:      Queue = Queue(maxsize=4)
        self._has_imu      = False
        self._has_semantic = False
        self._setup()

    @staticmethod
    def _put_latest(q, data):
        while q.full():
            try: q.get_nowait()
            except Empty: break
        q.put(data)

    @staticmethod
    def _img_to_array(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        return arr[:, :, ::-1].copy()

    @staticmethod
    def _get_latest(q):
        item = None
        while True:
            try: item = q.get_nowait()
            except Empty: break
        return item

    def _setup(self):
        bp = self.world.get_blueprint_library()

        # ── LiDAR ─────────────────────────────────────────────────────────
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",              str(self.cfg.lidar_range))
        lidar_bp.set_attribute("channels",           str(self.cfg.lidar_channels))
        lidar_bp.set_attribute("points_per_second",  str(self.cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", str(self.cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("upper_fov",          str(self.cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov",          str(self.cfg.lidar_lower_fov))
        lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.7, z=2.25)),
            attach_to=self.vehicle,
        )
        lidar.listen(lambda d: self._put_latest(self.lidar_q, d))
        self.actors.append(lidar)

        # ── RGB cameras ───────────────────────────────────────────────────
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov",          str(self.cfg.camera_fov))
        front = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.7, z=1.9)),
            attach_to=self.vehicle,
        )
        front.listen(lambda img: self._put_latest(self.front_q, self._img_to_array(img)))
        self.actors.append(front)
        rear = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=-1.7, z=1.9), carla.Rotation(yaw=180)),
            attach_to=self.vehicle,
        )
        rear.listen(lambda img: self._put_latest(self.rear_q, self._img_to_array(img)))
        self.actors.append(rear)

        # ── Semantic segmentation camera ──────────────────────────────────
        try:
            seg_bp = bp.find("sensor.camera.semantic_segmentation")
            seg_bp.set_attribute("image_size_x", str(self.cfg.semantic_camera_width))
            seg_bp.set_attribute("image_size_y", str(self.cfg.semantic_camera_height))
            seg_bp.set_attribute("fov",          str(self.cfg.camera_fov))
            seg = self.world.spawn_actor(
                seg_bp,
                carla.Transform(carla.Location(x=1.7, z=1.9)),
                attach_to=self.vehicle,
            )
            def _seg_callback(image):
                raw  = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr  = raw.reshape((image.height, image.width, 4))
                # Keep as BGRA so TerrainClassifier can read R=index 2
                self._put_latest(self.semantic_q, arr.copy())
            seg.listen(_seg_callback)
            self.actors.append(seg)
            self._has_semantic = True
            print("[SENSOR] Semantic segmentation camera online")
        except Exception as e:
            print(f"[SENSOR] Semantic camera unavailable: {e} – using color fallback")

        # ── IMU ───────────────────────────────────────────────────────────
        try:
            imu_bp = bp.find("sensor.other.imu")
            imu_bp.set_attribute("noise_accel_stddev_x", "0.01")
            imu_bp.set_attribute("noise_accel_stddev_y", "0.01")
            imu_bp.set_attribute("noise_accel_stddev_z", "0.01")
            imu_bp.set_attribute("noise_gyro_stddev_x",  "0.001")
            imu_bp.set_attribute("noise_gyro_stddev_y",  "0.001")
            imu_bp.set_attribute("noise_gyro_stddev_z",  "0.001")
            imu = self.world.spawn_actor(
                imu_bp,
                carla.Transform(carla.Location(x=0.0, z=0.5)),
                attach_to=self.vehicle,
            )
            imu.listen(lambda d: self._put_latest(self.imu_q, d))
            self.actors.append(imu)
            self._has_imu = True
            print("[SENSOR] IMU online")
        except Exception as e:
            print(f"[SENSOR] IMU unavailable: {e}")

    def get_lidar_points(self):
        data = self._get_latest(self.lidar_q)
        if data is None: return None
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        d   = np.linalg.norm(pts[:, :2], axis=1)
        return pts[(d > 1.0) & (d < self.cfg.lidar_range)]

    def get_front_image(self):
        return self._get_latest(self.front_q)

    def get_rear_image(self):
        return self._get_latest(self.rear_q)

    def get_semantic_image(self) -> Optional[np.ndarray]:
        return self._get_latest(self.semantic_q)

    def get_imu_data(self):
        """Return the latest IMU measurement (raw CARLA object or None)."""
        return self._get_latest(self.imu_q)

    @property
    def has_imu(self):
        return self._has_imu

    @property
    def has_semantic(self):
        return self._has_semantic

    def destroy(self):
        for a in self.actors:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# §17  RESULT LOGGER
# ╚════════════════════════════════════════════════════════════════════════════╝

class ResultLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.rows: List[dict] = []
        self.terrain_rows: List[dict] = []

    def log(self, t, st, cmd, cand, goal_dist, mode, cost, reward,
            semantic_class: str = "UNKNOWN", imu_vibration: float = 0.0,
            safe_topo_nodes: int = 0) -> None:
        angular_velocity = (st.speed / max(self.cfg.wheel_base, 1e-6)) * math.tan(cmd.steer)
        subgoal_x = getattr(cand, 'wx', None)
        if subgoal_x is None:
            wp = getattr(cand, 'world_pos', None)
            subgoal_x = float(wp[0]) if wp is not None and len(wp) >= 2 else st.x
        subgoal_y = getattr(cand, 'wy', None)
        if subgoal_y is None:
            wp = getattr(cand, 'world_pos', None)
            subgoal_y = float(wp[1]) if wp is not None and len(wp) >= 2 else st.y
        self.rows.append({
            "time": t, "x": st.x, "y": st.y, "z": st.z,
            "yaw": st.yaw, "pitch_deg": st.pitch_deg, "roll_deg": st.roll_deg,
            "speed": st.speed,
            "linear_velocity": st.speed,
            "angular_velocity": angular_velocity,
            "throttle": cmd.throttle, "steer": cmd.steer,
            "brake": cmd.brake, "reverse": int(cmd.reverse),
            "goal_distance": goal_dist, "mode": mode,
            "cost": cost, "reward": reward,
            "subgoal_x": subgoal_x, "subgoal_y": subgoal_y,
            "flatness":       getattr(cand, 'flatness',      getattr(cand, 'traversability', 0.0)),
            "slope":          getattr(cand, 'slope',         0.0),
            "roughness":      getattr(cand, 'roughness',     0.0),
            "progress":       getattr(cand, 'progress',      getattr(cand, 'goal_progress', 0.0)),
            "obstacle_risk":  getattr(cand, 'obstacle_risk', getattr(cand, 'occupancy', 0.0)),
            "clearance_risk": getattr(cand, 'clearance_risk', 0.0),
            "semantic_class": semantic_class,
            "imu_vibration":  imu_vibration,
            "safe_topo_nodes": safe_topo_nodes,
        })

    def log_terrain_points(self, t, st, pts) -> None:
        if pts is None or len(pts) < 10: return
        step = max(1, len(pts) // 800)
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        for p in pts[::step]:
            lx, ly, lz = float(p[0]), float(p[1]), float(p[2])
            self.terrain_rows.append({
                "time": t,
                "x": st.x + c * lx - s * ly,
                "y": st.y + s * lx + c * ly,
                "z": st.z + lz,
            })

    def save_csv(self):
        if not self.rows: return
        with open(os.path.join(self.cfg.out_dir, "navigation_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader(); w.writerows(self.rows)
        for name, keys in [
            ("controls.csv",     ["time","throttle","steer","brake","reverse","linear_velocity","angular_velocity","speed"]),
            ("trajectory.csv",   ["time","x","y","z","goal_distance"]),
            ("cost_reward.csv",  ["time","cost","reward","goal_distance","flatness","slope","roughness","progress","clearance_risk"]),
            ("semantics_imu.csv",["time","semantic_class","imu_vibration","safe_topo_nodes","speed"]),
        ]:
            with open(os.path.join(self.cfg.out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow({k: r.get(k, "") for k in keys})
        if self.terrain_rows:
            with open(os.path.join(self.cfg.out_dir, "terrain_samples.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.terrain_rows[0].keys()))
                w.writeheader(); w.writerows(self.terrain_rows)

    def _line(self, x, y, title, ylabel, filename):
        plt.figure(figsize=(8, 4.5)); plt.plot(x, y, linewidth=2)
        plt.grid(True, alpha=0.3); plt.xlabel("Time [s]"); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, filename), format="svg"); plt.close()

    def save_plots(self):
        if not self.rows: return
        t      = np.array([r["time"]            for r in self.rows])
        x      = np.array([r["x"]               for r in self.rows])
        y      = np.array([r["y"]               for r in self.rows])
        speed  = np.array([r["speed"]           for r in self.rows])
        cost   = np.array([r["cost"]            for r in self.rows])
        reward = np.array([r["reward"]          for r in self.rows])
        gdist  = np.array([r["goal_distance"]   for r in self.rows])
        vibs   = np.array([r["imu_vibration"]   for r in self.rows])
        nodes  = np.array([r["safe_topo_nodes"] for r in self.rows])

        self._line(t, cost,   "MPPI Cost",              "Cost",   "nmpc_cost.svg")
        self._line(t, reward, "Reward",                 "Reward", "reward.svg")
        self._line(t, speed,  "Vehicle Speed",          "m/s",    "vehicle_speed.svg")
        self._line(t, gdist,  "Distance to Goal",       "m",      "goal_distance.svg")
        self._line(t, vibs,   "IMU Vibration Index",    "0–1",    "imu_vibration.svg")
        self._line(t, nodes,  "Safe Topological Nodes", "count",  "topo_safe_nodes.svg")

        plt.figure(figsize=(7, 7))
        plt.plot(x, y, linewidth=2, label="trajectory")
        plt.scatter([self.cfg.spawn_x], [self.cfg.spawn_y], marker="o", s=80, label="start")
        plt.scatter([self.cfg.goal_x],  [self.cfg.goal_y],  marker="x", s=120, c="red", label="goal")
        plt.axis("equal"); plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("World Trajectory")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "trajectory.svg"), format="svg"); plt.close()

        plt.figure(figsize=(7, 7))
        sc = plt.scatter(x, y, c=speed, s=18, cmap="RdYlGn")
        plt.colorbar(sc, label="Speed [m/s]")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=120, c="red")
        plt.axis("equal"); plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("Speed Heat Map")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "speed_heatmap.svg"), format="svg"); plt.close()

        if len(self.terrain_rows) > 50:
            tx = np.array([r["x"] for r in self.terrain_rows])
            ty = np.array([r["y"] for r in self.terrain_rows])
            tz = np.array([r["z"] for r in self.terrain_rows])
            plt.figure(figsize=(7, 7))
            sc = plt.scatter(tx, ty, c=tz, s=5)
            plt.colorbar(sc, label="Terrain height [m]")
            plt.plot(x, y, linewidth=1.5)
            plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=120, c="red")
            plt.axis("equal"); plt.grid(True, alpha=0.3)
            plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("Terrain Height Map")
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.out_dir, "flatness_heatmap.svg"), format="svg")
            plt.close()

    def finalize(self):
        print(f"\n[SAVE] Saving in {os.path.abspath(self.cfg.out_dir)}")
        self.save_csv(); self.save_plots(); print("[SAVE] Done")


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18  VISUALIZATION MODULE
# ╚════════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    C = {
        'bg': (17,17,17), 'panel': (26,26,46), 'header': (35,35,65),
        'grid': (45,45,70), 'border': (60,60,95), 'traj': (0,206,201),
        'veh': (253,203,110), 'goal': (255,60,60), 'subgoal': (50,255,120),
        'memory': (255,110,50), 'cost': (255,107,107),
        'speed': (85,239,196), 'steer': (162,155,254), 'white': (230,230,230),
        'dim': (110,110,110), 'danger': (255,50,50), 'ok': (80,220,80),
        'warn': (255,200,0), 'topo_safe': (50,220,50), 'topo_bad': (220,50,50),
    }
    _SEM_COLORS: Dict[TerrainClass, Tuple[int,int,int]] = {
        TerrainClass.UNKNOWN:    (100, 100, 100),
        TerrainClass.DIRT:       (160, 110,  50),
        TerrainClass.GRASS:      ( 50, 200,  60),
        TerrainClass.GRAVEL:     (160, 160, 130),
        TerrainClass.ROCK:       (120, 100,  80),
        TerrainClass.SNOW:       (220, 240, 255),
        TerrainClass.VEGETATION: ( 30, 140,  30),
        TerrainClass.ROAD:       (180, 180, 180),
    }

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._screen = None; self._font_s = None
        mx = cfg.traj_history
        self._traj_x: deque = deque(maxlen=mx); self._traj_y: deque = deque(maxlen=mx)
        self._costs:  deque = deque(maxlen=400); self._speeds: deque = deque(maxlen=400)
        self._rewards: deque = deque(maxlen=400)
        self._front_img = None; self._rear_img = None; self._lidar_pts = None
        self._goal = GOAL
        self._subgoal = None; self._subgoal_candidates: List = []; self._mem_pts: list = []
        self._mode = "AUTO"; self._speed = 0.0; self._steer = 0.0
        self._cost = 0.0; self._reward = 0.0; self._dist_goal = 0.0
        self._obs_warn = False; self._pitch = 0.0; self._roll = 0.0
        self._flatness = 0.0; self._progress = 0.0
        self._semantic_class = TerrainClass.UNKNOWN
        self._imu_vibration = 0.0; self._imu_slip = 0.0; self._stability = 1.0
        self._safety_active = False; self._safety_reason = ""
        self._topo_nodes: List[TopologicalNode] = []

    def init(self, screen):
        self._screen = screen; pygame.font.init()
        self._font_s = pygame.font.SysFont("monospace", 11)

    def push(self, *, pos, speed, steer, perc, mode, cost, reward,
             subgoal, all_candidates, mem_pts, dist_goal, obs_warn,
             flatness=0.0, progress=0.0, pitch=0.0, roll=0.0,
             semantic=None, imu=None, safety_active=False, safety_reason="",
             topo_nodes=None):
        self._traj_x.append(float(pos[0])); self._traj_y.append(float(pos[1]))
        self._costs.append(cost); self._speeds.append(speed); self._rewards.append(reward)
        self._mode = mode; self._speed = speed; self._steer = steer
        self._cost = cost; self._reward = reward
        self._subgoal = subgoal.copy() if subgoal is not None else None
        self._subgoal_candidates = [c.world_pos for c in all_candidates] if all_candidates else []
        self._mem_pts = list(mem_pts); self._dist_goal = dist_goal; self._obs_warn = obs_warn
        self._flatness = flatness; self._progress = progress; self._pitch = pitch; self._roll = roll
        if semantic:
            self._semantic_class = semantic.dominant_class
        if imu:
            self._imu_vibration = imu.vibration
            self._imu_slip      = imu.slip_index
            self._stability     = imu.stability_score
        self._safety_active = safety_active; self._safety_reason = safety_reason
        if topo_nodes is not None: self._topo_nodes = topo_nodes
        if perc and perc.raw_points is not None and len(perc.raw_points):
            self._lidar_pts = perc.raw_points.copy()

    def set_front(self, img): self._front_img = img
    def set_rear(self, img):  self._rear_img  = img

    def render(self):
        if self._screen is None: return
        W, H  = self._screen.get_size()
        CAM_H = 240; col_w = W // 3; bot_y = CAM_H; bot_h = H - bot_y; map_w = W // 2
        self._screen.fill(self.C['bg'])
        self._draw_camera(self._front_img, "Front camera",   0,        0, col_w,     CAM_H)
        self._draw_camera(self._rear_img,  "Rear camera",    col_w,    0, col_w,     CAM_H)
        self._draw_lidar_3d("LiDAR view",                    col_w*2,  0, W-col_w*2, CAM_H)
        self._draw_trajectory(0,    bot_y, map_w,     bot_h)
        self._draw_cost_plot(map_w, bot_y, W - map_w, bot_h // 2)
        self._draw_speed_plot(map_w, bot_y + bot_h // 2, W - map_w, bot_h - bot_h // 2)
        self._draw_hud(W, H)
        pygame.display.flip()

    def _draw_camera(self, img, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x+4, y+3))
        if img is not None:
            try:
                img_r = self._resize(img, w, h - 18)
                surf  = pygame.surfarray.make_surface(img_r.swapaxes(0, 1))
                self._screen.blit(surf, (x, y + 18))
            except Exception: pass
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_lidar_3d(self, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x+4, y+3))
        cy0 = y + 18; ch = h - 18; lr = self.cfg.lidar_range
        cx = x + w // 2; cy = cy0 + ch // 2; scale = (min(w, ch) * 0.44) / lr
        for rr in [lr*0.33, lr*0.67, lr]:
            pygame.draw.circle(self._screen, C['grid'], (cx, cy), int(rr*scale), 1)
        pygame.draw.line(self._screen, C['grid'], (cx, cy), (cx, cy-int(lr*scale)), 1)
        if self._lidar_pts is not None and len(self._lidar_pts):
            pts = self._lidar_pts
            z_min = pts[:,2].min(); z_max = pts[:,2].max(); z_range = max(z_max-z_min,1e-6)
            zn = (pts[:,2]-z_min)/z_range
            for i in range(0, len(pts), 3):
                px = int(cx - pts[i,1]*scale); py = int(cy - pts[i,0]*scale)
                if x <= px < x+w and cy0 <= py < cy0+ch:
                    t2 = float(zn[i])
                    if   t2<0.2: r2,g2,b2 = 255,int(255*(t2/0.2)),0
                    elif t2<0.4: r2,g2,b2 = int(255*((0.4-t2)/0.2)),255,0
                    elif t2<0.6: r2,g2,b2 = 0,255,int(255*((t2-0.4)/0.2))
                    elif t2<0.8: r2,g2,b2 = 0,int(255*((0.8-t2)/0.2)),255
                    else:        r2,g2,b2 = int(255*((t2-0.8)/0.2)),0,255
                    pygame.draw.circle(self._screen, (r2,g2,b2), (px,py), 1)
        pygame.draw.circle(self._screen, C['veh'], (cx,cy), 4)
        pygame.draw.line(self._screen, C['veh'], (cx,cy), (cx,cy-14), 2)
        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_trajectory(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        sem_col = self._SEM_COLORS.get(self._semantic_class, (100,100,100))
        header_txt = f"Route Map  terrain={self._semantic_class.name}"
        self._screen.blit(self._font_s.render(header_txt, True, sem_col), (x+4,y+3))

        tx = list(self._traj_x); ty = list(self._traj_y)
        if not tx:
            pygame.draw.rect(self._screen, C['border'], (x,y,w,h),1); return

        ax = tx+[self._goal[0]]; ay = ty+[self._goal[1]]
        for px2,py2,pw2,_ in self._mem_pts:
            ax.append(px2); ay.append(py2)
        if self._subgoal is not None:
            ax.append(float(self._subgoal[0])); ay.append(float(self._subgoal[1]))
        for n in self._topo_nodes:
            ax.append(n.x); ay.append(n.y)

        span = max(max(ax)-min(ax), max(ay)-min(ay), 20.0); pad = span*0.12
        mn_x=min(ax)-pad; mx_x=max(ax)+pad; mn_y=min(ay)-pad; mx_y=max(ay)+pad
        px0=x+6; py0=y+22; pw=w-12; ph=h-26

        def w2s(wx2,wy2):
            sx = px0 + int((wx2-mn_x)/(mx_x-mn_x)*pw)
            sy = py0 + int((1.0-(wy2-mn_y)/(mx_y-mn_y))*ph)
            return sx,sy

        for gx2 in np.linspace(mn_x,mx_x,5):
            sx,_ = w2s(gx2,mn_y); pygame.draw.line(self._screen,C['grid'],(sx,py0),(sx,py0+ph),1)
        for gy2 in np.linspace(mn_y,mx_y,5):
            _,sy = w2s(mn_x,gy2); pygame.draw.line(self._screen,C['grid'],(px0,sy),(px0+pw,sy),1)

        # Draw topological nodes (small dots)
        for n in self._topo_nodes:
            sx,sy = w2s(n.x,n.y)
            col = C['topo_safe'] if n.is_safe else C['topo_bad']
            pygame.draw.circle(self._screen, col, (sx,sy), 2)

        # Goal marker
        sx,sy = w2s(self._goal[0], self._goal[1])
        for d in [(-7,-7,7,7),(-7,7,7,-7)]:
            pygame.draw.line(self._screen, C['goal'], (sx+d[0],sy+d[1]),(sx+d[2],sy+d[3]),2)
        self._screen.blit(self._font_s.render("GOAL", True, C['goal']), (sx+8,sy-6))

        for px2,py2,pw2,_ in self._mem_pts:
            sx,sy = w2s(px2,py2); r = max(4,int(pw2*14))
            blob = pygame.Surface((r*2,r*2),pygame.SRCALPHA)
            pygame.draw.circle(blob,(*C['memory'],90),(r,r),r)
            self._screen.blit(blob,(sx-r,sy-r)); pygame.draw.circle(self._screen,C['memory'],(sx,sy),4)

        for i,c in enumerate(self._subgoal_candidates):
            if c is not None:
                sx,sy = w2s(c[0],c[1])
                col = C['subgoal'] if i==0 else tuple(v//2 for v in C['subgoal'])
                pygame.draw.circle(self._screen,col,(sx,sy),4 if i==0 else 2)

        if len(tx)>1:
            pts_s = [w2s(tx[i],ty[i]) for i in range(len(tx))]
            pygame.draw.lines(self._screen,C['traj'],False,pts_s,2)
        if tx:
            vx2,vy2 = w2s(tx[-1],ty[-1])
            pygame.draw.circle(self._screen,C['veh'],(vx2,vy2),6)
            pygame.draw.circle(self._screen,C['white'],(vx2,vy2),6,1)
        if self._subgoal is not None:
            ssx,ssy = w2s(float(self._subgoal[0]),float(self._subgoal[1]))
            pygame.draw.circle(self._screen,C['subgoal'],(ssx,ssy),6)
            pygame.draw.circle(self._screen,C['white'],(ssx,ssy),6,1)
        pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1)

    def _draw_cost_plot(self, x, y, w, h):
        self._draw_graph(list(self._costs), "MPPI Cost", x, y, w, h, self.C['cost'])

    def _draw_speed_plot(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen,C['panel'],(x,y,w,h))
        pygame.draw.rect(self._screen,C['header'],(x,y,w,18))
        self._screen.blit(self._font_s.render(
            f"Spd:{self._speed:.1f}m/s  Vib:{self._imu_vibration:.2f}  Slip:{self._imu_slip:.2f}",
            True, C['speed']), (x+4,y+3))
        if len(self._speeds) > 2:
            data = list(self._speeds); mn,mx = min(data),max(data); rng = max(mx-mn,1e-3)
            px0,py0 = x+5,y+22; pw,ph = w-10,h-26; n = len(data)
            pts_s = [(px0+int(i/(n-1)*pw),py0+int((1-(v-mn)/rng)*ph)) for i,v in enumerate(data)]
            if len(pts_s)>1: pygame.draw.lines(self._screen,C['speed'],False,pts_s,2)
        pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1)

    def _draw_graph(self, data, title, x, y, w, h, color):
        C = self.C
        pygame.draw.rect(self._screen,C['panel'],(x,y,w,h))
        pygame.draw.rect(self._screen,C['header'],(x,y,w,18))
        lbl = f"{title}  {data[-1]:.2f}" if data else title
        self._screen.blit(self._font_s.render(lbl,True,color),(x+4,y+3))
        if len(data)<2: pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1); return
        px0,py0=x+5,y+22; pw,ph=w-10,h-26
        mn,mx=min(data),max(data); rng=max(mx-mn,1e-3); n=len(data)
        pts_s=[(px0+int(i/(n-1)*pw),py0+int((1-(v-mn)/rng)*ph)) for i,v in enumerate(data)]
        if len(pts_s)>1: pygame.draw.lines(self._screen,color,False,pts_s,2)
        self._screen.blit(self._font_s.render(f"{mx:.1f}",True,C['dim']),(px0+2,py0+1))
        self._screen.blit(self._font_s.render(f"{mn:.1f}",True,C['dim']),(px0+2,py0+ph-12))
        pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1)

    def _draw_hud(self, W, H):
        C = self.C
        mode_c   = C['ok'] if self._mode == "AUTO" else C['warn']
        sem_col  = self._SEM_COLORS.get(self._semantic_class, (100,100,100))
        stab_col = C['ok'] if self._stability > 0.7 else (C['warn'] if self._stability > 0.4 else C['danger'])
        lines = [
            (f"MODE : {self._mode}",                      mode_c),
            (f"Speed: {self._speed:5.2f} m/s",            C['speed']),
            (f"Steer: {self._steer:+6.3f}",               C['steer']),
            (f"Cost : {self._cost:8.1f}",                 C['cost']),
            (f"Goal : {self._dist_goal:6.1f} m",          C['white']),
            (f"Terrain: {self._semantic_class.name}",     sem_col),
            (f"Vib  : {self._imu_vibration:5.3f}",        stab_col),
            (f"Slip : {self._imu_slip:5.3f}",             stab_col),
            (f"Stab : {self._stability:5.3f}",            stab_col),
            (f"Flat : {self._flatness:6.2f}",             C['dim']),
            (f"Pitch: {self._pitch:+5.1f}°",              C['dim']),
            (f"Roll : {self._roll:+5.1f}°",               C['dim']),
            (f"TopoSafe: {sum(1 for n in self._topo_nodes if n.is_safe)}", C['topo_safe']),
        ]
        if self._obs_warn:
            lines.append(("!!! OBSTACLE !!!", C['danger']))
        if self._safety_active:
            lines.append((f"SAFETY: {self._safety_reason}", C['danger']))
        BW = 230; BH = len(lines)*17+8; bx = W-BW-4; by = 4
        bg = pygame.Surface((BW,BH),pygame.SRCALPHA); bg.fill((0,0,0,170))
        self._screen.blit(bg,(bx,by))
        for i,(txt,col) in enumerate(lines):
            self._screen.blit(self._font_s.render(txt,True,col),(bx+5,by+4+i*17))

    @staticmethod
    def _resize(img, w, h):
        if h<=0 or w<=0: return img
        sh,sw = img.shape[:2]
        ri = (np.arange(h)*sh/h).astype(int).clip(0,sh-1)
        ci = (np.arange(w)*sw/w).astype(int).clip(0,sw-1)
        return img[np.ix_(ri,ci)]


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18.5  MISSION MANAGER  (no waypoints – single goal)
# ╚════════════════════════════════════════════════════════════════════════════╝

class MissionManager:
    def __init__(self, goal: Tuple[float, float], cfg: Config):
        self.goal = goal; self.cfg = cfg
        self._complete = False
        self.last_dist = float('inf')
        self.stall_counter = 0

    def update(self, st: VehicleState, t: float) -> bool:
        if self._complete: return False
        dist = math.hypot(st.x - self.goal[0], st.y - self.goal[1])
        if dist < self.cfg.goal_tolerance:
            print(f"\n[MISSION] ✓ Goal reached!  t={t:.1f}s  dist={dist:.2f}m")
            self._complete = True
            return True
        if dist < self.last_dist - 0.3:
            self.stall_counter = 0
        else:
            self.stall_counter += 1
        self.last_dist = dist
        return False

    @property
    def current_goal(self) -> Tuple[float, float]:
        return self.goal

    @property
    def goal_array(self) -> np.ndarray:
        return np.array(self.goal, dtype=np.float32)

    @property
    def is_final(self) -> bool:
        return True

    @property
    def complete(self) -> bool:
        return self._complete

    def goal_dist(self, st: VehicleState) -> float:
        return math.hypot(st.x - self.goal[0], st.y - self.goal[1])


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18.6  BEHAVIOR PLANNER
# ╚════════════════════════════════════════════════════════════════════════════╝

class BehaviorPlanner:
    AUTO = "AUTO"; AVOID_OBSTACLE = "AVOID_OBSTACLE"
    RECOVERY = "RECOVERY"; GOAL_REACHED = "GOAL_REACHED"; MANUAL = "MANUAL"

    def __init__(self):
        self.mode = self.AUTO; self._prev_mode = self.AUTO; self._manual_flag = False

    def set_manual(self, enable: bool):
        self._manual_flag = enable; self.mode = self.MANUAL if enable else self.AUTO; self._prev_mode = self.mode

    def update(self, *, mission_complete, obstacle_active, in_recovery):
        if self._manual_flag: return
        self._prev_mode = self.mode
        if mission_complete:         self.mode = self.GOAL_REACHED
        elif in_recovery:            self.mode = self.RECOVERY
        elif obstacle_active:        self.mode = self.AVOID_OBSTACLE
        else:                        self.mode = self.AUTO

    @property
    def changed(self): return self.mode != self._prev_mode

    @property
    def is_autonomous(self): return self.mode in (self.AUTO, self.AVOID_OBSTACLE)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18.7  SAFETY MONITOR  (semantic + IMU aware)
# ╚════════════════════════════════════════════════════════════════════════════╝

class SafetyMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._bad_imu_start: Optional[float] = None

    def check(
        self,
        st: VehicleState,
        imu: Optional[IMUData] = None,
        semantic: Optional[SemanticOutput] = None,
        void_risk: float = 0.0,
        forward_clearance: float = 99.0,
        wall_t: float = 0.0,
    ) -> Tuple[bool, str]:
        # 1. Hard structural overrides (roll/pitch danger)
        if abs(st.roll_deg)  > self.cfg.anti_tip_roll_danger:
            return True, f"ROLL {st.roll_deg:.1f}° DANGER"
        if abs(st.pitch_deg) > self.cfg.anti_tip_pitch_danger:
            return True, f"PITCH {st.pitch_deg:.1f}° DANGER"

        # 2. Cliff / void ahead
        if void_risk > 0.82 and forward_clearance < 6.0:
            return True, f"CLIFF/VOID ahead: void={void_risk:.2f} fwd={forward_clearance:.1f}m"

        # 3. IMU sustained instability
        if imu is not None:
            if imu.vibration > self.cfg.imu_vibration_danger:
                if self._bad_imu_start is None:
                    self._bad_imu_start = wall_t
                elif wall_t - self._bad_imu_start > self.cfg.imu_bad_duration_s:
                    return True, f"IMU vib={imu.vibration:.2f} sustained"
            elif imu.slip_index > self.cfg.imu_slip_danger:
                return True, f"IMU slip={imu.slip_index:.2f}"
            else:
                self._bad_imu_start = None
        else:
            self._bad_imu_start = None

        # 4. Semantic + geometric veto
        if semantic is not None:
            if (semantic.forward_class == TerrainClass.ROCK and
                    abs(st.pitch_deg) > self.cfg.sem_rock_slope_veto * 60.0):
                return True, f"ROCK+SLOPE pitch={st.pitch_deg:.1f}°"
            if (semantic.forward_class == TerrainClass.SNOW and
                    abs(st.pitch_deg) > self.cfg.sem_snow_slope_veto * 45.0):
                return True, f"SNOW+SLOPE pitch={st.pitch_deg:.1f}°"

        return False, ""

    def get_safe_stop_command(self) -> ControlState:
        return ControlState(throttle=0.0, steer=0.0, brake=1.0)

    def semantic_speed_limit(
        self,
        semantic: Optional[SemanticOutput],
        imu: Optional[IMUData],
        base_speed: float,
    ) -> float:
        """Return the maximum allowed speed given semantic + IMU context."""
        cap = base_speed
        if semantic is not None:
            cap = min(cap, semantic.speed_cap)
        if imu is not None:
            if imu.vibration > self.cfg.imu_vibration_warn:
                vib_factor = 1.0 - 0.5 * (imu.vibration - self.cfg.imu_vibration_warn) / (
                    self.cfg.imu_vibration_danger - self.cfg.imu_vibration_warn + 1e-6)
                cap = min(cap, base_speed * clamp(vib_factor, 0.3, 1.0))
            if imu.slip_index > 0.4:
                cap = min(cap, 5.0)
        return max(cap, 1.5)   # always allow some minimum speed


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18.8  CONTROL ARBITRATOR
# ╚════════════════════════════════════════════════════════════════════════════╝

class ControlArbitrator:
    def __init__(self, cfg: Config, safety_monitor: SafetyMonitor):
        self.cfg = cfg; self.safety = safety_monitor

    def arbitrate(self, behavior_mode, safety_unsafe, recovery_cmd,
                  reactive_cmd, mppi_cmd, manual_cmd, st,
                  goal_dist, slope, roughness, obstacle_risk,
                  semantic=None, imu=None):
        if safety_unsafe:
            return self.safety.get_safe_stop_command(), "SAFETY_STOP"
        if behavior_mode == BehaviorPlanner.MANUAL and manual_cmd is not None:
            safe_manual = reactive_cmd if reactive_cmd.brake > 0.1 else manual_cmd
            return safe_manual, "MANUAL"
        if behavior_mode == BehaviorPlanner.GOAL_REACHED:
            return ControlState(brake=1.0), "GOAL_REACHED"
        if behavior_mode == BehaviorPlanner.RECOVERY and recovery_cmd is not None:
            return recovery_cmd, "RECOVERY"
        if behavior_mode == BehaviorPlanner.AVOID_OBSTACLE:
            return reactive_cmd, "AVOID_OBSTACLE"
        # AUTO: apply semantic speed cap on top of MPPI output
        cmd = reactive_cmd
        if semantic is not None or imu is not None:
            spd_cap = self.safety.semantic_speed_limit(semantic, imu, self.cfg.target_speed_flat)
            if st.speed > spd_cap * 1.05:
                # Soft-brake to honour speed cap
                excess = (st.speed - spd_cap) / max(spd_cap, 1e-3)
                cmd = ControlState(
                    throttle=cmd.throttle * max(0.0, 1.0 - excess),
                    steer=cmd.steer,
                    brake=clamp(excess * 0.3, 0.0, self.cfg.max_brake),
                    reverse=cmd.reverse,
                )
        return cmd, "AUTO"


# ╔════════════════════════════════════════════════════════════════════════════╗
# §19  NAVIGATION SYSTEM
# ╚════════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._client = self._world = self._vehicle = self._sensors = None
        self._original_settings = None

        # ── Perception stack ───────────────────────────────────────────────
        self._perception   = PerceptionModule(cfg)
        self._terrain_cls  = TerrainClassifier()
        self._imu_proc     = IMUProcessor(cfg)

        # ── Planning ───────────────────────────────────────────────────────
        self._subgoal_pl   = VSGPSubgoalPlanner(cfg)
        self._mppi         = MPPIPlanner(cfg)
        self._terrain      = TerrainAnalyzer(cfg)
        self._flat_planner = FlatGoalPlanner(cfg, self._terrain)

        # ── Memory layers ──────────────────────────────────────────────────
        self._stuck_memory   = StuckMemory(cfg)
        self._terrain_memory = TerrainMemory(cfg.terrain_memory_size, cfg.terrain_memory_resolution)
        self._topo_memory    = TopologicalMemory(cfg)

        # ── Safety / control ───────────────────────────────────────────────
        self._reactive   = ReactiveObstacleAvoidance(cfg)
        self._controller = Controller(cfg)
        self._manual     = ManualController(cfg)
        self._safety     = SafetyMonitor(cfg)
        self._arbitrator = ControlArbitrator(cfg, self._safety)

        # ── Mission ────────────────────────────────────────────────────────
        self._mission    = MissionManager(GOAL, cfg)
        self._behavior   = BehaviorPlanner()
        self._recovery   = StuckRecovery(cfg, self._flat_planner,
                                         self._terrain_memory, self._topo_memory)

        # ── Logger + viz ───────────────────────────────────────────────────
        self._logger     = ResultLogger(cfg)
        self._viz        = VisualizationModule(cfg)

        self._step            = 0
        self._start_wall      = time.time()
        self._nav6_prev_thr   = 0.0

        # Cached semantic / IMU
        self._last_semantic: Optional[SemanticOutput] = None
        self._last_imu:      Optional[IMUData]        = None

        pygame.init()
        self._screen = pygame.display.set_mode(
            (cfg.viz_win_w, cfg.viz_win_h), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            "VSGP-MPPI Off-Road Navigator  |  TAB=AUTO/MANUAL  ESC=quit  R=recover"
        )
        self._viz.init(self._screen)
        print(f"\n[MISSION] Goal → ({GOAL[0]:.1f}, {GOAL[1]:.1f})  "
              f"tolerance={cfg.goal_tolerance}m\n"
              "System plans its own path using VSGP + topological memory.\n")

    # ── CARLA helpers ──────────────────────────────────────────────────────

    def connect(self):
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port}")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world = self._client.get_world()
        if self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode = True
            s.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self._world.apply_settings(s)
        for _ in range(5):
            self._safe_tick()

    def _safe_tick(self, n=1):
        for _ in range(max(1, n)):
            self._world.tick() if self.cfg.synchronous else self._world.wait_for_tick()

    def spawn_vehicle(self):
        bp_lib = self._world.get_blueprint_library()
        bp     = bp_lib.find(self.cfg.vehicle_blueprint)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")
        self._vehicle = None
        for dz in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            tf = carla.Transform(
                carla.Location(x=self.cfg.spawn_x, y=self.cfg.spawn_y,
                               z=self.cfg.spawn_z + dz),
                carla.Rotation(pitch=self.cfg.spawn_pitch,
                               yaw=self.cfg.spawn_yaw,
                               roll=self.cfg.spawn_roll),
            )
            try:
                self._vehicle = self._world.try_spawn_actor(bp, tf)
            except RuntimeError:
                self._vehicle = None
            if self._vehicle is not None:
                print(f"[SPAWN] OK at z+{dz:.1f}")
                break
        if self._vehicle is None:
            raise RuntimeError("Could not spawn vehicle.")
        self._vehicle.set_simulate_physics(True)
        self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self._safe_tick(int(self.cfg.post_spawn_wait_s / self.cfg.fixed_delta_seconds))
        self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self._sensors = SensorManager(self._world, self._vehicle, self.cfg)
        print(f"[GOAL] ({GOAL[0]:.2f}, {GOAL[1]:.2f})")

    def get_state(self) -> VehicleState:
        tf    = self._vehicle.get_transform()
        vel   = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return VehicleState(
            float(tf.location.x), float(tf.location.y), float(tf.location.z),
            math.radians(float(tf.rotation.yaw)),
            float(tf.rotation.pitch), float(tf.rotation.roll), float(speed),
        )

    def apply_control(self, cmd: ControlState):
        self._vehicle.apply_control(carla.VehicleControl(
            throttle=clamp(cmd.throttle, 0, 1),
            steer=clamp(cmd.steer, -1, 1),
            brake=clamp(cmd.brake, 0, 1),
            reverse=bool(cmd.reverse),
            hand_brake=False,
        ))

    def force_velocity_push(self, st: VehicleState):
        if self._recovery.mode == StuckRecovery.FORWARD and st.speed < 0.35:
            vx = self.cfg.recovery_push_speed * math.cos(st.yaw)
            vy = self.cfg.recovery_push_speed * math.sin(st.yaw)
            try:
                self._vehicle.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            except Exception:
                pass

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        self.connect(); self.spawn_vehicle()
        self._running = True
        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[CLEANUP] Stopping vehicle")
        try:
            if self._vehicle and self._vehicle.is_alive:
                self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception: pass
        try:
            if self._sensors: self._sensors.destroy()
        except Exception: pass
        try:
            if self._vehicle and self._vehicle.is_alive: self._vehicle.destroy()
        except Exception: pass
        try:
            if self._world and self._original_settings:
                self._world.apply_settings(self._original_settings)
        except Exception: pass
        self._logger.finalize()
        pygame.quit()

    # ── Per-tick logic ─────────────────────────────────────────────────────

    def _tick(self):
        # 0. World tick
        if self.cfg.synchronous: self._world.tick()
        else:                    self._world.wait_for_tick()
        self._step += 1

        # 1. Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_TAB:
                    enter_manual = (self._behavior.mode != BehaviorPlanner.MANUAL)
                    self._behavior.set_manual(enter_manual)
                    self._controller.reset()
                    if not enter_manual: self._manual.reset()
                    print(f"[MODE] {'MANUAL' if enter_manual else 'AUTO'}")
                elif event.key == pygame.K_r:
                    if self._behavior.mode != BehaviorPlanner.MANUAL:
                        t_ev = time.time() - self._start_wall
                        self._recovery.force(t_ev, self.get_state(), self._mission.current_goal)
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._viz.init(self._screen)
        keys = pygame.key.get_pressed()

        # 2. Time + state
        t  = time.time() - self._start_wall
        st = self.get_state()

        # 3. Mission update
        if self._behavior.mode != BehaviorPlanner.MANUAL:
            if self._mission.update(st, t):
                self.apply_control(ControlState(brake=1.0))
                self._running = False; return

        goal_xy  = self._mission.current_goal
        goal_arr = self._mission.goal_array
        goal_dist = math.hypot(goal_arr[0] - st.x, goal_arr[1] - st.y)

        # 4. Sensors
        pts  = self._sensors.get_lidar_points()
        self._terrain.update(pts)
        if self._step % 10 == 0:
            self._logger.log_terrain_points(t, st, pts)

        # 4a. LiDAR → geometric perception
        perc = (self._perception.process_lidar(pts)
                if pts is not None
                else self._perception.last_output)

        # 4b. IMU
        imu_raw = self._sensors.get_imu_data()
        if imu_raw is not None:
            self._last_imu = self._imu_proc.process(imu_raw)
        imu = self._last_imu   # may be None if IMU not yet received

        # 4c. Semantic segmentation
        seg_img  = self._sensors.get_semantic_image()
        rgb_img  = self._sensors.get_front_image()
        if seg_img is not None or rgb_img is not None:
            # Re-classify every 5 ticks (expensive-ish)
            if self._step % 5 == 0 or self._last_semantic is None:
                self._last_semantic = self._terrain_cls.classify(seg_img, rgb_img, self.cfg)
        semantic = self._last_semantic

        # 4d. Fuse all perception layers
        if perc is not None:
            perc = self._perception.fuse(perc, semantic, imu)

        # 4e. Camera images for viz
        front = rgb_img
        rear  = self._sensors.get_rear_image()
        if front is not None: self._viz.set_front(front)
        if rear  is not None: self._viz.set_rear(rear)

        raw_pts = perc.raw_points if perc is not None else None

        # 4f. Terrain memory update from LiDAR
        if pts is not None and len(pts) > 0:
            self._terrain_memory.update_from_lidar(
                pts, st.x, st.y, self.cfg.terrain_memory_lidar_downsample,
                alpha_grid=perc.alpha_grid if perc is not None else None,
                void_risk_map=perc.void_risk_map if perc is not None else None,
            )

        # 4g. Topological memory update
        geo_slope, geo_rough, _, _, _ = self._terrain.local_risk(0.0, 0.0, radius=3.0)
        self._topo_memory.update(
            x=st.x, y=st.y, t=t,
            speed=st.speed,
            imu=imu,
            semantic=semantic,
            geo_slope=geo_slope,
            geo_rough=geo_rough,
        )

        # 5. Behavior
        in_recovery = self._recovery.mode != StuckRecovery.IDLE
        if self._behavior.mode != BehaviorPlanner.MANUAL:
            self._behavior.update(
                mission_complete=self._mission.complete,
                obstacle_active=self._reactive.obstacle_active,
                in_recovery=in_recovery,
            )
        bmode = self._behavior.mode

        # 6. Safety check (now includes semantic + IMU)
        fwd_clearance = self._terrain.forward_clearance()
        void_risk     = perc.void_risk if perc is not None else 0.0
        safety_unsafe, safety_reason = self._safety.check(
            st, imu=imu, semantic=semantic,
            void_risk=void_risk, forward_clearance=fwd_clearance,
            wall_t=t,
        )

        # Tell MPPI about current semantic + IMU context
        self._mppi.set_semantic_imu(semantic, imu)

        # 7. Planning
        subgoal_viz = None; all_cands = None
        cur_cost = cur_reward = cur_flatness = cur_progress = 0.0
        cur_slope = cur_rough = cur_obs = cur_clearance = 0.0
        manual_cmd = recovery_cmd = None
        mppi_cmd   = ControlState(throttle=0.1)

        if bmode == BehaviorPlanner.MANUAL:
            manual_cmd = self._manual.tick(keys, st.roll_deg, st.pitch_deg)
            manual_cmd = self._reactive.process(raw_pts, manual_cmd, st.speed)

        elif bmode == BehaviorPlanner.RECOVERY:
            recovery_cmd = self._recovery.cmd() or ControlState(throttle=0.1)
            self._stuck_memory.update(np.array([st.x, st.y]), stuck=True)

        else:  # AUTO / AVOID_OBSTACLE
            if perc is not None:
                # VSGP subgoal with all three memory layers + semantics
                best_sg, all_cands = self._subgoal_pl.plan(
                    perc,
                    vehicle_pos=np.array([st.x, st.y, st.z], dtype=np.float32),
                    vehicle_yaw_rad=st.yaw,
                    goal_pos=goal_arr,
                    terrain_memory=self._terrain_memory,
                    topo_memory=self._topo_memory,
                    semantic=semantic,
                )

                clearance = 0.0
                if best_sg is not None:
                    _, _, _, _, clearance = self._terrain.local_risk(
                        float(best_sg.local_pos[0]),
                        float(best_sg.local_pos[1]), radius=5.0,
                    )
                    cur_clearance = clearance
                    if clearance > self.cfg.clearance_block_threshold:
                        best_sg = None

                if best_sg is None:
                    cand       = self._flat_planner.plan(st, goal_xy)
                    subgoal_viz = np.array([cand.wx, cand.wy, 0.0])
                    cur_flatness = cand.flatness; cur_progress = cand.progress
                    cur_reward = cand.reward; cur_slope = cand.slope
                    cur_rough  = cand.roughness; cur_obs = cand.obstacle_risk
                    cur_clearance = cand.clearance_risk
                else:
                    subgoal_viz  = best_sg.world_pos
                    cur_flatness = best_sg.traversability; cur_progress = best_sg.goal_progress
                    cur_slope    = best_sg.slope; cur_rough = best_sg.roughness
                    cur_obs      = best_sg.occupancy; cur_clearance = clearance

                state_vec  = [st.x, st.y, st.yaw, st.speed]
                mem_tensor = self._stuck_memory.get_tensor(self._mppi.device)
                mppi_cmd, cur_cost = self._mppi.plan(state_vec, perc, subgoal_viz, mem_tensor)
                mppi_cmd = self._controller.apply_safety_filters(
                    mppi_cmd, st.roll_deg, st.pitch_deg, speed=st.speed
                )

                self._recovery.update(t, st, mppi_cmd, goal_xy)
                self._stuck_memory.update(np.array([st.x, st.y]), stuck=False)

                rec_cmd2 = self._recovery.cmd()
                if rec_cmd2 is not None and self._recovery.mode != StuckRecovery.IDLE:
                    recovery_cmd = rec_cmd2
                    self._behavior.update(
                        mission_complete=self._mission.complete,
                        obstacle_active=self._reactive.obstacle_active,
                        in_recovery=True,
                    )
                    bmode = self._behavior.mode
                    self._controller.reset()
                    self._stuck_memory.update(np.array([st.x, st.y]), stuck=True)
                    self._terrain_memory.mark_stuck(st.x, st.y, cost=1.0)
                    self._topo_memory.mark_stuck(st.x, st.y)
                else:
                    mppi_cmd = self._reactive.process(raw_pts, mppi_cmd, st.speed)
                    mppi_cmd = self._apply_throttle_bias(
                        mppi_cmd, st, goal_dist, cur_slope, cur_rough, cur_obs, semantic, imu
                    )

        # 8. Arbitration
        final_cmd, final_mode = self._arbitrator.arbitrate(
            behavior_mode=bmode,
            safety_unsafe=safety_unsafe,
            recovery_cmd=recovery_cmd,
            reactive_cmd=mppi_cmd,
            mppi_cmd=mppi_cmd,
            manual_cmd=manual_cmd,
            st=st,
            goal_dist=goal_dist,
            slope=cur_slope, roughness=cur_rough, obstacle_risk=cur_obs,
            semantic=semantic, imu=imu,
        )
        if safety_unsafe:
            final_mode = f"SAFETY ({safety_reason})"

        self.apply_control(final_cmd)
        self.force_velocity_push(st)

        # 9. Memory decay
        self._terrain_memory.decay()

        # 10. Logging
        cost_total   = cur_cost + 0.3*final_cmd.throttle**2 + 0.8*final_cmd.steer**2 + 0.4*final_cmd.brake**2
        reward_total = cur_reward + 8.0*st.speed - 0.35*goal_dist

        if all_cands and all_cands[0] is not None:
            log_cand = all_cands[0]
        else:
            log_cand = Candidate(
                st.x, st.y, 0, 0, 0, 0,
                cur_progress, goal_dist, 0,
                cur_slope, cur_rough, cur_obs, cur_flatness, cur_clearance, 0.0,
                cur_cost, cur_reward,
            )
        self._logger.log(
            t, st, final_cmd, log_cand, goal_dist, final_mode,
            cost_total, reward_total,
            semantic_class=semantic.dominant_class.name if semantic else "UNKNOWN",
            imu_vibration=imu.vibration if imu else 0.0,
            safe_topo_nodes=self._topo_memory.safe_node_count(),
        )

        if self._step % (self.cfg.viz_every_n_ticks * 5) == 0:
            sem_str  = semantic.dominant_class.name if semantic else "?"
            vib_str  = f"{imu.vibration:.2f}" if imu else "N/A"
            slip_str = f"{imu.slip_index:.2f}" if imu else "N/A"
            print(
                f"\r[NAV] t={t:6.1f}s dist={goal_dist:7.2f}m spd={st.speed:5.2f} "
                f"thr={final_cmd.throttle:4.2f} steer={final_cmd.steer:+5.2f} "
                f"terrain={sem_str:<10} vib={vib_str} slip={slip_str} "
                f"mode={final_mode:>22s}",
                end="", flush=True,
            )

        # 11. Visualization
        self._viz.push(
            pos=np.array([st.x, st.y, st.z]),
            speed=st.speed, steer=final_cmd.steer, perc=perc,
            mode=final_mode, cost=cost_total, reward=reward_total,
            subgoal=subgoal_viz, all_candidates=all_cands,
            mem_pts=self._stuck_memory.points, dist_goal=goal_dist,
            obs_warn=self._reactive.obstacle_active,
            flatness=cur_flatness, progress=cur_progress,
            pitch=st.pitch_deg, roll=st.roll_deg,
            semantic=semantic, imu=imu,
            safety_active=safety_unsafe, safety_reason=safety_reason,
            topo_nodes=list(self._topo_memory.nodes),
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()

    def _apply_throttle_bias(
        self, ctrl, st, goal_dist, slope, roughness, obstacle_risk,
        semantic=None, imu=None,
    ) -> ControlState:
        out = ControlState(ctrl.throttle, ctrl.steer, ctrl.brake, ctrl.reverse)
        if out.reverse or out.brake > 0.05 or self._recovery.mode != StuckRecovery.IDLE:
            self._nav6_prev_thr = out.throttle; return out

        bad = clamp(0.65*slope + 0.35*roughness + 0.25*obstacle_risk, 0.0, 1.0)
        target_speed = (1.0 - bad)*self.cfg.target_speed_flat + bad*self.cfg.target_speed_bad
        if goal_dist < 30.0:
            target_speed = min(target_speed, self.cfg.target_speed_near_goal)
        target_speed *= max(0.60, 1.0 - 0.20*abs(out.steer)/max(self.cfg.max_steer, 1e-6))

        # Apply semantic speed limit
        if semantic is not None or imu is not None:
            target_speed = min(target_speed,
                               self._safety.semantic_speed_limit(semantic, imu, target_speed))

        min_thr = (self.cfg.min_throttle_hill
                   if (abs(st.pitch_deg) > 7.0 or slope > 0.38)
                   else self.cfg.min_throttle_auto)
        desired = clamp(self.cfg.kp_speed*(target_speed - st.speed), min_thr, self.cfg.max_throttle)
        out.throttle = max(out.throttle, desired)
        out.throttle = clamp(out.throttle,
                             self._nav6_prev_thr - self.cfg.throttle_rate,
                             self._nav6_prev_thr + self.cfg.throttle_rate)
        tol = self.cfg.goal_tolerance
        if goal_dist > tol and st.speed < 0.5:
            out.throttle = max(out.throttle, min_thr)
        out.throttle = clamp(out.throttle, 0.0, self.cfg.max_throttle)
        self._nav6_prev_thr = out.throttle
        return out


# ╔════════════════════════════════════════════════════════════════════════════╗
# §20  ENTRY POINT
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    nav = NavigationSystem(CFG)
    try:
        nav.run()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}"); traceback.print_exc()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
