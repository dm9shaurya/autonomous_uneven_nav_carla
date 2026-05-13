"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VSGP-MPPI OFF-ROAD NAVIGATOR + TCN CORRIDOR PLANNER  v4.0                 ║
║  Base: VSGP-MPPI Navigator (v3.0)                                           ║
║  Added: Topology-Inspired Corridor Navigation (TCN v3.0)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture:                                                               ║
║    LiDAR → VSGP Perception → Fused Traversability (spherical GP grid)       ║
║    LiDAR → ElevationGrid → TCN Corridor Graph → Corridor Waypoints          ║
║    Subgoal: VSGP candidates + TCN corridor bias + TCN fallback               ║
║    MPPI trajectory optimisation w/ terrain + semantic + IMU cost             ║
║    Reactive safety overlay + anti-tip controller                             ║
║    Topological & dense terrain memory for long-range path quality            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import csv
import enum
import json
import math
import os
import random
import sqlite3
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Deque, Dict, List, Optional, Set, Tuple, Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# scikit-image  (required by TCN skeleton step)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from skimage.morphology import skeletonize
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print("[WARN] scikit-image not found – TCN skeleton step disabled. "
          "pip install scikit-image")

# ─────────────────────────────────────────────────────────────────────────────
# scipy  (shared by VSGP kernel and TCN classifier)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.spatial import KDTree
    from scipy.special import softmax as scipy_softmax
    from scipy.ndimage import convolve as ndimage_convolve
    from scipy.ndimage import gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not found – TCN classifier disabled. pip install scipy")

# ─────────────────────────────────────────────────────────────────────────────
# sklearn
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
# §0  GOAL DEFINITION
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
    spawn_x: float = 882.52
    spawn_y: float = -738.89
    spawn_z: float = 20.07
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
    imu_vibration_window: int = 20
    imu_slip_threshold: float = 0.55
    imu_vibration_danger: float = 2.0
    imu_vibration_warn: float = 0.40
    imu_slip_danger: float = 0.90
    imu_bad_duration_s: float = 2.0

    # ── Semantic terrain
    semantic_forward_rows_frac: float = 0.60
    semantic_center_cols_frac: float = 0.40
    sem_rock_slope_veto: float = 0.45
    sem_snow_slope_veto: float = 0.35

    # ── Topological memory
    topo_node_spacing: float = 4.0
    topo_max_nodes: int = 600
    topo_safe_cost_thresh: float = 0.40
    topo_backtrack_min_dist: float = 8.0
    topo_decay_per_tick: float = 0.9995
    topo_corridor_radius: float = 6.0
    topo_corridor_bonus: float = 3.0

    # ── VSGP Subgoal Planning
    subgoal_distance: float = 10.0
    subgoal_min_distance: float = 2.0
    subgoal_num_angles: int = 12
    subgoal_num_depth: int = 12

    # ── MPPI
    mppi_horizon: int = 30
    mppi_num_samples: int = 1024
    mppi_lambda: float = 1.0
    mppi_noise_throttle: float = 0.2
    mppi_noise_steer: float = 0.3

    # ── Cost Weights
    w_goal: float = 500.0
    w_heading: float = 40.0
    w_terrain_risk: float = 8.0
    w_memory: float = 30.0
    w_learned: float = 0.5
    w_semantic: float = 10.0
    w_imu: float = 2.5

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
    kp_speed: float = 1.32
    throttle_rate: float = 0.30

    # ── Safety / Anti-tip
    max_safe_roll_deg: float = 30.0
    max_safe_pitch_deg: float = 35.0
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float = 0.5
    anti_tip_roll_warn: float = 30.0
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

    # ── Memory
    memory_radius: float = 6.0
    terrain_memory_cost_weight: float = 2.0

    # ── Obstacle Detection
    obstacle_z_min: float = 1.85
    obstacle_y_half_width: float = 2.4
    emergency_clearance: float = 1.40
    danger_clearance: float = 2.50

    # ── Mission stall
    mission_stall_ticks: int = 200

    # ── TerrainMemory (dense grid)
    terrain_memory_size: int = 600
    terrain_memory_resolution: float = 1.0
    terrain_memory_lidar_downsample: int = 50
    terrain_memory_lookahead_steps: int = 3
    terrain_memory_lookahead_weight: float = 1.5

    # ── Visualization
    viz_win_w: int = 1280
    viz_win_h: int = 720
    viz_every_n_ticks: int = 2
    traj_history: int = 1200

    # ── Output
    out_dir: str = "unified_nav_results"

    # ── TCN Corridor Planner  ◄ NEW
    tcn_enabled: bool = True          # master switch
    tcn_update_freq: int = 5          # run TCN every N ticks
    tcn_grid_size: int = 64           # elevation grid cells (NxN)
    tcn_grid_res: float = 0.5         # metres per cell
    tcn_corridor_bias: float = 20.0   # cost reduction for VSGP cands in corridor
    tcn_valley_prob_thresh: float = 0.50   # min P(valley) to accept corridor pixel
    tcn_rough_prob_thresh: float = 0.20    # max P(rough)  to accept corridor pixel


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

SEMANTIC_TRAV_MULT: Dict[TerrainClass, float] = {
    TerrainClass.UNKNOWN:    0.55,
    TerrainClass.DIRT:       0.92,
    TerrainClass.GRASS:      0.85,
    TerrainClass.GRAVEL:     0.75,
    TerrainClass.ROCK:       0.50,
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
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    vibration: float = 0.0
    slip_index: float = 0.0
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    stability_score: float = 1.0


@dataclass
class SemanticOutput:
    dominant_class: TerrainClass = TerrainClass.UNKNOWN
    forward_class: TerrainClass = TerrainClass.UNKNOWN
    class_counts: Dict[int, int] = field(default_factory=dict)
    traversability_mult: float = 0.6
    speed_cap: float = 8.0
    semantic_risk: float = 0.35


@dataclass
class TopologicalNode:
    x: float; y: float
    cost: float = 0.5
    visit_count: int = 0
    semantic_class: TerrainClass = TerrainClass.UNKNOWN
    geometric_cost: float = 0.5
    imu_vibration: float = 0.0
    success_score: float = 0.0
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
    semantic: Optional[SemanticOutput] = None
    imu: Optional[IMUData] = None
    fused_traversability: Optional[np.ndarray] = None


@dataclass
class Subgoal:
    alpha: float; beta: float; distance: float
    local_pos: np.ndarray; world_pos: np.ndarray
    slope: float; roughness: float; occupancy: float
    variance: float; traversability: float; terrain_cost: float
    goal_progress: float; heading_error: float; safe: bool
    width_m: float; cost: float = 0.0


@dataclass
class RouteGuidance:
    active: bool
    target_xy: Optional[np.ndarray]
    nearest_xy: Optional[np.ndarray]
    nearest_idx: int = -1
    target_idx: int = -1
    route_length: float = 0.0
    remaining_m: float = 0.0
    progress_ratio: float = 0.0
    corridor_distance: float = float("inf")
    end_distance: float = float("inf")
    run_id: int = -1


# ╔════════════════════════════════════════════════════════════════════════════╗
# §4  TCN – TOPOLOGY-INSPIRED CORRIDOR NAVIGATION  ◄ NEW
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║  §4.0  TopoClassifier  – real probability distributions over terrain      ║
# ║  §4.1  CorridorEdge / CorridorGraph – skeleton-traced corridor graph      ║
# ║  §4.2  ElevationGridBuilder – LiDAR → Cartesian height grid               ║
# ║  §4.3  TCNPlanner – top-level: classify → skeleton → graph → nav point    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class TopoClassifier:
    """
    Computes P(terrain_label | features) over a 2-D elevation window.

    Features per cell: [MeanCurvature, HeightStd, GradientMagnitude]
    Labels (internal): 1=VALLEY, 2=FLAT, 4=RIDGE, 7=ROUGH
    Uses softmax of negative squared distances to feature centroids.
    """
    # Feature centroids: [MeanCurv, Roughness(std), GradMag]
    _CENTROIDS: Dict[int, List[float]] = {
        1: [-0.15, 0.05, 0.10],   # VALLEY – concave, smooth, gentle slope
        4: [ 0.20, 0.10, 0.20],   # RIDGE  – convex
        7: [ 0.00, 0.60, 0.40],   # ROUGH  – high height variance
        2: [ 0.00, 0.02, 0.05],   # FLAT   – near-zero everything
    }
    N_LABELS = 8   # probability array width (indices 0–7)

    def __init__(self):
        self.keys = list(self._CENTROIDS.keys())
        self.mu   = np.array([self._CENTROIDS[k] for k in self.keys], dtype=np.float32)

    def compute_probs(
        self,
        mc: np.ndarray,      # (H,W) mean curvature
        h_std: np.ndarray,   # (H,W) height standard deviation
        grad: np.ndarray,    # (H,W) gradient magnitude
    ) -> np.ndarray:         # (H,W,8) probability tensor
        if not SCIPY_OK:
            H, W = mc.shape
            return np.zeros((H, W, self.N_LABELS), dtype=np.float32)

        features = np.stack([mc, h_std, grad], axis=-1)   # (H,W,3)
        diff     = features[..., np.newaxis, :] - self.mu  # (H,W,N,3)
        dist_sq  = np.sum(diff ** 2, axis=-1)              # (H,W,N)

        probs_raw = scipy_softmax(-dist_sq * 10.0, axis=-1)  # (H,W,N)

        H, W = mc.shape
        full = np.zeros((H, W, self.N_LABELS), dtype=np.float32)
        for i, lbl in enumerate(self.keys):
            full[..., lbl] = probs_raw[..., i]
        return full


@dataclass
class CorridorEdge:
    node_a: int
    node_b: int
    weight: float                        # accumulated roughness cost
    path_pixels: List[Tuple[int, int]]


class CorridorGraph:
    """
    Builds a sparse graph by tracing skeleton pixels between junction nodes.

    FIX vs original: world coordinate conversion corrected from
      (ox + pos - half) * res   →   ox + (pos - half) * res
    so that ox/oy are vehicle world-metres, not grid-unit offsets.
    """

    def __init__(self, res: float):
        self.res   = res
        self.nodes: Dict[int, np.ndarray] = {}   # id → [wx, wy]
        self.edges: List[CorridorEdge] = []

    def build_from_skeleton(
        self,
        skel:  np.ndarray,   # (H,W) bool skeleton
        probs: np.ndarray,   # (H,W,8) terrain probabilities
        ox:    float,        # vehicle world-x [m]
        oy:    float,        # vehicle world-y [m]
        size:  int,
    ) -> None:
        self.nodes.clear()
        self.edges.clear()

        if not SCIPY_OK or not np.any(skel):
            return

        # ── 1. Identify junction / endpoint pixels ──────────────────────────
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.int32)
        neighs = ndimage_convolve(skel.astype(np.int32), kernel, mode='constant')
        keypoint_mask = skel & ((neighs == 11) | (neighs > 12))

        kp_indices   = np.argwhere(keypoint_mask)          # list of [row, col]
        pixel_to_id  = {tuple(pos): i for i, pos in enumerate(kp_indices)}

        # ── 2. Convert keypoints to world nodes ─────────────────────────────
        half = size // 2
        for pos_t, nid in pixel_to_id.items():
            # FIX: ox/oy are in metres; pos is in grid cells
            wx = ox + (pos_t[0] - half) * self.res
            wy = oy + (pos_t[1] - half) * self.res
            self.nodes[nid] = np.array([wx, wy], dtype=np.float32)

        # ── 3. Trace edges between adjacent keypoints ────────────────────────
        visited_edges: Set[Tuple[int, int]] = set()

        for pos_t, start_id in pixel_to_id.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (pos_t[0] + dr, pos_t[1] + dc)
                    if self._is_skel(skel, nbr) and nbr not in pixel_to_id:
                        end_id, path = self._trace_path(nbr, pos_t, skel, pixel_to_id)
                        if end_id is None:
                            continue
                        key = tuple(sorted((start_id, end_id)))
                        if key not in visited_edges:
                            weight = self._path_weight(path, probs)
                            self.edges.append(
                                CorridorEdge(start_id, end_id, weight, path)
                            )
                            visited_edges.add(key)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _trace_path(
        self,
        current: Tuple[int, int],
        prev:    Tuple[int, int],
        skel:    np.ndarray,
        keypoints: Dict[Tuple[int, int], int],
        max_steps: int = 2000,          # FIX: guard against infinite loops
    ) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        path  = [current]
        steps = 0
        while current not in keypoints:
            if steps >= max_steps:
                return None, []
            found = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (current[0] + dr, current[1] + dc)
                    if self._is_skel(skel, nxt) and nxt != prev:
                        prev, current = current, nxt
                        path.append(current)
                        found = True
                        break
                if found:
                    break
            if not found:
                return None, []   # dead-end branch
            steps += 1
        return keypoints[current], path

    @staticmethod
    def _is_skel(skel: np.ndarray, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < skel.shape[0] and 0 <= c < skel.shape[1] and bool(skel[r, c])

    @staticmethod
    def _path_weight(path: List[Tuple[int, int]], probs: np.ndarray) -> float:
        # Accumulate roughness probability (label index 7) along the path
        total = 0.0
        for r, c in path:
            if 0 <= r < probs.shape[0] and 0 <= c < probs.shape[1]:
                total += float(probs[r, c, 7]) + 0.1
        return total


class ElevationGridBuilder:
    """
    Converts raw LiDAR points (vehicle-local frame) into a Cartesian
    height-grid required by TCNPlanner.

    Returns:
        (h_win, c_win, s_win): height mean, curvature proxy, height std – each (size×size)
        (vehicle_x, vehicle_y, size, res): metadata for coordinate conversion
    """

    def __init__(self, size: int = 64, resolution: float = 0.5):
        self.size = size
        self.res  = resolution

    def build(
        self,
        pts:       np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
               Optional[Tuple[float, float, int, float]]]:
        if pts is None or len(pts) < 20:
            return None, None

        half  = self.size // 2
        h_sum = np.zeros((self.size, self.size), dtype=np.float64)
        h_sq  = np.zeros((self.size, self.size), dtype=np.float64)
        count = np.zeros((self.size, self.size), dtype=np.int32)

        gx = np.floor(pts[:, 0] / self.res + half).astype(np.int32)
        gy = np.floor(pts[:, 1] / self.res + half).astype(np.int32)
        lz = pts[:, 2]

        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)
        gx, gy, lz = gx[valid], gy[valid], lz[valid]

        np.add.at(h_sum, (gx, gy), lz)
        np.add.at(h_sq,  (gx, gy), lz * lz)
        np.add.at(count, (gx, gy), 1)

        mask   = count > 0
        c_safe = np.maximum(count, 1).astype(np.float64)
        h_mean = np.where(mask, h_sum / c_safe, 0.0).astype(np.float32)
        h_std  = np.where(
            mask,
            np.sqrt(np.maximum(h_sq / c_safe - (h_sum / c_safe) ** 2, 0.0)),
            0.0,
        ).astype(np.float32)

        # Curvature: Laplacian of Gaussian-smoothed height
        try:
            h_smooth = gaussian_filter(h_mean, sigma=1.5) if SCIPY_OK else h_mean
        except Exception:
            h_smooth = h_mean

        dz_dx  = np.gradient(h_smooth, self.res, axis=1)
        dz_dy  = np.gradient(h_smooth, self.res, axis=0)
        d2z_dx = np.gradient(dz_dx, self.res, axis=1)
        d2z_dy = np.gradient(dz_dy, self.res, axis=0)
        c_win  = np.abs((d2z_dx + d2z_dy) / 2.0).astype(np.float32)

        meta = (vehicle_x, vehicle_y, self.size, self.res)
        return (h_mean, c_win, h_std), meta


class TCNPlanner:
    """
    Topology-Inspired Corridor Navigation Planner.

    Pipeline per tick (update()):
      1. Compute real terrain probabilities via TopoClassifier (softmax).
      2. Build binary valley/corridor mask from probability thresholds.
      3. Skeletonise the mask → topological backbone.
      4. Build CorridorGraph: nodes at junctions, edges weighted by roughness.
      5. Return the node closest to the final goal as a nav waypoint.

    The returned point is in world coordinates and can be used as:
      - A corridor-bias hint for the VSGP subgoal planner.
      - A fallback nav waypoint when VSGP finds no safe candidates.
    """

    def __init__(self, goal_x: float, goal_y: float, res: float = 0.5):
        self.classifier = TopoClassifier()
        self.graph      = CorridorGraph(res)
        self.goal       = np.array([goal_x, goal_y], dtype=np.float32)
        self.res        = res
        self._last_nav: Optional[np.ndarray] = None

    def update(
        self,
        elevation_window: Tuple[np.ndarray, np.ndarray, np.ndarray],
        window_meta:      Tuple[float, float, int, float],
        valley_thresh: float = 0.50,
        rough_thresh:  float = 0.20,
    ) -> Optional[np.ndarray]:
        """
        Args:
            elevation_window: (h_win, c_win, s_win) – height, curvature, std grids
            window_meta:      (ox_m, oy_m, size, res) – vehicle world pos + grid info
            valley_thresh:    min P(valley) to include cell in corridor mask
            rough_thresh:     max P(rough)  to include cell in corridor mask
        Returns:
            World-coordinate nav point [wx, wy] or None.
        """
        if not SKIMAGE_OK or not SCIPY_OK:
            return self._last_nav

        h_win, c_win, s_win = elevation_window
        ox, oy, size, _     = window_meta

        # ── 1. Derivatives for probability features ──────────────────────────
        dx = self.res
        dz_dx = np.gradient(h_win, dx, axis=1)
        dz_dy = np.gradient(h_win, dx, axis=0)
        grad  = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        d2z_dx2 = np.gradient(dz_dx, dx, axis=1)
        d2z_dy2 = np.gradient(dz_dy, dx, axis=0)
        mc = (d2z_dx2 + d2z_dy2) / 2.0

        # ── 2. Terrain probabilities ─────────────────────────────────────────
        probs = self.classifier.compute_probs(mc, s_win, grad)

        # ── 3. Valley / corridor mask ────────────────────────────────────────
        mask = (probs[..., 1] > valley_thresh) & (probs[..., 7] < rough_thresh)
        if not np.any(mask):
            return self._last_nav

        # ── 4. Skeletonise ───────────────────────────────────────────────────
        try:
            skel = skeletonize(mask)
        except Exception:
            return self._last_nav

        # ── 5. Build corridor graph ──────────────────────────────────────────
        self.graph.build_from_skeleton(skel, probs, ox, oy, size)

        # ── 6. Select best nav node ──────────────────────────────────────────
        nav = self._select_nav_node()
        if nav is not None:
            self._last_nav = nav
        return nav

    def _select_nav_node(self) -> Optional[np.ndarray]:
        if not self.graph.nodes:
            return None
        best_pos  = None
        min_dist  = float('inf')
        for nid, pos in self.graph.nodes.items():
            d = float(np.linalg.norm(pos - self.goal))
            if d < min_dist:
                min_dist = d
                best_pos = pos
        return best_pos.copy() if best_pos is not None else None


# ╔════════════════════════════════════════════════════════════════════════════╗
# §5  VARIATIONAL SPARSE GAUSSIAN PROCESS
# ╚════════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha=1.0, length_scale=0.3, variance=1.0):
        self.alpha, self.length_scale, self.variance = alpha, length_scale, variance

    def __call__(self, X, Y):
        if not len(X) or not len(Y):
            return np.zeros((len(X), len(Y)))
        diff  = X[:, None, :] - Y[None, :, :]
        sq_d  = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X):
        return self.variance * np.ones(len(X))


class VSGP:
    def __init__(self, cfg: Config = CFG):
        self.cfg     = cfg
        self.kernel  = RationalQuadraticKernel(cfg.vsgp_alpha, cfg.vsgp_length_scale)
        self.noise_var  = cfg.vsgp_noise_var
        self.n_ind      = cfg.vsgp_n_inducing
        self.lr         = cfg.vsgp_lr
        self.update_freq = cfg.vsgp_update_freq
        side = int(math.sqrt(self.n_ind))
        A, B = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, side),
            np.linspace(-np.pi / 4, np.pi / 4, side),
        )
        self.Z  = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]
        m = len(self.Z)
        self.mu = np.zeros(m)
        self.Su = np.eye(m) * 0.1
        self._Kuu_inv:    Optional[np.ndarray] = None
        self._trained     = False
        self._update_count = 0
        self._last_X_hash  = 0

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
        m   = len(self.Z)
        Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6
        try:
            self._Kuu_inv = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            self._Kuu_inv = np.linalg.pinv(Kuu)
        Kfu    = self.kernel(X, self.Z)
        A      = Kfu @ self._Kuu_inv
        ni     = 1.0 / self.noise_var
        Lambda = ni * (A.T @ A) + self._Kuu_inv
        rhs    = ni * (A.T @ y)
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
        Ksu      = self.kernel(Xs, self.Z)
        A        = Ksu @ self._Kuu_inv
        mean     = A @ self.mu
        Kss_diag = self.kernel.diag(Xs)
        var_exp  = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
        var_var  = np.sum(A @ self.Su * A, axis=1)
        var      = np.clip(Kss_diag - var_exp + var_var + self.noise_var, 1e-6, None)
        return mean, var


# ╔════════════════════════════════════════════════════════════════════════════╗
# §6  PERCEPTION MODULE  (fused: geometry + semantics + IMU)
# ╚════════════════════════════════════════════════════════════════════════════╝

class PerceptionModule:
    ALPHA_RES = 60; BETA_RES = 30; ROC = 5.0
    SLOPE_WEIGHT = 1.2; OCC_WEIGHT = 1.0
    VAR_FREE_WEIGHT = 1.0; ROUGHNESS_WEIGHT = 3.0

    def __init__(self, cfg: Config = CFG):
        self.cfg  = cfg
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
        gp    = pts[np.abs(pts[:, 2] - z_med) < 1.0]
        if len(gp) < 10:
            return 0.0
        centered = gp - gp.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        normal   = vh[-1]
        cos_ang  = np.abs(normal[2]) / (np.linalg.norm(normal) + 1e-9)
        return float(np.clip(np.degrees(np.arccos(np.clip(cos_ang, -1, 1))), 0, 90))

    def _directional_void_risk(self, pts, alpha_centers):
        if pts is None or len(pts) < 10:
            void = 1.0
            return void, np.full((self.BETA_RES, len(alpha_centers)), void, dtype=np.float32)

        forward_pts = pts[pts[:, 0] > 1.5]
        if len(forward_pts) < 10:
            void = 1.0
            return void, np.full((self.BETA_RES, len(alpha_centers)), void, dtype=np.float32)

        angles    = np.arctan2(forward_pts[:, 1], forward_pts[:, 0])
        ranges    = np.linalg.norm(forward_pts[:, :2], axis=1)
        sector_ids = np.argmin(np.abs(angles[:, None] - alpha_centers[None, :]), axis=1)

        sector_void = np.ones(len(alpha_centers), dtype=np.float32)
        for idx in range(len(alpha_centers)):
            sector_ranges = np.sort(ranges[sector_ids == idx])
            if sector_ranges.size == 0:
                sector_void[idx] = 1.0; continue
            density = np.clip(sector_ranges.size / 16.0, 0.0, 1.0)
            if sector_ranges.size > 1:
                gap  = np.clip(np.max(np.diff(sector_ranges)) / 8.0, 0.0, 1.0)
                span = np.clip((sector_ranges[-1] - sector_ranges[0]) / self.cfg.lidar_range, 0.0, 1.0)
            else:
                gap = 0.35; span = 0.0
            sector_void[idx] = np.clip(
                0.55 * (1.0 - density) + 0.25 * gap
                + 0.15 * span + 0.05 * float(sector_ranges[0] > 12.0),
                0.0, 1.0,
            )

        mid   = len(alpha_centers) // 2
        lo, hi = max(0, mid - 3), min(len(alpha_centers), mid + 4)
        scalar_void  = float(np.clip(np.mean(sector_void[lo:hi]), 0.0, 1.0))
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
        pts  = raw[(dist > 0.5) & (dist < self.cfg.lidar_range)]
        if pts.shape[0] < 10:
            return self._last_output

        ground_slope = self._estimate_ground_slope(pts)
        r     = np.linalg.norm(pts, axis=1) + 1e-9
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta  = np.arcsin(np.clip(pts[:, 2] / r, -1.0, 1.0))
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
        slope     = np.sqrt(da ** 2 + db ** 2)
        roughness = np.sqrt(np.gradient(da, axis=1) ** 2 + np.gradient(db, axis=0) ** 2)
        d2a = np.gradient(da, axis=1); d2b = np.gradient(db, axis=0)
        curvature = np.sqrt(d2a ** 2 + d2b ** 2)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return np.zeros_like(arr) if mx - mn < 1e-6 else (arr - mn) / (mx - mn)

        slope_n = _norm(slope); rough_n = _norm(roughness)
        curv_n  = _norm(curvature); occ_n = _norm(mean); var_n = _norm(var)
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

    def fuse(self, out, semantic, imu):
        if out is None:
            return out
        fused = out.traversability.copy()
        if semantic is not None:
            fused *= semantic.traversability_mult
            out.semantic = semantic
        if imu is not None:
            stability_pen = 1.0 - (
                0.5 * imu.vibration
                + 0.3 * imu.slip_index
                + 0.2 * clamp(abs(imu.pitch_rate) / 30.0, 0.0, 1.0)
            )
            fused *= clamp(stability_pen, 0.25, 1.0)
            out.imu = imu
        out.fused_traversability = np.clip(fused, 0.0, 1.0)
        return out

    @property
    def last_output(self):
        return self._last_output


# ╔════════════════════════════════════════════════════════════════════════════╗
# §7  TERRAIN CLASSIFIER
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainClassifier:
    _LABEL_ROAD       = {6, 7, 8}
    _LABEL_VEGETATION = {9}
    _LABEL_GROUND     = {14, 22}
    _LABEL_WATER      = {21}
    _LABEL_SKY        = {13}

    def classify(self, seg_img, rgb_img, cfg: Config) -> SemanticOutput:
        if seg_img is None:
            return SemanticOutput()
        h, w = seg_img.shape[:2]
        if seg_img.ndim == 3 and seg_img.shape[2] >= 3:
            labels = seg_img[:, :, 2].astype(np.int32)
        else:
            return SemanticOutput()

        r0 = int(h * (1.0 - cfg.semantic_forward_rows_frac))
        c0 = int(w * (1.0 - cfg.semantic_center_cols_frac) / 2.0)
        c1 = w - c0
        roi_labels = labels[r0:, c0:c1]
        roi_rgb    = rgb_img[r0:, c0:c1] if rgb_img is not None else None

        counts: Dict[int, int] = {}
        unique, cnts = np.unique(roi_labels, return_counts=True)
        for u, c in zip(unique.tolist(), cnts.tolist()):
            counts[u] = c

        total = max(roi_labels.size, 1)
        road_frac   = sum(counts.get(l, 0) for l in self._LABEL_ROAD)      / total
        veg_frac    = sum(counts.get(l, 0) for l in self._LABEL_VEGETATION) / total
        ground_frac = sum(counts.get(l, 0) for l in self._LABEL_GROUND)    / total
        water_frac  = sum(counts.get(l, 0) for l in self._LABEL_WATER)     / total

        if road_frac > 0.40:
            dominant = TerrainClass.ROAD
        elif veg_frac > 0.50:
            dominant = TerrainClass.VEGETATION
        elif water_frac > 0.10:
            dominant = TerrainClass.UNKNOWN
        elif ground_frac > 0.25:
            dominant = self._subclassify_by_color(roi_rgb, roi_labels)
        else:
            dominant = TerrainClass.UNKNOWN

        fwd_r0 = int(h * 0.75)
        fwd_c0 = int(w * 0.35); fwd_c1 = w - fwd_c0
        fwd_labels = labels[fwd_r0:, fwd_c0:fwd_c1]
        fwd_rgb    = rgb_img[fwd_r0:, fwd_c0:fwd_c1] if rgb_img is not None else None
        forward_class = self._subclassify_by_color(fwd_rgb, fwd_labels)

        trav_mult = SEMANTIC_TRAV_MULT.get(dominant, 0.55)
        speed_cap = SEMANTIC_SPEED_CAP.get(dominant, 7.0)
        return SemanticOutput(
            dominant_class=dominant, forward_class=forward_class,
            class_counts=counts, traversability_mult=trav_mult,
            speed_cap=speed_cap, semantic_risk=1.0 - trav_mult,
        )

    def _subclassify_by_color(self, rgb, labels) -> TerrainClass:
        if rgb is None or rgb.size == 0:
            return TerrainClass.DIRT
        pix = rgb.reshape(-1, 3).astype(np.float32)
        if len(pix) == 0:
            return TerrainClass.DIRT
        r = float(pix[:, 0].mean())
        g = float(pix[:, 1].mean())
        b = float(pix[:, 2].mean())
        brightness = (r + g + b) / 3.0
        saturation = max(r, g, b) - min(r, g, b)
        if brightness > 210 and saturation < 25:
            return TerrainClass.SNOW
        if g > r * 1.15 and g > b * 1.05 and saturation > 20:
            return TerrainClass.GRASS
        if saturation < 30 and 50 < brightness < 180:
            return TerrainClass.ROCK
        if saturation < 55 and brightness > 140:
            return TerrainClass.GRAVEL
        if r > g * 1.05 and r > b * 1.15:
            return TerrainClass.DIRT
        return TerrainClass.DIRT


# ╔════════════════════════════════════════════════════════════════════════════╗
# §8  IMU PROCESSOR
# ╚════════════════════════════════════════════════════════════════════════════╝

class IMUProcessor:
    def __init__(self, cfg: Config = CFG):
        self.cfg  = cfg
        self._buf: Deque[Tuple[float, float, float]] = deque(maxlen=cfg.imu_vibration_window)
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
        if len(self._buf) >= 5:
            arr  = np.array(self._buf, dtype=np.float32)
            dev  = arr - arr.mean(axis=0)
            vibration = clamp(float(np.sqrt((dev ** 2).mean())) / 4.0, 0.0, 1.0)
        else:
            vibration = 0.0

        slip_index = clamp(abs(ay) / 9.81 / self.cfg.imu_slip_threshold, 0.0, 1.0)
        pitch_rate = math.degrees(gy)
        roll_rate  = math.degrees(gx)
        stability  = clamp(
            1.0 - 0.45 * vibration
            - 0.35 * slip_index
            - 0.20 * clamp(abs(pitch_rate) / 35.0, 0.0, 1.0),
            0.0, 1.0,
        )
        self._last = IMUData(
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            vibration=vibration, slip_index=slip_index,
            pitch_rate=pitch_rate, roll_rate=roll_rate,
            stability_score=stability,
        )
        return self._last

    @property
    def last(self) -> IMUData:
        return self._last


# ╔════════════════════════════════════════════════════════════════════════════╗
# §9  TERRAIN ANALYZER
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
        z         = local[:, 2]
        z_span    = float(np.percentile(z, 90) - np.percentile(z, 10))
        bump_h    = float(np.percentile(z, 95) - np.percentile(z, 20))
        rough     = clamp(z_span / 3.0, 0.0, 1.0)
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
        if bump_h <= self.cfg.max_step_height:
            clearance_risk = 0.0
        elif bump_h >= self.cfg.vehicle_ground_clearance:
            clearance_risk = 1.0
        else:
            clearance_risk = clamp(
                (bump_h - self.cfg.max_step_height)
                / max(self.cfg.vehicle_ground_clearance - self.cfg.max_step_height, 1e-6),
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
# §10  TERRAIN MEMORY  (dense cost grid)
# ╚════════════════════════════════════════════════════════════════════════════╝

class TerrainMemory:
    _EMA_ALPHA  = 0.15
    _TICK_DECAY = 0.995

    def __init__(self, size=600, resolution=1.0):
        self.res  = float(resolution); self.size = int(size)
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

    def update_from_lidar(self, pts, vehicle_x, vehicle_y, downsample=50,
                          alpha_grid=None, void_risk_map=None):
        if pts is None or len(pts) < downsample:
            return
        sampled  = pts[::downsample]
        world_x  = vehicle_x + sampled[:, 0]
        world_y  = vehicle_y + sampled[:, 1]
        rough_cost = np.clip(np.abs(sampled[:, 2]) * 0.10, 0.0, 1.0)
        void_cost  = None
        if alpha_grid is not None and void_risk_map is not None and void_risk_map.size > 0:
            try:
                alpha_centers = np.asarray(alpha_grid[0], dtype=np.float32)
                local_alpha   = np.arctan2(sampled[:, 1], sampled[:, 0])
                sector_ids    = np.argmin(
                    np.abs(local_alpha[:, None] - alpha_centers[None, :]), axis=1
                )
                void_cost = np.asarray(
                    void_risk_map[void_risk_map.shape[0] // 2, sector_ids], dtype=np.float32
                )
            except Exception:
                void_cost = None
        for i in range(len(sampled)):
            cost = float(rough_cost[i])
            if void_cost is not None:
                cost = float(np.clip(0.55 * cost + 0.45 * void_cost[i], 0.0, 1.0))
            self.update(float(world_x[i]), float(world_y[i]), cost)


class PersistentPathMemory:
    """SQLite-backed experience memory for successful and failed traversals."""

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.dir = Path(cfg.persistent_memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / cfg.persistent_memory_db
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._current_run_id: Optional[int] = None
        self._last_saved_xy: Optional[Tuple[float, float]] = None
        self._sample_seq = 0
        self._success_pts: Optional[np.ndarray] = None
        self._failure_pts: Optional[np.ndarray] = None
        self._success_tree = None
        self._failure_tree = None
        self._cached_seed_nodes: List[TopologicalNode] = []
        self._route_run_id: Optional[int] = None
        self._route_points: Optional[np.ndarray] = None
        self._route_cumdist: Optional[np.ndarray] = None
        self._route_tree = None
        self._route_length: float = 0.0
        self._load_cache()
        self._load_best_route()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                goal_x REAL NOT NULL,
                goal_y REAL NOT NULL,
                note TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                t REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                yaw REAL NOT NULL,
                speed REAL NOT NULL,
                terrain_cost REAL NOT NULL,
                traversability REAL NOT NULL,
                success INTEGER NOT NULL,
                stuck INTEGER NOT NULL,
                mode TEXT NOT NULL,
                semantic TEXT,
                imu_vibration REAL NOT NULL,
                slip REAL NOT NULL,
                goal_dist REAL NOT NULL,
                goal_progress REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topo_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                cost REAL NOT NULL,
                visit_count INTEGER NOT NULL,
                semantic_class INTEGER NOT NULL,
                geometric_cost REAL NOT NULL,
                imu_vibration REAL NOT NULL,
                success_score REAL NOT NULL,
                is_safe INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def begin_run(self, goal_xy: Tuple[float, float], note: str = "live") -> int:
        cur = self._conn.cursor()
        cur.execute("INSERT INTO runs(started_at, goal_x, goal_y, note) VALUES (?, ?, ?, ?)",
                    (time.time(), float(goal_xy[0]), float(goal_xy[1]), note))
        self._conn.commit()
        self._current_run_id = int(cur.lastrowid)
        self._sample_seq = 0
        self._last_saved_xy = None
        return self._current_run_id

    def _load_cache(self) -> None:
        try:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT x, y, cost, visit_count, semantic_class, geometric_cost, imu_vibration, success_score, is_safe, timestamp, source FROM topo_nodes ORDER BY id ASC LIMIT ?",
                (int(self.cfg.persistent_seed_limit),),
            ).fetchall()
            seed_nodes: List[TopologicalNode] = []
            for r in rows:
                seed_nodes.append(
                    TopologicalNode(
                        x=float(r["x"]),
                        y=float(r["y"]),
                        cost=float(r["cost"]),
                        visit_count=int(r["visit_count"]),
                        semantic_class=TerrainClass(int(r["semantic_class"])),
                        geometric_cost=float(r["geometric_cost"]),
                        imu_vibration=float(r["imu_vibration"]),
                        success_score=float(r["success_score"]),
                        is_safe=bool(r["is_safe"]),
                        timestamp=float(r["timestamp"]),
                    )
                )
            self._cached_seed_nodes = seed_nodes
        except Exception:
            self._cached_seed_nodes = []

        if not self._cached_seed_nodes and self._success_pts is not None and len(self._success_pts) > 0:
            derived: List[TopologicalNode] = []
            stride = max(1, len(self._success_pts) // max(1, self.cfg.persistent_seed_limit))
            for row in self._success_pts[::stride]:
                terrain_cost = float(np.clip(row[2], 0.0, 1.0))
                trav = float(np.clip(row[3], 0.0, 1.0))
                derived.append(
                    TopologicalNode(
                        x=float(row[0]),
                        y=float(row[1]),
                        cost=terrain_cost,
                        visit_count=1,
                        semantic_class=TerrainClass.UNKNOWN,
                        geometric_cost=terrain_cost,
                        imu_vibration=0.0,
                        success_score=trav,
                        is_safe=trav > 0.5,
                        timestamp=0.0,
                    )
                )
            self._cached_seed_nodes = derived

        try:
            self._success_pts = self._load_points_from_samples(success=1)
            self._failure_pts = self._load_points_from_samples(stuck=1)
            if self._success_pts is not None and len(self._success_pts) > 0 and SCIPY_OK:
                self._success_tree = KDTree(self._success_pts[:, :2])
            if self._failure_pts is not None and len(self._failure_pts) > 0 and SCIPY_OK:
                self._failure_tree = KDTree(self._failure_pts[:, :2])
        except Exception:
            self._success_pts = None
            self._failure_pts = None
            self._success_tree = None
            self._failure_tree = None

    def _load_points_from_samples(self, success: Optional[int] = None, stuck: Optional[int] = None) -> Optional[np.ndarray]:
        clauses = []
        params: List[int] = []
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))
        if stuck is not None:
            clauses.append("stuck = ?")
            params.append(int(stuck))
        where = " AND ".join(clauses) if clauses else "1=1"
        q = f"SELECT x, y, terrain_cost, traversability, yaw, speed, goal_dist, goal_progress FROM samples WHERE {where} ORDER BY id ASC LIMIT ?"
        params.append(int(self.cfg.persistent_seed_limit))
        rows = self._conn.execute(q, tuple(params)).fetchall()
        if not rows:
            return None
        arr = np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])] for r in rows], dtype=np.float32)
        return arr

    def _load_route_points(self, run_id: int) -> Optional[np.ndarray]:
        rows = self._conn.execute(
            "SELECT x, y FROM samples WHERE run_id = ? ORDER BY seq ASC",
            (int(run_id),),
        ).fetchall()
        if not rows:
            return None
        pts = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=np.float32)
        if len(pts) < 2:
            return None
        return pts

    def _load_best_route(self) -> None:
        self._route_run_id = None
        self._route_points = None
        self._route_cumdist = None
        self._route_tree = None
        self._route_length = 0.0

        try:
            rows = self._conn.execute(
                """
                SELECT run_id, COUNT(*) AS n_samples, MAX(success) AS success_hits,
                       MIN(goal_dist) AS min_goal_dist, MAX(goal_progress) AS max_progress
                FROM samples
                GROUP BY run_id
                ORDER BY n_samples DESC
                """
            ).fetchall()
        except Exception:
            return

        scored: List[Tuple[bool, float, float, float, int, np.ndarray]] = []
        fallback: List[Tuple[bool, float, float, float, int, np.ndarray]] = []

        for r in rows:
            run_id = int(r[0])
            success_hits = int(r[2] or 0)
            min_goal_dist = float(r[3]) if r[3] is not None else float('inf')
            max_progress = float(r[4]) if r[4] is not None else 0.0
            pts = self._load_route_points(run_id)
            if pts is None or len(pts) < 2:
                continue
            seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            length = float(np.sum(seg)) if len(seg) else 0.0
            entry = (success_hits > 0, length, max_progress, -min_goal_dist, run_id, pts)
            fallback.append(entry)
            if success_hits > 0:
                scored.append(entry)

        chosen = scored if scored else fallback
        if not chosen:
            return

        chosen.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        _, length, _, _, run_id, pts = chosen[0]
        self._route_run_id = int(run_id)
        self._route_points = pts.astype(np.float32)
        self._route_length = float(length)
        if len(pts) >= 2:
            seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            self._route_cumdist = np.concatenate(([0.0], np.cumsum(seg))).astype(np.float32)
            if SCIPY_OK:
                self._route_tree = KDTree(self._route_points[:, :2])

    def route_guidance(
        self,
        wx: float,
        wy: float,
        lookahead_m: Optional[float] = None,
        corridor_radius: Optional[float] = None,
    ) -> RouteGuidance:
        if self._route_points is None or self._route_cumdist is None or len(self._route_points) < 2:
            return RouteGuidance(False, None, None)

        radius = float(corridor_radius if corridor_radius is not None else self.cfg.persistent_route_corridor_radius_m)
        lookahead = float(lookahead_m if lookahead_m is not None else self.cfg.persistent_route_lookahead_m)
        lookahead = max(0.0, lookahead)

        if self._route_tree is not None:
            corridor_distance, nearest_idx = self._route_tree.query([wx, wy], k=1)
            corridor_distance = float(corridor_distance)
            nearest_idx = int(nearest_idx)
        else:
            diffs = self._route_points[:, :2] - np.array([wx, wy], dtype=np.float32)
            d2 = np.sum(diffs * diffs, axis=1)
            nearest_idx = int(np.argmin(d2))
            corridor_distance = float(math.sqrt(float(d2[nearest_idx])))

        nearest_idx = int(np.clip(nearest_idx, 0, len(self._route_points) - 1))
        nearest_s = float(self._route_cumdist[nearest_idx])
        target_s = min(nearest_s + lookahead, float(self._route_cumdist[-1]))
        target_idx = int(np.searchsorted(self._route_cumdist, target_s, side="left"))
        target_idx = int(np.clip(target_idx, 0, len(self._route_points) - 1))
        target_xy = self._route_points[target_idx, :2].astype(np.float32).copy()
        nearest_xy = self._route_points[nearest_idx, :2].astype(np.float32).copy()
        remaining_m = max(0.0, float(self._route_length - self._route_cumdist[target_idx]))
        progress_ratio = 0.0 if self._route_length <= 1e-6 else float(self._route_cumdist[target_idx] / self._route_length)
        end_distance = float(np.linalg.norm(self._route_points[-1, :2] - np.array([wx, wy], dtype=np.float32)))
        active = remaining_m > float(self.cfg.persistent_route_end_margin_m)
        if corridor_distance > radius * 6.0 and remaining_m > 0.0:
            # Stay on the route even when far away; only disable if it is basically exhausted.
            active = True

        return RouteGuidance(
            active=active,
            target_xy=target_xy,
            nearest_xy=nearest_xy,
            nearest_idx=nearest_idx,
            target_idx=target_idx,
            route_length=float(self._route_length),
            remaining_m=remaining_m,
            progress_ratio=progress_ratio,
            corridor_distance=corridor_distance,
            end_distance=end_distance,
            run_id=int(self._route_run_id or -1),
        )

    def seed_topology(self, topo_memory: "TopologicalMemory") -> None:
        if not self._cached_seed_nodes:
            return
        topo_memory.ingest_nodes(self._cached_seed_nodes)

    def seed_terrain(self, terrain_memory: TerrainMemory) -> None:
        if self._success_pts is None or len(self._success_pts) == 0:
            return
        for row in self._success_pts[::max(1, len(self._success_pts)//2000)]:
            terrain_memory.update(float(row[0]), float(row[1]), float(np.clip(row[2], 0.0, 1.0)))

    def corridor_bias(self, wx: float, wy: float) -> float:
        if self._success_tree is None and self._failure_tree is None and self._route_tree is None:
            return 0.0

        succ = 0.0
        fail = 0.0
        route_penalty = 0.0
        route_bonus = 0.0

        if self._success_tree is not None:
            d, _ = self._success_tree.query([wx, wy], k=1)
            succ = self.cfg.persistent_success_weight * math.exp(-(float(d) ** 2) / (2.0 * self.cfg.persistent_route_sigma_m ** 2))
        if self._failure_tree is not None:
            d, _ = self._failure_tree.query([wx, wy], k=1)
            fail = self.cfg.persistent_failure_weight * math.exp(-(float(d) ** 2) / (2.0 * self.cfg.persistent_failure_dist_scale ** 2))
        if self._route_tree is not None and self._route_points is not None:
            d, _ = self._route_tree.query([wx, wy], k=1)
            d = float(d)
            route_bonus = self.cfg.persistent_route_follow_weight * math.exp(-(d ** 2) / (2.0 * self.cfg.persistent_route_sigma_m ** 2))
            if d > self.cfg.persistent_route_corridor_radius_m:
                route_penalty = self.cfg.persistent_route_offtrack_weight * ((d - self.cfg.persistent_route_corridor_radius_m) / self.cfg.persistent_route_corridor_radius_m)

        return fail - succ + route_penalty - route_bonus

    def record_sample(self, *, t: float, x: float, y: float, yaw: float, speed: float, terrain_cost: float, traversability: float, goal_dist: float, goal_progress: float, mode: str, semantic: Optional[SemanticOutput], imu: Optional[IMUData], success: bool, stuck: bool) -> None:
        if self._current_run_id is None:
            self.begin_run((x, y), note="auto")
        if self._last_saved_xy is not None:
            if math.hypot(x - self._last_saved_xy[0], y - self._last_saved_xy[1]) < self.cfg.persistent_min_move_m:
                return
        self._last_saved_xy = (x, y)
        self._sample_seq += 1
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO samples(run_id, seq, t, x, y, yaw, speed, terrain_cost, traversability, success, stuck, mode, semantic, imu_vibration, slip, goal_dist, goal_progress)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(self._current_run_id), self._sample_seq, float(t), float(x), float(y), float(yaw), float(speed),
                float(terrain_cost), float(traversability), int(bool(success)), int(bool(stuck)), str(mode),
                semantic.dominant_class.name if semantic else None,
                float(imu.vibration) if imu else 0.0,
                float(imu.slip_index) if imu else 0.0,
                float(goal_dist), float(goal_progress),
            ),
        )

    def record_topological_node(self, node: TopologicalNode) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO topo_nodes(x, y, cost, visit_count, semantic_class, geometric_cost, imu_vibration, success_score, is_safe, timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(node.x), float(node.y), float(node.cost), int(node.visit_count), int(node.semantic_class),
                float(node.geometric_cost), float(node.imu_vibration), float(node.success_score), int(bool(node.is_safe)),
                float(node.timestamp), "live",
            ),
        )

    def record_topological_nodes(self, nodes: List[TopologicalNode]) -> None:
        if not nodes:
            return
        stride = max(1, len(nodes) // max(1, self.cfg.persistent_seed_limit // 4))
        for n in nodes[::stride]:
            self.record_topological_node(n)

    def flush(self) -> None:
        try:
            self._conn.commit()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self._conn.close()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# §11  TOPOLOGICAL MEMORY
# ╚════════════════════════════════════════════════════════════════════════════╝

class TopologicalMemory:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.nodes: List[TopologicalNode] = []
        self._last_node_x: float = float('nan')
        self._last_node_y: float = float('nan')

    def ingest_nodes(self, nodes: List[TopologicalNode]) -> None:
        if not nodes:
            return
        self.nodes.extend(nodes)
        if len(self.nodes) > self.cfg.topo_max_nodes:
            self.nodes = self.nodes[-self.cfg.topo_max_nodes:]

    def export_nodes(self) -> List[TopologicalNode]:
        return list(self.nodes)

    def update(self, x, y, t, speed, imu, semantic, geo_slope, geo_rough):
        for n in self.nodes:
            n.cost = clamp(n.cost * self.cfg.topo_decay_per_tick, 0.0, 1.0)

        dist = (math.hypot(x - self._last_node_x, y - self._last_node_y)
                if not math.isnan(self._last_node_x) else float('inf'))

        if dist < self.cfg.topo_node_spacing:
            if self.nodes:
                self._blend_node(self.nodes[-1], speed, imu, semantic, geo_slope, geo_rough)
            return

        vib      = imu.vibration       if imu      else 0.3
        slip     = imu.slip_index      if imu      else 0.0
        sem_risk = semantic.semantic_risk if semantic else 0.35
        sem_cls  = semantic.dominant_class if semantic else TerrainClass.UNKNOWN

        quality_cost = clamp(
            0.30 * (1.0 - clamp(speed / 8.0, 0.0, 1.0))
            + 0.25 * vib + 0.15 * slip + 0.20 * sem_risk + 0.10 * geo_slope,
            0.0, 1.0,
        )
        node = TopologicalNode(
            x=x, y=y, cost=quality_cost, visit_count=1,
            semantic_class=sem_cls,
            geometric_cost=clamp(0.5 * geo_slope + 0.5 * geo_rough, 0.0, 1.0),
            imu_vibration=vib,
            success_score=clamp(1.0 - quality_cost, 0.0, 1.0),
            is_safe=quality_cost < self.cfg.topo_safe_cost_thresh,
            timestamp=t,
        )
        self.nodes.append(node)
        self._last_node_x = x; self._last_node_y = y

        if len(self.nodes) > self.cfg.topo_max_nodes:
            self.nodes = self.nodes[-self.cfg.topo_max_nodes:]

    def _blend_node(self, n, speed, imu, semantic, geo_slope, geo_rough):
        alpha    = 0.15
        vib      = imu.vibration       if imu      else 0.3
        risk     = semantic.semantic_risk if semantic else 0.35
        new_cost = clamp(
            0.30 * (1.0 - clamp(speed / 8.0, 0.0, 1.0))
            + 0.25 * vib + 0.20 * risk + 0.10 * geo_slope, 0.0, 1.0,
        )
        n.cost = (1 - alpha) * n.cost + alpha * new_cost
        n.success_score = clamp((1.0 - n.cost) * 0.6 + n.success_score * 0.4, 0.0, 1.0)
        n.is_safe = n.cost < self.cfg.topo_safe_cost_thresh
        n.visit_count += 1

    def corridor_bias(self, wx, wy) -> float:
        r, r2 = self.cfg.topo_corridor_radius, self.cfg.topo_corridor_radius ** 2
        best_safe = worst_bad = 0.0
        for n in self.nodes:
            d2 = (n.x - wx) ** 2 + (n.y - wy) ** 2
            if d2 > r2: continue
            prox = 1.0 - math.sqrt(d2) / r
            if n.is_safe:
                best_safe = max(best_safe, prox * max(0.05, n.success_score))
            else:
                worst_bad = max(worst_bad, prox * n.cost)
        return worst_bad * 4.0 - best_safe * self.cfg.topo_corridor_bonus

    def safe_node_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_safe)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §12  VSGP SUBGOAL PLANNER
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

    def plan(self, perc, vehicle_pos, vehicle_yaw_rad, goal_pos,
             terrain_memory=None, topo_memory=None, semantic=None,
             persistent_memory=None,
             tcn_nav_world: Optional[np.ndarray] = None,   # ◄ NEW param
             tcn_corridor_bias: float = 20.0,              # ◄ NEW param
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        if perc.occupancy_mean is None:
            return None, []

        curv = getattr(perc, "curvature_map", None)
        if curv is None:
            curv = np.zeros_like(perc.traversability)
        void_map = getattr(perc, "void_risk_map", None)
        if void_map is None:
            void_map = np.zeros_like(perc.traversability)


        dx_world    = goal_pos[0] - vehicle_pos[0]
        dy_world    = goal_pos[1] - vehicle_pos[1]
        goal_yaw    = math.atan2(dy_world, dx_world)
        goal_rel_y  = self._norm_angle(goal_yaw - vehicle_yaw_rad)

        alphas       = perc.alpha_grid[0]
        betas        = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        trav  = perc.fused_traversability if perc.fused_traversability is not None else perc.traversability
        occ   = perc.occupancy_mean
        var   = perc.occupancy_var
        slope = perc.slope_map
        rough = perc.roughness_map

        half_fov    = math.pi / 2.2
        scan_centre = goal_rel_y if abs(goal_rel_y) < half_fov else 0.0
        alpha_range = np.linspace(scan_centre - half_fov, scan_centre + half_fov, self.n_angles)

        sem_speed_cap = semantic.speed_cap if semantic else 15.0
        dyn_max_dist  = min(self.max_dist, max(self.min_dist + 2, sem_speed_cap * 1.5))
        candidates: List[Subgoal] = []

        for a in alpha_range:
            a_idx = int(np.argmin(np.abs(alphas - a)))
            if not (0 <= a_idx < len(alphas)):
                continue

            depths       = np.linspace(self.min_dist, dyn_max_dist, self.n_depth)
            valid_points = []

            for d in depths:
                o_v  = float(occ [mid_beta_idx, a_idx])
                v_v  = float(var [mid_beta_idx, a_idx])
                s_v  = float(slope[mid_beta_idx, a_idx])
                r_v  = float(rough[mid_beta_idx, a_idx])
                t_v  = float(trav [mid_beta_idx, a_idx])
                c_v  = float(curv [mid_beta_idx, a_idx])
                vd_v = float(void_map[mid_beta_idx, a_idx])
                if (o_v < self.OCC_FREE_THRESH and v_v < self.VAR_STABLE_THRESH
                        and s_v < self.SLOPE_SAFE_THRESH and r_v < self.ROUGH_SAFE_THRESH
                        and c_v < self.CURV_SAFE_THRESH  and vd_v < self.VOID_SAFE_THRESH):
                    valid_points.append((d, o_v, v_v, s_v, r_v, t_v, c_v, vd_v))

            if not valid_points:
                continue

            bd, bo, bv, bs, br, bt, bc, bvd = valid_points[-1]
            lx = bd * math.cos(a); ly = bd * math.sin(a)
            wx = vehicle_pos[0] + math.cos(vehicle_yaw_rad) * lx - math.sin(vehicle_yaw_rad) * ly
            wy = vehicle_pos[1] + math.sin(vehicle_yaw_rad) * lx + math.cos(vehicle_yaw_rad) * ly

            dist_c = math.hypot(goal_pos[0] - wx, goal_pos[1] - wy)
            dist_v = math.hypot(goal_pos[0] - vehicle_pos[0], goal_pos[1] - vehicle_pos[1])
            progress    = dist_v - dist_c
            heading_err = abs(self._norm_angle(
                math.atan2(goal_pos[1] - wy, goal_pos[0] - wx) - vehicle_yaw_rad
            ))

            terrain_cost = (
                self.W_SLOPE * bs + self.W_ROUGH * br + self.W_OCC * bo
                + self.W_TERRAIN * (1.0 - bt) + 2.5 * bc + self.W_VOID * bvd
            )
            sem_cost    = self.cfg.w_semantic * semantic.semantic_risk if semantic else 0.0
            mem_cost    = lookahead_c = topo_cost = 0.0
            if terrain_memory:
                mem_cost   = terrain_memory.get_cost(wx, wy)
                cand_hdg   = math.atan2(wy - vehicle_pos[1], wx - vehicle_pos[0])
                lookahead_c = terrain_memory.lookahead_cost(
                    wx, wy, cand_hdg,
                    steps=self.cfg.terrain_memory_lookahead_steps,
                    step_m=1.0, weight=self.cfg.terrain_memory_lookahead_weight,
                )
            if topo_memory:
                topo_cost = topo_memory.corridor_bias(wx, wy)
            persistent_cost = 0.0
            if persistent_memory is not None:
                persistent_cost = persistent_memory.corridor_bias(wx, wy)

            cost = (
                terrain_cost
                - self.W_GOAL_DIST * max(0.0, progress)
                + self.W_HEADING   * heading_err
                + self.cfg.terrain_memory_cost_weight * mem_cost
                + lookahead_c + sem_cost + topo_cost + persistent_cost
            )

            sg = Subgoal(
                alpha=a, beta=0.0, distance=bd,
                local_pos=np.array([lx, ly, 0.0], dtype=np.float32),
                world_pos=np.array([wx, wy, 0.0], dtype=np.float32),
                slope=bs, roughness=br, occupancy=bo, variance=bv, traversability=bt,
                terrain_cost=terrain_cost, goal_progress=progress,
                heading_error=heading_err, safe=True,
                width_m=self.cfg.wheel_base, cost=cost,
            )
            candidates.append(sg)

        # ── TCN corridor bias: reduce cost for candidates inside the corridor ◄ NEW
        if tcn_nav_world is not None and candidates:
            tcn_pos = tcn_nav_world[:2].astype(np.float32)
            for sg in candidates:
                d_corr = float(np.linalg.norm(sg.world_pos[:2] - tcn_pos))
                if d_corr < self.cfg.topo_corridor_radius:
                    sg.cost -= tcn_corridor_bias * (1.0 - d_corr / self.cfg.topo_corridor_radius)

        if candidates:
            candidates.sort(key=lambda s: s.cost)
            return candidates[0], candidates

        # ── Relaxed fallback ─────────────────────────────────────────────────
        print("[SUBGOAL] No constrained candidates – relaxed fallback")
        dist_v = math.hypot(goal_pos[0] - vehicle_pos[0], goal_pos[1] - vehicle_pos[1])
        for a in alpha_range:
            d = self.min_dist + 2.0
            lx = d * math.cos(a); ly = d * math.sin(a)
            wx = vehicle_pos[0] + math.cos(vehicle_yaw_rad) * lx - math.sin(vehicle_yaw_rad) * ly
            wy = vehicle_pos[1] + math.sin(vehicle_yaw_rad) * lx + math.cos(vehicle_yaw_rad) * ly
            dist_c   = math.hypot(goal_pos[0] - wx, goal_pos[1] - wy)
            progress = dist_v - dist_c
            h_err    = abs(self._norm_angle(
                math.atan2(goal_pos[1] - wy, goal_pos[0] - wx) - vehicle_yaw_rad
            ))
            cost = (self.W_GOAL_DIST * dist_c + self.W_HEADING * h_err
                    - self.W_GOAL_DIST * max(0.0, progress))
            if persistent_memory is not None:
                cost += persistent_memory.corridor_bias(wx, wy)
            candidates.append(Subgoal(
                alpha=a, beta=0.0, distance=d,
                local_pos=np.array([lx, ly, 0.0], dtype=np.float32),
                world_pos=np.array([wx, wy, 0.0], dtype=np.float32),
                slope=0.5, roughness=0.5, occupancy=0.5, variance=0.5, traversability=0.3,
                terrain_cost=5.0, goal_progress=progress, heading_error=h_err,
                safe=False, width_m=self.cfg.wheel_base, cost=cost,
            ))

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
# §13  STUCK MEMORY
# ╚════════════════════════════════════════════════════════════════════════════╝

class StuckMemory:
    def __init__(self, cfg: Config):
        self.decay     = 0.995
        self.radius    = cfg.memory_radius
        self.threshold = 0.05
        self.points: list = []

    def update(self, pos: np.ndarray) -> None:
        self.points = [
            (x, y, w * self.decay, t)
            for x, y, w, t in self.points
            if w * self.decay > self.threshold
        ]

    def get_tensor(self, device):
        if not self.points:
            return None
        type_map = {"stall": 0.0, "slip": 1.0, "obstacle": 2.0}
        data = [[p[0], p[1], p[2], type_map.get(p[3], 0.0)] for p in self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §14  LEARNED COST NETWORK
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
# §15  MPPI PLANNER
# ╚════════════════════════════════════════════════════════════════════════════╝

class MPPIPlanner:
    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon     = cfg.mppi_horizon
        self.num_samples = cfg.mppi_num_samples
        self.lambda_     = cfg.mppi_lambda
        self.dt          = cfg.fixed_delta_seconds
        self._u_nom      = torch.zeros((self.horizon, 2), device=self.device)
        self._noise      = torch.zeros((self.num_samples, self.horizon, 2), device=self.device)
        self.cost_net    = CostNet().to(self.device)
        self.cost_net.eval()
        self._semantic_risk: float = 0.0
        self._speed_cap:     float = 15.0
        self._imu_vibration: float = 0.0
        self._imu_slip:      float = 0.0
        self._prev_steer:    float = 0.0

    def set_semantic_imu(self, semantic, imu):
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

        trav_arr = (perc.fused_traversability
                    if perc.fused_traversability is not None else perc.traversability)
        occ      = torch.tensor(perc.occupancy_mean, device=device).float()
        slope    = torch.tensor(perc.slope_map,      device=device).float()
        rough    = torch.tensor(perc.roughness_map,  device=device).float()
        var      = torch.tensor(perc.occupancy_var,  device=device).float()
        trav     = torch.tensor(trav_arr,            device=device).float()
        curv     = torch.tensor(perc.curvature_map,  device=device).float()
        void_map = torch.tensor(perc.void_risk_map,  device=device).float()

        sem_risk_t  = torch.tensor(self._semantic_risk, device=device).float()
        imu_vib_t   = torch.tensor(self._imu_vibration, device=device).float()
        imu_slip_t  = torch.tensor(self._imu_slip,      device=device).float()
        speed_cap_t = torch.tensor(self._speed_cap,     device=device).float()

        costs = self._rollout(state, u, occ, slope, rough, var, trav, curv, void_map,
                              sem_risk_t, imu_vib_t, imu_slip_t, speed_cap_t)

        beta    = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum()
        self._u_nom = (weights[:, None, None] * u).sum(dim=0).detach()

        u0            = self._u_nom[0]
        raw_steer     = float(u0[1].cpu())
        smoothed_steer = 0.7 * self._prev_steer + 0.3 * raw_steer
        self._prev_steer = smoothed_steer

        return ControlState(
            throttle=float(u0[0].cpu()),
            steer=smoothed_steer,
            brake=0.0, reverse=False,
        ), float(beta.cpu())

    def _rollout(self, state, u, occ, slope, rough, var, trav, curv, void_map,
                 sem_risk, imu_vib, imu_slip, speed_cap):
        K, T, _ = u.shape
        device   = self.device
        n_a, n_b = occ.shape[1], occ.shape[0]
        a_max    = np.pi / 2

        x = torch.tensor(state, device=device).float().unsqueeze(0).repeat(K, 1)
        veh_x, veh_y, veh_yaw = x[0, 0].item(), x[0, 1].item(), x[0, 2].item()
        total   = torch.zeros(K, device=device)
        prev_gd = torch.sqrt((self.goal[0] - x[:, 0]) ** 2 + (self.goal[1] - x[:, 1]) ** 2)

        for t in range(T):
            throttle = u[:, t, 0]; steer = u[:, t, 1]
            dx = x[:, 0] - veh_x; dy = x[:, 1] - veh_y
            rel_yaw = torch.atan2(
                torch.sin(torch.atan2(dy, dx) - veh_yaw),
                torch.cos(torch.atan2(dy, dx) - veh_yaw),
            )
            ai = ((rel_yaw + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi = torch.full_like(ai, n_b // 2)
            speed_scale = torch.exp(-3.0 * (slope[bi, ai] + rough[bi, ai])).clamp(0.15, 1.0)
            v   = (x[:, 3] + throttle * speed_scale * self.dt).clamp(0.0, speed_cap)
            yaw = x[:, 2] + (v / self.cfg.wheel_base) * torch.tan(steer.clamp(-0.99, 0.99)) * self.dt
            xp  = x[:, 0] + v * torch.cos(yaw) * self.dt
            yp  = x[:, 1] + v * torch.sin(yaw) * self.dt
            x   = torch.stack([xp, yp, yaw, v], dim=1)

            dx2 = xp - veh_x; dy2 = yp - veh_y
            rel_yaw2 = torch.atan2(
                torch.sin(torch.atan2(dy2, dx2) - veh_yaw),
                torch.cos(torch.atan2(dy2, dx2) - veh_yaw),
            )
            ai2 = ((rel_yaw2 + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi2 = torch.full_like(ai2, n_b // 2)

            gdx = self.goal[0] - xp; gdy = self.goal[1] - yp
            curr_gd = torch.sqrt(gdx ** 2 + gdy ** 2)
            goal_h  = torch.atan2(gdy, gdx)
            h_err   = torch.atan2(torch.sin(goal_h - yaw), torch.cos(goal_h - yaw)).abs()

            progress  = prev_gd - curr_gd; prev_gd = curr_gd
            C_prog    = -25.0 * progress
            terrain_c = (
                5.0  * occ[bi2, ai2] + 3.5 * slope[bi2, ai2]
                + 3.0 * rough[bi2, ai2] + 8.0 * curv[bi2, ai2]
                + 120.0 * void_map[bi2, ai2] + 10.0 * var[bi2, ai2]
                + 12.0 * (1.0 - trav[bi2, ai2]) ** 2
            )
            ctrl_c = 0.1 * (throttle ** 2 + steer ** 2)
            if t > 0:
                jerk = (u[:, t, :] - u[:, t-1, :]).abs().sum(dim=1)
                total += 2.5 * jerk

            total += (terrain_c + self.cfg.w_heading * h_err + ctrl_c
                      + self.cfg.w_semantic * sem_risk
                      + self.cfg.w_imu * (0.6 * imu_vib + 0.4 * imu_slip)
                      + torch.relu(v - speed_cap) * 5.0
                      + C_prog)
        return total


# ╔════════════════════════════════════════════════════════════════════════════╗
# §16  REACTIVE OBSTACLE AVOIDANCE
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
        x, y, z = raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2]
        dist_2d = np.sqrt(x ** 2 + y ** 2)
        angle   = np.degrees(np.arctan2(y, x))
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
            ratio    = (min_dist - self.cfg.react_emergency_dist) / (
                self.cfg.react_danger_dist - self.cfg.react_emergency_dist)
            out.throttle = ctrl.throttle * ratio * 0.2
            out.brake    = self.cfg.max_brake * (1.0 - ratio * 0.4)
            obs_y    = float(fwd_pts[closest_idx, 1])
            steer_dir = -np.sign(obs_y) if abs(obs_y) > 0.1 else -np.sign(ctrl.steer + 1e-6)
            steer_mag = min(self.cfg.max_steer, 0.4 / (min_dist + 0.1))
            out.steer = float(np.clip(ctrl.steer + steer_dir * steer_mag,
                                      -self.cfg.max_steer, self.cfg.max_steer))
        else:
            ratio    = (min_dist - self.cfg.react_danger_dist) / (
                self.cfg.react_warn_dist - self.cfg.react_danger_dist)
            out.throttle = ctrl.throttle * (0.3 + ratio * 0.7)
            out.brake    = ctrl.brake + (1.0 - ratio) * 0.25
        if min_dist < 2.5 and not out.reverse:
            out.reverse = True; out.throttle = 0.3
        return out


# ╔════════════════════════════════════════════════════════════════════════════╗
# §17  CONTROLLER
# ╚════════════════════════════════════════════════════════════════════════════╝

class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg    = cfg
        self._state = ControlState()
        self._spd_ema = self._str_ema = 0.0

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
        st = float(np.clip(st + cs, -max_steer_at_speed, max_steer_at_speed))
        st -= 0.2 * (st - self._state.steer)
        th -= 0.1 * (th - self._state.throttle)
        th = float(np.clip(th, 0.0, self.cfg.max_throttle))
        st = float(np.clip(st, -self.cfg.max_steer, self.cfg.max_steer))
        self._state = ControlState(throttle=th, steer=st, brake=br, reverse=ctrl.reverse)
        return self._state

    def _stability(self, roll, pitch):
        ts = 1.0; cs = 0.0; ar = abs(roll)
        wr, dr = self.cfg.anti_tip_roll_warn, self.cfg.anti_tip_roll_danger
        if ar > wr:
            ratio = clamp((ar - wr) / max(dr - wr, 1.0), 0.0, 1.0)
            ts = min(ts, 1.0 - ratio * (1.0 - self.cfg.stability_throttle_scale))
            cs = -math.copysign(
                min(self.cfg.roll_corrective_steer * ratio, self.cfg.roll_corrective_steer), roll
            )
        ap = abs(pitch)
        wp, dp = self.cfg.anti_tip_pitch_warn, self.cfg.anti_tip_pitch_danger
        if ap > wp:
            ratio = clamp((ap - wp) / max(dp - wp, 1.0), 0.0, 1.0)
            ts = min(ts, 1.0 - ratio * (1.0 - self.cfg.stability_throttle_scale * 0.5))
        return float(ts), float(cs)

    def reset(self):
        self._spd_ema = self._str_ema = 0.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# §18  FLAT GOAL PLANNER
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
        goal_x, goal_y = goal_xy
        cur_gd     = math.hypot(goal_x - st.x, goal_y - st.y)
        goal_h     = math.atan2(goal_y - st.y, goal_x - st.x)
        rel_goal   = angle_wrap(goal_h - st.yaw)
        center     = clamp(rel_goal, -math.radians(80), math.radians(80))
        half       = math.radians(self.fov_deg) / 2.0
        angles     = np.linspace(center - half, center + half, self.candidate_count)
        candidates: List[Candidate] = []

        for dist in self.candidate_distances:
            for alpha in angles:
                lx = dist * math.cos(alpha); ly = dist * math.sin(alpha)
                if lx < 1.0: continue
                wx, wy     = local_to_world(lx, ly, st.yaw, st.x, st.y)
                new_gd     = math.hypot(goal_x - wx, goal_y - wy)
                progress   = cur_gd - new_gd
                h_err      = abs(angle_wrap(math.atan2(wy - st.y, wx - st.x) - st.yaw))
                slope, rough, obs, flat, clear = self.terrain.local_risk(lx, ly, radius=5.0)
                mem = self.memory_penalty(wx, wy)
                cost = (self.w_goal * new_gd - self.w_progress * progress
                        + self.w_heading * h_err + self.w_slope * slope
                        + self.w_rough * rough + self.w_obstacle * obs
                        + self.w_clearance * clear + self.w_memory * mem)
                reward = (120.0 * progress + 45.0 * flat - 25.0 * obs
                          - 12.0 * slope - 8.0 * rough - 35.0 * clear - 40.0 * mem)
                candidates.append(Candidate(
                    wx, wy, lx, ly, alpha, dist, progress, new_gd, h_err,
                    slope, rough, obs, flat, clear, mem, cost, reward,
                ))

        if not candidates:
            dist = min(25.0, cur_gd)
            lx = dist * math.cos(rel_goal); ly = dist * math.sin(rel_goal)
            wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
            slope, rough, obs, flat, clear = self.terrain.local_risk(lx, ly, radius=5.0)
            self.best = Candidate(wx, wy, lx, ly, rel_goal, dist, 1.0, cur_gd,
                                  abs(rel_goal), slope, rough, obs, flat, clear,
                                  0.0, cur_gd, 0.0)
            return self.best

        candidates.sort(key=lambda c: c.cost)
        best = candidates[0]
        if max(c.progress for c in candidates) < 0.2:
            candidates.sort(key=lambda c: (abs(c.heading_error),
                                           c.goal_dist + 30.0 * c.obstacle_risk))
            best = candidates[0]
        self.best = best; return best


# ╔════════════════════════════════════════════════════════════════════════════╗
# §19  MANUAL CONTROLLER
# ╚════════════════════════════════════════════════════════════════════════════╝

class ManualController:
    THROTTLE_STEP = 0.04; STEER_STEP = 0.06; STEER_DECAY = 0.80

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg; self._throttle = self._steer = 0.0; self._reverse = False

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
            self._throttle = 0.0; self._steer *= 0.5; self._reverse = False
            brake = self.cfg.max_brake
        if abs(roll_deg) > self.cfg.max_safe_roll_deg:
            corr = -math.copysign(
                min(0.15 * (abs(roll_deg) - self.cfg.max_safe_roll_deg) / 10.0, 0.15), roll_deg
            )
            self._steer = float(np.clip(self._steer + corr, -self.cfg.max_steer, self.cfg.max_steer))
        return ControlState(round(self._throttle, 4), round(self._steer, 4), brake, self._reverse)

    def reset(self):
        self._throttle = self._steer = 0.0; self._reverse = False


# ╔════════════════════════════════════════════════════════════════════════════╗
# §20  SENSOR MANAGER
# ╚════════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config):
        self.world   = world; self.vehicle = vehicle; self.cfg = cfg
        self.actors: List = []
        self.lidar_q    = Queue(maxsize=2); self.front_q    = Queue(maxsize=2)
        self.rear_q     = Queue(maxsize=2); self.semantic_q = Queue(maxsize=2)
        self.imu_q      = Queue(maxsize=4)
        self._has_imu = self._has_semantic = False
        self._setup()

    @staticmethod
    def _put_latest(q, data):
        while q.full():
            try: q.get_nowait()
            except Empty: break
        q.put(data)

    @staticmethod
    def _img_to_array(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))[:, :, :3]
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

        lidar_bp = bp.find("sensor.lidar.ray_cast")
        for k, v in [("range",              str(self.cfg.lidar_range)),
                     ("channels",           str(self.cfg.lidar_channels)),
                     ("points_per_second",  str(self.cfg.lidar_points_per_sec)),
                     ("rotation_frequency", str(self.cfg.lidar_rotation_freq)),
                     ("upper_fov",          str(self.cfg.lidar_upper_fov)),
                     ("lower_fov",          str(self.cfg.lidar_lower_fov))]:
            lidar_bp.set_attribute(k, v)
        lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.7, z=2.25)),
            attach_to=self.vehicle)
        lidar.listen(lambda d: self._put_latest(self.lidar_q, d))
        self.actors.append(lidar)

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov",          str(self.cfg.camera_fov))
        for tf_args, q in [
            (carla.Transform(carla.Location(x=1.7, z=1.9)), self.front_q),
            (carla.Transform(carla.Location(x=-1.7, z=1.9), carla.Rotation(yaw=180)), self.rear_q),
        ]:
            c = self.world.spawn_actor(cam_bp, tf_args, attach_to=self.vehicle)
            c.listen(lambda img, _q=q: self._put_latest(_q, self._img_to_array(img)))
            self.actors.append(c)

        try:
            seg_bp = bp.find("sensor.camera.semantic_segmentation")
            seg_bp.set_attribute("image_size_x", str(self.cfg.semantic_camera_width))
            seg_bp.set_attribute("image_size_y", str(self.cfg.semantic_camera_height))
            seg_bp.set_attribute("fov",          str(self.cfg.camera_fov))
            seg = self.world.spawn_actor(
                seg_bp, carla.Transform(carla.Location(x=1.7, z=1.9)),
                attach_to=self.vehicle)
            seg.listen(lambda img: self._put_latest(
                self.semantic_q,
                np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                    (img.height, img.width, 4)).copy()))
            self.actors.append(seg)
            self._has_semantic = True
            print("[SENSOR] Semantic camera online")
        except Exception as e:
            print(f"[SENSOR] Semantic camera unavailable: {e}")

        try:
            imu_bp = bp.find("sensor.other.imu")
            for k in ["noise_accel_stddev_x","noise_accel_stddev_y","noise_accel_stddev_z"]:
                imu_bp.set_attribute(k, "0.01")
            for k in ["noise_gyro_stddev_x","noise_gyro_stddev_y","noise_gyro_stddev_z"]:
                imu_bp.set_attribute(k, "0.001")
            imu = self.world.spawn_actor(
                imu_bp, carla.Transform(carla.Location(x=0.0, z=0.5)),
                attach_to=self.vehicle)
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

    def get_front_image(self):  return self._get_latest(self.front_q)
    def get_rear_image(self):   return self._get_latest(self.rear_q)
    def get_semantic_image(self): return self._get_latest(self.semantic_q)
    def get_imu_data(self):     return self._get_latest(self.imu_q)

    @property
    def has_imu(self):      return self._has_imu
    @property
    def has_semantic(self): return self._has_semantic

    def destroy(self):
        for a in self.actors:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# §21  RESULT LOGGER
# ╚════════════════════════════════════════════════════════════════════════════╝

class ResultLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.rows: List[dict] = []
        self.terrain_rows: List[dict] = []

    def log(self, t, st, cmd, cand, goal_dist, mode, cost, reward,
            semantic_class="UNKNOWN", imu_vibration=0.0, safe_topo_nodes=0,
            tcn_active=False) -> None:                          # ◄ NEW: tcn_active
        aw = (st.speed / max(self.cfg.wheel_base, 1e-6)) * math.tan(cmd.steer)
        swx = getattr(cand, 'wx', None)
        if swx is None:
            wp = getattr(cand, 'world_pos', None)
            swx = float(wp[0]) if wp is not None and len(wp) >= 2 else st.x
        swy = getattr(cand, 'wy', None)
        if swy is None:
            wp = getattr(cand, 'world_pos', None)
            swy = float(wp[1]) if wp is not None and len(wp) >= 2 else st.y
        self.rows.append({
            "time": t, "x": st.x, "y": st.y, "z": st.z,
            "yaw": st.yaw, "pitch_deg": st.pitch_deg, "roll_deg": st.roll_deg,
            "speed": st.speed, "linear_velocity": st.speed, "angular_velocity": aw,
            "throttle": cmd.throttle, "steer": cmd.steer,
            "brake": cmd.brake, "reverse": int(cmd.reverse),
            "goal_distance": goal_dist, "mode": mode,
            "cost": cost, "reward": reward,
            "subgoal_x": swx, "subgoal_y": swy,
            "flatness":       getattr(cand, 'flatness',      getattr(cand, 'traversability', 0.0)),
            "slope":          getattr(cand, 'slope',         0.0),
            "roughness":      getattr(cand, 'roughness',     0.0),
            "progress":       getattr(cand, 'progress',      getattr(cand, 'goal_progress', 0.0)),
            "obstacle_risk":  getattr(cand, 'obstacle_risk', getattr(cand, 'occupancy', 0.0)),
            "clearance_risk": getattr(cand, 'clearance_risk', 0.0),
            "semantic_class": semantic_class, "imu_vibration": imu_vibration,
            "safe_topo_nodes": safe_topo_nodes,
            "tcn_active": int(tcn_active),               # ◄ NEW column
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
            ("semantics_imu.csv",["time","semantic_class","imu_vibration","safe_topo_nodes","speed","tcn_active"]),
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
        plt.axis("equal"); plt.grid(True, alpha=0.3); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("World Trajectory"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, "trajectory.svg"), format="svg"); plt.close()

        plt.figure(figsize=(7, 7))
        sc = plt.scatter(x, y, c=speed, s=18, cmap="RdYlGn")
        plt.colorbar(sc, label="Speed [m/s]")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=120, c="red")
        plt.axis("equal"); plt.grid(True, alpha=0.3); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("Speed Heat Map"); plt.tight_layout()
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
            plt.axis("equal"); plt.grid(True, alpha=0.3); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
            plt.title("Terrain Height Map"); plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.out_dir, "flatness_heatmap.svg"), format="svg")
            plt.close()

    def finalize(self):
        print(f"\n[SAVE] Saving in {os.path.abspath(self.cfg.out_dir)}")
        self.save_csv(); self.save_plots(); print("[SAVE] Done")


# ╔════════════════════════════════════════════════════════════════════════════╗
# §22  VISUALIZATION MODULE
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
        'tcn': (180, 80, 255),   # ◄ NEW: purple for TCN corridor point
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
        self._costs:   deque = deque(maxlen=400); self._speeds: deque = deque(maxlen=400)
        self._rewards: deque = deque(maxlen=400)
        self._front_img = self._rear_img = self._lidar_pts = None
        self._goal = GOAL; self._subgoal = None
        self._subgoal_candidates: List = []; self._mem_pts: list = []
        self._tcn_nav: Optional[np.ndarray] = None  # ◄ NEW
        self._mode = "AUTO"; self._speed = self._steer = 0.0
        self._cost = self._reward = self._dist_goal = 0.0
        self._obs_warn = False; self._pitch = self._roll = 0.0
        self._flatness = self._progress = 0.0
        self._semantic_class = TerrainClass.UNKNOWN
        self._imu_vibration = self._imu_slip = self._stability = 0.0
        self._stability = 1.0
        self._safety_active = False; self._safety_reason = ""
        self._topo_nodes: List[TopologicalNode] = []
        self._tcn_active = False   # ◄ NEW

    def init(self, screen):
        self._screen = screen; pygame.font.init()
        self._font_s = pygame.font.SysFont("monospace", 11)

    def push(self, *, pos, speed, steer, perc, mode, cost, reward,
             subgoal, all_candidates, mem_pts, dist_goal, obs_warn,
             flatness=0.0, progress=0.0, pitch=0.0, roll=0.0,
             semantic=None, imu=None, safety_active=False, safety_reason="",
             topo_nodes=None, tcn_nav=None, tcn_active=False):   # ◄ NEW params
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
        self._tcn_nav    = tcn_nav     # ◄ NEW
        self._tcn_active = tcn_active  # ◄ NEW
        if perc and perc.raw_points is not None and len(perc.raw_points):
            self._lidar_pts = perc.raw_points.copy()

    def set_front(self, img): self._front_img = img
    def set_rear(self, img):  self._rear_img  = img

    def render(self):
        if self._screen is None: return
        W, H  = self._screen.get_size()
        CAM_H = 240; col_w = W // 3; bot_y = CAM_H; bot_h = H - bot_y; map_w = W // 2
        self._screen.fill(self.C['bg'])
        self._draw_camera(self._front_img, "Front camera", 0,       0, col_w, CAM_H)
        self._draw_camera(self._rear_img,  "Rear camera",  col_w,   0, col_w, CAM_H)
        self._draw_lidar_3d("LiDAR view",                  col_w*2, 0, W-col_w*2, CAM_H)
        self._draw_trajectory(0,    bot_y, map_w,     bot_h)
        self._draw_cost_plot(map_w, bot_y, W - map_w, bot_h // 2)
        self._draw_speed_plot(map_w, bot_y + bot_h // 2, W - map_w, bot_h - bot_h // 2)
        self._draw_hud(W, H)
        pygame.display.flip()

    def _draw_camera(self, img, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x+4,y+3))
        if img is not None:
            try:
                img_r = self._resize(img, w, h - 18)
                surf  = pygame.surfarray.make_surface(img_r.swapaxes(0, 1))
                self._screen.blit(surf, (x, y + 18))
            except Exception: pass
        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_lidar_3d(self, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x+4,y+3))
        cy0 = y+18; ch = h-18; lr = self.cfg.lidar_range
        cx = x + w // 2; cy = cy0 + ch // 2; scale = (min(w, ch) * 0.44) / lr
        for rr in [lr*0.33, lr*0.67, lr]:
            pygame.draw.circle(self._screen, C['grid'], (cx, cy), int(rr*scale), 1)
        pygame.draw.line(self._screen, C['grid'], (cx, cy), (cx, cy-int(lr*scale)), 1)
        if self._lidar_pts is not None and len(self._lidar_pts):
            pts = self._lidar_pts
            z_min = pts[:,2].min(); z_max = pts[:,2].max()
            zn = (pts[:,2] - z_min) / max(z_max - z_min, 1e-6)
            for i in range(0, len(pts), 3):
                px = int(cx - pts[i,1]*scale); py = int(cy - pts[i,0]*scale)
                if x <= px < x+w and cy0 <= py < cy0+ch:
                    t2 = float(zn[i])
                    if   t2 < 0.2: r2,g2,b2 = 255, int(255*(t2/0.2)), 0
                    elif t2 < 0.4: r2,g2,b2 = int(255*((0.4-t2)/0.2)), 255, 0
                    elif t2 < 0.6: r2,g2,b2 = 0, 255, int(255*((t2-0.4)/0.2))
                    elif t2 < 0.8: r2,g2,b2 = 0, int(255*((0.8-t2)/0.2)), 255
                    else:          r2,g2,b2 = int(255*((t2-0.8)/0.2)), 0, 255
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
        if self._tcn_active:
            header_txt += "  [TCN]"
        self._screen.blit(self._font_s.render(header_txt, True, sem_col), (x+4,y+3))

        tx = list(self._traj_x); ty = list(self._traj_y)
        if not tx:
            pygame.draw.rect(self._screen, C['border'], (x,y,w,h),1); return

        ax = tx + [self._goal[0]]; ay = ty + [self._goal[1]]
        for px2,py2,pw2,_ in self._mem_pts:
            ax.append(px2); ay.append(py2)
        if self._subgoal is not None:
            ax.append(float(self._subgoal[0])); ay.append(float(self._subgoal[1]))
        if self._tcn_nav is not None:
            ax.append(float(self._tcn_nav[0])); ay.append(float(self._tcn_nav[1]))
        for n in self._topo_nodes:
            ax.append(n.x); ay.append(n.y)

        span = max(max(ax)-min(ax), max(ay)-min(ay), 20.0); pad = span*0.12
        mn_x = min(ax)-pad; mx_x = max(ax)+pad; mn_y = min(ay)-pad; mx_y = max(ay)+pad
        px0 = x+6; py0 = y+22; pw = w-12; ph = h-26

        def w2s(wx2, wy2):
            sx = px0 + int((wx2-mn_x)/(mx_x-mn_x)*pw)
            sy = py0 + int((1.0-(wy2-mn_y)/(mx_y-mn_y))*ph)
            return sx, sy

        for gx2 in np.linspace(mn_x, mx_x, 5):
            sx,_ = w2s(gx2, mn_y); pygame.draw.line(self._screen, C['grid'], (sx,py0), (sx,py0+ph), 1)
        for gy2 in np.linspace(mn_y, mx_y, 5):
            _,sy = w2s(mn_x, gy2); pygame.draw.line(self._screen, C['grid'], (px0,sy), (px0+pw,sy), 1)

        for n in self._topo_nodes:
            sx, sy = w2s(n.x, n.y)
            col = C['topo_safe'] if n.is_safe else C['topo_bad']
            pygame.draw.circle(self._screen, col, (sx,sy), 2)

        # Goal marker
        sx, sy = w2s(self._goal[0], self._goal[1])
        for d in [(-7,-7,7,7),(-7,7,7,-7)]:
            pygame.draw.line(self._screen, C['goal'], (sx+d[0],sy+d[1]),(sx+d[2],sy+d[3]),2)
        self._screen.blit(self._font_s.render("GOAL", True, C['goal']), (sx+8,sy-6))

        # Memory blobs
        for px2,py2,pw2,_ in self._mem_pts:
            sx, sy = w2s(px2, py2); r = max(4, int(pw2*14))
            blob = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
            pygame.draw.circle(blob, (*C['memory'],90), (r,r), r)
            self._screen.blit(blob,(sx-r,sy-r)); pygame.draw.circle(self._screen, C['memory'], (sx,sy), 4)

        # VSGP candidates
        for i, c in enumerate(self._subgoal_candidates):
            if c is not None:
                sx, sy = w2s(c[0], c[1])
                col = C['subgoal'] if i == 0 else tuple(v//2 for v in C['subgoal'])
                pygame.draw.circle(self._screen, col, (sx,sy), 4 if i==0 else 2)

        # TCN corridor point  ◄ NEW
        if self._tcn_nav is not None:
            tsx, tsy = w2s(float(self._tcn_nav[0]), float(self._tcn_nav[1]))
            pygame.draw.circle(self._screen, C['tcn'], (tsx, tsy), 7)
            pygame.draw.circle(self._screen, C['white'], (tsx, tsy), 7, 1)
            self._screen.blit(self._font_s.render("TCN", True, C['tcn']), (tsx+9, tsy-6))

        # Trajectory line
        if len(tx) > 1:
            pts_s = [w2s(tx[i], ty[i]) for i in range(len(tx))]
            pygame.draw.lines(self._screen, C['traj'], False, pts_s, 2)
        if tx:
            vx2, vy2 = w2s(tx[-1], ty[-1])
            pygame.draw.circle(self._screen, C['veh'], (vx2,vy2), 6)
            pygame.draw.circle(self._screen, C['white'], (vx2,vy2), 6, 1)
        if self._subgoal is not None:
            ssx, ssy = w2s(float(self._subgoal[0]), float(self._subgoal[1]))
            pygame.draw.circle(self._screen, C['subgoal'], (ssx,ssy), 6)
            pygame.draw.circle(self._screen, C['white'], (ssx,ssy), 6, 1)

        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_cost_plot(self, x, y, w, h):
        self._draw_graph(list(self._costs), "MPPI Cost", x, y, w, h, self.C['cost'])

    def _draw_speed_plot(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        self._screen.blit(self._font_s.render(
            f"Spd:{self._speed:.1f}m/s  Vib:{self._imu_vibration:.2f}  Slip:{self._imu_slip:.2f}",
            True, C['speed']), (x+4,y+3))
        if len(self._speeds) > 2:
            data = list(self._speeds); mn,mx = min(data),max(data); rng = max(mx-mn,1e-3)
            px0,py0 = x+5,y+22; pw,ph = w-10,h-26; n = len(data)
            pts_s = [(px0+int(i/(n-1)*pw), py0+int((1-(v-mn)/rng)*ph)) for i,v in enumerate(data)]
            if len(pts_s)>1: pygame.draw.lines(self._screen, C['speed'], False, pts_s, 2)
        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_graph(self, data, title, x, y, w, h, color):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        lbl = f"{title}  {data[-1]:.2f}" if data else title
        self._screen.blit(self._font_s.render(lbl, True, color), (x+4,y+3))
        if len(data) < 2:
            pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1); return
        px0,py0 = x+5,y+22; pw,ph = w-10,h-26
        mn,mx = min(data),max(data); rng = max(mx-mn,1e-3); n = len(data)
        pts_s = [(px0+int(i/(n-1)*pw), py0+int((1-(v-mn)/rng)*ph)) for i,v in enumerate(data)]
        if len(pts_s)>1: pygame.draw.lines(self._screen, color, False, pts_s, 2)
        self._screen.blit(self._font_s.render(f"{mx:.1f}", True, C['dim']), (px0+2, py0+1))
        self._screen.blit(self._font_s.render(f"{mn:.1f}", True, C['dim']), (px0+2, py0+ph-12))
        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_hud(self, W, H):
        C = self.C
        mode_c  = C['ok'] if self._mode == "AUTO" else C['warn']
        sem_col = self._SEM_COLORS.get(self._semantic_class, (100,100,100))
        stab_c  = (C['ok'] if self._stability > 0.7
                   else (C['warn'] if self._stability > 0.4 else C['danger']))
        tcn_c   = C['tcn'] if self._tcn_active else C['dim']   # ◄ NEW
        lines = [
            (f"MODE : {self._mode}",                                 mode_c),
            (f"Speed: {self._speed:5.2f} m/s",                       C['speed']),
            (f"Steer: {self._steer:+6.3f}",                          C['steer']),
            (f"Cost : {self._cost:8.1f}",                            C['cost']),
            (f"Goal : {self._dist_goal:6.1f} m",                     C['white']),
            (f"Terrain: {self._semantic_class.name}",                sem_col),
            (f"Vib  : {self._imu_vibration:5.3f}",                   stab_c),
            (f"Slip : {self._imu_slip:5.3f}",                        stab_c),
            (f"Stab : {self._stability:5.3f}",                       stab_c),
            (f"Flat : {self._flatness:6.2f}",                        C['dim']),
            (f"Pitch: {self._pitch:+5.1f}°",                         C['dim']),
            (f"Roll : {self._roll:+5.1f}°",                          C['dim']),
            (f"TopoSafe: {sum(1 for n in self._topo_nodes if n.is_safe)}", C['topo_safe']),
            (f"TCN  : {'ACTIVE' if self._tcn_active else 'idle'}",   tcn_c),  # ◄ NEW
        ]
        if self._obs_warn:
            lines.append(("!!! OBSTACLE !!!", C['danger']))
        if self._safety_active:
            lines.append((f"SAFETY: {self._safety_reason}", C['danger']))
        BW = 240; BH = len(lines)*17+8; bx = W-BW-4; by = 4
        bg = pygame.Surface((BW, BH), pygame.SRCALPHA); bg.fill((0,0,0,170))
        self._screen.blit(bg, (bx, by))
        for i, (txt, col) in enumerate(lines):
            self._screen.blit(self._font_s.render(txt, True, col), (bx+5, by+4+i*17))

    @staticmethod
    def _resize(img, w, h):
        if h <= 0 or w <= 0: return img
        sh, sw = img.shape[:2]
        ri = (np.arange(h)*sh/h).astype(int).clip(0,sh-1)
        ci = (np.arange(w)*sw/w).astype(int).clip(0,sw-1)
        return img[np.ix_(ri, ci)]


# ╔════════════════════════════════════════════════════════════════════════════╗
# §23  MISSION MANAGER
# ╚════════════════════════════════════════════════════════════════════════════╝

class MissionManager:
    def __init__(self, goal: Tuple[float, float], cfg: Config):
        self.goal = goal; self.cfg = cfg
        self._complete   = False
        self.last_dist   = float('inf')
        self.stall_counter = 0

    def update(self, st: VehicleState, t: float) -> bool:
        if self._complete: return False
        dist = math.hypot(st.x - self.goal[0], st.y - self.goal[1])
        if dist < self.cfg.goal_tolerance:
            print(f"\n[MISSION] ✓ Goal reached!  t={t:.1f}s  dist={dist:.2f}m")
            self._complete = True; return True
        self.stall_counter = 0 if dist < self.last_dist - 0.3 else self.stall_counter + 1
        self.last_dist = dist; return False

    @property
    def current_goal(self) -> Tuple[float, float]: return self.goal
    @property
    def goal_array(self) -> np.ndarray: return np.array(self.goal, dtype=np.float32)
    @property
    def is_final(self) -> bool: return True
    @property
    def complete(self) -> bool: return self._complete

    def goal_dist(self, st: VehicleState) -> float:
        return math.hypot(st.x - self.goal[0], st.y - self.goal[1])


# ╔════════════════════════════════════════════════════════════════════════════╗
# §24  BEHAVIOR PLANNER
# ╚════════════════════════════════════════════════════════════════════════════╝

class BehaviorPlanner:
    AUTO = "AUTO"; AVOID_OBSTACLE = "AVOID_OBSTACLE"
    GOAL_REACHED = "GOAL_REACHED"; MANUAL = "MANUAL"

    def __init__(self):
        self.mode = self.AUTO; self._prev_mode = self.AUTO; self._manual_flag = False

    def set_manual(self, enable: bool):
        self._manual_flag = enable
        self.mode = self.MANUAL if enable else self.AUTO
        self._prev_mode = self.mode

    def update(self, *, mission_complete: bool, obstacle_active: bool):
        if self._manual_flag: return
        self._prev_mode = self.mode
        if mission_complete:    self.mode = self.GOAL_REACHED
        elif obstacle_active:   self.mode = self.AVOID_OBSTACLE
        else:                   self.mode = self.AUTO

    @property
    def changed(self): return self.mode != self._prev_mode
    @property
    def is_autonomous(self): return self.mode in (self.AUTO, self.AVOID_OBSTACLE)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §25  SAFETY MONITOR
# ╚════════════════════════════════════════════════════════════════════════════╝

class SafetyMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._bad_imu_start: Optional[float] = None

    def check(self, st, imu=None, semantic=None, void_risk=0.0,
              forward_clearance=99.0, wall_t=0.0) -> Tuple[bool, str]:
        if abs(st.roll_deg)  > self.cfg.anti_tip_roll_danger:
            return True, f"ROLL {st.roll_deg:.1f}° DANGER"
        if abs(st.pitch_deg) > self.cfg.anti_tip_pitch_danger:
            return True, f"PITCH {st.pitch_deg:.1f}° DANGER"
        if void_risk > 0.82 and forward_clearance < 6.0:
            return True, f"CLIFF/VOID ahead: void={void_risk:.2f} fwd={forward_clearance:.1f}m"
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
        if semantic is not None:
            if (semantic.forward_class == TerrainClass.ROCK
                    and abs(st.pitch_deg) > self.cfg.sem_rock_slope_veto * 60.0):
                return True, f"ROCK+SLOPE pitch={st.pitch_deg:.1f}°"
            if (semantic.forward_class == TerrainClass.SNOW
                    and abs(st.pitch_deg) > self.cfg.sem_snow_slope_veto * 45.0):
                return True, f"SNOW+SLOPE pitch={st.pitch_deg:.1f}°"
        return False, ""

    def get_safe_stop_command(self) -> ControlState:
        return ControlState(throttle=0.0, steer=0.0, brake=1.0)

    def semantic_speed_limit(self, semantic, imu, base_speed) -> float:
        cap = base_speed
        if semantic is not None:
            cap = min(cap, semantic.speed_cap)
        if imu is not None:
            if imu.vibration > self.cfg.imu_vibration_warn:
                vf = 1.0 - 0.5 * (imu.vibration - self.cfg.imu_vibration_warn) / (
                    self.cfg.imu_vibration_danger - self.cfg.imu_vibration_warn + 1e-6)
                cap = min(cap, base_speed * clamp(vf, 0.3, 1.0))
            if imu.slip_index > 0.4:
                cap = min(cap, 5.0)
        return max(cap, 1.5)


# ╔════════════════════════════════════════════════════════════════════════════╗
# §26  CONTROL ARBITRATOR
# ╚════════════════════════════════════════════════════════════════════════════╝

class ControlArbitrator:
    def __init__(self, cfg: Config, safety_monitor: SafetyMonitor):
        self.cfg = cfg; self.safety = safety_monitor

    def arbitrate(self, behavior_mode, safety_unsafe, reactive_cmd, mppi_cmd,
                  manual_cmd, st, goal_dist, slope, roughness, obstacle_risk,
                  semantic=None, imu=None):
        if safety_unsafe:
            return self.safety.get_safe_stop_command(), "SAFETY_STOP"
        if behavior_mode == BehaviorPlanner.MANUAL and manual_cmd is not None:
            return (reactive_cmd if reactive_cmd.brake > 0.1 else manual_cmd), "MANUAL"
        if behavior_mode == BehaviorPlanner.GOAL_REACHED:
            return ControlState(brake=1.0), "GOAL_REACHED"
        if behavior_mode == BehaviorPlanner.AVOID_OBSTACLE:
            return reactive_cmd, "AVOID_OBSTACLE"
        # AUTO: apply semantic/IMU speed cap
        cmd = reactive_cmd
        if semantic is not None or imu is not None:
            spd_cap = self.safety.semantic_speed_limit(semantic, imu, self.cfg.target_speed_flat)
            if st.speed > spd_cap * 1.05:
                excess = (st.speed - spd_cap) / max(spd_cap, 1e-3)
                cmd = ControlState(
                    throttle=cmd.throttle * max(0.0, 1.0 - excess),
                    steer=cmd.steer,
                    brake=clamp(excess * 0.3, 0.0, self.cfg.max_brake),
                    reverse=cmd.reverse,
                )
        return cmd, "AUTO"


# ╔════════════════════════════════════════════════════════════════════════════╗
# §27  NAVIGATION SYSTEM   ◄ Integrated TCN throughout
# ╚════════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._client  = self._world = self._vehicle = self._sensors = None
        self._original_settings = None

        # ── Perception stack
        self._perception  = PerceptionModule(cfg)
        self._terrain_cls = TerrainClassifier()
        self._imu_proc    = IMUProcessor(cfg)

        # ── Planning stack
        self._subgoal_pl   = VSGPSubgoalPlanner(cfg)
        self._mppi         = MPPIPlanner(cfg)
        self._terrain      = TerrainAnalyzer(cfg)
        self._flat_planner = FlatGoalPlanner(cfg, self._terrain)

        # ── Memory
        self._stuck_memory   = StuckMemory(cfg)
        self._terrain_memory = TerrainMemory(cfg.terrain_memory_size, cfg.terrain_memory_resolution)
        self._topo_memory    = TopologicalMemory(cfg)
        self._persistent_memory = PersistentPathMemory(cfg)
        self._persistent_memory.seed_topology(self._topo_memory)
        self._persistent_memory.seed_terrain(self._terrain_memory)
        self._persistent_memory.begin_run(GOAL, note="live")

        # ── TCN Corridor Planner  ◄ NEW
        self._elev_builder = ElevationGridBuilder(
            size       = cfg.tcn_grid_size,
            resolution = cfg.tcn_grid_res,
        )
        self._tcn_planner = TCNPlanner(
            goal_x = cfg.goal_x,
            goal_y = cfg.goal_y,
            res    = cfg.tcn_grid_res,
        )
        self._tcn_last_nav: Optional[np.ndarray] = None   # cached between ticks

        # ── Control stack
        self._reactive   = ReactiveObstacleAvoidance(cfg)
        self._controller = Controller(cfg)
        self._manual     = ManualController(cfg)
        self._safety     = SafetyMonitor(cfg)
        self._arbitrator = ControlArbitrator(cfg, self._safety)

        # ── Mission / behaviour
        self._mission  = MissionManager(GOAL, cfg)
        self._behavior = BehaviorPlanner()

        # ── Output
        self._logger = ResultLogger(cfg)
        self._viz    = VisualizationModule(cfg)

        self._step         = 0
        self._start_wall   = time.time()
        self._nav6_prev_thr = 0.0

        self._last_semantic: Optional[SemanticOutput] = None
        self._last_imu:      Optional[IMUData]        = None

        pygame.init()
        self._screen = pygame.display.set_mode(
            (cfg.viz_win_w, cfg.viz_win_h), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            "VSGP-MPPI + TCN Navigator  |  TAB=AUTO/MANUAL  ESC=quit"
        )
        self._viz.init(self._screen)
        print(f"\n[MISSION] Goal → ({GOAL[0]:.1f}, {GOAL[1]:.1f})  "
              f"tolerance={cfg.goal_tolerance}m")
        print("[TCN] Corridor planner "
              + ("ENABLED" if cfg.tcn_enabled and SKIMAGE_OK and SCIPY_OK else "DISABLED")
              + f"  grid={cfg.tcn_grid_size}×{cfg.tcn_grid_size}  res={cfg.tcn_grid_res}m\n")

    # ── CARLA helpers ─────────────────────────────────────────────────────────

    def connect(self):
        print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port}")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world  = self._client.get_world()
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
        for dz in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
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
                print(f"[SPAWN] OK at z+{dz:.1f}"); break
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

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.connect(); self.spawn_vehicle(); self._running = True
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
        try:
            self._persistent_memory.record_topological_nodes(self._topo_memory.export_nodes())
            self._persistent_memory.flush()
            self._persistent_memory.close()
        except Exception:
            pass
        self._logger.finalize(); pygame.quit()

    # ── Per-tick logic ────────────────────────────────────────────────────────

    def _tick(self):
        if self.cfg.synchronous: self._world.tick()
        else:                    self._world.wait_for_tick()
        self._step += 1

        # ── Events
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
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._viz.init(self._screen)
        keys = pygame.key.get_pressed()

        # ── State / time
        t  = time.time() - self._start_wall
        st = self.get_state()

        # ── Mission
        if self._behavior.mode != BehaviorPlanner.MANUAL:
            if self._mission.update(st, t):
                self.apply_control(ControlState(brake=1.0)); self._running = False; return

        goal_xy   = self._mission.current_goal
        goal_arr  = self._mission.goal_array
        goal_dist = math.hypot(goal_arr[0] - st.x, goal_arr[1] - st.y)

        # ── Sensors
        pts = self._sensors.get_lidar_points()
        self._terrain.update(pts)
        if self._step % 10 == 0:
            self._logger.log_terrain_points(t, st, pts)

        # ── Geometric perception (VSGP)
        perc = (self._perception.process_lidar(pts)
                if pts is not None else self._perception.last_output)

        # ── IMU
        imu_raw = self._sensors.get_imu_data()
        if imu_raw is not None:
            self._last_imu = self._imu_proc.process(imu_raw)
        imu = self._last_imu

        # ── Semantic segmentation
        seg_img = self._sensors.get_semantic_image()
        rgb_img = self._sensors.get_front_image()
        if seg_img is not None or rgb_img is not None:
            if self._step % 5 == 0 or self._last_semantic is None:
                self._last_semantic = self._terrain_cls.classify(seg_img, rgb_img, self.cfg)
        semantic = self._last_semantic

        # ── Fuse perception layers
        if perc is not None:
            perc = self._perception.fuse(perc, semantic, imu)

        rear = self._sensors.get_rear_image()
        if rgb_img is not None: self._viz.set_front(rgb_img)
        if rear    is not None: self._viz.set_rear(rear)

        raw_pts = perc.raw_points if perc is not None else None

        # ── Dense terrain memory update
        if pts is not None and len(pts) > 0:
            self._terrain_memory.update_from_lidar(
                pts, st.x, st.y, self.cfg.terrain_memory_lidar_downsample,
                alpha_grid=perc.alpha_grid if perc is not None else None,
                void_risk_map=perc.void_risk_map if perc is not None else None,
            )

        # ── Topological memory update
        geo_slope, geo_rough, _, _, _ = self._terrain.local_risk(0.0, 0.0, radius=3.0)
        self._topo_memory.update(
            x=st.x, y=st.y, t=t, speed=st.speed,
            imu=imu, semantic=semantic,
            geo_slope=geo_slope, geo_rough=geo_rough,
        )

        # ── TCN Corridor Planner  ◄ NEW ──────────────────────────────────────
        tcn_nav_world = self._tcn_last_nav   # default: reuse cached value
        tcn_active    = False

        if (self.cfg.tcn_enabled and SKIMAGE_OK and SCIPY_OK
                and pts is not None and len(pts) > 20
                and self._step % self.cfg.tcn_update_freq == 0):
            try:
                elev_data, meta = self._elev_builder.build(pts, st.x, st.y)
                if elev_data is not None:
                    nav_pt = self._tcn_planner.update(
                        elev_data, meta,
                        valley_thresh=self.cfg.tcn_valley_prob_thresh,
                        rough_thresh =self.cfg.tcn_rough_prob_thresh,
                    )
                    if nav_pt is not None:
                        self._tcn_last_nav = nav_pt
                        tcn_nav_world = nav_pt
            except Exception as e:
                pass   # TCN errors are non-fatal; degrade gracefully
        # ─────────────────────────────────────────────────────────────────────

        # ── Behaviour state
        if self._behavior.mode != BehaviorPlanner.MANUAL:
            self._behavior.update(
                mission_complete=self._mission.complete,
                obstacle_active =self._reactive.obstacle_active,
            )
        bmode = self._behavior.mode

        # ── Safety check
        fwd_clearance = self._terrain.forward_clearance()
        void_risk     = perc.void_risk if perc is not None else 0.0
        safety_unsafe, safety_reason = self._safety.check(
            st, imu=imu, semantic=semantic,
            void_risk=void_risk, forward_clearance=fwd_clearance, wall_t=t,
        )

        self._mppi.set_semantic_imu(semantic, imu)

        # ── Planning
        subgoal_viz  = None; all_cands = None
        cur_cost = cur_reward = cur_flatness = cur_progress = 0.0
        cur_slope = cur_rough = cur_obs = cur_clearance = 0.0
        manual_cmd = None; mppi_cmd = ControlState(throttle=0.1)

        if bmode == BehaviorPlanner.MANUAL:
            manual_cmd = self._manual.tick(keys, st.roll_deg, st.pitch_deg)
            manual_cmd = self._reactive.process(raw_pts, manual_cmd, st.speed)

        else:
            if perc is not None:
                # ── VSGP Subgoal (with TCN corridor bias)  ◄ MODIFIED
                best_sg, all_cands = self._subgoal_pl.plan(
                    perc,
                    vehicle_pos=np.array([st.x, st.y, st.z], dtype=np.float32),
                    vehicle_yaw_rad=st.yaw,
                    goal_pos=goal_arr,
                    terrain_memory=self._terrain_memory,
                    topo_memory=self._topo_memory,
                    semantic=semantic,
                    persistent_memory=self._persistent_memory,
                    tcn_nav_world=tcn_nav_world,             # ◄ pass TCN hint
                    tcn_corridor_bias=self.cfg.tcn_corridor_bias,
                )

                # Clearance veto
                clearance = 0.0
                if best_sg is not None:
                    _, _, _, _, clearance = self._terrain.local_risk(
                        float(best_sg.local_pos[0]),
                        float(best_sg.local_pos[1]), radius=5.0,
                    )
                    cur_clearance = clearance
                    if clearance > self.cfg.clearance_block_threshold:
                        best_sg = None

                # ── TCN fallback when VSGP returns nothing  ◄ NEW
                if best_sg is None and tcn_nav_world is not None:
                    best_sg = self._build_tcn_subgoal(st, tcn_nav_world, goal_arr, goal_dist)
                    if best_sg is not None:
                        tcn_active = True

                # ── FlatGoalPlanner as last resort
                if best_sg is None:
                    cand         = self._flat_planner.plan(st, goal_xy)
                    subgoal_viz  = np.array([cand.wx, cand.wy, 0.0])
                    cur_flatness = cand.flatness;   cur_progress = cand.progress
                    cur_reward   = cand.reward;     cur_slope    = cand.slope
                    cur_rough    = cand.roughness;  cur_obs      = cand.obstacle_risk
                    cur_clearance = cand.clearance_risk
                else:
                    subgoal_viz  = best_sg.world_pos
                    cur_flatness = best_sg.traversability; cur_progress = best_sg.goal_progress
                    cur_slope    = best_sg.slope;          cur_rough    = best_sg.roughness
                    cur_obs      = best_sg.occupancy;      cur_clearance = clearance

                # ── MPPI trajectory optimisation
                state_vec  = [st.x, st.y, st.yaw, st.speed]
                mem_tensor = self._stuck_memory.get_tensor(self._mppi.device)
                mppi_cmd, cur_cost = self._mppi.plan(state_vec, perc, subgoal_viz, mem_tensor)
                mppi_cmd = self._controller.apply_safety_filters(
                    mppi_cmd, st.roll_deg, st.pitch_deg, speed=st.speed
                )

                self._stuck_memory.update(np.array([st.x, st.y]))
                mppi_cmd = self._reactive.process(raw_pts, mppi_cmd, st.speed)
                mppi_cmd = self._apply_throttle_bias(
                    mppi_cmd, st, goal_dist, cur_slope, cur_rough, cur_obs, semantic, imu
                )

        # ── Control arbitration
        final_cmd, final_mode = self._arbitrator.arbitrate(
            behavior_mode=bmode,
            safety_unsafe=safety_unsafe,
            reactive_cmd=mppi_cmd,
            mppi_cmd=mppi_cmd,
            manual_cmd=manual_cmd,
            st=st, goal_dist=goal_dist,
            slope=cur_slope, roughness=cur_rough, obstacle_risk=cur_obs,
            semantic=semantic, imu=imu,
        )
        if safety_unsafe:
            final_mode = f"SAFETY ({safety_reason})"

        self.apply_control(final_cmd)
        self._terrain_memory.decay()

        # ── Logging
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
            tcn_active=tcn_active,        # ◄ NEW
        )

        live_success = (goal_dist < max(self.cfg.goal_tolerance * 2.0, 18.0) and cur_progress > 0.0 and not safety_unsafe)
        live_stuck = (self._recovery.active or (st.speed < 0.25 and goal_dist > self.cfg.goal_tolerance * 1.5))
        self._persistent_memory.record_sample(
            t=t, x=st.x, y=st.y, yaw=st.yaw, speed=st.speed,
            terrain_cost=cur_cost if cur_cost > 0 else (cur_slope + cur_rough + cur_obs),
            traversability=cur_flatness, goal_dist=goal_dist, goal_progress=cur_progress,
            mode=final_mode, semantic=semantic, imu=imu,
            success=live_success, stuck=live_stuck,
        )
        if self._step % self.cfg.persistent_flush_ticks == 0:
            self._persistent_memory.flush()

        if self._step % (self.cfg.viz_every_n_ticks * 5) == 0:
            sem_str  = semantic.dominant_class.name if semantic else "?"
            vib_str  = f"{imu.vibration:.2f}"       if imu else "N/A"
            slip_str = f"{imu.slip_index:.2f}"      if imu else "N/A"
            tcn_str  = "TCN" if tcn_active else "   "
            print(
                f"\r[NAV] t={t:6.1f}s dist={goal_dist:7.2f}m spd={st.speed:5.2f} "
                f"thr={final_cmd.throttle:4.2f} steer={final_cmd.steer:+5.2f} "
                f"terrain={sem_str:<10} vib={vib_str} slip={slip_str} "
                f"{tcn_str} mode={final_mode:>20s}",
                end="", flush=True,
            )

        # ── Visualisation
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
            tcn_nav=tcn_nav_world,    # ◄ NEW
            tcn_active=tcn_active,    # ◄ NEW
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()

    # ── TCN subgoal helper  ◄ NEW ─────────────────────────────────────────────

    def _build_tcn_subgoal(
        self,
        st: VehicleState,
        tcn_nav_world: np.ndarray,
        goal_arr: np.ndarray,
        goal_dist: float,
    ) -> Optional[Subgoal]:
        """
        Convert a TCN corridor nav point to a Subgoal dataclass.
        Returns None if the point is behind the vehicle, too close, or blocked.
        """
        twx, twy = float(tcn_nav_world[0]), float(tcn_nav_world[1])
        dx = twx - st.x; dy = twy - st.y

        # Rotate displacement into vehicle local frame
        c, s  = math.cos(-st.yaw), math.sin(-st.yaw)
        lx    = c * dx - s * dy
        ly    = s * dx + c * dy
        dist  = math.hypot(dx, dy)

        # Reject points behind vehicle or too close
        if lx <= 0.5 or dist < self.cfg.subgoal_min_distance:
            return None

        slope, rough, obs, flat, clear = self._terrain.local_risk(lx, ly, radius=5.0)
        if clear > self.cfg.clearance_block_threshold:
            return None    # hard terrain block

        dist_g   = math.hypot(goal_arr[0] - twx, goal_arr[1] - twy)
        progress = goal_dist - dist_g
        h_err    = abs(math.atan2(ly, lx))

        return Subgoal(
            alpha=math.atan2(ly, lx), beta=0.0, distance=dist,
            local_pos=np.array([lx, ly, 0.0],   dtype=np.float32),
            world_pos=np.array([twx, twy, 0.0], dtype=np.float32),
            slope=slope, roughness=rough, occupancy=obs,
            variance=0.3, traversability=flat,
            terrain_cost=slope + rough,
            goal_progress=progress, heading_error=h_err,
            safe=True, width_m=self.cfg.wheel_base,
            cost=dist_g - 5.0 * progress + 3.0 * h_err,
        )

    # ── Throttle bias  ────────────────────────────────────────────────────────

    def _apply_throttle_bias(self, ctrl, st, goal_dist, slope, roughness,
                             obstacle_risk, semantic=None, imu=None) -> ControlState:
        out = ControlState(ctrl.throttle, ctrl.steer, ctrl.brake, ctrl.reverse)
        if out.reverse or out.brake > 0.05:
            self._nav6_prev_thr = out.throttle; return out

        bad = clamp(0.65*slope + 0.35*roughness + 0.25*obstacle_risk, 0.0, 1.0)
        target_speed = (1.0-bad)*self.cfg.target_speed_flat + bad*self.cfg.target_speed_bad
        if goal_dist < 30.0:
            target_speed = min(target_speed, self.cfg.target_speed_near_goal)
        target_speed *= max(0.60, 1.0 - 0.20*abs(out.steer)/max(self.cfg.max_steer, 1e-6))
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
# §27.1  ENHANCED TERRAIN / RECOVERY OVERLAY
# ╚════════════════════════════════════════════════════════════════════════════╝


@dataclass
class EnhancedConfig(Config):
    # ── Terrain geometry
    wheel_track_front: float = 1.68
    wheel_track_rear: float = 1.68
    wheel_contact_radius: float = 0.85
    terrain_patch_radius: float = 1.75
    terrain_plane_min_points: int = 18

    # ── Elevation costmap
    elevation_costmap_size: int = 96
    elevation_costmap_resolution: float = 0.50
    elevation_costmap_decay: float = 0.985
    elevation_costmap_blend: float = 0.18
    elevation_costmap_cost_weight: float = 8.0

    # ── Recovery
    recovery_stuck_ticks: int = 10
    recovery_spin_ticks: int = 6
    recovery_reverse_ticks: int = 18
    recovery_turn_ticks: int = 16
    recovery_escape_ticks: int = 20
    recovery_return_ticks: int = 16
    recovery_exit_stable_ticks: int = 8
    recovery_cooldown_ticks: int = 45
    recovery_safe_radius: float = 5.5
    recovery_speed_exit: float = 0.35
    recovery_min_progress: float = 0.08
    recovery_progress_window: int = 16
    recovery_spin_threshold: float = 0.45
    recovery_slip_threshold: float = 0.70
    recovery_throttle_reverse: float = 0.32
    recovery_throttle_escape: float = 0.34
    recovery_multipoint_steer: float = 0.62
    recovery_escape_steer_gain: float = 1.05
    recovery_escape_heading_span_deg: float = 180.0
    recovery_max_ticks: int = 90

    # ── Slip-aware MPPI
    mppi_traction_weight: float = 7.5
    mppi_slip_weight: float = 11.0
    mppi_slide_weight: float = 8.0
    mppi_clearance_weight: float = 14.0
    mppi_axle_twist_weight: float = 9.0
    mppi_roll_moment_weight: float = 10.0
    mppi_step_weight: float = 12.0
    mppi_costmap_weight: float = 5.0

    # ── Persistent experience memory
    persistent_memory_dir: str = "persistent_nav_memory"
    persistent_memory_db: str = "trajectory_memory.sqlite3"
    persistent_flush_ticks: int = 25
    persistent_min_move_m: float = 2.5
    persistent_seed_limit: int = 4000
    persistent_route_sigma_m: float = 8.0
    persistent_success_weight: float = 7.0
    persistent_failure_weight: float = 5.0
    persistent_stuck_weight: float = 6.0
    persistent_failure_dist_scale: float = 4.0
    persistent_route_follow_weight: float = 18.0
    persistent_route_offtrack_weight: float = 14.0
    persistent_route_corridor_radius_m: float = 5.0
    persistent_route_lookahead_m: float = 12.0
    persistent_route_end_margin_m: float = 18.0


CFG = EnhancedConfig()


class StuckMemory:
    def __init__(self, cfg: Config):
        self.decay = 0.995
        self.radius = cfg.memory_radius
        self.threshold = 0.05
        self.points: list = []

    def update(self, pos: np.ndarray, kind: str = "stall") -> None:
        kept = []
        for x, y, w, t in self.points:
            w2 = w * self.decay
            if w2 > self.threshold:
                kept.append((x, y, w2, t))
        self.points = kept
        px, py = float(pos[0]), float(pos[1])
        if not self.points:
            self.points.append((px, py, 1.0, kind))
            return
        if all(math.hypot(px - x, py - y) > self.radius * 0.35 for x, y, _, _ in self.points):
            self.points.append((px, py, 1.0, kind))
        else:
            x, y, w, t = self.points[-1]
            self.points[-1] = (x, y, min(4.0, w + 0.25), kind)

    def get_tensor(self, device):
        if not self.points:
            return None
        type_map = {"stall": 0.0, "slip": 1.0, "obstacle": 2.0}
        data = [[p[0], p[1], p[2], type_map.get(p[3], 0.0)] for p in self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


@dataclass
class CostMapSample:
    cost: float
    height: float
    slope: float
    step: float
    confidence: float


@dataclass
class WheelTerrainSample:
    name: str
    local_pos: np.ndarray
    height: float
    slope_deg: float
    step_height: float
    normal: np.ndarray
    support_points: int


@dataclass
class VehicleTerrainReport:
    wheel_samples: Dict[str, WheelTerrainSample]
    local_plane_normal: np.ndarray
    local_plane_height: float
    surface_continuity: float
    discontinuity: float
    axle_twist: float
    roll_moment: float
    chassis_clearance_margin: float
    traction: float
    lateral_slide_risk: float
    wheel_spin_risk: float
    elevation_cost: float
    safe: bool
    dominant_terrain: str = "UNKNOWN"


class ElevationCostMap:
    """
    Ego-centric elevation cost map. It is updated from the latest LiDAR cloud
    and queried in vehicle-local coordinates, so it can feed both MPPI and
    recovery logic without pretending the world is flat.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.size = int(cfg.elevation_costmap_size)
        self.res = float(cfg.elevation_costmap_resolution)
        self.half = self.size // 2

        self.height = np.zeros((self.size, self.size), dtype=np.float32)
        self.h_sq = np.zeros((self.size, self.size), dtype=np.float32)
        self.count = np.zeros((self.size, self.size), dtype=np.float32)
        self.slope = np.zeros((self.size, self.size), dtype=np.float32)
        self.step = np.zeros((self.size, self.size), dtype=np.float32)
        self.cost = np.zeros((self.size, self.size), dtype=np.float32)

    def clear(self) -> None:
        self.height.fill(0.0)
        self.h_sq.fill(0.0)
        self.count.fill(0.0)
        self.slope.fill(0.0)
        self.step.fill(0.0)
        self.cost.fill(0.0)

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    def update(self, pts: Optional[np.ndarray], st: VehicleState,
               terrain_report: Optional[VehicleTerrainReport] = None) -> None:
        self.clear()
        if pts is None or len(pts) < 20:
            return

        raw = pts[:, :3].astype(np.float32)
        x = raw[:, 0]
        y = raw[:, 1]
        z = raw[:, 2]

        limit = (self.size // 2 - 2) * self.res
        keep = (np.abs(x) <= limit) & (np.abs(y) <= limit)
        if not np.any(keep):
            return
        x = x[keep]
        y = y[keep]
        z = z[keep]

        gx = np.floor(x / self.res + self.half).astype(np.int32)
        gy = np.floor(y / self.res + self.half).astype(np.int32)
        valid = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)
        gx = gx[valid]
        gy = gy[valid]
        z = z[valid]

        np.add.at(self.height, (gx, gy), z)
        np.add.at(self.h_sq, (gx, gy), z * z)
        np.add.at(self.count, (gx, gy), 1.0)

        mask = self.count > 0.0
        safe = np.maximum(self.count, 1.0)
        mean = np.where(mask, self.height / safe, 0.0).astype(np.float32)
        var = np.where(mask, np.maximum(self.h_sq / safe - mean ** 2, 0.0), 0.0).astype(np.float32)

        if SCIPY_OK:
            try:
                sm = gaussian_filter(mean, sigma=1.15).astype(np.float32)
            except Exception:
                sm = mean
        else:
            sm = mean
        dmx = np.abs(np.gradient(sm, self.res, axis=1))
        dmy = np.abs(np.gradient(sm, self.res, axis=0))
        slope = np.sqrt(dmx ** 2 + dmy ** 2).astype(np.float32)

        if SCIPY_OK:
            try:
                local_avg = gaussian_filter(sm, sigma=2.25)
            except Exception:
                local_avg = sm
        else:
            local_avg = (np.roll(sm, 1, 0) + np.roll(sm, -1, 0) + np.roll(sm, 1, 1) + np.roll(sm, -1, 1)) / 4.0

        step = np.abs(sm - local_avg).astype(np.float32)

        slope_n = self._norm(slope)
        step_n = self._norm(step)
        var_n = self._norm(var)
        base_cost = np.clip(
            0.40 * slope_n + 0.35 * step_n + 0.25 * var_n,
            0.0, 1.0
        ).astype(np.float32)

        if terrain_report is not None:
            base_cost = np.clip(
                (1.0 - self.cfg.elevation_costmap_blend) * base_cost
                + self.cfg.elevation_costmap_blend * float(terrain_report.elevation_cost),
                0.0, 1.0
            ).astype(np.float32)

        self.slope = slope_n
        self.step = step_n
        self.cost = base_cost

    def _sample_grid(self, grid: np.ndarray, lx: float, ly: float) -> float:
        gx = lx / self.res + self.half
        gy = ly / self.res + self.half
        if gx < 0 or gy < 0 or gx > self.size - 1 or gy > self.size - 1:
            return float(np.max(grid) if grid.size else 1.0)
        x0 = int(math.floor(gx))
        y0 = int(math.floor(gy))
        x1 = min(x0 + 1, self.size - 1)
        y1 = min(y0 + 1, self.size - 1)
        fx = gx - x0
        fy = gy - y0
        v00 = float(grid[x0, y0])
        v10 = float(grid[x1, y0])
        v01 = float(grid[x0, y1])
        v11 = float(grid[x1, y1])
        return (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )

    def sample_local(self, lx: float, ly: float) -> CostMapSample:
        safe_height = np.where(self.count > 0, self.height / np.maximum(self.count, 1.0), 0.0)
        cost = self._sample_grid(self.cost, lx, ly)
        height = self._sample_grid(safe_height, lx, ly)
        slope = self._sample_grid(self.slope, lx, ly)
        step = self._sample_grid(self.step, lx, ly)
        confidence = self._sample_grid(np.clip(self.count / 5.0, 0.0, 1.0), lx, ly)
        return CostMapSample(
            cost=float(clamp(cost, 0.0, 1.0)),
            height=float(height),
            slope=float(clamp(slope, 0.0, 1.0)),
            step=float(clamp(step, 0.0, 1.0)),
            confidence=float(clamp(confidence, 0.0, 1.0)),
        )

    def sample_world(self, wx: float, wy: float, st: VehicleState) -> CostMapSample:
        lx, ly = world_to_local(wx - st.x, wy - st.y, st.yaw)
        return self.sample_local(lx, ly)

    def best_escape_heading(self, st: VehicleState,
                            terrain_report: Optional[VehicleTerrainReport] = None) -> float:
        span = math.radians(self.cfg.recovery_escape_heading_span_deg)
        rels = np.linspace(-span / 2.0, span / 2.0, 25)
        best_rel = 0.0
        best_cost = float("inf")
        for rel in rels:
            score = 0.0
            for dist in (2.0, 4.0, 6.0, 8.0):
                lx = dist * math.cos(rel)
                ly = dist * math.sin(rel)
                s = self.sample_local(lx, ly)
                score += 2.5 * s.cost + 1.4 * s.step + 0.8 * s.slope + 0.15 * abs(ly)
            if terrain_report is not None:
                score += 0.7 * (1.0 - terrain_report.traction)
                score += 0.5 * terrain_report.lateral_slide_risk
                score += 0.35 * max(0.0, -terrain_report.chassis_clearance_margin)
            if score < best_cost:
                best_cost = score
                best_rel = rel
        return angle_wrap(st.yaw + best_rel)


class TerrainSurfaceEstimator:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.last_report: Optional[VehicleTerrainReport] = None

    @staticmethod
    def _fit_plane(pts: np.ndarray) -> Tuple[np.ndarray, float]:
        if pts is None or len(pts) < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float32)])
        try:
            coeff, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
            normal = np.array([-a, -b, 1.0], dtype=np.float32)
            nrm = float(np.linalg.norm(normal))
            if nrm < 1e-9:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                normal /= nrm
            return normal, c
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), float(np.median(pts[:, 2]))

    def _wheel_positions(self) -> Dict[str, np.ndarray]:
        half_wb = self.cfg.wheel_base / 2.0
        half_front = self.cfg.wheel_track_front / 2.0
        half_rear = self.cfg.wheel_track_rear / 2.0
        return {
            "front_left": np.array([+half_wb, -half_front, 0.0], dtype=np.float32),
            "front_right": np.array([+half_wb, +half_front, 0.0], dtype=np.float32),
            "rear_left": np.array([-half_wb, -half_rear, 0.0], dtype=np.float32),
            "rear_right": np.array([-half_wb, +half_rear, 0.0], dtype=np.float32),
        }

    def _sample_wheel(self, pts: np.ndarray, name: str, pos: np.ndarray,
                      global_normal: np.ndarray, global_height: float) -> WheelTerrainSample:
        if pts is None or len(pts) < 3:
            return WheelTerrainSample(name, pos.copy(), float(global_height), 0.0, 0.0, global_normal.copy(), 0)

        dx = pts[:, 0] - pos[0]
        dy = pts[:, 1] - pos[1]
        mask = (dx * dx + dy * dy) <= (self.cfg.terrain_patch_radius ** 2)
        local = pts[mask]
        if len(local) < self.cfg.terrain_plane_min_points:
            mask = (dx * dx + dy * dy) <= ((self.cfg.terrain_patch_radius * 1.75) ** 2)
            local = pts[mask]

        if len(local) >= 3:
            normal, plane_h = self._fit_plane(local)
            z = local[:, 2]
            height = float(np.median(z))
            slope_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
            q95 = float(np.percentile(z, 95))
            q05 = float(np.percentile(z, 5))
            step = max(0.0, q95 - q05)
            support = int(len(local))
        else:
            normal = global_normal.copy()
            height = float(global_height)
            slope_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
            step = 0.0
            support = int(len(local))

        return WheelTerrainSample(
            name=name,
            local_pos=pos.copy(),
            height=height,
            slope_deg=slope_deg,
            step_height=float(step),
            normal=normal.copy(),
            support_points=support,
        )

    def evaluate(self, pts: Optional[np.ndarray], st: VehicleState,
                 semantic: Optional[SemanticOutput] = None,
                 imu: Optional[IMUData] = None) -> VehicleTerrainReport:
        if pts is None or len(pts) < 15:
            fallback = VehicleTerrainReport(
                wheel_samples={},
                local_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                local_plane_height=0.0,
                surface_continuity=0.0,
                discontinuity=1.0,
                axle_twist=0.0,
                roll_moment=0.0,
                chassis_clearance_margin=self.cfg.vehicle_ground_clearance,
                traction=0.35,
                lateral_slide_risk=0.75,
                wheel_spin_risk=0.65,
                elevation_cost=0.85,
                safe=False,
                dominant_terrain=(semantic.dominant_class.name if semantic else "UNKNOWN"),
            )
            self.last_report = fallback
            return fallback

        raw = pts[:, :3].astype(np.float32)
        global_normal, global_h = self._fit_plane(raw)
        wheel_pos = self._wheel_positions()

        wheel_samples: Dict[str, WheelTerrainSample] = {}
        for name, pos in wheel_pos.items():
            wheel_samples[name] = self._sample_wheel(raw, name, pos, global_normal, global_h)

        fl = wheel_samples["front_left"]
        fr = wheel_samples["front_right"]
        rl = wheel_samples["rear_left"]
        rr = wheel_samples["rear_right"]

        front_twist = fl.height - fr.height
        rear_twist = rl.height - rr.height
        axle_twist = abs(front_twist - rear_twist)
        roll_moment = abs(9.81 * 0.5 * (front_twist + rear_twist))

        step_h = max(w.step_height for w in wheel_samples.values())
        slope_h = float(np.mean([w.slope_deg for w in wheel_samples.values()]))
        continuity = float(np.mean([abs(w.normal[2]) for w in wheel_samples.values()]))
        discontinuity = float(clamp(1.0 - continuity, 0.0, 1.0))

        clearance_margin = float(self.cfg.vehicle_ground_clearance - step_h)
        roughness = float(clamp(step_h / max(self.cfg.max_step_height, 1e-6), 0.0, 1.0))
        slope_norm = float(clamp(slope_h / 45.0, 0.0, 1.0))
        twist_norm = float(clamp(axle_twist / max(self.cfg.vehicle_ground_clearance, 1e-6), 0.0, 1.5))
        roll_norm = float(clamp(roll_moment / 9.81, 0.0, 1.5))

        semantic_mult = SEMANTIC_TRAV_MULT.get(semantic.dominant_class, 0.75) if semantic else 0.75
        traction = clamp(
            (1.0
             - 0.40 * slope_norm
             - 0.35 * roughness
             - 0.20 * twist_norm
             - 0.15 * discontinuity)
            * semantic_mult,
            0.12, 1.0
        )
        lateral_slide_risk = clamp(
            0.45 * (1.0 - traction)
            + 0.30 * slope_norm
            + 0.20 * roll_norm
            + 0.15 * discontinuity,
            0.0, 1.0
        )
        wheel_spin_risk = clamp(
            0.55 * (1.0 - traction)
            + 0.25 * roughness
            + 0.20 * max(0.0, -clearance_margin / max(self.cfg.vehicle_ground_clearance, 1e-6)),
            0.0, 1.0
        )
        elevation_cost = clamp(
            0.38 * slope_norm
            + 0.32 * roughness
            + 0.18 * discontinuity
            + 0.12 * max(0.0, -clearance_margin / max(self.cfg.vehicle_ground_clearance, 1e-6)),
            0.0, 1.0
        )

        safe = bool(
            clearance_margin > -0.08
            and axle_twist < self.cfg.vehicle_ground_clearance * 2.8
            and roll_norm < 1.40
            and traction > 0.16
        )

        report = VehicleTerrainReport(
            wheel_samples=wheel_samples,
            local_plane_normal=global_normal.astype(np.float32),
            local_plane_height=float(global_h),
            surface_continuity=float(clamp(continuity, 0.0, 1.0)),
            discontinuity=float(discontinuity),
            axle_twist=float(axle_twist),
            roll_moment=float(roll_moment),
            chassis_clearance_margin=float(clearance_margin),
            traction=float(traction),
            lateral_slide_risk=float(lateral_slide_risk),
            wheel_spin_risk=float(wheel_spin_risk),
            elevation_cost=float(elevation_cost),
            safe=safe,
            dominant_terrain=(semantic.dominant_class.name if semantic else "UNKNOWN"),
        )
        self.last_report = report
        return report



class RecoveryManager:
    """
    Recovery state machine:
      normal -> reverse -> multipoint turn -> escape -> return-to-safe -> normal
    It exits automatically and should not get trapped in recovery forever.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.active = False
        self.stage = "normal"
        self.stage_ticks = 0
        self.total_ticks = 0
        self.cooldown = 0
        self.exit_stable_ticks = 0
        self.spin_ticks = 0
        self.stuck_ticks = 0
        self._progress_hist: deque = deque(maxlen=cfg.recovery_progress_window)
        self._last_goal_dist: Optional[float] = None
        self.last_safe_pose: Optional[Tuple[float, float, float]] = None
        self.reason = ""
        self.just_exited = False

    def reset(self) -> None:
        self.active = False
        self.stage = "normal"
        self.stage_ticks = 0
        self.total_ticks = 0
        self.cooldown = 0
        self.exit_stable_ticks = 0
        self.spin_ticks = 0
        self.stuck_ticks = 0
        self._progress_hist.clear()
        self._last_goal_dist = None
        self.reason = ""
        self.just_exited = False

    def update_safe_pose(self, st: VehicleState, report: VehicleTerrainReport,
                         goal_dist: float) -> None:
        if report.safe and st.speed > 0.12:
            self.last_safe_pose = (float(st.x), float(st.y), float(st.yaw))
        if self._last_goal_dist is None:
            self._last_goal_dist = float(goal_dist)

    def should_enter(self, st: VehicleState, imu: Optional[IMUData],
                     report: VehicleTerrainReport, goal_dist: float,
                     last_cmd: Optional[ControlState]) -> bool:
        if self.cooldown > 0 or self.active:
            return False

        if self._last_goal_dist is not None:
            self._progress_hist.append(self._last_goal_dist - goal_dist)
        self._last_goal_dist = float(goal_dist)

        progress = float(np.mean(self._progress_hist)) if self._progress_hist else 0.0
        cmd_thr = abs(last_cmd.throttle) if last_cmd is not None else 0.0

        low_motion = st.speed < 0.18
        poor_motion = progress < self.cfg.recovery_min_progress
        spin_signal = 0.0
        if imu is not None:
            spin_signal = max(spin_signal, float(imu.slip_index))
        spin_signal = max(spin_signal, float(report.wheel_spin_risk))

        bad_surface = (
            report.chassis_clearance_margin < -0.10
            or report.axle_twist > self.cfg.vehicle_ground_clearance * 2.4
            or report.lateral_slide_risk > 0.92
        )

        if low_motion and poor_motion and cmd_thr > 0.10:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(self.stuck_ticks - 1, 0)

        if spin_signal > self.cfg.recovery_spin_threshold and cmd_thr > 0.08:
            self.spin_ticks += 1
        else:
            self.spin_ticks = max(self.spin_ticks - 1, 0)

        trigger = (
            self.stuck_ticks >= self.cfg.recovery_stuck_ticks
            or self.spin_ticks >= self.cfg.recovery_spin_ticks
            or bad_surface
        )

        if trigger:
            self.active = True
            self.stage = "reverse"
            self.stage_ticks = 0
            self.total_ticks = 0
            self.exit_stable_ticks = 0
            self.reason = (
                "wheel-spin" if self.spin_ticks >= self.cfg.recovery_spin_ticks else
                "stuck" if self.stuck_ticks >= self.cfg.recovery_stuck_ticks else
                "terrain-instability"
            )
            return True
        return False

    def _safe_pose_dist(self, st: VehicleState) -> float:
        if self.last_safe_pose is None:
            return float("inf")
        sx, sy, _ = self.last_safe_pose
        return math.hypot(st.x - sx, st.y - sy)

    def _advance_stage(self) -> None:
        order = ("reverse", "turn", "escape", "return")
        try:
            nxt = order[order.index(self.stage) + 1]
        except Exception:
            nxt = "return"
        self.stage = nxt
        self.stage_ticks = 0

    def update(self, st: VehicleState, goal_dist: float,
               imu: Optional[IMUData], report: VehicleTerrainReport) -> Tuple[bool, str]:
        self.just_exited = False
        if self.cooldown > 0:
            self.cooldown -= 1

        if not self.active:
            return False, ""

        self.total_ticks += 1
        self.stage_ticks += 1

        safe_dist = self._safe_pose_dist(st)

        stable = (
            report.safe
            and st.speed < self.cfg.recovery_speed_exit
            and (self.last_safe_pose is None or safe_dist < self.cfg.recovery_safe_radius * 1.25)
            and (imu is None or imu.slip_index < self.cfg.recovery_slip_threshold)
            and report.wheel_spin_risk < self.cfg.recovery_spin_threshold
        )
        if stable:
            self.exit_stable_ticks += 1
        else:
            self.exit_stable_ticks = max(self.exit_stable_ticks - 1, 0)

        # Never linger in recovery forever.
        if self.total_ticks >= self.cfg.recovery_max_ticks:
            self.active = False
            self.stage = "normal"
            self.stage_ticks = 0
            self.total_ticks = 0
            self.cooldown = self.cfg.recovery_cooldown_ticks
            self.just_exited = True
            self.reason = "recovery-timeout"
            return False, self.reason

        if self.stage == "reverse" and self.stage_ticks >= self.cfg.recovery_reverse_ticks:
            self._advance_stage()
        elif self.stage == "turn" and self.stage_ticks >= self.cfg.recovery_turn_ticks:
            self._advance_stage()
        elif self.stage == "escape" and self.stage_ticks >= self.cfg.recovery_escape_ticks:
            self._advance_stage()
        elif self.stage == "return" and self.stage_ticks >= self.cfg.recovery_return_ticks:
            if stable or self.exit_stable_ticks >= 2:
                self.active = False
                self.stage = "normal"
                self.stage_ticks = 0
                self.total_ticks = 0
                self.cooldown = self.cfg.recovery_cooldown_ticks
                self.just_exited = True
                self.reason = "recovered"
                return False, self.reason

        if self.exit_stable_ticks >= self.cfg.recovery_exit_stable_ticks:
            self.active = False
            self.stage = "normal"
            self.stage_ticks = 0
            self.total_ticks = 0
            self.cooldown = self.cfg.recovery_cooldown_ticks
            self.just_exited = True
            self.reason = "recovered"
            return False, self.reason

        return True, self.stage

    def command(self, st: VehicleState, report: VehicleTerrainReport,
                costmap: ElevationCostMap) -> ControlState:
        if not self.active:
            return ControlState()

        target_yaw = None
        if self.last_safe_pose is not None:
            sx, sy, _ = self.last_safe_pose
            dx = sx - st.x
            dy = sy - st.y
            if math.hypot(dx, dy) > 1e-3:
                target_yaw = math.atan2(dy, dx)
        if target_yaw is None:
            target_yaw = costmap.best_escape_heading(st, report)

        speed_scale = clamp(report.traction, 0.15, 1.0)
        steer_error = angle_wrap(target_yaw - st.yaw)
        steer_cmd = clamp(
            steer_error * self.cfg.recovery_escape_steer_gain,
            -self.cfg.max_steer, self.cfg.max_steer
        )

        roll_bias = clamp(report.axle_twist * 0.7 + report.lateral_slide_risk * 0.2, 0.0, 0.35)
        if report.roll_moment > 0:
            steer_cmd -= math.copysign(roll_bias, steer_cmd if abs(steer_cmd) > 1e-3 else report.roll_moment)

        if self.stage == "reverse":
            return ControlState(
                throttle=clamp(self.cfg.recovery_throttle_reverse * speed_scale, 0.12, 0.35),
                steer=clamp(-steer_cmd, -self.cfg.max_steer, self.cfg.max_steer),
                brake=0.0,
                reverse=True,
            )

        if self.stage == "turn":
            pulse = -1.0 if (self.stage_ticks // 4) % 2 == 0 else 1.0
            return ControlState(
                throttle=clamp(0.16 * speed_scale, 0.08, 0.24),
                steer=clamp(pulse * self.cfg.recovery_multipoint_steer, -self.cfg.max_steer, self.cfg.max_steer),
                brake=0.0,
                reverse=(pulse < 0),
            )

        if self.stage == "escape":
            return ControlState(
                throttle=clamp(self.cfg.recovery_throttle_escape * speed_scale, 0.10, 0.33),
                steer=steer_cmd,
                brake=0.0,
                reverse=False,
            )

        return ControlState(
            throttle=clamp(0.14 * speed_scale, 0.06, 0.22),
            steer=steer_cmd,
            brake=0.0,
            reverse=False,
        )


class SlipAwareMPPIPlanner(MPPIPlanner):

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._terrain_report: Optional[VehicleTerrainReport] = None
        self._elevation_costmap: Optional[ElevationCostMap] = None

    def set_terrain_context(self, terrain_report: Optional[VehicleTerrainReport],
                            elevation_costmap: Optional[ElevationCostMap]) -> None:
        self._terrain_report = terrain_report
        self._elevation_costmap = elevation_costmap

    def plan(self, state, perc, goal_xy, memory_points,
             terrain_report: Optional[VehicleTerrainReport] = None,
             elevation_costmap: Optional[ElevationCostMap] = None):
        self.set_terrain_context(terrain_report, elevation_costmap)
        return super().plan(state, perc, goal_xy, memory_points)

    def _rollout(self, state, u, occ, slope, rough, var, trav, curv, void_map,
                 sem_risk, imu_vib, imu_slip, speed_cap):
        K, T, _ = u.shape
        device = self.device
        n_a, n_b = occ.shape[1], occ.shape[0]
        a_max = np.pi / 2.0

        x = torch.tensor(state, device=device).float().unsqueeze(0).repeat(K, 1)
        veh_x, veh_y, veh_yaw = x[0, 0].item(), x[0, 1].item(), x[0, 2].item()
        total = torch.zeros(K, device=device)
        prev_gd = torch.sqrt((self.goal[0] - x[:, 0]) ** 2 + (self.goal[1] - x[:, 1]) ** 2)

        terrain_pen = 0.0
        if self._terrain_report is not None:
            terrain_pen = (
                self.cfg.mppi_traction_weight * (1.0 - self._terrain_report.traction)
                + self.cfg.mppi_slip_weight * self._terrain_report.lateral_slide_risk
                + self.cfg.mppi_clearance_weight * max(0.0, -self._terrain_report.chassis_clearance_margin)
                + self.cfg.mppi_axle_twist_weight * clamp(self._terrain_report.axle_twist / max(self.cfg.vehicle_ground_clearance, 1e-6), 0.0, 2.0)
                + self.cfg.mppi_roll_moment_weight * clamp(self._terrain_report.roll_moment / 9.81, 0.0, 2.0)
                + self.cfg.mppi_step_weight * max(
                    self._terrain_report.wheel_samples["front_left"].step_height if "front_left" in self._terrain_report.wheel_samples else 0.0,
                    self._terrain_report.wheel_samples["front_right"].step_height if "front_right" in self._terrain_report.wheel_samples else 0.0,
                    self._terrain_report.wheel_samples["rear_left"].step_height if "rear_left" in self._terrain_report.wheel_samples else 0.0,
                    self._terrain_report.wheel_samples["rear_right"].step_height if "rear_right" in self._terrain_report.wheel_samples else 0.0,
                )
            )

        for t in range(T):
            throttle = u[:, t, 0]
            steer = u[:, t, 1]

            dx = x[:, 0] - veh_x
            dy = x[:, 1] - veh_y
            rel_yaw = torch.atan2(torch.sin(torch.atan2(dy, dx) - veh_yaw),
                                  torch.cos(torch.atan2(dy, dx) - veh_yaw))
            ai = ((rel_yaw + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi = torch.full_like(ai, n_b // 2)

            terrain_scale = torch.exp(-2.25 * (slope[bi, ai] + rough[bi, ai]))
            terrain_scale = torch.clamp(terrain_scale, 0.12, 1.0)
            traction = torch.tensor(1.0, device=device).float()
            if self._terrain_report is not None:
                traction = torch.tensor(self._terrain_report.traction, device=device).float().clamp(0.12, 1.0)

            base_v = x[:, 3].clamp(0.0, speed_cap)
            accel = throttle * terrain_scale * traction * (1.05 + 0.25 * (1.0 - rough[bi, ai]))
            brake_pen = torch.relu(-throttle) * 2.5
            v = (base_v + (accel - brake_pen) * self.dt).clamp(0.0, speed_cap)

            steer_eff = torch.tan(steer.clamp(-0.99, 0.99))
            slip_gain = torch.clamp(1.0 - 0.65 * (1.0 - traction), 0.25, 1.0)
            beta = torch.atan2(v * steer_eff * (1.0 - traction) * 0.55,
                               torch.abs(v) + 1.0 + 0.5 * traction)
            yaw_rate = (v / max(self.cfg.wheel_base, 1e-6)) * steer_eff * slip_gain
            yaw = x[:, 2] + yaw_rate * self.dt

            xp = x[:, 0] + v * torch.cos(yaw + beta) * self.dt
            yp = x[:, 1] + v * torch.sin(yaw + beta) * self.dt
            x = torch.stack([xp, yp, yaw, v], dim=1)

            dx2 = xp - veh_x
            dy2 = yp - veh_y
            rel_yaw2 = torch.atan2(torch.sin(torch.atan2(dy2, dx2) - veh_yaw),
                                   torch.cos(torch.atan2(dy2, dx2) - veh_yaw))
            ai2 = ((rel_yaw2 + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi2 = torch.full_like(ai2, n_b // 2)

            gdx = self.goal[0] - xp
            gdy = self.goal[1] - yp
            curr_gd = torch.sqrt(gdx ** 2 + gdy ** 2)
            goal_h = torch.atan2(gdy, gdx)
            h_err = torch.atan2(torch.sin(goal_h - yaw), torch.cos(goal_h - yaw)).abs()

            progress = prev_gd - curr_gd
            prev_gd = curr_gd

            terrain_map_c = (
                5.0 * occ[bi2, ai2]
                + 3.2 * slope[bi2, ai2]
                + 2.8 * rough[bi2, ai2]
                + 8.0 * curv[bi2, ai2]
                + 120.0 * void_map[bi2, ai2]
                + 10.0 * var[bi2, ai2]
                + 12.0 * (1.0 - trav[bi2, ai2]) ** 2
            )


            costmap_pen = torch.zeros(K, device=device)
            if self._terrain_report is not None:
                # Reuse the already-computed local terrain summary instead of
                # sampling the elevation grid for every rollout particle.
                costmap_pen = torch.full(
                    (K,),
                    float(self._terrain_report.elevation_cost),
                    device=device,
                    dtype=torch.float32,
                )

            wheel_spin_pen = torch.relu(throttle * (1.0 - traction) - v / max(speed_cap, 1e-3)) * 18.0
            slide_pen = torch.abs(beta) * 12.0
            jerk_pen = torch.zeros(K, device=device)
            if t > 0:
                jerk = (u[:, t, :] - u[:, t - 1, :]).abs().sum(dim=1)
                jerk_pen = 2.5 * jerk

            total += (
                terrain_map_c
                + self.cfg.w_heading * h_err
                + 0.10 * (throttle ** 2 + steer ** 2)
                + self.cfg.w_semantic * sem_risk
                + self.cfg.w_imu * (0.6 * imu_vib + 0.4 * imu_slip)
                + torch.relu(v - speed_cap) * 6.0
                + (-25.0 * progress)
                + terrain_pen
                + self.cfg.mppi_costmap_weight * costmap_pen
                + self.cfg.mppi_slide_weight * slide_pen
                + wheel_spin_pen
                + jerk_pen
            )

        return total


class TerrainAwareNavigationSystem(NavigationSystem):
    def __init__(self, cfg: Config = CFG):
        super().__init__(cfg)
        self._mppi = SlipAwareMPPIPlanner(cfg)
        self._terrain_surface = TerrainSurfaceEstimator(cfg)
        self._elevation_costmap = ElevationCostMap(cfg)
        self._recovery = RecoveryManager(cfg)

        self._last_control = ControlState()
        self._last_goal_dist: Optional[float] = None
        self._recovery_reason = ""
        self._route_guidance: Optional[RouteGuidance] = None
        self._nav_mode: str = "EXPLORE"

        print("[TERRAIN] Wheel-patch terrain evaluator, elevation costmap, and recovery state machine online")

    def _build_terrain_report(self, pts: Optional[np.ndarray], st: VehicleState,
                              semantic: Optional[SemanticOutput], imu: Optional[IMUData]) -> VehicleTerrainReport:
        report = self._terrain_surface.evaluate(pts, st, semantic, imu)
        self._elevation_costmap.update(pts, st, report)
        return report


def _terrain_veto(self, report: VehicleTerrainReport) -> Optional[str]:
    # Only veto catastrophic cases. Rough, steep, or stepped terrain should
    # influence cost, not hard-stop the vehicle long before the edge.
    if report.chassis_clearance_margin < -0.12 and report.discontinuity > 0.55:
        return f"clearance={report.chassis_clearance_margin:.2f}"
    if report.axle_twist > self.cfg.vehicle_ground_clearance * 4.0:
        return f"axle_twist={report.axle_twist:.2f}"
    if report.roll_moment > 1.9 * 9.81:
        return f"roll_moment={report.roll_moment:.2f}"
    return None

def _tick(self):
    if self.cfg.synchronous:
        self._world.tick()
    else:
        self._world.wait_for_tick()
    self._step += 1

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
                    self._manual.reset()
                    self._recovery.reset()
                    print(f"[MODE] {'MANUAL' if enter_manual else 'AUTO'}")
        elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._viz.init(self._screen)

        keys = pygame.key.get_pressed()

        t = time.time() - self._start_wall
        st = self.get_state()

        if self._behavior.mode != BehaviorPlanner.MANUAL:
            if self._mission.update(st, t):
                self.apply_control(ControlState(brake=1.0))
                self._running = False
                return

        goal_xy = self._mission.current_goal
        goal_arr = self._mission.goal_array
        goal_dist = math.hypot(goal_arr[0] - st.x, goal_arr[1] - st.y)
        goal_progress = 0.0 if self._last_goal_dist is None else (self._last_goal_dist - goal_dist)
        self._last_goal_dist = goal_dist

        pts = self._sensors.get_lidar_points()
        self._terrain.update(pts)
        if self._step % 10 == 0:
            self._logger.log_terrain_points(t, st, pts)

        perc = (self._perception.process_lidar(pts) if pts is not None else self._perception.last_output)

        imu_raw = self._sensors.get_imu_data()
        if imu_raw is not None:
            self._last_imu = self._imu_proc.process(imu_raw)
        imu = self._last_imu

        seg_img = self._sensors.get_semantic_image()
        rgb_img = self._sensors.get_front_image()
        if seg_img is not None or rgb_img is not None:
            if self._step % 5 == 0 or self._last_semantic is None:
                self._last_semantic = self._terrain_cls.classify(seg_img, rgb_img, self.cfg)
        semantic = self._last_semantic

        if perc is not None:
            perc = self._perception.fuse(perc, semantic, imu)

        rear = self._sensors.get_rear_image()
        if rgb_img is not None:
            self._viz.set_front(rgb_img)
        if rear is not None:
            self._viz.set_rear(rear)

        raw_pts = perc.raw_points if perc is not None else None

        if pts is not None and len(pts) > 0:
            self._terrain_memory.update_from_lidar(
                pts, st.x, st.y, self.cfg.terrain_memory_lidar_downsample,
                alpha_grid=perc.alpha_grid if perc is not None else None,
                void_risk_map=perc.void_risk_map if perc is not None else None,
            )

        geo_slope, geo_rough, _, _, _ = self._terrain.local_risk(0.0, 0.0, radius=3.0)
        self._topo_memory.update(
            x=st.x, y=st.y, t=t, speed=st.speed,
            imu=imu, semantic=semantic,
            geo_slope=geo_slope, geo_rough=geo_rough,
        )

        terrain_report = self._build_terrain_report(raw_pts, st, semantic, imu)
        terrain_veto = self._terrain_veto(terrain_report)

        route_goal_xy = np.array([goal_arr[0], goal_arr[1]], dtype=np.float32)
        route_info = self._persistent_memory.route_guidance(
            st.x,
            st.y,
            lookahead_m=max(self.cfg.persistent_route_lookahead_m, st.speed * 1.75),
            corridor_radius=self.cfg.persistent_route_corridor_radius_m,
        )
        self._route_guidance = route_info
        route_active = bool(route_info.active and route_info.target_xy is not None)
        if route_active:
            route_goal_xy = np.array(route_info.target_xy[:2], dtype=np.float32)
            self._nav_mode = "FOLLOW_MEMORY"
            if route_info.remaining_m <= self.cfg.persistent_route_end_margin_m:
                route_active = False
                route_goal_xy = np.array([goal_arr[0], goal_arr[1]], dtype=np.float32)
                self._nav_mode = "EXPLORE"
        else:
            self._nav_mode = "EXPLORE"

        tcn_nav_world = self._tcn_last_nav
        tcn_active = False
        if (self.cfg.tcn_enabled and SKIMAGE_OK and SCIPY_OK
                and pts is not None and len(pts) > 20
                and self._step % self.cfg.tcn_update_freq == 0):
            try:
                elev_data, meta = self._elev_builder.build(pts, st.x, st.y)
                if elev_data is not None:
                    nav_pt = self._tcn_planner.update(
                        elev_data, meta,
                        valley_thresh=self.cfg.tcn_valley_prob_thresh,
                        rough_thresh=self.cfg.tcn_rough_prob_thresh,
                    )
                    if nav_pt is not None:
                        self._tcn_last_nav = nav_pt
                        tcn_nav_world = nav_pt
            except Exception:
                pass

        if self._behavior.mode != BehaviorPlanner.MANUAL:
            self._behavior.update(
                mission_complete=self._mission.complete,
                obstacle_active=self._reactive.obstacle_active,
            )
        bmode = self._behavior.mode

        fwd_clearance = self._terrain.forward_clearance()
        void_risk = perc.void_risk if perc is not None else 0.0
        safety_unsafe, safety_reason = self._safety.check(
            st, imu=imu, semantic=semantic,
            void_risk=void_risk, forward_clearance=fwd_clearance, wall_t=t,
        )

        if terrain_veto is not None:
            safety_unsafe = True
            safety_reason = f"TERRAIN {terrain_veto}"

        self._mppi.set_semantic_imu(semantic, imu)
        self._recovery.update_safe_pose(st, terrain_report, goal_dist)

        subgoal_viz = None
        all_cands = None
        cur_cost = cur_reward = cur_flatness = cur_progress = 0.0
        cur_slope = cur_rough = cur_obs = cur_clearance = 0.0
        manual_cmd = None
        mppi_cmd = ControlState(throttle=0.1)

        if bmode == BehaviorPlanner.MANUAL:
            self._recovery.reset()
            manual_cmd = self._manual.tick(keys, st.roll_deg, st.pitch_deg)
            manual_cmd = self._reactive.process(raw_pts, manual_cmd, st.speed)
            final_cmd = manual_cmd
            final_mode = "MANUAL"
        else:
            self._recovery.should_enter(st, imu, terrain_report, goal_dist, self._last_control)

            recovery_active, recovery_stage = self._recovery.update(st, goal_dist, imu, terrain_report)
            if recovery_active:
                final_cmd = self._recovery.command(st, terrain_report, self._elevation_costmap)
                final_cmd = self._controller.apply_safety_filters(
                    final_cmd, st.roll_deg, st.pitch_deg, speed=st.speed
                )
                final_cmd = self._reactive.process(raw_pts, final_cmd, st.speed)
                final_mode = f"RECOVERY/{recovery_stage}"
                cur_slope = terrain_report.surface_continuity
                cur_rough = terrain_report.discontinuity
                cur_obs = terrain_report.lateral_slide_risk
                cur_clearance = terrain_report.chassis_clearance_margin
                cur_flatness = terrain_report.surface_continuity
                cur_progress = goal_progress
                cur_reward = 0.0
            else:
                if perc is not None:
                    best_sg, all_cands = self._subgoal_pl.plan(
                        perc,
                        vehicle_pos=np.array([st.x, st.y, st.z], dtype=np.float32),
                        vehicle_yaw_rad=st.yaw,
                        goal_pos=route_goal_xy,
                        terrain_memory=self._terrain_memory,
                        topo_memory=self._topo_memory,
                        semantic=semantic,
                        tcn_nav_world=tcn_nav_world,
                        tcn_corridor_bias=self.cfg.tcn_corridor_bias,
                    )

                    clearance = 0.0
                    if best_sg is not None:
                        _, _, _, _, clearance = self._terrain.local_risk(
                            float(best_sg.local_pos[0]),
                            float(best_sg.local_pos[1]), radius=5.0,
                        )
                        cur_clearance = clearance
                        cmap_sample = self._elevation_costmap.sample_local(
                            float(best_sg.local_pos[0]), float(best_sg.local_pos[1])
                        )
                        if clearance > self.cfg.clearance_block_threshold or cmap_sample.cost > 0.82:
                            best_sg = None

                    if best_sg is None and tcn_nav_world is not None:
                        best_sg = self._build_tcn_subgoal(st, tcn_nav_world, route_goal_xy, goal_dist)
                        if best_sg is not None:
                            tcn_active = True

                    if best_sg is None:
                        cand = self._flat_planner.plan(st, tuple(route_goal_xy.tolist()))
                        subgoal_viz = np.array([cand.wx, cand.wy, 0.0])
                        cur_flatness = cand.flatness
                        cur_progress = cand.progress
                        cur_reward = cand.reward
                        cur_slope = cand.slope
                        cur_rough = cand.roughness
                        cur_obs = cand.obstacle_risk
                        cur_clearance = cand.clearance_risk
                    else:
                        subgoal_viz = best_sg.world_pos
                        cur_flatness = best_sg.traversability
                        cur_progress = best_sg.goal_progress
                        cur_slope = best_sg.slope
                        cur_rough = best_sg.roughness
                        cur_obs = best_sg.occupancy
                        cur_clearance = clearance

                    state_vec = [st.x, st.y, st.yaw, st.speed]
                    mem_tensor = self._stuck_memory.get_tensor(self._mppi.device)
                    mppi_cmd, cur_cost = self._mppi.plan(
                        state_vec, perc, subgoal_viz, mem_tensor,
                        terrain_report=terrain_report,
                        elevation_costmap=self._elevation_costmap,
                    )
                    mppi_cmd = self._controller.apply_safety_filters(
                        mppi_cmd, st.roll_deg, st.pitch_deg, speed=st.speed
                    )

                    self._stuck_memory.update(np.array([st.x, st.y]), kind="stall")
                    mppi_cmd = self._reactive.process(raw_pts, mppi_cmd, st.speed)
                    mppi_cmd = self._apply_throttle_bias(
                        mppi_cmd, st, goal_dist, cur_slope, cur_rough, cur_obs, semantic, imu
                    )
                    final_cmd = mppi_cmd
                    final_mode = "AUTO"

        if safety_unsafe:
            final_cmd, final_mode = self._arbitrator.arbitrate(
                behavior_mode=BehaviorPlanner.AUTO,
                safety_unsafe=True,
                reactive_cmd=final_cmd,
                mppi_cmd=final_cmd,
                manual_cmd=manual_cmd,
                st=st, goal_dist=goal_dist,
                slope=cur_slope, roughness=cur_rough, obstacle_risk=cur_obs,
                semantic=semantic, imu=imu,
            )
            final_mode = f"SAFETY ({safety_reason})"

        self._last_control = final_cmd
        self.apply_control(final_cmd)
        self._terrain_memory.decay()

        cost_total = cur_cost + 0.3 * final_cmd.throttle ** 2 + 0.8 * final_cmd.steer ** 2 + 0.4 * final_cmd.brake ** 2
        reward_total = cur_reward + 8.0 * st.speed - 0.35 * goal_dist

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
            tcn_active=tcn_active,
        )

        if self._step % (self.cfg.viz_every_n_ticks * 5) == 0:
            sem_str = semantic.dominant_class.name if semantic else "?"
            vib_str = f"{imu.vibration:.2f}" if imu else "N/A"
            slip_str = f"{imu.slip_index:.2f}" if imu else "N/A"
            rec_str = "REC" if self._recovery.active else "   "
            tcn_str = "TCN" if tcn_active else "   "
            print(
                f"\r[NAV] t={t:6.1f}s dist={goal_dist:7.2f}m spd={st.speed:5.2f} "
                f"thr={final_cmd.throttle:4.2f} steer={final_cmd.steer:+5.2f} "
                f"terrain={sem_str:<10} vib={vib_str} slip={slip_str} "
                f"{tcn_str} {rec_str} mode={final_mode:>20s}",
                end="", flush=True,
            )

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
            tcn_nav=tcn_nav_world,
            tcn_active=tcn_active,
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()

        if self._recovery.just_exited:
            self._controller.reset()
            self._nav6_prev_thr = 0.0

# Rebind the public names so the entry point uses the enhanced versions.
NavigationSystem = TerrainAwareNavigationSystem
MPPIPlanner = SlipAwareMPPIPlanner


# ╔════════════════════════════════════════════════════════════════════════════╗
# §28  ENTRY POINT
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    nav = NavigationSystem(CFG)
    try:
        nav.run()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()