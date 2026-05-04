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

    # ── Vehicle
    vehicle_blueprint: str      = "vehicle.tesla.cybertruck"
    spawn_z_offset: float       = 5.0
    post_spawn_wait_s: float    = 3.0
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

    # ── Local Planner
    robot_width: float          = 0.8
    safety_margin: float        = 2.80
    max_slope_deg: float        = 15.0
    subgoal_distance: float     = 5.0
    n_subgoals_max: int         = 5

    # ── GLOBAL PLANNER CONFIG
    goal_location: Tuple[float, float, float] = (-1169.18, -413.52, 5.0)
    goal_tolerance: float       = 5.0
    global_planner_enabled: bool = True
    waypoint_spacing: float     = 10.0
    lookahead_distance: float   = 15.0

    # ── Cost weights
    w_direction: float          = 4.0
    w_distance: float          = 2.5
    w_steepness: float         = 50.0
    w_collision: float         = 500.0
    w_oscillation: float       = 50.0
    w_flatness: float          = 5.0
    w_progress: float          = 10.0       # NEW: Progress toward goal reward
    w_exploration: float        = 3.0        # NEW: Exploration bonus

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

    # ── Stuck detection & recovery
    stuck_window_s: float           = 2.0
    stuck_disp_thresh: float        = 0.6
    stuck_oscillation_thresh: float = 0.5
    recovery_duration_s: float      = 4.0
    recovery_phase_durations: Tuple[float, float, float] = (1.5, 1.5, 1.0)

    # ── Memory lane for stuck locations
    memory_grid_size: float          = 2.0       # Grid cell size in meters
    memory_max_entries: int          = 50        # Max stuck locations to remember

    # ── VSGP
    vsgp_n_inducing: int        = 50
    vsgp_alpha: float           = 1.0
    vsgp_length_scale: float    = 0.3
    vsgp_noise_var: float       = 0.05
    vsgp_lr: float              = 0.02
    vsgp_update_freq: int       = 10

    # ── Visualization
    viz_update_hz: float        = 3.0
    traj_history: int          = 500


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
# §2  MEMORY LANE (Stuck Location Memory)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MemoryLane:
    """
    Remembers locations where the vehicle got stuck and penalizes 
    revisiting those areas to encourage exploration of new routes.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.grid_size = cfg.memory_grid_size
        self.max_entries = cfg.memory_max_entries
        self._stuck_locations: deque = deque(maxlen=self.max_entries)
        self._recent_positions: deque = deque(maxlen=100)  # Track recent path

    def add_stuck_location(self, pos: np.ndarray) -> None:
        """Record a location where the vehicle got stuck."""
        grid_x = round(pos[0] / self.grid_size) * self.grid_size
        grid_y = round(pos[1] / self.grid_size) * self.grid_size
        grid_z = round(pos[2] / self.grid_size) * self.grid_size
        
        entry = (grid_x, grid_y, grid_z, time.time())
        self._stuck_locations.append(entry)
        print(f"[MEMORY] Added stuck location: ({grid_x:.1f}, {grid_y:.1f})")

    def add_position(self, pos: np.ndarray) -> None:
        """Track recent positions to detect loops."""
        grid_x = round(pos[0] / self.grid_size) * self.grid_size
        grid_y = round(pos[1] / self.grid_size) * self.grid_size
        self._recent_positions.append((grid_x, grid_y, time.time()))

    def get_penalty(self, pos: np.ndarray) -> float:
        """
        Returns penalty for being near a stuck location.
        Higher penalty = more likely to avoid this area.
        """
        grid_x = round(pos[0] / self.grid_size) * self.grid_size
        grid_y = round(pos[1] / self.grid_size) * self.grid_size
        
        penalty = 0.0
        now = time.time()
        
        for stuck_x, stuck_y, stuck_z, stuck_time in self._stuck_locations:
            dist = math.sqrt((grid_x - stuck_x) ** 2 + (grid_y - stuck_y) ** 2)
            
            # Age factor - older stuck locations matter less
            age = now - stuck_time
            age_factor = max(0.3, 1.0 - age / 60.0)  # Decay over 60 seconds
            
            if dist < self.grid_size * 2:
                penalty += (3.0 - dist) * age_factor  # Strong penalty for close locations
            elif dist < self.grid_size * 4:
                penalty += (5.0 - dist) * 0.5 * age_factor  # Weaker penalty for farther
        
        return min(penalty, 10.0)  # Cap penalty at 10.0

    def check_loops(self, pos: np.ndarray) -> float:
        """
        Detect if vehicle is looping in circles.
        Returns penalty if similar position was visited recently.
        """
        grid_x = round(pos[0] / self.grid_size) * self.grid_size
        grid_y = round(pos[1] / self.grid_size) * self.grid_size
        
        loop_penalty = 0.0
        now = time.time()
        
        recent_positions = list(self._recent_positions)
        for i, (px, py, pt) in enumerate(recent_positions):
            if now - pt < 5.0:  # Only check last 5 seconds
                dist = math.sqrt((grid_x - px) ** 2 + (grid_y - py) ** 2)
                if dist < self.grid_size * 0.5:  # Very close position
                    loop_penalty += 1.0
        
        return min(loop_penalty, 5.0)

    def clear_old_entries(self, max_age_seconds: float = 120.0) -> None:
        """Clear stuck locations older than max_age_seconds."""
        now = time.time()
        self._stuck_locations = deque(
            [entry for entry in self._stuck_locations if now - entry[3] < max_age_seconds],
            maxlen=self.max_entries
        )
        self._recent_positions = deque(
            [entry for entry in self._recent_positions if now - entry[2] < 30.0],
            maxlen=100
        )

    def get_stuck_count_near(self, pos: np.ndarray, radius: float = 10.0) -> int:
        """Count how many times we've been stuck near this position."""
        grid_x = round(pos[0] / self.grid_size) * self.grid_size
        grid_y = round(pos[1] / self.grid_size) * self.grid_size
        
        count = 0
        for stuck_x, stuck_y, _, _ in self._stuck_locations:
            dist = math.sqrt((grid_x - stuck_x) ** 2 + (grid_y - stuck_y) ** 2)
            if dist < radius:
                count += 1
        return count


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §3  PROGRESS TRACKER (Forward Bias & Exploration)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ProgressTracker:
    """
    Tracks vehicle progress toward goal and provides:
    - Progress rewards when moving closer
    - Exploration bonuses for trying new areas
    - Forward bias to keep the vehicle moving
    """

    def __init__(self, cfg: Config = CFG, goal_location: np.ndarray = None):
        self.cfg = cfg
        self.goal_location = goal_location if goal_location is not None else np.array(cfg.goal_location)
        self._initial_distance: Optional[float] = None
        self._last_distance: float = float('inf')
        self._last_position: Optional[np.ndarray] = None
        self._progress_history: deque = deque(maxlen=50)
        self._total_progress: float = 0.0
        self._stuck_episodes: int = 0
        self._best_distance: float = float('inf')
        self._explored_areas: deque = deque(maxlen=100)  # Track explored grid cells

    def initialize(self, current_pos: np.ndarray) -> None:
        """Initialize with starting position."""
        self._initial_distance = self._get_distance_to_goal(current_pos)
        self._last_distance = self._initial_distance
        self._last_position = current_pos.copy()
        self._best_distance = self._initial_distance
        self._total_progress = 0.0
        print(f"[PROGRESS] Initialized. Start dist: {self._initial_distance:.1f}m")

    def _get_distance_to_goal(self, pos: np.ndarray) -> float:
        return np.linalg.norm(pos[:2] - self.goal_location[:2])

    def update(self, current_pos: np.ndarray) -> dict:
        """
        Update progress tracking and return metrics.
        Returns dict with progress_score, exploration_bonus, forward_bias.
        """
        current_dist = self._get_distance_to_goal(current_pos)
        now = time.time()
        
        metrics = {
            'progress_score': 0.0,
            'exploration_bonus': 0.0,
            'forward_bias': 0.0,
            'distance_to_goal': current_dist,
            'improvement_rate': 0.0,
        }
        
        if self._last_position is None:
            self._last_position = current_pos.copy()
            return metrics
        
        # Calculate movement
        movement = np.linalg.norm(current_pos[:2] - self._last_position[:2])
        distance_change = self._last_distance - current_dist
        
        # Progress score: positive when moving closer, negative when moving away
        if distance_change > 0.1:
            # Moving toward goal - reward based on improvement rate
            metrics['progress_score'] = min(distance_change * self.cfg.w_progress, 5.0)
            self._total_progress += distance_change
            if current_dist < self._best_distance:
                self._best_distance = current_dist
        elif distance_change < -0.5:
            # Moving away from goal - penalty
            metrics['progress_score'] = max(distance_change * 2.0, -3.0)
        
        # Forward bias: encourage consistent forward progress
        if movement > 0.1:
            forward_progress_ratio = max(0, distance_change) / (movement + 0.01)
            metrics['forward_bias'] = forward_progress_ratio * self.cfg.w_direction * 0.5
        
        # Exploration bonus: reward for visiting new areas
        grid_x = round(current_pos[0] / 3.0) * 3.0
        grid_y = round(current_pos[1] / 3.0) * 3.0
        current_cell = (grid_x, grid_y, now)
        
        is_new_area = True
        for area_x, area_y, _ in self._explored_areas:
            if abs(grid_x - area_x) < 6.0 and abs(grid_y - area_y) < 6.0:
                is_new_area = False
                break
        
        if is_new_area:
            self._explored_areas.append(current_cell)
            metrics['exploration_bonus'] = self.cfg.w_exploration * 0.5
        else:
            # Small bonus for maintaining exploration intent
            metrics['exploration_bonus'] = 0.1
        
        # Update state
        self._last_distance = current_dist
        self._last_position = current_pos.copy()
        self._progress_history.append((now, current_dist, distance_change))
        
        return metrics

    def on_stuck(self) -> None:
        """Called when vehicle gets stuck - resets some tracking."""
        self._stuck_episodes += 1
        # Keep best distance but allow fresh start
        print(f"[PROGRESS] Stuck episode #{self._stuck_episodes}. Best: {self._best_distance:.1f}m")

    def get_overall_progress(self) -> float:
        """Get overall progress as percentage of initial distance."""
        if self._initial_distance is None or self._initial_distance == 0:
            return 0.0
        return (self._initial_distance - self._best_distance) / self._initial_distance * 100.0

    def reset(self, current_pos: np.ndarray) -> None:
        """Reset with new position (for replanning)."""
        self._last_position = current_pos.copy()
        self._last_distance = self._get_distance_to_goal(current_pos)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §4  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
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

        current_hash = hash(X.tobytes())
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
# §5  PERCEPTION MODULE
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
    distance_map: Optional[np.ndarray] = None


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
            np.linalg.norm(normal) * np.linalg.norm(z_axis) + 1e-9)
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

        # FIX: Compute actual distance map (in meters)
        distance_map = self.ROC - mean

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
            distance_map=distance_map,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §6  SUBGOAL PLANNER (with Memory & Progress Integration)
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
    escape_direction: Optional[str] = None
    
    # NEW: Cost breakdown for visualization
    cost_direction: float = 0.0
    cost_steepness: float = 0.0
    cost_memory: float = 0.0
    cost_progress: float = 0.0


class VSGPSubgoalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.min_free_distance = 3.0
        self.max_var_thresh = 0.4
        self.max_slope_thresh = 0.25
        self.min_dist = 3.0
        self.max_dist = cfg.subgoal_distance
        
        # NEW: Memory lane integration
        self._memory_lane: Optional[MemoryLane] = None
        self._progress_tracker: Optional[ProgressTracker] = None

    def set_memory(self, memory_lane: MemoryLane) -> None:
        self._memory_lane = memory_lane

    def set_progress_tracker(self, progress_tracker: ProgressTracker) -> None:
        self._progress_tracker = progress_tracker

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pitch_rad: float,
        vehicle_transform: carla.Transform,
        global_waypoint: Optional[np.ndarray] = None,
        mode: str = "forward",
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        if perc.occupancy_mean is None or perc.distance_map is None:
            return None, []

        alpha_range = np.linspace(-np.pi / 2.5, np.pi / 2.5, 15)
        candidates = []

        occ = perc.occupancy_mean
        var = perc.occupancy_var
        slope = perc.slope_map
        dist_map = perc.distance_map

        alphas = perc.alpha_grid[0]
        betas = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        # Calculate preferred direction from global waypoint
        preferred_alpha = 0.0
        preferred_direction = np.array([0.0, 0.0])
        vehicle_pos = np.array([vehicle_transform.location.x,
                               vehicle_transform.location.y,
                               vehicle_transform.location.z])
        
        if global_waypoint is not None:
            direction = global_waypoint[:2] - vehicle_pos[:2]
            dist_to_wp = np.linalg.norm(direction)
            if dist_to_wp > 0.1:
                preferred_direction = direction / dist_to_wp
            preferred_alpha = math.atan2(direction[1], direction[0])
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            preferred_alpha = preferred_alpha - vehicle_yaw

        # Mode-specific behavior
        escape_direction = None
        if mode == "escape_left":
            escape_direction = "left"
        elif mode == "escape_right":
            escape_direction = "right"
        elif mode == "escape":
            escape_direction = "toward_global"

        # Get progress metrics
        progress_metrics = {'progress_score': 0.0, 'exploration_bonus': 0.0, 'forward_bias': 0.0}
        if self._progress_tracker is not None:
            progress_metrics = self._progress_tracker.update(vehicle_pos)

        for a in alpha_range:
            a_idx = np.argmin(np.abs(alphas - a))

            best_point_cost = float('inf')
            best_point_dist = 0.0
            s_val = 0.0
            
            # Cost breakdown for visualization
            cost_dir = 0.0
            cost_steep = 0.0
            cost_mem = 0.0
            cost_prog = 0.0

            dist_samples = np.linspace(self.min_dist, self.max_dist, 5)

            for d in dist_samples:
                step_ratio = d / self.max_dist
                pitch_correction = vehicle_pitch_rad * step_ratio * 5
                b_idx = int(mid_beta_idx - step_ratio * (mid_beta_idx - 1) + pitch_correction)
                b_idx = int(np.clip(b_idx, 0, occ.shape[0] - 1))

                if not (0 <= a_idx < occ.shape[1]):
                    break

                actual_distance = dist_map[b_idx, a_idx]
                is_free = actual_distance > self.min_free_distance
                is_stable = var[b_idx, a_idx] < self.max_var_thresh
                is_safe = slope[b_idx, a_idx] < self.max_slope_thresh

                if is_free and is_stable and is_safe:
                    direction_cost = abs(a) / (np.pi / 2)
                    slope_cost = slope[b_idx, a_idx]
                    
                    # Calculate world position for memory penalty
                    lx = d * np.cos(a)
                    ly = d * np.sin(a)
                    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                    test_x = vehicle_pos[0] + lx * math.cos(vehicle_yaw) - ly * math.sin(vehicle_yaw)
                    test_y = vehicle_pos[1] + lx * math.sin(vehicle_yaw) + ly * math.cos(vehicle_yaw)
                    test_pos = np.array([test_x, test_y, vehicle_pos[2]])
                    
                    # Memory lane penalty
                    memory_penalty = 0.0
                    if self._memory_lane is not None:
                        memory_penalty = self._memory_lane.get_penalty(test_pos)
                        loop_penalty = self._memory_lane.check_loops(test_pos)
                        memory_penalty += loop_penalty
                    
                    # Progress/reward bonus for this direction
                    progress_bonus = 0.0
                    if self._progress_tracker is not None:
                        # Reward going toward goal
                        local_dir = np.array([np.cos(a), np.sin(a)])
                        alignment = np.dot(local_dir, preferred_direction)
                        progress_bonus = alignment * progress_metrics.get('forward_bias', 0.0)
                        
                        # Exploration bonus for new areas
                        progress_bonus += progress_metrics.get('exploration_bonus', 0.0) * abs(a) / (np.pi / 2 + 0.01)

                    # Mode-specific cost adjustments
                    if mode in ("escape_left", "escape_right", "escape"):
                        escape_bonus = 0.0
                        if mode == "escape_left" and a < -0.3:
                            escape_bonus = -2.0
                        elif mode == "escape_right" and a > 0.3:
                            escape_bonus = -2.0
                        elif mode == "escape" and escape_direction == "toward_global":
                            alignment = np.dot(local_dir, preferred_direction)
                            escape_bonus = -alignment * 2.0
                        
                        cost = (
                            self.cfg.w_direction * direction_cost +
                            self.cfg.w_flatness * slope_cost +
                            0.5 * var[b_idx, a_idx] +
                            escape_bonus +
                            memory_penalty -
                            progress_bonus
                        )
                    else:
                        # Normal forward mode with all biases
                        global_penalty = abs(a - preferred_alpha) * 1.0 if global_waypoint is not None else 0.0
                        
                        # Forward bias - strongly prefer directions toward goal
                        forward_bonus = 0.0
                        alignment = np.dot(local_dir, preferred_direction)
                        if alignment > 0.7:  # Strong forward alignment
                            forward_bonus = -2.0 * alignment
                        
                        cost = (
                            self.cfg.w_direction * direction_cost +
                            self.cfg.w_flatness * slope_cost +
                            0.5 * var[b_idx, a_idx] +
                            global_penalty +
                            memory_penalty -
                            progress_bonus -
                            forward_bonus
                        )
                        cost -= d * 0.3

                    if cost < best_point_cost:
                        best_point_cost = cost
                        best_point_dist = d
                        s_val = slope[b_idx, a_idx]
                        cost_dir = direction_cost
                        cost_steep = slope_cost
                        cost_mem = memory_penalty
                        cost_prog = progress_bonus
                else:
                    break

            if best_point_dist > 0:
                lx = best_point_dist * np.cos(a)
                ly = best_point_dist * np.sin(a)

                vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                world_x = vehicle_pos[0] + lx * math.cos(vehicle_yaw) - ly * math.sin(vehicle_yaw)
                world_y = vehicle_pos[1] + lx * math.sin(vehicle_yaw) + ly * math.cos(vehicle_yaw)
                world_pos = np.array([world_x, world_y, vehicle_pos[2]])

                sg = Subgoal(
                    alpha=a,
                    beta=0.0,
                    distance=best_point_dist,
                    local_pos=np.array([lx, ly, 0.0]),
                    slope_deg=s_val * 45.0,
                    safe=True,
                    width_m=self.cfg.vehicle_width,
                    cost=best_point_cost,
                    world_pos=world_pos,
                    escape_direction=escape_direction,
                    cost_direction=cost_dir * self.cfg.w_direction,
                    cost_steepness=cost_steep * self.cfg.w_steepness,
                    cost_memory=cost_mem,
                    cost_progress=cost_prog,
                )
                candidates.append(sg)

        if not candidates:
            return None, []

        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §7  GLOBAL PLANNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class GlobalWaypoint:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    reached: bool = False


class GlobalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.goal_location = np.array(cfg.goal_location)
        self.waypoint_spacing = cfg.waypoint_spacing
        self.waypoints: List[GlobalWaypoint] = []
        self.current_waypoint_idx = 0
        self.path_computed = False
        self.goal_reached = False
        self._replan_counter = 0
        self._last_replan_time = 0.0

    def compute_path(self, start_pos: np.ndarray, start_yaw: float) -> List[GlobalWaypoint]:
        self.waypoints = []

        direction = self.goal_location[:2] - start_pos[:2]
        total_distance = np.linalg.norm(direction)

        if total_distance < self.cfg.goal_tolerance:
            self.goal_reached = True
            print(f"[GLOBAL] Already at goal! Distance: {total_distance:.2f}m")
            return self.waypoints

        if total_distance > 0:
            direction = direction / total_distance

        num_waypoints = max(1, int(total_distance / self.waypoint_spacing))

        for i in range(num_waypoints + 1):
            progress = i / num_waypoints
            wp_x = start_pos[0] + direction[0] * total_distance * progress
            wp_y = start_pos[1] + direction[1] * total_distance * progress
            wp_z = start_pos[2] + (self.goal_location[2] - start_pos[2]) * progress

            if i < num_waypoints:
                next_progress = (i + 1) / num_waypoints
                next_wp_x = start_pos[0] + direction[0] * total_distance * next_progress
                next_wp_y = start_pos[1] + direction[1] * total_distance * next_progress
                wp_yaw = math.degrees(math.atan2(next_wp_y - wp_y, next_wp_x - wp_x))
            else:
                wp_yaw = -82.22

            waypoint = GlobalWaypoint(x=wp_x, y=wp_y, z=wp_z, yaw=wp_yaw)
            self.waypoints.append(waypoint)

        self.waypoints[-1].x = self.goal_location[0]
        self.waypoints[-1].y = self.goal_location[1]
        self.waypoints[-1].z = self.goal_location[2]
        self.waypoints[-1].yaw = -82.22

        self.path_computed = True
        self.current_waypoint_idx = 0

        print(f"[GLOBAL] Path computed: {len(self.waypoints)} waypoints")
        print(f"[GLOBAL] Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}) -> Goal: ({self.goal_location[0]:.1f}, {self.goal_location[1]:.1f})")
        print(f"[GLOBAL] Distance: {total_distance:.1f}m")

        return self.waypoints

    def get_current_waypoint(self, vehicle_pos: np.ndarray) -> Optional[GlobalWaypoint]:
        if not self.path_computed or len(self.waypoints) == 0:
            return None

        dist_to_goal = np.linalg.norm(vehicle_pos[:2] - self.goal_location[:2])
        if dist_to_goal < self.cfg.goal_tolerance:
            self.goal_reached = True
            print(f"[GLOBAL] 🎯 GOAL REACHED! Distance: {dist_to_goal:.2f}m")
            return None

        while self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            wp_pos = np.array([wp.x, wp.y, wp.z])
            dist_to_wp = np.linalg.norm(vehicle_pos - wp_pos)

            if dist_to_wp < self.waypoint_spacing * 0.5:
                if not wp.reached:
                    print(f"[GLOBAL] Waypoint {self.current_waypoint_idx} reached (dist: {dist_to_wp:.1f}m)")
                wp.reached = True
                self.current_waypoint_idx += 1
            else:
                return wp

        return GlobalWaypoint(
            x=self.goal_location[0],
            y=self.goal_location[1],
            z=self.goal_location[2],
            yaw=-82.22
        )

    def get_lookahead_waypoint(self, vehicle_pos: np.ndarray) -> Optional[np.ndarray]:
        if not self.path_computed or len(self.waypoints) == 0:
            return None

        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            wp_pos = np.array([wp.x, wp.y, wp.z])
            dist = np.linalg.norm(vehicle_pos - wp_pos)

            if dist >= self.cfg.lookahead_distance:
                return np.array([wp.x, wp.y, wp.z])

        return self.goal_location.copy()

    def check_replan_needed(self, vehicle_pos: np.ndarray, t: float) -> bool:
        if not self.path_computed or len(self.waypoints) == 0:
            return False

        if self.current_waypoint_idx >= len(self.waypoints):
            return False

        current_wp = self.waypoints[self.current_waypoint_idx]
        wp_pos = np.array([current_wp.x, current_wp.y, current_wp.z])
        dist_to_wp = np.linalg.norm(vehicle_pos - wp_pos)

        if dist_to_wp > self.waypoint_spacing * 2:
            return True

        return False

    def reset(self, start_pos: np.ndarray, start_yaw: float) -> None:
        self.goal_reached = False
        self.current_waypoint_idx = 0
        self.path_computed = False
        self.compute_path(start_pos, start_yaw)

    def is_goal_reached(self) -> bool:
        return self.goal_reached


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §8  CONTROLLER MODULE
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
            1.0 - terrain_slope_deg / self.cfg.max_slope_deg * 0.5
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
        """FIXED: Proper manual control with correct direction handling."""
        # Positive steer = turn right, Negative steer = turn left (in vehicle frame)
        # Positive vx = forward, Negative vx = reverse
        
        is_reverse = vx < 0.0
        abs_vx = abs(vx)
        
        # Throttle based on speed magnitude
        throttle = np.clip(abs_vx / self.cfg.base_speed, 0.0, self.cfg.max_throttle)
        
        return self._apply_smooth(throttle, steer, 0.0, is_reverse)

    def compute_recovery(self, phase: int, direction_hint: Optional[str] = None) -> ControlState:
        if direction_hint == 'left' or direction_hint == 'right':
            steer = 0.6 if direction_hint == 'left' else -0.6
        elif direction_hint == 'toward_global':
            steer = 0.0
        else:
            if phase == 0:
                steer = 0.0
            elif phase == 1:
                steer = 0.7
            else:
                steer = -0.7

        if phase == 0:
            return ControlState(throttle=0.45, steer=steer, brake=0.0, reverse=True)
        elif phase == 1:
            return ControlState(throttle=0.5, steer=steer, brake=0.0, reverse=True)
        else:
            return ControlState(throttle=0.4, steer=steer, brake=0.0, reverse=True)

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
            corrective_steer = -np.sign(roll_deg) * min(
                self.cfg.roll_corrective_steer * (excess / 10.0),
                self.cfg.roll_corrective_steer
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
        self._steer_ema = a * steer + (1 - a) * self._steer_ema

        out_throttle = self._speed_ema
        out_steer = self._steer_ema
        out_brake = brake

        if brake > 0.1:
            out_throttle = 0.0
            out_brake = brake

        max_dr = self.cfg.max_steer_rate
        max_dt = self.cfg.max_throttle_rate
        out_steer = float(np.clip(out_steer,
                                  self._state.steer - max_dr,
                                  self._state.steer + max_dr))
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
# §9  STUCK DETECTION & RECOVERY
# ╚═══════════════════════════════════════════════════════════════════════════╝

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
        self._last_stuck_time: float = 0.0
        self._consecutive_stuck_count: int = 0
        self._recovery_direction: Optional[str] = None
        self._stuck_callbacks: list = []

    def register_callback(self, callback) -> None:
        """Register callback to be called when stuck is detected."""
        self._stuck_callbacks.append(callback)

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
        self._pos_history.append((t, pos.copy()))
        self._steer_history.append(steer)

        if self._state == self.RECOVERING:
            elapsed = t - self._recovery_start
            durations = self.cfg.recovery_phase_durations
            
            phase_end_times = [durations[0], durations[0] + durations[1], 
                              durations[0] + durations[1] + durations[2]]
            
            if elapsed >= sum(durations):
                print(f"[STUCK] Recovery complete. Resetting to IDLE.")
                self._state = self.IDLE
                self._recovery_phase = 0
                self._recovery_direction = None
                self._pos_history.clear()
                return False, 0
            
            if elapsed < phase_end_times[0]:
                self._recovery_phase = 0
            elif elapsed < phase_end_times[1]:
                self._recovery_phase = 1
            else:
                self._recovery_phase = 2
            
            return True, self._recovery_phase

        is_stuck = False
        stuck_reason = ""

        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    is_stuck = True
                    stuck_reason = f"low displacement ({disp:.2f}m < {self.cfg.stuck_disp_thresh}m)"
        
        if len(self._steer_history) >= self._steer_history.maxlen:
            arr = np.array(list(self._steer_history))
            signs = np.sign(arr + 1e-9)
            flips = int(np.sum(np.diff(signs) != 0))
            flip_ratio = flips / (len(signs) - 1) if len(signs) > 1 else 0
            
            if flip_ratio > self.cfg.stuck_oscillation_thresh:
                is_stuck = True
                stuck_reason = f"steering oscillation ({flip_ratio:.1%})"

        if is_stuck:
            if t - self._last_stuck_time < 3.0:
                self._consecutive_stuck_count += 1
            else:
                self._consecutive_stuck_count = 1
            
            self._last_stuck_time = t
            
            if self._consecutive_stuck_count >= 2:
                self._recovery_direction = self._determine_escape_direction()
            else:
                self._recovery_direction = "left" if self._recovery_phase % 2 == 0 else "right"
            
            self._trigger_recovery(t, stuck_reason)
            return True, 0

        return False, 0

    def _check_oscillation(self) -> bool:
        if len(self._steer_history) < 10:
            return False
        
        arr = np.array(list(self._steer_history))
        signs = np.sign(arr + 1e-9)
        flips = np.sum(np.diff(signs) != 0)
        
        return flips > len(signs) * self.cfg.stuck_oscillation_thresh

    def _determine_escape_direction(self) -> str:
        if len(self._steer_history) < 5:
            return "left"
        
        arr = np.array(list(self._steer_history))
        mean_steer = np.mean(arr)
        
        if mean_steer > 0.2:
            return "left"
        elif mean_steer < -0.2:
            return "right"
        else:
            return "straight"

    def _trigger_recovery(self, t: float, reason: str) -> None:
        if self._state == self.IDLE:
            print(f"[STUCK] Recovery triggered! Reason: {reason}")
            print(f"[STUCK] Direction hint: {self._recovery_direction}, Count: {self._consecutive_stuck_count}")
            self._state = self.RECOVERING
            self._recovery_start = t
            self._recovery_phase = 0
            
            # Call registered callbacks (e.g., to add to memory lane)
            for callback in self._stuck_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[STUCK] Callback error: {e}")

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING

    def get_recovery_direction(self) -> Optional[str]:
        return self._recovery_direction

    def reset(self) -> None:
        self._state = self.IDLE
        self._recovery_phase = 0
        self._recovery_direction = None
        self._consecutive_stuck_count = 0
        self._pos_history.clear()
        self._steer_history.clear()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §10  VISUALIZATION MODULE (Fixed Colors)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._data_lock = threading.Lock()
        self._subgoal = None
        self._vehicle_pos = None
        self._speed_ms = 0.0
        self._steer = 0.0
        self._perc = None
        self._mode = "AUTO"
        self._front_img = None
        self._rear_img = None
        self._lidar_pts = None
        self._global_path = None
        self._goal_location = None
        self._is_stuck = False
        self._recovery_phase = 0

        maxlen = cfg.traj_history
        # FIX: Store proper cost breakdown for visualization
        self._cost_total: deque = deque(maxlen=maxlen)
        self._cost_dir: deque = deque(maxlen=maxlen)
        self._cost_dist: deque = deque(maxlen=maxlen)
        self._cost_steep: deque = deque(maxlen=maxlen)
        self._cost_progress: deque = deque(maxlen=maxlen)   # NEW
        self._cost_memory: deque = deque(maxlen=maxlen)      # NEW
        self._vel_lin: deque = deque(maxlen=maxlen)
        self._vel_ang: deque = deque(maxlen=maxlen)
        self._traj_x: deque = deque(maxlen=maxlen)
        self._traj_y: deque = deque(maxlen=maxlen)
        self._subgoal_x: deque = deque(maxlen=20)
        self._subgoal_y: deque = deque(maxlen=20)

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
        global_path: Optional[List[GlobalWaypoint]] = None,
        goal_location: Optional[np.ndarray] = None,
        is_stuck: bool = False,
        recovery_phase: int = 0,
    ) -> None:
        with self._data_lock:
            self._traj_x.append(float(vehicle_pos[0]))
            self._traj_y.append(float(vehicle_pos[1]))

            if subgoal is not None:
                self._cost_total.append(max(0.0, subgoal.cost))
                self._cost_dir.append(subgoal.cost_direction)
                self._cost_dist.append(
                    abs(subgoal.distance - self.cfg.subgoal_distance) / self.cfg.subgoal_distance * self.cfg.w_distance)
                self._cost_steep.append(subgoal.cost_steepness)
                self._cost_progress.append(subgoal.cost_progress)
                self._cost_memory.append(subgoal.cost_memory)
                self._subgoal_x.append(vehicle_pos[0] + subgoal.local_pos[0])
                self._subgoal_y.append(vehicle_pos[1] + subgoal.local_pos[1])
            else:
                self._cost_total.append(0.0)
                self._cost_dir.append(0.0)
                self._cost_dist.append(0.0)
                self._cost_steep.append(0.0)
                self._cost_progress.append(0.0)
                self._cost_memory.append(0.0)

            self._vel_lin.append(speed_ms)
            self._vel_ang.append(steer * self.cfg.base_speed)

            if perc is not None and perc.raw_points is not None:
                self._lidar_pts = perc.raw_points.copy()

            self._subgoal = subgoal
            self._vehicle_pos = vehicle_pos.copy()
            self._speed_ms = speed_ms
            self._steer = steer
            self._mode = mode
            self._global_path = global_path
            self._goal_location = goal_location
            self._is_stuck = is_stuck
            self._recovery_phase = recovery_phase

    def set_front_image(self, arr: np.ndarray) -> None:
        with self._data_lock:
            self._front_img = arr

    def set_rear_image(self, arr: np.ndarray) -> None:
        with self._data_lock:
            self._rear_img = arr

    def _render_loop(self) -> None:
        plt.ion()
        fig = plt.figure("VSGP Navigator", figsize=(18, 9))
        fig.patch.set_facecolor("#111")
        gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

        ax_front = fig.add_subplot(gs[0, 0])
        ax_lidar = fig.add_subplot(gs[0, 1])
        ax_rear = fig.add_subplot(gs[0, 2])
        ax_cost = fig.add_subplot(gs[0, 3])  # Fixed: cost now gets full column
        ax_traj = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_progress = fig.add_subplot(gs[1, 2:])  # NEW: Progress panel

        for ax in fig.get_axes():
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#ddd")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")

        last_update = 0.0
        interval = self._interval

        while self._running:
            now = time.time()
            if (now - last_update) < interval:
                time.sleep(0.01)
                continue
            last_update = now

            try:
                with self._data_lock:
                    self._draw_camera(ax_front, self._front_img, "Front Camera")
                    self._draw_camera(ax_rear, self._rear_img, "Rear Camera")
                    self._draw_lidar(ax_lidar)
                    self._draw_cost(ax_cost)
                    self._draw_trajectory(ax_traj)
                    self._draw_velocity(ax_vel)
                    self._draw_progress(ax_progress)

                status = f"Mode: {self._mode}"
                if self._is_stuck:
                    status += f" | STUCK - Phase {self._recovery_phase}"
                
                fig.suptitle(
                    f"VSGP Mapless Navigator  |  {status}",
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
            z = pts[:, 2]
            z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)
            ax.scatter(pts[:, 1], pts[:, 0], c=z_n, cmap="plasma", s=0.5, alpha=0.6)
            ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
            ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
        ax.set_xlabel("Y [m]", fontsize=7)
        ax.set_ylabel("X [m]", fontsize=7)

    def _draw_cost(self, ax):
        """FIXED: Proper cost function visualization with correct colors and scaling."""
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Cost Function (weighted components)", fontsize=8)
        
        n = len(self._cost_total)
        if n == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                   transform=ax.transAxes, color="#888", fontsize=10)
            return
        
        xs = np.arange(n)
        
        # FIX: Use actual weighted costs (not raw normalized values)
        # Total = direction + steepness + memory + progress (all already weighted)
        total_vals = list(self._cost_total)
        dir_vals = list(self._cost_dir)
        steep_vals = list(self._cost_steep)
        progress_vals = list(self._cost_progress)
        memory_vals = list(self._cost_memory)
        
        # Plot with proper colors - total should be SUM of others
        ax.plot(xs, total_vals, c="#ff4757", lw=2.0, label="Total Cost", zorder=5)
        ax.plot(xs, steep_vals, c="#ff6b81", lw=1.2, label="Steepness", linestyle='--')
        ax.plot(xs, dir_vals, c="#ffa502", lw=1.0, label="Direction", linestyle=':')
        ax.plot(xs, memory_vals, c="#7bed9f", lw=1.0, label="Memory Penalty", linestyle='-.')
        ax.plot(xs, progress_vals, c="#70a1ff", lw=1.0, label="Progress Bonus")
        
        ax.legend(fontsize=6, loc="upper right", ncol=2,
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Cost (weighted)", fontsize=7)
        ax.grid(True, alpha=0.2)

    def _draw_trajectory(self, ax):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Trajectory & Global Path", fontsize=8)
        tx, ty = list(self._traj_x), list(self._traj_y)
        if len(tx) > 1:
            color = "#e17055" if self._is_stuck else "#00cec9"
            ax.plot(ty, tx, c=color, lw=1.2, label="Actual")
        if len(tx) > 0:
            ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)
        sx, sy = list(self._subgoal_x), list(self._subgoal_y)
        if sx:
            ax.scatter(sy, sx, c="#e17055", s=12, marker="x", zorder=4, label="Subgoals")

        if self._global_path is not None and len(self._global_path) > 0:
            global_x = [wp.y for wp in self._global_path]
            global_y = [wp.x for wp in self._global_path]
            ax.plot(global_x, global_y, c="#a29bfe", lw=2.0,
                   linestyle='--', label="Global Path", alpha=0.7)
            reached_x = [wp.y for wp in self._global_path if wp.reached]
            reached_y = [wp.x for wp in self._global_path if wp.reached]
            if reached_x:
                ax.scatter(reached_x, reached_y, c="#00b894", s=8,
                          marker='o', label="Reached WP", alpha=0.5)

        if self._goal_location is not None:
            ax.scatter([self._goal_location[1]], [self._goal_location[0]],
                      c="#ff7675", s=50, marker='*', label="GOAL", zorder=10)

        ax.legend(fontsize=6, loc="upper left", facecolor="#222", edgecolor="#555", labelcolor="#ddd")
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
        ax.legend(fontsize=6, loc="upper left", facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("m/s", fontsize=7)
        ax.grid(True, alpha=0.2)

    def _draw_progress(self, ax):
        """NEW: Draw progress toward goal."""
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Progress Metrics", fontsize=8)
        
        # We don't have dedicated progress history, but can show cost breakdown
        n = len(self._cost_total)
        if n == 0:
            ax.text(0.5, 0.5, "Initializing...", ha='center', va='center',
                   transform=ax.transAxes, color="#888", fontsize=10)
            return
        
        xs = np.arange(n)
        
        # Show stacked area for cost components
        steep_vals = np.array(list(self._cost_steep))
        dir_vals = np.array(list(self._cost_dir))
        mem_vals = np.array(list(self._cost_memory))
        
        # Ensure values are positive for stacking
        steep_vals = np.maximum(steep_vals, 0)
        dir_vals = np.maximum(dir_vals, 0)
        mem_vals = np.maximum(mem_vals, 0)
        
        ax.fill_between(xs, 0, steep_vals, alpha=0.5, color="#ff6b81", label="Steepness")
        ax.fill_between(xs, steep_vals, steep_vals + dir_vals, alpha=0.5, color="#ffa502", label="Direction")
        ax.fill_between(xs, steep_vals + dir_vals, steep_vals + dir_vals + mem_vals, 
                       alpha=0.5, color="#7bed9f", label="Memory")
        
        ax.plot(xs, list(self._cost_total), c="#ff4757", lw=1.5, label="Total")
        
        ax.legend(fontsize=6, loc="upper left", ncol=2,
                  facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Cumulative Cost", fontsize=7)
        ax.grid(True, alpha=0.2)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §11  CARLA SENSOR WRAPPERS
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg = cfg
        self.vehicle = vehicle
        self.world = world
        self._actors: list = []

        self._lidar_queue: Queue = Queue(maxsize=1)
        self._front_queue: Queue = Queue(maxsize=1)
        self._rear_queue: Queue = Queue(maxsize=1)

        bp_lib = world.get_blueprint_library()

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
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
        cam_bp.set_attribute("fov", str(self.cfg.camera_fov))
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
# §12  MANUAL KEYBOARD CONTROLLER (FIXED)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ManualController:
    """
    FIXED: Proper manual control with correct key mappings.
    W = Forward (positive velocity)
    S = Reverse (negative velocity)
    A = Turn Left (negative steer in CARLA = turn left when facing forward)
    D = Turn Right (positive steer in CARLA = turn right when facing forward)
    SPACE = Center steering
    """
    VX_SPEED = 3.0
    STEER_SPEED = 0.5

    def __init__(self, cfg: Config = CFG):
        self.ctrl = Controller(cfg)
        self._steer = 0.0
        self._vx = 0.0
        self.cfg = cfg

    def tick(self, keys) -> ControlState:
        vx = 0.0
        reverse = False
        
        # FIX: W/S for forward/reverse with proper handling
        if keys[pygame.K_w]:
            vx = self.VX_SPEED       # Forward
            reverse = False
        if keys[pygame.K_s]:
            vx = -self.VX_SPEED      # Reverse
            reverse = True
        
        # FIX: A/D for steering - in CARLA, steer > 0 = turn right, < 0 = turn left
        # W key moves forward (positive Y in CARLA coordinates)
        # A key should turn left = negative steer
        # D key should turn right = positive steer
        if keys[pygame.K_a]:
            self._steer = np.clip(self._steer - self.STEER_SPEED,
                                 -self.cfg.max_steer, self.cfg.max_steer)
        if keys[pygame.K_d]:
            self._steer = np.clip(self._steer + self.STEER_SPEED,
                                 -self.cfg.max_steer, self.cfg.max_steer)
        
        # SPACE to center steering
        if keys[pygame.K_SPACE]:
            self._steer = 0.0
            vx = 0.0
        
        # Use compute_manual with proper steering and reverse flag
        return self.ctrl.compute_manual(vx, self._steer)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §13  VEHICLE SPAWNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    spawn_transform = carla.Transform(
        carla.Location(x=906.66, y=-875.78, z=4.7620),
        carla.Rotation(pitch=-4.50, yaw=88.0, roll=0.0),
    )

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

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
# §14  MAIN NAVIGATION LOOP
# ╚═══════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._running = False
        self._mode = "AUTO"

        self._client: Optional[carla.Client] = None
        self._world = None
        self._vehicle = None
        self._sensors: Optional[SensorManager] = None

        self._perception = PerceptionModule(cfg)
        self._planner = VSGPSubgoalPlanner(cfg)
        self._controller = Controller(cfg)
        self._manual_ctrl = ManualController(cfg)
        self._stuck = StuckDetector(cfg)
        self._viz = VisualizationModule(cfg)
        self._global_planner = GlobalPlanner(cfg)
        
        # NEW: Memory lane and progress tracker
        self._memory_lane = MemoryLane(cfg)
        self._progress_tracker = ProgressTracker(cfg, np.array(cfg.goal_location))
        
        # Connect memory lane to stuck detector
        self._stuck.register_callback(self._on_stuck_detected)
        
        # Connect to planner
        self._planner.set_memory(self._memory_lane)
        self._planner.set_progress_tracker(self._progress_tracker)

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit, R=reset stuck")
        self._screen = pygame.display.set_mode((280, 110))
        self._font = pygame.font.SysFont("monospace", 13)

        self._step = 0
        self._t0 = 0.0

    def _on_stuck_detected(self) -> None:
        """Callback when stuck is detected - add to memory lane."""
        loc = self._vehicle.get_location()
        pos = np.array([loc.x, loc.y, loc.z])
        self._memory_lane.add_stuck_location(pos)
        self._progress_tracker.on_stuck()

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

        loc = self._vehicle.get_location()
        transform = self._vehicle.get_transform()
        start_pos = np.array([loc.x, loc.y, loc.z])
        start_yaw = math.radians(transform.rotation.yaw)

        if self.cfg.global_planner_enabled:
            self._global_planner.reset(start_pos, start_yaw)
            print(f"[GLOBAL] Global planner initialized")

        # NEW: Initialize progress tracker
        self._progress_tracker.initialize(start_pos)
        self._memory_lane.add_position(start_pos)

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

        ctrl_state: ControlState = ControlState()
        best_goal: Optional[Subgoal] = None
        is_stuck = False
        recovery_phase = 0

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
                    self._stuck.reset()
                    print(f"[MODE] Switched to {self._mode}")
                if event.key == pygame.K_r:
                    self._stuck.reset()
                    self._memory_lane.clear_old_entries(30.0)  # Clear recent memory
                    print("[MODE] Stuck state and recent memory reset")

        keys = pygame.key.get_pressed()

        loc = self._vehicle.get_location()
        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        pos = np.array([loc.x, loc.y, loc.z])

        transform = self._vehicle.get_transform()
        vehicle_yaw = math.radians(transform.rotation.yaw)
        pitch_rad = math.radians(transform.rotation.pitch)
        roll_rad = math.radians(transform.rotation.roll)
        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)

        # Track position in memory lane
        self._memory_lane.add_position(pos)

        # Clear old memory entries periodically
        if self._step % 100 == 0:
            self._memory_lane.clear_old_entries(120.0)

        if self.cfg.global_planner_enabled and self._global_planner.is_goal_reached():
            print("[NAV] 🎯 GOAL REACHED! Stopping vehicle.")
            self._vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
            self._running = False
            return

        # Perception
        lidar_data = self._sensors.lidar_data
        if lidar_data is not None:
            perc = self._perception.process_lidar(lidar_data)
        else:
            perc = self._perception.last_output

        # Global Planner Integration
        global_waypoint = None
        global_path = None
        if self.cfg.global_planner_enabled and self._mode == "AUTO":
            global_waypoint = self._global_planner.get_lookahead_waypoint(pos)
            global_path = self._global_planner.waypoints

        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)
        else:
            stuck, rec_phase = self._stuck.update(pos, ctrl_state.steer, t_now)
            is_stuck = stuck
            recovery_phase = rec_phase

            if stuck:
                print(f"[NAV] 🚗 STUCK DETECTED! Recovery phase {rec_phase}")
                
                recovery_direction = self._stuck.get_recovery_direction()
                ctrl_state = self._controller.compute_recovery(rec_phase, recovery_direction)
                
                if perc is not None:
                    if rec_phase == 1 and recovery_direction:
                        escape_mode = "escape_left" if recovery_direction == "left" else "escape_right"
                        escape_goal, _ = self._planner.plan(
                            perc, pitch_rad, transform, 
                            global_waypoint, mode=escape_mode
                        )
                        if escape_goal is not None and abs(escape_goal.alpha) > 0.5:
                            print(f"[NAV] Using escape goal: alpha={escape_goal.alpha:.2f}")
                            ctrl_state = self._controller.compute(
                                escape_goal, speed, 0.0, roll_deg, pitch_deg
                            )
            else:
                if perc is not None:
                    best_goal, _ = self._planner.plan(
                        perc, pitch_rad, transform,
                        global_waypoint, mode="forward"
                    )

                if best_goal:
                    ctrl_state = self._controller.compute(
                        best_goal, speed, best_goal.slope_deg,
                        roll_deg=roll_deg, pitch_deg=pitch_deg,
                    )
                else:
                    print("[NAV] ⚠️ No valid path found!")
                    
                    if speed < 0.5:
                        self._stuck.update(pos, ctrl_state.steer, t_now)
                    
                    if global_waypoint is not None:
                        direction = global_waypoint[:2] - pos[:2]
                        dist_to_goal_dir = np.linalg.norm(direction)
                        if dist_to_goal_dir > 0.1:
                            direction = direction / dist_to_goal_dir
                            alpha = math.atan2(direction[1], direction[0]) - vehicle_yaw
                            while alpha > math.pi:
                                alpha -= 2 * math.pi
                            while alpha < -math.pi:
                                alpha += 2 * math.pi
                            
                            fallback_goal = Subgoal(
                                alpha=alpha,
                                beta=0.0,
                                distance=2.0,
                                local_pos=np.array([2.0 * np.cos(alpha), 2.0 * np.sin(alpha), 0.0]),
                                slope_deg=0.0,
                                safe=False,
                                width_m=self.cfg.vehicle_width,
                                cost=999.0,
                            )
                        else:
                            fallback_goal = None
                    else:
                        fallback_goal = None
                    
                    if fallback_goal:
                        ctrl_state = self._controller.compute(
                            fallback_goal, speed, 0.0,
                            roll_deg=roll_deg, pitch_deg=pitch_deg,
                        )
                    else:
                        ctrl_state = self._controller.compute(
                            None, speed, 0.0,
                            roll_deg=roll_deg, pitch_deg=pitch_deg,
                        )

        # Apply Control
        if self.cfg.invert_drive:
            applied_steer = -ctrl_state.steer
            applied_reverse = not ctrl_state.reverse
        else:
            applied_steer = ctrl_state.steer
            applied_reverse = ctrl_state.reverse

        self._vehicle.apply_control(carla.VehicleControl(
            throttle=float(ctrl_state.throttle),
            steer=float(applied_steer),
            brake=float(ctrl_state.brake),
            reverse=applied_reverse,
        ))

        # Pygame HUD
        goal_dist = np.linalg.norm(pos[:2] - np.array(self.cfg.goal_location[:2]))
        progress_pct = self._progress_tracker.get_overall_progress()
        stuck_count = self._memory_lane.get_stuck_count_near(pos)
        
        self._screen.fill((20, 20, 40))
        mode_color = (255, 100, 100) if is_stuck else (200, 220, 255)
        status = "STUCK!" if is_stuck else self._mode
        lines = [
            f"Mode: {status}",
            f"Speed: {speed:.1f} m/s  Rev: {ctrl_state.reverse}",
            f"Thr:{ctrl_state.throttle:.2f}  Str:{ctrl_state.steer:.2f}",
            f"Roll:{roll_deg:+.1f}°  Pitch:{pitch_deg:+.1f}°",
            f"Goal: {goal_dist:.0f}m  Progress: {progress_pct:.1f}%",
            f"Step: {self._step}  Stuck Memories: {stuck_count}",
        ]
        for i, line in enumerate(lines):
            color = mode_color if i == 0 else (200, 220, 255)
            surf = self._font.render(line, True, color)
            self._screen.blit(surf, (5, 4 + i * 18))
        pygame.display.flip()

        # Push to visualization
        self._viz.push_data(
            subgoal=best_goal if self._mode == "AUTO" else None,
            vehicle_pos=pos,
            speed_ms=speed,
            steer=ctrl_state.steer,
            perc=perc,
            mode=self._mode,
            global_path=global_path,
            goal_location=np.array(self.cfg.goal_location),
            is_stuck=is_stuck,
            recovery_phase=recovery_phase,
        )
        front_img = self._sensors.front_image
        rear_img = self._sensors.rear_image
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
