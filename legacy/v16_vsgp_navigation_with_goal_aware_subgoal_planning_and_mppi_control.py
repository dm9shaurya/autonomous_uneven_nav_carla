from __future__ import annotations

import math
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from queue import Queue, Empty

import numpy as np
import pygame

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
# PyTorch (required)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available – running on CPU")
except ImportError:
    print("[FATAL] PyTorch not found.  pip install torch")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CARLA (required)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print("[FATAL] carla package not found.  Put CARLA egg on PYTHONPATH.")
    sys.exit(1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── CARLA
    carla_host: str              = "127.0.0.1"
    carla_port: int              = 2000
    carla_timeout: float         = 20.0
    synchronous: bool            = True
    fixed_delta_seconds: float   = 0.05

    # ── Vehicle
    vehicle_blueprint: str       = "vehicle.tesla.cybertruck"
    post_spawn_wait_s: float    = 3.0

    # ── Geometry
    wheel_base: float            = 3.807
    vehicle_width: float        = 2.0
    vehicle_length: float       = 4.5    # for clearance checks

    # ── Sensors
    lidar_range: float          = 15.0
    lidar_points_per_sec: int    = 80_000
    lidar_rotation_freq: float   = 20.0
    lidar_channels: int         = 64
    lidar_upper_fov: float       = 15.0
    lidar_lower_fov: float       = -25.0
    camera_width: int            = 320
    camera_height: int           = 240
    camera_fov: int              = 90

    # ── Goal
    # goal_x: float                = -164.10
    # goal_y: float                = 127.96
    # goal_tolerance: float        = 2.18


    goal_x: float = 770.10
    goal_y: float = -287.96
    goal_tolerance: float = 5.0

#     Location: x=-164.10, y=127.96, z=2.18
# Rotation: pitch=-1.01, yaw=-105.50, roll=0.00


    # ── Subgoal planning
    subgoal_distance: float      = 20.0    # look-ahead horizon (m)
    subgoal_min_distance: float  = 2.0    # minimum before this is ignored
    subgoal_num_angles: int      = 20    # rays to scan
    subgoal_num_depth: int       = 10     # samples per ray
    subgoal_clearance_margin: float = 0.8  # extra safety buffer (m)

    # ── MPPI
    mppi_horizon: int            = 20
    mppi_num_samples: int        = 1024
    mppi_lambda: float           = 1.0
    mppi_noise_throttle: float   = 0.2
    mppi_noise_steer: float      = 0.3
    safety_margin: float         = 4.00

    # ── Cost weights
    w_goal: float               = 15.0
    w_heading: float            = 4.0
    w_terrain_risk: float       = 8.0
    w_memory: float             = 30.0
    w_learned: float            = 0.0

    # ── Stuck memory
    memory_decay: float         = 0.995
    memory_radius: float        = 8.0
    memory_threshold: float     = 0.05

    # ── Control limits
    max_throttle: float         = 0.55
    max_steer: float            = 0.9
    max_brake: float            = 0.8

    # ── AUTO control smoothing
    ema_alpha: float            = 0.3
    max_steer_rate: float       = 0.15
    max_throttle_rate: float   = 0.08
    steer_penalty: float        = 0.2
    throttle_penalty: float    = 0.1
    base_speed: float           = 3.0
    kp_throttle: float          = 0.5
    kp_steer: float             = 1.5

    # ── Stability
    max_safe_roll_deg: float    = 10.0
    max_safe_pitch_deg: float    = 15.0
    stability_throttle_scale: float = 0.4
    roll_corrective_steer: float    = 0.5

    # ── Stuck detection
    stuck_window_s: float       = 2.0
    stuck_disp_thresh: float    = 0.6
    recovery_duration_s: float = 2.5

    # ── VSGP
    vsgp_n_inducing: int        = 50
    vsgp_alpha: float           = 1.0
    vsgp_length_scale: float    = 0.3
    vsgp_noise_var: float       = 0.05
    vsgp_lr: float              = 0.02
    vsgp_update_freq: int       = 10

    # ── Reactive avoidance
    react_warn_dist: float      = 10.0
    react_danger_dist: float    = 6.0
    react_emergency_dist: float = 3.8
    react_forward_cone: float   = 50.0

    # ── Visualization
    viz_win_w: int              = 1280
    viz_win_h: int              = 720
    viz_every_n_ticks: int      = 3
    traj_history: int           = 600


CFG = Config()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §1  DATA CLASSES
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ControlState:
    throttle: float = 0.0
    steer: float    = 0.0
    brake: float    = 0.0
    reverse: bool   = False


@dataclass
class PerceptionOutput:
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS
# ╚═══════════════════════════════════════════════════════════════════════════╝

class RationalQuadraticKernel:
    def __init__(self, alpha: float = 1.0, length_scale: float = 0.3, variance: float = 1.0):
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
    """Variational Sparse Gaussian Process for terrain height regression."""

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.kernel = RationalQuadraticKernel(
            alpha=cfg.vsgp_alpha,
            length_scale=cfg.vsgp_length_scale,
        )
        self.noise_var    = cfg.vsgp_noise_var
        self.n_ind        = cfg.vsgp_n_inducing
        self.lr           = cfg.vsgp_lr
        self.update_freq  = cfg.vsgp_update_freq

        side = int(math.sqrt(self.n_ind))
        A, B = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, side),
            np.linspace(-np.pi / 4, np.pi / 4, side),
        )
        self.Z  = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]
        m = len(self.Z)
        self.mu = np.zeros(m)
        self.Su = np.eye(m) * 0.1
        self._Kuu_inv: Optional[np.ndarray] = None
        self._trained      = False
        self._update_count = 0
        self._last_X_hash  = 0

    def _select_inducing(self, X: np.ndarray) -> None:
        n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50, random_state=0)
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
        Kfu  = self.kernel(X, self.Z)
        A    = Kfu @ self._Kuu_inv
        ni   = 1.0 / self.noise_var
        Lambda = ni * (A.T @ A) + self._Kuu_inv
        rhs  = ni * (A.T @ y)
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
        Ksu      = self.kernel(Xs, self.Z)
        A        = Ksu @ self._Kuu_inv
        mean     = A @ self.mu
        Kss_diag = self.kernel.diag(Xs)
        var_exp  = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
        var_var  = np.sum(A @ self.Su * A, axis=1)
        var = np.clip(Kss_diag - var_exp + var_var + self.noise_var, 1e-6, None)
        return mean, var


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PerceptionModule:
    ALPHA_RES        = 60
    BETA_RES         = 30
    ROC              = 5.0
    SLOPE_WEIGHT     = 1.2
    OCC_WEIGHT       = 1.0
    VAR_FREE_WEIGHT  = 1.0
    ROUGHNESS_WEIGHT = 0.8

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
        return float(np.clip(np.degrees(np.arccos(np.clip(cos_ang, -1, 1))), 0, 90))

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(raw) < 20:
            return self._last_output
        dist = np.linalg.norm(raw, axis=1)
        pts = raw[(dist > 0.5) & (dist < self.cfg.lidar_range)]
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
            X, y, pts = X[idx], y[idx], pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)
        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var = var.reshape(self.BETA_RES, self.ALPHA_RES)

        da = np.gradient(mean, axis=1)
        db = np.gradient(mean, axis=0)
        slope = np.sqrt(da ** 2 + db ** 2)
        roughness = np.sqrt(np.gradient(da, axis=1) ** 2 + np.gradient(db, axis=0) ** 2)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-6)

        slope = _norm(slope)
        roughness = _norm(roughness)
        occ_norm = _norm(mean)
        var_norm = _norm(var)

        traversability = np.clip(
            1.0
            - self.OCC_WEIGHT       * occ_norm
            - self.SLOPE_WEIGHT     * slope
            - self.VAR_FREE_WEIGHT  * var_norm
            - self.ROUGHNESS_WEIGHT * roughness,
            0.0, 1.0,
        )

        out = PerceptionOutput(
            alpha_grid=self._AG,
            beta_grid=self._BG,
            occupancy_mean=mean,
            occupancy_var=var,
            slope_map=slope,
            roughness_map=roughness,
            traversability=traversability,
            raw_points=pts,
            free_mask=traversability > 0.5,
            mean_surface=mean,
            ground_slope_deg=ground_slope,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §4  SUBGOAL DATA STRUCTURE
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Subgoal:
    """
    A candidate subgoal with full terrain context.
    """
    alpha: float                  # azimuth angle in vehicle frame (rad)
    beta: float                   # elevation angle in vehicle frame (rad)
    distance: float               # range (m)
    local_pos: np.ndarray          # (x, y, z) in vehicle-local frame
    world_pos: np.ndarray         # (x, y, z) in world frame

    # Terrain quality at this point
    slope: float                  # normalised slope [0, 1]
    roughness: float              # normalised roughness [0, 1]
    occupancy: float              # normalised occupancy [0, 1]
    variance: float              # normalised variance [0, 1]
    traversability: float        # normalised traversability [0, 1]

    # Scoring
    terrain_cost: float           # cost from terrain quality alone
    goal_progress: float         # how much closer this gets us to the goal (m)
    heading_error: float          # angular deviation from goal direction (rad)
    safe: bool                   # is this point safe to traverse?
    width_m: float
    cost: float = 0.0             # total weighted cost


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §5  GOAL-AWARE VSGP SUBGOAL PLANNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VSGPSubgoalPlanner:
    """
    Goal-aware subgoal planner using VSGP terrain perception.

    Strategy:
      1. Compute the desired heading from vehicle to global goal.
      2. Scan N rays spanning the forward view (constrained to ~±90°).
      3. For each ray, march outward from min_dist to max_dist.
         At each step sample the VSGP grid and check:
           • FREE    — occupancy below threshold
           • STABLE  — variance below threshold
           • SAFE    — slope & roughness below threshold
           • CLEAR   — roughness gradient small (no sudden drops)
      4. Terminate ray when any check fails (obstacle / unsafe terrain hit).
      5. Score every valid endpoint:
           score = w_terrain * terrain_cost
                 + w_goal_dist * (1 / (dist_to_goal + 1))
                 + w_heading   * |heading_error|
                 + w_slope     * slope
                 + w_rough     * roughness
      6. Return the lowest-cost candidate. If none → None (trigger recovery).

    The planner knows where the goal is and biases toward it while still
    responding to local terrain. The MPPI then tracks the selected subgoal.
    """

    # Terrain safety thresholds (normalised 0-1)
    OCC_FREE_THRESH   = 0.6
    VAR_STABLE_THRESH = 0.7
    SLOPE_SAFE_THRESH = 0.7
    ROUGH_SAFE_THRESH = 0.7

    # Scoring weights
    W_TERRAIN   = 3.0    # terrain quality
    W_GOAL_DIST = 2.0    # prefer points closer to goal
    W_HEADING   = 1.5    # penalise off-goal-heading angles
    W_SLOPE     = 2.0    # penalise steep terrain
    W_ROUGH     = 1.5    # penalise rough terrain
    W_OCC       = 2.0    # penalise occupied areas

    def __init__(self, cfg: Config = CFG):
        self.cfg            = cfg
        self.min_dist       = cfg.subgoal_min_distance
        self.max_dist       = cfg.subgoal_distance
        self.n_angles       = cfg.subgoal_num_angles
        self.n_depth        = cfg.subgoal_num_depth

    def plan(
        self,
        perc: PerceptionOutput,
        vehicle_pos: np.ndarray,
        vehicle_yaw_rad: float,
        goal_pos: np.ndarray,
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        """
        Returns (best_subgoal, all_candidates).
        """
        if perc.occupancy_mean is None:
            return None, []

        # ── Goal direction in vehicle frame ────────────────────────────────
        dx_world = goal_pos[0] - vehicle_pos[0]
        dy_world = goal_pos[1] - vehicle_pos[1]
        goal_yaw = math.atan2(dy_world, dx_world)
        # Relative angle from vehicle heading to goal
        goal_rel_yaw = self._norm_angle(goal_yaw - vehicle_yaw_rad)

        # ── Cache grid ─────────────────────────────────────────────────────
        alphas = perc.alpha_grid[0]        # (ALPHA_RES,)
        betas  = perc.beta_grid[:, 0]       # (BETA_RES,)
        mid_beta_idx = len(betas) // 2

        occ   = perc.occupancy_mean
        var   = perc.occupancy_var
        slope = perc.slope_map
        rough = perc.roughness_map
        trav  = perc.traversability

        # ── Scan rays ─────────────────────────────────────────────────────
        # Constrain to forward hemisphere; bias toward goal heading
        half_fov = math.pi / 2.2
        if abs(goal_rel_yaw) < half_fov:
            # Goal is within scan range — centre scan on goal direction
            scan_centre = goal_rel_yaw
        else:
            # Goal is behind; scan full forward range
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

            # ── March along this ray ────────────────────────────────────────
            depths = np.linspace(self.min_dist, self.max_dist, self.n_depth)

            valid_points: List[Tuple[float, float, float, float, float, float]] = []

            for d in depths:
                o_val = float(occ[mid_beta_idx, a_idx])
                v_val = float(var[mid_beta_idx, a_idx])
                s_val = float(slope[mid_beta_idx, a_idx])
                r_val = float(rough[mid_beta_idx, a_idx])
                t_val = float(trav[mid_beta_idx, a_idx])

                is_free   = o_val < self.OCC_FREE_THRESH
                is_stable = v_val < self.VAR_STABLE_THRESH
                is_safe   = s_val < self.SLOPE_SAFE_THRESH and r_val < self.ROUGH_SAFE_THRESH

                if not (is_free and is_stable and is_safe):
                    continue   # skip this point, but keep exploring

                valid_points.append((d, o_val, v_val, s_val, r_val, t_val))

            if not valid_points:
                continue

            # ── Pick the farthest valid point on this ray ──────────────────
            best_d, best_o, best_v, best_s, best_r, best_t = valid_points[-1]

            # ── Compute goal progress for this point ──────────────────────
            # World position of candidate
            lx = best_d * math.cos(a)
            ly = best_d * math.sin(a)

            wx = vehicle_pos[0] + (math.cos(vehicle_yaw_rad) * lx
                                   - math.sin(vehicle_yaw_rad) * ly)
            wy = vehicle_pos[1] + (math.sin(vehicle_yaw_rad) * lx
                                   + math.cos(vehicle_yaw_rad) * ly)

            dist_to_goal_from_cand = math.hypot(
                goal_pos[0] - wx, goal_pos[1] - wy)
            dist_to_goal_from_veh  = math.hypot(
                goal_pos[0] - vehicle_pos[0], goal_pos[1] - vehicle_pos[1])
            goal_progress = dist_to_goal_from_veh - dist_to_goal_from_cand

            # Heading error at this candidate
            cand_yaw = vehicle_yaw_rad + a
            cand_goal_yaw = math.atan2(goal_pos[1] - wy, goal_pos[0] - wx)
            heading_err = abs(self._norm_angle(cand_goal_yaw - vehicle_yaw_rad))

            # ── Terrain cost ─────────────────────────────────────────────
            terrain_cost = (
                self.W_SLOPE * best_s +
                self.W_ROUGH * best_r +
                self.W_OCC  * best_o +
                self.W_TERRAIN * (1.0 - best_t)
            )

            # ── Total score ────────────────────────────────────────────────
            # Higher score = worse.  We minimise.
            cost = (
                terrain_cost
                - self.W_GOAL_DIST * max(0.0, goal_progress)   # reward progress
                + self.W_HEADING   * heading_err               # penalise off-heading
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
                width_m=self.cfg.vehicle_width,
                cost=cost,
            )
            candidates.append(sg)

        if not candidates:
            return None, []

        # Sort by total cost (ascending)
        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates

    @staticmethod
    def _norm_angle(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while a > math.pi:  a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §6  REACTIVE OBSTACLE AVOIDANCE
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ReactiveObstacleAvoidance:
    """
    Last-resort layer — runs after MPPI. Directly reads raw LiDAR points
    and overrides throttle/steer/brake when an obstacle is imminently close.
    """

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
        angle   = np.degrees(np.arctan2(y, x))

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

        # Emergency
        if min_dist <= self.cfg.react_emergency_dist:
            out.throttle *= 0.5
            out.brake    = self.cfg.max_brake
            out.reverse  = True
            out.throttle = 0.3
            # out.steer += steer_dir * 0.2
            print(f"[REACTIVE] EMERGENCY STOP  dist={min_dist:.2f} m")

        # Danger
        elif min_dist <= self.cfg.react_danger_dist:
            ratio = (min_dist - self.cfg.react_emergency_dist) / (
                self.cfg.react_danger_dist - self.cfg.react_emergency_dist)
            out.throttle = ctrl.throttle * ratio * 0.2
            out.brake    = self.cfg.max_brake * (1.0 - ratio * 0.4)
            obs_y = float(fwd_pts[closest_idx, 1])
            steer_dir = -np.sign(obs_y) if abs(obs_y) > 0.1 else -np.sign(ctrl.steer + 1e-6)
            steer_mag = min(self.cfg.max_steer, 0.4 / (min_dist + 0.1))
            out.steer = float(np.clip(
                ctrl.steer + steer_dir * steer_mag,
                -self.cfg.max_steer, self.cfg.max_steer,
            ))
            print(f"[REACTIVE] Danger  dist={min_dist:.2f} m  steer_corr={steer_dir * steer_mag:+.3f}")

        # Warning
        else:
            ratio = (min_dist - self.cfg.react_danger_dist) / (
                self.cfg.react_warn_dist - self.cfg.react_danger_dist)
            out.throttle = ctrl.throttle * (0.3 + ratio * 0.7)
            out.brake    = ctrl.brake + (1.0 - ratio) * 0.25

        # Auto-reverse for very close obstacles
        if min_dist < 2.5 and not out.reverse:
            out.reverse  = True
            out.throttle = 0.3

        return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §7  LEARNED COST NETWORK
# ╚═══════════════════════════════════════════════════════════════════════════╝

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §8  STUCK MEMORY
# ╚═══════════════════════════════════════════════════════════════════════════╝

class StuckMemory:
    def __init__(self, cfg: Config):
        self.decay     = cfg.memory_decay
        self.radius    = cfg.memory_radius
        self.threshold = cfg.memory_threshold
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
        print(f"[MEMORY] Stuck at ({pos[0]:.1f},{pos[1]:.1f}) "
              f"type={failure_type}  total={len(self.points)}")

    def get_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.points:
            return None
        type_map = {"stall": 0.0, "slip": 1.0, "obstacle": 2.0}
        data = [[p[0], p[1], p[2], type_map.get(p[3], 0.0)] for p in self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §9  MPPI PLANNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MPPIPlanner:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon      = cfg.mppi_horizon
        self.num_samples  = cfg.mppi_num_samples
        self.lambda_      = cfg.mppi_lambda
        self.dt           = cfg.fixed_delta_seconds

        self._u_nom = torch.zeros((self.horizon, 2), device=self.device)
        self._noise = torch.zeros((self.num_samples, self.horizon, 2), device=self.device)
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
        self.goal    = torch.tensor(goal_xy, device=device).float()
        self.mem_pts = memory_points

        # Shift nominal sequence
        self._u_nom = torch.roll(self._u_nom, -1, dims=0)
        self._u_nom[-1] = self._u_nom[-2]

        # Sample noise
        self._noise[:, :, 0].normal_(0, self.cfg.mppi_noise_throttle)
        self._noise[:, :, 1].normal_(0, self.cfg.mppi_noise_steer)
        u = (self._u_nom.unsqueeze(0) + self._noise).clone()
        u[:, :, 0] = u[:, :, 0].clamp(0.0, self.cfg.max_throttle)
        u[:, :, 1] = u[:, :, 1].clamp(-self.cfg.max_steer, self.cfg.max_steer)

        occ   = torch.tensor(perc.occupancy_mean,  device=device).float()
        slope = torch.tensor(perc.slope_map,        device=device).float()
        rough = torch.tensor(perc.roughness_map,    device=device).float()
        var   = torch.tensor(perc.occupancy_var,   device=device).float()
        trav  = torch.tensor(perc.traversability,   device=device).float()

        costs = self._rollout(state, u, occ, slope, rough, var, trav)

        beta    = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum()

        self._u_nom = (weights[:, None, None] * u).sum(dim=0).detach()

        u0 = self._u_nom[0]
        ctrl = ControlState(
            throttle=float(u0[0].cpu()),
            steer   =float(u0[1].cpu()),
            brake   =0.0,
            reverse =False,
        )
        return ctrl, float(beta.cpu())

    def _rollout(self, state, u, occ, slope, rough, var, trav):
        K, T, _ = u.shape
        device  = self.device
        n_a, n_b = occ.shape[1], occ.shape[0]
        a_max    = np.pi / 2

        x = torch.tensor(state, device=device).float().unsqueeze(0).repeat(K, 1)
        veh_x, veh_y, veh_yaw = x[0, 0].item(), x[0, 1].item(), x[0, 2].item()
        total = torch.zeros(K, device=device)

        for t in range(T):
            throttle = u[:, t, 0]
            steer    = u[:, t, 1]

            dx = x[:, 0] - veh_x
            dy = x[:, 1] - veh_y
            ry = torch.atan2(
                torch.sin(torch.atan2(dy, dx) - veh_yaw),
                torch.cos(torch.atan2(dy, dx) - veh_yaw),
            )
            ai = ((ry + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi = torch.full_like(ry, n_b // 2, dtype=torch.long)
            sp_scale = torch.exp(-3.0 * (slope[bi, ai] + rough[bi, ai])).clamp(0.15, 1.0)

            v   = x[:, 3] + throttle * sp_scale * self.dt
            yr  = (v / self.cfg.wheel_base) * torch.tan(steer.clamp(-0.99, 0.99))
            yaw = x[:, 2] + yr * self.dt
            xp  = x[:, 0] + v * torch.cos(yaw) * self.dt
            yp  = x[:, 1] + v * torch.sin(yaw) * self.dt
            x   = torch.stack([xp, yp, yaw, v], dim=1)

            dx = xp - veh_x
            dy = yp - veh_y
            ry = torch.atan2(
                torch.sin(torch.atan2(dy, dx) - veh_yaw),
                torch.cos(torch.atan2(dy, dx) - veh_yaw),
            )
            ai = ((ry + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi = torch.full_like(ry, n_b // 2, dtype=torch.long)

            gdx   = self.goal[0] - xp
            gdy   = self.goal[1] - yp
            gdist = torch.sqrt(gdx ** 2 + gdy ** 2)
            gh    = torch.atan2(gdy, gdx)
            he    = torch.atan2(torch.sin(gh - yaw), torch.cos(gh - yaw))

            occ_v   = occ[bi, ai]
            slope_v = slope[bi, ai]
            rough_v = rough[bi, ai]
            var_v   = var[bi, ai]
            trav_v  = trav[bi, ai]

            cost = (
                5.0 * occ_v
                + 3.0 * slope_v
                + 2.0 * rough_v
                + 6.0 * var_v
                + self.cfg.w_goal    * gdist
                + self.cfg.w_heading * he.abs()
                + 15.0 * (1.0 - trav_v) ** 2 * (1.0 + 0.2 * t)
                + 5.0 * (slope_v + rough_v) * v
                + 0.1 * (throttle ** 2 + steer ** 2)
                - 3.0 / (gdist + 1e-3)
            )

            # Collision penalty
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
                            * torch.exp(-d2 / (2 * self.cfg.memory_radius ** 2))
                            * (1.0 + 0.5 * torch.cos(ang - yaw.unsqueeze(1)))).sum(dim=1)
                ed  = torch.sqrt(d2 + 1e-6)
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §10  CONTROLLER (AUTO smoothing)
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Controller:
    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._state   = ControlState()
        self._spd_ema = 0.0
        self._str_ema = 0.0

    def apply_safety_filters(
        self, ctrl: ControlState, roll_deg: float, pitch_deg: float
    ) -> ControlState:
        a = self.cfg.ema_alpha
        self._spd_ema = a * ctrl.throttle + (1 - a) * self._spd_ema
        self._str_ema = a * ctrl.steer    + (1 - a) * self._str_ema

        th = self._spd_ema
        st = self._str_ema
        br = ctrl.brake

        if br > 0.1:
            th = 0.0

        st = float(np.clip(
            st,
            self._state.steer - self.cfg.max_steer_rate,
            self._state.steer + self.cfg.max_steer_rate,
        ))
        th = float(np.clip(
            th,
            self._state.throttle - self.cfg.max_throttle_rate,
            self._state.throttle + self.cfg.max_throttle_rate,
        ))

        ts, cs = self._stability(roll_deg, pitch_deg)
        th *= ts
        st  = float(np.clip(st + cs, -self.cfg.max_steer, self.cfg.max_steer))

        st -= self.cfg.steer_penalty    * (st - self._state.steer)
        th -= self.cfg.throttle_penalty * (th - self._state.throttle)

        th = float(np.clip(th, 0.0,              self.cfg.max_throttle))
        st = float(np.clip(st, -self.cfg.max_steer, self.cfg.max_steer))

        self._state = ControlState(throttle=th, steer=st, brake=br, reverse=ctrl.reverse)
        return self._state

    def _stability(self, roll: float, pitch: float) -> Tuple[float, float]:
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §11  STUCK DETECTOR
# ╚═══════════════════════════════════════════════════════════════════════════╝

class StuckDetector:
    IDLE       = "IDLE"
    RECOVERING = "RECOVERING"

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._pos_hist: deque    = deque(maxlen=200)
        self._str_hist: deque    = deque(maxlen=40)
        self._state             = self.IDLE
        self._rec_start: float   = 0.0

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int, str]:
        self._pos_hist.append((t, pos.copy()))
        self._str_hist.append(steer)
        ftype = "stall"

        if self._state == self.RECOVERING:
            elapsed   = t - self._rec_start
            phase_dur = self.cfg.recovery_duration_s / 3.0
            if elapsed > self.cfg.recovery_duration_s:
                self._state = self.IDLE
                return False, 0, ftype
            return True, min(int(elapsed / phase_dur), 2), ftype

        if len(self._pos_hist) >= 2:
            ot, op = self._pos_hist[0]
            if (t - ot) >= self.cfg.stuck_window_s:
                if np.linalg.norm(pos - op) < self.cfg.stuck_disp_thresh:
                    self._trigger(t)
                    return True, 0, ftype

        if len(self._str_hist) == self._str_hist.maxlen:
            arr   = np.array(self._str_hist)
            flips = int(np.sum(np.diff(np.sign(arr)) != 0))
            if flips > len(arr) * 0.7:
                self._trigger(t)
                return True, 0, "oscillation"

        return False, 0, ftype

    def _trigger(self, t: float) -> None:
        if self._state == self.IDLE:
            print("[STUCK] Recovery triggered!")
            self._state     = self.RECOVERING
            self._rec_start = t
            self._pos_hist.clear()

    def get_recovery_control(self, phase: int) -> ControlState:
        steer_map = {0: 0.0, 1: random.uniform(-0.6, 0.6), 2: random.uniform(-0.8, 0.8)}
        return ControlState(
            throttle=0.3,
            steer=steer_map.get(phase, 0.0),
            brake=0.0,
            reverse=True,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §12  MANUAL CONTROLLER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ManualController:
    THROTTLE_STEP = 0.04
    STEER_STEP    = 0.06
    STEER_DECAY   = 0.80

    def __init__(self, cfg: Config = CFG):
        self.cfg       = cfg
        self._throttle = 0.0
        self._steer    = 0.0
        self._reverse  = False

    def tick(self, keys, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> ControlState:
        brake = 0.0

        if keys[pygame.K_w]:
            self._reverse  = False
            self._throttle = min(self._throttle + self.THROTTLE_STEP, self.cfg.max_throttle)
        elif keys[pygame.K_s]:
            self._reverse  = True
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
            self._throttle = 0.0
            self._steer   *= 0.5
            self._reverse  = False
            brake          = self.cfg.max_brake

        if abs(roll_deg) > self.cfg.max_safe_roll_deg:
            corr = -math.copysign(
                min(0.15 * (abs(roll_deg) - self.cfg.max_safe_roll_deg) / 10.0, 0.15),
                roll_deg,
            )
            self._steer = float(np.clip(
                self._steer + corr, -self.cfg.max_steer, self.cfg.max_steer))

        return ControlState(
            throttle=round(self._throttle, 4),
            steer   =round(self._steer, 4),
            brake   =brake,
            reverse =self._reverse,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §13  VISUALIZATION MODULE
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VisualizationModule:
    C = {
        'bg':       (17,  17,  17),
        'panel':    (26,  26,  46),
        'header':   (35,  35,  65),
        'grid':     (45,  45,  70),
        'border':   (60,  60,  95),
        'traj':     (0,   206, 201),
        'veh':      (253, 203, 110),
        'goal':     (255, 60,  60),
        'subgoal':  (50,  255, 120),
        'memory':   (255, 110, 50),
        'cost':     (255, 107, 107),
        'speed':    (85,  239, 196),
        'steer':    (162, 155, 254),
        'white':    (230, 230, 230),
        'dim':      (110, 110, 110),
        'danger':   (255, 50,  50),
        'ok':       (80,  220, 80),
        'warn':     (255, 200, 0),
    }

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._screen: Optional[pygame.Surface] = None
        self._font_s: Optional[pygame.font.Font] = None

        mx = cfg.traj_history
        self._traj_x:  deque = deque(maxlen=mx)
        self._traj_y:  deque = deque(maxlen=mx)
        self._costs:   deque = deque(maxlen=400)
        self._speeds:  deque = deque(maxlen=400)

        self._front_img: Optional[np.ndarray] = None
        self._rear_img:  Optional[np.ndarray] = None
        self._lidar_pts: Optional[np.ndarray] = None

        self._goal          = np.array([cfg.goal_x, cfg.goal_y])
        self._subgoal: Optional[np.ndarray] = None
        self._subgoal_candidates: List[np.ndarray] = []
        self._mem_pts: list = []

        self._mode      = "AUTO"
        self._speed     = 0.0
        self._steer     = 0.0
        self._cost      = 0.0
        self._dist_goal = 0.0
        self._obs_warn  = False

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
        subgoal: Optional[np.ndarray],
        all_candidates: Optional[List[Subgoal]],
        mem_pts: list,
        dist_goal: float,
        obs_warn: bool,
    ) -> None:
        self._traj_x.append(float(pos[0]))
        self._traj_y.append(float(pos[1]))
        self._costs.append(cost)
        self._speeds.append(speed)
        self._mode      = mode
        self._speed     = speed
        self._steer     = steer
        self._cost      = cost
        self._subgoal   = subgoal.copy() if subgoal is not None else None
        # Store all candidate world positions for debug rendering
        self._subgoal_candidates = [
            c.world_pos for c in all_candidates
        ] if all_candidates else []
        self._mem_pts   = list(mem_pts)
        self._dist_goal = dist_goal
        self._obs_warn  = obs_warn
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

        self._draw_camera(self._front_img, "Front camera", 0,        0, col_w,   CAM_H)
        self._draw_camera(self._rear_img,  "Rear camera",  col_w,    0, col_w,   CAM_H)
        self._draw_lidar("LiDAR (top)",   col_w * 2, 0, W - col_w * 2, CAM_H)
        self._draw_trajectory(0,     bot_y, map_w, bot_h)
        self._draw_metrics(map_w, bot_y, W - map_w, bot_h)
        self._draw_hud(W, H)

        pygame.display.flip()

    def _draw_camera(self, img, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x + 4, y + 3))
        if img is not None:
            try:
                img_r = self._resize(img, w, h - 18)
                surf  = pygame.surfarray.make_surface(img_r.swapaxes(0, 1))
                self._screen.blit(surf, (x, y + 18))
            except Exception:
                pass
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_lidar(self, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render(title, True, C['white']), (x + 4, y + 3))

        content_y = y + 18
        content_h = h - 18

        lr    = self.cfg.lidar_range
        cx    = x + w // 2
        cy    = content_y + content_h // 2
        scale = (min(w, content_h) * 0.44) / lr

        for ring_r in [lr * 0.33, lr * 0.67, lr]:
            pr = int(ring_r * scale)
            pygame.draw.circle(self._screen, C['grid'], (cx, cy), pr, 1)
        pygame.draw.line(self._screen, C['grid'], (cx, cy), (cx, cy - int(lr * scale)), 1)

        if self._lidar_pts is not None and len(self._lidar_pts):
            pts = self._lidar_pts
            zn  = (pts[:, 2] - pts[:, 2].min()) / (np.ptp(pts[:, 2]) + 1e-6)
            for i in range(0, len(pts), 3):
                px = int(cx - pts[i, 1] * scale)
                py = int(cy - pts[i, 0] * scale)
                if x <= px < x + w and content_y <= py < content_y + content_h:
                    t  = float(zn[i])
                    r  = int(60  + 195 * min(t * 2, 1.0))
                    g  = int(80  + 100 * t)
                    b  = int(200 - 180 * min(t * 2, 1.0))
                    pygame.draw.circle(self._screen, (r, g, b), (px, py), 1)

        pygame.draw.circle(self._screen, C['veh'], (cx, cy), 4)
        pygame.draw.line(self._screen, C['veh'], (cx, cy), (cx, cy - 14), 2)
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_trajectory(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font_s.render("Trajectory / memory", True, C['white']),
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
        pad  = span * 0.12
        mn_x, mx_x = min(ax) - pad, max(ax) + pad
        mn_y, mx_y = min(ay) - pad, max(ay) + pad

        px0 = x + 6
        py0 = y + 22
        pw  = w - 12
        ph  = h - 26

        def w2s(wx, wy):
            sx = px0 + int((wx - mn_x) / (mx_x - mn_x) * pw)
            sy = py0 + int((1.0 - (wy - mn_y) / (mx_y - mn_y)) * ph)
            return sx, sy

        for gx2 in np.linspace(mn_x, mx_x, 5):
            sx, _ = w2s(gx2, mn_y)
            pygame.draw.line(self._screen, C['grid'], (sx, py0), (sx, py0 + ph), 1)
        for gy2 in np.linspace(mn_y, mx_y, 5):
            _, sy = w2s(mn_x, gy2)
            pygame.draw.line(self._screen, C['grid'], (px0, sy), (px0 + pw, sy), 1)

        for px2, py2, pw2, _ in self._mem_pts:
            sx, sy = w2s(px2, py2)
            r = max(4, int(pw2 * 14))
            blob = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(blob, (*C['memory'], 90), (r, r), r)
            self._screen.blit(blob, (sx - r, sy - r))
            pygame.draw.circle(self._screen, C['memory'], (sx, sy), 4)

        # Debug: all candidate subgoal points
        for i, c in enumerate(self._subgoal_candidates):
            if c is not None:
                sx, sy = w2s(c[0], c[1])
                colour = C['subgoal'] if i == 0 else tuple(v // 2 for v in C['subgoal'])
                pygame.draw.circle(self._screen, colour, (sx, sy), 4 if i == 0 else 2)

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
                             (gsx + d[0], gsy + d[1]), (gsx + d[2], gsy + d[3]), 2)
        self._screen.blit(self._font_s.render("GOAL", True, C['goal']), (gsx + 8, gsy - 8))

        if self._subgoal is not None:
            ssx, ssy = w2s(self._subgoal[0], self._subgoal[1])
            pygame.draw.circle(self._screen, C['subgoal'], (ssx, ssy), 6)
            pygame.draw.circle(self._screen, C['white'],   (ssx, ssy), 6, 1)
            self._screen.blit(self._font_s.render("SUB", True, C['subgoal']),
                              (ssx + 6, ssy - 7))

        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_metrics(self, x, y, w, h):
        half = h // 2
        self._draw_graph(list(self._costs),  "MPPI cost",    x, y,       w, half,      self.C['cost'])
        self._draw_graph(list(self._speeds), "Speed (m/s)", x, y + half, w, h - half,  self.C['speed'])

    def _draw_graph(self, data, title, x, y, w, h, color):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        lbl = f"{title}   {data[-1]:.2f}" if data else title
        self._screen.blit(self._font_s.render(lbl, True, color), (x + 4, y + 3))

        if len(data) < 2:
            pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)
            return

        px0, py0 = x + 5, y + 22
        pw, ph   = w - 10, h - 26
        mn, mx   = min(data), max(data)
        rng      = max(mx - mn, 1e-3)

        if mn < 0 < mx:
            zy = py0 + int((1 - (0 - mn) / rng) * ph)
            pygame.draw.line(self._screen, C['grid'], (px0, zy), (px0 + pw, zy), 1)

        n   = len(data)
        pts = [
            (px0 + int(i / (n - 1) * pw),
             py0 + int((1 - (v - mn) / rng) * ph))
            for i, v in enumerate(data)
        ]
        if len(pts) > 1:
            pygame.draw.lines(self._screen, color, False, pts, 2)

        self._screen.blit(self._font_s.render(f"{mx:.1f}", True, C['dim']), (px0 + 2, py0 + 1))
        self._screen.blit(self._font_s.render(f"{mn:.1f}", True, C['dim']), (px0 + 2, py0 + ph - 12))
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_hud(self, W: int, H: int) -> None:
        C = self.C
        mode_c = C['ok'] if self._mode == "AUTO" else C['warn']

        lines = [
            (f"MODE : {self._mode}",             mode_c),
            (f"Speed: {self._speed:5.2f} m/s",   C['speed']),
            (f"Steer: {self._steer:+6.3f}",       C['steer']),
            (f"Cost : {self._cost:8.1f}",         C['cost']),
            (f"Goal : {self._dist_goal:6.1f} m",  C['white']),
            (f"Mem  : {len(self._mem_pts)}",      C['dim']),
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §14  SENSOR MANAGER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SensorManager:
    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self._actors = []
        bp           = world.get_blueprint_library()

        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",               str(cfg.lidar_range))
        lidar_bp.set_attribute("points_per_second",   str(cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency",  str(cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("channels",            str(cfg.lidar_channels))
        lidar_bp.set_attribute("upper_fov",           str(cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov",           str(cfg.lidar_lower_fov))

        self._lidar_q: Queue = Queue(maxsize=2)
        self._front_q: Queue = Queue(maxsize=2)
        self._rear_q:  Queue = Queue(maxsize=2)

        lidar = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.5, z=2.0)),
            attach_to=vehicle,
        )
        lidar.listen(lambda d: self._lidar_q.put(d) if not self._lidar_q.full() else None)
        self._actors.append(lidar)

        for tag, tf in [
            ("front", carla.Transform(carla.Location(x=1.5, z=1.8))),
            ("rear",  carla.Transform(carla.Location(x=-1.5, z=1.8), carla.Rotation(yaw=180))),
        ]:
            cam_bp = bp.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.camera_width))
            cam_bp.set_attribute("image_size_y", str(cfg.camera_height))
            cam_bp.set_attribute("fov",          str(cfg.camera_fov))
            cam = world.spawn_actor(cam_bp, tf, attach_to=vehicle)
            q   = self._front_q if tag == "front" else self._rear_q
            cam.listen(lambda img, _q=q: _q.put(self._img_to_arr(img))
                       if not _q.full() else None)
            self._actors.append(cam)

    @staticmethod
    def _img_to_arr(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        return arr[:, :, ::-1].copy()

    def _get(self, q: Queue):
        try:
            return q.get_nowait()
        except Empty:
            return None

    @property
    def lidar_data(self):   return self._get(self._lidar_q)
    @property
    def front_image(self):  return self._get(self._front_q)
    @property
    def rear_image(self):   return self._get(self._rear_q)

    def destroy(self) -> None:
        for a in self._actors:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §15  VEHICLE SPAWNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

def spawn_vehicle(world, cfg: Config = CFG):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)
    tf = carla.Transform(
        carla.Location(x=906.66, y=-875.78, z=4.762),
        carla.Rotation(pitch=-4.5, yaw=88.0),
    )
    # tf = carla.Transform(
    #     carla.Location(x=240.35, y=223.16, z=2.15),
    #     carla.Rotation(pitch=-14.03, yaw=103.85, roll=0.00),
    # )
#     ----------------------------------------
# Location: x=240.35, y=-223.16, z=5.15
# Rotation: pitch=-14.03, yaw=103.85, roll=0.00
# ----------------------------------------

    vehicle = world.try_spawn_actor(vehicle_bp, tf)
    if vehicle is None:
        pts     = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(pts)) if pts else None
    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle")

    print(f"[SPAWN] at {vehicle.get_location()}")
    t0 = time.time()
    while time.time() - t0 < cfg.post_spawn_wait_s:
        world.tick()

    return vehicle, vehicle.get_location().z


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §16  MAIN NAVIGATION SYSTEM
# ╚═══════════════════════════════════════════════════════════════════════════╝

class NavigationSystem:
    def __init__(self, cfg: Config = CFG):
        self.cfg    = cfg
        self._mode  = "AUTO"
        self._running = False

        self._client  = None
        self._world   = None
        self._vehicle = None
        self._sensors: Optional[SensorManager] = None

        self._perception = PerceptionModule(cfg)
        self._planner    = MPPIPlanner(cfg)
        self._controller = Controller(cfg)
        self._manual     = ManualController(cfg)
        self._stuck      = StuckDetector(cfg)
        self._memory     = StuckMemory(cfg)
        self._subgoal_pl = VSGPSubgoalPlanner(cfg)
        self._reactive   = ReactiveObstacleAvoidance(cfg)
        self._viz        = VisualizationModule(cfg)

        # Global goal in world coords
        self._goal = np.array([cfg.goal_x, cfg.goal_y], dtype=np.float32)

        self._step = 0
        self._t0   = 0.0

        pygame.init()
        self._screen = pygame.display.set_mode(
            (cfg.viz_win_w, cfg.viz_win_h), pygame.RESIZABLE)
        pygame.display.set_caption(
            "VSGP-MPPI Navigator  |  TAB = AUTO/MANUAL  |  ESC = quit")
        self._viz.init(self._screen)

    def connect(self) -> None:
        print(f"[CARLA] connecting to {self.cfg.carla_host}:{self.cfg.carla_port}")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world  = self._client.get_world()
        if self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode     = True
            s.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self._world.apply_settings(s)
            print("[CARLA] synchronous mode ON")

    def setup(self) -> None:
        self._vehicle, _ = spawn_vehicle(self._world, self.cfg)
        self._sensors    = SensorManager(self._world, self._vehicle, self.cfg)
        self._t0         = time.time()
        print("[SETUP] sensors ready – navigation starting")

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
        print("[CLEANUP]")
        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.apply_control(carla.VehicleControl(brake=1.0))
        if self._sensors:
            self._sensors.destroy()
        if self._world and self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode = False
            self._world.apply_settings(s)
        pygame.quit()

    def _tick(self) -> None:
        # 1. Advance world
        if self.cfg.synchronous:
            self._world.tick()
        else:
            self._world.wait_for_tick()
        self._step += 1

        # 2. Events
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
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._viz.init(self._screen)

        keys = pygame.key.get_pressed()

        # 3. Vehicle state
        tf        = self._vehicle.get_transform()
        loc       = tf.location
        vel       = self._vehicle.get_velocity()
        speed     = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        roll_deg  = float(tf.rotation.roll)
        pitch_deg = float(tf.rotation.pitch)
        yaw_rad   = math.radians(tf.rotation.yaw)
        pos       = np.array([loc.x, loc.y, loc.z], dtype=np.float32)

        t_now     = time.time() - self._t0

        # 4. Goal check
        dist_goal = float(np.linalg.norm(pos[:2] - self._goal))
        if dist_goal < self.cfg.goal_tolerance:
            print("[NAV] *** GOAL REACHED ***")
            self._vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
            self._running = False
            return

        # 5. Sensor data
        lidar_raw = self._sensors.lidar_data
        perc = self._perception.process_lidar(lidar_raw) if lidar_raw is not None \
               else self._perception.last_output

        front = self._sensors.front_image
        rear  = self._sensors.rear_image
        if front is not None:  self._viz.set_front(front)
        if rear  is not None:  self._viz.set_rear(rear)

        # 6. Planning
        ctrl_state: Optional[ControlState] = None
        subgoal_for_viz: Optional[np.ndarray] = None
        all_candidates:  Optional[List[Subgoal]] = None
        current_cost = 0.0

        if self._mode == "MANUAL":
            ctrl_state = self._manual.tick(keys, roll_deg, pitch_deg)

        else:  # AUTO
            stuck, rec_phase, ftype = self._stuck.update(
                pos, self._controller._state.steer, t_now)
            self._memory.update(pos, stuck, ftype)

            if stuck:
                print(f"[NAV] Stuck ({ftype}) – recovering phase {rec_phase}")
                ctrl_state = self._stuck.get_recovery_control(rec_phase)

            elif perc is not None:
                # ── Goal-aware VSGP subgoal planning ─────────────────────
                best_subgoal, all_candidates = self._subgoal_pl.plan(
                    perc,
                    vehicle_pos=pos,
                    vehicle_yaw_rad=yaw_rad,
                    goal_pos=self._goal,
                )

                if best_subgoal is None:
                    # Grid blocked — force recovery
                    print("[NAV] VSGP grid blocked – entering recovery")
                    self._stuck.update(pos, self._controller._state.steer, t_now)
                    stuck, rec_phase, ftype = True, 0, "grid_blocked"
                    self._memory.update(pos, True, ftype)
                    ctrl_state = self._stuck.get_recovery_control(0)

                else:
                    subgoal_world = best_subgoal.world_pos
                    subgoal_for_viz = subgoal_world

                    state_vec  = [loc.x, loc.y, yaw_rad, speed]
                    mem_tensor = self._memory.get_tensor(self._planner.device)
                    ctrl_state, current_cost = self._planner.plan(
                        state_vec, perc, subgoal_world, mem_tensor)
                    ctrl_state = self._controller.apply_safety_filters(
                        ctrl_state, roll_deg, pitch_deg)

                    # Log selected subgoal info
                    print(f"[SUBGOAL] alpha={best_subgoal.alpha:.2f} "
                          f"dist={best_subgoal.distance:.1f}m "
                          f"progress={best_subgoal.goal_progress:.2f}m "
                          f"slope={best_subgoal.slope:.2f} "
                          f"rough={best_subgoal.roughness:.2f} "
                          f"cost={best_subgoal.cost:.2f}")

        if ctrl_state is None:
            ctrl_state = ControlState(throttle=0.1)
            print("[NAV] fallback: creeping forward")

        # 7. Reactive avoidance
        raw_pts    = perc.raw_points if perc is not None else None
        ctrl_state = self._reactive.process(raw_pts, ctrl_state, speed)

        # 8. Apply to CARLA
        self._vehicle.apply_control(carla.VehicleControl(
            throttle=float(np.clip(ctrl_state.throttle, 0.0,  self.cfg.max_throttle)),
            steer   =float(np.clip(ctrl_state.steer,   -self.cfg.max_steer, self.cfg.max_steer)),
            brake   =float(np.clip(ctrl_state.brake,    0.0,  self.cfg.max_brake)),
            reverse =bool(ctrl_state.reverse),
        ))

        # 9. Render
        self._viz.push(
            pos=pos, speed=speed, steer=ctrl_state.steer,
            perc=perc, mode=self._mode, cost=current_cost,
            subgoal=subgoal_for_viz,
            all_candidates=all_candidates,
            mem_pts=self._memory.points,
            dist_goal=dist_goal,
            obs_warn=self._reactive.obstacle_active,
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()

        if not self.cfg.synchronous:
            time.sleep(self.cfg.fixed_delta_seconds)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §17  ENTRY POINT
# ╚═══════════════════════════════════════════════════════════════════════════╝

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
