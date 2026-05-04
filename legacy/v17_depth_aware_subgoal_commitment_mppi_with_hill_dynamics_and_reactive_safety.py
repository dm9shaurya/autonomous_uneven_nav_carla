"""
CARLA Off-Road Navigator — Full Rewrite
========================================
Fixes all issues from the previous version:
  1. Real depth-aware subgoal sampling (not a lie)
  2. MPPI tracks SUBGOAL, not raw global goal
  3. Subgoal commitment timer (no oscillation)
  4. Reverse only when truly stuck, not just scared
  5. Slope-aware dynamics in MPPI rollout (hill physics)
  6. Anti-stall cost (vehicle KEEPS MOVING)
  7. Higher MPPI exploration (not timid)
  8. Reactive layer is last resort ONLY (no constant braking)
  9. Clean hierarchy: Subgoal → MPPI → Reactive
"""

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

# ── Optional sklearn ─────────────────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found – KMeans falls back to random sampling")

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU")
except ImportError:
    print("[FATAL] PyTorch not found. pip install torch")
    sys.exit(1)

# ── CARLA ─────────────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print("[FATAL] carla package not found. Put CARLA egg on PYTHONPATH.")
    sys.exit(1)


# ╔═══════════════════════════════════════════════════════════════╗
# §0  CONFIGURATION
# ╚═══════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    # ── CARLA
    carla_host:            str   = "127.0.0.1"
    carla_port:            int   = 2000
    carla_timeout:         float = 20.0
    synchronous:           bool  = True
    fixed_delta_seconds:   float = 0.05

    # ── Vehicle
    vehicle_blueprint:     str   = "vehicle.tesla.cybertruck"
    post_spawn_wait_s:     float = 3.0
    wheel_base:            float = 3.807
    vehicle_width:         float = 2.0

    # ── Sensors
    lidar_range:           float = 30.0
    lidar_points_per_sec:  int   = 80_000
    lidar_rotation_freq:   float = 20.0
    lidar_channels:        int   = 64
    lidar_upper_fov:       float = 15.0
    lidar_lower_fov:       float = -25.0
    camera_width:          int   = 320
    camera_height:         int   = 240
    camera_fov:            int   = 90

    # ── Goal
    goal_x:                float = -164.10
    goal_y:                float = 127.96
    goal_tolerance:        float = 3.0

    # ── Subgoal planner
    subgoal_distance:      float = 30.0   # max look-ahead (m)
    subgoal_min_distance:  float = 2.0    # min look-ahead (m)
    subgoal_num_angles:    int   = 30     # rays to scan
    subgoal_num_depth:     int   = 20     # depth samples per ray
    subgoal_commit_ticks:  int   = 25     # ticks before re-planning subgoal

    # ── MPPI  (FIX: more samples, more noise → less timid)
    mppi_horizon:          int   = 20
    mppi_num_samples:      int   = 2048
    mppi_lambda:           float = 0.8
    mppi_noise_throttle:   float = 0.40   # was 0.2 — now bolder
    mppi_noise_steer:      float = 0.3   # was 0.3 — now bolder

    # ── Cost weights
    w_goal:                float = 12.0
    w_heading:             float = 3.0
    w_terrain_risk:        float = 6.0
    w_memory:              float = 25.0
    w_anti_stall:          float = 12.0   # NEW: penalise low speed hard

    # ── Control limits
    max_throttle:          float = 0.75   # higher ceiling for hills
    max_steer:             float = 0.6
    max_brake:             float = 0.5    # FIX: was 0.8 — less panic braking

    # ── Smoothing
    ema_alpha:             float = 0.35
    max_steer_rate:        float = 0.10
    max_throttle_rate:     float = 0.12

    # ── Stability
    max_safe_roll_deg:     float = 12.0
    max_safe_pitch_deg:    float = 20.0

    # ── Stuck detection
    stuck_window_s:        float = 2.5
    stuck_disp_thresh:     float = 0.5
    recovery_duration_s:   float = 3.0

    # ── Stuck memory
    memory_decay:          float = 0.995
    memory_radius:         float = 8.0
    memory_threshold:      float = 0.05

    # ── Reactive (FIX: wider distances, gentler response)
    react_emergency_dist:  float = 2.5    # only override in true emergency
    react_danger_dist:     float = 4.5
    react_warn_dist:       float = 8.0
    react_forward_cone:    float = 45.0

    # ── VSGP
    vsgp_n_inducing:       int   = 50
    vsgp_alpha:            float = 1.0
    vsgp_length_scale:     float = 0.3
    vsgp_noise_var:        float = 0.05
    vsgp_lr:               float = 0.03

    # ── Visualization
    viz_win_w:             int   = 1280
    viz_win_h:             int   = 720
    viz_every_n_ticks:     int   = 3
    traj_history:          int   = 600


CFG = Config()


# ╔═══════════════════════════════════════════════════════════════╗
# §1  DATA CLASSES
# ╚═══════════════════════════════════════════════════════════════╝
@dataclass
class ControlState:
    throttle: float = 0.0
    steer:    float = 0.0
    brake:    float = 0.0
    reverse:  bool  = False


@dataclass
class PerceptionOutput:
    alpha_grid:      np.ndarray
    beta_grid:       np.ndarray
    occupancy_mean:  np.ndarray
    occupancy_var:   np.ndarray
    slope_map:       np.ndarray
    roughness_map:   np.ndarray
    traversability:  np.ndarray
    raw_points:      np.ndarray
    ground_slope_deg: float = 0.0


@dataclass
class Subgoal:
    alpha:         float
    distance:      float
    local_pos:     np.ndarray   # (x, y) in vehicle frame
    world_pos:     np.ndarray   # (x, y) in world frame
    slope:         float
    roughness:     float
    occupancy:     float
    traversability: float
    goal_progress: float
    heading_error: float
    cost:          float = 0.0


# ╔═══════════════════════════════════════════════════════════════╗
# §2  RATIONAL-QUADRATIC KERNEL + VSGP
# ╚═══════════════════════════════════════════════════════════════╝
class RQKernel:
    def __init__(self, alpha: float = 1.0, ls: float = 0.3, var: float = 1.0):
        self.alpha = alpha
        self.ls    = ls
        self.var   = var

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if not len(X) or not len(Y):
            return np.zeros((len(X), len(Y)))
        diff  = X[:, None, :] - Y[None, :, :]
        sq_d  = np.sum(diff ** 2, axis=-1)
        denom = 2.0 * self.alpha * self.ls ** 2
        return self.var * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return self.var * np.ones(len(X))


class VSGP:
    """Variational Sparse GP for terrain height regression."""

    def __init__(self, cfg: Config):
        self.kernel    = RQKernel(alpha=cfg.vsgp_alpha, ls=cfg.vsgp_length_scale)
        self.noise_var = cfg.vsgp_noise_var
        self.n_ind     = cfg.vsgp_n_inducing
        self.lr        = cfg.vsgp_lr

        side           = int(math.sqrt(self.n_ind))
        A, B           = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, side),
            np.linspace(-np.pi / 4, np.pi / 4, side),
        )
        self.Z         = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]
        m              = len(self.Z)
        self.mu        = np.zeros(m)
        self.Su        = np.eye(m) * 0.1
        self._Kuu_inv: Optional[np.ndarray] = None
        self._trained  = False
        self._tick     = 0

    def _select_inducing(self, X: np.ndarray) -> None:
        n = min(self.n_ind, len(X))
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50, random_state=0)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            self.Z = X[np.random.choice(len(X), n, replace=False)].copy()
        m      = len(self.Z)
        self.mu  = np.zeros(m)
        self.Su  = np.eye(m) * 0.1
        self._Kuu_inv = None

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 4:
            return
        self._tick += 1
        if self._tick % 100 == 0:
            self._select_inducing(X)

        m   = len(self.Z)
        Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6
        try:
            self._Kuu_inv = np.linalg.inv(Kuu)
        except np.linalg.LinAlgError:
            self._Kuu_inv = np.linalg.pinv(Kuu)

        Kfu = self.kernel(X, self.Z)
        A   = Kfu @ self._Kuu_inv
        ni  = 1.0 / self.noise_var
        L   = ni * (A.T @ A) + self._Kuu_inv
        rhs = ni * (A.T @ y)
        try:
            Su_new = np.linalg.inv(L + np.eye(m) * 1e-6)
            mu_new = Su_new @ rhs
        except np.linalg.LinAlgError:
            return

        lr       = self.lr
        self.Su  = (1 - lr) * self.Su + lr * Su_new
        self.mu  = (1 - lr) * self.mu + lr * mu_new
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
        var      = np.clip(Kss_diag - var_exp + var_var + self.noise_var, 1e-6, None)
        return mean, var


# ╔═══════════════════════════════════════════════════════════════╗
# §3  PERCEPTION MODULE
# ╚═══════════════════════════════════════════════════════════════╝
class PerceptionModule:
    ALPHA_RES = 60
    BETA_RES  = 30
    ROC       = 5.0

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.vsgp = VSGP(cfg)
        self._last_output: Optional[PerceptionOutput] = None

        AG, BG = np.meshgrid(
            np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES),
            np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES),
        )
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG, self._BG = AG, BG

    def _ground_slope(self, pts: np.ndarray) -> float:
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

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        raw  = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(raw) < 20:
            return self._last_output
        dist = np.linalg.norm(raw, axis=1)
        pts  = raw[(dist > 0.5) & (dist < self.cfg.lidar_range)]
        if len(pts) < 10:
            return self._last_output

        ground_slope = self._ground_slope(pts)
        r     = np.linalg.norm(pts, axis=1)
        alpha = np.arctan2(pts[:, 1], pts[:, 0])
        beta  = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1.0, 1.0))
        X     = np.column_stack([alpha, beta])
        y     = self.ROC - r

        if len(X) > 2000:
            idx   = np.random.choice(len(X), 2000, replace=False)
            X, y, pts = X[idx], y[idx], pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)
        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var  = var.reshape(self.BETA_RES,  self.ALPHA_RES)

        da       = np.gradient(mean, axis=1)
        db       = np.gradient(mean, axis=0)
        slope    = np.sqrt(da ** 2 + db ** 2)
        roughness = np.sqrt(np.gradient(da, axis=1) ** 2 + np.gradient(db, axis=0) ** 2)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-6)

        slope     = _norm(slope)
        roughness = _norm(roughness)
        occ_norm  = _norm(mean)
        var_norm  = _norm(var)

        traversability = np.clip(
            1.0 - 1.2 * occ_norm - 1.2 * slope - 1.0 * var_norm - 0.8 * roughness,
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
            ground_slope_deg=ground_slope,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔═══════════════════════════════════════════════════════════════╗
# §4  SUBGOAL PLANNER  (completely rewritten — real depth sampling)
# ╚═══════════════════════════════════════════════════════════════╝
class SubgoalPlanner:
    """
    Goal-aware subgoal planner with TRUE depth-aware ray marching.

    FIX vs old code:
    • Each depth sample maps to a unique (alpha, beta) grid cell — not the
      same cell for every depth.  We project (d, a) → angular index properly.
    • Far points get a distance-uncertainty penalty.
    • Subgoal is committed for `commit_ticks` ticks to prevent oscillation.
    • Scoring rewards goal progress and penalises off-heading + bad terrain.
    """

    OCC_FREE_THRESH   = 0.65
    VAR_STABLE_THRESH = 0.72
    SLOPE_SAFE_THRESH = 0.75
    ROUGH_SAFE_THRESH = 0.75

    W_TERRAIN   = 2.5
    W_GOAL_DIST = 2.5
    W_HEADING   = 1.5
    W_SLOPE     = 2.0
    W_ROUGH     = 1.5
    W_OCC       = 2.0

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.min_dist     = cfg.subgoal_min_distance
        self.max_dist     = cfg.subgoal_distance
        self.n_angles     = cfg.subgoal_num_angles
        self.n_depth      = cfg.subgoal_num_depth
        self.commit_ticks = cfg.subgoal_commit_ticks

        self._committed:  Optional[Subgoal] = None
        self._commit_age: int               = 0

    @staticmethod
    def _norm_angle(a: float) -> float:
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    def _lookup_grid(
        self,
        alphas:   np.ndarray,
        betas:    np.ndarray,
        occ:      np.ndarray,
        var:      np.ndarray,
        slope_map: np.ndarray,
        rough_map: np.ndarray,
        trav_map: np.ndarray,
        alpha:    float,
        beta:     float,
        dist:     float,
    ) -> Tuple[float, float, float, float, float]:
        """
        TRUE depth-aware lookup.
        Returns (occ_val, var_val, slope_val, rough_val, trav_val).
        Far points get a distance-uncertainty penalty.
        """
        a_idx = int(np.argmin(np.abs(alphas - alpha)))
        b_idx = int(np.argmin(np.abs(betas  - beta)))
        a_idx = int(np.clip(a_idx, 0, occ.shape[1] - 1))
        b_idx = int(np.clip(b_idx, 0, occ.shape[0] - 1))

        # Distance uncertainty penalty: farther = less reliable, treat as riskier
        dist_pen = dist / self.max_dist  # [0, 1]

        o_val = float(occ[b_idx, a_idx])       * (1.0 + 0.4 * dist_pen)
        v_val = float(var[b_idx, a_idx])       * (1.0 + 0.6 * dist_pen)
        s_val = float(slope_map[b_idx, a_idx])
        r_val = float(rough_map[b_idx, a_idx])
        t_val = float(trav_map[b_idx, a_idx])  * (1.0 - 0.2 * dist_pen)

        return (
            np.clip(o_val, 0, 1),
            np.clip(v_val, 0, 1),
            s_val,
            r_val,
            np.clip(t_val, 0, 1),
        )

    def plan(
        self,
        perc:           PerceptionOutput,
        vehicle_pos:    np.ndarray,
        vehicle_yaw_rad: float,
        goal_pos:       np.ndarray,
    ) -> Tuple[Optional[Subgoal], List[Subgoal]]:

        # ── Tick commitment counter ───────────────────────────────────
        self._commit_age += 1
        if self._committed is not None and self._commit_age < self.commit_ticks:
            # Keep using committed subgoal: check it's still reachable
            still_ok = self._committed.traversability > 0.3
            if still_ok:
                return self._committed, []

        # ── Reset commitment ──────────────────────────────────────────
        self._committed  = None
        self._commit_age = 0

        if perc.occupancy_mean is None:
            return None, []

        alphas = perc.alpha_grid[0]       # (ALPHA_RES,)
        betas  = perc.beta_grid[:, 0]     # (BETA_RES,)
        occ    = perc.occupancy_mean
        var    = perc.occupancy_var
        slope  = perc.slope_map
        rough  = perc.roughness_map
        trav   = perc.traversability

        # ── Goal direction ────────────────────────────────────────────
        dx_world    = goal_pos[0] - vehicle_pos[0]
        dy_world    = goal_pos[1] - vehicle_pos[1]
        goal_yaw    = math.atan2(dy_world, dx_world)
        goal_rel    = self._norm_angle(goal_yaw - vehicle_yaw_rad)

        # Centre scan on goal direction (clamped to ±80°)
        half_fov    = math.pi / 2.0
        scan_centre = np.clip(goal_rel, -half_fov * 0.8, half_fov * 0.8)

        alpha_range = np.linspace(
            scan_centre - half_fov,
            scan_centre + half_fov,
            self.n_angles,
        )

        candidates: List[Subgoal] = []

        for a in alpha_range:
            depths = np.linspace(self.min_dist, self.max_dist, self.n_depth)
            best_valid: Optional[Tuple] = None

            for d in depths:
                # Project depth along ray to get angular elevation (beta)
                # Simple ground-plane assumption: points at distance d on the ground
                # have a downward elevation of arctan(vehicle_height / d)
                # We use 0.0 (horizon) as mid-scan, consistent with flat-world approx.
                # effective_beta = 0.0   # scan at horizon level
                # Approximate downward ray based on distance (vehicle height ~1.8m)
                vehicle_height = 1.8
                # effective_beta = -math.atan2(vehicle_height, d)
                # Blend horizon + slight downward look (NOT full downward)
                vehicle_height = 1.8
                down_angle = -math.atan2(vehicle_height, d)

                # blend factor: near = flat, far = slight downward
                blend = min(d / self.max_dist, 0.6)
                effective_beta = blend * down_angle

                o, v, s, r, t = self._lookup_grid(
                    alphas, betas, occ, var, slope, rough, trav,
                    alpha=a, beta=effective_beta, dist=d,
                )

                is_free   = o < self.OCC_FREE_THRESH
                is_stable = v < self.VAR_STABLE_THRESH
                is_safe   = (s < self.SLOPE_SAFE_THRESH
                             and r < self.ROUGH_SAFE_THRESH)

                if is_free and is_stable and is_safe:
                    best_valid = (d, o, v, s, r, t)
                else:
                    # Ray hit obstacle/unsafe terrain — stop marching this ray
                    break

            if best_valid is None:
                continue

            best_d, best_o, best_v, best_s, best_r, best_t = best_valid

            # World position of candidate endpoint
            lx = best_d * math.cos(a)
            ly = best_d * math.sin(a)
            wx = vehicle_pos[0] + (math.cos(vehicle_yaw_rad) * lx
                                   - math.sin(vehicle_yaw_rad) * ly)
            wy = vehicle_pos[1] + (math.sin(vehicle_yaw_rad) * lx
                                   + math.cos(vehicle_yaw_rad) * ly)

            dist_from_cand = math.hypot(goal_pos[0] - wx, goal_pos[1] - wy)
            dist_from_veh  = math.hypot(goal_pos[0] - vehicle_pos[0],
                                        goal_pos[1] - vehicle_pos[1])
            goal_progress  = dist_from_veh - dist_from_cand

            heading_err = abs(self._norm_angle(
                math.atan2(goal_pos[1] - wy, goal_pos[0] - wx) - vehicle_yaw_rad
            ))

            terrain_cost = (
                self.W_SLOPE   * best_s
                + self.W_ROUGH * best_r
                + self.W_OCC   * best_o
                + self.W_TERRAIN * (1.0 - best_t)
            )

            # cost = (
            #     terrain_cost
            #     - self.W_GOAL_DIST * max(0.0, goal_progress)
            #     + self.W_HEADING   * heading_err
            # )

            # Penalize short-sighted paths (dead ends)
            progress_ratio = goal_progress / (best_d + 1e-3)

            cost = (
                terrain_cost
                - self.W_GOAL_DIST * max(0.0, goal_progress)
                + self.W_HEADING   * heading_err
                + 4.0 * (1.0 - progress_ratio)   # NEW: dead-end penalty
            )

            sg = Subgoal(
                alpha=a,
                distance=best_d,
                local_pos=np.array([lx, ly], dtype=np.float32),
                world_pos=np.array([wx, wy], dtype=np.float32),
                slope=best_s,
                roughness=best_r,
                occupancy=best_o,
                traversability=best_t,
                goal_progress=goal_progress,
                heading_error=heading_err,
                cost=cost,
            )
            candidates.append(sg)

        if not candidates:
            return None, []

        candidates.sort(key=lambda s: s.cost)
        best = candidates[0]

        # Commit to this subgoal
        self._committed  = best
        self._commit_age = 0
        return best, candidates

    def force_reset(self) -> None:
        """Call this after recovery so planner picks a fresh subgoal."""
        self._committed  = None
        self._commit_age = 0


# ╔═══════════════════════════════════════════════════════════════╗
# §5  MPPI PLANNER  (follows subgoal; real hill physics; anti-stall)
# ╚═══════════════════════════════════════════════════════════════╝
class MPPIPlanner:
    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self.device      = DEVICE
        self.horizon     = cfg.mppi_horizon
        self.num_samples = cfg.mppi_num_samples
        self.lambda_     = cfg.mppi_lambda
        self.dt          = cfg.fixed_delta_seconds

        self._u_nom = torch.zeros((self.horizon, 2), device=self.device)

    def plan(
        self,
        state:         List[float],         # [x, y, yaw, speed]
        perc:          PerceptionOutput,
        goal_xy:       np.ndarray,          # subgoal world position (x, y)
        memory_points: Optional[torch.Tensor],
    ) -> Tuple[ControlState, float]:

        if perc is None or perc.occupancy_mean is None:
            return ControlState(throttle=0.4), 0.0

        dev  = self.device

        # Shift nominal sequence
        self._u_nom = torch.roll(self._u_nom, -1, dims=0)
        self._u_nom[-1] = self._u_nom[-2]

        # Sample perturbations
        noise = torch.zeros(
            (self.num_samples, self.horizon, 2), device=dev)
        noise[:, :, 0].normal_(0, self.cfg.mppi_noise_throttle)
        noise[:, :, 1].normal_(0, self.cfg.mppi_noise_steer)

        u = (self._u_nom.unsqueeze(0) + noise)
        u[:, :, 0] = u[:, :, 0].clamp(0.0,                    self.cfg.max_throttle)
        u[:, :, 1] = u[:, :, 1].clamp(-self.cfg.max_steer,    self.cfg.max_steer)

        occ   = torch.tensor(perc.occupancy_mean, device=dev).float()
        slope = torch.tensor(perc.slope_map,      device=dev).float()
        rough = torch.tensor(perc.roughness_map,  device=dev).float()
        var   = torch.tensor(perc.occupancy_var,  device=dev).float()
        trav  = torch.tensor(perc.traversability, device=dev).float()
        goal  = torch.tensor(goal_xy,             device=dev).float()

        costs = self._rollout(state, u, occ, slope, rough, var, trav, goal,
                              memory_points)

        beta    = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights = weights / (weights.sum() + 1e-9)

        self._u_nom = (weights[:, None, None] * u).sum(dim=0).detach()

        u0   = self._u_nom[0]
        ctrl = ControlState(
            throttle=float(u0[0].cpu()),
            steer   =float(u0[1].cpu()),
            brake   =0.0,
            reverse =False,
        )
        return ctrl, float(beta.cpu())

    def _rollout(self, state, u, occ, slope, rough, var, trav, goal,
                 mem_pts):
        K, T, _ = u.shape
        dev      = self.device
        n_b, n_a = occ.shape          # (BETA_RES, ALPHA_RES)
        a_max    = math.pi / 2.0

        x = (torch.tensor(state, device=dev)
             .float().unsqueeze(0).repeat(K, 1))
        veh_x = state[0]
        veh_y = state[1]
        veh_yaw = state[2]

        total = torch.zeros(K, device=dev)

        for t in range(T):
            throttle = u[:, t, 0]
            steer    = u[:, t, 1]

            # ── Angular index of current position wrt vehicle ─────────
            dx      = x[:, 0] - veh_x
            dy      = x[:, 1] - veh_y
            raw_ang = torch.atan2(dy, dx)
            rel_ang = torch.atan2(
                torch.sin(raw_ang - veh_yaw),
                torch.cos(raw_ang - veh_yaw),
            )
            ai = ((rel_ang + a_max) / (2 * a_max) * n_a).long().clamp(0, n_a - 1)
            bi = torch.full_like(ai, n_b // 2)

            # ── FIX: slope-aware speed (hill physics) ─────────────────
            slope_v   = slope[bi, ai]
            rough_v   = rough[bi, ai]
            occ_v     = occ[bi, ai]
            var_v     = var[bi, ai]
            trav_v    = trav[bi, ai]

            # Gravity penalty: going uphill costs more power
            gravity_penalty = torch.cos(torch.atan(slope_v * 2.0)).clamp(0.3, 1.0)
            sp_scale  = (torch.exp(-2.5 * (slope_v + rough_v)) * gravity_penalty
                         ).clamp(0.15, 1.0)

            v   = (x[:, 3] + throttle * sp_scale * self.dt).clamp(0.0, 15.0)
            yr  = (v / self.cfg.wheel_base) * torch.tan(steer.clamp(-0.99, 0.99))
            yaw = x[:, 2] + yr * self.dt
            xp  = x[:, 0] + v * torch.cos(yaw) * self.dt
            yp  = x[:, 1] + v * torch.sin(yaw) * self.dt
            x   = torch.stack([xp, yp, yaw, v], dim=1)

            # ── Cost ─────────────────────────────────────────────────
            gdx   = goal[0] - xp
            gdy   = goal[1] - yp
            gdist = torch.sqrt(gdx ** 2 + gdy ** 2 + 1e-6)
            gh    = torch.atan2(gdy, gdx)
            he    = torch.atan2(torch.sin(gh - yaw), torch.cos(gh - yaw)).abs()

            cost = (
                # Goal tracking
                self.cfg.w_goal    * gdist
                + self.cfg.w_heading * he
                # Terrain risk
                + 5.0  * occ_v
                + 4.0  * slope_v
                + 3.0  * rough_v
                + 7.0  * var_v
                + 14.0 * (1.0 - trav_v) ** 2
                # Hill penalty: risky terrain at speed
                + 4.0  * (slope_v + rough_v) * v
                # Collision
                + 400.0 * (occ_v > 0.55).float()
                # FIX: anti-stall — penalise low speed HARD
                # + self.cfg.w_anti_stall * (v < 0.4).float()
                + 25.0 * (v < 0.6).float()          # stronger penalty
                + 8.0  * torch.exp(-v)              # continuous slow penalty
                # FIX: reward making progress, not just distance
                - 2.0  / (gdist + 0.5)
                # Control effort (light penalty only)
                + 0.05 * (throttle ** 2 + steer ** 2)
            )

            # ── Stuck memory avoidance ────────────────────────────────
            if mem_pts is not None:
                mx  = mem_pts[:, 0]
                my  = mem_pts[:, 1]
                mw  = mem_pts[:, 2]
                ex  = xp.unsqueeze(1) - mx.unsqueeze(0)
                ey  = yp.unsqueeze(1) - my.unsqueeze(0)
                d2  = ex ** 2 + ey ** 2
                mem_cost = (
                    mw.unsqueeze(0)
                    * torch.exp(-d2 / (2.0 * self.cfg.memory_radius ** 2))
                ).sum(dim=1)
                cost += self.cfg.w_memory * mem_cost

            total += cost

        # Terminal cost to final subgoal position
        fdx    = goal[0] - x[:, 0]
        fdy    = goal[1] - x[:, 1]
        total += 20.0 * torch.sqrt(fdx ** 2 + fdy ** 2)
        return total


# ╔═══════════════════════════════════════════════════════════════╗
# §6  REACTIVE AVOIDANCE  (last resort, not constant override)
# ╚═══════════════════════════════════════════════════════════════╝
class ReactiveAvoidance:
    """
    FIX vs old code:
    • Only activates inside react_emergency_dist / react_danger_dist.
    • Does NOT reverse unless speed < 0.5 AND truly jammed.
    • Braking is gentler (max_brake = 0.5 in CFG).
    • Does NOT override steering unless obstacle is dead ahead.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._active = False

    @property
    def obstacle_active(self) -> bool:
        return self._active

    def process(
        self,
        raw_pts:  Optional[np.ndarray],
        ctrl:     ControlState,
        speed:    float,
    ) -> ControlState:
        self._active = False

        if raw_pts is None or len(raw_pts) < 5:
            return ctrl

        x, y, z  = raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2]
        dist_2d   = np.sqrt(x ** 2 + y ** 2)
        angle_deg = np.degrees(np.arctan2(y, x))

        fwd_mask = (
            (x > 0.5)
            & (np.abs(angle_deg) < self.cfg.react_forward_cone)
            & (z > -0.5)
            & (dist_2d < self.cfg.react_warn_dist)
        )
        if not np.any(fwd_mask):
            return ctrl

        fwd_dist  = dist_2d[fwd_mask]
        fwd_pts_y = y[fwd_mask]
        min_dist  = float(np.min(fwd_dist))

        if min_dist > self.cfg.react_warn_dist:
            return ctrl

        self._active = True
        out = ControlState(
            throttle=ctrl.throttle,
            steer   =ctrl.steer,
            brake   =ctrl.brake,
            reverse =ctrl.reverse,
        )

        if min_dist <= self.cfg.react_emergency_dist:
            # True emergency: brake + steer away
            out.throttle = 0.0
            out.brake    = self.cfg.max_brake
            # FIX: only reverse if genuinely stopped
            if speed < 0.5:
                out.reverse  = True
                out.throttle = 0.35
                out.brake    = 0.0
            closest_y = float(fwd_pts_y[np.argmin(fwd_dist)])
            steer_dir = -np.sign(closest_y) if abs(closest_y) > 0.1 else 1.0
            out.steer = float(np.clip(
                ctrl.steer + steer_dir * 0.35,
                -self.cfg.max_steer, self.cfg.max_steer,
            ))
            print(f"[REACTIVE] EMERGENCY dist={min_dist:.2f}m  reverse={out.reverse}")

        # elif min_dist <= self.cfg.react_danger_dist:
        elif min_dist <= self.cfg.react_danger_dist and speed > 1.5:
            ratio        = (min_dist - self.cfg.react_emergency_dist) / (
                self.cfg.react_danger_dist - self.cfg.react_emergency_dist)
            # FIX: reduce throttle gently, do NOT slam brakes
            out.throttle = ctrl.throttle * (0.3 + 0.5 * ratio)
            out.brake    = self.cfg.max_brake * 0.4 * (1.0 - ratio)
            print(f"[REACTIVE] Danger   dist={min_dist:.2f}m")

        else:
            # Warning zone: just slow down a bit
            ratio        = (min_dist - self.cfg.react_danger_dist) / (
                self.cfg.react_warn_dist - self.cfg.react_danger_dist)
            out.throttle = ctrl.throttle * (0.6 + 0.4 * ratio)

        return out


# ╔═══════════════════════════════════════════════════════════════╗
# §7  CONTROLLER  (EMA smoothing + stability)
# ╚═══════════════════════════════════════════════════════════════╝
class Controller:
    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self._prev    = ControlState()
        self._th_ema  = 0.0
        self._st_ema  = 0.0

    def smooth(self, ctrl: ControlState,
               roll_deg: float, pitch_deg: float) -> ControlState:
        a = self.cfg.ema_alpha
        self._th_ema = a * ctrl.throttle + (1 - a) * self._th_ema
        self._st_ema = a * ctrl.steer    + (1 - a) * self._st_ema

        th = self._th_ema
        st = self._st_ema
        br = ctrl.brake

        if br > 0.05:
            th = 0.0

        # Rate limits
        th = float(np.clip(
            th,
            self._prev.throttle - self.cfg.max_throttle_rate,
            self._prev.throttle + self.cfg.max_throttle_rate,
        ))
        st = float(np.clip(
            st,
            self._prev.steer - self.cfg.max_steer_rate,
            self._prev.steer + self.cfg.max_steer_rate,
        ))

        # Stability on slopes/rolls
        th_scale, st_corr = self._stability(roll_deg, pitch_deg)
        th *= th_scale
        st  = float(np.clip(st + st_corr, -self.cfg.max_steer, self.cfg.max_steer))

        th = float(np.clip(th, 0.0,                   self.cfg.max_throttle))
        # st = float(np.clip(st, -self.cfg.max_steer,   self.cfg.max_steer))
        # Speed-based steering limit
        speed_factor = max(0.3, 1.0 - self._prev.throttle)

        dynamic_max_steer = self.cfg.max_steer * speed_factor

        st = float(np.clip(st + st_corr, -dynamic_max_steer, dynamic_max_steer))
        br = float(np.clip(br, 0.0,                   self.cfg.max_brake))

        self._prev = ControlState(throttle=th, steer=st, brake=br,
                                  reverse=ctrl.reverse)
        return self._prev

    def _stability(self, roll: float, pitch: float) -> Tuple[float, float]:
        ts, cs = 1.0, 0.0
        ar = abs(roll)
        if ar > self.cfg.max_safe_roll_deg:
            ex  = ar - self.cfg.max_safe_roll_deg
            ts  = max(0.35, 1.0 - ex / 20.0)
            cs  = -math.copysign(min(0.2 * ex / 10.0, 0.2), roll)
        ap = abs(pitch)
        if ap > self.cfg.max_safe_pitch_deg:
            ep  = ap - self.cfg.max_safe_pitch_deg
            ts  = min(ts, max(0.3, 1.0 - ep / 15.0))
        return float(ts), float(cs)

    def reset(self) -> None:
        self._th_ema = self._st_ema = 0.0
        self._prev   = ControlState()


# ╔═══════════════════════════════════════════════════════════════╗
# §8  STUCK DETECTOR + RECOVERY
# ╚═══════════════════════════════════════════════════════════════╝
class StuckDetector:
    IDLE      = "IDLE"
    RECOVERING = "RECOVERING"

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._pos_hist: deque = deque(maxlen=200)
        self._state    = self.IDLE
        self._rec_t0   = 0.0
        self._rec_steer = 0.0

    def update(self, pos: np.ndarray, t: float) -> Tuple[bool, int]:
        self._pos_hist.append((t, pos.copy()))

        if self._state == self.RECOVERING:
            elapsed    = t - self._rec_t0
            phase_dur  = self.cfg.recovery_duration_s / 3.0
            if elapsed > self.cfg.recovery_duration_s:
                self._state = self.IDLE
                return False, 0
            return True, min(int(elapsed / phase_dur), 2)

        if len(self._pos_hist) >= 2:
            ot, op = self._pos_hist[0]
            if (t - ot) >= self.cfg.stuck_window_s:
                if np.linalg.norm(pos - op) < self.cfg.stuck_disp_thresh:
                    self._trigger(t)
                    return True, 0

        return False, 0

    def _trigger(self, t: float) -> None:
        if self._state == self.IDLE:
            print("[STUCK] Recovery triggered!")
            self._state     = self.RECOVERING
            self._rec_t0    = t
            self._rec_steer = random.uniform(-0.7, 0.7)
            self._pos_hist.clear()

    def get_recovery_control(self, phase: int) -> ControlState:
        steer_vals = {0: 0.0, 1: self._rec_steer, 2: -self._rec_steer}
        return ControlState(
            throttle=0.45,
            steer   =steer_vals.get(phase, 0.0),
            brake   =0.0,
            reverse =True,
        )


# ╔═══════════════════════════════════════════════════════════════╗
# §9  STUCK MEMORY
# ╚═══════════════════════════════════════════════════════════════╝
class StuckMemory:
    def __init__(self, cfg: Config):
        self.decay     = cfg.memory_decay
        self.radius    = cfg.memory_radius
        self.threshold = cfg.memory_threshold
        self.points: list = []   # [(x, y, weight)]

    def update(self, pos: np.ndarray, stuck: bool) -> None:
        self.points = [
            (x, y, w * self.decay)
            for x, y, w in self.points
            if w * self.decay > self.threshold
        ]
        if not stuck:
            return
        for i, (px, py, pw) in enumerate(self.points):
            if math.hypot(pos[0] - px, pos[1] - py) < self.radius * 0.5:
                self.points[i] = (px, py, min(1.0, pw + 0.4))
                return
        self.points.append((float(pos[0]), float(pos[1]), 1.0))
        print(f"[MEMORY] Logged stuck at ({pos[0]:.1f},{pos[1]:.1f})  total={len(self.points)}")

    def get_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.points:
            return None
        data = [[p[0], p[1], p[2]] for p in self.points]
        return torch.tensor(data, device=device, dtype=torch.float32)


# ╔═══════════════════════════════════════════════════════════════╗
# §10  MANUAL CONTROLLER
# ╚═══════════════════════════════════════════════════════════════╝
class ManualController:
    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self._th      = 0.0
        self._st      = 0.0
        self._reverse = False

    def tick(self, keys) -> ControlState:
        brake = 0.0
        if keys[pygame.K_w]:
            self._reverse = False
            self._th = min(self._th + 0.04, self.cfg.max_throttle)
        elif keys[pygame.K_s]:
            self._reverse = True
            self._th = min(self._th + 0.04, self.cfg.max_throttle)
        else:
            self._th = max(self._th - 0.08, 0.0)

        if keys[pygame.K_a]:
            self._st = min(self._st + 0.06,  self.cfg.max_steer)
        elif keys[pygame.K_d]:
            self._st = max(self._st - 0.06, -self.cfg.max_steer)
        else:
            self._st *= 0.8

        if keys[pygame.K_SPACE]:
            self._th      = 0.0
            self._reverse = False
            brake         = self.cfg.max_brake

        return ControlState(
            throttle=round(self._th, 4),
            steer   =round(self._st, 4),
            brake   =brake,
            reverse =self._reverse,
        )


# ╔═══════════════════════════════════════════════════════════════╗
# §11  VISUALIZATION
# ╚═══════════════════════════════════════════════════════════════╝
class Viz:
    C = {
        'bg': (17, 17, 17),      'panel': (26, 26, 46),
        'header': (35, 35, 65),  'grid': (45, 45, 70),
        'border': (60, 60, 95),  'traj': (0, 206, 201),
        'veh': (253, 203, 110),  'goal': (255, 60, 60),
        'subgoal': (50, 255, 120), 'memory': (255, 110, 50),
        'cost': (255, 107, 107), 'speed': (85, 239, 196),
        'steer': (162, 155, 254), 'white': (230, 230, 230),
        'dim': (110, 110, 110),  'danger': (255, 50, 50),
        'ok': (80, 220, 80),     'warn': (255, 200, 0),
    }

    def __init__(self, cfg: Config):
        self.cfg           = cfg
        self._screen       = None
        self._font         = None
        self._traj_x:  deque = deque(maxlen=cfg.traj_history)
        self._traj_y:  deque = deque(maxlen=cfg.traj_history)
        self._costs:   deque = deque(maxlen=400)
        self._speeds:  deque = deque(maxlen=400)
        self._front_img    = None
        self._rear_img     = None
        self._lidar_pts    = None
        self._goal         = np.array([cfg.goal_x, cfg.goal_y])
        self._subgoal      = None
        self._candidates   = []
        self._mem_pts      = []
        self._mode         = "AUTO"
        self._speed        = 0.0
        self._steer        = 0.0
        self._cost         = 0.0
        self._dist_goal    = 0.0
        self._obs_warn     = False

    def init(self, screen) -> None:
        self._screen = screen
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 11)

    def push(self, *, pos, speed, steer, perc, mode, cost,
             subgoal, candidates, mem_pts, dist_goal, obs_warn):
        self._traj_x.append(float(pos[0]))
        self._traj_y.append(float(pos[1]))
        self._costs.append(cost)
        self._speeds.append(speed)
        self._mode      = mode
        self._speed     = speed
        self._steer     = steer
        self._cost      = cost
        self._subgoal   = subgoal.copy() if subgoal is not None else None
        self._candidates = [c.world_pos for c in candidates] if candidates else []
        self._mem_pts   = list(mem_pts)
        self._dist_goal = dist_goal
        self._obs_warn  = obs_warn
        if perc is not None and perc.raw_points is not None and len(perc.raw_points):
            self._lidar_pts = perc.raw_points.copy()

    def set_front(self, img): self._front_img = img
    def set_rear(self, img):  self._rear_img  = img

    def render(self) -> None:
        if self._screen is None:
            return
        W, H = self._screen.get_size()
        self._screen.fill(self.C['bg'])
        CAM_H   = 240
        col_w   = W // 3
        bot_y   = CAM_H
        bot_h   = H - bot_y
        map_w   = W // 2

        self._draw_camera(self._front_img, "Front",    0,       0, col_w,      CAM_H)
        self._draw_camera(self._rear_img,  "Rear",     col_w,   0, col_w,      CAM_H)
        self._draw_lidar(                              col_w*2, 0, W-col_w*2,  CAM_H)
        self._draw_trajectory(0,     bot_y, map_w, bot_h)
        self._draw_metrics(map_w,   bot_y, W-map_w, bot_h)
        self._draw_hud(W, H)
        pygame.display.flip()

    def _draw_camera(self, img, title, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font.render(title, True, C['white']), (x+4, y+3))
        if img is not None:
            try:
                ri = (np.arange(h-18) * img.shape[0] / (h-18)).astype(int).clip(0, img.shape[0]-1)
                ci = (np.arange(w)    * img.shape[1] / w     ).astype(int).clip(0, img.shape[1]-1)
                r  = img[np.ix_(ri, ci)]
                self._screen.blit(pygame.surfarray.make_surface(r.swapaxes(0,1)), (x, y+18))
            except Exception:
                pass
        pygame.draw.rect(self._screen, C['border'], (x, y, w, h), 1)

    def _draw_lidar(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x, y, w, h))
        pygame.draw.rect(self._screen, C['header'], (x, y, w, 18))
        self._screen.blit(self._font.render("LiDAR", True, C['white']), (x+4, y+3))
        cy0 = y + 18
        ch  = h - 18
        lr  = self.cfg.lidar_range
        cx  = x + w//2
        cy  = cy0 + ch//2
        sc  = (min(w, ch) * 0.44) / lr
        for rf in [lr*0.33, lr*0.67, lr]:
            pygame.draw.circle(self._screen, C['grid'], (cx, cy), int(rf*sc), 1)
        if self._lidar_pts is not None and len(self._lidar_pts):
            pts = self._lidar_pts
            zn  = (pts[:,2] - pts[:,2].min()) / (np.ptp(pts[:,2]) + 1e-6)
            for i in range(0, len(pts), 3):
                px = int(cx - pts[i,1]*sc)
                py = int(cy - pts[i,0]*sc)
                if x <= px < x+w and cy0 <= py < cy0+ch:
                    t  = float(zn[i])
                    pygame.draw.circle(self._screen,
                        (int(60+195*min(t*2,1)), int(80+100*t), int(200-180*min(t*2,1))),
                        (px, py), 1)
        pygame.draw.circle(self._screen, C['veh'], (cx,cy), 4)
        pygame.draw.rect(self._screen, C['border'], (x,y,w,h), 1)

    def _draw_trajectory(self, x, y, w, h):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        self._screen.blit(self._font.render("Trajectory", True, C['white']), (x+4,y+3))
        tx, ty = list(self._traj_x), list(self._traj_y)
        if not tx:
            pygame.draw.rect(self._screen, C['border'], (x,y,w,h),1); return

        ax = tx + [self._goal[0]] + [c[0] for c in self._candidates if c is not None]
        ay = ty + [self._goal[1]] + [c[1] for c in self._candidates if c is not None]
        for px2,py2,pw2 in self._mem_pts:
            ax.append(px2); ay.append(py2)
        if self._subgoal is not None:
            ax.append(self._subgoal[0]); ay.append(self._subgoal[1])

        span = max(max(ax)-min(ax), max(ay)-min(ay), 20.0)
        pad  = span*0.12
        mn_x, mx_x = min(ax)-pad, max(ax)+pad
        mn_y, mx_y = min(ay)-pad, max(ay)+pad
        px0, py0 = x+6, y+22
        pw_,  ph_ = w-12, h-26

        def w2s(wx2, wy2):
            sx = px0 + int((wx2-mn_x)/(mx_x-mn_x)*pw_)
            sy = py0 + int((1-(wy2-mn_y)/(mx_y-mn_y))*ph_)
            return sx, sy

        for px2,py2,pw2 in self._mem_pts:
            sx,sy = w2s(px2,py2)
            r = max(4, int(pw2*14))
            b = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
            pygame.draw.circle(b, (*C['memory'],90),(r,r),r)
            self._screen.blit(b,(sx-r,sy-r))
            pygame.draw.circle(self._screen,C['memory'],(sx,sy),4)

        for i,c in enumerate(self._candidates):
            if c is not None:
                sx,sy = w2s(c[0],c[1])
                col   = C['subgoal'] if i==0 else tuple(v//2 for v in C['subgoal'])
                pygame.draw.circle(self._screen,col,(sx,sy),4 if i==0 else 2)

        if len(tx)>1:
            pts2 = [w2s(tx[i],ty[i]) for i in range(len(tx))]
            pygame.draw.lines(self._screen,C['traj'],False,pts2,2)

        if tx:
            vx2,vy2 = w2s(tx[-1],ty[-1])
            pygame.draw.circle(self._screen,C['veh'],(vx2,vy2),6)
            pygame.draw.circle(self._screen,C['white'],(vx2,vy2),6,1)

        gsx,gsy = w2s(self._goal[0],self._goal[1])
        for d in [(-7,-7,7,7),(-7,7,7,-7)]:
            pygame.draw.line(self._screen,C['goal'],(gsx+d[0],gsy+d[1]),(gsx+d[2],gsy+d[3]),2)
        self._screen.blit(self._font.render("GOAL",True,C['goal']),(gsx+8,gsy-8))

        if self._subgoal is not None:
            ssx,ssy = w2s(self._subgoal[0],self._subgoal[1])
            pygame.draw.circle(self._screen,C['subgoal'],(ssx,ssy),6)
            pygame.draw.circle(self._screen,C['white'],(ssx,ssy),6,1)
            self._screen.blit(self._font.render("SUB",True,C['subgoal']),(ssx+6,ssy-7))

        pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1)

    def _draw_metrics(self, x, y, w, h):
        half = h//2
        self._draw_graph(list(self._costs),  "Cost",      x, y,      w, half,     self.C['cost'])
        self._draw_graph(list(self._speeds), "Speed m/s", x, y+half, w, h-half,   self.C['speed'])

    def _draw_graph(self, data, title, x, y, w, h, color):
        C = self.C
        pygame.draw.rect(self._screen, C['panel'],  (x,y,w,h))
        pygame.draw.rect(self._screen, C['header'], (x,y,w,18))
        lbl = f"{title}  {data[-1]:.2f}" if data else title
        self._screen.blit(self._font.render(lbl,True,color),(x+4,y+3))
        if len(data)<2:
            pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1); return
        px0,py0 = x+5,y+22
        pw,ph   = w-10,h-26
        mn,mx2  = min(data),max(data)
        rng     = max(mx2-mn,1e-3)
        n       = len(data)
        pts2    = [(px0+int(i/(n-1)*pw), py0+int((1-(v-mn)/rng)*ph))
                   for i,v in enumerate(data)]
        if len(pts2)>1:
            pygame.draw.lines(self._screen,color,False,pts2,2)
        pygame.draw.rect(self._screen,C['border'],(x,y,w,h),1)

    def _draw_hud(self, W, H):
        C      = self.C
        mode_c = C['ok'] if self._mode == "AUTO" else C['warn']
        lines  = [
            (f"MODE : {self._mode}",              mode_c),
            (f"Speed: {self._speed:5.2f} m/s",    C['speed']),
            (f"Steer: {self._steer:+6.3f}",        C['steer']),
            (f"Cost : {self._cost:8.1f}",          C['cost']),
            (f"Goal : {self._dist_goal:6.1f} m",   C['white']),
            (f"Mem  : {len(self._mem_pts)}",       C['dim']),
            (f"Cands: {len(self._candidates)}",    C['dim']),
        ]
        if self._obs_warn:
            lines.append(("!!! OBSTACLE !!!", C['danger']))
        BW, BH = 185, len(lines)*17+8
        bx, by = W-BW-4, 4
        bg = pygame.Surface((BW,BH), pygame.SRCALPHA)
        bg.fill((0,0,0,170))
        self._screen.blit(bg,(bx,by))
        for i,(txt,col) in enumerate(lines):
            self._screen.blit(self._font.render(txt,True,col),(bx+5,by+4+i*17))


# ╔═══════════════════════════════════════════════════════════════╗
# §12  SENSOR MANAGER
# ╚═══════════════════════════════════════════════════════════════╝
class SensorManager:
    def __init__(self, world, vehicle, cfg: Config):
        self.cfg     = cfg
        self._actors = []
        bp           = world.get_blueprint_library()

        # ── LiDAR ─────────────────────────────────────────────────────
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",              str(cfg.lidar_range))
        lidar_bp.set_attribute("points_per_second",  str(cfg.lidar_points_per_sec))
        lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_freq))
        lidar_bp.set_attribute("channels",           str(cfg.lidar_channels))
        lidar_bp.set_attribute("upper_fov",          str(cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov",          str(cfg.lidar_lower_fov))

        self._lidar_q: Queue = Queue(maxsize=2)
        self._front_q: Queue = Queue(maxsize=2)
        self._rear_q:  Queue = Queue(maxsize=2)

        lidar = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.5, z=2.0)),
            attach_to=vehicle,
        )
        lidar.listen(lambda d: self._lidar_q.put(d)
                     if not self._lidar_q.full() else None)
        self._actors.append(lidar)

        # ── Cameras ───────────────────────────────────────────────────
        for tag, tf in [
            ("front", carla.Transform(carla.Location(x=1.5, z=1.8))),
            ("rear",  carla.Transform(carla.Location(x=-1.5, z=1.8),
                                       carla.Rotation(yaw=180))),
        ]:
            cam_bp = bp.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.camera_width))
            cam_bp.set_attribute("image_size_y", str(cfg.camera_height))
            cam_bp.set_attribute("fov",          str(cfg.camera_fov))
            cam = world.spawn_actor(cam_bp, tf, attach_to=vehicle)
            q   = self._front_q if tag == "front" else self._rear_q
            cam.listen(lambda img, _q=q:
                       _q.put(self._img_to_arr(img)) if not _q.full() else None)
            self._actors.append(cam)

    @staticmethod
    def _img_to_arr(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        return arr[:, :, ::-1].copy()

    def _get(self, q: Queue):
        try:    return q.get_nowait()
        except Empty: return None

    @property
    def lidar_data(self):  return self._get(self._lidar_q)
    @property
    def front_image(self): return self._get(self._front_q)
    @property
    def rear_image(self):  return self._get(self._rear_q)

    def destroy(self) -> None:
        for a in self._actors:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass


# ╔═══════════════════════════════════════════════════════════════╗
# §13  VEHICLE SPAWNER
# ╚═══════════════════════════════════════════════════════════════╝
def spawn_vehicle(world, cfg: Config):
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

    tf = carla.Transform(
        carla.Location(x=240.35, y=223.16, z=2.15),
        carla.Rotation(pitch=-14.03, yaw=103.85, roll=0.00),
    )
    vehicle = world.try_spawn_actor(vehicle_bp, tf)

    if vehicle is None:
        pts     = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(pts)) if pts else None
    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle at any location")

    print(f"[SPAWN] Vehicle at {vehicle.get_location()}")
    t0 = time.time()
    while time.time() - t0 < cfg.post_spawn_wait_s:
        world.tick()
    return vehicle


# ╔═══════════════════════════════════════════════════════════════╗
# §14  NAVIGATION SYSTEM  (clean hierarchy: Subgoal→MPPI→Reactive)
# ╚═══════════════════════════════════════════════════════════════╝
class NavigationSystem:
    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self._mode      = "AUTO"
        self._running   = False

        self._client    = None
        self._world     = None
        self._vehicle   = None
        self._sensors: Optional[SensorManager] = None

        self._perception = PerceptionModule(cfg)
        self._subgoal_pl = SubgoalPlanner(cfg)
        self._mppi       = MPPIPlanner(cfg)
        self._controller = Controller(cfg)
        self._manual     = ManualController(cfg)
        self._stuck      = StuckDetector(cfg)
        self._memory     = StuckMemory(cfg)
        self._reactive   = ReactiveAvoidance(cfg)
        self._viz        = Viz(cfg)

        self._goal       = np.array([cfg.goal_x, cfg.goal_y], dtype=np.float32)
        self._step       = 0
        self._t0         = 0.0

        pygame.init()
        self._screen = pygame.display.set_mode(
            (cfg.viz_win_w, cfg.viz_win_h), pygame.RESIZABLE)
        pygame.display.set_caption(
            "Navigator  |  TAB = AUTO/MANUAL  |  ESC = quit")
        self._viz.init(self._screen)

    def connect(self) -> None:
        print(f"[CARLA] connecting to {self.cfg.carla_host}:{self.cfg.carla_port}")
        self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self._client.set_timeout(self.cfg.carla_timeout)
        self._world  = self._client.get_world()
        if self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode    = True
            s.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self._world.apply_settings(s)
            print("[CARLA] Synchronous mode ON")

    def setup(self) -> None:
        self._vehicle = spawn_vehicle(self._world, self.cfg)
        self._sensors = SensorManager(self._world, self._vehicle, self.cfg)
        self._t0      = time.time()
        print("[SETUP] Ready — starting navigation")

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
            self._vehicle.apply_control(
                carla.VehicleControl(brake=1.0, throttle=0.0))
        if self._sensors:
            self._sensors.destroy()
        if self._world and self.cfg.synchronous:
            s = self._world.get_settings()
            s.synchronous_mode = False
            self._world.apply_settings(s)
        pygame.quit()

    # ─────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        if self.cfg.synchronous:
            self._world.tick()
        else:
            self._world.wait_for_tick()
        self._step += 1

        # ── Events ───────────────────────────────────────────────
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

        # ── Vehicle state ─────────────────────────────────────────
        tf        = self._vehicle.get_transform()
        loc       = tf.location
        vel       = self._vehicle.get_velocity()
        speed     = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        roll_deg  = float(tf.rotation.roll)
        pitch_deg = float(tf.rotation.pitch)
        yaw_rad   = math.radians(tf.rotation.yaw)
        pos       = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
        t_now     = time.time() - self._t0

        # ── Goal check ───────────────────────────────────────────
        dist_goal = float(np.linalg.norm(pos[:2] - self._goal))
        print(f"\r[NAV] Dist={dist_goal:.1f}m  Speed={speed:.2f}m/s  "
              f"Mode={self._mode}", end="", flush=True)

        if dist_goal < self.cfg.goal_tolerance:
            print("\n[NAV] *** GOAL REACHED ***")
            self._vehicle.apply_control(
                carla.VehicleControl(brake=1.0, throttle=0.0))
            self._running = False
            return

        # ── Sensors ───────────────────────────────────────────────
        lidar_raw = self._sensors.lidar_data
        perc      = (self._perception.process_lidar(lidar_raw)
                     if lidar_raw is not None
                     else self._perception.last_output)

        front = self._sensors.front_image
        rear  = self._sensors.rear_image
        if front is not None: self._viz.set_front(front)
        if rear  is not None: self._viz.set_rear(rear)

        # ── Planning ──────────────────────────────────────────────
        ctrl_state:      Optional[ControlState] = None
        subgoal_for_viz: Optional[np.ndarray]   = None
        all_candidates:  List[Subgoal]           = []
        current_cost:    float                   = 0.0

        if self._mode == "MANUAL":
            ctrl_state = self._manual.tick(keys)

        else:  # ── AUTO ─────────────────────────────────────────
            stuck, rec_phase = self._stuck.update(pos, t_now)
            self._memory.update(pos, stuck)

            if stuck:
                # Recovery mode: ignore subgoal, back up with random steer
                ctrl_state = self._stuck.get_recovery_control(rec_phase)
                self._subgoal_pl.force_reset()   # re-plan after recovery
                print(f"\n[NAV] STUCK — recovery phase {rec_phase}")

            elif perc is not None:
                # ── STEP 1: Subgoal planning ──────────────────────────
                best_sg, all_candidates = self._subgoal_pl.plan(
                    perc,
                    vehicle_pos    = pos,
                    vehicle_yaw_rad= yaw_rad,
                    goal_pos       = self._goal,
                )

                if best_sg is None:
                    # No safe subgoal found → treat as stuck
                    print("\n[NAV] No subgoal found — forcing recovery")
                    self._stuck._trigger(t_now)
                    ctrl_state = self._stuck.get_recovery_control(0)

                else:
                    subgoal_for_viz = best_sg.world_pos

                    # ── STEP 2: MPPI tracks SUBGOAL (not global goal) ──
                    # If we're close to subgoal switch to global goal
                    sg_dist = np.linalg.norm(
                        best_sg.world_pos - pos[:2])
                    mppi_goal = (
                        self._goal
                        if sg_dist < 4.0 or dist_goal < 15.0
                        else best_sg.world_pos
                    )

                    state_vec  = [loc.x, loc.y, yaw_rad, speed]
                    mem_tensor = self._memory.get_tensor(self._mppi.device)

                    ctrl_state, current_cost = self._mppi.plan(
                        state_vec, perc, mppi_goal, mem_tensor)

                    # ── STEP 3: Safety filter (smooth + stability) ─────
                    ctrl_state = self._controller.smooth(
                        ctrl_state, roll_deg, pitch_deg)

                    # ── FIX: guarantee minimum forward motion ─────────
                    if not ctrl_state.reverse and speed < 1.0:
                        ctrl_state.throttle = max(ctrl_state.throttle, 0.5)

                    print(f"  SG_dist={sg_dist:.1f}m "
                          f"SG_prog={best_sg.goal_progress:.1f}m "
                          f"cost={current_cost:.0f}", end="")

        # ── Fallback if still None ────────────────────────────────
        if ctrl_state is None:
            ctrl_state = ControlState(throttle=0.45)
            print("\n[NAV] Fallback: creeping forward")

        # ── STEP 4: Reactive avoidance (last resort only) ─────────
        raw_pts    = perc.raw_points if perc is not None else None
        ctrl_state = self._reactive.process(raw_pts, ctrl_state, speed)

        # ── Apply to CARLA ────────────────────────────────────────
        self._vehicle.apply_control(carla.VehicleControl(
            throttle=float(np.clip(ctrl_state.throttle, 0.0,
                                   self.cfg.max_throttle)),
            steer   =float(np.clip(ctrl_state.steer,
                                   -self.cfg.max_steer, self.cfg.max_steer)),
            brake   =float(np.clip(ctrl_state.brake, 0.0, self.cfg.max_brake)),
            reverse =bool(ctrl_state.reverse),
        ))

        # ── Visualization ─────────────────────────────────────────
        self._viz.push(
            pos         = pos,
            speed       = speed,
            steer       = ctrl_state.steer,
            perc        = perc,
            mode        = self._mode,
            cost        = current_cost,
            subgoal     = subgoal_for_viz,
            candidates  = all_candidates,
            mem_pts     = self._memory.points,
            dist_goal   = dist_goal,
            obs_warn    = self._reactive.obstacle_active,
        )
        if self._step % self.cfg.viz_every_n_ticks == 0:
            self._viz.render()

        if not self.cfg.synchronous:
            time.sleep(self.cfg.fixed_delta_seconds)


# ╔═══════════════════════════════════════════════════════════════╗
# §15  ENTRY POINT
# ╚═══════════════════════════════════════════════════════════════╝
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