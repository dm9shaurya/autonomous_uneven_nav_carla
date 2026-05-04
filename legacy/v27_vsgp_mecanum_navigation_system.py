"""
================================================================================
CARLA VSGP Mapless Navigation System
================================================================================
A complete autonomous navigation system for CARLA using:
  - Variational Sparse Gaussian Process (VSGP) for terrain perception
  - Subgoal-based planner with a rich cost function
  - Mecanum-wheel kinematic model (adapted to CARLA non-holonomic control)
  - Stuck detection and recovery
  - Manual holonomic keyboard control mode
  - Full 6-panel live visualization

Vehicle: vehicle.tesla.cybertruck
Python:  3.7+
Deps:    carla, numpy, scipy, matplotlib, pygame, scikit-learn

Run:
    python carla_vsgp_navigator.py
================================================================================
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

import numpy as np
import pygame
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────────────
# Optional scipy / sklearn imports (used in VSGP)
# ──────────────────────────────────────────────────────────────────────────────
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

# ╔══════════════════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── CARLA connection ───────────────────────────────────────────────────────
    carla_host: str          = "127.0.0.1"
    carla_port: int          = 2000
    carla_timeout: float     = 20.0
    synchronous: bool        = True
    fixed_delta_seconds: float = 0.05   # 20 Hz physics

    # ── Vehicle ────────────────────────────────────────────────────────────────
    vehicle_blueprint: str   = "vehicle.tesla.cybertruck"
    spawn_z_offset: float    = 5.0
    post_spawn_wait_s: float = 3.0

    # ── Sensors ───────────────────────────────────────────────────────────────
    lidar_range: float       = 20.0
    lidar_points_per_sec: int = 100_000
    lidar_rotation_freq: float = 20.0
    lidar_channels: int      = 64
    lidar_upper_fov: float   = 15.0
    lidar_lower_fov: float   = -25.0
    camera_width: int        = 320
    camera_height: int       = 240
    camera_fov: int          = 90

    # ── Mecanum geometry ──────────────────────────────────────────────────────
    wheel_radius: float      = 0.15    # rw  [m]
    lx: float                = 0.25    # half wheelbase [m]
    ly: float                = 0.20    # half track width [m]
    max_wheel_omega: float   = 40.0    # rad/s

    # ── Planner ───────────────────────────────────────────────────────────────
    robot_width: float       = 0.8     # [m]
    safety_margin: float     = 0.4     # [m]
    max_slope_deg: float     = 20.0    # degrees
    subgoal_distance: float  = 4.0     # [m] preferred subgoal radius
    n_subgoals_max: int      = 8

    # ── Cost weights ──────────────────────────────────────────────────────────
    w_direction: float       = 1.2
    w_distance: float        = 0.6
    w_steepness: float       = 1.5
    w_collision: float       = 3.0
    w_oscillation: float     = 0.8
    w_flatness: float        = 0.5

    # ── Control ───────────────────────────────────────────────────────────────
    max_throttle: float      = 0.55
    max_steer: float         = 0.6
    max_brake: float         = 0.8
    ema_alpha: float         = 0.35    # smoothing factor
    max_steer_rate: float    = 0.12    # per step
    max_throttle_rate: float = 0.08    # per step
    base_speed: float        = 4.0     # m/s target forward

    # ── Stuck detection ───────────────────────────────────────────────────────
    stuck_window_s: float    = 4.0
    stuck_disp_thresh: float = 0.6     # [m]
    recovery_duration_s: float = 2.5

    # ── VSGP ──────────────────────────────────────────────────────────────────
    vsgp_n_inducing: int     = 64
    vsgp_alpha: float        = 1.0     # RQ kernel alpha
    vsgp_length_scale: float = 0.3
    vsgp_noise_var: float    = 0.05
    vsgp_lr: float           = 0.02    # inducing mean update rate

    # ── Visualization ─────────────────────────────────────────────────────────
    viz_update_hz: float     = 5.0
    traj_history: int        = 400


CFG = Config()


# ╔══════════════════════════════════════════════════════════════════════════════
# §1  MECANUM KINEMATIC MODEL
# ╚══════════════════════════════════════════════════════════════════════════════

class MecanumKinematics:
    """
    Standard four-mecanum-wheel kinematic model.

    Wheel numbering (top view):
        FL=0  FR=1
        RL=2  RR=3

    Forward kinematics:
        [vx, vy, ω]ᵀ = (rw/4) · Jfwd · [ω1,ω2,ω3,ω4]ᵀ

    Inverse kinematics:
        [ω1,ω2,ω3,ω4]ᵀ = (1/rw) · Jinv · [vx,vy,ω]ᵀ
    """

    def __init__(self, cfg: Config = CFG):
        rw = cfg.wheel_radius
        L  = cfg.lx + cfg.ly          # geometric constant

        # ── Forward kinematics matrix (3×4) ───────────────────────────────────
        self.Jfwd = (rw / 4.0) * np.array([
            [ 1,  1,  1,  1],
            [-1,  1,  1, -1],
            [-1/L, 1/L, -1/L, 1/L],
        ], dtype=np.float64)

        # ── Inverse kinematics matrix (4×3) ───────────────────────────────────
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
        # clamp to physical limits
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

        CARLA is non-holonomic: no direct lateral velocity.
        Approximation:
          throttle ← ||(vx, vy)|| / v_max
          steer    ← omega component + lateral bias from vy
          brake    ← 0 unless reversing
        """
        speed = math.sqrt(vx ** 2 + vy ** 2)
        throttle = np.clip(speed / cfg.base_speed, 0.0, cfg.max_throttle)
        brake    = 0.0

        if vx < -0.1:                   # reversing
            throttle = np.clip(-vx / cfg.base_speed, 0.0, cfg.max_throttle)
            brake    = 0.0

        # steering: heading correction (omega) + lateral bias (vy influence)
        steer_omega  = omega / (cfg.base_speed + 1e-6)   # normalise
        steer_lateral = -vy / (cfg.base_speed + 1e-6) * 0.4
        steer = np.clip(steer_omega + steer_lateral,
                        -cfg.max_steer, cfg.max_steer)

        return float(throttle), float(steer), float(brake)


# ╔══════════════════════════════════════════════════════════════════════════════
# §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# ╚══════════════════════════════════════════════════════════════════════════════

class RationalQuadraticKernel:
    """
    k(x,x') = σ² · (1 + ||x-x'||² / (2·α·l²))^(-α)

    Parameters
    ----------
    alpha      : shape parameter
    length_scale : characteristic length
    variance   : signal variance σ²
    """

    def __init__(self, alpha: float = 1.0,
                 length_scale: float = 0.3,
                 variance: float = 1.0):
        self.alpha        = alpha
        self.length_scale = length_scale
        self.variance     = variance

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Returns (n,m) kernel matrix."""
        # squared Euclidean distances
        diff  = X[:, None, :] - Y[None, :, :]        # (n,m,d)
        sq_d  = np.sum(diff ** 2, axis=-1)            # (n,m)
        denom = 2.0 * self.alpha * self.length_scale ** 2
        return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Returns (n,) diagonal k(x,x) = σ²."""
        return self.variance * np.ones(len(X))


class VSGP:
    """
    Variational Sparse Gaussian Process.

    Inputs:   (alpha_angle, beta_angle)  – spherical angles from LiDAR
    Output:   f(z) = roc - r            – signed occupancy

    Uses m inducing points Z ⊂ ℝ² to approximate the full GP posterior.
    Posterior predictive:
        q(f*)  =  N( Ks* Kuu⁻¹ mu,
                     Kss - Ks* Kuu⁻¹ (Kuu - Su) Kuu⁻¹ Ks*ᵀ )

    where  mu, Su  are the variational parameters (updated online via SGD-style).
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg    = cfg
        self.kernel = RationalQuadraticKernel(
            alpha        = cfg.vsgp_alpha,
            length_scale = cfg.vsgp_length_scale,
            variance     = 1.0,
        )
        self.noise_var  = cfg.vsgp_noise_var
        self.n_ind      = cfg.vsgp_n_inducing
        self.lr         = cfg.vsgp_lr

        # Inducing inputs Z: (m, 2)
        # Initialised on a uniform grid in spherical angle space
        a_vals = np.linspace(-np.pi / 2, np.pi / 2, int(math.sqrt(self.n_ind)))
        b_vals = np.linspace(-np.pi / 4, np.pi / 4, int(math.sqrt(self.n_ind)))
        A, B   = np.meshgrid(a_vals, b_vals)
        self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

        # Variational parameters
        self.mu = np.zeros(self.n_ind)          # variational mean
        self.Su = np.eye(self.n_ind) * 0.1      # variational covariance

        # Caches
        self._Kuu       : Optional[np.ndarray] = None
        self._Kuu_inv   : Optional[np.ndarray] = None
        self._trained   : bool = False

    # ── inducing point selection ───────────────────────────────────────────────

    def _select_inducing_points(self, X: np.ndarray) -> None:
        """Choose inducing points from data using K-Means or random."""
        if SKLEARN_OK and len(X) >= self.n_ind:
            km = KMeans(n_clusters=self.n_ind, n_init=3, max_iter=50)
            km.fit(X)
            self.Z = km.cluster_centers_.copy()
        else:
            idx = np.random.choice(len(X), min(self.n_ind, len(X)), replace=False)
            self.Z = X[idx].copy()
        self.mu[:] = 0.0
        self.Su[:] = np.eye(self.n_ind) * 0.1

    # ── update step ───────────────────────────────────────────────────────────

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform a single variational update from a mini-batch (X, y).

        X: (n,2)  spherical angles (alpha, beta)
        y: (n,)   occupancy labels  roc - r
        """
        if len(X) < 4:
            return

        n = len(X)

        # Re-select inducing points every update for online tracking
        self._select_inducing_points(X)

        Kuu  = self.kernel(self.Z, self.Z) + np.eye(self.n_ind) * 1e-6
        Kfu  = self.kernel(X, self.Z)                   # (n, m)
        Kff_diag = self.kernel.diag(X)                  # (n,)

        Kuu_inv = np.linalg.inv(Kuu)
        A       = Kfu @ Kuu_inv                          # (n, m)

        # ELBO-based natural gradient update (simplified)
        noise_inv = 1.0 / self.noise_var
        Lambda = noise_inv * (A.T @ A) + Kuu_inv        # (m, m)
        rhs    = noise_inv * (A.T @ y)                  # (m,)

        # posterior natural parameters
        Su_new  = np.linalg.inv(Lambda + np.eye(self.n_ind) * 1e-6)
        mu_new  = Su_new @ rhs

        # EMA update (online learning)
        lr = self.lr
        self.Su = (1 - lr) * self.Su + lr * Su_new
        self.mu = (1 - lr) * self.mu + lr * mu_new

        self._Kuu     = Kuu
        self._Kuu_inv = Kuu_inv
        self._trained = True

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and variance at test points Xs (n,2).

        Returns
        -------
        mean : (n,)
        var  : (n,)   – high var ↔ uncertain ↔ navigable candidate
        """
        if not self._trained or self._Kuu_inv is None:
            n = len(Xs)
            return np.zeros(n), np.ones(n)

        Ksu = self.kernel(Xs, self.Z)                    # (n, m)
        A   = Ksu @ self._Kuu_inv                        # (n, m)

        mean    = A @ self.mu                            # (n,)

        # variance: prior - explained + variational
        Kss_diag = self.kernel.diag(Xs)                 # (n,)
        var_explained = np.sum(A * (A @ self._Kuu), axis=1)  # (n,)
        var_variational = np.sum(A @ self.Su * A, axis=1)    # (n,)
        var = Kss_diag - var_explained + var_variational + self.noise_var
        var = np.clip(var, 1e-6, None)

        return mean, var


# ╔══════════════════════════════════════════════════════════════════════════════
# §3  PERCEPTION MODULE
# ╚══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerceptionOutput:
    """Structured output from the perception module."""
    alpha_grid  : np.ndarray            # (H, W)  azimuth angles
    beta_grid   : np.ndarray            # (H, W)  elevation angles
    mean_surface: np.ndarray            # (H, W)  VSGP mean
    var_surface : np.ndarray            # (H, W)  VSGP variance
    free_mask   : np.ndarray            # (H, W)  bool: navigable
    raw_points  : np.ndarray            # (N, 3)  xyz of LiDAR hit


class PerceptionModule:
    """
    Converts raw LiDAR point cloud → VSGP occupancy surface.

    Pipeline
    --------
    1. Receive carla.LidarMeasurement
    2. Convert to numpy xyz
    3. Convert to spherical (alpha, beta, r)
    4. Compute occupancy label: occ = roc - r
       roc = a reference occupancy radius (constant here)
    5. Train VSGP online
    6. Predict mean + variance on a regular spherical grid
    7. Threshold variance → free-space mask
    """

    ALPHA_RES  = 60    # grid columns  (azimuth)
    BETA_RES   = 30    # grid rows     (elevation)
    ROC        = 10.0  # reference occupancy radius [m]
    VAR_FREE_THRESH = 0.35   # variance above this → "navigable" candidate

    def __init__(self, cfg: Config = CFG):
        self.cfg  = cfg
        self.vsgp = VSGP(cfg)
        self._last_output: Optional[PerceptionOutput] = None

        # Pre-compute angular grids
        self._a_lin = np.linspace(-math.pi / 2, math.pi / 2, self.ALPHA_RES)
        self._b_lin = np.linspace(-math.pi / 6, math.pi / 6, self.BETA_RES)
        AG, BG = np.meshgrid(self._a_lin, self._b_lin)
        self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
        self._AG = AG
        self._BG = BG

    # ── lidar callback data buffer ────────────────────────────────────────────

    def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
        """
        Process a carla.LidarMeasurement.
        Returns PerceptionOutput or None if insufficient data.
        """
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        raw = raw.reshape(-1, 4)[:, :3]          # (N,3) xyz (ignore intensity)

        if len(raw) < 20:
            return self._last_output

        # Filter out ground points and self-hits
        dist = np.linalg.norm(raw, axis=1)
        mask = (dist > 0.5) & (dist < self.cfg.lidar_range - 0.5)
        pts  = raw[mask]

        if len(pts) < 10:
            return self._last_output

        # Spherical conversion
        r     = np.linalg.norm(pts, axis=1)
        alpha = np.arctan2(pts[:, 1], pts[:, 0])   # azimuth
        beta  = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1, 1))  # elevation

        # Occupancy label
        occ   = self.ROC - r                        # positive → occupied region

        # Subsample for efficiency
        n_max = 2000
        if len(alpha) > n_max:
            idx   = np.random.choice(len(alpha), n_max, replace=False)
            alpha, beta, occ = alpha[idx], beta[idx], occ[idx]
            pts   = pts[idx]

        X_train = np.column_stack([alpha, beta])

        # VSGP update
        self.vsgp.update(X_train, occ)

        # Predict on grid
        mean_flat, var_flat = self.vsgp.predict(self._grid_pts)
        mean_surf = mean_flat.reshape(self.BETA_RES, self.ALPHA_RES)
        var_surf  = var_flat.reshape(self.BETA_RES, self.ALPHA_RES)

        # Free-space mask: high variance → uncertain → navigable candidate
        free_mask = var_surf > self.VAR_FREE_THRESH

        out = PerceptionOutput(
            alpha_grid  = self._AG,
            beta_grid   = self._BG,
            mean_surface= mean_surf,
            var_surface = var_surf,
            free_mask   = free_mask,
            raw_points  = pts,
        )
        self._last_output = out
        return out

    @property
    def last_output(self) -> Optional[PerceptionOutput]:
        return self._last_output


# ╔══════════════════════════════════════════════════════════════════════════════
# §4  SUBGOAL GENERATION
# ╚══════════════════════════════════════════════════════════════════════════════

@dataclass
class Subgoal:
    alpha     : float          # azimuth direction [rad]
    beta      : float          # elevation angle   [rad]
    distance  : float          # target distance   [m]
    local_pos : np.ndarray     # (x, y, z) robot frame
    slope_deg : float          # terrain slope magnitude [deg]
    safe      : bool           # passed safety checks
    width_m   : float          # estimated passable width [m]

    # runtime
    cost      : float = 0.0
    world_pos : Optional[np.ndarray] = None


class SubgoalPlanner:
    """
    Extracts navigable segments from the VSGP variance surface and
    proposes ranked subgoals using a multi-term cost function.
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg              = cfg
        self._prev_alpha      : float = 0.0       # for oscillation tracking
        self._alpha_history   : deque = deque(maxlen=8)
        self._prev_subgoal_idx: int   = -1
        self._exploration_bias: float = 0.0       # increases during recovery

    # ── segment extraction ────────────────────────────────────────────────────

    def extract_segments(self, perc: PerceptionOutput) -> List[dict]:
        """
        Scan the middle elevation row(s) of the free_mask for contiguous
        navigable segments.

        Returns list of dicts: {alpha_center, beta_mean, width_alpha, width_m,
                                 slope_deg, alpha_min, alpha_max}
        """
        mid_row = perc.free_mask.shape[0] // 2
        # Use a band of 3 rows around mid
        rows = perc.free_mask[max(0, mid_row-1):mid_row+2, :]
        row  = np.any(rows, axis=0)                 # (W,) bool

        alpha_arr  = perc.alpha_grid[0, :]          # azimuth per column
        beta_arr   = perc.beta_grid[mid_row, :]     # elevation per column
        mean_row   = perc.mean_surface[mid_row, :]

        segments   = []
        in_seg     = False
        seg_start  = 0

        for i, free in enumerate(row):
            if free and not in_seg:
                in_seg = True
                seg_start = i
            elif (not free or i == len(row) - 1) and in_seg:
                seg_end = i
                in_seg  = False

                width_alpha = alpha_arr[seg_end] - alpha_arr[seg_start]
                alpha_ctr   = 0.5 * (alpha_arr[seg_start] + alpha_arr[seg_end])
                beta_mean   = float(np.mean(beta_arr[seg_start:seg_end + 1]))
                slope_deg   = abs(math.degrees(beta_mean))

                # Angular width → metric width at subgoal_distance
                width_m = self.cfg.subgoal_distance * abs(width_alpha)

                segments.append({
                    "alpha_center": float(alpha_ctr),
                    "alpha_min"   : float(alpha_arr[seg_start]),
                    "alpha_max"   : float(alpha_arr[seg_end]),
                    "beta_mean"   : beta_mean,
                    "slope_deg"   : slope_deg,
                    "width_m"     : width_m,
                    "mean_occ"    : float(np.mean(mean_row[seg_start:seg_end])),
                })

        return segments

    # ── subgoal generation from segment ──────────────────────────────────────

    def _segment_to_subgoals(self, seg: dict, cfg: Config) -> List[Subgoal]:
        """Convert a segment dict to one or more Subgoal objects."""
        req_width = cfg.robot_width + cfg.safety_margin
        w         = seg["width_m"]
        safe      = (w >= req_width) and (seg["slope_deg"] < cfg.max_slope_deg)

        goals = []
        d     = cfg.subgoal_distance
        alpha = seg["alpha_center"]
        beta  = seg["beta_mean"]

        def _make(a: float) -> Subgoal:
            lx = d * math.cos(beta) * math.cos(a)
            ly = d * math.cos(beta) * math.sin(a)
            lz = d * math.sin(beta)
            return Subgoal(
                alpha    = a,
                beta     = beta,
                distance = d,
                local_pos= np.array([lx, ly, lz]),
                slope_deg= seg["slope_deg"],
                safe     = safe,
                width_m  = w,
            )

        goals.append(_make(alpha))

        # Extra subgoals for wider segments
        if w > 2.0 * req_width:
            offset = (seg["alpha_max"] - seg["alpha_min"]) * 0.3
            goals.append(_make(alpha - offset))
            goals.append(_make(alpha + offset))

        return goals

    # ── cost function ─────────────────────────────────────────────────────────

    def _compute_cost(self, sg: Subgoal, cfg: Config) -> float:
        """
        J(g) = w1·dir + w2·dist + w3·steep + w4·coll + w5·osc – w6·flat
        """
        # direction cost: deviation from straight-ahead (alpha=0)
        dir_cost  = abs(sg.alpha) / (math.pi / 2)

        # distance cost: prefer targets at the desired distance
        dist_cost = abs(sg.distance - cfg.subgoal_distance) / cfg.subgoal_distance

        # steepness cost
        steep_cost = sg.slope_deg / cfg.max_slope_deg

        # collision cost: penalise narrow passages
        req_w   = cfg.robot_width + cfg.safety_margin
        coll_cost = max(0.0, (req_w - sg.width_m) / req_w) ** 2

        # oscillation penalty: penalise alternating left/right
        osc_penalty = 0.0
        if len(self._alpha_history) >= 2:
            signs = [math.copysign(1, a) for a in self._alpha_history]
            alternations = sum(
                1 for i in range(1, len(signs)) if signs[i] != signs[i - 1]
            )
            osc_penalty = alternations / max(len(signs) - 1, 1)

        # flatness reward (negative cost contribution)
        flatness_reward = 1.0 - steep_cost

        # exploration bias (increases when stuck)
        dir_cost *= max(0.1, 1.0 - self._exploration_bias)

        cost = (
            cfg.w_direction   * dir_cost
            + cfg.w_distance  * dist_cost
            + cfg.w_steepness * steep_cost
            + cfg.w_collision * coll_cost
            + cfg.w_oscillation * osc_penalty
            - cfg.w_flatness  * flatness_reward
        )
        return float(cost)

    # ── main planning call ────────────────────────────────────────────────────

    def plan(self, perc: PerceptionOutput) -> Tuple[Optional[Subgoal], List[Subgoal]]:
        """
        Returns (best_subgoal, all_subgoals).
        best_subgoal is None if no safe option exists.
        """
        segments = self.extract_segments(perc)
        all_goals: List[Subgoal] = []

        for seg in segments:
            all_goals.extend(self._segment_to_subgoals(seg, self.cfg))

        if not all_goals:
            return None, []

        # Compute costs
        safe_goals = [g for g in all_goals if g.safe]
        pool       = safe_goals if safe_goals else all_goals   # fallback

        for g in pool:
            g.cost = self._compute_cost(g, self.cfg)

        pool.sort(key=lambda g: g.cost)
        best = pool[0]

        # Update history
        self._alpha_history.append(best.alpha)
        self._prev_alpha = best.alpha

        return best, all_goals

    def set_exploration_bias(self, v: float) -> None:
        self._exploration_bias = np.clip(v, 0.0, 1.0)


# ╔══════════════════════════════════════════════════════════════════════════════
# §5  CONTROLLER MODULE
# ╚══════════════════════════════════════════════════════════════════════════════

@dataclass
class ControlState:
    throttle : float = 0.0
    steer    : float = 0.0
    brake    : float = 0.0
    reverse  : bool  = False


class Controller:
    """
    Converts planned subgoal → CARLA vehicle control.

    1. Compute desired (vx, vy, ω) from subgoal
    2. Validate with mecanum inverse kinematics
    3. Map back to CARLA control via to_carla_control()
    4. Apply EMA smoothing + rate limiting
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self.mec      = MecanumKinematics(cfg)
        self._state   = ControlState()
        self._vx_ema  = 0.0
        self._vy_ema  = 0.0
        self._om_ema  = 0.0

    def compute(
        self,
        subgoal    : Optional[Subgoal],
        vehicle_speed_ms: float,
        terrain_slope_deg: float = 0.0,
    ) -> ControlState:
        """
        Derive desired twist from subgoal, run mecanum IK, return CARLA state.
        """
        if subgoal is None:
            # No goal – gentle brake
            return self._apply_smooth(0.0, 0.0, 0.0)

        alpha = subgoal.alpha
        beta  = subgoal.beta
        slope = subgoal.slope_deg

        # ── desired twist ──────────────────────────────────────────────────
        # Speed modulation: slow on slopes & sharp turns
        speed_factor = max(0.3, 1.0
                           - slope / self.cfg.max_slope_deg * 0.5
                           - abs(alpha) / (math.pi / 2) * 0.3)
        vx_desired = self.cfg.base_speed * speed_factor

        # lateral preference from subgoal azimuth
        vy_desired = -math.sin(alpha) * self.cfg.base_speed * 0.2   # subtle

        # heading correction proportional to azimuth error
        omega_desired = alpha * 1.8    # P-controller gain

        # ── mecanum IK  (for internal validation / wheel speed tracking) ──
        ws = self.mec.inverse(vx_desired, vy_desired, omega_desired)
        # Reconstruct achievable twist from clamped wheel speeds
        vx_ach, vy_ach, om_ach = self.mec.forward(ws)

        return self._apply_smooth(vx_ach, vy_ach, om_ach)

    def compute_manual(self, vx: float, vy: float, omega: float) -> ControlState:
        """Holonomic manual control: twist → mecanum IK → CARLA."""
        ws = self.mec.inverse(vx, vy, omega)
        vx_a, vy_a, om_a = self.mec.forward(ws)
        return self._apply_smooth(vx_a, vy_a, om_a)

    def compute_recovery(self, phase: int) -> ControlState:
        """Recovery manoeuvre: reverse + steer."""
        if phase == 0:          # reverse straight
            return ControlState(throttle=0.3, steer=0.0, brake=0.0, reverse=True)
        elif phase == 1:        # reverse + steer right
            return ControlState(throttle=0.3, steer=0.4, brake=0.0, reverse=True)
        else:                   # reverse + steer left
            return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _apply_smooth(self, vx: float, vy: float, omega: float) -> ControlState:
        a = self.cfg.ema_alpha

        self._vx_ema = a * vx    + (1 - a) * self._vx_ema
        self._vy_ema = a * vy    + (1 - a) * self._vy_ema
        self._om_ema = a * omega + (1 - a) * self._om_ema

        throttle, steer, brake = MecanumKinematics.to_carla_control(
            self._vx_ema, self._vy_ema, self._om_ema, self.cfg
        )

        # Rate limiting
        max_dr = self.cfg.max_steer_rate
        max_dt = self.cfg.max_throttle_rate
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


# ╔══════════════════════════════════════════════════════════════════════════════
# §6  STUCK DETECTION & RECOVERY
# ╚══════════════════════════════════════════════════════════════════════════════

class StuckDetector:
    """
    Monitors vehicle displacement and steering oscillations.
    Triggers a finite-state recovery sequence when stuck is detected.
    """

    IDLE        = "IDLE"
    RECOVERING  = "RECOVERING"

    def __init__(self, cfg: Config = CFG):
        self.cfg            = cfg
        self._pos_history   : deque = deque(maxlen=200)
        self._steer_history : deque = deque(maxlen=40)
        self._state         = self.IDLE
        self._recovery_start: float = 0.0
        self._recovery_phase: int   = 0

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
        """
        Returns (is_stuck, recovery_phase).
        Call every control tick.
        """
        self._pos_history.append((t, pos.copy()))
        self._steer_history.append(steer)

        if self._state == self.RECOVERING:
            elapsed = t - self._recovery_start
            phase_dur = self.cfg.recovery_duration_s / 3.0

            if elapsed > self.cfg.recovery_duration_s:
                self._state = self.IDLE
                return False, 0

            phase = min(int(elapsed / phase_dur), 2)
            return True, phase

        # Check displacement
        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    self._trigger_recovery(t)
                    return True, 0

        # Check steering oscillation
        if len(self._steer_history) == self._steer_history.maxlen:
            arr   = np.array(self._steer_history)
            signs = np.sign(arr)
            flips = np.sum(np.diff(signs) != 0)
            if flips > len(signs) * 0.7:      # >70% flips → oscillation
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


# ╔══════════════════════════════════════════════════════════════════════════════
# §7  VISUALIZATION MODULE
# ╚══════════════════════════════════════════════════════════════════════════════

class VisualizationModule:
    """
    Six-panel matplotlib dashboard (non-blocking, updated at viz_update_hz).

    Layout
    ------
    Row 0: [Front camera] [LiDAR scatter] [Rear camera]
    Row 1: [Cost graph]   [Trajectory]    [Velocity]
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg  = cfg
        plt.ion()
        self.fig  = plt.figure("VSGP Navigator", figsize=(16, 9))
        self.fig.patch.set_facecolor("#111")
        gs        = GridSpec(2, 3, figure=self.fig,
                             hspace=0.35, wspace=0.3)

        # ── axes ──────────────────────────────────────────────────────────────
        self.ax_front  = self.fig.add_subplot(gs[0, 0])
        self.ax_lidar  = self.fig.add_subplot(gs[0, 1])
        self.ax_rear   = self.fig.add_subplot(gs[0, 2])
        self.ax_cost   = self.fig.add_subplot(gs[1, 0])
        self.ax_traj   = self.fig.add_subplot(gs[1, 1])
        self.ax_vel    = self.fig.add_subplot(gs[1, 2])

        for ax in self.fig.get_axes():
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#ddd")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")

        # ── data stores ───────────────────────────────────────────────────────
        maxlen = cfg.traj_history
        self._cost_total   : deque = deque(maxlen=maxlen)
        self._cost_dir     : deque = deque(maxlen=maxlen)
        self._cost_dist    : deque = deque(maxlen=maxlen)
        self._cost_steep   : deque = deque(maxlen=maxlen)
        self._vel_lin      : deque = deque(maxlen=maxlen)
        self._vel_ang      : deque = deque(maxlen=maxlen)
        self._traj_x       : deque = deque(maxlen=maxlen)
        self._traj_y       : deque = deque(maxlen=maxlen)
        self._subgoal_x    : deque = deque(maxlen=20)
        self._subgoal_y    : deque = deque(maxlen=20)

        self._front_img    = None
        self._rear_img     = None
        self._lidar_pts    = None
        self._mode_text    = "AUTO"

        self._last_update  = 0.0
        self._interval     = 1.0 / cfg.viz_update_hz

    # ── update from nav loop ──────────────────────────────────────────────────

    def push_data(
        self,
        *,
        subgoal        : Optional[Subgoal],
        vehicle_pos    : np.ndarray,
        speed_ms       : float,
        steer          : float,
        perc           : Optional[PerceptionOutput],
        mode           : str = "AUTO",
    ) -> None:
        """Called every navigation tick to push latest data."""
        self._traj_x.append(vehicle_pos[0])
        self._traj_y.append(vehicle_pos[1])

        if subgoal is not None:
            self._cost_total.append(subgoal.cost)
            dir_c  = abs(subgoal.alpha) / (math.pi / 2) * CFG.w_direction
            dist_c = abs(subgoal.distance - CFG.subgoal_distance) / CFG.subgoal_distance * CFG.w_distance
            st_c   = subgoal.slope_deg / CFG.max_slope_deg * CFG.w_steepness
            self._cost_dir.append(dir_c)
            self._cost_dist.append(dist_c)
            self._cost_steep.append(st_c)

            # World position of subgoal (approximate)
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
        self._vel_ang.append(steer * CFG.base_speed)   # approximate rad/s

        if perc is not None:
            self._lidar_pts = perc.raw_points

        self._mode_text = mode

    def set_front_image(self, arr: np.ndarray) -> None:
        self._front_img = arr

    def set_rear_image(self, arr: np.ndarray) -> None:
        self._rear_img = arr

    # ── render ────────────────────────────────────────────────────────────────

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
            pass   # silently skip a frame on render errors

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
            z_n = (z - z.min()) / (z.ptp() + 1e-9)
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
        ax.plot(xs, list(self._cost_total), c="#ff6b6b",  lw=1.2, label="Total")
        ax.plot(xs, list(self._cost_dir),   c="#ffa36c",  lw=0.8, label="Direction")
        ax.plot(xs, list(self._cost_dist),  c="#c3f584",  lw=0.8, label="Distance")
        ax.plot(xs, list(self._cost_steep), c="#74b9ff",  lw=0.8, label="Steepness")
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
        ax.set_ylabel("m/s", fontsize=7)

    def close(self) -> None:
        plt.close(self.fig)


# ╔══════════════════════════════════════════════════════════════════════════════
# §8  CARLA SENSOR WRAPPERS
# ╚══════════════════════════════════════════════════════════════════════════════

class SensorManager:
    """Attaches all sensors to the ego vehicle and buffers their latest data."""

    def __init__(self, world, vehicle, cfg: Config = CFG):
        self.cfg     = cfg
        self.vehicle = vehicle
        self.world   = world
        self._actors = []

        self._lidar_data   = None
        self._front_image  = None
        self._rear_image   = None

        bp_lib = world.get_blueprint_library()

        # ── LiDAR ─────────────────────────────────────────────────────────────
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

        # ── Front RGB camera ──────────────────────────────────────────────────
        self._actors.append(self._spawn_camera(
            bp_lib, vehicle,
            carla.Transform(carla.Location(x=1.5, z=1.8)),
            "front",
        ))

        # ── Rear RGB camera ───────────────────────────────────────────────────
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

    # ── sensor callbacks ──────────────────────────────────────────────────────

    def _on_lidar(self, data) -> None:
        self._lidar_data = data

    def _on_front_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._front_image = arr[:, :, ::-1]   # BGR→RGB

    def _on_rear_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        self._rear_image = arr[:, :, ::-1]

    # ── getters ───────────────────────────────────────────────────────────────

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


# ╔══════════════════════════════════════════════════════════════════════════════
# §9  MANUAL KEYBOARD CONTROLLER (Holonomic)
# ╚══════════════════════════════════════════════════════════════════════════════

class ManualHolonomicController:
    """
    Maps keyboard → holonomic twist → mecanum → CARLA control.

    W / S  → vx  (forward / backward)
    A / D  → vy  (strafe left / right)
    Q / E  → ω   (rotate CCW / CW)
    TAB    → toggle AUTO/MANUAL (handled externally)
    """

    VX_SPEED = 3.0    # m/s
    VY_SPEED = 2.0
    OM_SPEED = 1.2    # rad/s

    def __init__(self, cfg: Config = CFG):
        self.ctrl = Controller(cfg)
        self._vx = self._vy = self._om = 0.0

    def tick(self, keys) -> ControlState:
        vx = vy = om = 0.0
        if keys[pygame.K_w]: vx =  self.VX_SPEED
        if keys[pygame.K_s]: vx = -self.VX_SPEED
        if keys[pygame.K_a]: vy =  self.VY_SPEED
        if keys[pygame.K_d]: vy = -self.VY_SPEED
        if keys[pygame.K_q]: om =  self.OM_SPEED
        if keys[pygame.K_e]: om = -self.OM_SPEED

        return self.ctrl.compute_manual(vx, vy, om)


# ╔══════════════════════════════════════════════════════════════════════════════
# §10  VEHICLE SPAWNER (strict per spec)
# ╚══════════════════════════════════════════════════════════════════════════════

def spawn_vehicle(world, cfg: Config = CFG):
    """
    Spawn vehicle.tesla.cybertruck following the EXACT spec:
    - shuffle spawn points
    - offset z by 5.0
    - try each point in order
    - wait 3s after successful spawn
    Returns (vehicle, spawn_z)
    """
    bp_lib    = world.get_blueprint_library()
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

    # Wait ~3 seconds
    wait_start = time.time()
    while time.time() - wait_start < cfg.post_spawn_wait_s:
        world.tick()

    spawn_z = vehicle.get_location().z
    print(f"[SPAWN] Settled Z = {spawn_z:.2f} m")
    return vehicle, spawn_z


# ╔══════════════════════════════════════════════════════════════════════════════
# §11  MAIN NAVIGATION LOOP
# ╚══════════════════════════════════════════════════════════════════════════════

class NavigationSystem:
    """
    Top-level orchestrator.

    Modes
    -----
    AUTO   – full VSGP + planner + controller autonomous navigation
    MANUAL – holonomic keyboard control via mecanum model

    Key bindings
    ------------
    TAB       – toggle AUTO/MANUAL
    ESC / Q   – quit (in pygame window when it has focus; or Ctrl+C)
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg      = cfg
        self._running = False
        self._mode    = "AUTO"

        # CARLA
        self._client  : Optional[carla.Client]  = None
        self._world   : Optional[carla.World]   = None
        self._vehicle = None
        self._sensors : Optional[SensorManager] = None
        self._tm      = None

        # Modules
        self._perception = PerceptionModule(cfg)
        self._planner    = SubgoalPlanner(cfg)
        self._controller = Controller(cfg)
        self._manual_ctrl= ManualHolonomicController(cfg)
        self._stuck      = StuckDetector(cfg)
        self._viz        = VisualizationModule(cfg)

        # pygame (for key capture + HUD)
        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit")
        self._screen = pygame.display.set_mode((200, 80))
        self._font   = pygame.font.SysFont("monospace", 13)

        # Logging
        self._step          = 0
        self._t0            = 0.0
        self._prev_location = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

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
        self._t0       = time.time()
        self._prev_location = self._vehicle.get_location()
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

    # ── per-tick logic ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # ── advance simulation ────────────────────────────────────────────────
        if self.cfg.synchronous:
            self._world.tick()

        t_now = time.time() - self._t0
        self._step += 1

        # ── pygame events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    self._running = False
                    return
                if event.key == pygame.K_TAB:
                    self._mode = "MANUAL" if self._mode == "AUTO" else "AUTO"
                    self._controller.reset_ema()
                    print(f"[MODE] Switched to {self._mode}")

        keys = pygame.key.get_pressed()

        # ── vehicle state ─────────────────────────────────────────────────────
        loc   = self._vehicle.get_location()
        vel   = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        pos   = np.array([loc.x, loc.y, loc.z])

        # ── perception ────────────────────────────────────────────────────────
        perc = None
        if self._sensors.lidar_data is not None:
            perc = self._perception.process_lidar(self._sensors.lidar_data)

        # ── control decision ──────────────────────────────────────────────────
        if self._mode == "MANUAL":
            ctrl_state = self._manual_ctrl.tick(keys)
            best_goal  = None
        else:
            # AUTO
            stuck, rec_phase = self._stuck.update(pos, 0.0, t_now)

            if stuck:
                ctrl_state = self._controller.compute_recovery(rec_phase)
                self._planner.set_exploration_bias(0.8)
                best_goal  = None
            else:
                self._planner.set_exploration_bias(0.0)
                if perc is not None:
                    best_goal, _ = self._planner.plan(perc)
                else:
                    best_goal = None

                slope = best_goal.slope_deg if best_goal else 0.0
                ctrl_state = self._controller.compute(best_goal, speed, slope)

            # Update stuck detector with latest steer
            self._stuck.update(pos, ctrl_state.steer, t_now)

        # ── apply to CARLA ────────────────────────────────────────────────────
        carla_ctrl = carla.VehicleControl(
            throttle = float(ctrl_state.throttle),
            steer    = float(ctrl_state.steer),
            brake    = float(ctrl_state.brake),
            reverse  = ctrl_state.reverse,
        )
        self._vehicle.apply_control(carla_ctrl)

        # ── pygame HUD ────────────────────────────────────────────────────────
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

        # ── visualization ─────────────────────────────────────────────────────
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

        # ── rate limiting (async mode) ────────────────────────────────────────
        if not self.cfg.synchronous:
            time.sleep(self.cfg.fixed_delta_seconds)


# ╔══════════════════════════════════════════════════════════════════════════════
# §12  ENTRY POINT
# ╚══════════════════════════════════════════════════════════════════════════════

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
    finally:
        pass   # cleanup is called inside run()


if __name__ == "__main__":
    main()