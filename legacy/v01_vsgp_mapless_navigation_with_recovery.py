# from __future__ import annotations

# import math
# import random
# import sys
# import time
# import threading
# import traceback
# from collections import deque
# from dataclasses import dataclass, field
# from typing import List, Optional, Tuple
# from queue import Queue

# import numpy as np
# import pygame
# from matplotlib import pyplot as plt
# from matplotlib.gridspec import GridSpec

# # ─────────────────────────────────────────────────────────────────────────────
# # Optional scipy / sklearn
# # ─────────────────────────────────────────────────────────────────────────────
# try:
#     from scipy.spatial.distance import cdist
#     from sklearn.cluster import KMeans
#     SKLEARN_OK = True
# except ImportError:
#     SKLEARN_OK = False
#     print("[WARN] scikit-learn not found – inducing points fall back to random selection")

# try:
#     import carla
# except ImportError:
#     print("[FATAL] carla Python package not found. Put CARLA's egg on PYTHONPATH.")
#     sys.exit(1)


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §0  CONFIGURATION
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# @dataclass
# class Config:
#     # ── CARLA connection
#     carla_host: str             = "127.0.0.1"
#     carla_port: int             = 2000
#     carla_timeout: float        = 20.0
#     synchronous: bool           = True
#     fixed_delta_seconds: float  = 0.05          # 20 Hz physics

#     # ── Vehicle (Spawn point kept intact as requested)
#     vehicle_blueprint: str      = "vehicle.tesla.cybertruck"
#     spawn_z_offset: float       = 5.0
#     post_spawn_wait_s: float    = 3.0

#     # FIX #5 – Cybertruck physics inversion flag.
#     invert_drive: bool          = False

#     # ── Vehicle Geometry
#     vehicle_length: float       = 4.5
#     vehicle_width: float        = 2.0
#     vehicle_height: float       = 1.8
#     wheel_base: float           = 2.8
#     track_width: float          = 1.7
#     wheel_radius_real: float    = 0.43

#     # ── Sensors
#     lidar_range: float          = 10.0
#     lidar_points_per_sec: int   = 80000
#     lidar_rotation_freq: float  = 20.0
#     lidar_channels: int         = 64
#     lidar_upper_fov: float      = 15.0
#     lidar_lower_fov: float      = -25.0
#     camera_width: int           = 320
#     camera_height: int          = 240
#     camera_fov: int             = 90

#     # ── Planner
#     robot_width: float          = 0.8
#     safety_margin: float        = 2.80
#     max_slope_deg: float        = 15.0
#     subgoal_distance: float     = 5.0
#     n_subgoals_max: int         = 5

#     # ── GLOBAL PLANNER CONFIG (NEW)
#     goal_location: Tuple[float, float, float] = (-1169.18, -413.52, 15.16)
#     goal_tolerance: float       = 5.0           # meters to consider "arrived"
#     global_planner_enabled: bool = True
#     waypoint_spacing: float     = 10.0          # meters between global waypoints
#     lookahead_distance: float   = 15.0          # meters to look ahead on global path

#     # ── Cost weights
#     w_direction: float          = 4.0
#     w_distance: float           = 2.5
#     w_steepness: float          = 50.0
#     w_collision: float          = 500.0
#     w_oscillation: float        = 50.0
#     w_flatness: float           = 5.0

#     # ── Control (Ackermann)
#     max_throttle: float         = 0.55
#     max_steer: float            = 0.9
#     max_brake: float            = 0.8
#     ema_alpha: float            = 0.3
#     max_steer_rate: float       = 0.15
#     max_throttle_rate: float    = 0.08
#     base_speed: float           = 3.0
#     kp_throttle: float          = 0.5
#     kp_steer: float             = 1.5

#     # ── Stability (roll / pitch)
#     max_safe_roll_deg: float        = 10.0
#     max_safe_pitch_deg: float       = 15.0
#     stability_throttle_scale: float = 0.4
#     roll_corrective_steer: float    = 0.5

#     # ── Stuck detection
#     stuck_window_s: float       = 2.0
#     stuck_disp_thresh: float    = 0.6
#     recovery_duration_s: float  = 2.5

#     # ── VSGP
#     vsgp_n_inducing: int        = 50
#     vsgp_alpha: float           = 1.0
#     vsgp_length_scale: float    = 0.3
#     vsgp_noise_var: float       = 0.05
#     vsgp_lr: float              = 0.02
#     vsgp_update_freq: int       = 10

#     # ── Visualization
#     viz_update_hz: float        = 3.0
#     traj_history: int           = 500


# CFG = Config()


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §1  ACKERMANN KINEMATIC MODEL
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class AckermannKinematics:
#     """Standard bicycle model for Ackermann steering vehicles."""

#     def __init__(self, cfg: Config = CFG):
#         self.wheel_base = cfg.wheel_base
#         self.cfg = cfg

#     def steering_angle_to_curvature(self, steer: float) -> float:
#         if abs(steer) < 1e-6:
#             return 0.0
#         return math.tan(steer) / self.wheel_base

#     def curvature_to_steering_angle(self, curvature: float) -> float:
#         return math.atan(curvature * self.wheel_base)

#     @staticmethod
#     def to_carla_control(
#         vx_desired: float,
#         yaw_error: float,
#         current_speed: float,
#         cfg: Config = CFG,
#     ) -> Tuple[float, float, float, bool]:
#         """Returns: (throttle, steer, brake, reverse)"""
#         reverse = False

#         speed_error = vx_desired - current_speed
#         throttle = cfg.kp_throttle * speed_error
#         throttle = float(np.clip(throttle, 0.0, cfg.max_throttle))

#         brake = 0.0
#         if speed_error < -1.0:
#             brake = min(0.5, abs(speed_error) * 0.3)
#             throttle = 0.0

#         steer = cfg.kp_steer * yaw_error
#         steer = float(np.clip(steer, -cfg.max_steer, cfg.max_steer))

#         return throttle, steer, brake, reverse


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §2  VARIATIONAL SPARSE GAUSSIAN PROCESS (VSGP)
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class RationalQuadraticKernel:
#     def __init__(self, alpha: float = 1.0,
#                  length_scale: float = 0.3,
#                  variance: float = 1.0):
#         self.alpha = alpha
#         self.length_scale = length_scale
#         self.variance = variance

#     def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
#         if len(X) == 0 or len(Y) == 0:
#             return np.zeros((len(X), len(Y)))
#         diff = X[:, None, :] - Y[None, :, :]
#         sq_d = np.sum(diff ** 2, axis=-1)
#         denom = 2.0 * self.alpha * self.length_scale ** 2
#         return self.variance * (1.0 + sq_d / denom) ** (-self.alpha)

#     def diag(self, X: np.ndarray) -> np.ndarray:
#         return self.variance * np.ones(len(X))


# class VSGP:
#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self.kernel = RationalQuadraticKernel(
#             alpha=cfg.vsgp_alpha,
#             length_scale=cfg.vsgp_length_scale,
#             variance=1.0,
#         )
#         self.noise_var = cfg.vsgp_noise_var
#         self.n_ind = cfg.vsgp_n_inducing
#         self.lr = cfg.vsgp_lr
#         self.update_freq = cfg.vsgp_update_freq

#         side = int(math.sqrt(self.n_ind))
#         a_v = np.linspace(-np.pi / 2, np.pi / 2, side)
#         b_v = np.linspace(-np.pi / 4, np.pi / 4, side)
#         A, B = np.meshgrid(a_v, b_v)
#         self.Z = np.column_stack([A.ravel(), B.ravel()])[:self.n_ind]

#         self.mu = np.zeros(len(self.Z))
#         self.Su = np.eye(len(self.Z)) * 0.1

#         self._Kuu_inv: Optional[np.ndarray] = None
#         self._trained: bool = False
#         self._update_count: int = 0
#         self._last_X_hash: int = 0

#     def _select_inducing_points(self, X: np.ndarray) -> None:
#         actual_n = min(self.n_ind, len(X))
#         if SKLEARN_OK and len(X) >= self.n_ind:
#             km = KMeans(n_clusters=self.n_ind, n_init=3,
#                         max_iter=50, random_state=0)
#             km.fit(X)
#             self.Z = km.cluster_centers_.copy()
#         else:
#             idx = np.random.choice(len(X), actual_n, replace=False)
#             self.Z = X[idx].copy()

#         n = len(self.Z)
#         self.mu = np.zeros(n)
#         self.Su = np.eye(n) * 0.1
#         self._Kuu_inv = None

#     def update(self, X: np.ndarray, y: np.ndarray) -> None:
#         if len(X) < 4:
#             return

#         self._update_count += 1

#         if self._update_count % (50 * self.update_freq) == 0:
#             self._select_inducing_points(X)

#         current_hash = hash(X.tobytes())
#         if current_hash == self._last_X_hash and self._trained:
#             return
#         self._last_X_hash = current_hash

#         m = len(self.Z)
#         Kuu = self.kernel(self.Z, self.Z) + np.eye(m) * 1e-6

#         try:
#             self._Kuu_inv = np.linalg.inv(Kuu)
#         except np.linalg.LinAlgError:
#             self._Kuu_inv = np.linalg.pinv(Kuu)

#         Kfu = self.kernel(X, self.Z)
#         A = Kfu @ self._Kuu_inv

#         noise_inv = 1.0 / self.noise_var
#         Lambda = noise_inv * (A.T @ A) + self._Kuu_inv
#         rhs = noise_inv * (A.T @ y)

#         try:
#             Su_new = np.linalg.inv(Lambda + np.eye(m) * 1e-6)
#             mu_new = Su_new @ rhs
#         except np.linalg.LinAlgError:
#             return

#         lr = self.lr
#         self.Su = (1 - lr) * self.Su + lr * Su_new
#         self.mu = (1 - lr) * self.mu + lr * mu_new
#         self._trained = True

#     def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         if not self._trained or self._Kuu_inv is None:
#             n = len(Xs)
#             return np.zeros(n), np.ones(n) * self.noise_var

#         Ksu = self.kernel(Xs, self.Z)
#         A = Ksu @ self._Kuu_inv
#         mean = A @ self.mu

#         Kss_diag = self.kernel.diag(Xs)
#         var_explained = np.sum(A * (A @ self._Kuu_inv.T), axis=1)
#         var_variational = np.sum(A @ self.Su * A, axis=1)
#         var = Kss_diag - var_explained + var_variational + self.noise_var
#         var = np.clip(var, 1e-6, None)

#         return mean, var


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §3  PERCEPTION MODULE
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# @dataclass
# class PerceptionOutput:
#     alpha_grid: np.ndarray
#     beta_grid: np.ndarray
#     occupancy_mean: np.ndarray
#     occupancy_var: np.ndarray
#     slope_map: np.ndarray
#     traversability: np.ndarray
#     raw_points: np.ndarray
#     free_mask: Optional[np.ndarray] = None
#     mean_surface: Optional[np.ndarray] = None
#     ground_slope_deg: float = 0.0


# class PerceptionModule:
#     ALPHA_RES = 60
#     BETA_RES = 30
#     ROC = 5.0
#     VAR_FREE_WEIGHT = 1.0
#     SLOPE_WEIGHT = 1.2
#     OCC_WEIGHT = 1.0

#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self.vsgp = VSGP(cfg)
#         self._last_output: Optional[PerceptionOutput] = None

#         self._a_lin = np.linspace(-np.pi / 2, np.pi / 2, self.ALPHA_RES)
#         self._b_lin = np.linspace(-np.pi / 6, np.pi / 6, self.BETA_RES)
#         AG, BG = np.meshgrid(self._a_lin, self._b_lin)
#         self._grid_pts = np.column_stack([AG.ravel(), BG.ravel()])
#         self._AG = AG
#         self._BG = BG

#     def _estimate_ground_slope(self, pts: np.ndarray) -> float:
#         if len(pts) < 10:
#             return 0.0
#         z_median = np.median(pts[:, 2])
#         ground_mask = np.abs(pts[:, 2] - z_median) < 1.0
#         ground_pts = pts[ground_mask]
#         if len(ground_pts) < 10:
#             return 0.0
#         centroid = np.mean(ground_pts, axis=0)
#         centered = ground_pts - centroid
#         _, _, vh = np.linalg.svd(centered)
#         normal = vh[-1, :]
#         z_axis = np.array([0, 0, 1])
#         cos_angle = np.abs(np.dot(normal, z_axis)) / (
#             np.linalg.norm(normal) * np.linalg.norm(z_axis))
#         slope_rad = np.arccos(np.clip(cos_angle, -1, 1))
#         return float(np.clip(np.degrees(slope_rad), 0, 90))

#     def process_lidar(self, measurement) -> Optional[PerceptionOutput]:
#         raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
#         if len(raw) < 20:
#             return self._last_output

#         dist = np.linalg.norm(raw, axis=1)
#         mask = (dist > 0.5) & (dist < self.cfg.lidar_range)
#         pts = raw[mask]
#         if len(pts) < 10:
#             return self._last_output

#         ground_slope = self._estimate_ground_slope(pts)

#         r = np.linalg.norm(pts, axis=1)
#         alpha = np.arctan2(pts[:, 1], pts[:, 0])
#         beta = np.arcsin(np.clip(pts[:, 2] / (r + 1e-9), -1.0, 1.0))

#         X = np.column_stack([alpha, beta])
#         y = self.ROC - r

#         if len(X) > 2000:
#             idx = np.random.choice(len(X), 2000, replace=False)
#             X, y = X[idx], y[idx]
#             pts = pts[idx]

#         self.vsgp.update(X, y)
#         mean, var = self.vsgp.predict(self._grid_pts)

#         mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
#         var = var.reshape(self.BETA_RES, self.ALPHA_RES)

#         d_alpha = np.gradient(mean, axis=1)
#         d_beta = np.gradient(mean, axis=0)
#         slope = np.sqrt(d_alpha ** 2 + d_beta ** 2)
#         slope = slope / (np.max(slope) + 1e-6)

#         mean_range = mean.max() - mean.min()
#         occ_norm = (mean - mean.min()) / (mean_range + 1e-6) if mean_range > 0 else mean
#         var_range = var.max() - var.min()
#         var_norm = (var - var.min()) / (var_range + 1e-6) if var_range > 0 else var

#         traversability = np.clip(
#             1.0
#             - self.OCC_WEIGHT * occ_norm
#             - self.SLOPE_WEIGHT * slope
#             - self.VAR_FREE_WEIGHT * var_norm,
#             0.0, 1.0,
#         )
#         free_mask = traversability > 0.5

#         out = PerceptionOutput(
#             alpha_grid=self._AG,
#             beta_grid=self._BG,
#             occupancy_mean=mean,
#             occupancy_var=var,
#             slope_map=slope,
#             traversability=traversability,
#             raw_points=pts,
#             free_mask=free_mask,
#             mean_surface=mean,
#             ground_slope_deg=ground_slope,
#         )
#         self._last_output = out
#         return out

#     @property
#     def last_output(self) -> Optional[PerceptionOutput]:
#         return self._last_output


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §4  SUBGOAL PLANNER
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# @dataclass
# class Subgoal:
#     alpha: float
#     beta: float
#     distance: float
#     local_pos: np.ndarray
#     slope_deg: float
#     safe: bool
#     width_m: float
#     cost: float = 0.0
#     world_pos: Optional[np.ndarray] = None


# class VSGPSubgoalPlanner:
#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self.max_occ_thresh = 0.35
#         self.max_var_thresh = 0.4
#         self.max_slope_thresh = 0.2
#         self.min_dist = 3.0
#         self.max_dist = cfg.subgoal_distance

#     def plan(
#         self,
#         perc: PerceptionOutput,
#         vehicle_pitch_rad: float,
#         vehicle_transform: carla.Transform,
#         global_waypoint: Optional[np.ndarray] = None,
#         mode: str = "forward",
#     ) -> Tuple[Optional[Subgoal], List[Subgoal]]:
#         if perc.occupancy_mean is None:
#             return None, []

#         alpha_range = np.linspace(-np.pi / 2.5, np.pi / 2.5, 15)
#         candidates = []

#         occ = perc.occupancy_mean
#         var = perc.occupancy_var
#         slope = perc.slope_map

#         alphas = perc.alpha_grid[0]
#         betas = perc.beta_grid[:, 0]
#         mid_beta_idx = len(betas) // 2

#         # Calculate preferred direction from global waypoint if available
#         preferred_alpha = 0.0
#         if global_waypoint is not None:
#             vehicle_pos = np.array([vehicle_transform.location.x, 
#                                    vehicle_transform.location.y, 
#                                    vehicle_transform.location.z])
#             direction = global_waypoint[:2] - vehicle_pos[:2]
#             preferred_alpha = math.atan2(direction[1], direction[0])
#             # Convert to vehicle frame
#             vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
#             preferred_alpha = preferred_alpha - vehicle_yaw

#         for a in alpha_range:
#             a_idx = np.argmin(np.abs(alphas - a))

#             best_point_cost = float('inf')
#             best_point_dist = 0.0
#             s_val = 0.0

#             dist_samples = np.linspace(self.min_dist, self.max_dist, 5)

#             for d in dist_samples:
#                 step_ratio = d / self.max_dist
#                 pitch_correction = vehicle_pitch_rad * step_ratio * 5
#                 b_idx = int(mid_beta_idx - step_ratio * (mid_beta_idx - 1) + pitch_correction)
#                 b_idx = int(np.clip(b_idx, 0, occ.shape[0] - 1))

#                 if not (0 <= a_idx < occ.shape[1]):
#                     break

#                 o_val = occ[b_idx, a_idx]
#                 v_val = var[b_idx, a_idx]
#                 s_val = slope[b_idx, a_idx]

#                 is_free = o_val < self.max_occ_thresh
#                 is_stable = v_val < self.max_var_thresh
#                 is_safe = s_val < self.max_slope_thresh

#                 if is_free and is_stable and is_safe:
#                     # Add penalty for deviating from global direction
#                     global_penalty = abs(a - preferred_alpha) * 2.0 if global_waypoint is not None else 0.0
                    
#                     cost = (
#                         2.0 * abs(a) / (np.pi / 2) +
#                         1.0 * s_val +
#                         0.5 * v_val +
#                         global_penalty
#                     )
#                     cost -= d * 0.5

#                     if cost < best_point_cost:
#                         best_point_cost = cost
#                         best_point_dist = d
#                 else:
#                     break

#             if best_point_dist > 0:
#                 lx = best_point_dist * np.cos(a)
#                 ly = best_point_dist * np.sin(a)

#                 # Calculate world position
#                 vehicle_pos = np.array([vehicle_transform.location.x, 
#                                        vehicle_transform.location.y, 
#                                        vehicle_transform.location.z])
#                 vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                
#                 # Transform local to world
#                 world_x = vehicle_pos[0] + lx * math.cos(vehicle_yaw) - ly * math.sin(vehicle_yaw)
#                 world_y = vehicle_pos[1] + lx * math.sin(vehicle_yaw) + ly * math.cos(vehicle_yaw)
#                 world_pos = np.array([world_x, world_y, vehicle_pos[2]])

#                 sg = Subgoal(
#                     alpha=a,
#                     beta=0.0,
#                     distance=best_point_dist,
#                     local_pos=np.array([lx, ly, 0.0]),
#                     slope_deg=s_val * 45.0,
#                     safe=True,
#                     width_m=self.cfg.vehicle_width,
#                     cost=best_point_cost,
#                     world_pos=world_pos,
#                 )
#                 candidates.append(sg)

#         if not candidates:
#             return None, []

#         candidates.sort(key=lambda s: s.cost)
#         return candidates[0], candidates


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §4.5  GLOBAL PLANNER (NEW)
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# @dataclass
# class GlobalWaypoint:
#     x: float
#     y: float
#     z: float
#     yaw: float = 0.0
#     reached: bool = False


# class GlobalPlanner:
#     """
#     Generates a global path from current position to final destination.
#     Creates intermediate waypoints that the local planner follows.
#     """
    
#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self.goal_location = np.array(cfg.goal_location)
#         self.waypoint_spacing = cfg.waypoint_spacing
#         self.waypoints: List[GlobalWaypoint] = []
#         self.current_waypoint_idx = 0
#         self.path_computed = False
#         self.goal_reached = False
        
#     def compute_path(self, start_pos: np.ndarray, start_yaw: float) -> List[GlobalWaypoint]:
#         """
#         Compute waypoints from start position to goal.
#         Uses straight-line interpolation with optional obstacle avoidance.
#         """
#         self.waypoints = []
        
#         # Calculate direction to goal
#         direction = self.goal_location[:2] - start_pos[:2]
#         total_distance = np.linalg.norm(direction)
        
#         if total_distance < self.cfg.goal_tolerance:
#             self.goal_reached = True
#             print(f"[GLOBAL] Already at goal! Distance: {total_distance:.2f}m")
#             return self.waypoints
        
#         # Normalize direction
#         if total_distance > 0:
#             direction = direction / total_distance
        
#         # Generate waypoints along the path
#         num_waypoints = int(total_distance / self.waypoint_spacing) + 1
        
#         for i in range(num_waypoints + 1):
#             progress = i / num_waypoints
#             wp_x = start_pos[0] + direction[0] * total_distance * progress
#             wp_y = start_pos[1] + direction[1] * total_distance * progress
#             wp_z = start_pos[2] + (self.goal_location[2] - start_pos[2]) * progress
            
#             # Calculate yaw for this waypoint (direction to next waypoint)
#             if i < num_waypoints:
#                 next_wp_x = start_pos[0] + direction[0] * total_distance * (i + 1) / num_waypoints
#                 next_wp_y = start_pos[1] + direction[1] * total_distance * (i + 1) / num_waypoints
#                 wp_yaw = math.degrees(math.atan2(next_wp_y - wp_y, next_wp_x - wp_x))
#             else:
#                 # Final waypoint - use goal orientation
#                 wp_yaw = -82.22  # From your specified goal yaw
            
#             waypoint = GlobalWaypoint(x=wp_x, y=wp_y, z=wp_z, yaw=wp_yaw)
#             self.waypoints.append(waypoint)
        
#         # Set final waypoint to exact goal coordinates
#         self.waypoints[-1].x = self.goal_location[0]
#         self.waypoints[-1].y = self.goal_location[1]
#         self.waypoints[-1].z = self.goal_location[2]
#         self.waypoints[-1].yaw = -82.22  # Your specified goal yaw
        
#         self.path_computed = True
#         self.current_waypoint_idx = 0
        
#         print(f"[GLOBAL] Path computed: {len(self.waypoints)} waypoints to goal")
#         print(f"[GLOBAL] Goal: X={self.goal_location[0]:.2f}, Y={self.goal_location[1]:.2f}, Z={self.goal_location[2]:.2f}")
        
#         return self.waypoints
    
#     def get_current_waypoint(self, vehicle_pos: np.ndarray) -> Optional[GlobalWaypoint]:
#         """
#         Get the current target waypoint based on vehicle position.
#         Advances to next waypoint when current one is reached.
#         """
#         if not self.path_computed or len(self.waypoints) == 0:
#             return None
        
#         # Check if we've reached the goal
#         dist_to_goal = np.linalg.norm(vehicle_pos[:2] - self.goal_location[:2])
#         if dist_to_goal < self.cfg.goal_tolerance:
#             self.goal_reached = True
#             print(f"[GLOBAL] 🎯 GOAL REACHED! Distance: {dist_to_goal:.2f}m")
#             return None
        
#         # Find the next unreach waypoint
#         while self.current_waypoint_idx < len(self.waypoints):
#             wp = self.waypoints[self.current_waypoint_idx]
#             wp_pos = np.array([wp.x, wp.y, wp.z])
#             dist_to_wp = np.linalg.norm(vehicle_pos - wp_pos)
            
#             if dist_to_wp < self.waypoint_spacing * 0.5:
#                 # Waypoint reached, move to next
#                 wp.reached = True
#                 self.current_waypoint_idx += 1
#                 print(f"[GLOBAL] Waypoint {self.current_waypoint_idx - 1} reached")
#             else:
#                 # Return current target waypoint
#                 return wp
        
#         # All waypoints reached, return goal
#         return GlobalWaypoint(
#             x=self.goal_location[0],
#             y=self.goal_location[1],
#             z=self.goal_location[2],
#             yaw=-82.22
#         )
    
#     def get_lookahead_waypoint(self, vehicle_pos: np.ndarray) -> Optional[np.ndarray]:
#         """
#         Get a lookahead point on the global path for the local planner.
#         Returns the position of a waypoint ahead of the vehicle.
#         """
#         if not self.path_computed or len(self.waypoints) == 0:
#             return None
        
#         # Find waypoint at lookahead distance
#         for i in range(self.current_waypoint_idx, len(self.waypoints)):
#             wp = self.waypoints[i]
#             wp_pos = np.array([wp.x, wp.y, wp.z])
#             dist = np.linalg.norm(vehicle_pos - wp_pos)
            
#             if dist >= self.cfg.lookahead_distance:
#                 return np.array([wp.x, wp.y, wp.z])
        
#         # If no waypoint at lookahead distance, return goal
#         return self.goal_location.copy()
    
#     def reset(self, start_pos: np.ndarray, start_yaw: float) -> None:
#         """Reset the global planner with new start position."""
#         self.goal_reached = False
#         self.current_waypoint_idx = 0
#         self.path_computed = False
#         self.compute_path(start_pos, start_yaw)
    
#     def is_goal_reached(self) -> bool:
#         return self.goal_reached


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §5  CONTROLLER MODULE
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# @dataclass
# class ControlState:
#     throttle: float = 0.0
#     steer: float    = 0.0
#     brake: float    = 0.0
#     reverse: bool   = False


# class Controller:
#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self.ackermann = AckermannKinematics(cfg)
#         self._state = ControlState()
#         self._speed_ema = 0.0
#         self._steer_ema = 0.0

#     def compute(
#         self,
#         subgoal: Optional[Subgoal],
#         vehicle_speed_ms: float,
#         terrain_slope_deg: float = 0.0,
#         roll_deg: float = 0.0,
#         pitch_deg: float = 0.0,
#     ) -> ControlState:
#         if subgoal is None:
#             return self._apply_smooth(0.0, 0.0, roll_deg=roll_deg,
#                                       pitch_deg=pitch_deg, brake=1.0)

#         alpha = subgoal.alpha
#         speed_factor = max(
#             0.3,
#             1.0
#             - terrain_slope_deg / self.cfg.max_slope_deg * 0.5
#             - abs(alpha) / (math.pi / 2) * 0.3,
#         )
#         vx_desired = self.cfg.base_speed * speed_factor
#         yaw_error = alpha

#         throttle, steer, brake, reverse = AckermannKinematics.to_carla_control(
#             vx_desired, yaw_error, vehicle_speed_ms, self.cfg
#         )
#         return self._apply_smooth(throttle, steer, brake, reverse,
#                                   roll_deg=roll_deg, pitch_deg=pitch_deg)

#     def compute_manual(self, vx: float, steer: float) -> ControlState:
#         throttle = np.clip(vx / self.cfg.base_speed, 0.0, self.cfg.max_throttle)
#         return self._apply_smooth(throttle, steer, 0.0, False)

#     def compute_recovery(self, phase: int) -> ControlState:
#         if phase == 0:
#             return ControlState(throttle=0.3, steer=0.0,  brake=0.0, reverse=True)
#         elif phase == 1:
#             return ControlState(throttle=0.3, steer=0.4,  brake=0.0, reverse=True)
#         else:
#             return ControlState(throttle=0.3, steer=-0.4, brake=0.0, reverse=True)

#     def _stability_modifiers(self, roll_deg: float, pitch_deg: float) -> Tuple[float, float]:
#         throttle_scale = 1.0
#         corrective_steer = 0.0
#         abs_roll = abs(roll_deg)

#         if abs_roll > self.cfg.max_safe_roll_deg:
#             excess = abs_roll - self.cfg.max_safe_roll_deg
#             roll_factor = 1.0 - min(excess / 15.0, 1.0)
#             throttle_scale = min(
#                 throttle_scale,
#                 self.cfg.stability_throttle_scale
#                 + (1.0 - self.cfg.stability_throttle_scale) * roll_factor,
#             )
#             corrective_steer = -math.copysign(
#                 min(self.cfg.roll_corrective_steer * (excess / 10.0),
#                     self.cfg.roll_corrective_steer),
#                 roll_deg,
#             )

#         abs_pitch = abs(pitch_deg)
#         if abs_pitch > self.cfg.max_safe_pitch_deg:
#             excess_p = abs_pitch - self.cfg.max_safe_pitch_deg
#             pitch_factor = 1.0 - min(excess_p / 10.0, 0.7)
#             throttle_scale = min(
#                 throttle_scale,
#                 self.cfg.stability_throttle_scale * pitch_factor + 0.15,
#             )

#         return float(throttle_scale), float(corrective_steer)

#     def _apply_smooth(
#         self,
#         throttle: float,
#         steer: float,
#         brake: float = 0.0,
#         reverse: bool = False,
#         roll_deg: float = 0.0,
#         pitch_deg: float = 0.0,
#     ) -> ControlState:
#         a = self.cfg.ema_alpha
#         self._speed_ema = a * throttle + (1 - a) * self._speed_ema
#         self._steer_ema = a * steer   + (1 - a) * self._steer_ema

#         out_throttle = self._speed_ema
#         out_steer    = self._steer_ema
#         out_brake    = brake

#         if brake > 0.1:
#             out_throttle = 0.0
#             out_brake    = brake

#         max_dr = self.cfg.max_steer_rate
#         max_dt = self.cfg.max_throttle_rate
#         out_steer    = float(np.clip(out_steer,
#                                      self._state.steer    - max_dr,
#                                      self._state.steer    + max_dr))
#         out_throttle = float(np.clip(out_throttle,
#                                      self._state.throttle - max_dt,
#                                      self._state.throttle + max_dt))

#         thr_scale, cor_steer = self._stability_modifiers(roll_deg, pitch_deg)
#         out_throttle *= thr_scale
#         out_steer = float(np.clip(out_steer + cor_steer,
#                                   -self.cfg.max_steer, self.cfg.max_steer))

#         self._state = ControlState(throttle=out_throttle, steer=out_steer,
#                                    brake=out_brake, reverse=reverse)
#         return self._state

#     def reset_ema(self) -> None:
#         self._speed_ema = self._steer_ema = 0.0


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §6  STUCK DETECTION & RECOVERY
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class StuckDetector:
#     IDLE      = "IDLE"
#     RECOVERING = "RECOVERING"

#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self._pos_history: deque   = deque(maxlen=200)
#         self._steer_history: deque = deque(maxlen=40)
#         self._state = self.IDLE
#         self._recovery_start: float = 0.0
#         self._recovery_phase: int   = 0

#     def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
#         self._pos_history.append((t, pos.copy()))
#         self._steer_history.append(steer)

#         if self._state == self.RECOVERING:
#             elapsed   = t - self._recovery_start
#             phase_dur = self.cfg.recovery_duration_s / 3.0
#             if elapsed > self.cfg.recovery_duration_s:
#                 self._state = self.IDLE
#                 return False, 0
#             phase = min(int(elapsed / phase_dur), 2)
#             return True, phase

#         if len(self._pos_history) >= 2:
#             oldest_t, oldest_pos = self._pos_history[0]
#             if (t - oldest_t) >= self.cfg.stuck_window_s:
#                 disp = np.linalg.norm(pos - oldest_pos)
#                 if disp < self.cfg.stuck_disp_thresh:
#                     self._trigger_recovery(t)
#                     return True, 0

#         if len(self._steer_history) == self._steer_history.maxlen:
#             arr   = np.array(self._steer_history)
#             signs = np.sign(arr)
#             flips = int(np.sum(np.diff(signs) != 0))
#             if flips > len(signs) * 0.7:
#                 self._trigger_recovery(t)
#                 return True, 0

#         return False, 0

#     def _trigger_recovery(self, t: float) -> None:
#         if self._state == self.IDLE:
#             print("[STUCK] Recovery triggered!")
#             self._state            = self.RECOVERING
#             self._recovery_start   = t
#             self._recovery_phase   = 0
#             self._pos_history.clear()

#     def is_recovering(self) -> bool:
#         return self._state == self.RECOVERING


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §7  VISUALIZATION MODULE (Threaded)
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class VisualizationModule:
#     def __init__(self, cfg: Config = CFG):
#         self.cfg = cfg
#         self._running = False
#         self._thread: Optional[threading.Thread] = None

#         self._data_lock = threading.Lock()
#         self._subgoal    = None
#         self._vehicle_pos = None
#         self._speed_ms    = 0.0
#         self._steer       = 0.0
#         self._perc        = None
#         self._mode        = "AUTO"
#         self._front_img   = None
#         self._rear_img    = None
#         self._lidar_pts   = None
#         self._global_path = None  # NEW
#         self._goal_location = None  # NEW

#         maxlen = cfg.traj_history
#         self._cost_total: deque  = deque(maxlen=maxlen)
#         self._cost_dir: deque    = deque(maxlen=maxlen)
#         self._cost_dist: deque   = deque(maxlen=maxlen)
#         self._cost_steep: deque  = deque(maxlen=maxlen)
#         self._vel_lin: deque     = deque(maxlen=maxlen)
#         self._vel_ang: deque     = deque(maxlen=maxlen)
#         self._traj_x: deque      = deque(maxlen=maxlen)
#         self._traj_y: deque      = deque(maxlen=maxlen)
#         self._subgoal_x: deque   = deque(maxlen=20)
#         self._subgoal_y: deque   = deque(maxlen=20)

#         self._interval: float = 1.0 / cfg.viz_update_hz

#     def start(self) -> None:
#         self._running = True
#         self._thread = threading.Thread(target=self._render_loop, daemon=True)
#         self._thread.start()

#     def stop(self) -> None:
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=2.0)
#         plt.close("all")

#     def push_data(
#         self,
#         *,
#         subgoal: Optional[Subgoal],
#         vehicle_pos: np.ndarray,
#         speed_ms: float,
#         steer: float,
#         perc: Optional[PerceptionOutput],
#         mode: str = "AUTO",
#         global_path: Optional[List[GlobalWaypoint]] = None,  # NEW
#         goal_location: Optional[np.ndarray] = None,  # NEW
#     ) -> None:
#         with self._data_lock:
#             self._traj_x.append(float(vehicle_pos[0]))
#             self._traj_y.append(float(vehicle_pos[1]))

#             if subgoal is not None:
#                 self._cost_total.append(subgoal.cost)
#                 self._cost_dir.append(
#                     abs(subgoal.alpha) / (math.pi / 2) * CFG.w_direction)
#                 self._cost_dist.append(
#                     abs(subgoal.distance - CFG.subgoal_distance)
#                     / CFG.subgoal_distance * CFG.w_distance)
#                 self._cost_steep.append(
#                     subgoal.slope_deg / CFG.max_slope_deg * CFG.w_steepness)
#                 self._subgoal_x.append(vehicle_pos[0] + subgoal.local_pos[0])
#                 self._subgoal_y.append(vehicle_pos[1] + subgoal.local_pos[1])
#             else:
#                 self._cost_total.append(0)
#                 self._cost_dir.append(0)
#                 self._cost_dist.append(0)
#                 self._cost_steep.append(0)

#             self._vel_lin.append(speed_ms)
#             self._vel_ang.append(steer * CFG.base_speed)

#             if perc is not None and perc.raw_points is not None:
#                 self._lidar_pts = perc.raw_points.copy()

#             self._subgoal     = subgoal
#             self._vehicle_pos = vehicle_pos.copy()
#             self._speed_ms    = speed_ms
#             self._steer       = steer
#             self._mode        = mode
#             self._global_path = global_path  # NEW
#             self._goal_location = goal_location  # NEW

#     def set_front_image(self, arr: np.ndarray) -> None:
#         with self._data_lock:
#             self._front_img = arr

#     def set_rear_image(self, arr: np.ndarray) -> None:
#         with self._data_lock:
#             self._rear_img = arr

#     def _render_loop(self) -> None:
#         plt.ion()
#         fig = plt.figure("VSGP Navigator", figsize=(16, 9))
#         fig.patch.set_facecolor("#111")
#         gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

#         ax_front = fig.add_subplot(gs[0, 0])
#         ax_lidar = fig.add_subplot(gs[0, 1])
#         ax_rear  = fig.add_subplot(gs[0, 2])
#         ax_cost  = fig.add_subplot(gs[1, 0])
#         ax_traj  = fig.add_subplot(gs[1, 1])
#         ax_vel   = fig.add_subplot(gs[1, 2])

#         for ax in fig.get_axes():
#             ax.set_facecolor("#1a1a2e")
#             for sp in ax.spines.values():
#                 sp.set_color("#444")
#             ax.tick_params(colors="#ccc", labelsize=7)
#             ax.title.set_color("#ddd")
#             ax.xaxis.label.set_color("#aaa")
#             ax.yaxis.label.set_color("#aaa")

#         last_update = 0.0
#         interval    = self._interval

#         while self._running:
#             now = time.time()
#             if (now - last_update) < interval:
#                 time.sleep(0.01)
#                 continue
#             last_update = now

#             try:
#                 with self._data_lock:
#                     self._draw_camera(ax_front, self._front_img, "Front Camera")
#                     self._draw_camera(ax_rear,  self._rear_img,  "Rear Camera")
#                     self._draw_lidar(ax_lidar)
#                     self._draw_cost(ax_cost)
#                     self._draw_trajectory(ax_traj)
#                     self._draw_velocity(ax_vel)

#                 fig.suptitle(
#                     f"VSGP Mapless Navigator  |  Mode: {self._mode}",
#                     color="#eee", fontsize=11, y=0.99,
#                 )
#                 fig.canvas.draw_idle()
#                 fig.canvas.flush_events()
#             except Exception:
#                 pass

#         plt.close(fig)

#     def _draw_camera(self, ax, img, title):
#         ax.clear()
#         ax.set_facecolor("#1a1a2e")
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#         if img is not None:
#             ax.imshow(img)

#     def _draw_lidar(self, ax):
#         ax.clear()
#         ax.set_facecolor("#1a1a2e")
#         ax.set_title("LiDAR (top view)", fontsize=8)
#         if self._lidar_pts is not None and len(self._lidar_pts) > 0:
#             pts = self._lidar_pts
#             z   = pts[:, 2]
#             z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)
#             ax.scatter(pts[:, 1], pts[:, 0], c=z_n,
#                        cmap="plasma", s=0.5, alpha=0.6)
#             ax.set_xlim(-self.cfg.lidar_range, self.cfg.lidar_range)
#             ax.set_ylim(-self.cfg.lidar_range, self.cfg.lidar_range)
#         ax.set_xlabel("Y [m]", fontsize=7)
#         ax.set_ylabel("X [m]", fontsize=7)

#     def _draw_cost(self, ax):
#         ax.clear()
#         ax.set_facecolor("#1a1a2e")
#         ax.set_title("Cost Function", fontsize=8)
#         n = len(self._cost_total)
#         if n == 0:
#             return
#         xs = np.arange(n)
#         ax.plot(xs, list(self._cost_total), c="#ff6b6b", lw=1.2, label="Total")
#         ax.plot(xs, list(self._cost_dir),   c="#ffa36c", lw=0.8, label="Direction")
#         ax.plot(xs, list(self._cost_dist),  c="#c3f584", lw=0.8, label="Distance")
#         ax.plot(xs, list(self._cost_steep), c="#74b9ff", lw=0.8, label="Steepness")
#         ax.legend(fontsize=6, loc="upper left",
#                   facecolor="#222", edgecolor="#555", labelcolor="#ddd")
#         ax.set_xlabel("Step", fontsize=7)
#         ax.set_ylabel("Cost",  fontsize=7)

#     def _draw_trajectory(self, ax):
#         ax.clear()
#         ax.set_facecolor("#1a1a2e")
#         ax.set_title("Trajectory & Global Path", fontsize=8)
#         tx, ty = list(self._traj_x), list(self._traj_y)
#         if len(tx) > 1:
#             ax.plot(ty, tx, c="#00cec9", lw=1.2, label="Actual")
#         if len(tx) > 0:
#             ax.scatter([ty[-1]], [tx[-1]], c="#fdcb6e", s=18, zorder=5)
#         sx, sy = list(self._subgoal_x), list(self._subgoal_y)
#         if sx:
#             ax.scatter(sy, sx, c="#e17055", s=12, marker="x",
#                        zorder=4, label="Subgoals")
        
#         # NEW: Draw global path
#         if self._global_path is not None and len(self._global_path) > 0:
#             global_x = [wp.y for wp in self._global_path]
#             global_y = [wp.x for wp in self._global_path]
#             ax.plot(global_x, global_y, c="#a29bfe", lw=2.0, 
#                    linestyle='--', label="Global Path", alpha=0.7)
#             # Mark reached waypoints
#             reached_x = [wp.y for wp in self._global_path if wp.reached]
#             reached_y = [wp.x for wp in self._global_path if wp.reached]
#             if reached_x:
#                 ax.scatter(reached_x, reached_y, c="#00b894", s=8, 
#                           marker='o', label="Reached WP", alpha=0.5)
        
#         # NEW: Draw goal location
#         if self._goal_location is not None:
#             ax.scatter([self._goal_location[1]], [self._goal_location[0]], 
#                       c="#ff7675", s=30, marker='*', label="GOAL", zorder=10)
        
#         ax.legend(fontsize=6, loc="upper left",
#                   facecolor="#222", edgecolor="#555", labelcolor="#ddd")
#         ax.set_xlabel("Y [m]", fontsize=7)
#         ax.set_ylabel("X [m]", fontsize=7)
#         ax.set_aspect("equal", adjustable="datalim")

#     def _draw_velocity(self, ax):
#         ax.clear()
#         ax.set_facecolor("#1a1a2e")
#         ax.set_title("Velocities", fontsize=8)
#         n = len(self._vel_lin)
#         if n == 0:
#             return
#         xs = np.arange(n)
#         ax.plot(xs, list(self._vel_lin), c="#55efc4", lw=1.2, label="Linear (m/s)")
#         ax.plot(xs, list(self._vel_ang), c="#a29bfe", lw=1.0, label="ω·v (m/s)")
#         ax.legend(fontsize=6, loc="upper left",
#                   facecolor="#222", edgecolor="#555", labelcolor="#ddd")
#         ax.set_xlabel("Step", fontsize=7)
#         ax.set_ylabel("m/s",  fontsize=7)


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §8  CARLA SENSOR WRAPPERS
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class SensorManager:
#     def __init__(self, world, vehicle, cfg: Config = CFG):
#         self.cfg     = cfg
#         self.vehicle = vehicle
#         self.world   = world
#         self._actors: list = []

#         self._lidar_queue: Queue = Queue(maxsize=1)
#         self._front_queue: Queue = Queue(maxsize=1)
#         self._rear_queue:  Queue = Queue(maxsize=1)

#         bp_lib = world.get_blueprint_library()

#         lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
#         lidar_bp.set_attribute("range",              str(cfg.lidar_range))
#         lidar_bp.set_attribute("points_per_second",  str(cfg.lidar_points_per_sec))
#         lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_freq))
#         lidar_bp.set_attribute("channels",           str(cfg.lidar_channels))
#         lidar_bp.set_attribute("upper_fov",          str(cfg.lidar_upper_fov))
#         lidar_bp.set_attribute("lower_fov",          str(cfg.lidar_lower_fov))
#         lidar = world.spawn_actor(
#             lidar_bp,
#             carla.Transform(carla.Location(x=0.5, z=2.0)),
#             attach_to=vehicle,
#         )
#         lidar.listen(self._on_lidar)
#         self._actors.append(lidar)

#         self._actors.append(self._spawn_camera(
#             bp_lib, vehicle,
#             carla.Transform(carla.Location(x=1.5, z=1.8)),
#             "front",
#         ))
#         self._actors.append(self._spawn_camera(
#             bp_lib, vehicle,
#             carla.Transform(carla.Location(x=-1.5, z=1.8),
#                             carla.Rotation(yaw=180)),
#             "rear",
#         ))

#     def _spawn_camera(self, bp_lib, vehicle, transform, tag: str):
#         cam_bp = bp_lib.find("sensor.camera.rgb")
#         cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
#         cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
#         cam_bp.set_attribute("fov",          str(self.cfg.camera_fov))
#         cam = self.world.spawn_actor(cam_bp, transform, attach_to=vehicle)
#         if tag == "front":
#             cam.listen(self._on_front_image)
#         else:
#             cam.listen(self._on_rear_image)
#         return cam

#     def _on_lidar(self, data) -> None:
#         if not self._lidar_queue.full():
#             self._lidar_queue.put(data)

#     def _on_front_image(self, image) -> None:
#         arr = np.frombuffer(image.raw_data, dtype=np.uint8)
#         arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
#         arr = arr[:, :, ::-1].copy()
#         if not self._front_queue.full():
#             self._front_queue.put(arr)

#     def _on_rear_image(self, image) -> None:
#         arr = np.frombuffer(image.raw_data, dtype=np.uint8)
#         arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
#         arr = arr[:, :, ::-1].copy()
#         if not self._rear_queue.full():
#             self._rear_queue.put(arr)

#     @property
#     def lidar_data(self):
#         try:
#             return self._lidar_queue.get_nowait()
#         except Exception:
#             return None

#     @property
#     def front_image(self) -> Optional[np.ndarray]:
#         try:
#             return self._front_queue.get_nowait()
#         except Exception:
#             return None

#     @property
#     def rear_image(self) -> Optional[np.ndarray]:
#         try:
#             return self._rear_queue.get_nowait()
#         except Exception:
#             return None

#     def destroy(self) -> None:
#         for a in self._actors:
#             try:
#                 if a.is_alive:
#                     a.destroy()
#             except Exception:
#                 pass


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §9  MANUAL KEYBOARD CONTROLLER
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class ManualController:
#     VX_SPEED    = 3.0
#     STEER_SPEED = 0.5

#     def __init__(self, cfg: Config = CFG):
#         self.ctrl   = Controller(cfg)
#         self._steer = 0.0

#     def tick(self, keys) -> ControlState:
#         vx = 0.0
#         if keys[pygame.K_w]:
#             vx = self.VX_SPEED
#         if keys[pygame.K_s]:
#             vx = -self.VX_SPEED
#         if keys[pygame.K_a]:
#             self._steer = np.clip(self._steer + self.STEER_SPEED,
#                                   -CFG.max_steer, CFG.max_steer)
#         if keys[pygame.K_d]:
#             self._steer = np.clip(self._steer - self.STEER_SPEED,
#                                   -CFG.max_steer, CFG.max_steer)
#         if keys[pygame.K_SPACE]:
#             self._steer = 0.0
#         return self.ctrl.compute_manual(vx, self._steer)


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §10  VEHICLE SPAWNER
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# def spawn_vehicle(world, cfg: Config = CFG):
#     bp_lib     = world.get_blueprint_library()
#     vehicle_bp = bp_lib.find(cfg.vehicle_blueprint)

#     spawn_transform = carla.Transform(
#         carla.Location(x=906.66, y=-875.78, z=4.7620),
#         carla.Rotation(pitch=-4.50, yaw=88.0, roll=0.0),
#     )

#     vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

#     if vehicle is None:
#         print("[WARN] Hardcoded spawn failed, trying dynamic spawn points...")
#         spawn_points = world.get_map().get_spawn_points()
#         if spawn_points:
#             spawn_transform = random.choice(spawn_points)
#             vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

#     if vehicle is None:
#         raise RuntimeError(f"Spawn failed at {spawn_transform.location}")

#     print(f"[SPAWN] Spawned at {vehicle.get_location()}")

#     wait_start = time.time()
#     while time.time() - wait_start < cfg.post_spawn_wait_s:
#         world.tick()

#     settled_z = vehicle.get_location().z
#     print(f"[SPAWN] Settled Z = {settled_z:.2f} m")
#     return vehicle, settled_z


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # §11  MAIN NAVIGATION LOOP
# # ╚═══════════════════════════════════════════════════════════════════════════╝

# class NavigationSystem:
#     def __init__(self, cfg: Config = CFG):
#         self.cfg      = cfg
#         self._running = False
#         self._mode    = "AUTO"

#         self._client: Optional[carla.Client] = None
#         self._world   = None
#         self._vehicle = None
#         self._sensors: Optional[SensorManager] = None

#         self._perception  = PerceptionModule(cfg)
#         self._planner     = VSGPSubgoalPlanner(cfg)
#         self._controller  = Controller(cfg)
#         self._manual_ctrl = ManualController(cfg)
#         self._stuck       = StuckDetector(cfg)
#         self._viz         = VisualizationModule(cfg)
#         self._global_planner = GlobalPlanner(cfg)  # NEW

#         pygame.init()
#         pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit")
#         self._screen = pygame.display.set_mode((220, 88))
#         self._font   = pygame.font.SysFont("monospace", 13)

#         self._step = 0
#         self._t0   = 0.0

#     def connect(self) -> None:
#         print(f"[CARLA] Connecting to {self.cfg.carla_host}:{self.cfg.carla_port} …")
#         self._client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
#         self._client.set_timeout(self.cfg.carla_timeout)
#         self._world = self._client.get_world()
#         if self.cfg.synchronous:
#             settings = self._world.get_settings()
#             settings.synchronous_mode = True
#             settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
#             self._world.apply_settings(settings)
#             print("[CARLA] Synchronous mode ON")

#     def setup(self) -> None:
#         self._vehicle, self._spawn_z = spawn_vehicle(self._world, self.cfg)
#         self._sensors = SensorManager(self._world, self._vehicle, self.cfg)
#         self._t0 = time.time()
#         self._viz.start()
        
#         # NEW: Initialize global planner with spawn position
#         loc = self._vehicle.get_location()
#         transform = self._vehicle.get_transform()
#         start_pos = np.array([loc.x, loc.y, loc.z])
#         start_yaw = math.radians(transform.rotation.yaw)
        
#         if self.cfg.global_planner_enabled:
#             self._global_planner.reset(start_pos, start_yaw)
#             print(f"[GLOBAL] Global planner initialized")
#             print(f"[GLOBAL] Start: X={start_pos[0]:.2f}, Y={start_pos[1]:.2f}, Z={start_pos[2]:.2f}")
#             print(f"[GLOBAL] Goal:  X={self.cfg.goal_location[0]:.2f}, Y={self.cfg.goal_location[1]:.2f}, Z={self.cfg.goal_location[2]:.2f}")
        
#         print("[SETUP] All sensors attached. Starting navigation loop.")

#     def run(self) -> None:
#         self._running = True
#         try:
#             while self._running:
#                 self._tick()
#         except KeyboardInterrupt:
#             print("\n[EXIT] KeyboardInterrupt")
#         finally:
#             self.cleanup()

#     def cleanup(self) -> None:
#         print("[CLEANUP] Stopping actors …")
#         if self._vehicle and self._vehicle.is_alive:
#             self._vehicle.apply_control(carla.VehicleControl(brake=1.0))
#         if self._sensors:
#             self._sensors.destroy()
#         if self._vehicle and self._vehicle.is_alive:
#             self._vehicle.destroy()
#         if self._world and self.cfg.synchronous:
#             settings = self._world.get_settings()
#             settings.synchronous_mode = False
#             self._world.apply_settings(settings)
#         self._viz.stop()
#         pygame.quit()
#         print("[CLEANUP] Done.")

#     def _tick(self) -> None:
#         if self.cfg.synchronous:
#             self._world.tick()

#         t_now = time.time() - self._t0
#         self._step += 1

#         ctrl_state: ControlState = ControlState()
#         best_goal:  Optional[Subgoal] = None

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self._running = False
#                 return
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     self._running = False
#                     return
#                 if event.key == pygame.K_TAB:
#                     self._mode = "MANUAL" if self._mode == "AUTO" else "AUTO"
#                     self._controller.reset_ema()
#                     print(f"[MODE] Switched to {self._mode}")

#         keys = pygame.key.get_pressed()

#         loc   = self._vehicle.get_location()
#         vel   = self._vehicle.get_velocity()
#         speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
#         pos   = np.array([loc.x, loc.y, loc.z])

#         transform = self._vehicle.get_transform()
#         pitch_rad = math.radians(transform.rotation.pitch)
#         roll_rad  = math.radians(transform.rotation.roll)
#         pitch_deg = math.degrees(pitch_rad)
#         roll_deg  = math.degrees(roll_rad)

#         # ── NEW: Check if goal is reached
#         if self.cfg.global_planner_enabled and self._global_planner.is_goal_reached():
#             print("[NAV] 🎯 GOAL REACHED! Stopping vehicle.")
#             self._vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
#             self._running = False
#             return

#         # ── Perception
#         lidar_data = self._sensors.lidar_data
#         if lidar_data is not None:
#             perc = self._perception.process_lidar(lidar_data)
#         else:
#             perc = self._perception.last_output

#         # ── Global Planner Integration (NEW)
#         global_waypoint = None
#         global_path = None
#         if self.cfg.global_planner_enabled and self._mode == "AUTO":
#             global_waypoint = self._global_planner.get_lookahead_waypoint(pos)
#             global_path = self._global_planner.waypoints

#         if self._mode == "MANUAL":
#             ctrl_state = self._manual_ctrl.tick(keys)

#         else:  # ── AUTO ────────────────────────────────────────────────────
#             stuck, rec_phase = self._stuck.update(pos, ctrl_state.steer, t_now)

#             if stuck:
#                 print("[NAV] Stuck detected. Initiating recovery.")
#                 if perc is not None:
#                     best_goal, _ = self._planner.plan(perc, pitch_rad, transform, 
#                                                       global_waypoint, mode="reverse")

#                 if best_goal:
#                     ctrl_state = self._controller.compute(
#                         best_goal, speed, 0.0, roll_deg, pitch_deg)
#                 else:
#                     ctrl_state = self._controller.compute_recovery(rec_phase)

#             else:
#                 if perc is not None:
#                     # NEW: Pass global waypoint to local planner
#                     best_goal, _ = self._planner.plan(perc, pitch_rad, transform, 
#                                                       global_waypoint, mode="forward")

#                 if best_goal:
#                     ctrl_state = self._controller.compute(
#                         best_goal, speed, best_goal.slope_deg,
#                         roll_deg=roll_deg, pitch_deg=pitch_deg,
#                     )
#                 else:
#                     # Fallback: creep toward global waypoint
#                     if global_waypoint is not None:
#                         direction = global_waypoint[:2] - pos[:2]
#                         alpha = math.atan2(direction[1], direction[0])
#                         vehicle_yaw = math.radians(transform.rotation.yaw)
#                         alpha = alpha - vehicle_yaw
                        
#                         fallback_goal = Subgoal(
#                             alpha=alpha,
#                             beta=0.0,
#                             distance=2.0,
#                             local_pos=np.array([2.0 * np.cos(alpha), 2.0 * np.sin(alpha), 0.0]),
#                             slope_deg=0.0,
#                             safe=False,
#                             width_m=self.cfg.vehicle_width,
#                             cost=999.0,
#                         )
#                     else:
#                         fallback_goal = Subgoal(
#                             alpha=0.0,
#                             beta=0.0,
#                             distance=2.0,
#                             local_pos=np.array([2.0, 0.0, 0.0]),
#                             slope_deg=0.0,
#                             safe=False,
#                             width_m=self.cfg.vehicle_width,
#                             cost=999.0,
#                         )
#                     ctrl_state = self._controller.compute(
#                         fallback_goal, speed, 0.0,
#                         roll_deg=roll_deg, pitch_deg=pitch_deg,
#                     )
#                     print("[NAV] Fallback: creeping forward")

#         # ── Apply Control
#         if self.cfg.invert_drive:
#             applied_steer   = -ctrl_state.steer
#             applied_reverse = not ctrl_state.reverse
#         else:
#             applied_steer   = ctrl_state.steer
#             applied_reverse = ctrl_state.reverse

#         self._vehicle.apply_control(carla.VehicleControl(
#             throttle=float(ctrl_state.throttle),
#             steer=float(applied_steer),
#             brake=float(ctrl_state.brake),
#             reverse=applied_reverse,
#         ))

#         # ── Pygame HUD
#         goal_dist = np.linalg.norm(pos[:2] - np.array(self.cfg.goal_location[:2]))
#         self._screen.fill((20, 20, 40))
#         lines = [
#             f"Mode: {self._mode}",
#             f"Speed: {speed:.1f} m/s",
#             f"Thr:{ctrl_state.throttle:.2f}  Str:{ctrl_state.steer:.2f}",
#             f"Roll:{roll_deg:+.1f}°  Pitch:{pitch_deg:+.1f}°",
#             f"Step: {self._step}   t={t_now:.1f}s",
#             f"Goal Dist: {goal_dist:.1f}m",  # NEW
#         ]
#         for i, line in enumerate(lines):
#             surf = self._font.render(line, True, (200, 220, 255))
#             self._screen.blit(surf, (5, 4 + i * 16))
#         pygame.display.flip()

#         # ── Push to visualization thread
#         self._viz.push_data(
#             subgoal=best_goal if self._mode == "AUTO" else None,
#             vehicle_pos=pos,
#             speed_ms=speed,
#             steer=ctrl_state.steer,
#             perc=perc,
#             mode=self._mode,
#             global_path=global_path,  # NEW
#             goal_location=np.array(self.cfg.goal_location),  # NEW
#         )
#         front_img = self._sensors.front_image
#         rear_img  = self._sensors.rear_image
#         if front_img is not None:
#             self._viz.set_front_image(front_img)
#         if rear_img is not None:
#             self._viz.set_rear_image(rear_img)

#         if not self.cfg.synchronous:
#             time.sleep(self.cfg.fixed_delta_seconds)


# def main() -> None:
#     nav = NavigationSystem(CFG)
#     try:
#         nav.connect()
#         nav.setup()
#         nav.run()
#     except RuntimeError as exc:
#         print(f"[FATAL] {exc}")
#         traceback.print_exc()
#     except Exception:
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()


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
    goal_location: Tuple[float, float, float] = (-1169.18, -413.52, 5.0)  # FIX: Z=5.0 instead of 15.16
    goal_tolerance: float       = 5.0
    global_planner_enabled: bool = True
    waypoint_spacing: float     = 10.0
    lookahead_distance: float   = 15.0

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

    # ── Stuck detection & recovery
    stuck_window_s: float       = 2.0
    stuck_disp_thresh: float    = 0.6
    stuck_oscillation_thresh: float = 0.5  # FIX: Lowered from 0.7
    recovery_duration_s: float  = 4.0      # FIX: Longer recovery
    recovery_phase_durations: Tuple[float, float, float] = (1.5, 1.5, 1.0)  # Phase timings

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
    # FIX: Add distance map - actual range in meters
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
        y = self.ROC - r  # Keep this for GP training

        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
            pts = pts[idx]

        self.vsgp.update(X, y)
        mean, var = self.vsgp.predict(self._grid_pts)

        mean = mean.reshape(self.BETA_RES, self.ALPHA_RES)
        var = var.reshape(self.BETA_RES, self.ALPHA_RES)

        # FIX: Compute actual distance map (in meters)
        # mean = ROC - r, so r = ROC - mean
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
            distance_map=distance_map,  # FIX: Include distance map
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
    escape_direction: Optional[str] = None  # FIX: Track if this is an escape maneuver


class VSGPSubgoalPlanner:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        # FIX: Adjusted thresholds based on distance_map interpretation
        self.min_free_distance = 3.0   # Minimum distance to be considered "free"
        self.max_var_thresh = 0.4
        self.max_slope_thresh = 0.25
        self.min_dist = 3.0
        self.max_dist = cfg.subgoal_distance

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
        dist_map = perc.distance_map  # FIX: Use distance map for free space detection

        alphas = perc.alpha_grid[0]
        betas = perc.beta_grid[:, 0]
        mid_beta_idx = len(betas) // 2

        # Calculate preferred direction from global waypoint if available
        preferred_alpha = 0.0
        preferred_direction = np.array([0.0, 0.0])
        if global_waypoint is not None:
            vehicle_pos = np.array([vehicle_transform.location.x, 
                                   vehicle_transform.location.y, 
                                   vehicle_transform.location.z])
            direction = global_waypoint[:2] - vehicle_pos[:2]
            preferred_direction = direction / (np.linalg.norm(direction) + 1e-9)
            preferred_alpha = math.atan2(direction[1], direction[0])
            # Convert to vehicle frame
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            preferred_alpha = preferred_alpha - vehicle_yaw

        # FIX: Mode-specific behavior
        escape_direction = None
        if mode == "escape_left":
            escape_direction = "left"
        elif mode == "escape_right":
            escape_direction = "right"
        elif mode == "escape":
            # Default to turning toward global direction
            escape_direction = "toward_global"

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

                # FIX: Use distance_map for free space detection
                # d_map value = actual distance in meters
                # If distance > threshold, it's free
                actual_distance = dist_map[b_idx, a_idx]
                is_free = actual_distance > self.min_free_distance
                is_stable = var[b_idx, a_idx] < self.max_var_thresh
                is_safe = slope[b_idx, a_idx] < self.max_slope_thresh

                if is_free and is_stable and is_safe:
                    # Cost calculation
                    direction_cost = abs(a) / (np.pi / 2)  # Straight is best

                    # FIX: Add mode-specific cost adjustments
                    if mode in ("escape_left", "escape_right", "escape"):
                        # Prefer sharp turns for escape
                        escape_bonus = 0.0
                        if mode == "escape_left" and a < -0.3:
                            escape_bonus = -2.0  # Reward going left
                        elif mode == "escape_right" and a > 0.3:
                            escape_bonus = -2.0  # Reward going right
                        elif mode == "escape" and escape_direction == "toward_global":
                            # Bonus for going toward global direction
                            local_dir = np.array([np.cos(a), np.sin(a)])
                            alignment = np.dot(local_dir, preferred_direction)
                            escape_bonus = -alignment * 1.5
                        
                        cost = (
                            self.cfg.w_direction * direction_cost +
                            self.cfg.w_flatness * s_val +
                            0.5 * var[b_idx, a_idx] +
                            escape_bonus
                        )
                    else:
                        # Normal forward mode
                        global_penalty = abs(a - preferred_alpha) * 1.0 if global_waypoint is not None else 0.0
                        cost = (
                            self.cfg.w_direction * direction_cost +
                            self.cfg.w_flatness * s_val +
                            0.5 * var[b_idx, a_idx] +
                            global_penalty
                        )
                        cost -= d * 0.3  # Prefer longer distances

                    if cost < best_point_cost:
                        best_point_cost = cost
                        best_point_dist = d
                        s_val = slope[b_idx, a_idx]
                else:
                    # Path blocked - don't continue to farther distances
                    break

            if best_point_dist > 0:
                lx = best_point_dist * np.cos(a)
                ly = best_point_dist * np.sin(a)

                # Calculate world position
                vehicle_pos = np.array([vehicle_transform.location.x, 
                                       vehicle_transform.location.y, 
                                       vehicle_transform.location.z])
                vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                
                # Transform local to world
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
                )
                candidates.append(sg)

        if not candidates:
            return None, []

        candidates.sort(key=lambda s: s.cost)
        return candidates[0], candidates


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §4.5  GLOBAL PLANNER
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class GlobalWaypoint:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    reached: bool = False


class GlobalPlanner:
    """Generates a global path from current position to final destination."""

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
        """Compute waypoints from start position to goal using straight-line interpolation."""
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
        """Get the current target waypoint. Advances when reached."""
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
        """Get a lookahead point on the global path for the local planner."""
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
        """Check if path replanning is needed (stuck or significant deviation)."""
        if not self.path_computed or len(self.waypoints) == 0:
            return False

        if self.current_waypoint_idx >= len(self.waypoints):
            return False

        # Check if we're significantly off the path
        current_wp = self.waypoints[self.current_waypoint_idx]
        wp_pos = np.array([current_wp.x, current_wp.y, current_wp.z])
        dist_to_wp = np.linalg.norm(vehicle_pos - wp_pos)

        # If we can't reach current waypoint within 2x spacing, replan
        if dist_to_wp > self.waypoint_spacing * 2:
            return True

        return False

    def reset(self, start_pos: np.ndarray, start_yaw: float) -> None:
        """Reset the global planner with new start position."""
        self.goal_reached = False
        self.current_waypoint_idx = 0
        self.path_computed = False
        self.compute_path(start_pos, start_yaw)

    def is_goal_reached(self) -> bool:
        return self.goal_reached


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
        throttle = np.clip(vx / self.cfg.base_speed, 0.0, self.cfg.max_throttle)
        return self._apply_smooth(throttle, steer, 0.0, False)

    def compute_recovery(self, phase: int, direction_hint: Optional[str] = None) -> ControlState:
        """
        Recovery maneuver that respects vehicle physics and provides variety.
        Phase 0: Reverse straight (clear behind)
        Phase 1: Reverse while turning (away from obstacle)
        Phase 2: Reverse while turning (other direction or toward escape)
        
        direction_hint: 'left', 'right', 'toward_global', or None
        """
        # Determine steering based on direction hint
        if direction_hint == 'left' or direction_hint == 'right':
            steer = 0.6 if direction_hint == 'left' else -0.6
        elif direction_hint == 'toward_global':
            steer = 0.0  # Go straight toward goal
        else:
            # Use phase to determine direction
            if phase == 0:
                steer = 0.0
            elif phase == 1:
                steer = 0.7  # Turn left
            else:
                steer = -0.7  # Turn right

        if phase == 0:
            # Reverse straight with moderate speed
            return ControlState(throttle=0.45, steer=steer, brake=0.0, reverse=True)
        elif phase == 1:
            # Reverse with turn - faster escape
            return ControlState(throttle=0.5, steer=steer, brake=0.0, reverse=True)
        else:
            # Continue reversing with opposite turn
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
# §6  STUCK DETECTION & RECOVERY (FIXED)
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

    def update(self, pos: np.ndarray, steer: float, t: float) -> Tuple[bool, int]:
        self._pos_history.append((t, pos.copy()))
        self._steer_history.append(steer)

        if self._state == self.RECOVERING:
            elapsed = t - self._recovery_start
            durations = self.cfg.recovery_phase_durations
            
            # Calculate cumulative time for each phase
            phase_end_times = [durations[0], durations[0] + durations[1], 
                              durations[0] + durations[1] + durations[2]]
            
            if elapsed >= sum(durations):
                # Recovery complete
                print(f"[STUCK] Recovery complete. Resetting to IDLE.")
                self._state = self.IDLE
                self._recovery_phase = 0
                self._recovery_direction = None
                self._pos_history.clear()
                return False, 0
            
            # Determine current phase
            if elapsed < phase_end_times[0]:
                self._recovery_phase = 0
            elif elapsed < phase_end_times[1]:
                self._recovery_phase = 1
            else:
                self._recovery_phase = 2
            
            return True, self._recovery_phase

        # Check for stuck condition
        is_stuck = False
        stuck_reason = ""

        # Check displacement over time window
        if len(self._pos_history) >= 2:
            oldest_t, oldest_pos = self._pos_history[0]
            if (t - oldest_t) >= self.cfg.stuck_window_s:
                disp = np.linalg.norm(pos - oldest_pos)
                if disp < self.cfg.stuck_disp_thresh:
                    is_stuck = True
                    stuck_reason = f"low displacement ({disp:.2f}m < {self.cfg.stuck_disp_thresh}m)"
                elif disp < self.cfg.stuck_disp_thresh * 2:
                    # Check for oscillation
                    if self._check_oscillation():
                        is_stuck = True
                        stuck_reason = "oscillation detected"
        
        # Also check steering oscillation
        if len(self._steer_history) >= self._steer_history.maxlen:
            arr = np.array(self._steer_history)
            signs = np.sign(arr + 1e-9)
            flips = int(np.sum(np.diff(signs) != 0))
            flip_ratio = flips / (len(signs) - 1)
            
            if flip_ratio > self.cfg.stuck_oscillation_thresh:
                is_stuck = True
                stuck_reason = f"steering oscillation ({flip_ratio:.1%})"

        if is_stuck:
            # Prevent rapid re-triggering
            if t - self._last_stuck_time < 3.0:
                self._consecutive_stuck_count += 1
            else:
                self._consecutive_stuck_count = 1
            
            self._last_stuck_time = t
            
            # Determine recovery direction
            if self._consecutive_stuck_count >= 2:
                # After multiple stuck events, try to determine best escape direction
                self._recovery_direction = self._determine_escape_direction()
            else:
                # First stuck - alternate directions
                self._recovery_direction = "left" if self._recovery_phase % 2 == 0 else "right"
            
            self._trigger_recovery(t, stuck_reason)
            return True, 0

        return False, 0

    def _check_oscillation(self) -> bool:
        """Check if vehicle is oscillating (alternating direction)."""
        if len(self._steer_history) < 10:
            return False
        
        arr = np.array(list(self._steer_history))
        signs = np.sign(arr + 1e-9)
        flips = np.sum(np.diff(signs) != 0)
        
        return flips > len(signs) * self.cfg.stuck_oscillation_thresh

    def _determine_escape_direction(self) -> str:
        """Determine which direction to escape based on steering history."""
        if len(self._steer_history) < 5:
            return "left"
        
        arr = np.array(list(self._steer_history))
        mean_steer = np.mean(arr)
        
        # If been turning right a lot, escape left and vice versa
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

    def is_recovering(self) -> bool:
        return self._state == self.RECOVERING

    def get_recovery_direction(self) -> Optional[str]:
        """Get the preferred recovery direction."""
        return self._recovery_direction

    def reset(self) -> None:
        """Manually reset stuck state."""
        self._state = self.IDLE
        self._recovery_phase = 0
        self._recovery_direction = None
        self._consecutive_stuck_count = 0
        self._pos_history.clear()
        self._steer_history.clear()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §7  VISUALIZATION MODULE (Threaded)
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
        self._cost_total: deque = deque(maxlen=maxlen)
        self._cost_dir: deque = deque(maxlen=maxlen)
        self._cost_dist: deque = deque(maxlen=maxlen)
        self._cost_steep: deque = deque(maxlen=maxlen)
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
                self._cost_total.append(subgoal.cost)
                self._cost_dir.append(abs(subgoal.alpha) / (math.pi / 2) * CFG.w_direction)
                self._cost_dist.append(
                    abs(subgoal.distance - CFG.subgoal_distance) / CFG.subgoal_distance * CFG.w_distance)
                self._cost_steep.append(subgoal.slope_deg / CFG.max_slope_deg * CFG.w_steepness)
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
        fig = plt.figure("VSGP Navigator", figsize=(16, 9))
        fig.patch.set_facecolor("#111")
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        ax_front = fig.add_subplot(gs[0, 0])
        ax_lidar = fig.add_subplot(gs[0, 1])
        ax_rear = fig.add_subplot(gs[0, 2])
        ax_cost = fig.add_subplot(gs[1, 0])
        ax_traj = fig.add_subplot(gs[1, 1])
        ax_vel = fig.add_subplot(gs[1, 2])

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
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Cost Function", fontsize=8)
        n = len(self._cost_total)
        if n == 0:
            return
        xs = np.arange(n)
        ax.plot(xs, list(self._cost_total), c="#ff6b6b", lw=1.2, label="Total")
        ax.plot(xs, list(self._cost_dir), c="#ffa36c", lw=0.8, label="Direction")
        ax.plot(xs, list(self._cost_dist), c="#c3f584", lw=0.8, label="Distance")
        ax.plot(xs, list(self._cost_steep), c="#74b9ff", lw=0.8, label="Steepness")
        ax.legend(fontsize=6, loc="upper left", facecolor="#222", edgecolor="#555", labelcolor="#ddd")
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Cost", fontsize=7)

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# §8  CARLA SENSOR WRAPPERS
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
# §9  MANUAL KEYBOARD CONTROLLER
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ManualController:
    VX_SPEED = 3.0
    STEER_SPEED = 0.5

    def __init__(self, cfg: Config = CFG):
        self.ctrl = Controller(cfg)
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
# §10  VEHICLE SPAWNER
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
# §11  MAIN NAVIGATION LOOP (FIXED)
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

        pygame.init()
        pygame.display.set_caption("VSGP Nav – TAB=toggle, ESC=quit")
        self._screen = pygame.display.set_mode((260, 100))
        self._font = pygame.font.SysFont("monospace", 13)

        self._step = 0
        self._t0 = 0.0

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
                    self._stuck.reset()  # FIX: Reset stuck detector on mode change
                    print(f"[MODE] Switched to {self._mode}")
                # FIX: R key to reset stuck state manually
                if event.key == pygame.K_r:
                    self._stuck.reset()
                    print("[MODE] Stuck state reset")

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

        # FIX: Check if goal reached
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

        else:  # AUTO mode
            # FIX: Use proper stuck detection
            stuck, rec_phase = self._stuck.update(pos, ctrl_state.steer, t_now)
            is_stuck = stuck
            recovery_phase = rec_phase

            if stuck:
                print(f"[NAV] 🚗 STUCK DETECTED! Recovery phase {rec_phase}")
                
                # FIX: Always use recovery maneuvers when stuck
                recovery_direction = self._stuck.get_recovery_direction()
                ctrl_state = self._controller.compute_recovery(rec_phase, recovery_direction)
                
                # FIX: Also try to find an escape route during recovery
                if perc is not None:
                    # Try to find a clear escape path
                    if rec_phase == 1 and recovery_direction:
                        # In phase 1, look for escape toward the hint direction
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
                # Normal navigation
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
                    # FIX: No valid path - be more aggressive with recovery
                    print("[NAV] ⚠️ No valid path found!")
                    
                    # Check if we should trigger stuck detection manually
                    if speed < 0.5:
                        # Very slow or stopped - likely stuck
                        self._stuck.update(pos, ctrl_state.steer, t_now)
                    
                    # Fallback: creep toward global waypoint
                    if global_waypoint is not None:
                        direction = global_waypoint[:2] - pos[:2]
                        dist_to_goal_dir = np.linalg.norm(direction)
                        if dist_to_goal_dir > 0.1:
                            direction = direction / dist_to_goal_dir
                            # Compute alpha in vehicle frame
                            alpha = math.atan2(direction[1], direction[0]) - vehicle_yaw
                            # Normalize to [-pi, pi]
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
                            # At waypoint, just stop
                            fallback_goal = None
                    else:
                        fallback_goal = None
                    
                    if fallback_goal:
                        ctrl_state = self._controller.compute(
                            fallback_goal, speed, 0.0,
                            roll_deg=roll_deg, pitch_deg=pitch_deg,
                        )
                    else:
                        # Complete stop
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
        self._screen.fill((20, 20, 40))
        mode_color = (255, 100, 100) if is_stuck else (200, 220, 255)
        status = "STUCK!" if is_stuck else self._mode
        lines = [
            f"Mode: {status}",
            f"Speed: {speed:.1f} m/s",
            f"Thr:{ctrl_state.throttle:.2f}  Str:{ctrl_state.steer:.2f}",
            f"Roll:{roll_deg:+.1f}°  Pitch:{pitch_deg:+.1f}°",
            f"Step: {self._step}   t={t_now:.1f}s",
            f"Goal Dist: {goal_dist:.1f}m",
        ]
        for i, line in enumerate(lines):
            color = mode_color if i == 0 else (200, 220, 255)
            surf = self._font.render(line, True, color)
            self._screen.blit(surf, (5, 4 + i * 16))
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
