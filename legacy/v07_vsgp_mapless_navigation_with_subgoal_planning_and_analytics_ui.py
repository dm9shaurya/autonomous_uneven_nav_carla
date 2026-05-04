import carla
import argparse
import random
import time
import math
import colorsys
import numpy as np
from collections import deque

# ─────────────────────────────────────────────────────────────
# NEW IMPORTS FOR ADVANCED NAV
# ─────────────────────────────────────────────────────────────
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not found. GP will use fallback variance estimation.")

try:
    import pygame
    from pygame.locals import (K_ESCAPE, K_q, K_r, K_w, K_s, K_a, K_d, K_p, K_m)
except ImportError:
    raise RuntimeError("pygame not installed — run: pip install pygame")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for pygame embedding
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not found. Plots will use simplified pygame rendering.")


# ─────────────────────────────────────────────────────────────
# CONSTANTS & WEIGHTS (PAPER-BASED)
# ─────────────────────────────────────────────────────────────

# ── Legacy Constants (Preserved for CARLA setup) ────────────
FRONT_WARN_DIST  =  8.0
FRONT_STOP_DIST  =  5.0
FRONT_EMRG_DIST  =  3.0
OBSTACLE_Z_MIN   = -1.8
OBSTACLE_Z_MAX   =  2.5
SLOPE_BLOCK_Z    =  1.2
MAX_SPEED_MS     =  7.0
THROTTLE_CRUISE  =  0.85
THROTTLE_CAUTION =  0.45
STUCK_WINDOW_FRAMES = 90
STUCK_MIN_TRAVEL_M  =  1.5
REVERSE_FRAMES      = 50
OSC_WINDOW      = 24
RAINBOW_Z_MIN = -2.5
RAINBOW_Z_MAX =  3.0
FALL_Z_THRESH  = -10.0
SPAWN_SETTLE_S =   3.0
LIDAR_CHANNELS     = "32"
LIDAR_PTS_PER_SEC  = "56000"
LIDAR_ROT_FREQ     = "10"
LIDAR_RANGE        = "50"
LIDAR_FRAME_SKIP   = 2
RGB_WIDTH  = 320
RGB_HEIGHT = 180
INVERT_DRIVE = True
_R_FWD = INVERT_DRIVE
_R_REV = not INVERT_DRIVE

# ── NEW: VSGP Navigation Constants (PAPER-BASED) ────────────
# Spherical projection
ROC = 7.0  # Reference occupancy surface radius (meters)

# GP/VSGP Parameters
GP_LENGTH_SCALE = 0.5  # radians
GP_NOISE = 0.1
GP_MAX_POINTS = 300  # Sparse GP - limit training points
GP_UPDATE_FREQ = 3   # Update GP every N frames

# Variance Threshold for Segmentation
VARIANCE_THRESHOLD = 0.7

# Subgoal Parameters
SUBGOAL_DISTANCE_MIN = 3.0
SUBGOAL_DISTANCE_MAX = 12.0
SUBGOAL_ANGLES = np.linspace(-45, 45, 19) * (math.pi / 180.0)  # -45° to +45°
ROBOT_WIDTH = 2.0
ROBOT_MARGIN = 0.5

# Cost Function Weights (STRICTLY FOLLOW PAPER)
K_DIR = 0.2  # Direction cost weight
K_DST = 0.3  # Distance/forward cost weight
K_STP = 0.5  # Steepness cost weight

# Safety Constraints (radians)
MAX_ROLL  = 0.524   # 30 degrees
MAX_PITCH = 0.785   # 45 degrees

# Control Smoothing
STEER_EMA_ALPHA = 0.3
THROTTLE_EMA_ALPHA = 0.4


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

class CustomTimer:
    def __init__(self):
        self.timer = time.perf_counter
    def time(self):
        return self.timer()


def _hsv_to_rgb_vectorised(hue_arr):
    """Pure-numpy HSV→RGB for array of hues (S=1, V=1)."""
    h6 = hue_arr * 6.0
    hi = np.floor(h6).astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    q  = (1.0 - f)
    rgb = np.zeros((len(hue_arr), 3), dtype=np.float32)
    lut = [(1.0, f, 0.0), (q, 1.0, 0.0), (0.0, 1.0, f),
           (0.0, q, 1.0), (f, 0.0, 1.0), (1.0, 0.0, q)]
    for s, (rv, gv, bv) in enumerate(lut):
        m = hi == s
        if not m.any(): continue
        rgb[m, 0] = rv if np.isscalar(rv) else rv[m]
        rgb[m, 1] = gv if np.isscalar(gv) else gv[m]
        rgb[m, 2] = bv if np.isscalar(bv) else bv[m]
    return (rgb * 255).astype(np.uint8)


def _precompute_colour_bar(height, bar_w=12):
    """Build (bar_w, height, 3) uint8 rainbow legend."""
    bar = np.zeros((bar_w, height, 3), dtype=np.uint8)
    for py in range(height):
        norm = 1.0 - py / height
        hue  = (1.0 - norm) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bar[:, py] = (int(r * 255), int(g * 255), int(b * 255))
    return bar


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian (x, y, z) to spherical (alpha, beta, r)
    alpha: azimuth angle (-π to π)
    beta: elevation angle (-π/2 to π/2)
    r: radial distance
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arctan2(y, x)
    beta = np.arcsin(np.clip(z / r, -1, 1))
    return alpha, beta, r


def spherical_to_cartesian(alpha, beta, r):
    """Convert spherical back to Cartesian."""
    x = r * np.cos(beta) * np.cos(alpha)
    y = r * np.cos(beta) * np.sin(alpha)
    z = r * np.sin(beta)
    return x, y, z


# ─────────────────────────────────────────────────────────────
# VSGP PERCEPTION MODULE (PAPER-BASED)
# ─────────────────────────────────────────────────────────────

class VSGPPerception:
    """
    Sparse Gaussian Process Perception for Mapless Navigation.
    Based on: "Autonomous Mapless Navigation on Uneven Terrains"
    
    Key Concepts:
    1. LiDAR → Spherical Occupancy Surface (alpha, beta, r)
    2. f(z) = roc - r (occupancy function)
    3. GP predicts mean & variance over (alpha, beta)
    4. Variance threshold segments free/occupied space
    """
    
    def __init__(self):
        self.gp = None
        self.X_train = None  # (alpha, beta) training points
        self.y_train = None  # f(z) = roc - r values
        self.variance_map = None
        self.mean_map = None
        self.frame_count = 0
        self.use_gp = SKLEARN_AVAILABLE
        
        if self.use_gp:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(GP_LENGTH_SCALE, (1e-2, 1e2)) + WhiteKernel(GP_NOISE)
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=0,
                normalize_y=True
            )
    
    def update(self, lidar_points):
        """
        Process LiDAR points through VSGP pipeline.
        
        Returns:
            free_segments: list of (alpha_center, beta_center, width)
            variance_map: 2D variance surface
            mean_map: 2D mean occupancy surface
        """
        self.frame_count += 1
        
        if lidar_points is None or len(lidar_points) < 10:
            return [], None, None
        
        # ── Step 1: Cartesian → Spherical ────────────────────
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        alpha, beta, r = cartesian_to_spherical(x, y, z)
        
        # Filter valid points (forward hemisphere, reasonable range)
        valid_mask = (r > 0.5) & (r < ROC * 2) & (np.abs(beta) < math.pi/3)
        alpha = alpha[valid_mask]
        beta = beta[valid_mask]
        r = r[valid_mask]
        
        if len(alpha) < 20:
            return [], None, None
        
        # ── Step 2: Compute Occupancy Function f(z) = roc - r ─
        f_z = ROC - r
        
        # ── Step 3: Downsample for Sparse GP ─────────────────
        if len(alpha) > GP_MAX_POINTS:
            indices = np.random.choice(len(alpha), GP_MAX_POINTS, replace=False)
            alpha = alpha[indices]
            beta = beta[indices]
            f_z = f_z[indices]
        
        X = np.column_stack([alpha, beta])
        y = f_z
        
        # ── Step 4: GP Training (every GP_UPDATE_FREQ frames) ─
        if self.frame_count % GP_UPDATE_FREQ == 0 and self.use_gp:
            try:
                self.X_train = X
                self.y_train = y
                self.gp.fit(X, y)
            except Exception:
                pass
        
        # ── Step 5: Generate Variance & Mean Maps ────────────
        # Create grid over (alpha, beta) space
        alpha_grid = np.linspace(-math.pi/2, math.pi/2, 40)
        beta_grid = np.linspace(-math.pi/6, math.pi/6, 20)
        Alpha, Beta = np.meshgrid(alpha_grid, beta_grid)
        X_test = np.column_stack([Alpha.ravel(), Beta.ravel()])
        
        if self.use_gp and self.gp is not None:
            try:
                mean, std = self.gp.predict(X_test, return_std=True)
                self.mean_map = mean.reshape(Alpha.shape)
                self.variance_map = (std ** 2).reshape(Alpha.shape)
            except Exception:
                self.mean_map = np.zeros_like(Alpha)
                self.variance_map = np.ones_like(Alpha) * VARIANCE_THRESHOLD
        else:
            # Fallback: simple variance estimation
            self.variance_map = np.var(y) * np.ones((20, 40))
            self.mean_map = np.mean(y) * np.ones((20, 40))
        
        # ── Step 6: Variance-based Segmentation ──────────────
        free_segments = self._segment_free_space(alpha, beta, self.variance_map, Alpha, Beta)
        
        return free_segments, self.variance_map, self.mean_map
    
    def _segment_free_space(self, alpha, beta, var_map, Alpha_grid, Beta_grid):
        """
        Extract free space segments from variance map.
        Vth = 0.7: variance below threshold = free space
        """
        segments = []
        
        if var_map is None:
            return segments
        
        # Find regions where variance < threshold (free space)
        free_mask = var_map < VARIANCE_THRESHOLD
        
        # For each column (alpha), find contiguous free regions (beta)
        for col_idx in range(var_map.shape[1]):
            col_free = free_mask[:, col_idx]
            
            # Find contiguous segments
            start_idx = None
            for row_idx in range(len(col_free)):
                if col_free[row_idx] and start_idx is None:
                    start_idx = row_idx
                elif not col_free[row_idx] and start_idx is not None:
                    # Segment ended
                    beta_center = np.mean(Beta_grid[start_idx:row_idx, col_idx])
                    alpha_center = Alpha_grid[start_idx, col_idx]
                    width = row_idx - start_idx
                    if width >= 2:  # Minimum segment width
                        segments.append({
                            'alpha': alpha_center,
                            'beta': beta_center,
                            'width': width,
                            'variance': np.mean(var_map[start_idx:row_idx, col_idx])
                        })
                    start_idx = None
            
            # Handle segment at end of column
            if start_idx is not None:
                beta_center = np.mean(Beta_grid[start_idx:, col_idx])
                alpha_center = Alpha_grid[start_idx, col_idx]
                width = len(col_free) - start_idx
                if width >= 2:
                    segments.append({
                        'alpha': alpha_center,
                        'beta': beta_center,
                        'width': width,
                        'variance': np.mean(var_map[start_idx:, col_idx])
                    })
        
        return segments


# ─────────────────────────────────────────────────────────────
# SUBGOAL PLANNER (PAPER-BASED)
# ─────────────────────────────────────────────────────────────

class SubgoalPlanner:
    """
    Subgoal-based planning with cost function:
    J(g) = kdir * direction + kdst * distance + kstp * steepness
    
    Steepness Cost: Cstp = dz² + exp(sign(pitch) * dz) * |pitch|
    """
    
    def __init__(self):
        self.subgoals = []
        self.selected_subgoal = None
        self.cost_history = deque(maxlen=100)
        self.trajectory_history = deque(maxlen=200)
        self.control_history = deque(maxlen=100)
    
    def generate_subgoals(self, free_segments, vehicle_transform, vsgp_perception):
        """
        Generate candidate subgoals from free space segments.
        """
        self.subgoals = []
        
        if not free_segments:
            return self.subgoals
        
        vehicle_loc = vehicle_transform.location
        vehicle_rot = vehicle_transform.rotation
        vehicle_yaw = math.radians(vehicle_rot.yaw)
        
        for seg in free_segments:
            # Convert spherical segment to Cartesian subgoal
            for distance in np.linspace(SUBGOAL_DISTANCE_MIN, SUBGOAL_DISTANCE_MAX, 3):
                alpha = seg['alpha']
                beta = seg['beta']
                
                # Global angle = vehicle_yaw + alpha
                global_alpha = vehicle_yaw + alpha
                
                # Cartesian coordinates relative to vehicle
                x = distance * np.cos(beta) * np.cos(alpha)
                y = distance * np.cos(beta) * np.sin(alpha)
                z = distance * np.sin(beta)
                
                # Get terrain info from GP
                mean_z = 0.0
                variance = seg.get('variance', 1.0)
                
                if vsgp_perception.mean_map is not None:
                    # Lookup mean from GP map
                    try:
                        alpha_idx = int((alpha + math.pi/2) / math.pi * 40)
                        beta_idx = int((beta + math.pi/6) / (math.pi/3) * 20)
                        alpha_idx = np.clip(alpha_idx, 0, 39)
                        beta_idx = np.clip(beta_idx, 0, 19)
                        mean_z = vsgp_perception.mean_map[beta_idx, alpha_idx]
                    except Exception:
                        mean_z = 0.0
                
                # Calculate pitch and roll estimates
                pitch = beta  # Approximation from elevation angle
                roll = 0.0    # Would need lateral variance for true roll
                
                subgoal = {
                    'x': x,
                    'y': y,
                    'z': z + mean_z,
                    'alpha': alpha,
                    'beta': beta,
                    'distance': distance,
                    'variance': variance,
                    'pitch': pitch,
                    'roll': roll,
                    'safe': True,
                    'cost': float('inf'),
                    'cost_breakdown': {}
                }
                
                # Safety constraints
                if abs(roll) >= MAX_ROLL or abs(pitch) >= MAX_PITCH:
                    subgoal['safe'] = False
                
                self.subgoals.append(subgoal)
        
        return self.subgoals
    
    def evaluate_costs(self, vehicle_transform):
        """
        Evaluate cost function for all subgoals.
        J(g) = kdir * direction + kdst * distance + kstp * steepness
        """
        vehicle_rot = vehicle_transform.rotation
        vehicle_yaw = math.radians(vehicle_rot.yaw)
        
        for sg in self.subgoals:
            if not sg['safe']:
                sg['cost'] = float('inf')
                continue
            
            # Direction cost (prefer straight ahead)
            direction_cost = abs(sg['alpha']) / (math.pi / 2)  # Normalize to [0, 1]
            
            # Distance cost (prefer forward progress, not too far)
            distance_cost = 1.0 - (sg['distance'] / SUBGOAL_DISTANCE_MAX)  # Prefer farther
            distance_cost = max(0, distance_cost)
            
            # Steepness cost (MANDATORY formula from paper)
            dz = sg['z']  # Height difference
            pitch = sg['pitch']
            steepness_cost = dz**2 + np.exp(np.sign(pitch) * dz) * abs(pitch)
            steepness_cost = min(steepness_cost, 10.0)  # Cap for stability
            steepness_cost = steepness_cost / 10.0  # Normalize
            
            # Total cost
            total_cost = (K_DIR * direction_cost + 
                         K_DST * distance_cost + 
                         K_STP * steepness_cost)
            
            sg['cost'] = total_cost
            sg['cost_breakdown'] = {
                'direction': K_DIR * direction_cost,
                'distance': K_DST * distance_cost,
                'steepness': K_STP * steepness_cost,
                'total': total_cost
            }
        
        # Store cost history for plotting
        if self.subgoals:
            best_sg = min(self.subgoals, key=lambda s: s['cost'])
            self.cost_history.append(best_sg['cost_breakdown'])
    
    def select_best_subgoal(self):
        """Select minimum cost safe subgoal."""
        safe_subgoals = [sg for sg in self.subgoals if sg['safe']]
        
        if not safe_subgoals:
            self.selected_subgoal = None
            return None
        
        self.selected_subgoal = min(safe_subgoals, key=lambda s: s['cost'])
        return self.selected_subgoal
    
    def add_trajectory_point(self, location):
        """Add actual vehicle position to trajectory history."""
        self.trajectory_history.append((location.x, location.y))
    
    def add_control_point(self, speed, steer):
        """Add control inputs to history."""
        self.control_history.append({'speed': speed, 'steer': steer})


# ─────────────────────────────────────────────────────────────
# VISUALIZATION MANAGER (REDESIGNED UI)
# ─────────────────────────────────────────────────────────────

class VisualizationManager:
    """
    Redesigned UI with:
    - Top Row: Front Cam, LiDAR, Right Cam (unchanged)
    - Bottom Row: Cost Plot, Trajectory+Subgoals Map, Control Plot
    - Extra: Subgoal decision overlay under LiDAR
    """
    
    def __init__(self, display_manager):
        self.dm = display_manager
        self.font = pygame.font.SysFont("monospace", 11)
        self.bold_font = pygame.font.SysFont("monospace", 13, bold=True)
        
        # Colors
        self.C_FREE = (0, 255, 0)
        self.C_OCC = (255, 0, 0)
        self.C_UNK = (100, 100, 100)
        self.C_TRAJ = (0, 200, 100)  # GREEN
        self.C_PRED = (0, 200, 255)  # CYAN
        self.C_SUBGOAL_BEST = (255, 255, 0)  # YELLOW
        self.C_SUBGOAL_REJECT = (255, 50, 50)  # RED
        self.C_SUBGOAL_CANDIDATE = (100, 100, 255)  # BLUE
        self.C_SUBGOAL_UNCERTAIN = (255, 200, 0)  # ORANGE/YELLOW
        
        # Plot surfaces (cached)
        self.cost_surface = None
        self.traj_surface = None
        self.ctrl_surface = None
        self.subgoal_overlay = None
        
        # Initialize matplotlib figures if available
        if MATPLOTLIB_AVAILABLE:
            self._init_matplotlib_plots()
    
    def _init_matplotlib_plots(self):
        """Initialize matplotlib figures for high-quality plots."""
        # Cost Plot
        self.fig_cost = plt.figure(figsize=(3, 2), dpi=80)
        self.ax_cost = self.fig_cost.add_subplot(111)
        self.canvas_cost = FigureCanvas(self.fig_cost)
        
        # Trajectory Plot
        self.fig_traj = plt.figure(figsize=(3, 2), dpi=80)
        self.ax_traj = self.fig_traj.add_subplot(111)
        self.canvas_traj = FigureCanvas(self.fig_traj)
        
        # Control Plot
        self.fig_ctrl = plt.figure(figsize=(3, 2), dpi=80)
        self.ax_ctrl = self.fig_ctrl.add_subplot(111)
        self.canvas_ctrl = FigureCanvas(self.fig_ctrl)
    
    def update_plots(self, planner, vehicle):
        """Update all plot data."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # ── Cost Function Plot ───────────────────────────────
        self.ax_cost.clear()
        if len(planner.cost_history) > 1:
            costs = list(planner.cost_history)
            x_vals = range(len(costs))
            
            dir_costs = [c['direction'] for c in costs]
            dst_costs = [c['distance'] for c in costs]
            stp_costs = [c['steepness'] for c in costs]
            tot_costs = [c['total'] for c in costs]
            
            self.ax_cost.plot(x_vals, tot_costs, 'k-', linewidth=2, label='Total')
            self.ax_cost.plot(x_vals, dir_costs, 'b--', linewidth=1, label='Direction')
            self.ax_cost.plot(x_vals, dst_costs, 'g-.', linewidth=1, label='Distance')
            self.ax_cost.plot(x_vals, stp_costs, 'r:', linewidth=1, label='Steepness')
            
            self.ax_cost.set_xlabel('Frame')
            self.ax_cost.set_ylabel('Cost')
            self.ax_cost.legend(loc='upper right', fontsize=7)
            self.ax_cost.set_title('Cost Function Over Time', fontsize=9)
            self.ax_cost.grid(True, alpha=0.3)
            self.ax_cost.set_ylim(0, max(1.5, max(tot_costs) * 1.2))
        
        self.canvas_cost.draw()
        self.cost_surface = self._canvas_to_pygame(self.canvas_cost)
        
        # ── Trajectory + Subgoals Map ────────────────────────
        self.ax_traj.clear()
        
        # Plot actual trajectory
        if len(planner.trajectory_history) > 1:
            traj = list(planner.trajectory_history)
            traj_x = [p[0] for p in traj]
            traj_y = [p[1] for p in traj]
            self.ax_traj.plot(traj_x, traj_y, 'g-', linewidth=2, label='Actual')
        
        # Plot subgoals - FIXED: Initialize sg_x and sg_y BEFORE the if block
        sg_x = []
        sg_y = []
        
        if planner.subgoals:
            sg_x = [sg['x'] for sg in planner.subgoals]
            sg_y = [sg['y'] for sg in planner.subgoals]
            
            for sg in planner.subgoals:
                if sg == planner.selected_subgoal:
                    color = 'yellow'
                    marker = 'o'
                    size = 100
                elif not sg['safe']:
                    color = 'red'
                    marker = 'x'
                    size = 50
                elif sg['variance'] > VARIANCE_THRESHOLD * 0.8:
                    color = 'orange'
                    marker = 's'
                    size = 60
                else:
                    color = 'blue'
                    marker = '.'
                    size = 30
                
                self.ax_traj.scatter(sg['x'], sg['y'], c=color, marker=marker, 
                                   s=size, alpha=0.7, edgecolors='black')
        
        # Plot vehicle at origin
        self.ax_traj.scatter(0, 0, c='green', marker='^', s=150, label='Vehicle')
        
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.legend(loc='upper right', fontsize=7)
        self.ax_traj.set_title('Trajectory & Subgoals', fontsize=9)
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.set_aspect('equal')
        
        # Set reasonable bounds - FIXED: sg_x and sg_y now always defined
        if len(planner.trajectory_history) > 0:
            all_x = [p[0] for p in planner.trajectory_history] + sg_x
            all_y = [p[1] for p in planner.trajectory_history] + sg_y
            margin = 2.0
            self.ax_traj.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax_traj.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        self.canvas_traj.draw()
        self.traj_surface = self._canvas_to_pygame(self.canvas_traj)
        
        # ── Control Plot ─────────────────────────────────────
        self.ax_ctrl.clear()
        if len(planner.control_history) > 1:
            ctrls = list(planner.control_history)
            x_vals = range(len(ctrls))
            
            speeds = [c['speed'] for c in ctrls]
            steers = [c['steer'] for c in ctrls]
            
            self.ax_ctrl.plot(x_vals, speeds, 'b-', linewidth=2, label='Velocity (m/s)')
            self.ax_ctrl.plot(x_vals, steers, 'r--', linewidth=2, label='Steering')
            
            self.ax_ctrl.set_xlabel('Frame')
            self.ax_ctrl.set_ylabel('Value')
            self.ax_ctrl.legend(loc='upper right', fontsize=7)
            self.ax_ctrl.set_title('Control Inputs', fontsize=9)
            self.ax_ctrl.grid(True, alpha=0.3)
        
        self.canvas_ctrl.draw()
        self.ctrl_surface = self._canvas_to_pygame(self.canvas_ctrl)
    
    def _canvas_to_pygame(self, canvas):
        """Convert matplotlib canvas to pygame surface."""
        canvas.draw()
        buf = canvas.buffer_rgba()
        arr = np.asarray(buf)
        arr = arr[:, :, :3]  # Remove alpha
        arr = np.flipud(arr)  # Flip for pygame
        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        return surface
    
    def render_subgoal_overlay(self, surface, planner, slot_size):
        """
        Render subgoal decision visualization as overlay.
        Shows candidate subgoals, best subgoal, rejected ones.
        """
        if not planner.subgoals:
            return
        
        w, h = slot_size
        cx, cy = w // 2, h // 2
        scale = min(w, h) / 20.0  # Scale to fit
        
        # Draw polar grid
        for r in [5, 10, 15]:
            pygame.draw.circle(surface, (50, 50, 50), (cx, cy), int(r * scale), 1)
        
        for angle in np.linspace(0, 2*math.pi, 8):
            x1 = cx + int(15 * scale * math.cos(angle))
            y1 = cy - int(15 * scale * math.sin(angle))
            x2 = cx + int(18 * scale * math.cos(angle))
            y2 = cy - int(18 * scale * math.sin(angle))
            pygame.draw.line(surface, (50, 50, 50), (x1, y1), (x2, y2), 1)
        
        # Draw subgoals
        for sg in planner.subgoals:
            angle = sg['alpha']
            dist = sg['distance']
            
            x = cx + int(dist * scale * math.cos(angle))
            y = cy - int(dist * scale * math.sin(angle))
            
            if sg == planner.selected_subgoal:
                color = self.C_SUBGOAL_BEST
                radius = 6
            elif not sg['safe']:
                color = self.C_SUBGOAL_REJECT
                radius = 3
            elif sg['variance'] > VARIANCE_THRESHOLD * 0.8:
                color = self.C_SUBGOAL_UNCERTAIN
                radius = 4
            else:
                color = self.C_SUBGOAL_CANDIDATE
                radius = 3
            
            pygame.draw.circle(surface, color, (x, y), radius)
            
            # Draw line from center to subgoal
            if sg == planner.selected_subgoal:
                pygame.draw.line(surface, color, (cx, cy), (x, y), 2)
        
        # Draw vehicle at center
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 4)
        
        # Label
        lbl = self.font.render("SUBGOALS", True, (200, 200, 200))
        surface.blit(lbl, (5, 5))
    
    def render(self, planner, vehicle, display_manager):
        """Render all visualization panels."""
        ds = display_manager.get_display_size()
        slot_w, slot_h = ds[0], ds[1]
        
        # ── Bottom Left: Cost Function Plot ──────────────────
        if self.cost_surface:
            cost_scaled = pygame.transform.scale(self.cost_surface, (slot_w, slot_h))
            offset = display_manager.get_display_offset([1, 0])
            display_manager.display.blit(cost_scaled, offset)
            pygame.draw.rect(display_manager.display, (60, 60, 80), 
                           (offset[0], offset[1], slot_w, slot_h), 1)
            lbl = self.bold_font.render("COST FUNCTION", True, (200, 200, 200))
            display_manager.display.blit(lbl, (offset[0] + 4, offset[1] + 4))
        
        # ── Bottom Center: Trajectory + Subgoals Map ─────────
        if self.traj_surface:
            traj_scaled = pygame.transform.scale(self.traj_surface, (slot_w, slot_h))
            offset = display_manager.get_display_offset([1, 1])
            display_manager.display.blit(traj_scaled, offset)
            pygame.draw.rect(display_manager.display, (60, 60, 80), 
                           (offset[0], offset[1], slot_w, slot_h), 1)
            lbl = self.bold_font.render("TRAJECTORY & SUBGOALS", True, (200, 200, 200))
            display_manager.display.blit(lbl, (offset[0] + 4, offset[1] + 4))
        
        # ── Bottom Right: Control Plot ───────────────────────
        if self.ctrl_surface:
            ctrl_scaled = pygame.transform.scale(self.ctrl_surface, (slot_w, slot_h))
            offset = display_manager.get_display_offset([1, 2])
            display_manager.display.blit(ctrl_scaled, offset)
            pygame.draw.rect(display_manager.display, (60, 60, 80), 
                           (offset[0], offset[1], slot_w, slot_h), 1)
            lbl = self.bold_font.render("CONTROL INPUTS", True, (200, 200, 200))
            display_manager.display.blit(lbl, (offset[0] + 4, offset[1] + 4))
        
        # ── Extra: Subgoal Overlay under LiDAR ───────────────
        # Create overlay surface for LiDAR slot [0, 1]
        overlay_surface = pygame.Surface((slot_w, slot_h), pygame.SRCALPHA)
        overlay_surface.fill((0, 0, 0, 0))  # Transparent
        self.render_subgoal_overlay(overlay_surface, planner, (slot_w, slot_h))
        
        offset = display_manager.get_display_offset([0, 1])
        display_manager.display.blit(overlay_surface, offset)


# ─────────────────────────────────────────────────────────────
# DISPLAY MANAGER (UPDATED FOR NEW LAYOUT)
# ─────────────────────────────────────────────────────────────

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA VSGP Mapless Navigation v5")
        self.grid_size   = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self._font       = pygame.font.SysFont("monospace", 13)
        self._hud_lines  = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [
            int(self.window_size[0] / self.grid_size[1]),
            int(self.window_size[1] / self.grid_size[0])
        ]

    def get_display_offset(self, grid_pos):
        ds = self.get_display_size()
        return [int(grid_pos[1] * ds[0]), int(grid_pos[0] * ds[1])]

    def add_sensor(self, s):
        self.sensor_list.append(s)

    def get_sensor_list(self):
        return self.sensor_list

    def set_hud(self, lines):
        self._hud_lines = lines

    def render(self):
        if self.display is None:
            return
        self.display.fill((10, 10, 20))
        for s in self.sensor_list:
            s.render()
        self._draw_labels()
        self._draw_hud()
        pygame.display.flip()

    def _draw_labels(self):
        ds = self.get_display_size()
        for s in self.sensor_list:
            off = self.get_display_offset(s.display_pos)
            lbl = self._font.render(s.label, True, (200, 200, 200))
            self.display.blit(lbl, (off[0] + 4, off[1] + 4))
            pygame.draw.rect(self.display, (60, 60, 80),
                             (off[0], off[1], ds[0], ds[1]), 1)

    def _draw_hud(self):
        x = 10
        y = self.window_size[1] - 20 * len(self._hud_lines) - 6
        for line in self._hud_lines:
            surf = self._font.render(line, True, (0, 255, 120))
            self.display.blit(surf, (x, y))
            y += 18

    def render_enabled(self):
        return self.display is not None

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()


# ─────────────────────────────────────────────────────────────
# SENSOR MANAGER (REUSED - UNCHANGED LOGIC)
# ─────────────────────────────────────────────────────────────

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform,
                 attached, sensor_options, display_pos, label=""):
        self.surface        = None
        self.world          = world
        self.display_man    = display_man
        self.display_pos    = display_pos
        self.label          = label
        self.sensor_options = sensor_options
        self.timer          = CustomTimer()
        self.time_proc      = 0.0
        self.ticks          = 0
        self.lidar_points = None
        self.depth_array  = None
        self._lidar_frame   = 0
        self._colour_bar    = None
        self.sensor = self._init_sensor(sensor_type, transform,
                                        attached, sensor_options)
        self.display_man.add_sensor(self)

    def _init_sensor(self, sensor_type, transform, attached, opts):
        bp_lib = self.world.get_blueprint_library()
        ds     = self.display_man.get_display_size()

        if sensor_type == "RGBCamera":
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items(): bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rgb)
            return actor
        elif sensor_type == "DepthCamera":
            bp = bp_lib.find("sensor.camera.depth")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items(): bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._depth_combined_callback)
            return actor
        elif sensor_type == "RainbowLiDAR":
            bp = bp_lib.find("sensor.lidar.ray_cast")
            bp.set_attribute("range", LIDAR_RANGE)
            bp.set_attribute("channels", LIDAR_CHANNELS)
            bp.set_attribute("points_per_second", LIDAR_PTS_PER_SEC)
            bp.set_attribute("rotation_frequency", LIDAR_ROT_FREQ)
            bp.set_attribute("dropoff_general_rate", "0.0")
            for k, v in opts.items(): bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rainbow_lidar)
            self._colour_bar = _precompute_colour_bar(ds[1])
            return actor
        return None

    def _save_rgb(self, image):
        t0 = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        arr  = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr  = np.reshape(arr, (image.height, image.width, 4))
        arr  = arr[:, :, :3][:, :, ::-1]
        ds   = self.display_man.get_display_size()
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        self.surface = pygame.transform.scale(surf, (ds[0], ds[1]))
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def _depth_combined_callback(self, image):
        t0 = self.timer.time()
        arr_raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr_raw = np.reshape(arr_raw, (image.height, image.width, 4))
        self.depth_array = arr_raw[:, :, 0].copy()
        image.convert(carla.ColorConverter.LogarithmicDepth)
        arr_vis = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr_vis = np.reshape(arr_vis, (image.height, image.width, 4))
        arr_vis = arr_vis[:, :, :3][:, :, ::-1]
        ds      = self.display_man.get_display_size()
        surf    = pygame.surfarray.make_surface(arr_vis.swapaxes(0, 1))
        self.surface = pygame.transform.scale(surf, (ds[0], ds[1]))
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def _save_rainbow_lidar(self, image):
        t0 = self.timer.time()
        self._lidar_frame += 1
        pts = np.frombuffer(image.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (-1, 4))
        xyz = pts[:, :3].copy()
        self.lidar_points = xyz
        if self._lidar_frame % LIDAR_FRAME_SKIP != 0:
            self.time_proc += self.timer.time() - t0
            self.ticks += 1
            return
        ds          = self.display_man.get_display_size()
        lidar_range = 2.0 * float(LIDAR_RANGE)
        xy = xyz[:, :2].copy()
        xy *= min(ds) / lidar_range
        xy += (0.5 * ds[0], 0.5 * ds[1])
        xy  = xy.astype(np.int32)
        z_norm = np.clip((xyz[:, 2] - RAINBOW_Z_MIN) / (RAINBOW_Z_MAX - RAINBOW_Z_MIN), 0.0, 1.0)
        hue    = ((1.0 - z_norm) * 0.75).astype(np.float32)
        rgb_u8 = _hsv_to_rgb_vectorised(hue)
        img   = np.zeros((ds[0], ds[1], 3), dtype=np.uint8)
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < ds[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < ds[1]))
        img[xy[valid, 0], xy[valid, 1]] = rgb_u8[valid]
        cx, cy = ds[0] // 2, ds[1] // 2
        r = 7
        img[cx - 1 : cx + 2, max(0, cy - r) : min(ds[1], cy + r)] = 255
        img[max(0, cx - r) : min(ds[0], cx + r), cy - 1 : cy + 2] = 255
        bar_w = self._colour_bar.shape[0]
        img[ds[0] - bar_w : ds[0], :] = self._colour_bar
        self.surface = pygame.surfarray.make_surface(img)
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


# ─────────────────────────────────────────────────────────────
# NAVIGATION CONTROLLER (VSGP-BASED)
# ─────────────────────────────────────────────────────────────

class NavigationController:
    def __init__(self):
        # Legacy safety systems (preserved)
        self.steer_ema       = 0.0
        self.steer_ema_alpha = STEER_EMA_ALPHA
        self.throttle_ema    = 0.0
        self.throttle_ema_alpha = THROTTLE_EMA_ALPHA
        self.pos_history   = deque(maxlen=STUCK_WINDOW_FRAMES)
        self.reverse_count = 0
        self.reverse_steer = 0.0
        self.spawn_z         = None
        self.emergency_count = 0
        
        # NEW: VSGP Navigation Systems
        self.vsgp = VSGPPerception()
        self.planner = SubgoalPlanner()
        
        self.diag = {
            "state": "INIT", "front": 0.0, "fl": 0.0, "fr": 0.0,
            "speed": 0.0, "steer": 0.0, "throttle": 0.0,
            "subgoals": 0, "best_cost": 0.0
        }
    
    def _filter_lidar_for_vsgp(self, pts):
        """Filter and downsample LiDAR for VSGP processing."""
        if pts is None:
            return np.empty((0, 3))
        
        # ROI Filter (forward hemisphere)
        mask = (pts[:, 0] > 0) & (pts[:, 0] < 20.0) & \
               (pts[:, 1] > -10.0) & (pts[:, 1] < 10.0) & \
               (pts[:, 2] > -2.0) & (pts[:, 2] < 3.0)
        filtered = pts[mask]
        
        # Downsample
        if len(filtered) > GP_MAX_POINTS * 2:
            indices = np.random.choice(len(filtered), GP_MAX_POINTS * 2, replace=False)
            filtered = filtered[indices]
        
        return filtered
    
    def _analyse_lidar_legacy(self, pts):
        """Legacy LiDAR analysis for safety fallback."""
        if pts is None or len(pts) == 0:
            return None
        mask = (pts[:, 2] > OBSTACLE_Z_MIN) & (pts[:, 2] < OBSTACLE_Z_MAX)
        obs  = pts[mask]
        cap  = float(LIDAR_RANGE)
        empty = dict(front=cap, fl=cap, fr=cap)
        if len(obs) == 0:
            return empty
        ang  = np.arctan2(obs[:, 1], obs[:, 0])
        dist = np.hypot(obs[:, 0], obs[:, 1])
        sw   = math.pi / 8
        def sector_min(lo, hi):
            m = (ang >= lo) & (ang < hi)
            return float(np.min(dist[m])) if m.any() else cap
        return {
            "front": sector_min(-sw, sw),
            "fl": sector_min(sw, 3*sw),
            "fr": sector_min(-3*sw, -sw),
        }
    
    def _check_stuck(self):
        if len(self.pos_history) < STUCK_WINDOW_FRAMES:
            return False
        old = self.pos_history[0]
        new = self.pos_history[-1]
        dist = math.hypot(new[0]-old[0], new[1]-old[1])
        return dist < STUCK_MIN_TRAVEL_M and self.diag["speed"] < 0.5
    
    def compute_control(self, lidar_pts, depth_raw, vehicle):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.pos_history.append((loc.x, loc.y))
        self.diag["speed"] = speed
        
        # Record trajectory
        self.planner.add_trajectory_point(loc)
        
        # ── FALL recovery ────────────────────────────────────
        if self.spawn_z is not None and loc.z < self.spawn_z - 8.0:
            self.emergency_count = 35
        if self.emergency_count > 0:
            self.emergency_count -= 1
            self.diag["state"] = "FALL-RECOV"
            return carla.VehicleControl(throttle=0.6, reverse=_R_REV)
        
        # ── STUCK check ──────────────────────────────────────
        if self._check_stuck():
            self.reverse_count = REVERSE_FRAMES
            self.pos_history.clear()
            self.diag["state"] = "STUCK→REV"
            return carla.VehicleControl(throttle=0.5, steer=-0.5, reverse=_R_REV)
        
        if self.reverse_count > 0:
            self.reverse_count -= 1
            self.diag["state"] = "REVERSING"
            return carla.VehicleControl(throttle=0.5, steer=-0.5, reverse=_R_REV)
        
        # ── VSGP NAVIGATION PIPELINE ─────────────────────────
        nav_pts = self._filter_lidar_for_vsgp(lidar_pts)
        
        # Step 1: VSGP Perception
        free_segments, var_map, mean_map = self.vsgp.update(nav_pts)
        
        # Step 2: Generate Subgoals
        self.planner.generate_subgoals(free_segments, vehicle.get_transform(), self.vsgp)
        
        # Step 3: Evaluate Costs
        self.planner.evaluate_costs(vehicle.get_transform())
        
        # Step 4: Select Best Subgoal
        best_sg = self.planner.select_best_subgoal()
        
        # Update diagnostics
        self.diag["subgoals"] = len(self.planner.subgoals)
        self.diag["best_cost"] = best_sg['cost'] if best_sg else float('inf')
        
        # ── CONTROL GENERATION ───────────────────────────────
        throttle = THROTTLE_CRUISE
        target_steer = 0.0
        brake = 0.0
        
        if best_sg:
            self.diag["state"] = "VSGP-NAV"
            
            # Convert subgoal to steering
            # alpha is relative angle, convert to steer (-1 to 1)
            target_steer = np.clip(best_sg['alpha'] / (math.pi / 3), -1, 1)
            
            # Throttle based on distance and safety
            if best_sg['distance'] < 5.0:
                throttle = THROTTLE_CAUTION
            elif best_sg['variance'] > VARIANCE_THRESHOLD * 0.5:
                throttle = THROTTLE_CAUTION
            
            # Legacy safety override
            analysis = self._analyse_lidar_legacy(lidar_pts)
            if analysis:
                front = analysis["front"]
                if front < FRONT_EMRG_DIST:
                    self.diag["state"] = "EMRG-STOP"
                    brake = 0.6
                    throttle = 0.0
                    target_steer = 0.0
                elif front < FRONT_STOP_DIST:
                    self.diag["state"] = "SLOW"
                    throttle = 0.2
        else:
            self.diag["state"] = "NO-PATH"
            throttle = 0.0
            brake = 0.3
            target_steer = 0.0
        
        # Low speed boost
        if speed < 0.8 and brake == 0.0:
            throttle = max(throttle, 0.4)
        
        # EMA smoothing
        self.steer_ema = (self.steer_ema_alpha * target_steer + 
                         (1 - self.steer_ema_alpha) * self.steer_ema)
        self.throttle_ema = (self.throttle_ema_alpha * throttle + 
                            (1 - self.throttle_ema_alpha) * self.throttle_ema)
        
        self.diag["steer"] = self.steer_ema
        self.diag["throttle"] = self.throttle_ema
        
        # Record control for plotting
        self.planner.add_control_point(speed, self.steer_ema)
        
        return carla.VehicleControl(
            throttle=float(np.clip(self.throttle_ema, 0, 1)),
            steer=float(np.clip(self.steer_ema, -1, 1)),
            brake=float(np.clip(brake, 0, 1)),
            reverse=_R_FWD
        )


# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────

def run_simulation(args, client):
    display_manager = None
    actors          = []
    nav             = NavigationController()
    visualizer      = None

    try:
        world             = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            tm       = client.get_trafficmanager(8000)
            settings = world.get_settings()
            tm.set_synchronous_mode(True)
            settings.synchronous_mode       = True
            settings.fixed_delta_seconds    = 0.05
            settings.max_substep_delta_time = 0.01
            settings.max_substeps           = 10
            world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_bp   = bp_lib.find("vehicle.tesla.cybertruck")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle = None
        for sp in spawn_points:
            sp.location.z += 5.0
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"✅  Spawned at ({sp.location.x:.1f}, {sp.location.y:.1f}, {sp.location.z:.1f})")
                break

        if vehicle is None:
            raise RuntimeError("❌  No valid spawn point found")
        actors.append(vehicle)

        print(f"⏳  Settling {SPAWN_SETTLE_S}s …")
        t0 = time.time()
        while time.time() - t0 < SPAWN_SETTLE_S:
            world.tick() if args.sync else world.wait_for_tick()

        nav.spawn_z = vehicle.get_location().z
        print(f"📍  Settled Z = {nav.spawn_z:.2f}m")
        print(f"ℹ️   INVERT_DRIVE={INVERT_DRIVE}  (_R_FWD={_R_FWD}, _R_REV={_R_REV})")
        print(f"🧠   VSGP Navigation: GP={SKLEARN_AVAILABLE}, Plots={MATPLOTLIB_AVAILABLE}")

        # Display Manager - 2×3 grid
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])
        
        # Visualization Manager
        if MATPLOTLIB_AVAILABLE:
            visualizer = VisualizationManager(display_manager)
        
        # ── TOP ROW: Front, LiDAR, Right (PRESERVED) ─────────
        front_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[0, 0], label="FRONT CAM")
        actors.append(front_cam.sensor)

        rainbow_lidar = SensorManager(world, display_manager, "RainbowLiDAR",
            carla.Transform(carla.Location(z=2.5)), vehicle, {},
            display_pos=[0, 1], label="LiDAR + SUBGOALS")
        actors.append(rainbow_lidar.sensor)

        right_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=0.8, z=1.5), carla.Rotation(yaw=90)),
            vehicle, {}, display_pos=[0, 2], label="RIGHT CAM")
        actors.append(right_cam.sensor)

        # ── BOTTOM ROW: Cost, Trajectory, Control (NEW) ──────
        # Left camera (kept for completeness, slot [1,0] now shows Cost Plot)
        left_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=-0.8, z=1.5), carla.Rotation(yaw=-90)),
            vehicle, {}, display_pos=[1, 0], label="COST PLOT")
        actors.append(left_cam.sensor)

        # Depth camera (slot [1,1] now shows Trajectory Map)
        depth_cam = SensorManager(world, display_manager, "DepthCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-5)),
            vehicle, {}, display_pos=[1, 1], label="TRAJECTORY MAP")
        actors.append(depth_cam.sensor)

        # Rear camera (slot [1,2] now shows Control Plot)
        rear_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.5, z=2.0), carla.Rotation(yaw=180, pitch=-10)),
            vehicle, {}, display_pos=[1, 2], label="CONTROL PLOT")
        actors.append(rear_cam.sensor)

        # ── Main loop ─────────────────────────────────────────
        clock        = pygame.time.Clock()
        running      = True
        frame        = 0
        manual_mode  = False
        manual_steer = 0.0

        print("🚗  Running — ESC/Q: quit   R: reset   P: toggle manual/auto")
        print("         Manual — W: forward   S: brake   A/D: steer")

        while running:
            clock.tick(20)
            frame += 1

            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_r:
                        nav.pos_history.clear()
                        nav.planner.trajectory_history.clear()
                        nav.planner.cost_history.clear()
                        nav.planner.control_history.clear()
                        nav.reverse_count = 0
                        nav.emergency_count = 0
                        print("⟳  Nav state reset")
                    elif event.key == K_p:
                        manual_mode = not manual_mode
                        manual_steer = 0.0
                        tag = "🕹  MANUAL" if manual_mode else "🤖  VSGP-AUTO"
                        print(f"[P] Switched to {tag} mode")

            if manual_mode:
                keys = pygame.key.get_pressed()
                m_throttle, m_brake, m_reverse, m_steer_tgt = 0.0, 0.08, _R_FWD, 0.0
                if keys[K_w]:
                    m_throttle, m_brake, m_reverse = 0.65, 0.0, _R_FWD
                if keys[K_s]:
                    vel = vehicle.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    if spd > 0.5:
                        m_brake, m_throttle, m_reverse = 0.8, 0.0, _R_FWD
                    else:
                        m_throttle, m_brake, m_reverse = 0.55, 0.0, _R_REV
                if keys[K_a]:
                    m_steer_tgt = -0.6
                if keys[K_d]:
                    m_steer_tgt =  0.6
                manual_steer = 0.45 * m_steer_tgt + 0.55 * manual_steer
                control = carla.VehicleControl(
                    throttle=float(np.clip(m_throttle, 0, 1)),
                    steer=float(np.clip(manual_steer, -1, 1)),
                    brake=float(np.clip(m_brake, 0, 1)),
                    reverse=m_reverse
                )
                vehicle.apply_control(control)
                mode_tag = "🕹  MANUAL"
            else:
                lidar_pts = rainbow_lidar.lidar_points
                depth_raw = depth_cam.depth_array
                control = nav.compute_control(lidar_pts, depth_raw, vehicle)
                vehicle.apply_control(control)
                mode_tag = "🤖  VSGP-AUTO"
                
                # Update visualizations - FIXED: Added try-except for safety
                if visualizer and MATPLOTLIB_AVAILABLE:
                    try:
                        visualizer.update_plots(nav.planner, vehicle)
                        visualizer.render(nav.planner, vehicle, display_manager)
                    except Exception as e:
                        print(f"⚠  Visualization error (frame {frame}): {e}")
                        # Continue without visualization

            # ── HUD ───────────────────────────────────────────
            d = nav.diag
            display_manager.set_hud([
                f"MODE  : {mode_tag}          FRAME: {frame}",
                f"STATE : {d['state']:<16}  VSGP: {SKLEARN_AVAILABLE}",
                f"SPEED : {d['speed']*3.6:5.1f} km/h   STEER: {d['steer']:+.3f}",
                f"THR   : {d['throttle']:.2f}   SUBGOALS: {d['subgoals']:3d}",
                f"COST  : {d['best_cost']:.3f}   VAR_TH: {VARIANCE_THRESHOLD}",
                f"STUCK : {'YES' if nav._check_stuck() else 'no':<4}  EMRG: {nav.emergency_count:3d}",
            ])

            try:
                display_manager.render()
            except Exception as e:
                print(f"⚠  Render skip (frame {frame}): {e}")

    finally:
        print("\n🧹  Cleaning up …")
        if display_manager:
            display_manager.destroy()
        client.apply_batch(
            [carla.command.DestroyActor(a) for a in actors if a is not None]
        )
        world.apply_settings(original_settings)
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')
        pygame.quit()
        print("✅  Done.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARLA VSGP Mapless Navigation v5")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--sync",  action="store_true",  default=True)
    parser.add_argument("--async", dest="sync", action="store_false")
    parser.add_argument("--res",   default="1280x720", metavar="WxH")
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        run_simulation(args, client)
    except KeyboardInterrupt:
        print("\n⚠  Interrupted by user.")


if __name__ == "__main__":
    main()
