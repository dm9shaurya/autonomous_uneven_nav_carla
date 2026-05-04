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
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not found. GP Terrain Estimation disabled. Falling back to simple variance.")

try:
    import pygame
    from pygame.locals import (K_ESCAPE, K_q, K_r,
                               K_w, K_s, K_a, K_d, K_p, K_m)
except ImportError:
    raise RuntimeError("pygame not installed — run: pip install pygame")


# ─────────────────────────────────────────────────────────────
# CONSTANTS & WEIGHTS
# ─────────────────────────────────────────────────────────────

# ── Legacy Constants (Preserved) ────────────────────────────
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

# ── NEW: Advanced Nav Constants ─────────────────────────────
# LiDAR ROI for processing
NAV_X_MIN, NAV_X_MAX = 0.0, 20.0
NAV_Y_MIN, NAV_Y_MAX = -10.0, 10.0
NAV_Z_MIN, NAV_Z_MAX = -2.0, 3.0
NAV_DOWNSAMPLE_TARGET = 800  # Points for GP/Grid

# Occupancy Grid
GRID_SIZE_M = 40.0       # 40m x 40m area
GRID_RES_M  = 0.5        # 0.5m per cell
GRID_DIM    = int(GRID_SIZE_M / GRID_RES_M)

# Cost Weights (Tunable)
W_SLOPE      = 2.0
W_UNCERTAINTY= 1.5
W_OCCUPANCY  = 10.0
W_HEADING    = 1.0
W_PROGRESS   = 0.5

# Planning
PLAN_HORIZON_M = 15.0
PLAN_STEP_M    = 1.0
CANDIDATE_ANGLES = np.linspace(-30, 30, 13) * (math.pi / 180.0)  # -30° to +30°

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

class CustomTimer:
    def __init__(self):
        self.timer = time.perf_counter
    def time(self):
        return self.timer()

def _hsv_to_rgb_vectorised(hue_arr):
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
    bar = np.zeros((bar_w, height, 3), dtype=np.uint8)
    for py in range(height):
        norm = 1.0 - py / height
        hue  = (1.0 - norm) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bar[:, py] = (int(r * 255), int(g * 255), int(b * 255))
    return bar

# ─────────────────────────────────────────────────────────────
# NEW MODULE: OCCUPANCY GRID
# ─────────────────────────────────────────────────────────────

class OccupancyGrid:
    def __init__(self, size_m, resolution_m):
        self.dim = int(size_m / resolution_m)
        self.res = resolution_m
        self.offset = size_m / 2.0
        # 0: Unknown, 1: Free, 2: Occupied
        self.grid = np.zeros((self.dim, self.dim), dtype=np.uint8)
        self.center_idx = self.dim // 2

    def clear(self):
        self.grid[:] = 0  # Reset to Unknown

    def world_to_grid(self, x, y):
        # Transform world relative coords to grid indices
        # x is forward, y is left
        # Grid: row=0 is top (negative y), col=0 is left (negative x)
        # But we want vehicle centered: x=0,y=0 -> center_idx
        gx = int((x + self.offset) / self.res)
        gy = int((y + self.offset) / self.res)
        return gx, gy

    def update(self, points):
        """
        points: (N, 3) numpy array (x, y, z) relative to vehicle
        """
        self.clear()
        
        # 1. Mark Occupied
        for x, y, z in points:
            if not (NAV_X_MIN <= x <= NAV_X_MAX and NAV_Y_MIN <= y <= NAV_Y_MAX):
                continue
            gx, gy = self.world_to_grid(x, y)
            if 0 <= gx < self.dim and 0 <= gy < self.dim:
                self.grid[gx, gy] = 2  # Occupied

        # 2. Raycast Free Space (Simplified Bresenham-like)
        # For every occupied point, mark line from center to point as Free
        # This is computationally heavy in Python, so we do a simplified version:
        # Just mark a cone in front as free if no points are found? 
        # Better: Iterate points, draw line.
        cx, cy = self.center_idx, self.center_idx
        
        # Optimization: Only process a subset of points for raycasting to save FPS
        step = max(1, len(points) // 200)
        for i in range(0, len(points), step):
            x, y, _ = points[i]
            if x < 0: continue # Don't raycast behind
            
            gx, gy = self.world_to_grid(x, y)
            if not (0 <= gx < self.dim and 0 <= gy < self.dim):
                continue
            
            # Simple line drawing
            dx, dy = gx - cx, gy - cy
            steps = max(abs(dx), abs(dy))
            if steps == 0: continue
            
            for k in range(1, steps):
                ix = int(cx + dx * k / steps)
                iy = int(cy + dy * k / steps)
                if 0 <= ix < self.dim and 0 <= iy < self.dim:
                    if self.grid[ix, iy] != 2: # Don't overwrite occupied
                        self.grid[ix, iy] = 1  # Free

    def is_occupied(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.dim and 0 <= gy < self.dim:
            return self.grid[gx, gy] == 2
        return True # Out of bounds = occupied/safe assumption

# ─────────────────────────────────────────────────────────────
# NEW MODULE: TERRAIN ESTIMATOR (GP / VSGP Style)
# ─────────────────────────────────────────────────────────────

class TerrainEstimator:
    def __init__(self):
        self.gp = None
        self.use_gp = SKLEARN_AVAILABLE
        if self.use_gp:
            # Lightweight kernel for real-time
            kernel = C(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        
        self.last_mean = 0.0
        self.last_var = 0.0

    def update(self, points):
        """
        points: (N, 3) relative to vehicle.
        Returns: mean_slope, uncertainty
        """
        if len(points) < 10:
            return 0.0, 1.0

        # Filter for ground points (low z) to estimate terrain
        # Heuristic: lowest 20% of points in each sector
        # Simplified: Just take points with z < 0.5 (assuming vehicle z=0)
        ground_mask = points[:, 2] < 0.5
        ground_pts = points[ground_mask]

        if len(ground_pts) < 5:
            return 0.0, 1.0

        # Downsample for GP speed
        if len(ground_pts) > 100:
            indices = np.random.choice(len(ground_pts), 100, replace=False)
            ground_pts = ground_pts[indices]

        X = ground_pts[:, :2] # x, y
        y = ground_pts[:, 2]  # z

        if self.use_gp:
            try:
                self.gp.fit(X, y)
                # Predict at a look-ahead point (e.g., 5m forward)
                X_test = np.array([[5.0, 0.0]])
                mean, std = self.gp.predict(X_test, return_std=True)
                self.last_mean = mean[0]
                self.last_var = std[0]
                
                # Estimate slope from GP gradient (approx)
                # Compare mean at 5m vs 0m
                mean_0, _ = self.gp.predict(np.array([[0.0, 0.0]]), return_std=True)
                slope = abs(self.last_mean - mean_0[0]) / 5.0
                return slope, self.last_var
            except Exception:
                return 0.0, 1.0
        else:
            # Fallback: Simple Variance of Z as Uncertainty/Roughness
            self.last_var = np.var(y)
            # Slope approx: max_z - min_z over range
            slope = (np.max(y) - np.min(y)) / 10.0
            return slope, self.last_var

# ─────────────────────────────────────────────────────────────
# NEW MODULE: TELEMETRY VISUALIZER (Pygame Based)
# ─────────────────────────────────────────────────────────────

class TelemetryVisualizer:
    def __init__(self, display_manager):
        self.dm = display_manager
        self.font = pygame.font.SysFont("monospace", 12)
        
        # History for plots
        self.cost_history = deque(maxlen=100)
        self.ctrl_history = deque(maxlen=100)
        self.traj_history = deque(maxlen=200) # Actual
        self.plan_history = deque(maxlen=10)  # Predicted
        
        # Colors
        self.C_FREE   = (0, 255, 0)
        self.C_OCC    = (255, 0, 0)
        self.C_UNK    = (100, 100, 100)
        self.C_TRAJ   = (0, 200, 255)
        self.C_PLAN   = (255, 255, 0)

    def update(self, vehicle, cost_breakdown, planned_angle, occupancy_grid):
        loc = vehicle.get_location()
        self.traj_history.append((loc.x, loc.y))
        self.plan_history.append(planned_angle)
        
        # Store cost components
        self.cost_history.append(cost_breakdown)
        
        vel = vehicle.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.ctrl_history.append(speed)

    def render(self, surface, occupancy_grid, vehicle, planned_angle):
        """
        Draws the Local Map, Cost Bars, and Trajectory on a dedicated surface
        or overlays on existing display. Here we draw on a passed surface.
        """
        w, h = surface.get_width(), surface.get_height()
        surface.fill((20, 20, 30))

        # 1. Draw Occupancy Grid (Top-Left Quadrant)
        grid_size = 150
        offset_x, offset_y = 10, 10
        cell_px = grid_size / occupancy_grid.dim
        
        # Center of grid on surface
        cx, cy = offset_x + grid_size/2, offset_y + grid_size/2
        
        # Draw Grid
        for r in range(occupancy_grid.dim):
            for c in range(occupancy_grid.dim):
                val = occupancy_grid.grid[r, c]
                if val == 0: color = self.C_UNK
                elif val == 1: color = (0, 100, 0) # Free (dark green)
                else: color = self.C_OCC
                
                # Optimization: Only draw occupied or near center to save fill calls
                if val == 2 or (abs(r - occupancy_grid.center_idx) < 10 and abs(c - occupancy_grid.center_idx) < 10):
                    rect = pygame.Rect(
                        offset_x + c * cell_px,
                        offset_y + (occupancy_grid.dim - 1 - r) * cell_px, # Flip Y
                        cell_px, cell_px
                    )
                    pygame.draw.rect(surface, color, rect)
        
        # Draw Vehicle Arrow
        pygame.draw.circle(surface, (255, 255, 255), (int(cx), int(cy)), 5)
        # Heading line
        end_x = cx + math.cos(0) * 20 # Assume 0 heading relative to grid
        end_y = cy - math.sin(0) * 20
        pygame.draw.line(surface, (255, 255, 0), (cx, cy), (end_x, end_y), 2)

        # Label
        lbl = self.font.render("LOCAL MAP (Occ)", True, (200, 200, 200))
        surface.blit(lbl, (offset_x, offset_y - 15))

        # 2. Draw Cost Breakdown (Bottom-Left)
        if len(self.cost_history) > 0:
            last_cost = self.cost_history[-1]
            y_start = h - 100
            x_start = 10
            bar_w = 20
            components = ['Slope', 'Unc', 'Occ', 'Head', 'Total']
            values = [
                last_cost.get('slope', 0), 
                last_cost.get('unc', 0), 
                last_cost.get('occ', 0), 
                last_cost.get('head', 0),
                last_cost.get('total', 0)
            ]
            colors = [(200, 100, 0), (100, 100, 200), (200, 0, 0), (0, 200, 0), (255, 255, 255)]
            
            for i, (name, val, col) in enumerate(zip(components, values, colors)):
                # Normalize for display (assume max 10)
                h_bar = min(50, val * 5)
                rect = pygame.Rect(x_start + i * (bar_w + 5), y_start, bar_w, h_bar)
                pygame.draw.rect(surface, col, rect)
                lbl = self.font.render(name[:3], True, col)
                surface.blit(lbl, (rect.x, rect.y - 12))
            
            lbl = self.font.render("COST METRICS", True, (200, 200, 200))
            surface.blit(lbl, (x_start, y_start - 15))

        # 3. Draw Trajectory (Right Side Overlay)
        # Convert world coords to local screen coords relative to last point
        if len(self.traj_history) > 1:
            # Simple relative plot
            base_x, base_y = self.traj_history[-1]
            for i in range(1, len(self.traj_history)):
                p1 = self.traj_history[i-1]
                p2 = self.traj_history[i]
                # Scale and offset to fit right side
                sx1 = w - 150 + (p1[0] - base_x) * 10
                sy1 = 100 + (p1[1] - base_y) * 10
                sx2 = w - 150 + (p2[0] - base_x) * 10
                sy2 = 100 + (p2[1] - base_y) * 10
                pygame.draw.line(surface, self.C_TRAJ, (sx1, sy1), (sx2, sy2), 2)
            
            lbl = self.font.render("TRAJECTORY", True, self.C_TRAJ)
            surface.blit(lbl, (w - 140, 85))

        # 4. Planned Direction Indicator
        # Draw an arc representing the chosen angle
        pygame.draw.arc(surface, (0, 255, 255), 
                        (w//2 - 50, h//2 - 50, 100, 100), 
                        math.pi/2 - 0.2, math.pi/2 + 0.2, 2)


# ─────────────────────────────────────────────────────────────
# DISPLAY MANAGER (Modified to support Telemetry)
# ─────────────────────────────────────────────────────────────

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Advanced Nav v4")
        self.grid_size   = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self._font       = pygame.font.SysFont("monospace", 13)
        self._hud_lines  = []
        self.telemetry_surface = None

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
        
        # Render Telemetry Overlay if available (replaces Depth Cam slot [1,1])
        if self.telemetry_surface:
            off = self.get_display_offset([1, 1])
            ds = self.get_display_size()
            # Scale telemetry surface to fit slot
            tel_scaled = pygame.transform.scale(self.telemetry_surface, (ds[0], ds[1]))
            self.display.blit(tel_scaled, off)
            # Border
            pygame.draw.rect(self.display, (60, 60, 80), (off[0], off[1], ds[0], ds[1]), 1)
            lbl = self._font.render("NAV TELEMETRY", True, (200, 200, 200))
            self.display.blit(lbl, (off[0] + 4, off[1] + 4))

        self._draw_labels()
        self._draw_hud()
        pygame.display.flip()

    def _draw_labels(self):
        ds = self.get_display_size()
        for s in self.sensor_list:
            # Skip drawing label for slot [1,1] as we overlay telemetry
            if s.display_pos == [1, 1]:
                continue
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
# SENSOR MANAGER (Unchanged Logic, Extended Data)
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
# NAVIGATION CONTROLLER (ADVANCED)
# ─────────────────────────────────────────────────────────────

class NavigationController:
    def __init__(self):
        self.steer_ema       = 0.0
        self.steer_ema_alpha = 0.35
        self.pos_history   = deque(maxlen=STUCK_WINDOW_FRAMES)
        self.reverse_count = 0
        self.reverse_steer = 0.0
        self.steer_sign_hist = deque(maxlen=OSC_WINDOW)
        self.forced_steer    = 0.0
        self.forced_count    = 0
        self.commit_steer = 0.0
        self.commit_count = 0
        self.spawn_z         = None
        self.emergency_count = 0
        
        # New Advanced Modules
        self.occ_grid = OccupancyGrid(GRID_SIZE_M, GRID_RES_M)
        self.terrain  = TerrainEstimator()
        self.diag = {
            "state": "INIT", "front": 0.0, "fl": 0.0, "fr": 0.0,
            "speed": 0.0, "steer": 0.0, "cost_total": 0.0
        }
        self.last_plan_angle = 0.0

    def _analyse_lidar(self, pts):
        if pts is None or len(pts) == 0: return None
        mask = (pts[:, 2] > OBSTACLE_Z_MIN) & (pts[:, 2] < OBSTACLE_Z_MAX)
        obs  = pts[mask]
        cap  = float(LIDAR_RANGE)
        empty = dict(front=cap, fl=cap, fr=cap, left=cap, right=cap,
                     front_slope_blocked=False, fl_slope_blocked=False, fr_slope_blocked=False)
        if len(obs) == 0: return empty
        ang  = np.arctan2(obs[:, 1], obs[:, 0])
        dist = np.hypot(obs[:, 0], obs[:, 1])
        sw   = math.pi / 8
        def sector_min(lo, hi):
            m = (ang >= lo) & (ang < hi)
            return float(np.min(dist[m])) if m.any() else cap
        def slope_blocked(lo, hi):
            m = (ang >= lo) & (ang < hi) & (obs[:, 0] > 1.0) & (obs[:, 0] < 8.0)
            return bool(m.any() and np.max(obs[m, 2]) > SLOPE_BLOCK_Z)
        return {
            "front": sector_min(-sw, sw), "fl": sector_min(sw, 3*sw), "fr": sector_min(-3*sw, -sw),
            "left": sector_min(3*sw, 5*sw), "right": sector_min(-5*sw, -3*sw),
            "front_slope_blocked": slope_blocked(-sw, sw),
            "fl_slope_blocked": slope_blocked(sw, 3*sw),
            "fr_slope_blocked": slope_blocked(-3*sw, -sw),
        }

    def _analyse_depth(self, depth_arr):
        if depth_arr is None: return None
        h, w = depth_arr.shape
        roi = depth_arr[h//4:h*3//4, w//4:w*3//4].astype(np.float32)
        return (float(roi.mean()) / 255.0) * 100.0

    def _check_stuck(self):
        if len(self.pos_history) < STUCK_WINDOW_FRAMES: return False
        old = self.pos_history[0]
        new = self.pos_history[-1]
        dist = math.hypot(new[0]-old[0], new[1]-old[1])
        return dist < STUCK_MIN_TRAVEL_M and self.diag["speed"] < 0.5

    def _filter_lidar_for_nav(self, pts):
        if pts is None: return np.empty((0, 3))
        # ROI Filter
        mask = (pts[:, 0] >= NAV_X_MIN) & (pts[:, 0] <= NAV_X_MAX) & \
               (pts[:, 1] >= NAV_Y_MIN) & (pts[:, 1] <= NAV_Y_MAX) & \
               (pts[:, 2] >= NAV_Z_MIN) & (pts[:, 2] <= NAV_Z_MAX)
        filtered = pts[mask]
        # Downsample
        if len(filtered) > NAV_DOWNSAMPLE_TARGET:
            indices = np.random.choice(len(filtered), NAV_DOWNSAMPLE_TARGET, replace=False)
            filtered = filtered[indices]
        return filtered

    def _compute_plan(self, nav_pts, vehicle_transform):
        """
        Core Advanced Planning Logic
        """
        # 1. Update Occupancy Grid
        self.occ_grid.update(nav_pts)
        
        # 2. Update Terrain Model
        slope, uncertainty = self.terrain.update(nav_pts)
        
        # 3. Evaluate Candidates
        best_angle = 0.0
        min_cost = float('inf')
        cost_breakdown = {'slope':0, 'unc':0, 'occ':0, 'head':0, 'total':0}
        
        current_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        for angle in CANDIDATE_ANGLES:
            # Calculate cost components
            c_slope = slope * W_SLOPE
            c_unc   = uncertainty * W_UNCERTAINTY
            
            # Check Occupancy along ray
            c_occ = 0.0
            for d in np.arange(PLAN_STEP_M, PLAN_HORIZON_M, PLAN_STEP_M):
                lx = d * math.cos(angle)
                ly = d * math.sin(angle)
                if self.occ_grid.is_occupied(lx, ly):
                    c_occ += 2.0 # Penalty per cell
            c_occ *= W_OCCUPANCY
            
            # Heading Alignment (prefer straight)
            c_head = abs(angle) * W_HEADING
            
            # Progress (prefer forward)
            c_prog = -d * W_PROGRESS # Negative cost for distance
            
            total = c_slope + c_unc + c_occ + c_head + c_prog
            
            if total < min_cost:
                min_cost = total
                best_angle = angle
                cost_breakdown = {
                    'slope': c_slope, 'unc': c_unc, 'occ': c_occ, 
                    'head': c_head, 'total': total
                }
        
        self.last_plan_angle = best_angle
        return best_angle, cost_breakdown

    def compute_control(self, lidar_pts, depth_raw, vehicle):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.pos_history.append((loc.x, loc.y))
        self.diag["speed"] = speed

        # ── FALL recovery ────────────────────────────────────────────
        if self.spawn_z is not None and loc.z < self.spawn_z - 8.0:
            self.emergency_count = 35
        if self.emergency_count > 0:
            self.emergency_count -= 1
            self.diag["state"] = "FALL-RECOV"
            return carla.VehicleControl(throttle=0.6, reverse=_R_REV)

        # ── ACTIVE REVERSE ──────────────────────────────────────────
        if self.reverse_count > 0:
            self.reverse_count -= 1
            self.diag["state"] = "REVERSING"
            return carla.VehicleControl(throttle=0.5, steer=-self.reverse_steer * 0.5, reverse=_R_REV)

        # ── STUCK check ──────────────────────────────────────────────
        if self._check_stuck():
            a = self._analyse_lidar(lidar_pts)
            self.reverse_steer = 0.6 if (a and a["fl"] < a["fr"]) else -0.6
            self.reverse_count = REVERSE_FRAMES
            self.pos_history.clear()
            self.diag["state"] = "STUCK→REV"
            return carla.VehicleControl(throttle=0.5, steer=self.reverse_steer, reverse=_R_REV)

        # ── ADVANCED PLANNING PIPELINE ───────────────────────────────
        nav_pts = self._filter_lidar_for_nav(lidar_pts)
        planned_angle, cost_breakdown = self._compute_plan(nav_pts, vehicle.get_transform())
        
        # Update Diag for HUD
        self.diag.update(cost_breakdown)
        self.diag["state"] = "PLANNING"

        # ── NORMAL NAV (Fallback / Hybrid) ───────────────────────────
        # We use the planned angle to set target steer, but keep legacy obstacle checks for safety
        analysis = self._analyse_lidar(lidar_pts)
        depth_m  = self._analyse_depth(depth_raw)
        
        throttle     = THROTTLE_CRUISE
        brake        = 0.0
        target_steer = 0.0

        # Convert planned angle to steer (-30 deg ~ -1.0 steer)
        # CARLA steer is approx -1 to 1, corresponding to ~ -60 to 60 deg usually
        target_steer = np.clip(planned_angle / (math.pi/3), -1, 1)

        if analysis:
            front = analysis["front"]
            if depth_m: front = min(front, depth_m * 0.65)
            
            # Safety Overrides
            if front < FRONT_EMRG_DIST:
                self.diag["state"] = "EMRG-STOP"
                brake = 0.6
                throttle = 0.0
                target_steer = 0.0 # Stop straight
            elif front < FRONT_STOP_DIST:
                self.diag["state"] = "SLOW-TURN"
                throttle = 0.2
            elif front < FRONT_WARN_DIST:
                self.diag["state"] = "CAUTION"
                throttle = THROTTLE_CAUTION
        
        # Low speed boost
        if speed < 0.8 and brake == 0.0:
            throttle = max(throttle, 0.4)
            target_steer *= 0.5

        self.steer_ema = (self.steer_ema_alpha * target_steer + (1 - self.steer_ema_alpha) * self.steer_ema)
        self.diag["steer"] = self.steer_ema

        return carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
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
    telemetry       = None

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

        # Display Manager
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])
        
        # Initialize Telemetry Visualizer
        telemetry = TelemetryVisualizer(display_manager)
        # Create a surface for telemetry to draw on (size of one grid slot)
        ds = display_manager.get_display_size()
        telemetry_surface = pygame.Surface((ds[0], ds[1]))
        display_manager.telemetry_surface = telemetry_surface

        # Sensors
        front_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[0, 0], label="FRONT CAM")
        actors.append(front_cam.sensor)

        rainbow_lidar = SensorManager(world, display_manager, "RainbowLiDAR",
            carla.Transform(carla.Location(z=2.5)), vehicle, {},
            display_pos=[0, 1], label="RAINBOW LiDAR")
        actors.append(rainbow_lidar.sensor)

        right_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=0.8, z=1.5), carla.Rotation(yaw=90)),
            vehicle, {}, display_pos=[0, 2], label="RIGHT CAM")
        actors.append(right_cam.sensor)

        left_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=-0.8, z=1.5), carla.Rotation(yaw=-90)),
            vehicle, {}, display_pos=[1, 0], label="LEFT CAM")
        actors.append(left_cam.sensor)

        # Depth Camera (Slot [1,1] is now overwritten by Telemetry in render, 
        # but we keep sensor for data)
        depth_cam = SensorManager(world, display_manager, "DepthCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-5)),
            vehicle, {}, display_pos=[1, 1], label="DEPTH (LOG)")
        actors.append(depth_cam.sensor)

        rear_cam = SensorManager(world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.5, z=2.0), carla.Rotation(yaw=180, pitch=-10)),
            vehicle, {}, display_pos=[1, 2], label="REAR CAM")
        actors.append(rear_cam.sensor)

        # ── Main loop ─────────────────────────────────────────────────
        clock        = pygame.time.Clock()
        running      = True
        frame        = 0
        manual_mode  = False
        manual_steer = 0.0

        print("🚗  Running — ESC/Q: quit   R: reset nav   P: toggle manual/auto   M: Toggle Map View")
        print("         Manual — W: forward   S: brake/reverse   A/D: steer")

        while running:
            clock.tick(20)
            frame += 1

            if args.sync: world.tick()
            else: world.wait_for_tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q): running = False
                    elif event.key == K_r:
                        nav.pos_history.clear()
                        nav.steer_sign_hist.clear()
                        nav.reverse_count = 0
                        nav.emergency_count = 0
                        print("⟳  Nav state reset")
                    elif event.key == K_p:
                        manual_mode = not manual_mode
                        manual_steer = 0.0
                        tag = "🕹  MANUAL" if manual_mode else "🤖  AUTO"
                        print(f"[P] Switched to {tag} mode")

            if manual_mode:
                keys = pygame.key.get_pressed()
                m_throttle, m_brake, m_reverse, m_steer_tgt = 0.0, 0.08, _R_FWD, 0.0
                if keys[K_w]: m_throttle, m_brake, m_reverse = 0.65, 0.0, _R_FWD
                if keys[K_s]:
                    vel = vehicle.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    if spd > 0.5: m_brake, m_throttle, m_reverse = 0.8, 0.0, _R_FWD
                    else: m_throttle, m_brake, m_reverse = 0.55, 0.0, _R_REV
                if keys[K_a]: m_steer_tgt = -0.6
                if keys[K_d]: m_steer_tgt =  0.6
                manual_steer = 0.45 * m_steer_tgt + 0.55 * manual_steer
                control = carla.VehicleControl(throttle=float(np.clip(m_throttle, 0, 1)),
                                               steer=float(np.clip(manual_steer, -1, 1)),
                                               brake=float(np.clip(m_brake, 0, 1)), reverse=m_reverse)
                vehicle.apply_control(control)
                mode_tag = "🕹  MANUAL"
            else:
                lidar_pts = rainbow_lidar.lidar_points
                depth_raw = depth_cam.depth_array
                control = nav.compute_control(lidar_pts, depth_raw, vehicle)
                vehicle.apply_control(control)
                mode_tag = "🤖  AUTO  "
                
                # Update Telemetry
                if telemetry:
                    telemetry.update(vehicle, nav.diag, nav.last_plan_angle, nav.occ_grid)
                    telemetry.render(telemetry_surface, nav.occ_grid, vehicle, nav.last_plan_angle)

            # ── HUD ─────────────────────────────────────────────────
            d = nav.diag
            display_manager.set_hud([
                f"MODE  : {mode_tag}          FRAME: {frame}",
                f"STATE : {d['state']:<16}  INV_DRIVE: {INVERT_DRIVE}",
                f"SPEED : {d['speed']*3.6:5.1f} km/h   STEER: {d['steer']:+.3f}",
                f"FRONT : {d['front']:5.1f}m   FL: {d['fl']:5.1f}m   FR: {d['fr']:5.1f}m",
                f"COST  : {d.get('total', 0):.2f} (Sl:{d.get('slope',0):.1f} Oc:{d.get('occ',0):.1f})",
                f"STUCK : {'YES' if nav._check_stuck() else 'no':<4}  REV: {nav.reverse_count:3d}"
            ])

            try:
                display_manager.render()
            except Exception as e:
                print(f"⚠  Render skip (frame {frame}): {e}")

    finally:
        print("\n🧹  Cleaning up …")
        if display_manager: display_manager.destroy()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors if a is not None])
        world.apply_settings(original_settings)
        pygame.quit()
        print("✅  Done.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARLA Advanced Nav v4")
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
