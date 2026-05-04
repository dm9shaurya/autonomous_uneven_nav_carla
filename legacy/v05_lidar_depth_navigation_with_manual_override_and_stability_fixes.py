#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          CARLA AUTONOMOUS NAVIGATOR — LiDAR + DEPTH          ║
║                     STABLE BUILD v3                          ║
║                                                              ║
║  Fix 1 : LiDAR load reduced  (56k pts / 32ch / 10Hz)        ║
║  Fix 2 : Depth dual-callback replaced → _depth_combined      ║
║  Fix 3 : Dead pygame.draw.line removed + HSV rewritten       ║
║  Fix 4 : clock.tick(20)                                      ║
║  Fix 5 : Render wrapped in try/except                        ║
║  Fix 6 : max_substep physics settings added                  ║
║  Fix 7 : INVERT_DRIVE — Cybertruck axis flip (was driving    ║
║          backward by default; all reverse= flags corrected)  ║
║  Fix 8 : Anti-rotation throttle no longer overrides          ║
║          obstacle brake (throttle boost only when brake==0)  ║
║  Fix 9 : spawn_z None-check (0.0 is falsy, skipped before)  ║
║  New   : WASD manual control + P toggle (auto ↔ manual)      ║
║  Opt   : Colour-bar precomputed, LiDAR frame-skip,           ║
║          RGB cam resolution reduced, nav stays on one sensor ║
╚══════════════════════════════════════════════════════════════╝
"""

import carla
import argparse
import random
import time
import math
import colorsys
import numpy as np
from collections import deque

try:
    import pygame
    from pygame.locals import (K_ESCAPE, K_q, K_r,
                               K_w, K_s, K_a, K_d, K_p)
except ImportError:
    raise RuntimeError("pygame not installed — run: pip install pygame")


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

FRONT_WARN_DIST  =  8.0
FRONT_STOP_DIST  =  5.0
FRONT_EMRG_DIST  =  3.0

OBSTACLE_Z_MIN   = -1.8
OBSTACLE_Z_MAX   =  2.5
SLOPE_BLOCK_Z    =  1.2

MAX_SPEED_MS     =  7.0
THROTTLE_CRUISE  =  0.85
THROTTLE_CAUTION =  0.45
THROTTLE_RECOV   =  0.55

STUCK_WINDOW_FRAMES = 90
STUCK_MIN_TRAVEL_M  =  1.5
REVERSE_FRAMES      = 50

OSC_WINDOW      = 24
OSC_FLIP_THRESH = 14
OSC_LOCK_FRAMES = 40

RAINBOW_Z_MIN = -2.5
RAINBOW_Z_MAX =  3.0

FALL_Z_THRESH  = -10.0
SPAWN_SETTLE_S =   3.0

# FIX 1 — reduced LiDAR load (was 280k pts / 64ch / 20Hz)
LIDAR_CHANNELS     = "32"
LIDAR_PTS_PER_SEC  = "56000"
LIDAR_ROT_FREQ     = "10"
LIDAR_RANGE        = "100"
LIDAR_FRAME_SKIP   = 2       # re-render rainbow every N callbacks

# Reduced RGB camera resolution (scales up for display)
RGB_WIDTH  = 320
RGB_HEIGHT = 180

# ── FIX 7 : Drive-axis inversion ─────────────────────────────
# The Cybertruck blueprint's physics mesh is oriented so that
#   reverse=False (physics "forward") drives the car BACKWARD visually
#   reverse=True  (physics "reverse") drives the car FORWARD  visually
#
# INVERT_DRIVE = True flips every VehicleControl reverse= flag so that
# "forward" commands actually go forward and "reverse" goes backward.
#
# If you switch to a vehicle that drives correctly, set this to False.
INVERT_DRIVE = True

# Convenience aliases used in every VehicleControl call below.
#   _R_FWD  → the reverse= value that makes the car go FORWARD
#   _R_REV  → the reverse= value that makes the car go BACKWARD
_R_FWD = INVERT_DRIVE        # True  when INVERT_DRIVE is True
_R_REV = not INVERT_DRIVE    # False when INVERT_DRIVE is True


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

class CustomTimer:
    def __init__(self):
        self.timer = time.perf_counter
    def time(self):
        return self.timer()


def _hsv_to_rgb_vectorised(hue_arr):
    """
    Pure-numpy HSV→RGB for array of hues (S=1, V=1).
    FIX 3: replaces the broken sector loop from v1.
    Returns uint8 (N, 3).
    """
    h6 = hue_arr * 6.0
    hi = np.floor(h6).astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    q  = (1.0 - f)

    rgb = np.zeros((len(hue_arr), 3), dtype=np.float32)
    lut = [
        (1.0,   f, 0.0),   # sector 0
        (  q, 1.0, 0.0),   # sector 1
        (0.0, 1.0,   f),   # sector 2
        (0.0,   q, 1.0),   # sector 3
        (  f, 0.0, 1.0),   # sector 4
        (1.0, 0.0,   q),   # sector 5
    ]
    for s, (rv, gv, bv) in enumerate(lut):
        m = hi == s
        if not m.any():
            continue
        rgb[m, 0] = rv if np.isscalar(rv) else rv[m]
        rgb[m, 1] = gv if np.isscalar(gv) else gv[m]
        rgb[m, 2] = bv if np.isscalar(bv) else bv[m]

    return (rgb * 255).astype(np.uint8)


def _precompute_colour_bar(height, bar_w=12):
    """
    Build (bar_w, height, 3) uint8 rainbow legend — computed once.
    Replaces the per-pixel Python loop that ran every callback in v1.
    """
    bar = np.zeros((bar_w, height, 3), dtype=np.uint8)
    for py in range(height):
        norm = 1.0 - py / height
        hue  = (1.0 - norm) * 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bar[:, py] = (int(r * 255), int(g * 255), int(b * 255))
    return bar


# ─────────────────────────────────────────────────────────────
# DISPLAY MANAGER
# ─────────────────────────────────────────────────────────────

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Autonomous Navigator v3")
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
# SENSOR MANAGER
# ─────────────────────────────────────────────────────────────

class SensorManager:
    """
    Supported sensor types: RGBCamera | DepthCamera | RainbowLiDAR

    FIX 2 — DepthCamera uses _depth_combined_callback (single listener)
             that writes both self.surface (display) and self.depth_array
             (navigation). No external .listen() override in main().

    FIX 3 — RainbowLiDAR uses clean vectorised HSV. No temp surfaces.
             Precomputed colour bar blit-ed each frame.
    """

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

        # Exposed data for navigation
        self.lidar_points = None   # (N, 3) float32 xyz
        self.depth_array  = None   # (H, W) uint8  raw depth channel

        # Rainbow internal state
        self._lidar_frame   = 0
        self._colour_bar    = None

        self.sensor = self._init_sensor(sensor_type, transform,
                                        attached, sensor_options)
        self.display_man.add_sensor(self)

    # ── Init ────────────────────────────────────────────────

    def _init_sensor(self, sensor_type, transform, attached, opts):
        bp_lib = self.world.get_blueprint_library()
        ds     = self.display_man.get_display_size()

        if sensor_type == "RGBCamera":
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rgb)
            return actor

        elif sensor_type == "DepthCamera":
            bp = bp_lib.find("sensor.camera.depth")
            bp.set_attribute("image_size_x", str(RGB_WIDTH))
            bp.set_attribute("image_size_y", str(RGB_HEIGHT))
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            # FIX 2: single combined callback, no external override
            actor.listen(self._depth_combined_callback)
            return actor

        elif sensor_type == "RainbowLiDAR":
            bp = bp_lib.find("sensor.lidar.ray_cast")
            # FIX 1: reduced load
            bp.set_attribute("range",              LIDAR_RANGE)
            bp.set_attribute("channels",           LIDAR_CHANNELS)
            bp.set_attribute("points_per_second",  LIDAR_PTS_PER_SEC)
            bp.set_attribute("rotation_frequency", LIDAR_ROT_FREQ)
            bp.set_attribute("dropoff_general_rate", "0.0")
            for k, v in opts.items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, transform, attach_to=attached)
            actor.listen(self._save_rainbow_lidar)
            # Precompute legend bar using cell height
            self._colour_bar = _precompute_colour_bar(ds[1])
            return actor

        return None

    # ── Callbacks ───────────────────────────────────────────

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
        """
        FIX 2: One listener does both jobs.
        Reads raw_data BEFORE colour conversion for nav buffer,
        then converts for display surface.
        """
        t0 = self.timer.time()

        # Nav buffer — raw channel 0 (linear depth proxy)
        arr_raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr_raw = np.reshape(arr_raw, (image.height, image.width, 4))
        self.depth_array = arr_raw[:, :, 0].copy()

        # Display surface — logarithmic colourmap
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
        """
        FIX 3: No temp pygame surface mid-callback.
        Uses clean vectorised HSV (replaces broken loop).
        Frame-skip for render, always updates nav data.
        """
        t0 = self.timer.time()
        self._lidar_frame += 1

        pts = np.frombuffer(image.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (-1, 4))
        xyz = pts[:, :3].copy()

        # Always refresh nav data every callback
        self.lidar_points = xyz

        # Skip re-rendering the surface every other frame
        if self._lidar_frame % LIDAR_FRAME_SKIP != 0:
            self.time_proc += self.timer.time() - t0
            self.ticks += 1
            return

        ds          = self.display_man.get_display_size()
        lidar_range = 2.0 * float(LIDAR_RANGE)

        # Top-down projection
        xy = xyz[:, :2].copy()
        xy *= min(ds) / lidar_range
        xy += (0.5 * ds[0], 0.5 * ds[1])
        xy  = xy.astype(np.int32)

        # Height → hue (low Z=blue, high Z=red)
        z_norm = np.clip(
            (xyz[:, 2] - RAINBOW_Z_MIN) / (RAINBOW_Z_MAX - RAINBOW_Z_MIN),
            0.0, 1.0
        )
        hue    = ((1.0 - z_norm) * 0.75).astype(np.float32)
        rgb_u8 = _hsv_to_rgb_vectorised(hue)

        # Paint valid pixels
        img   = np.zeros((ds[0], ds[1], 3), dtype=np.uint8)
        valid = (
            (xy[:, 0] >= 0) & (xy[:, 0] < ds[0]) &
            (xy[:, 1] >= 0) & (xy[:, 1] < ds[1])
        )
        img[xy[valid, 0], xy[valid, 1]] = rgb_u8[valid]

        # FIX 3: crosshair in numpy only — no temp surface
        cx, cy = ds[0] // 2, ds[1] // 2
        r = 7
        img[cx - 1 : cx + 2, max(0, cy - r) : min(ds[1], cy + r)] = 255
        img[max(0, cx - r) : min(ds[0], cx + r), cy - 1 : cy + 2] = 255

        # Precomputed colour bar on right edge
        bar_w = self._colour_bar.shape[0]
        img[ds[0] - bar_w : ds[0], :] = self._colour_bar

        self.surface = pygame.surfarray.make_surface(img)
        self.time_proc += self.timer.time() - t0
        self.ticks += 1

    # ── Render ──────────────────────────────────────────────

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


# ─────────────────────────────────────────────────────────────
# NAVIGATION CONTROLLER
# ─────────────────────────────────────────────────────────────

class NavigationController:
    def __init__(self):
        # Steering smoothing
        self.steer_ema       = 0.0
        self.steer_ema_alpha = 0.35

        # Motion tracking
        self.pos_history   = deque(maxlen=STUCK_WINDOW_FRAMES)
        self.reverse_count = 0
        self.reverse_steer = 0.0

        # Oscillation control
        self.steer_sign_hist = deque(maxlen=OSC_WINDOW)
        self.forced_steer    = 0.0
        self.forced_count    = 0

        # Forward commitment
        self.commit_steer = 0.0
        self.commit_count = 0

        self.spawn_z         = None   # set after settle; use `is not None` check
        self.emergency_count = 0

        self.diag = {
            "state": "INIT",
            "front": 0.0, "fl": 0.0, "fr": 0.0,
            "speed": 0.0, "steer": 0.0,
        }

    # ─────────────────────────────────────────────

    def _analyse_lidar(self, pts):
        if pts is None or len(pts) == 0:
            return None

        mask = (pts[:, 2] > OBSTACLE_Z_MIN) & (pts[:, 2] < OBSTACLE_Z_MAX)
        obs  = pts[mask]
        cap  = float(LIDAR_RANGE)

        empty = dict(front=cap, fl=cap, fr=cap, left=cap, right=cap,
                     front_slope_blocked=False,
                     fl_slope_blocked=False, fr_slope_blocked=False)
        if len(obs) == 0:
            return empty

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
            "front":  sector_min(-sw, sw),
            "fl":     sector_min(sw, 3*sw),
            "fr":     sector_min(-3*sw, -sw),
            "left":   sector_min(3*sw, 5*sw),
            "right":  sector_min(-5*sw, -3*sw),
            "front_slope_blocked": slope_blocked(-sw, sw),
            "fl_slope_blocked": slope_blocked(sw, 3*sw),
            "fr_slope_blocked": slope_blocked(-3*sw, -sw),
        }

    def _analyse_depth(self, depth_arr):
        if depth_arr is None:
            return None
        h, w = depth_arr.shape
        roi = depth_arr[h//4:h*3//4, w//4:w*3//4].astype(np.float32)
        return (float(roi.mean()) / 255.0) * 100.0

    def _check_stuck(self):
        if len(self.pos_history) < STUCK_WINDOW_FRAMES:
            return False
        old = self.pos_history[0]
        new = self.pos_history[-1]
        dist = math.hypot(new[0]-old[0], new[1]-old[1])
        return dist < STUCK_MIN_TRAVEL_M and self.diag["speed"] < 0.5

    # ─────────────────────────────────────────────

    def compute_control(self, lidar_pts, depth_raw, vehicle):
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        speed = math.hypot(vel.x, vel.y)

        self.pos_history.append((loc.x, loc.y))
        self.diag["speed"] = speed

        # ── FALL recovery ────────────────────────────────────────────
        # FIX 9: use `is not None` so spawn_z=0.0 isn't treated as falsy.
        if self.spawn_z is not None and loc.z < self.spawn_z - 8.0:
            self.emergency_count = 35

        if self.emergency_count > 0:
            self.emergency_count -= 1
            self.diag["state"] = "FALL-RECOV"
            # FIX 7: _R_REV makes the car actually reverse (away from the fall)
            return carla.VehicleControl(throttle=0.6, reverse=_R_REV)

        # ── ACTIVE REVERSE (stuck recovery countdown) ─────────────────
        if self.reverse_count > 0:
            self.reverse_count -= 1
            self.diag["state"] = "REVERSING"
            # FIX 7: _R_REV makes the car actually go backward
            return carla.VehicleControl(
                throttle=0.5,
                steer=-self.reverse_steer * 0.5,
                reverse=_R_REV
            )

        # ── STUCK check ──────────────────────────────────────────────
        if self._check_stuck():
            a = self._analyse_lidar(lidar_pts)
            self.reverse_steer = 0.6 if (a and a["fl"] < a["fr"]) else -0.6
            self.reverse_count = REVERSE_FRAMES
            self.pos_history.clear()
            self.diag["state"] = "STUCK→REV"
            # FIX 7: _R_REV makes the car actually go backward
            return carla.VehicleControl(
                throttle=0.5,
                steer=self.reverse_steer,
                reverse=_R_REV
            )

        # ── NORMAL NAV ───────────────────────────────────────────────
        analysis = self._analyse_lidar(lidar_pts)
        depth_m  = self._analyse_depth(depth_raw)

        throttle     = THROTTLE_CRUISE
        brake        = 0.0
        target_steer = 0.0

        if analysis:
            front = analysis["front"]
            fl, fr = analysis["fl"], analysis["fr"]
            self.diag.update(front=front, fl=fl, fr=fr)

            if depth_m:
                front = min(front, depth_m * 0.65)

            open_L = fl + analysis["left"] * 0.3
            open_R = fr + analysis["right"] * 0.3

            # ── FORWARD COMMITMENT ───────────────────────────────────
            if self.commit_count > 0:
                self.commit_count -= 1
                target_steer = self.commit_steer
            else:
                steer_raw = (open_L - open_R)
                target_steer = float(np.clip(steer_raw * 0.015, -0.12, 0.12))
                if abs(steer_raw) > 2.0:
                    self.commit_steer = 0.35 if steer_raw > 0 else -0.35
                    self.commit_count = 20

            # ── OBSTACLE HANDLING ────────────────────────────────────
            if front < FRONT_EMRG_DIST:
                self.diag["state"] = "EMRG-STOP"
                brake        = 0.6
                throttle     = 0.0
                target_steer = 0.6 if open_L > open_R else -0.6

            elif front < FRONT_STOP_DIST:
                self.diag["state"] = "SLOW-TURN"
                throttle     = 0.2
                target_steer = 0.5 if open_L > open_R else -0.5

            elif front < FRONT_WARN_DIST:
                self.diag["state"] = "CAUTION"
                throttle     = THROTTLE_CAUTION

            else:
                self.diag["state"] = "CRUISE"

        else:
            self.diag["state"] = "NO-SENSOR"

        # ── ANTI-ROTATION FIX (FIX 8) ────────────────────────────────
        # Only boost throttle when NOT actively braking for an obstacle.
        # Original bug: brake=0.6 (obstacle stop) + this override to
        # throttle=0.4 kept the vehicle frozen → stuck detector fired →
        # spurious REVERSE. Now we only boost throttle when brake is clear.
        if speed < 0.8:
            if brake == 0.0:
                throttle = max(throttle, 0.4)
            target_steer *= 0.5

        # EMA steer smoothing
        self.steer_ema = (
            self.steer_ema_alpha * target_steer +
            (1 - self.steer_ema_alpha) * self.steer_ema
        )
        self.diag["steer"] = self.steer_ema

        # FIX 7: _R_FWD makes the car actually go forward
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

    try:
        world             = client.get_world()
        original_settings = world.get_settings()

        # FIX 6 — substep physics
        if args.sync:
            tm       = client.get_trafficmanager(8000)
            settings = world.get_settings()
            tm.set_synchronous_mode(True)
            settings.synchronous_mode       = True
            settings.fixed_delta_seconds    = 0.05
            settings.max_substep_delta_time = 0.01   # FIX 6
            settings.max_substeps           = 10      # FIX 6
            world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        
        spawn_transform = carla.Transform(
        carla.Location(x=906.66, y=-875.78, z=4.7620),
        carla.Rotation(pitch=-4.50, yaw=88.0, roll=0.0),
    )

        # Spawn Cybertruck at z+5
        vehicle_bp   = bp_lib.find("vehicle.tesla.cybertruck")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle = None
        for sp in spawn_points:
            sp.location.z += 5.0
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"✅  Spawned at ({sp.location.x:.1f}, "
                      f"{sp.location.y:.1f}, {sp.location.z:.1f})")
                break

        if vehicle is None:
            raise RuntimeError("❌  No valid spawn point found")
        actors.append(vehicle)

        print(f"⏳  Settling {SPAWN_SETTLE_S}s …")
        t0 = time.time()
        while time.time() - t0 < SPAWN_SETTLE_S:
            world.tick() if args.sync else world.wait_for_tick()

        # FIX 9: explicit None check so spawn_z=0.0 is not treated as falsy
        nav.spawn_z = vehicle.get_location().z
        print(f"📍  Settled Z = {nav.spawn_z:.2f}m")
        print(f"ℹ️   INVERT_DRIVE={INVERT_DRIVE}  "
              f"(_R_FWD={_R_FWD}, _R_REV={_R_REV})")

        # Display Manager — 2×3 grid
        display_manager = DisplayManager(
            grid_size=[2, 3],
            window_size=[args.width, args.height]
        )

        # Row 0: Front | Rainbow LiDAR | Right
        # Row 1: Left  | Depth         | Rear

        front_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8),
                            carla.Rotation(pitch=-8)),
            vehicle, {}, display_pos=[0, 0], label="FRONT CAM"
        )
        actors.append(front_cam.sensor)

        rainbow_lidar = SensorManager(
            world, display_manager, "RainbowLiDAR",
            carla.Transform(carla.Location(z=2.5)),
            vehicle, {},
            display_pos=[0, 1], label="RAINBOW LiDAR (height)"
        )
        actors.append(rainbow_lidar.sensor)

        right_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=0.8, z=1.5),
                            carla.Rotation(yaw=90)),
            vehicle, {}, display_pos=[0, 2], label="RIGHT CAM"
        )
        actors.append(right_cam.sensor)

        left_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(y=-0.8, z=1.5),
                            carla.Rotation(yaw=-90)),
            vehicle, {}, display_pos=[1, 0], label="LEFT CAM"
        )
        actors.append(left_cam.sensor)

        depth_cam = SensorManager(
            world, display_manager, "DepthCamera",
            carla.Transform(carla.Location(x=2.0, z=1.8),
                            carla.Rotation(pitch=-5)),
            vehicle, {}, display_pos=[1, 1], label="DEPTH (LOG)"
        )
        actors.append(depth_cam.sensor)
        # FIX 2: depth_cam.depth_array populated inside _depth_combined_callback

        rear_cam = SensorManager(
            world, display_manager, "RGBCamera",
            carla.Transform(carla.Location(x=-3.5, z=2.0),
                            carla.Rotation(yaw=180, pitch=-10)),
            vehicle, {}, display_pos=[1, 2], label="REAR CAM"
        )
        actors.append(rear_cam.sensor)

        # ── Main loop ─────────────────────────────────────────────────
        clock        = pygame.time.Clock()
        running      = True
        frame        = 0
        manual_mode  = False   # P toggles AUTO ↔ MANUAL
        manual_steer = 0.0     # EMA steer for WASD — prevents wheel snap

        print("🚗  Running — ESC/Q: quit   R: reset nav   P: toggle manual/auto")
        print("         Manual — W: forward   S: brake/reverse   A/D: steer")

        while running:
            clock.tick(20)   # FIX 4
            frame += 1

            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # ── Events ──────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False

                    elif event.key == K_r:
                        nav.pos_history.clear()
                        nav.steer_sign_hist.clear()
                        nav.reverse_count   = 0
                        nav.forced_count    = 0
                        nav.emergency_count = 0
                        print("⟳  Nav state reset")

                    elif event.key == K_p:
                        manual_mode  = not manual_mode
                        manual_steer = 0.0   # reset steer on switch
                        tag = "🕹  MANUAL" if manual_mode else "🤖  AUTO"
                        print(f"[P] Switched to {tag} mode")

            # ── WASD manual control ──────────────────────────────────
            if manual_mode:
                keys = pygame.key.get_pressed()

                m_throttle  = 0.0
                m_brake     = 0.08    # gentle coast-stop when no key held
                m_reverse   = _R_FWD  # default idle: fwd ready
                m_steer_tgt = 0.0

                if keys[K_w]:
                    # FIX 7: _R_FWD = True (inverted) → car goes forward
                    m_throttle = 0.65
                    m_brake    = 0.0
                    m_reverse  = _R_FWD

                if keys[K_s]:
                    vel = vehicle.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    if spd > 0.5:
                        # Still rolling forward — brake first
                        m_brake    = 0.8
                        m_throttle = 0.0
                        m_reverse  = _R_FWD   # keep fwd flag while braking
                    else:
                        # Near-stopped — FIX 7: _R_REV = False → car goes backward
                        m_throttle = 0.55
                        m_brake    = 0.0
                        m_reverse  = _R_REV

                if keys[K_a]:
                    m_steer_tgt = -0.6
                if keys[K_d]:
                    m_steer_tgt =  0.6

                # Smooth steer so wheels don't snap hard
                manual_steer = 0.45 * m_steer_tgt + 0.55 * manual_steer

                control = carla.VehicleControl(
                    throttle = float(np.clip(m_throttle, 0, 1)),
                    steer    = float(np.clip(manual_steer, -1, 1)),
                    brake    = float(np.clip(m_brake, 0, 1)),
                    reverse  = m_reverse,
                )
                vehicle.apply_control(control)
                mode_tag = "🕹  MANUAL"

            else:
                # ── Autonomous nav ───────────────────────────────────
                lidar_pts = rainbow_lidar.lidar_points
                depth_raw = depth_cam.depth_array

                control = nav.compute_control(lidar_pts, depth_raw, vehicle)
                vehicle.apply_control(control)
                mode_tag = "🤖  AUTO  "

            # ── HUD ─────────────────────────────────────────────────
            d = nav.diag
            display_manager.set_hud([
                f"MODE  : {mode_tag}          FRAME: {frame}",
                f"STATE : {d['state']:<16}  INV_DRIVE: {INVERT_DRIVE}",
                f"SPEED : {d['speed']*3.6:5.1f} km/h   STEER: {d['steer']:+.3f}",
                f"FRONT : {d['front']:5.1f}m   FL: {d['fl']:5.1f}m   "
                f"FR: {d['fr']:5.1f}m",
                f"STUCK : {'YES' if nav._check_stuck() else 'no':<4}  "
                f"REV: {nav.reverse_count:3d}  "
                f"EMRG: {nav.emergency_count:3d}",
            ])

            # FIX 5: render wrapped
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
        pygame.quit()
        print("✅  Done.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARLA Autonomous Navigator v3")
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