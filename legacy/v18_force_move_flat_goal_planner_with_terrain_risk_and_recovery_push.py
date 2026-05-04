#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA Cybertruck Flat/Hill Goal Navigation — FORCE-MOVE FIXED VERSION
====================================================================
This version fixes the usual reason your vehicle stayed at 0.00 m/s:
1) The old forward obstacle check treated hill/ground LiDAR points as obstacles,
   so the controller kept braking.
2) The controller was too timid near hills.
3) If CARLA physics/wheels stick on a hill, this version uses a short physical
   push fallback with set_target_velocity during recovery.

Outputs are saved in: carla_force_move_results/
- navigation_log.csv
- controls.csv
- trajectory.csv
- cost_reward.csv
- terrain_samples.csv
- trajectory.svg
- nmpc_cost.svg
- reward.svg
- linear_velocity.svg
- angular_velocity.svg
- vehicle_speed.svg
- goal_distance.svg
- speed_heatmap.svg
- flatness_heatmap.svg

Run:
    python3 carla_flat_goal_FORCE_MOVE_fixed.py

Keys:
    ESC : quit
    R   : force recovery/push
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import carla
except ImportError:
    print("[FATAL] CARLA Python API not found. Add CARLA egg to PYTHONPATH.")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("[FATAL] pygame not found. Install with: pip install pygame")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[FATAL] matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 20.0
    synchronous: bool = True
    dt: float = 0.05

    vehicle_blueprint: str = "vehicle.tesla.cybertruck"

    # Same spawn and goal from your code
    spawn_x: float = 240.35
    spawn_y: float = 223.16
    spawn_z: float = 2.15
    spawn_pitch: float = -14.03
    spawn_yaw: float = 103.85
    spawn_roll: float = 0.00

    goal_x: float = -115.10
    goal_y: float = 187.96
    goal_tolerance: float = 5.0

    # LiDAR
    lidar_range: float = 55.0
    lidar_channels: int = 64
    lidar_points_per_second: int = 130000
    lidar_rotation_frequency: float = 20.0
    lidar_upper_fov: float = 20.0
    lidar_lower_fov: float = -35.0

    # Camera
    camera_width: int = 480
    camera_height: int = 270
    camera_fov: float = 90.0

    # Local planner
    candidate_count: int = 81
    fov_deg: float = 170.0
    candidate_distances: Tuple[float, ...] = (8.0, 14.0, 22.0, 32.0, 45.0)
    replan_every_ticks: int = 2

    # Cost weights: goal progress is dominant; flatness is preference only
    w_progress: float = 180.0
    w_goal: float = 1.1
    w_heading: float = 15.0
    w_slope: float = 18.0
    w_rough: float = 10.0
    w_obstacle: float = 45.0
    w_memory: float = 80.0

    # Controller
    wheel_base: float = 3.807
    max_steer: float = 0.70
    max_throttle: float = 0.95
    max_brake: float = 0.65
    min_throttle_auto: float = 0.50       # FORCE movement in AUTO
    min_throttle_hill: float = 0.68       # FORCE hill movement
    target_speed_flat: float = 9.0
    target_speed_bad: float = 5.0
    target_speed_near_goal: float = 3.0
    kp_speed: float = 0.20
    steer_gain: float = 1.45
    throttle_rate: float = 0.20           # faster than old code
    steer_rate: float = 0.12

    # IMPORTANT: obstacle detection only for high objects, not hill/ground
    obstacle_z_min: float = 0.85
    obstacle_y_half_width: float = 2.4
    emergency_clearance: float = 1.40
    danger_clearance: float = 2.50

    # Stuck and recovery
    stuck_speed_thresh: float = 0.20
    stuck_time_s: float = 1.50
    stuck_disp_thresh: float = 0.45
    recovery_reverse_s: float = 1.10
    recovery_turn_s: float = 1.00
    recovery_forward_s: float = 3.50
    recovery_reverse_throttle: float = 0.65
    recovery_forward_throttle: float = 0.95
    recovery_push_speed: float = 7.0      # physical set_target_velocity push if wheels stuck
    stuck_memory_radius: float = 15.0

    # Simulation/output
    post_spawn_wait_s: float = 1.5
    max_run_time_s: float = 900.0
    print_every: int = 5
    viz_every: int = 2
    out_dir: str = "carla_force_move_results"


CFG = Config()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def local_to_world(lx: float, ly: float, yaw: float, wx: float, wy: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return wx + c * lx - s * ly, wy + s * lx + c * ly


def world_to_local(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


@dataclass
class VehicleState:
    x: float
    y: float
    z: float
    yaw: float
    pitch_deg: float
    roll_deg: float
    speed: float


@dataclass
class Candidate:
    wx: float
    wy: float
    lx: float
    ly: float
    alpha: float
    distance: float
    progress: float
    goal_dist: float
    heading_error: float
    slope: float
    roughness: float
    obstacle_risk: float
    flatness: float
    memory_penalty: float
    cost: float
    reward: float


@dataclass
class ControlCmd:
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    reverse: bool = False


class ResultLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.rows: List[dict] = []
        self.terrain_rows: List[dict] = []

    def log(self, t: float, st: VehicleState, cmd: ControlCmd, cand: Candidate,
            goal_dist: float, mode: str, cost: float, reward: float) -> None:
        angular_velocity = (st.speed / max(self.cfg.wheel_base, 1e-6)) * math.tan(cmd.steer)
        self.rows.append({
            "time": t, "x": st.x, "y": st.y, "z": st.z,
            "yaw": st.yaw, "pitch_deg": st.pitch_deg, "roll_deg": st.roll_deg,
            "speed": st.speed, "linear_velocity": st.speed,
            "angular_velocity": angular_velocity,
            "throttle": cmd.throttle, "steer": cmd.steer, "brake": cmd.brake,
            "reverse": int(cmd.reverse), "goal_distance": goal_dist,
            "mode": mode, "cost": cost, "reward": reward,
            "subgoal_x": cand.wx, "subgoal_y": cand.wy,
            "flatness": cand.flatness, "slope": cand.slope,
            "roughness": cand.roughness, "progress": cand.progress,
            "obstacle_risk": cand.obstacle_risk,
        })

    def log_terrain_points(self, t: float, st: VehicleState, pts: Optional[np.ndarray]) -> None:
        if pts is None or len(pts) < 10:
            return
        step = max(1, len(pts) // 800)
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        for p in pts[::step]:
            lx, ly, lz = float(p[0]), float(p[1]), float(p[2])
            wx = st.x + c * lx - s * ly
            wy = st.y + s * lx + c * ly
            self.terrain_rows.append({"time": t, "x": wx, "y": wy, "z": st.z + lz})

    def save_csv(self) -> None:
        if not self.rows:
            return
        with open(os.path.join(self.cfg.out_dir, "navigation_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader(); w.writerows(self.rows)

        def write_subset(name: str, keys: List[str]) -> None:
            with open(os.path.join(self.cfg.out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow({k: r[k] for k in keys})

        write_subset("controls.csv", ["time", "throttle", "steer", "brake", "reverse", "linear_velocity", "angular_velocity", "speed"])
        write_subset("trajectory.csv", ["time", "x", "y", "z", "goal_distance"])
        write_subset("cost_reward.csv", ["time", "cost", "reward", "goal_distance", "flatness", "slope", "roughness", "progress"])

        if self.terrain_rows:
            with open(os.path.join(self.cfg.out_dir, "terrain_samples.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.terrain_rows[0].keys()))
                w.writeheader(); w.writerows(self.terrain_rows)

    def _line(self, x, y, title, ylabel, filename) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time [s]"); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.out_dir, filename), format="svg")
        plt.close()

    def save_plots(self) -> None:
        if not self.rows:
            return
        t = np.array([r["time"] for r in self.rows])
        x = np.array([r["x"] for r in self.rows])
        y = np.array([r["y"] for r in self.rows])
        speed = np.array([r["speed"] for r in self.rows])
        cost = np.array([r["cost"] for r in self.rows])
        reward = np.array([r["reward"] for r in self.rows])
        lin = np.array([r["linear_velocity"] for r in self.rows])
        ang = np.array([r["angular_velocity"] for r in self.rows])
        gdist = np.array([r["goal_distance"] for r in self.rows])

        self._line(t, cost, "NMPC-style Cost Function", "Cost", "nmpc_cost.svg")
        self._line(t, reward, "Reward", "Reward", "reward.svg")
        self._line(t, lin, "Linear Velocity", "m/s", "linear_velocity.svg")
        self._line(t, ang, "Angular Velocity", "rad/s", "angular_velocity.svg")
        self._line(t, speed, "Vehicle Speed", "m/s", "vehicle_speed.svg")
        self._line(t, gdist, "Distance to Goal", "m", "goal_distance.svg")

        plt.figure(figsize=(7, 7))
        plt.plot(x, y, linewidth=2, label="trajectory")
        plt.scatter([self.cfg.spawn_x], [self.cfg.spawn_y], marker="o", s=70, label="start")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=90, label="goal")
        plt.axis("equal"); plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("World Trajectory"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(self.cfg.out_dir, "trajectory.svg"), format="svg"); plt.close()

        plt.figure(figsize=(7, 7))
        sc = plt.scatter(x, y, c=speed, s=18)
        plt.colorbar(sc, label="Speed [m/s]")
        plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=90)
        plt.axis("equal"); plt.grid(True, alpha=0.3)
        plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("Vehicle Speed Heat Map")
        plt.tight_layout(); plt.savefig(os.path.join(self.cfg.out_dir, "speed_heatmap.svg"), format="svg"); plt.close()

        if len(self.terrain_rows) > 50:
            tx = np.array([r["x"] for r in self.terrain_rows])
            ty = np.array([r["y"] for r in self.terrain_rows])
            tz = np.array([r["z"] for r in self.terrain_rows])
            plt.figure(figsize=(7, 7))
            sc = plt.scatter(tx, ty, c=tz, s=5)
            plt.colorbar(sc, label="Terrain height proxy [m]")
            plt.plot(x, y, linewidth=1.5)
            plt.scatter([self.cfg.goal_x], [self.cfg.goal_y], marker="x", s=90)
            plt.axis("equal"); plt.grid(True, alpha=0.3)
            plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("Flatness / Terrain Height Heat Map")
            plt.tight_layout(); plt.savefig(os.path.join(self.cfg.out_dir, "flatness_heatmap.svg"), format="svg"); plt.close()

    def finalize(self) -> None:
        print(f"\n[SAVE] Saving results in {os.path.abspath(self.cfg.out_dir)}")
        self.save_csv(); self.save_plots()
        print("[SAVE] Done")


class SensorManager:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, cfg: Config):
        self.world = world; self.vehicle = vehicle; self.cfg = cfg
        self.actors: List[carla.Actor] = []
        self.lidar_q: Queue = Queue(maxsize=2)
        self.front_q: Queue = Queue(maxsize=2)
        self.rear_q: Queue = Queue(maxsize=2)
        self._setup()

    @staticmethod
    def _put_latest(q: Queue, data) -> None:
        while q.full():
            try: q.get_nowait()
            except Empty: break
        q.put(data)

    @staticmethod
    def _img_to_array(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        return arr[:, :, ::-1].copy()

    @staticmethod
    def _get_latest(q: Queue):
        item = None
        while True:
            try: item = q.get_nowait()
            except Empty: break
        return item

    def _setup(self) -> None:
        bp = self.world.get_blueprint_library()
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(self.cfg.lidar_range))
        lidar_bp.set_attribute("channels", str(self.cfg.lidar_channels))
        lidar_bp.set_attribute("points_per_second", str(self.cfg.lidar_points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(self.cfg.lidar_rotation_frequency))
        lidar_bp.set_attribute("upper_fov", str(self.cfg.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov", str(self.cfg.lidar_lower_fov))
        lidar = self.world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.7, z=2.25)), attach_to=self.vehicle)
        lidar.listen(lambda data: self._put_latest(self.lidar_q, data))
        self.actors.append(lidar)

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.cfg.camera_height))
        cam_bp.set_attribute("fov", str(self.cfg.camera_fov))
        front = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.7, z=1.9)), attach_to=self.vehicle)
        front.listen(lambda img: self._put_latest(self.front_q, self._img_to_array(img)))
        self.actors.append(front)
        rear = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.7, z=1.9), carla.Rotation(yaw=180)), attach_to=self.vehicle)
        rear.listen(lambda img: self._put_latest(self.rear_q, self._img_to_array(img)))
        self.actors.append(rear)

    def get_lidar_points(self) -> Optional[np.ndarray]:
        data = self._get_latest(self.lidar_q)
        if data is None: return None
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        d = np.linalg.norm(pts[:, :2], axis=1)
        return pts[(d > 1.0) & (d < self.cfg.lidar_range)]

    def get_front_image(self): return self._get_latest(self.front_q)
    def get_rear_image(self): return self._get_latest(self.rear_q)

    def destroy(self) -> None:
        for a in self.actors:
            try:
                if a.is_alive: a.destroy()
            except Exception: pass


class TerrainAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_points: Optional[np.ndarray] = None

    def update(self, pts: Optional[np.ndarray]) -> None:
        if pts is not None and len(pts) > 50:
            self.last_points = pts

    def local_risk(self, lx: float, ly: float, radius: float = 4.5) -> Tuple[float, float, float, float]:
        pts = self.last_points
        if pts is None or len(pts) < 30:
            return 0.12, 0.10, 0.0, 0.85
        dx = pts[:, 0] - lx; dy = pts[:, 1] - ly
        local = pts[(dx * dx + dy * dy) < radius * radius]
        if len(local) < 8:
            return 0.25, 0.20, 0.05, 0.70
        z = local[:, 2]
        z_span = float(np.percentile(z, 90) - np.percentile(z, 10))
        rough = clamp(z_span / 3.0, 0.0, 1.0)
        A = np.column_stack([local[:, 0], local[:, 1], np.ones(len(local))])
        try:
            coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
            slope_raw = math.sqrt(float(coeff[0] * coeff[0] + coeff[1] * coeff[1]))
        except Exception:
            slope_raw = 0.4
        slope = clamp(slope_raw / 0.9, 0.0, 1.0)

        ground = float(np.percentile(z, 20))
        high = local[z > ground + 1.20]
        obs = 0.0
        if len(high) > 0:
            hd = np.sqrt((high[:, 0] - lx) ** 2 + (high[:, 1] - ly) ** 2)
            obs = clamp(1.0 - float(np.min(hd)) / 6.0, 0.0, 1.0)
        flatness = clamp(1.0 - 0.65 * slope - 0.35 * rough - 0.25 * obs, 0.0, 1.0)
        return slope, rough, obs, flatness

    def forward_clearance(self) -> float:
        """High-object clearance only. Does NOT treat hill/ground points as an obstacle."""
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


class FlatGoalPlanner:
    def __init__(self, cfg: Config, terrain: TerrainAnalyzer):
        self.cfg = cfg; self.terrain = terrain
        self.best: Optional[Candidate] = None
        self.tick = 0
        self.memory: List[Tuple[float, float, float]] = []

    def add_memory(self, x: float, y: float) -> None:
        for i, (mx, my, w) in enumerate(self.memory):
            if math.hypot(x - mx, y - my) < self.cfg.stuck_memory_radius * 0.6:
                self.memory[i] = (mx, my, min(4.0, w + 1.0)); return
        self.memory.append((x, y, 1.0))
        self.memory = self.memory[-30:]

    def memory_penalty(self, wx: float, wy: float) -> float:
        p = 0.0
        for mx, my, w in self.memory:
            d = math.hypot(wx - mx, wy - my)
            p += w * math.exp(-(d * d) / (2.0 * self.cfg.stuck_memory_radius ** 2))
        return p

    def plan(self, st: VehicleState) -> Candidate:
        self.tick += 1
        if self.best is not None and self.tick % self.cfg.replan_every_ticks != 0:
            return self.best

        current_goal_dist = math.hypot(self.cfg.goal_x - st.x, self.cfg.goal_y - st.y)
        goal_heading = math.atan2(self.cfg.goal_y - st.y, self.cfg.goal_x - st.x)
        rel_goal = angle_wrap(goal_heading - st.yaw)
        center = clamp(rel_goal, -math.radians(80), math.radians(80))
        half = math.radians(self.cfg.fov_deg) / 2.0
        angles = np.linspace(center - half, center + half, self.cfg.candidate_count)
        candidates: List[Candidate] = []

        for dist in self.cfg.candidate_distances:
            for alpha in angles:
                lx = dist * math.cos(alpha)
                ly = dist * math.sin(alpha)
                if lx < 1.0:
                    continue
                wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
                new_gdist = math.hypot(self.cfg.goal_x - wx, self.cfg.goal_y - wy)
                progress = current_goal_dist - new_gdist
                heading_error = abs(angle_wrap(math.atan2(wy - st.y, wx - st.x) - st.yaw))
                slope, rough, obs, flat = self.terrain.local_risk(lx, ly, radius=5.0)
                mem = self.memory_penalty(wx, wy)
                cost = (
                    self.cfg.w_goal * new_gdist
                    - self.cfg.w_progress * progress
                    + self.cfg.w_heading * heading_error
                    + self.cfg.w_slope * slope
                    + self.cfg.w_rough * rough
                    + self.cfg.w_obstacle * obs
                    + self.cfg.w_memory * mem
                )
                reward = 120.0 * progress + 45.0 * flat - 25.0 * obs - 12.0 * slope - 8.0 * rough - 40.0 * mem
                candidates.append(Candidate(wx, wy, lx, ly, alpha, dist, progress, new_gdist, heading_error,
                                            slope, rough, obs, flat, mem, cost, reward))

        if not candidates:
            # direct-goal fallback
            dist = min(25.0, current_goal_dist)
            lx = dist * math.cos(rel_goal); ly = dist * math.sin(rel_goal)
            wx, wy = local_to_world(lx, ly, st.yaw, st.x, st.y)
            slope, rough, obs, flat = self.terrain.local_risk(lx, ly, radius=5.0)
            self.best = Candidate(wx, wy, lx, ly, rel_goal, dist, 1.0, current_goal_dist, abs(rel_goal),
                                  slope, rough, obs, flat, 0.0, current_goal_dist, 0.0)
            return self.best

        candidates.sort(key=lambda c: c.cost)
        best = candidates[0]
        # If all options are bad, still force a forward/goal-directed option.
        if max(c.progress for c in candidates) < 0.2:
            candidates.sort(key=lambda c: (abs(c.heading_error), c.goal_dist + 25.0 * c.obstacle_risk))
            best = candidates[0]
        self.best = best
        return best


class GoalController:
    def __init__(self, cfg: Config, terrain: TerrainAnalyzer):
        self.cfg = cfg; self.terrain = terrain
        self.prev_throttle = 0.0; self.prev_steer = 0.0

    def compute(self, st: VehicleState, cand: Candidate, goal_dist: float) -> ControlCmd:
        lx, ly = world_to_local(cand.wx - st.x, cand.wy - st.y, st.yaw)
        lookahead = max(5.0, math.hypot(lx, ly))
        curvature = 2.0 * ly / max(lookahead * lookahead, 1e-6)
        steer_raw = math.atan(self.cfg.wheel_base * curvature) * self.cfg.steer_gain
        steer_raw = clamp(steer_raw, -self.cfg.max_steer, self.cfg.max_steer)

        bad = clamp(0.65 * cand.slope + 0.35 * cand.roughness + 0.25 * cand.obstacle_risk, 0.0, 1.0)
        target_speed = (1.0 - bad) * self.cfg.target_speed_flat + bad * self.cfg.target_speed_bad
        if goal_dist < 30.0:
            target_speed = min(target_speed, self.cfg.target_speed_near_goal)
        target_speed *= max(0.45, 1.0 - 0.30 * abs(steer_raw) / self.cfg.max_steer)

        throttle_raw = self.cfg.kp_speed * (target_speed - st.speed)
        min_thr = self.cfg.min_throttle_hill if (abs(st.pitch_deg) > 7.0 or cand.slope > 0.38) else self.cfg.min_throttle_auto
        throttle_raw = clamp(throttle_raw, min_thr, self.cfg.max_throttle)

        brake = 0.0
        clearance = self.terrain.forward_clearance()
        if clearance < self.cfg.emergency_clearance:
            throttle_raw = 0.0; brake = self.cfg.max_brake
        elif clearance < self.cfg.danger_clearance and st.speed > 2.0:
            throttle_raw = min(throttle_raw, 0.30); brake = 0.15

        throttle = clamp(throttle_raw, self.prev_throttle - self.cfg.throttle_rate, self.prev_throttle + self.cfg.throttle_rate)
        steer = clamp(steer_raw, self.prev_steer - self.cfg.steer_rate, self.prev_steer + self.cfg.steer_rate)

        # No zero throttle while not at goal.
        if brake < 0.05 and goal_dist > self.cfg.goal_tolerance and st.speed < 0.5:
            throttle = max(throttle, min_thr)

        self.prev_throttle = throttle; self.prev_steer = steer
        return ControlCmd(throttle=throttle, steer=steer, brake=brake, reverse=False)

    def reset(self) -> None:
        self.prev_throttle = 0.0; self.prev_steer = 0.0


class StuckRecovery:
    IDLE = "AUTO"
    REVERSE = "RECOVERY_REVERSE"
    TURN = "RECOVERY_TURN"
    FORWARD = "RECOVERY_FORWARD_PUSH"

    def __init__(self, cfg: Config, planner: FlatGoalPlanner):
        self.cfg = cfg; self.planner = planner
        self.mode = self.IDLE; self.mode_start = 0.0
        self.hist: Deque[Tuple[float, float, float, float]] = deque(maxlen=400)
        self.stuck_start: Optional[float] = None
        self.recovery_steer = 0.55

    def force(self, t: float, st: VehicleState) -> None:
        self._start(t, st)

    def update(self, t: float, st: VehicleState, cmd: ControlCmd) -> None:
        self.hist.append((t, st.x, st.y, st.speed))
        if self.mode != self.IDLE:
            elapsed = t - self.mode_start
            if self.mode == self.REVERSE and elapsed > self.cfg.recovery_reverse_s:
                self.mode = self.TURN; self.mode_start = t
            elif self.mode == self.TURN and elapsed > self.cfg.recovery_turn_s:
                self.mode = self.FORWARD; self.mode_start = t
            elif self.mode == self.FORWARD and elapsed > self.cfg.recovery_forward_s:
                self.mode = self.IDLE; self.mode_start = t
            return

        if cmd.throttle < 0.35 or cmd.brake > 0.1:
            self.stuck_start = None; return
        if st.speed > self.cfg.stuck_speed_thresh:
            self.stuck_start = None; return
        old = None
        for h in self.hist:
            if t - h[0] >= self.cfg.stuck_time_s:
                old = h; break
        if old is None: return
        disp = math.hypot(st.x - old[1], st.y - old[2])
        if disp < self.cfg.stuck_disp_thresh:
            if self.stuck_start is None:
                self.stuck_start = t
            elif t - self.stuck_start > 0.5:
                self._start(t, st); self.stuck_start = None

    def _start(self, t: float, st: VehicleState) -> None:
        print(f"\n[RECOVERY] Vehicle stuck at ({st.x:.1f},{st.y:.1f}); using reverse + force push.")
        self.planner.add_memory(st.x, st.y)
        goal_heading = math.atan2(self.cfg.goal_y - st.y, self.cfg.goal_x - st.x)
        rel = angle_wrap(goal_heading - st.yaw)
        self.recovery_steer = -math.copysign(0.62, rel if abs(rel) > 0.1 else random.choice([-1.0, 1.0]))
        self.mode = self.REVERSE; self.mode_start = t

    def cmd(self) -> Optional[ControlCmd]:
        if self.mode == self.IDLE: return None
        if self.mode == self.REVERSE:
            return ControlCmd(throttle=self.cfg.recovery_reverse_throttle, steer=-self.recovery_steer, brake=0.0, reverse=True)
        if self.mode == self.TURN:
            return ControlCmd(throttle=0.65, steer=self.recovery_steer, brake=0.0, reverse=False)
        if self.mode == self.FORWARD:
            return ControlCmd(throttle=self.cfg.recovery_forward_throttle, steer=0.50 * self.recovery_steer, brake=0.0, reverse=False)
        return None


class LiveView:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("CARLA FORCE-MOVE Flat/Hill Goal Navigator")
        self.font = pygame.font.SysFont("monospace", 14)
        self.traj: Deque[Tuple[float, float]] = deque(maxlen=1200)
        self.cost: Deque[float] = deque(maxlen=300)
        self.reward: Deque[float] = deque(maxlen=300)
        self.front = None; self.rear = None

    def set_images(self, front, rear):
        if front is not None: self.front = front
        if rear is not None: self.rear = rear

    def draw(self, st: VehicleState, cand: Candidate, mode: str, goal_dist: float, cost: float, reward: float, cmd: ControlCmd):
        self.traj.append((st.x, st.y)); self.cost.append(cost); self.reward.append(reward)
        W, H = self.screen.get_size(); self.screen.fill((18, 18, 26))
        top_h = 270; panel_w = W // 3
        self._draw_image(self.front, 0, 0, panel_w, top_h, "Front camera")
        self._draw_image(self.rear, panel_w, 0, panel_w, top_h, "Rear camera")
        self._draw_text(2 * panel_w, 0, W - 2 * panel_w, top_h, st, cand, mode, goal_dist, cost, reward, cmd)
        self._draw_map(0, top_h, W // 2, H - top_h, cand)
        self._draw_graph(W // 2, top_h, W // 2, (H - top_h) // 2, list(self.cost), "NMPC cost")
        self._draw_graph(W // 2, top_h + (H - top_h) // 2, W // 2, (H - top_h) // 2, list(self.reward), "Reward")
        pygame.display.flip()

    def _draw_image(self, img, x, y, w, h, title):
        pygame.draw.rect(self.screen, (28, 28, 42), (x, y, w, h))
        pygame.draw.rect(self.screen, (45, 45, 60), (x, y, w, 22))
        self.screen.blit(self.font.render(title, True, (235,235,235)), (x+8, y+3))
        if img is not None:
            try:
                rr = (np.arange(h-22) * img.shape[0] / max(h-22,1)).astype(int).clip(0, img.shape[0]-1)
                cc = (np.arange(w) * img.shape[1] / max(w,1)).astype(int).clip(0, img.shape[1]-1)
                small = img[np.ix_(rr, cc)]
                self.screen.blit(pygame.surfarray.make_surface(small.swapaxes(0,1)), (x, y+22))
            except Exception: pass
        pygame.draw.rect(self.screen, (80,80,100), (x,y,w,h), 1)

    def _draw_text(self, x, y, w, h, st, cand, mode, goal_dist, cost, reward, cmd):
        pygame.draw.rect(self.screen, (28, 28, 42), (x, y, w, h))
        lines = [
            f"Mode       : {mode}",
            f"Dist goal  : {goal_dist:8.2f} m",
            f"Speed      : {st.speed:8.2f} m/s",
            f"Throttle   : {cmd.throttle:8.2f}",
            f"Steer      : {cmd.steer:+8.2f}",
            f"Brake      : {cmd.brake:8.2f}",
            f"Cost       : {cost:8.2f}",
            f"Reward     : {reward:8.2f}",
            f"Flatness   : {cand.flatness:8.2f}",
            f"Progress   : {cand.progress:8.2f}",
            f"Slope/Rgh  : {cand.slope:5.2f}/{cand.roughness:5.2f}",
            f"Pitch/Roll : {st.pitch_deg:5.1f}/{st.roll_deg:5.1f}",
        ]
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, (235,235,235)), (x+10, y+10+20*i))
        pygame.draw.rect(self.screen, (80,80,100), (x,y,w,h), 1)

    def _draw_map(self, x, y, w, h, cand):
        pygame.draw.rect(self.screen, (25,25,35), (x,y,w,h))
        pts = list(self.traj)
        if not pts: return
        xs = [p[0] for p in pts] + [self.cfg.spawn_x, self.cfg.goal_x, cand.wx]
        ys = [p[1] for p in pts] + [self.cfg.spawn_y, self.cfg.goal_y, cand.wy]
        span = max(max(xs)-min(xs), max(ys)-min(ys), 35.0)
        cx = 0.5*(max(xs)+min(xs)); cy = 0.5*(max(ys)+min(ys)); sc = 0.82*min(w,h)/span
        def conv(wx, wy): return int(x+w/2+(wx-cx)*sc), int(y+h/2-(wy-cy)*sc)
        if len(pts)>1: pygame.draw.lines(self.screen, (0,220,220), False, [conv(a,b) for a,b in pts], 2)
        sx, sy = conv(self.cfg.spawn_x, self.cfg.spawn_y); gx, gy = conv(self.cfg.goal_x, self.cfg.goal_y); vx, vy = conv(pts[-1][0], pts[-1][1]); cx2, cy2 = conv(cand.wx, cand.wy)
        pygame.draw.circle(self.screen, (255,220,90), (sx,sy), 6); pygame.draw.circle(self.screen, (255,255,255), (vx,vy), 6); pygame.draw.circle(self.screen, (80,255,120), (cx2,cy2), 5)
        pygame.draw.line(self.screen, (255,80,80), (gx-8,gy-8), (gx+8,gy+8), 2); pygame.draw.line(self.screen, (255,80,80), (gx-8,gy+8), (gx+8,gy-8), 2)
        self.screen.blit(self.font.render("World trajectory", True, (235,235,235)), (x+8,y+8))
        pygame.draw.rect(self.screen, (80,80,100), (x,y,w,h), 1)

    def _draw_graph(self, x, y, w, h, data, title):
        pygame.draw.rect(self.screen, (28,28,42), (x,y,w,h))
        self.screen.blit(self.font.render(title, True, (235,235,235)), (x+8,y+8))
        if len(data)>2:
            mn, mx = min(data), max(data); rng = max(mx-mn, 1e-6); pts=[]
            for i, v in enumerate(data):
                sx = x+10+int(i/(len(data)-1)*(w-20)); sy = y+h-10-int((v-mn)/rng*(h-35)); pts.append((sx,sy))
            pygame.draw.lines(self.screen, (255,110,120), False, pts, 2)
        pygame.draw.rect(self.screen, (80,80,100), (x,y,w,h), 1)


class NavigationSystem:
    def __init__(self, cfg: Config):
        pygame.init(); pygame.font.init()
        self.cfg = cfg
        self.client = None; self.world = None; self.vehicle = None; self.sensors = None
        self.original_settings = None
        self.terrain = TerrainAnalyzer(cfg)
        self.planner = FlatGoalPlanner(cfg, self.terrain)
        self.controller = GoalController(cfg, self.terrain)
        self.recovery = StuckRecovery(cfg, self.planner)
        self.logger = ResultLogger(cfg)
        self.view = LiveView(cfg)
        self.running = True
        self.start_wall = time.time()

    def connect(self):
        print(f"[CARLA] Connecting to {self.cfg.host}:{self.cfg.port}")
        self.client = carla.Client(self.cfg.host, self.cfg.port); self.client.set_timeout(self.cfg.timeout)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        if self.cfg.synchronous:
            settings = self.world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = self.cfg.dt
            self.world.apply_settings(settings); print("[CARLA] Synchronous mode ON")

    def spawn_vehicle(self):
        bp = self.world.get_blueprint_library().find(self.cfg.vehicle_blueprint)
        tf = carla.Transform(carla.Location(x=self.cfg.spawn_x, y=self.cfg.spawn_y, z=self.cfg.spawn_z),
                             carla.Rotation(pitch=self.cfg.spawn_pitch, yaw=self.cfg.spawn_yaw, roll=self.cfg.spawn_roll))
        self.vehicle = self.world.try_spawn_actor(bp, tf)
        if self.vehicle is None:
            for dz in [0.5, 1.0, 1.5, 2.0, 3.0]:
                tf.location.z = self.cfg.spawn_z + dz
                self.vehicle = self.world.try_spawn_actor(bp, tf)
                if self.vehicle is not None: break
        if self.vehicle is None:
            raise RuntimeError("Could not spawn vehicle. Delete old vehicle or adjust spawn position.")
        self.vehicle.set_autopilot(False)
        self.vehicle.set_simulate_physics(True)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False, manual_gear_shift=False))
        for _ in range(int(self.cfg.post_spawn_wait_s / self.cfg.dt)):
            self.world.tick()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False, manual_gear_shift=False))
        print(f"[SPAWN] Cybertruck at x={self.cfg.spawn_x:.2f}, y={self.cfg.spawn_y:.2f}, yaw={self.cfg.spawn_yaw:.2f}")
        self.sensors = SensorManager(self.world, self.vehicle, self.cfg)

    def get_state(self) -> VehicleState:
        tf = self.vehicle.get_transform(); vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)
        return VehicleState(float(tf.location.x), float(tf.location.y), float(tf.location.z), math.radians(float(tf.rotation.yaw)), float(tf.rotation.pitch), float(tf.rotation.roll), float(speed))

    def apply_control(self, cmd: ControlCmd):
        control = carla.VehicleControl(throttle=clamp(cmd.throttle,0,1), steer=clamp(cmd.steer,-1,1), brake=clamp(cmd.brake,0,1), reverse=bool(cmd.reverse), hand_brake=False, manual_gear_shift=False)
        self.vehicle.apply_control(control)

    def force_velocity_push(self, st: VehicleState, cmd: ControlCmd):
        """Last-resort push if the vehicle is commanded forward but CARLA wheels remain stuck."""
        if self.recovery.mode == StuckRecovery.FORWARD and st.speed < 0.35:
            vx = self.cfg.recovery_push_speed * math.cos(st.yaw)
            vy = self.cfg.recovery_push_speed * math.sin(st.yaw)
            try:
                self.vehicle.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            except Exception:
                pass

    def run(self):
        self.connect(); self.spawn_vehicle()
        tick = 0
        try:
            while self.running:
                tick += 1
                if self.cfg.synchronous: self.world.tick()
                else: self.world.wait_for_tick()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: self.running = False
                        elif event.key == pygame.K_r: self.recovery.force(time.time()-self.start_wall, self.get_state())

                t = time.time() - self.start_wall
                st = self.get_state()
                goal_dist = math.hypot(self.cfg.goal_x-st.x, self.cfg.goal_y-st.y)
                pts = self.sensors.get_lidar_points(); self.terrain.update(pts)
                if tick % 10 == 0: self.logger.log_terrain_points(t, st, pts)

                cand = self.planner.plan(st)
                cmd = self.controller.compute(st, cand, goal_dist)
                self.recovery.update(t, st, cmd)
                rec_cmd = self.recovery.cmd()
                mode = self.recovery.mode
                if rec_cmd is not None:
                    cmd = rec_cmd; self.controller.reset()

                if goal_dist < self.cfg.goal_tolerance:
                    print("\n[NAV] Goal reached.")
                    cmd = ControlCmd(0.0, 0.0, 1.0, False)
                    self.apply_control(cmd); self.running = False
                else:
                    self.apply_control(cmd)
                    self.force_velocity_push(st, cmd)

                cost = cand.cost + 0.3*cmd.throttle*cmd.throttle + 0.8*cmd.steer*cmd.steer + 0.4*cmd.brake*cmd.brake
                reward = cand.reward + 8.0*st.speed - 0.35*goal_dist
                self.logger.log(t, st, cmd, cand, goal_dist, mode, cost, reward)

                if tick % self.cfg.print_every == 0:
                    print(f"\r[NAV] t={t:6.1f}s dist={goal_dist:7.2f}m speed={st.speed:5.2f}m/s thr={cmd.throttle:4.2f} steer={cmd.steer:+5.2f} brake={cmd.brake:4.2f} mode={mode:>22s} flat={cand.flatness:4.2f} prog={cand.progress:6.2f} cost={cost:8.2f}", end="", flush=True)

                if tick % self.cfg.viz_every == 0:
                    self.view.set_images(self.sensors.get_front_image(), self.sensors.get_rear_image())
                    self.view.draw(st, cand, mode, goal_dist, cost, reward, cmd)

                if t > self.cfg.max_run_time_s:
                    print("\n[NAV] Max run time reached."); self.running = False
        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n[CLEANUP] Stop vehicle and save results")
        try:
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))
        except Exception: pass
        try:
            if self.sensors: self.sensors.destroy()
        except Exception: pass
        try:
            if self.vehicle and self.vehicle.is_alive: self.vehicle.destroy()
        except Exception: pass
        try:
            if self.world and self.original_settings: self.world.apply_settings(self.original_settings)
        except Exception: pass
        self.logger.finalize()
        pygame.quit()


def main():
    nav = NavigationSystem(CFG)
    nav.run()


if __name__ == "__main__":
    main()
