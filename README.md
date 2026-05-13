# Autonomous Mapless Navigation in Unstructured Terrain

Experimental terrain-aware autonomous navigation framework for off-road traversal in CARLA using VSGP perception, custom Topology-Inspired Corridor Navigation (TCN), MPPI trajectory optimization, semantic terrain reasoning, IMU stability analysis, and persistent terrain memory.

This repository focuses on mapless autonomous navigation in highly irregular terrain without relying on:
- predefined HD maps
- lane graphs
- waypoint routes
- classical graph-search planners such as A*, RRT, or PRM

The system was developed through iterative experimentation and controller evolution during research into autonomous off-road navigation in unstructured terrain.

---

# Project Status

This repository contains research-oriented experimental code.

The current implementations are **not final production-ready systems** and still require:
- additional tuning
- stability refinement
- broader terrain validation
- recovery optimization
- planner refinement
- controller modularization

The project is intended primarily for:
- research
- experimentation
- controller prototyping
- terrain-aware navigation studies
- autonomous off-road simulation

---

# Core Navigation Architecture

The current primary implementation:

```text
main_vsgp_tcn_single_vehicle_stable_main.py

System Pipeline :

LiDAR ──────────────┬──► VSGP Perception ──────────► Traversability Grid
                    │         │
                    │         └──► Slope / Roughness / Void-risk Maps
                    │
                    └──► ElevationGridBuilder ──► TCN Planner
                              │                      │
                    RGB ──────┤                      ▼
                              │             Corridor Graph Navigation
                    Semantic ─┤
                              │
                    IMU ──────┴──► Stability Analysis

                                ▼

                    Fused Traversability Estimation
                                │
                                ▼

                      SlipAware-MPPI Controller
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
        Reactive Safety    Anti-Tip Layer   Recovery Logic
                                │
                                ▼
                       Control Arbitration
                                │
                                ▼
                         Vehicle Control
                                │
                                ▼
                              CARLA
