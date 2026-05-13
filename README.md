# Autonomous Mapless Navigation in Unstructured Terrain

This repository contains experimental controller implementations for autonomous mapless navigation in uneven off-road terrain using the CARLA simulator.

The project focuses on terrain-aware navigation without relying on predefined maps, lane structures, or classical global path planners. It explores iterative controller designs for obstacle avoidance, traversability reasoning, trajectory selection, and recovery behavior in highly irregular environments.

Unlike standard navigation stacks that depend on planners such as A*, RRT, or similar graph-search methods, the `tcn`-based files in this repository implement a custom-built planner architecture designed specifically for terrain-aware mapless navigation.

The controller evolution and research background are described primarily in Chapters 5, 6, and 7 of the accompanying report.

## Project Status

This repository contains research-oriented and experimental controller code.

The current implementations are **not final** and still require:
- further refinement
- parameter tuning
- stability improvements
- recovery behavior improvements
- broader validation across terrain types

This project is intended for experimentation, development, and research in autonomous off-road navigation.

## Key Features

- Mapless autonomous navigation in unstructured terrain
- Terrain-aware control logic for uneven off-road environments
- Custom TCN-based planner implementations
- Reactive LiDAR-based navigation
- VSGP-based traversability reasoning
- Recovery-aware navigation behavior
- Multi-vehicle formation navigation experiments
- Iterative controller development and comparison

## Controller Progression

The repository reflects multiple stages of controller development:

1. **Reactive LiDAR baseline**  
   A local clearance-based navigation approach for initial motion and obstacle avoidance.

2. **VSGP-based mapless navigation**  
   Terrain estimation is handled through probabilistic modeling to improve traversability reasoning.

3. **Custom TCN-based planning**  
   The `tcn` files contain a custom-built planner rather than a standard native A* / RRT / PRM pipeline.

4. **Recovery-enhanced navigation**  
   Additional logic was added to help the vehicle escape stuck or unstable states.

5. **Formation navigation experiments**  
   Multi-vehicle coordination and formation-oriented traversal were explored as part of later iterations.

## Repository Structure

```text
legacy/                          Older versions kept for reference
tcn planner/                     Custom TCN-based planner experiments
main_vsgp_tcn_single_vehicle_stable_main.py
main_multiple_vehicle_formation_vsgp_tcn.py
LICENSE
.gitignore
