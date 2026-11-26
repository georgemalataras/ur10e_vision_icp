# UR10e Vision & 3D ICP Perception Workcell

Self-contained ROS 2 (Jazzy / Lyrical) perception and manipulation package for the **UR10e + Robotiq 2F-85** workcell in **Gazebo Harmonic**, **MoveIt 2**, and **RViz2**.

Implements a two-stage 6D pose estimation pipeline (`X, Y, Z, Yaw`) using an eye-in-hand RGB-D camera and a 3D CAD model (`red_cube.stl`), followed by autonomous grasp, lift, and place execution.

---

## Perception Pipeline Architecture

### Stage 1 — Full Unfiltered PointCloud Brute-Force + ICP (`icp_mode:=unfiltered`)
- Captures the complete unfiltered `/wrist_camera/points` (`PointCloud2`, table + cube + background).
- Runs a 3D coarse **Brute-Force** grid + yaw search with threshold gating (`COARSE_THRESHOLD_MM = 11.0 mm`) to reject empty-table scenes when the cube is outside the camera FOV.
- Refines the 6D pose via **Point-to-Plane ICP** directly against the full scene cloud.

### Stage 2 — Instance Segmentation + 3D Mask Filtering + Direct ICP (`icp_mode:=segmented`, default)
1. **Step 1/3 (Full FOV Capture):** Robot moves to angled observation pose (`OBSERVE_CENTER`) and publishes the full camera FOV `PointCloud2` to RViz2 (`/icp/target_cloud`).
2. **Step 2/3 (Instance Segmentation & 3D Masking):** Extracts the 2D cube segmentation mask $M(u, v)$, projects 3D points via camera intrinsics $K$, and filters **only** the 3D points belonging to the cube (removing 100% of table/background points, visualized with a 3D bounding box wireframe).
3. **Step 3/3 (Direct Pure ICP):** Initializes the CAD model (`red_cube.stl`) directly at the segmented cube centroid (no table brute-force sweep) and runs **Symmetric Point-to-Plane ICP** with live RViz2 correspondence visualization until `LOCKED` (`RMSE < 8.0 mm`).
4. **Pick & Place:** Updates MoveIt 2 `PlanningScene`, rotates the gripper fingers to match `cube_yaw`, grasps, attaches, lifts, places, and detaches the cube.

### Stage 3 — Multi-Object 6D Pose Estimation with NVIDIA FoundationPose (`v3.0.0`)
- Spawns 6 CAD parts (`Bracket 23-040-6`, `Red Cube`, `Lindapter`, `Clamping Bar`, `Square Nut`, `T-Connector`) in a 2×3 tabletop layout.
- Moves UR10e to an elevated angled observation pose (`OBSERVE_MULTI`) covering all 6 parts simultaneously within the wrist camera FOV.
- Runs **NVIDIA FoundationPose** (Scorer & Refiner networks, `nvdiffrast`, `PyTorch3D`) via a persistent GPU inference server (`foundationpose_server.py`) bridged to ROS 2 (`foundationpose_pick_and_place.py`) using `/wrist_camera/image`, `/wrist_camera/depth_image`, and `/wrist_camera/camera_info`.
- Displays all camera topics, always-visible 3D CAD meshes, 3D wireframe bounding boxes, labels, and per-object TF frames in RViz2.
- Opens an interactive 2×3 tabletop UI (with 3D CAD thumbnails and names) prompting *"Which part do you want to pick up?"*, computes a shape-agnostic 3D cross-section grasp with exact Robotiq 2F-85 linkage kinematics, lifts the part, holds it aloft for 2.5 s, and places it back at its original position.

---

## Repository Structure & Git Workflow

- `main`: Production-ready releases (`v1.0.0`, `v2.0.0`, `v2.0.1`, `v3.0.0`).
- `dev`: Integration branch where verified features are merged.
- `feature/1-icp-pose-estimation-lift`: Stage 1 implementation (Full-cloud Brute-Force + ICP).
- `feature/2-instance-segmentation-icp-lift`: Stage 2 implementation (2D Instance Segmentation $\rightarrow$ 3D PointCloud Masking $\rightarrow$ Direct ICP).
- `fix/3-cube-always-visible`: Persistent RViz cube visual during pick-and-place.
- `feature/4-foundationpose-multiobject-cad`: Stage 3 implementation (NVIDIA FoundationPose multi-object 6D estimation, interactive UI, 3D cross-section grasping).

### Package Contents
- `ur10e_perception_workcell/scripts/` — [`perception_pick_and_place.py`](ur10e_perception_workcell/scripts/perception_pick_and_place.py), [`publish_planning_scene.py`](ur10e_perception_workcell/scripts/publish_planning_scene.py), [`spawn_random_cube.py`](ur10e_perception_workcell/scripts/spawn_random_cube.py)
- `ur10e_perception_workcell/launch/` — [`workcell_complete.launch.py`](ur10e_perception_workcell/launch/workcell_complete.launch.py), [`sim.launch.py`](ur10e_perception_workcell/launch/sim.launch.py), [`move_group.launch.py`](ur10e_perception_workcell/launch/move_group.launch.py)
- `ur10e_perception_workcell/models/` — 3D CAD mesh (`red_cube/meshes/red_cube.stl`) and YOLO weights (`weights/best.pt`)
- `ur10e_perception_workcell/config/` — MoveIt 2, `ros2_control`, and RViz2 (`workcell.rviz`) configurations
- `run_vision_pick_and_place.sh` — One-command launcher with clean process management

---

## Build & Run

### 1. Build Workspace
```bash
cd /home/lms/ur10e_perception_workcell
source /opt/ros/lyrical/setup.bash   # or /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select ur10e_perception_workcell
source install/setup.bash
```

### 2. Run Complete Workcell (Stage 2 Default)
```bash
cd /home/lms/ur10e_perception_workcell
./run_vision_pick_and_place.sh
```

Or directly via `ros2 launch`:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch ur10e_perception_workcell workcell_complete.launch.py gazebo_gui:=true rviz:=true
```

### 3. Useful Options
- **Custom cube position:**
  ```bash
  ros2 launch ur10e_perception_workcell workcell_complete.launch.py cube_x:=0.90 cube_y:=0.05
  ```
- **Observation pose only:**
  ```bash
  ros2 launch ur10e_perception_workcell workcell_complete.launch.py observe_only:=true
  ```
- **Run Stage 1 (Unfiltered Brute-Force + ICP) standalone:**
  ```bash
  ros2 run ur10e_perception_workcell perception_pick_and_place.py --ros-args -p icp_mode:=unfiltered
  ```
