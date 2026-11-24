# ur10e_perception_workcell

Self-contained ROS 2 Jazzy package for the UR10e eye-in-hand perception workcell (Gazebo Harmonic + MoveIt 2 + YOLO11 pick-and-place).

## Contents
- `ur10e_perception_workcell/` — Full ROS 2 package:
  - `launch/` (`workcell_complete.launch.py`, `sim.launch.py`, `move_group.launch.py`, `collect_dataset.launch.py`)
  - `scripts/` (`perception_pick_and_place.py`, `spawn_random_cube.py`, `collect_dataset.py`)
  - `models/weights/best.pt` — Trained YOLO11 weights
  - `models/red_cube/model.sdf` — Red cube Gazebo model
  - `worlds/table_scene.sdf` — Gazebo world
  - `urdf/ur10e_workcell.urdf.xacro` & `srdf/ur10e_workcell.srdf`
  - `config/` — MoveIt 2, kinematics, joint limits, controllers & Gazebo GUI configs
- `run_vision_pick_and_place.sh` — Helper script to launch the complete vision-guided pick-and-place
- `run_dataset_collection.sh` — Helper script for dataset collection

## Setup (Ubuntu 24.04 / ROS 2 Jazzy)

1. Install required ROS 2 Jazzy packages (if not already installed):
   ```bash
   sudo apt update
   sudo apt install -y \
     ros-jazzy-ur-description \
     ros-jazzy-ur-controllers \
     ros-jazzy-ur-simulation-gz \
     ros-jazzy-robotiq-description \
     ros-jazzy-moveit \
     ros-jazzy-ros-gz-sim \
     ros-jazzy-ros-gz-bridge \
     ros-jazzy-cv-bridge \
     ros-jazzy-rmw-cyclonedds-cpp \
     ros-jazzy-xacro \
     python3-scipy python3-opencv python3-numpy
   ```

2. Install YOLO11 (`ultralytics`) in your Python environment (if not already installed):
   ```bash
   pip install ultralytics
   ```

3. Copy `ur10e_perception_workcell/` into your workspace `~/ros2_ws/src/` and build with `--symlink-install` (important so `models/weights/best.pt` resolves automatically relative to the script):
   ```bash
   mkdir -p ~/ros2_ws/src
   cp -r ur10e_perception_workcell ~/ros2_ws/src/
   cd ~/ros2_ws
   source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install --packages-select ur10e_perception_workcell
   source install/setup.bash
   ```

## Run
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch ur10e_perception_workcell workcell_complete.launch.py
```
