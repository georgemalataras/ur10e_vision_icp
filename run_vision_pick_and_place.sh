#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DISPLAY=${DISPLAY:-:0}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [ -f "/opt/ros/lyrical/setup.bash" ]; then
    source /opt/ros/lyrical/setup.bash
elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
fi
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
elif [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
fi

# Clean up any lingering zombie Gazebo or ROS processes before starting
echo "[INFO] Cleaning up any previous Gazebo / ROS 2 instances..."
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "move_group" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "perception_pick_and_place" 2>/dev/null || true
pkill -9 -f "collect_dataset" 2>/dev/null || true
sleep 1

# Setup trap to clean up child processes on exit/Ctrl+C
cleanup() {
    echo ""
    echo "[INFO] Shutting down simulation and cleaning up background processes..."
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
    pkill -9 -f "move_group" 2>/dev/null || true
    pkill -9 -f "rviz2" 2>/dev/null || true
    pkill -9 -f "perception_pick_and_place" 2>/dev/null || true
    pkill -9 -f "collect_dataset" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo " Starting UR10e Vision-Guided Pick and Place + RViz2 ...    "
echo "============================================================"
echo " - Spawning random cube on table with fixed orientation"
echo " - Bridging RGB, Depth, CameraInfo, and 3D PointCloud2"
echo " - Launching RViz2 with PointCloud2 & RGB-D displays"
echo "============================================================"

ros2 launch ur10e_perception_workcell workcell_complete.launch.py gazebo_gui:=true rviz:=true "$@"
