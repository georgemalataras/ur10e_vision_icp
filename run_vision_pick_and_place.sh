#!/usr/bin/env bash
set -e
export DISPLAY=${DISPLAY:-:10}
source /opt/ros/jazzy/setup.bash
source /home/gerogemal/ros2_ws/install/setup.bash

# Clean up any lingering zombie Gazebo or ROS processes before starting
echo "[INFO] Cleaning up any previous Gazebo / ROS 2 instances..."
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "move_group" 2>/dev/null || true
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
    pkill -9 -f "perception_pick_and_place" 2>/dev/null || true
    pkill -9 -f "collect_dataset" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo " Starting UR10e Vision-Guided Pick and Place (YOLO11) ...    "
echo "============================================================"
echo " - Spawning random cube on table with fixed orientation"
echo " - Moving to READY observation pose"
echo " - Detecting cube with YOLO11 model (best.pt)"
echo " - Calculating 3D position via ray-plane intersection"
echo " - Executing dynamic IK and MoveIt pick-and-place sequence"
echo "============================================================"

ros2 launch ur10e_perception_workcell workcell_complete.launch.py gazebo_gui:=true
