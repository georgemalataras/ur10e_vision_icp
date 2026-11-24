#!/usr/bin/env bash
set -e
export DISPLAY=:10
source /opt/ros/jazzy/setup.bash
source /home/gerogemal/ros2_ws/install/setup.bash

OUTPUT_DIR="/mnt/c/Users/admin/Desktop/ur10e_cube_dataset"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.jpg

echo "[INFO] Starting optimized 3D scanning routine with collect_dataset.launch.py..."
ros2 launch ur10e_perception_workcell collect_dataset.launch.py script_delay:=10.0 gazebo_gui:=true > /tmp/collect.log 2>&1 &
LAUNCH_PID=$!

WAIT_TIME=0
PREV_COUNT=-1
while kill -0 $LAUNCH_PID 2>/dev/null; do
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    COUNT=$(ls -1 "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ "$COUNT" -ne "$PREV_COUNT" ]; then
        echo "[INFO] Progress: $COUNT / 100 verified frames captured..."
        PREV_COUNT=$COUNT
    fi
    if [ "$WAIT_TIME" -ge 500 ]; then
        echo "[WARN] Timeout reached (500s)."
        break
    fi
done

echo "[INFO] Scan complete and robot returned to HOME. Cleaning up simulation..."
kill -SIGINT $LAUNCH_PID 2>/dev/null || true
sleep 2
pkill -9 -f "collect_dataset" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "move_group" 2>/dev/null || true
pkill -9 -f "ros_gz" 2>/dev/null || true

COUNT=$(ls -1 "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "============================================================"
echo "[INFO] Dataset collection finished!"
echo "[INFO] Total verified images saved in $OUTPUT_DIR: $COUNT"
echo "============================================================"
