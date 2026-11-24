#!/usr/bin/env python3

import os
import random
import sys
import time
import subprocess
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class RandomCubeSpawner(Node):
    def __init__(self):
        super().__init__("random_cube_spawner")
        self.get_logger().info("Initializing Random Cube Spawner for Gazebo Harmonic...")

        pkg_share = get_package_share_directory("ur10e_perception_workcell")
        self.model_sdf_path = os.path.join(pkg_share, "models", "red_cube", "model.sdf")

        if not os.path.exists(self.model_sdf_path):
            self.get_logger().error(f"Cube model file not found: {self.model_sdf_path}")
            sys.exit(1)

        # Wait briefly for Gazebo to be responsive
        time.sleep(3.0)
        self.spawn_cube()

    def spawn_cube(self):
        # Generate random position on tabletop
        rand_x = round(random.uniform(0.78, 0.92), 3)
        rand_y = round(random.uniform(-0.18, 0.18), 3)
        rand_z = 0.825

        # Fixed orientation as required: Yaw = 0.0 rad
        rand_yaw = 0.0

        self.get_logger().info(
            f"🎲 Spawning red_cube at random table pose: X={rand_x:.3f} m, Y={rand_y:.3f} m, Z={rand_z:.3f} m, Yaw={rand_yaw:.3f} rad"
        )

        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", "table_scene",
            "-file", self.model_sdf_path,
            "-name", "red_cube",
            "-x", str(rand_x),
            "-y", str(rand_y),
            "-z", str(rand_z),
            "-Y", str(rand_yaw),
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
            if res.returncode == 0:
                self.get_logger().info("✅ red_cube spawned successfully in Gazebo!")
            else:
                self.get_logger().warning(f"Spawner output: {res.stdout} {res.stderr}")
        except Exception as e:
            self.get_logger().error(f"Failed to spawn cube: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RandomCubeSpawner()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
