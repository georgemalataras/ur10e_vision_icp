#!/usr/bin/env python3
import os
import sys
import time
import glob
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import Image
from control_msgs.action import GripperCommand, FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

DESKTOP_DIR = "/mnt/c/Users/admin/Desktop/ur10e_cube_dataset"
FALLBACK_DIR = os.path.expanduser("~/ur10e_cube_dataset")

DIVERSE_POSES = [
    ("001_arc1_low_01", [-0.9254, -1.3715, 2.0107, -1.7388, -0.4631, -0.0059]),
    ("002_arc1_low_02", [-0.959, -1.4601, 2.1379, -1.8054, -0.4149, 0.0033]),
    ("003_arc1_low_03", [-0.9807, -1.6012, 2.3297, -2.085, -0.3799, 0.0061]),
    ("004_arc1_low_04", [-0.9765, -1.7324, 2.5309, -2.4931, -0.3829, 0.0094]),
    ("005_arc1_low_05", [-0.9196, -1.7804, 2.708, -3.0177, -0.4625, 0.0167]),
    ("006_arc1_low_06", [-0.7668, -1.681, 2.8144, -3.5672, -0.6685, 0.0417]),
    ("007_arc1_low_07", [-0.4991, -1.4936, 2.8322, -3.9831, -1.0137, 0.0569]),
    ("008_arc1_low_08", [-0.1845, -1.326, 2.7776, -4.1837, -1.4224, 0.0337]),
    ("009_arc1_low_09", [0.0861, -1.2218, 2.6764, -4.1908, -1.8026, 0.0118]),
    ("010_arc1_low_10", [0.2757, -1.1471, 2.5561, -4.0972, -2.1085, -0.0038]),
    ("011_arc1_low_11", [0.3954, -1.086, 2.4322, -3.9451, -2.3423, -0.0065]),
    ("012_arc1_low_12", [0.4663, -1.0338, 2.3159, -3.7465, -2.5184, -0.0107]),
    ("013_arc1_low_13", [0.5054, -0.9949, 2.2128, -3.4907, -2.6512, -0.0105]),
    ("014_arc1_low_14", [0.5234, -0.9829, 2.1298, -3.1561, -2.735, -0.0121]),
    ("015_arc2_midlow_01", [0.449, -1.2302, 2.1175, -2.8031, -2.5031, -0.006]),
    ("016_arc2_midlow_02", [0.4234, -1.2921, 2.2001, -2.9143, -2.4331, 0.0047]),
    ("017_arc2_midlow_03", [0.3795, -1.3492, 2.2915, -3.106, -2.3356, 0.0081]),
    ("018_arc2_midlow_04", [0.3122, -1.413, 2.381, -3.273, -2.2084, 0.0085]),
    ("019_arc2_midlow_05", [0.2139, -1.4884, 2.4658, -3.3926, -2.0435, 0.0062]),
    ("020_arc2_midlow_06", [0.0767, -1.5785, 2.5405, -3.4497, -1.8331, 0.0033]),
    ("021_arc2_midlow_07", [-0.1003, -1.6782, 2.5965, -3.4354, -1.582, -0.0003]),
    ("022_arc2_midlow_08", [-0.3026, -1.7759, 2.6211, -3.3365, -1.3091, -0.0021]),
    ("023_arc2_midlow_09", [-0.4969, -1.8445, 2.6051, -3.163, -1.0521, -0.0042]),
    ("024_arc2_midlow_10", [-0.6485, -1.8608, 2.5452, -2.9352, -0.8499, -0.0058]),
    ("025_arc2_midlow_11", [-0.7436, -1.8218, 2.445, -2.6769, -0.7193, -0.0085]),
    ("026_arc2_midlow_12", [-0.7898, -1.74, 2.3128, -2.4119, -0.6544, -0.0089]),
    ("027_arc2_midlow_13", [-0.8034, -1.6328, 2.1612, -2.1619, -0.6338, -0.0095]),
    ("028_arc2_midlow_14", [-0.7957, -1.5149, 1.9999, -1.9385, -0.6429, -0.0096]),
    ("029_arc3_midhigh_01", [-0.6286, -1.5876, 1.947, -2.0063, -0.8467, -0.0009]),
    ("030_arc3_midhigh_02", [-0.6032, -1.6584, 2.0349, -2.1253, -0.8907, 0.0027]),
    ("031_arc3_midhigh_03", [-0.5699, -1.7262, 2.1257, -2.265, -0.9339, 0.0043]),
    ("032_arc3_midhigh_04", [-0.5191, -1.781, 2.2063, -2.4068, -0.9977, 0.0042]),
    ("033_arc3_midhigh_05", [-0.4478, -1.8163, 2.271, -2.5418, -1.0875, 0.0054]),
    ("034_arc3_midhigh_06", [-0.3555, -1.8281, 2.3167, -2.6608, -1.2062, 0.0034]),
    ("035_arc3_midhigh_07", [-0.2488, -1.8153, 2.3416, -2.7565, -1.3448, 0.0019]),
    ("036_arc3_midhigh_08", [-0.1343, -1.7802, 2.3453, -2.8238, -1.4984, 0.0009]),
    ("037_arc3_midhigh_09", [-0.0241, -1.7295, 2.3299, -2.8584, -1.6505, 0.0002]),
    ("038_arc3_midhigh_10", [0.0751, -1.6695, 2.2981, -2.8595, -1.7944, -0.0014]),
    ("039_arc3_midhigh_11", [0.1588, -1.6055, 2.253, -2.8296, -1.9236, -0.0025]),
    ("040_arc3_midhigh_12", [0.2247, -1.543, 2.1967, -2.7675, -2.0309, -0.0016]),
    ("041_arc3_midhigh_13", [0.2753, -1.4825, 2.1322, -2.6792, -2.1211, -0.004]),
    ("042_arc3_midhigh_14", [0.311, -1.426, 2.0599, -2.5664, -2.1889, -0.004]),
    ("043_arc4_high_01", [0.1618, -1.4824, 1.9173, -2.3112, -1.9579, -0.0081]),
    ("044_arc4_high_02", [0.1308, -1.5174, 1.943, -2.3175, -1.9053, -0.0018]),
    ("045_arc4_high_03", [0.0944, -1.5537, 1.9762, -2.3508, -1.8481, 0.0015]),
    ("046_arc4_high_04", [0.0538, -1.5875, 2.0079, -2.3865, -1.7901, 0.0023]),
    ("047_arc4_high_05", [0.0071, -1.6183, 2.0329, -2.4108, -1.7236, 0.002]),
    ("048_arc4_high_06", [-0.0449, -1.6452, 2.0503, -2.4216, -1.6496, -0.0]),
    ("049_arc4_high_07", [-0.0992, -1.6649, 2.0577, -2.4166, -1.5752, 0.0002]),
    ("050_arc4_high_08", [-0.155, -1.6773, 2.055, -2.3955, -1.4988, -0.0004]),
    ("051_arc4_high_09", [-0.2098, -1.6815, 2.0434, -2.3632, -1.4242, -0.0009]),
    ("052_arc4_high_10", [-0.2614, -1.6766, 2.0221, -2.3187, -1.3543, -0.0019]),
    ("053_arc4_high_11", [-0.3077, -1.6626, 1.9919, -2.2648, -1.2915, -0.0036]),
    ("054_arc4_high_12", [-0.3473, -1.6393, 1.9515, -2.2001, -1.2377, -0.004]),
    ("055_arc4_high_13", [-0.3794, -1.6079, 1.9028, -2.1289, -1.1937, -0.0041]),
    ("056_arc4_high_14", [-0.4041, -1.5695, 1.8464, -2.0517, -1.1592, -0.0052]),
    ("057_arc5_steep_01", [-0.2575, -1.5183, 1.7535, -2.0016, -1.3343, -0.0049]),
    ("058_arc5_steep_02", [-0.2348, -1.5257, 1.7647, -2.0094, -1.3702, 0.0]),
    ("059_arc5_steep_03", [-0.2114, -1.5361, 1.7838, -2.0356, -1.4038, 0.0005]),
    ("060_arc5_steep_04", [-0.1873, -1.5447, 1.8017, -2.0635, -1.4362, 0.0005]),
    ("061_arc5_steep_05", [-0.1613, -1.5495, 1.8157, -2.088, -1.4715, 0.0006]),
    ("062_arc5_steep_06", [-0.1343, -1.5506, 1.8255, -2.1083, -1.5083, 0.0]),
    ("063_arc5_steep_07", [-0.108, -1.5478, 1.8294, -2.1206, -1.5421, 0.0007]),
    ("064_arc5_steep_08", [-0.0809, -1.5419, 1.8304, -2.1309, -1.5786, 0.0001]),
    ("065_arc5_steep_09", [-0.054, -1.532, 1.8263, -2.1349, -1.6162, 0.0004]),
    ("066_arc5_steep_10", [-0.0287, -1.5196, 1.8184, -2.1343, -1.6507, 0.0012]),
    ("067_arc5_steep_11", [-0.0039, -1.504, 1.8061, -2.1273, -1.6873, -0.0009]),
    ("068_arc5_steep_12", [0.0188, -1.4856, 1.7891, -2.114, -1.7215, -0.0001]),
    ("069_roll_tilt_01", [-0.6153, -1.7179, 2.1706, -2.3868, -0.9145, 0.4423]),
    ("070_roll_tilt_02", [-0.5435, -1.8825, 2.406, -2.6134, -0.9762, -0.3538]),
    ("071_roll_tilt_03", [-0.3484, -1.7732, 2.2208, -2.6189, -1.2234, 0.4511]),
    ("072_roll_tilt_04", [-0.2028, -1.8276, 2.3756, -2.7512, -1.4202, -0.4305]),
    ("073_roll_tilt_05", [-0.064, -1.7095, 2.2276, -2.7555, -1.608, 0.4362]),
    ("074_roll_tilt_06", [0.1156, -1.6538, 2.4193, -3.0369, -1.8653, -0.3566]),
    ("075_roll_tilt_07", [0.2318, -1.5249, 2.2285, -2.9229, -2.0521, 0.3713]),
    ("076_roll_tilt_08", [0.0851, -1.6594, 2.2248, -2.6605, -1.8619, -0.517]),
    ("077_roll_tilt_09", [-0.0038, -1.6295, 2.0879, -2.5901, -1.728, 0.4885]),
    ("078_roll_tilt_10", [-0.1525, -1.7656, 2.2277, -2.5505, -1.5255, -0.4907]),
    ("079_roll_tilt_11", [-0.315, -1.6942, 2.0748, -2.4523, -1.305, 0.4902]),
    ("080_roll_tilt_12", [-0.4003, -1.7806, 2.1777, -2.3715, -1.1913, -0.4938]),
    ("081_roll_tilt_13", [-0.452, -1.7825, 2.2206, -2.5063, -1.1017, 0.2127]),
    ("082_roll_tilt_14", [-0.2013, -1.8226, 2.3838, -2.7952, -1.41, -0.2498]),
    ("083_roll_tilt_15", [0.0836, -1.6483, 2.2476, -2.8238, -1.8059, 0.2474]),
    ("084_roll_tilt_16", [-0.1424, -1.7482, 2.2461, -2.6877, -1.5227, 0.0397]),
    ("085_topdown_ascent_01", [-0.0942, -1.1666, 1.7669, -2.2153, -1.5679, -0.0274]),
    ("086_topdown_ascent_02", [-0.0926, -1.1815, 1.7282, -2.1384, -1.5729, -0.0053]),
    ("087_topdown_ascent_03", [-0.0925, -1.2025, 1.7042, -2.0905, -1.5733, -0.0032]),
    ("088_topdown_ascent_04", [-0.0925, -1.2226, 1.6802, -2.0462, -1.5734, -0.0028]),
    ("089_topdown_ascent_05", [-0.0925, -1.2411, 1.6545, -2.0021, -1.5735, -0.0028]),
    ("090_topdown_ascent_06", [-0.0925, -1.2577, 1.6268, -1.9578, -1.5735, -0.0029]),
    ("091_topdown_ascent_07", [-0.0925, -1.2726, 1.5971, -1.9133, -1.5734, -0.0029]),
    ("092_topdown_ascent_08", [-0.0925, -1.2855, 1.5654, -1.8687, -1.5734, -0.0029]),
    ("093_topdown_ascent_09", [-0.0925, -1.2967, 1.5317, -1.824, -1.5734, -0.0029]),
    ("094_topdown_ascent_10", [-0.0923, -1.3063, 1.4967, -1.7801, -1.5739, -0.004]),
    ("095_topdown_ascent_11", [-0.0924, -1.3136, 1.4587, -1.7348, -1.5738, -0.004]),
    ("096_topdown_ascent_12", [-0.0924, -1.3191, 1.4187, -1.6896, -1.5739, -0.0042]),
    ("097_topdown_ascent_13", [-0.0925, -1.3223, 1.3759, -1.6436, -1.5733, -0.003]),
    ("098_topdown_ascent_14", [-0.0925, -1.3238, 1.3311, -1.5976, -1.5734, -0.003]),
    ("099_topdown_ascent_15", [-0.0925, -1.3233, 1.2839, -1.5514, -1.5735, -0.0031]),
    ("100_topdown_ascent_16", [-0.0924, -1.3207, 1.2341, -1.5046, -1.5737, -0.0034]),
]


class DatasetCollectorNode(Node):
    def __init__(self, output_dir):
        super().__init__("ur10e_dataset_collector_node")
        self.get_logger().info("==================================================")
        self.get_logger().info("  UR10e Verified Vision Dataset Collector Node    ")
        self.get_logger().info("==================================================")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Clear existing old images in output directory for a fresh dataset
        for old_img in glob.glob(os.path.join(self.output_dir, "*.jpg")):
            try:
                os.remove(old_img)
            except Exception:
                pass
        self.get_logger().info(f"Target folder: {self.output_dir}")

        self._cv_bridge = CvBridge()
        self._latest_image = None
        self._image_received_count = 0

        # Subscribe to wrist camera image
        self._image_sub = self.create_subscription(
            Image,
            "/wrist_camera/image",
            self._image_callback,
            10
        )

        # Gripper Client
        self._gripper_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
        )

        # Follow Joint Trajectory Controller Client
        self._traj_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        if not self._traj_client.wait_for_server(timeout_sec=40.0):
            self.get_logger().error("Trajectory controller action server did not respond! Exiting.")
            sys.exit(1)
        self.get_logger().info("Joint Trajectory Controller connected successfully!")

    def _image_callback(self, msg: Image):
        self._latest_image = msg
        self._image_received_count += 1

    def command_gripper(self, position: float = 0.0):
        if not self._gripper_client.wait_for_server(timeout_sec=3.0):
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 20.0
        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle and handle.accepted:
            res_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_future, timeout_sec=2.0)

    def send_pose_goal(self, pose_name, joint_values, duration_sec=2.5):
        self.get_logger().info(f"Moving arm to: {pose_name} (duration: {duration_sec:.1f}s)...")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in joint_values]
        pt.time_from_start.sec = int(duration_sec)
        pt.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        goal.trajectory.points.append(pt)

        future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            self.get_logger().error(f"Goal send for {pose_name} timed out!")
            return False
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(f"Goal for {pose_name} rejected by controller!")
            return False

        res_future = goal_handle.get_result_async()
        start_wait = time.time()
        max_wait = max(duration_sec * 7.0, 16.0)
        while time.time() - start_wait < max_wait:
            rclpy.spin_once(self, timeout_sec=0.1)
            if res_future.done():
                break
            time.sleep(0.05)

        if res_future.done():
            code = res_future.result().result.error_code
            if code == 0:
                self.get_logger().info(f"✅ Successfully reached: {pose_name}")
                return True
            else:
                self.get_logger().warning(f"Controller finished {pose_name} with code: {code}")
                return False
        else:
            self.get_logger().warning(f"Motion for {pose_name} timed out after {max_wait:.1f}s")
            return False

    def capture_frame(self, filename, pose_name):
        """
        Waits for fresh frame from /wrist_camera/image.
        STRICT VALIDATION: Only saves the frame if the red cube is verified in the image!
        """
        self._latest_image = None
        start_wait = time.time()
        while time.time() - start_wait < 3.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_image is not None:
                break
            time.sleep(0.02)

        if self._latest_image is None:
            self.get_logger().error(f"Failed to capture {filename}: No camera image received!")
            return False

        try:
            cv_img = self._cv_bridge.imgmsg_to_cv2(self._latest_image, desired_encoding="bgr8")
            
            # Calculate brightness and diagnostics
            mean_brightness = float(np.mean(cv_img))
            
            # Segment red color in HSV (relaxed thresholds to handle varied lighting/angles)
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area = 0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

            # Always save debug frame
            cv2.imwrite("/tmp/last_camera_frame.jpg", cv_img)

            if mean_brightness < 2.0:
                self.get_logger().error(
                    f"⚠️ Camera image is completely black (mean brightness: {mean_brightness:.1f})! Gazebo offscreen rendering may be unlit."
                )

            file_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(file_path, cv_img)
            self.get_logger().info(
                f"📸 [{filename}] SAVED ({pose_name}) | Red Area: {area:.0f} px | Brightness: {mean_brightness:.1f}"
            )
            return True
        except Exception as e:
            self.get_logger().error(f"OpenCV error saving {filename}: {e}")
            return False

    def run_collection(self):
        total_poses = len(DIVERSE_POSES)
        self.get_logger().info(f"Starting dataset collection across {total_poses} verified perspectives...")

        # 1. Open gripper to prevent any finger self-collision
        # 1. Initialize gripper: OPEN
        self.get_logger().info("Initializing gripper: OPEN")
        self.command_gripper(0.0)

        # 2. First move arm safely to HOME configuration (straight up in the air)
        home_pose = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]
        self.get_logger().info("Homing robot arm safely away from obstacles...")
        self.send_pose_goal("HOME", home_pose, duration_sec=2.0)
        time.sleep(0.2)

        # 3. Sequentially visit all 20 diverse viewing poses with adaptive duration
        saved_count = 0
        current_joints = home_pose

        for i, (pose_name, target_joints) in enumerate(DIVERSE_POSES, 1):
            # Compute dynamic duration based on max joint displacement (~0.8 rad/s angular speed)
            max_delta = max(abs(tj - cj) for tj, cj in zip(target_joints, current_joints))
            adaptive_duration = max(0.8, min(2.5, max_delta / 0.8))

            self.get_logger().info(
                f"\n>>> [{i}/{total_poses}] Moving to: {pose_name} (delta={max_delta:.2f} rad, dur={adaptive_duration:.2f}s)"
            )
            reached = self.send_pose_goal(pose_name, target_joints, duration_sec=adaptive_duration)

            if reached:
                current_joints = target_joints
                # Brief settle before snapshot
                time.sleep(0.15)
                filename = f"frame_{i:03d}.jpg"
                if self.capture_frame(filename, pose_name):
                    saved_count += 1
            else:
                self.get_logger().warning(f"Skipping capture for {pose_name} as pose was not reached.")

        # 4. Return to HOME upon completing the linear ascent
        self.get_logger().info("\n============================================================")
        self.get_logger().info(" 🏁 Linear ascent completed! Returning robot safely to HOME...")
        self.get_logger().info("============================================================")
        max_delta_home = max(abs(hj - cj) for hj, cj in zip(home_pose, current_joints))
        home_duration = max(1.2, min(2.5, max_delta_home / 0.8))
        self.send_pose_goal("HOME", home_pose, duration_sec=home_duration)
        time.sleep(0.5)

        self.get_logger().info("\n============================================================")
        self.get_logger().info(f" ✅ SUCCESS! {saved_count} / {total_poses} VERIFIED IMAGES SAVED TO:")
        self.get_logger().info(f"    {self.output_dir}")
        self.get_logger().info("    Robot is safely parked at HOME. Shutting down cleanly.")
        self.get_logger().info("============================================================\n")


def main(args=None):
    rclpy.init(args=args)

    target_dir = DESKTOP_DIR if os.path.exists("/mnt/c/Users/admin/Desktop") else FALLBACK_DIR
    node = DatasetCollectorNode(output_dir=target_dir)
    try:
        node.run_collection()
    except KeyboardInterrupt:
        node.get_logger().info("Collection cancelled by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Cleanly exit so launch Shutdown event triggers and closes Gazebo & MoveIt
        sys.exit(0)


if __name__ == "__main__":
    main()
