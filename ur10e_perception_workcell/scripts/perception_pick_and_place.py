#!/usr/bin/env python3
"""
UR10e Vision-Guided Pick-and-Place Node.

Executes autonomous pick-and-place using eye-in-hand perception (YOLO11 / OpenCV),
two-stage visual servoing, 6D inverse kinematics, and direct trajectory control.
"""

import os
import sys
import time
import math
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import Empty
from control_msgs.action import GripperCommand, FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from cv_bridge import CvBridge

import tf2_ros

# Append YOLO virtual environment if available
venv_path = "/home/gerogemal/Documents/yolo11-tutorial/venv/lib/python3.12/site-packages"
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.append(venv_path)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

POSES = {
    "HOME": [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0],
    "OBSERVE_CENTER": [-0.2063, -1.2352, 0.7682, -1.1038, -1.5708, 0.0],
    "OBSERVE_LEFT": [0.1671, -1.2196, 1.0250, -1.3762, -1.5708, 0.0],
    "OBSERVE_RIGHT": [-0.5530, -1.2196, 1.0250, -1.3762, -1.5708, 0.0],
    "SCAN_TRANSIT": [0.1671, -1.5707, 1.3104, -1.6457, -1.5708, 0.0],
    "SCAN_START": [0.1671, -1.2355, 1.3104, -1.6457, -1.5708, 0.0],
}

MOVEIT_ERROR_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -26: "START_STATE_INVALID",
    -27: "GOAL_STATE_INVALID",
    -28: "UNRECOGNIZED_GOAL_TYPE",
    -29: "CRASH",
    -30: "ABORT",
    -31: "NO_IK_SOLUTION",
}


class CalibratedKinematics:
    """Forward and inverse kinematics for the UR10e workcell."""

    def __init__(self):
        self.pedestal_z = 0.75
        # Perpendicular downward orientation with fingers parallel to world Y-axis
        self.R_target = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=float)

    def _tf_mat(self, xyz, rpy, axis_rot=0.0):
        T = np.eye(4)
        rot = R.from_euler('xyz', rpy).as_matrix()
        if axis_rot != 0.0:
            rot = rot @ R.from_euler('z', axis_rot).as_matrix()
        T[:3, :3] = rot
        T[:3, 3] = xyz
        return T

    def fk_tool0(self, q):
        """Compute forward kinematics for tool0 in world frame."""
        T = np.eye(4)
        T[2, 3] = self.pedestal_z
        T = T @ self._tf_mat([0, 0, 0], [0, 0, math.pi])
        T = T @ self._tf_mat([0, 0, 0.1807], [0, 0, 0], axis_rot=q[0])
        T = T @ self._tf_mat([0, 0, 0], [math.pi / 2.0, 0, 0], axis_rot=q[1])
        T = T @ self._tf_mat([-0.6127, 0, 0], [0, 0, 0], axis_rot=q[2])
        T = T @ self._tf_mat([-0.57155, 0, 0.17415], [0, 0, 0], axis_rot=q[3])
        T = T @ self._tf_mat([0, -0.11985, 0], [math.pi / 2.0, 0, 0], axis_rot=q[4])
        T = T @ self._tf_mat([0, 0.11655, 0], [math.pi / 2.0, math.pi, math.pi], axis_rot=q[5])
        T = T @ self._tf_mat([0, 0, 0], [0, -math.pi / 2.0, -math.pi / 2.0])
        T = T @ self._tf_mat([0, 0, 0], [math.pi / 2.0, 0, math.pi / 2.0])
        return T

    def fk_tcp(self, q):
        """Compute forward kinematics for tcp_link in world frame."""
        T = self.fk_tool0(q)
        T = T @ self._tf_mat([0, 0, 0.011], [0, 0, 1.570796])
        T = T @ self._tf_mat([0, 0, 0.14], [0, 0, 0])
        return T

    def fk_cam_optical(self, q):
        """Compute forward kinematics for wrist_camera_optical_frame in world frame."""
        T = self.fk_tool0(q)
        T = T @ self._tf_mat([0.095, 0, 0.135], [0, -math.pi / 2.0, 0])
        T = T @ self._tf_mat([0, 0, 0], [-math.pi / 2.0, 0, -math.pi / 2.0])
        return T

    def solve_ik(self, target_tcp, q_seed=None):
        """Solve 6D inverse kinematics for target_tcp with vertical downward orientation."""
        if q_seed is None:
            q_seed = np.array([-0.2063, -1.0972, 1.8092, -2.2828, -1.5708, 1.3645])
        target = np.array(target_tcp, dtype=float)

        def loss(q):
            T = self.fk_tcp(q)
            pos_err = np.sum((T[:3, 3] - target) ** 2)
            rot_err = np.sum((T[:3, :3] - self.R_target) ** 2)
            smooth_err = 0.005 * np.sum((q - q_seed) ** 2)
            return 1200.0 * pos_err + 350.0 * rot_err + smooth_err

        bounds = [
            (-6.28, 6.28), (-3.14, 0.0), (-3.14, 3.14),
            (-6.28, 6.28), (-6.28, 6.28), (-6.28, 6.28)
        ]
        res = minimize(loss, q_seed, method="SLSQP", bounds=bounds, options={"maxiter": 250, "ftol": 1e-8})
        return res.x


class UR10ePerceptionPickPlaceNode(Node):
    def __init__(self):
        super().__init__("ur10e_perception_pick_place_node")
        self.get_logger().info("Initializing UR10e Perception Pick-and-Place node...")

        self._cv_bridge = CvBridge()
        self._latest_image = None
        self._kinematics = CalibratedKinematics()
        self._current_joints = list(POSES["HOME"])

        model_candidates = [
            "/home/gerogemal/Documents/yolo11-tutorial/best.pt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights", "best.pt"),
            os.path.join(os.path.expanduser("~"), "Documents", "yolo11-tutorial", "best.pt"),
            "best.pt",
        ]
        self.model_path = next((p for p in model_candidates if os.path.exists(p)), model_candidates[0])
        if HAS_YOLO and os.path.exists(self.model_path):
            self.get_logger().info(f"Loading YOLO model: {self.model_path}")
            try:
                self._yolo = YOLO(self.model_path)
                self.get_logger().info("YOLO model loaded successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
                self._yolo = None
        else:
            self.get_logger().warn("YOLO model not found. Using HSV segmentation fallback.")
            self._yolo = None

        self._image_sub = self.create_subscription(
            Image,
            "/wrist_camera/image",
            self._image_callback,
            10
        )

        self._camera_info = None
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            "/wrist_camera/camera_info",
            self._camera_info_callback,
            10
        )

        self._joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10
        )

        self._tf_buffer = tf2_ros.Buffer(node=self)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        self._move_group_client = ActionClient(self, MoveGroup, "move_action")
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
        )

        self._attach_pub = self.create_publisher(Empty, "/cube/attach", 10)
        self._detach_pub = self.create_publisher(Empty, "/cube/detach", 10)
        self._current_gripper_pos = 0.0

        self.get_logger().info("Connecting to MoveGroup action server...")
        if not self._move_group_client.wait_for_server(timeout_sec=35.0):
            self.get_logger().error("MoveGroup action server unavailable. Aborting.")
            sys.exit(1)
        self.get_logger().info("MoveGroup action server connected.")

        self._has_traj_client = False
        if self._trajectory_client.wait_for_server(timeout_sec=5.0):
            self._has_traj_client = True
            self.get_logger().info("FollowJointTrajectory controller connected.")
        else:
            self.get_logger().warn("FollowJointTrajectory controller unavailable, falling back to MoveGroup.")

        self.get_logger().info("Connecting to Gripper action server...")
        if self._gripper_client.wait_for_server(timeout_sec=10.0):
            self._has_gripper = True
            self.get_logger().info("Gripper controller connected.")
        else:
            self._has_gripper = False
            self.get_logger().warn("Gripper controller unavailable (simulation mode).")

    def get_gripper_tcp_center(self):
        """Compute current 3D world coordinates of gripper TCP."""
        try:
            tf = self._tf_buffer.lookup_transform(
                "world",
                "tcp_link",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            return np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ], dtype=float)
        except Exception:
            return self._kinematics.fk_tcp(self._current_joints)[:3, 3]

    def _image_callback(self, msg: Image):
        self._latest_image = msg

    def _camera_info_callback(self, msg: CameraInfo):
        self._camera_info = msg

    def _joint_state_callback(self, msg: JointState):
        name_pos = dict(zip(msg.name, msg.position))
        if all(j in name_pos for j in JOINT_NAMES):
            self._current_joints = [name_pos[j] for j in JOINT_NAMES]
        if "robotiq_85_left_knuckle_joint" in name_pos:
            self._current_gripper_pos = float(name_pos["robotiq_85_left_knuckle_joint"])

    def command_gripper(self, position: float, max_effort: float = 35.0, timeout_sec: float = 3.0):
        if not self._has_gripper:
            return True

        self.get_logger().info(f"Gripper command: position={position:.3f} rad, max_effort={max_effort:.1f} N")
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.get_logger().warn("Gripper goal send timed out.")
            return False

        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Gripper goal rejected.")
            return False

        res_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future, timeout_sec=timeout_sec)

        if res_future.done() and res_future.result() is not None:
            res = res_future.result().result
            self.get_logger().info(
                f"Gripper action complete: pos={res.position:.3f} rad, effort={res.effort:.1f} N, reached={res.reached_goal}, stalled={res.stalled}"
            )
        else:
            self.get_logger().info(
                f"Gripper motion reached timeout window ({timeout_sec:.1f}s), live pos={self._current_gripper_pos:.3f} rad"
            )
        return True

    def send_exact_trajectory(self, pose_name, joint_values, duration_sec=2.5):
        """Execute trajectory using FollowJointTrajectory action client."""
        self.get_logger().info(f"Executing precision trajectory: {pose_name} ({duration_sec:.1f}s)")
        if not self._has_traj_client and not self._trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Trajectory controller unavailable, falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

        self._has_traj_client = True
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        p0 = JointTrajectoryPoint()
        p0.positions = [float(v) for v in self._current_joints]
        p0.velocities = [0.0] * 6
        p0.time_from_start = Duration(seconds=0).to_msg()
        traj.points.append(p0)

        p1 = JointTrajectoryPoint()
        p1.positions = [float(v) for v in joint_values]
        p1.velocities = [0.0] * 6
        sec = int(duration_sec)
        nsec = int((duration_sec - sec) * 1e9)
        p1.time_from_start = Duration(seconds=sec, nanoseconds=nsec).to_msg()
        traj.points.append(p1)

        goal.trajectory = traj
        future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        if not future.done() or future.result() is None:
            self.get_logger().warn(f"Trajectory goal send for {pose_name} timed out. Falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Trajectory goal for {pose_name} rejected. Falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        if not res_future.done() or res_future.result() is None:
            self.get_logger().warn(f"Trajectory execution for {pose_name} timed out.")
            return False

        res = res_future.result().result
        if res.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"Target reached: {pose_name}")
            self._current_joints = list(joint_values)
            time.sleep(0.05)
            return True
        else:
            self.get_logger().warn(f"Trajectory error code: {res.error_code}. Falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

    def send_pose_goal(
        self,
        pose_name,
        joint_values,
        tolerance=0.002,
        velocity_scaling=0.25,
        acceleration_scaling=0.25,
        max_retries=2,
    ):
        """Plan and execute joint goal using MoveGroup action client."""
        self.get_logger().info(f"MoveGroup planning: {pose_name} (vel: {velocity_scaling:.2f})")

        for attempt in range(1, max_retries + 1):
            goal = MoveGroup.Goal()
            req = MotionPlanRequest()
            req.group_name = "ur_manipulator"
            req.num_planning_attempts = 10 if attempt == 1 else 15
            req.allowed_planning_time = 5.0 if attempt == 1 else 8.0
            req.max_velocity_scaling_factor = float(velocity_scaling)
            req.max_acceleration_scaling_factor = float(acceleration_scaling)

            constraints = Constraints()
            for name, val in zip(JOINT_NAMES, joint_values):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = float(val)
                jc.tolerance_above = tolerance
                jc.tolerance_below = tolerance
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)

            req.goal_constraints.append(constraints)
            goal.request = req
            goal.planning_options.plan_only = False

            future = self._move_group_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().error(f"Goal for {pose_name} rejected by MoveIt (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            if result.error_code.val == 1:
                self.get_logger().info(f"Target reached: {pose_name}")
                self._current_joints = list(joint_values)
                return True
            else:
                err_name = MOVEIT_ERROR_NAMES.get(result.error_code.val, "UNKNOWN_ERROR")
                self.get_logger().warn(
                    f"MoveIt attempt {attempt}/{max_retries} finished with code {result.error_code.val} ({err_name})"
                )
                if attempt < max_retries:
                    time.sleep(0.5)

        self.get_logger().error(f"Failed to reach {pose_name} after {max_retries} attempts.")
        return False

    def solve_ik_target(self, target_pos, q_seed=None):
        """Compute deterministic 6D IK using CalibratedKinematics."""
        if q_seed is None:
            q_seed = self._current_joints
        return list(self._kinematics.solve_ik(target_pos, q_seed=q_seed))

    def _detect_cube_in_frame(self, cv_img):
        """
        Detect red cube top face centroid using HSV segmentation with YOLO fallback.
        Returns: (cx, cy, conf, best_box) or (None, None, 0.0, None)
        """
        cx, cy, conf, best_box = None, None, 0.0, None

        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 80, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 80:
                rect = cv2.minAreaRect(largest)
                (rx, ry), (rw, rh), _ = rect
                cx = float(rx)
                cy = float(ry)
                conf = 0.95
                bx, by, bw, bh = cv2.boundingRect(largest)
                best_box = [bx, by, bx + bw, by + bh]

        if cx is None and self._yolo is not None:
            try:
                results = self._yolo.predict(
                    source=cv_img,
                    conf=0.25,
                    device="cpu",
                    verbose=False
                )
                if results and len(results[0].boxes) > 0:
                    best_box_item = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    xyxy = best_box_item.xyxy[0].cpu().numpy()
                    cx = float((xyxy[0] + xyxy[2]) / 2.0)
                    cy = float((xyxy[1] + xyxy[3]) / 2.0)
                    conf = float(best_box_item.conf[0])
                    best_box = xyxy
            except Exception as e:
                self.get_logger().warn(f"YOLO inference exception: {e}")

        return cx, cy, conf, best_box

    def _ray_plane_intersect(self, u, v, width, height, target_z=0.825):
        """Intersect camera optical ray with horizontal plane Z = target_z."""
        if self._camera_info is not None and len(self._camera_info.k) >= 9 and self._camera_info.k[0] > 0.0:
            fx = float(self._camera_info.k[0])
            fy = float(self._camera_info.k[4])
            cx_cam = float(self._camera_info.k[2])
            cy_cam = float(self._camera_info.k[5])
        else:
            hfov = 1.2
            fx = (width / 2.0) / math.tan(hfov / 2.0)
            fy = fx
            cx_cam = width / 2.0
            cy_cam = height / 2.0

        rx = (u - cx_cam) / fx
        ry = (v - cy_cam) / fy
        ray_opt = np.array([rx, ry, 1.0], dtype=float)
        ray_opt /= np.linalg.norm(ray_opt)

        cam_pos = None
        rot_mat = None

        try:
            tf = self._tf_buffer.lookup_transform(
                "world",
                "wrist_camera_optical_frame",
                rclpy.time.Time(),
                timeout=Duration(seconds=1.5)
            )
            cam_pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            q = tf.transform.rotation
            rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        except Exception:
            T_opt = self._kinematics.fk_cam_optical(self._current_joints)
            cam_pos = T_opt[:3, 3]
            rot_mat = T_opt[:3, :3]

        if cam_pos is not None and rot_mat is not None:
            ray_world = rot_mat.dot(ray_opt)
            if abs(ray_world[2]) > 1e-4:
                t = (target_z - cam_pos[2]) / ray_world[2]
                intersection = cam_pos + t * ray_world
                return round(float(intersection[0]), 4), round(float(intersection[1]), 4)

        return None, None

    def _check_frame_for_cube(self, target_z=0.825, max_attempts=8):
        """Sample camera frames and compute median world coordinates."""
        samples = []
        best_cand = None

        for _ in range(max_attempts):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_image is None:
                time.sleep(0.06)
                continue

            try:
                cv_img = self._cv_bridge.imgmsg_to_cv2(self._latest_image, desired_encoding="bgr8")
            except Exception:
                time.sleep(0.06)
                continue

            h, w, _ = cv_img.shape
            cx, cy, conf, box = self._detect_cube_in_frame(cv_img)

            if cx is not None and cy is not None:
                world_x, world_y = self._ray_plane_intersect(cx, cy, w, h, target_z=target_z)
                if world_x is not None and world_y is not None:
                    dist_to_center = math.hypot(cx - (w / 2.0), cy - (h / 2.0))
                    cand = {
                        "cx": cx,
                        "cy": cy,
                        "world_x": world_x,
                        "world_y": world_y,
                        "conf": conf,
                        "box": box,
                        "dist_center": dist_to_center,
                        "img": cv_img.copy()
                    }
                    samples.append(cand)
                    if best_cand is None or dist_to_center < best_cand["dist_center"]:
                        best_cand = cand
                    if len(samples) >= 3:
                        break
            time.sleep(0.06)

        if samples:
            med_x = float(np.median([s["world_x"] for s in samples]))
            med_y = float(np.median([s["world_y"] for s in samples]))
            best_cand["world_x"] = round(med_x, 3)
            best_cand["world_y"] = round(med_y, 3)
            return best_cand
        return None

    def _finalize_detection(self, cand, target_z=0.825):
        """Annotate snapshot and log detection results."""
        best_x = cand["world_x"]
        best_y = cand["world_y"]

        self.get_logger().info(
            f"Target pose computed: X={best_x:.3f} m, Y={best_y:.3f} m, Z={target_z:.3f} m (Confidence: {cand['conf']:.2f})"
        )

        try:
            annotated = cand["img"]
            if cand["box"] is not None:
                x1, y1, x2, y2 = [int(v) for v in cand["box"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (int(cand["cx"]), int(cand["cy"])), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"Cube Center: X={best_x:.3f} Y={best_y:.3f} Z=0.825",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imwrite("/tmp/perception_detection.jpg", annotated)
            self.get_logger().info("Saved detection snapshot to: /tmp/perception_detection.jpg")
        except Exception as e:
            self.get_logger().warn(f"Failed to save snapshot: {e}")

        return best_x, best_y, 0.825

    def observe_and_detect_cube(self, target_z=0.825):
        """Execute wide table observation to obtain coarse cube coordinates."""
        self.get_logger().info("Starting adaptive table observation...")

        # Central high-altitude overview (Z=1.50m)
        self.get_logger().info("Moving to OBSERVE_CENTER...")
        if self.send_pose_goal("OBSERVE_CENTER", POSES["OBSERVE_CENTER"], velocity_scaling=0.30):
            time.sleep(0.5)
            cand = self._check_frame_for_cube(target_z=target_z)
            if cand is not None:
                self.get_logger().info(f"Cube detected in central overview (Conf: {cand['conf']:.2f})")
                return self._finalize_detection(cand, target_z=target_z)
            self.get_logger().info("Cube not detected in center. Initiating boundary sweep...")

        # Left quadrant sweep
        self.get_logger().info("Sweeping to OBSERVE_LEFT...")
        if self.send_pose_goal("OBSERVE_LEFT", POSES["OBSERVE_LEFT"], velocity_scaling=0.35):
            time.sleep(0.4)
            cand = self._check_frame_for_cube(target_z=target_z)
            if cand is not None:
                self.get_logger().info(f"Cube detected on left side (Conf: {cand['conf']:.2f})")
                return self._finalize_detection(cand, target_z=target_z)

        # Right quadrant sweep
        self.get_logger().info("Sweeping to OBSERVE_RIGHT...")
        if self.send_pose_goal("OBSERVE_RIGHT", POSES["OBSERVE_RIGHT"], velocity_scaling=0.35):
            time.sleep(0.4)
            cand = self._check_frame_for_cube(target_z=target_z)
            if cand is not None:
                self.get_logger().info(f"Cube detected on right side (Conf: {cand['conf']:.2f})")
                return self._finalize_detection(cand, target_z=target_z)

        self.get_logger().error("Cube could not be detected across table.")
        return None, None, None

    def closeup_visual_servoing(self, coarse_x, coarse_y, target_z=0.825):
        """Hover camera directly above coarse estimate to refine coordinates."""
        self.get_logger().info("Starting Stage 2 close-up visual servoing...")

        # Camera optical offset: X_cam = X_tcp - 0.095, Y_cam = Y_tcp
        inspect_tcp = [coarse_x + 0.095, coarse_y, 1.050]
        q_inspect = self.solve_ik_target(inspect_tcp, q_seed=self._current_joints)

        if not self.send_exact_trajectory("INSPECT_CLOSEUP", q_inspect, duration_sec=3.0):
            if not self.send_pose_goal("INSPECT_CLOSEUP", q_inspect, velocity_scaling=0.30):
                self.get_logger().warn("Close-up motion failed. Retaining coarse target.")
                return coarse_x, coarse_y, target_z

        time.sleep(0.5)

        # Top face plane of 50mm cube on table is Z = 0.850m
        cand = self._check_frame_for_cube(target_z=0.850, max_attempts=8)
        if cand is not None:
            fine_x = cand["world_x"]
            fine_y = cand["world_y"]
            dx_mm = (fine_x - coarse_x) * 1000.0
            dy_mm = (fine_y - coarse_y) * 1000.0
            self.get_logger().info(f"Stage 1 Coarse: X={coarse_x:.4f} m, Y={coarse_y:.4f} m")
            self.get_logger().info(f"Stage 2 Fine:   X={fine_x:.4f} m, Y={fine_y:.4f} m (Conf: {cand['conf']:.2f})")
            self.get_logger().info(f"Visual correction: dX={dx_mm:+.2f} mm, dY={dy_mm:+.2f} mm")

            try:
                annotated = cand["img"]
                if cand["box"] is not None:
                    x1, y1, x2, y2 = [int(v) for v in cand["box"]]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (int(cand["cx"]), int(cand["cy"])), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"Fine Center: X={fine_x:.3f} Y={fine_y:.3f} (Corr: {dx_mm:+.1f},{dy_mm:+.1f}mm)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.imwrite("/tmp/perception_stage2_closeup.jpg", annotated)
                self.get_logger().info("Saved close-up snapshot to: /tmp/perception_stage2_closeup.jpg")
            except Exception as e:
                self.get_logger().warn(f"Failed to save close-up snapshot: {e}")

            return fine_x, fine_y, target_z
        else:
            self.get_logger().warn("Cube not detected in close-up view. Retaining coarse target.")
            return coarse_x, coarse_y, target_z

    def execute_linear_scan(self, target_z=0.825, scan_seed=None):
        return self.observe_and_detect_cube(target_z=target_z)

    def run_sequence(self):
        """Execute full pick-and-place state machine."""
        time.sleep(1.5)

        # Release cube at startup so it settles naturally on table
        self.get_logger().info("[Step 0a] Releasing cube at startup...")
        try:
            import subprocess
            gz_bin = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
            if os.path.exists(gz_bin):
                subprocess.run(
                    [gz_bin, "topic", "-t", "/cube/detach", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=2.0, capture_output=True
                )
        except Exception:
            pass

        for _ in range(5):
            self._detach_pub.publish(Empty())
            time.sleep(0.05)

        time.sleep(2.0)

        # Initialize gripper
        self.get_logger().info("[Step 0b] Initializing gripper: OPEN")
        self.command_gripper(0.0, max_effort=30.0, timeout_sec=3.0)
        time.sleep(0.2)

        # Move to HOME
        self.get_logger().info("[Step 1] Moving to HOME")
        if not self.send_pose_goal("HOME", POSES["HOME"], velocity_scaling=0.25):
            return False
        time.sleep(0.3)

        # Stage 1: Coarse Overview
        self.get_logger().info("[Step 2] Executing Stage 1 Coarse Overview")
        coarse_x, coarse_y, coarse_z = self.observe_and_detect_cube(target_z=0.825)
        if coarse_x is None or coarse_y is None:
            self.get_logger().error("Cube detection failed. Aborting sequence.")
            self.send_pose_goal("HOME", POSES["HOME"], velocity_scaling=0.25)
            return False

        # Stage 2: Eye-in-Hand Visual Servoing
        cube_x, cube_y, cube_z = self.closeup_visual_servoing(coarse_x, coarse_y, target_z=0.825)

        # Compute trajectory waypoints with sequential seeding
        self.get_logger().info(f"[Step 3] Computing grasp kinematics for target: X={cube_x:.3f}, Y={cube_y:.3f}")
        pre_grasp_pos = [cube_x, cube_y, 0.980]
        grasp_pos     = [cube_x, cube_y, 0.825]
        lift_pos      = [cube_x, cube_y, 0.980]

        place_x = cube_x
        place_y = round(cube_y - 0.24, 3) if cube_y >= 0.0 else round(cube_y + 0.24, 3)
        place_approach_pos = [place_x, place_y, 0.980]
        place_pos          = [place_x, place_y, 0.825]

        q_curr = list(self._current_joints)
        q_pre_grasp = self.solve_ik_target(pre_grasp_pos, q_seed=q_curr)
        q_grasp     = self.solve_ik_target(grasp_pos, q_seed=q_pre_grasp)
        q_lift      = self.solve_ik_target(lift_pos, q_seed=q_grasp)
        q_place_app = self.solve_ik_target(place_approach_pos, q_seed=q_lift)
        q_place     = self.solve_ik_target(place_pos, q_seed=q_place_app)

        # Move to PRE_GRASP
        self.get_logger().info(f"[Step 4] Moving to PRE_GRASP at X={cube_x:.3f}, Y={cube_y:.3f}")
        if not self.send_exact_trajectory("PRE_GRASP", q_pre_grasp, duration_sec=3.0):
            return False
        time.sleep(0.2)

        # Verify alignment and execute safety gating check
        tcp_pos = self.get_gripper_tcp_center()
        align_x = (tcp_pos[0] - cube_x) * 1000.0
        align_y = (tcp_pos[1] - cube_y) * 1000.0
        self.get_logger().info(f"Pre-grasp alignment: TCP=({tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}), Target=({cube_x:.4f}, {cube_y:.4f})")
        self.get_logger().info(f"Axial deviation: dX={align_x:+.2f} mm, dY={align_y:+.2f} mm")

        MAX_SAFE_DEVIATION_MM = 12.0
        if abs(align_x) > MAX_SAFE_DEVIATION_MM or abs(align_y) > MAX_SAFE_DEVIATION_MM:
            self.get_logger().error(
                f"Safety threshold exceeded: dX={align_x:+.1f}mm, dY={align_y:+.1f}mm (> {MAX_SAFE_DEVIATION_MM}mm). Aborting descent."
            )
            self.send_exact_trajectory("HOME", POSES["HOME"], duration_sec=3.0)
            return False

        # Descend to GRASP
        self.get_logger().info("[Step 5] Descending to GRASP (Z=0.825m)")
        if not self.send_exact_trajectory("GRASP", q_grasp, duration_sec=1.5):
            return False
        time.sleep(0.2)

        # Close gripper and attach cube
        self.get_logger().info("[Step 6] Closing gripper (0.345 rad, 35.0 N)")
        self.command_gripper(0.345, max_effort=35.0, timeout_sec=3.0)
        time.sleep(0.2)
        for _ in range(5):
            self._attach_pub.publish(Empty())
            time.sleep(0.02)
        try:
            import subprocess
            gz_bin = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
            if os.path.exists(gz_bin):
                subprocess.run(
                    [gz_bin, "topic", "-t", "/cube/attach", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=2.0, capture_output=True
                )
        except Exception:
            pass
        time.sleep(0.15)

        # Lift cube
        self.get_logger().info("[Step 7] Lifting cube vertically")
        if not self.send_exact_trajectory("LIFT", q_lift, duration_sec=1.5):
            return False
        time.sleep(0.2)

        # Transfer to place approach
        self.get_logger().info(f"[Step 8] Transferring to PLACE_APPROACH at X={place_x:.3f}, Y={place_y:.3f}")
        if not self.send_exact_trajectory("PLACE_APPROACH", q_place_app, duration_sec=3.0):
            return False
        time.sleep(0.2)

        # Lower to table
        self.get_logger().info("[Step 9] Lowering to PLACE location")
        if not self.send_exact_trajectory("PLACE", q_place, duration_sec=1.5):
            return False
        time.sleep(0.25)

        # Open gripper and detach
        self.get_logger().info("[Step 10a] Opening gripper")
        self.command_gripper(0.0, max_effort=25.0, timeout_sec=3.0)
        time.sleep(0.2)

        self.get_logger().info("[Step 10b] Detaching cube on table")
        for _ in range(5):
            self._detach_pub.publish(Empty())
            time.sleep(0.02)
        try:
            import subprocess
            gz_bin = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
            if os.path.exists(gz_bin):
                subprocess.run(
                    [gz_bin, "topic", "-t", "/cube/detach", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=2.0, capture_output=True
                )
        except Exception:
            pass
        time.sleep(0.25)

        # Retract away from cube
        self.get_logger().info("[Step 11] Retracting vertically from cube")
        if not self.send_pose_goal("PLACE_APPROACH", q_place_app, velocity_scaling=0.20):
            return False
        time.sleep(0.2)

        # Return to HOME
        self.get_logger().info("[Step 12] Returning to HOME")
        if not self.send_pose_goal("HOME", POSES["HOME"], velocity_scaling=0.25):
            return False

        self.get_logger().info("Vision-guided pick-and-place sequence completed successfully.")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = UR10ePerceptionPickPlaceNode()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().info("Sequence interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
