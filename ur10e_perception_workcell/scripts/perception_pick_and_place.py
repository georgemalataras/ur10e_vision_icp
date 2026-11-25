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
import struct
import numpy as np
import cv2

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, JointState, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Empty, ColorRGBA
from geometry_msgs.msg import Pose, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand, FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    ObjectColor,
    PlanningScene,
)
from cv_bridge import CvBridge

import tf2_ros

# Append YOLO virtual environment if available
for venv_path in [
    os.path.expanduser("~/ros2_ai_venv/lib/python3.14/site-packages"),
    os.path.expanduser("~/ros2_ai_venv/lib/python3.12/site-packages"),
    "/home/gerogemal/Documents/yolo11-tutorial/venv/lib/python3.12/site-packages",
]:
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
    # Angled (~40 deg oblique 3D view) observation pose looking directly at the cube on the table
    "OBSERVE_CENTER": [-0.5699, -1.7262, 2.1257, -2.2650, -0.9339, 0.0043],
    "OBSERVE_ANGLED": [-0.5699, -1.7262, 2.1257, -2.2650, -0.9339, 0.0043],
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

    def fk_cam_link(self, q):
        """Compute forward kinematics for wrist_camera_link in world frame."""
        T = self.fk_tool0(q)
        T = T @ self._tf_mat([0.095, 0, 0.135], [0, -math.pi / 2.0, 0])
        return T

    def fk_cam_optical(self, q):
        """Compute forward kinematics for wrist_camera_optical_frame in world frame."""
        T = self.fk_cam_link(q)
        T = T @ self._tf_mat([0, 0, 0], [-math.pi / 2.0, 0, -math.pi / 2.0])
        return T

    def solve_ik(self, target_tcp, q_seed=None, target_yaw=0.0):
        """Solve 6D inverse kinematics for target_tcp with vertical downward orientation and yaw."""
        if q_seed is None:
            q_seed = np.array([-0.2063, -1.0972, 1.8092, -2.2828, -1.5708, 1.3645])
        target = np.array(target_tcp, dtype=float)
        R_des = R.from_euler('z', float(target_yaw)).as_matrix() @ self.R_target

        def loss(q):
            T = self.fk_tcp(q)
            pos_err = np.sum((T[:3, 3] - target) ** 2)
            rot_err = np.sum((T[:3, :3] - R_des) ** 2)
            smooth_err = 0.005 * np.sum((q - q_seed) ** 2)
            return 1200.0 * pos_err + 350.0 * rot_err + smooth_err

        bounds = [
            (-6.28, 6.28), (-3.14, 0.0), (-3.14, 3.14),
            (-6.28, 6.28), (-6.28, 6.28), (-6.28, 6.28)
        ]
        res = minimize(loss, q_seed, method="SLSQP", bounds=bounds, options={"maxiter": 250, "ftol": 1e-8})
        return res.x


def load_stl_cad_faces(stl_path: str, samples_per_axis: int = 11):
    """
    Parse ASCII STL triangles from red_cube.stl and sample dense 3D surface points
    grouped by outward face normal.
    """
    normals = []
    triangles = []
    with open(stl_path, "r") as f:
        cur_norm = None
        cur_verts = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "facet" and parts[1] == "normal":
                cur_norm = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
                cur_verts = []
            elif parts[0] == "vertex":
                cur_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "endfacet" and cur_norm is not None and len(cur_verts) == 3:
                normals.append(cur_norm)
                triangles.append(np.array(cur_verts, dtype=float))

    face_map = {}
    u_lin = np.linspace(0.0, 1.0, samples_per_axis)
    bary = [(u, v) for u in u_lin for v in u_lin if u + v <= 1.0 + 1e-6]
    for norm, tri in zip(normals, triangles):
        key = tuple(np.round(norm, 2))
        v0, v1, v2 = tri[0], tri[1], tri[2]
        pts = np.array([v0 + u * (v1 - v0) + v * (v2 - v0) for (u, v) in bary], dtype=float)
        face_map.setdefault(key, []).append(pts)

    faces = []
    for key, pts_list in face_map.items():
        all_pts = np.unique(np.round(np.vstack(pts_list), 6), axis=0)
        faces.append((all_pts, np.array(key, dtype=float)))
    return faces


class UR10ePerceptionPickPlaceNode(Node):
    def __init__(self):
        super().__init__("ur10e_perception_pick_place_node")
        self.get_logger().info("Initializing UR10e Perception Pick-and-Place node...")
        self.declare_parameter("observe_only", False)
        self._observe_only = bool(self.get_parameter("observe_only").value)

        self._cv_bridge = CvBridge()
        self._latest_image = None
        self._kinematics = CalibratedKinematics()
        self._current_joints = list(POSES["HOME"])

        model_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models", "weights", "best.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights", "best.pt"),
            "/home/gerogemal/Documents/yolo11-tutorial/best.pt",
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
            self.get_logger().warning("YOLO model not found. Using HSV segmentation fallback.")
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
        self._gripper_traj_client = ActionClient(
            self, FollowJointTrajectory, "/robotiq_gripper_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
        )

        self._attach_pub = self.create_publisher(Empty, "/cube/attach", 10)
        self._detach_pub = self.create_publisher(Empty, "/cube/detach", 10)
        self._planning_scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self._current_gripper_pos = 0.0

        # ---------------- 3D ICP Pose Estimation Parameters ----------------
        self.declare_parameter("icp_mode", "segmented")  # "segmented" (Stage 2: Instance Segmentation Mask + Direct ICP) or "unfiltered" (Stage 1: Full Cloud Brute-Force + ICP)
        self.declare_parameter("icp_step_delay", 0.15)   # seconds between visualized ICP steps
        self.declare_parameter("icp_max_iters", 30)
        self._icp_mode = str(self.get_parameter("icp_mode").value).lower()
        self._icp_step_delay = float(self.get_parameter("icp_step_delay").value)
        self._icp_max_iters = int(self.get_parameter("icp_max_iters").value)

        self._latest_cloud = None
        self._cloud_sub = self.create_subscription(
            PointCloud2, "/wrist_camera/points", self._cloud_callback, qos_profile_sensor_data
        )
        self._icp_target_pub = self.create_publisher(PointCloud2, "/icp/target_cloud", 1)
        self._icp_source_pub = self.create_publisher(PointCloud2, "/icp/source_cloud", 1)
        self._icp_marker_pub = self.create_publisher(MarkerArray, "/icp/markers", 1)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._cad_half = 0.025
        self._cad_faces = self._load_cad_model()

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
            self.get_logger().warning("FollowJointTrajectory controller unavailable, falling back to MoveGroup.")

        self.get_logger().info("Connecting to Gripper action server...")
        self._use_gripper_traj = False
        if self._gripper_traj_client.wait_for_server(timeout_sec=8.0):
            self._has_gripper = True
            self._use_gripper_traj = True
            self.get_logger().info("Gripper trajectory controller connected.")
        elif self._gripper_client.wait_for_server(timeout_sec=2.0):
            self._has_gripper = True
            self.get_logger().info("Gripper command controller connected.")
        else:
            self._has_gripper = False
            self.get_logger().warning("Gripper controller unavailable (simulation mode).")

    def update_cube_in_planning_scene(self, x: float, y: float, z: float = 0.825, yaw: float = 0.0):
        """Update red_cube CollisionObject pose (position + yaw) in MoveIt PlanningScene."""
        co = CollisionObject()
        co.header.frame_id = "world"
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "red_cube"
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        qz = R.from_euler("z", float(yaw)).as_quat()
        pose.orientation.z = float(qz[2])
        pose.orientation.w = float(qz[3])

        co.primitives.append(box)
        co.primitive_poses.append(pose)

        oc = ObjectColor()
        oc.id = "red_cube"
        oc.color = ColorRGBA(r=0.95, g=0.10, b=0.10, a=1.0)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)
        scene.object_colors.append(oc)
        self._planning_scene_pub.publish(scene)
        self.get_logger().info(f"Updated 'red_cube' CollisionObject in PlanningScene at ({x:.3f}, {y:.3f}, {z:.3f})")

    def attach_cube_in_planning_scene(self):
        """Attach red_cube CollisionObject to tcp_link in MoveIt PlanningScene during lift/transfer."""
        aco = AttachedCollisionObject()
        aco.link_name = "tcp_link"
        aco.object.header.frame_id = "tcp_link"
        aco.object.header.stamp = self.get_clock().now().to_msg()
        aco.object.id = "red_cube"
        aco.object.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.05]

        pose = Pose()
        pose.orientation.w = 1.0
        aco.object.primitives.append(box)
        aco.object.primitive_poses.append(pose)

        aco.touch_links = [
            "tcp_link",
            "robotiq_85_base_link",
            "robotiq_85_left_knuckle_link",
            "robotiq_85_right_knuckle_link",
            "robotiq_85_left_finger_link",
            "robotiq_85_right_finger_link",
            "robotiq_85_left_inner_knuckle_link",
            "robotiq_85_right_inner_knuckle_link",
            "robotiq_85_left_finger_tip_link",
            "robotiq_85_right_finger_tip_link",
        ]

        remove_world = CollisionObject()
        remove_world.id = "red_cube"
        remove_world.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(remove_world)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        self._planning_scene_pub.publish(scene)
        self.get_logger().info("Attached 'red_cube' to tcp_link in MoveIt PlanningScene")

    def detach_cube_in_planning_scene(self, place_x: float, place_y: float, place_z: float = 0.825):
        """Detach red_cube from tcp_link and place it back on table in MoveIt PlanningScene."""
        aco = AttachedCollisionObject()
        aco.link_name = "tcp_link"
        aco.object.id = "red_cube"
        aco.object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        self._planning_scene_pub.publish(scene)
        time.sleep(0.05)
        self.update_cube_in_planning_scene(place_x, place_y, place_z)

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
        if self._use_gripper_traj:
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = ["robotiq_85_left_knuckle_joint"]

            p1 = JointTrajectoryPoint()
            p1.positions = [float(position)]
            p1.velocities = [0.0]
            p1.time_from_start = Duration(seconds=1, nanoseconds=0).to_msg()
            traj.points.append(p1)
            goal.trajectory = traj

            future = self._gripper_traj_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                self.get_logger().warning("Gripper trajectory goal send timed out.")
                return False

            handle = future.result()
            if not handle.accepted:
                self.get_logger().warning("Gripper trajectory goal rejected.")
                return False

            res_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_future, timeout_sec=timeout_sec)
            self.get_logger().info(f"Gripper motion complete: live pos={self._current_gripper_pos:.3f} rad")
            return True

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.get_logger().warning("Gripper goal send timed out.")
            return False

        handle = future.result()
        if not handle.accepted:
            self.get_logger().warning("Gripper goal rejected.")
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
            self.get_logger().warning("Trajectory controller unavailable, falling back to MoveGroup.")
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
            self.get_logger().warning(f"Trajectory goal send for {pose_name} timed out. Falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning(f"Trajectory goal for {pose_name} rejected. Falling back to MoveGroup.")
            return self.send_pose_goal(pose_name, joint_values, tolerance=0.002)

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        if not res_future.done() or res_future.result() is None:
            self.get_logger().warning(f"Trajectory execution for {pose_name} timed out.")
            return False

        res = res_future.result().result
        if res.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"Target reached: {pose_name}")
            self._current_joints = list(joint_values)
            time.sleep(0.05)
            return True
        else:
            self.get_logger().warning(f"Trajectory error code: {res.error_code}. Falling back to MoveGroup.")
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
                self.get_logger().warning(
                    f"MoveIt attempt {attempt}/{max_retries} finished with code {result.error_code.val} ({err_name})"
                )
                if attempt < max_retries:
                    time.sleep(0.5)

        self.get_logger().error(f"Failed to reach {pose_name} after {max_retries} attempts.")
        return False

    def solve_ik_target(self, target_pos, q_seed=None, yaw=0.0):
        """Compute deterministic 6D IK using CalibratedKinematics (top-down grasp rotated by yaw)."""
        if q_seed is None:
            q_seed = self._current_joints
        return list(self._kinematics.solve_ik(target_pos, q_seed=q_seed, target_yaw=yaw))

    def _detect_cube_in_frame(self, cv_img):
        """
        Detect red cube top face centroid and 2D segmentation mask using HSV color segmentation with YOLO fallback.
        Returns: (cx, cy, conf, best_box, seg_mask)
        """
        cx, cy, conf, best_box = None, None, 0.0, None
        h, w = cv_img.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)

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
                cv2.drawContours(seg_mask, [largest], -1, 255, -1)

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
                    # If YOLO provides masks (YOLO-seg)
                    if hasattr(results[0], "masks") and results[0].masks is not None:
                        m_raw = results[0].masks.data[0].cpu().numpy()
                        seg_mask = (cv2.resize(m_raw, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
                    else:
                        x1, y1, x2, y2 = [int(v) for v in best_box]
                        seg_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
            except Exception as e:
                self.get_logger().warning(f"YOLO inference exception: {e}")

        return cx, cy, conf, best_box, seg_mask

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
            cx, cy, conf, box, _ = self._detect_cube_in_frame(cv_img)

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
            self.get_logger().warning(f"Failed to save snapshot: {e}")

        return best_x, best_y, 0.825

    # =====================================================================
    # Phase 1: 3D ICP Pose Estimation (CAD model <-> RGB-D PointCloud2)
    # =====================================================================
    TABLE_TOP_Z = 0.80

    def _load_cad_model(self):
        """Load red_cube.stl CAD model and sample its surface faces."""
        rel = os.path.join("models", "red_cube", "meshes", "red_cube.stl")
        candidates = [os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), rel)]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(get_package_share_directory("ur10e_perception_workcell"), rel))
        except Exception:
            pass
        for path in candidates:
            if os.path.exists(path):
                faces = load_stl_cad_faces(path)
                all_pts = np.vstack([f[0] for f in faces])
                self._cad_half = float(np.max(np.abs(all_pts)))
                self.get_logger().info(
                    f"Loaded CAD model {path}: {len(faces)} faces, {len(all_pts)} surface points, "
                    f"edge={2 * self._cad_half * 1000:.0f} mm"
                )
                return faces
        self.get_logger().error("CAD model red_cube.stl not found! ICP disabled.")
        return []

    def _cloud_callback(self, msg: PointCloud2):
        self._latest_cloud = msg

    @staticmethod
    def _pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
        """Fast numpy conversion of a sensor_msgs/PointCloud2 to an (N,3) float64 array."""
        offsets = {f.name: f.offset for f in msg.fields}
        n = msg.width * msg.height
        if n == 0 or not all(k in offsets for k in ("x", "y", "z")):
            return np.empty((0, 3))
        raw = np.frombuffer(msg.data, dtype=np.uint8)[: msg.row_step * msg.height]
        raw = raw.reshape(msg.height, msg.row_step)[:, : msg.width * msg.point_step].reshape(n, msg.point_step)
        dt = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
        xyz = np.stack(
            [raw[:, offsets[k]:offsets[k] + 4].copy().view(dt).reshape(n) for k in ("x", "y", "z")], axis=1
        ).astype(np.float64)
        return xyz[np.isfinite(xyz).all(axis=1)]

    @staticmethod
    def _xyz_to_pc2(points, rgb, frame_id, stamp) -> PointCloud2:
        """Pack an (N,3) array into a colored PointCloud2 (x, y, z, rgb)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        n = len(pts)
        rgb_u32 = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
        rgb_f = struct.unpack("f", struct.pack("I", rgb_u32))[0]
        data = np.zeros((n, 4), dtype=np.float32)
        if n:
            data[:, :3] = pts
            data[:, 3] = rgb_f
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.height, msg.width = 1, n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step, msg.row_step = 16, 16 * n
        msg.is_dense = True
        msg.data = data.tobytes()
        return msg

    def _lookup_world_tf(self, frame_id):
        """Return (R, t) mapping points from frame_id to world (TF, with FK fallback)."""
        try:
            tf = self._tf_buffer.lookup_transform("world", frame_id, rclpy.time.Time(), timeout=Duration(seconds=1.0))
            tr, q = tf.transform.translation, tf.transform.rotation
            return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([tr.x, tr.y, tr.z])
        except Exception as e:
            self.get_logger().warning(f"TF world<-{frame_id} unavailable ({e}); using FK of wrist_camera_link.")
            T = self._kinematics.fk_cam_link(self._current_joints)
            return T[:3, :3], T[:3, 3]

    def _capture_segmented_cube_cloud(self, timeout_sec=4.0):
        """
        Stage 2 (Instance Segmentation PointCloud Masking):
        1. Capture synchronized RGB image and 3D PointCloud2, and display the FULL camera FOV
           PointCloud2 in RViz2 so the user clearly sees the unfiltered scene first.
        2. Call Instance Segmentation to extract the 2D cube mask M(u, v).
        3. Project 3D points into the camera optical frame, filter ONLY points inside M(u, v),
           and display the filtered cube PointCloud2 + 3D bounding box in RViz2.
        Returns: (segmented_cube_points_world, full_scene_world, camera_pos_world, seg_mask_viz)
        """
        self._latest_cloud = None
        t0 = time.time()
        while (self._latest_cloud is None or self._latest_image is None) and (time.time() - t0 < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_cloud is None or self._latest_image is None:
            self.get_logger().error("Did not receive synchronized Image and PointCloud2.")
            return None, None, None, None

        try:
            cv_img = self._cv_bridge.imgmsg_to_cv2(self._latest_image, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")
            return None, None, None, None

        img_h, img_w = cv_img.shape[:2]

        # Extract 3D points from sensor_msgs/PointCloud2 (raw points in camera link frame)
        cloud_msg = self._latest_cloud
        pts_cam = self._pc2_to_xyz(cloud_msg)
        if len(pts_cam) < 30:
            self.get_logger().error("PointCloud2 contains insufficient points.")
            return None, None, None, None

        # Transform raw camera sensor points (wrist_camera_link: X fwd, Y left, Z up)
        # to optical frame (Z fwd, X right, Y down):
        x_opt = -pts_cam[:, 1]
        y_opt = -pts_cam[:, 2]
        z_opt = pts_cam[:, 0]
        valid_depth = (z_opt > 0.05) & (z_opt < 3.0)

        # Convert to world coordinates
        R_wc, t_wc = self._lookup_world_tf(cloud_msg.header.frame_id)
        pts_world = pts_cam @ R_wc.T + t_wc

        # Full FOV scene voxelized (4mm) for RViz2 visualization
        _, uniq_scene = np.unique(np.floor(pts_world[valid_depth] / 0.004).astype(int), axis=0, return_index=True)
        full_scene_world = pts_world[valid_depth][uniq_scene]

        # --- Visual Step 1: Show FULL Camera FOV PointCloud2 before filtering ---
        scene_center = np.median(full_scene_world, axis=0)
        self.get_logger().info(
            f"[Stage 2 - Step 1/3] Displaying FULL Camera FOV PointCloud2 ({len(full_scene_world)} pts)..."
        )
        self._publish_filtering_viz(
            full_scene_world,
            scene_center,
            f"[Step 1/3] Full Camera FOV PointCloud2 ({len(full_scene_world)} pts)",
            cloud_rgb=(220, 70, 70),
            bbox_pts=None,
        )
        time.sleep(1.8)

        # --- Visual Step 2: Run Instance Segmentation and filter PointCloud2 inside mask ---
        self.get_logger().info("[Stage 2 - Step 2/3] Running Instance Segmentation to filter PointCloud2...")
        cx, cy, conf, box, seg_mask = self._detect_cube_in_frame(cv_img)
        if seg_mask is None or np.count_nonzero(seg_mask) < 25:
            self._publish_filtering_viz(
                full_scene_world,
                scene_center,
                "[Step 2/3] REJECTED - Instance Segmentation found NO cube in FOV",
                cloud_rgb=(220, 70, 70),
                bbox_pts=None,
            )
            time.sleep(1.2)
            self.get_logger().warning("Instance segmentation found no cube mask in current image.")
            return None, full_scene_world, t_wc, None

        # Camera intrinsics K: fx, fy, cx, cy
        if self._camera_info is not None and len(self._camera_info.k) >= 9 and self._camera_info.k[0] > 0.0:
            fx = float(self._camera_info.k[0])
            fy = float(self._camera_info.k[4])
            cam_cx = float(self._camera_info.k[2])
            cam_cy = float(self._camera_info.k[5])
        else:
            hfov = 1.2
            fx = (img_w / 2.0) / math.tan(hfov / 2.0)
            fy = fx
            cam_cx = img_w / 2.0
            cam_cy = img_h / 2.0

        u = np.zeros(len(pts_cam), dtype=int)
        v = np.zeros(len(pts_cam), dtype=int)
        u[valid_depth] = np.round((x_opt[valid_depth] * fx / z_opt[valid_depth]) + cam_cx).astype(int)
        v[valid_depth] = np.round((y_opt[valid_depth] * fy / z_opt[valid_depth]) + cam_cy).astype(int)

        in_fov = valid_depth & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        # Mask lookup: retain ONLY points inside the 2D instance segmentation mask
        in_mask = np.zeros(len(pts_cam), dtype=bool)
        in_mask[in_fov] = seg_mask[v[in_fov], u[in_fov]] > 0

        # Filtered cube points
        cube_pts_raw = pts_world[in_mask]
        # Reject any stray background points outside the table elevation workspace [0.79, 0.95]
        cube_pts_world = cube_pts_raw[(cube_pts_raw[:, 2] > 0.79) & (cube_pts_raw[:, 2] < 0.95)]

        if len(cube_pts_world) < 20:
            self.get_logger().warning(
                f"Segmented cloud has only {len(cube_pts_world)} 3D points inside mask. Retrying..."
            )
            return None, full_scene_world, t_wc, None

        # Voxel downsampling (2.5mm) for balanced point distribution across top and sides
        _, uniq_cube = np.unique(np.floor(cube_pts_world / 0.0025).astype(int), axis=0, return_index=True)
        cube_pts_world = cube_pts_world[uniq_cube]

        kept_pct = 100.0 * len(cube_pts_world) / max(1, len(full_scene_world))
        self.get_logger().info(
            f"[Stage 2 Instance Segmentation Mask] Isolated {len(cube_pts_world)} pure cube points "
            f"out of {len(full_scene_world)} FOV points ({kept_pct:.1f}% kept)"
        )

        # Show ONLY the filtered cube point cloud + green segmentation wireframe in RViz2
        cube_mean = np.mean(cube_pts_world, axis=0)
        self._publish_filtering_viz(
            cube_pts_world,
            cube_mean,
            f"[Step 2/3] Instance Seg Filtered -> {len(cube_pts_world)} cube pts ({kept_pct:.1f}% of FOV)",
            cloud_rgb=(255, 70, 70),
            bbox_pts=cube_pts_world,
        )
        time.sleep(1.6)

        # Annotate & save snapshot for visual inspection
        try:
            annotated = cv_img.copy()
            if box is not None:
                x1, y1, x2, y2 = [int(p) for p in box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Overlay segmentation mask in translucent green
            mask_rgb = np.zeros_like(annotated)
            mask_rgb[seg_mask > 0] = [0, 255, 0]
            annotated = cv2.addWeighted(annotated, 0.75, mask_rgb, 0.25, 0)
            cv2.putText(
                annotated,
                f"Instance Seg: {len(cube_pts_world)} 3D points",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imwrite("/tmp/instance_segmentation_mask.jpg", annotated)
        except Exception:
            pass

        return cube_pts_world, full_scene_world, t_wc, seg_mask

    def _capture_full_scene_cloud(self, timeout_sec=4.0):
        """
        Phase 1 (Unfiltered Full-Scene Cloud):
        Grab the ENTIRE RGB-D PointCloud2 from /wrist_camera/points and transform it to
        the world frame WITHOUT removing the table plane or masking the cube.
        Applies only a light 4mm 3D voxel grid so KDTree & RViz2 stay responsive.
        Returns (full_scene_points_world, camera_position_world) or (None, None).
        """
        self._latest_cloud = None
        t0 = time.time()
        while self._latest_cloud is None and time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_cloud is None:
            self.get_logger().error("No PointCloud2 received on /wrist_camera/points.")
            return None, None

        msg = self._latest_cloud
        pts = self._pc2_to_xyz(msg)
        d = np.linalg.norm(pts, axis=1)
        pts = pts[(d > 0.05) & (d < 3.0)]
        R_wc, t_wc = self._lookup_world_tf(msg.header.frame_id)
        pts_w = pts @ R_wc.T + t_wc

        if len(pts_w) < 50:
            self.get_logger().error("PointCloud2 has insufficient valid 3D points.")
            return None, None

        # Keep the ENTIRE scene (table + cube + background), only 4mm voxel-downsampled
        _, uniq_idx = np.unique(np.floor(pts_w / 0.004).astype(int), axis=0, return_index=True)
        full_scene = pts_w[uniq_idx]
        self.get_logger().info(
            f"Full Unfiltered PointCloud2: {len(pts_w)} raw pts -> {len(full_scene)} voxel pts "
            f"(table + cube included, frame '{msg.header.frame_id}')"
        )
        return full_scene, t_wc

    def _visible_cad_points(self, center, yaw, cam_pos):
        """Transform the CAD model to (center, yaw) and return (visible_pts, outward_normals)."""
        Rz = R.from_euler("z", yaw).as_matrix()
        pts, norms = [], []
        for f_pts, f_n in self._cad_faces:
            n_w = Rz @ f_n
            if n_w[2] < -0.5:  # bottom face rests on the table -> never visible
                continue
            face_c = Rz @ (f_n * self._cad_half) + center
            v_cam = cam_pos - face_c
            v_cam /= max(np.linalg.norm(v_cam), 1e-6)
            if np.dot(n_w, v_cam) > 0.05:
                pw = f_pts @ Rz.T + center
                pts.append(pw)
                norms.append(np.tile(n_w, (len(pw), 1)))
        if not pts:
            return np.empty((0, 3)), np.empty((0, 3))
        return np.vstack(pts), np.vstack(norms)

    @staticmethod
    def _wrap_yaw(yaw):
        """Cube has 90 deg symmetry about Z -> wrap yaw into [-45, 45) deg."""
        return ((yaw + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0

    @staticmethod
    def _model_to_scene_rmse(source, tree_scene):
        """One-sided Model-to-Scene RMSE (mm) from visible CAD points to the full scene cloud."""
        d_st, idx = tree_scene.query(source)
        rmse = math.sqrt(float(np.mean(d_st ** 2))) * 1000.0
        return rmse, d_st, idx

    def _publish_icp_viz(self, target, source, matched, center, yaw, text, color, src_rgb=(40, 190, 255)):
        """Publish full target/source clouds, 3D STL CAD mesh marker, correspondence 'rubber bands', text and TF."""
        stamp = self.get_clock().now().to_msg()
        # Subsample full target cloud lightly for RViz2 marker rate if very large
        tgt_vis = target[::2] if len(target) > 12000 else target
        self._icp_target_pub.publish(self._xyz_to_pc2(tgt_vis, (220, 70, 70), "world", stamp))
        self._icp_source_pub.publish(self._xyz_to_pc2(source, src_rgb, "world", stamp))
        q = R.from_euler("z", yaw).as_quat()

        cube = Marker()
        cube.header.frame_id, cube.header.stamp = "world", stamp
        cube.ns, cube.id, cube.type, cube.action = "icp_cad_model", 0, Marker.MESH_RESOURCE, Marker.ADD
        cube.mesh_resource = "package://ur10e_perception_workcell/models/red_cube/meshes/red_cube.stl"
        cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = map(float, center)
        cube.pose.orientation.x, cube.pose.orientation.y = float(q[0]), float(q[1])
        cube.pose.orientation.z, cube.pose.orientation.w = float(q[2]), float(q[3])
        cube.scale.x = cube.scale.y = cube.scale.z = 1.0
        cube.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=0.65)

        lines = Marker()
        lines.header.frame_id, lines.header.stamp = "world", stamp
        lines.ns, lines.id, lines.type, lines.action = "icp_correspondences", 1, Marker.LINE_LIST, Marker.ADD
        lines.pose.orientation.w = 1.0
        lines.scale.x = 0.0012
        lines.color = ColorRGBA(r=1.0, g=0.9, b=0.1, a=0.85)
        if matched is not None and len(source):
            step = max(1, len(source) // 120)
            for s, t in zip(source[::step], matched[::step]):
                lines.points.append(Point(x=float(s[0]), y=float(s[1]), z=float(s[2])))
                lines.points.append(Point(x=float(t[0]), y=float(t[1]), z=float(t[2])))

        label = Marker()
        label.header.frame_id, label.header.stamp = "world", stamp
        label.ns, label.id, label.type, label.action = "icp_text", 2, Marker.TEXT_VIEW_FACING, Marker.ADD
        label.pose.position.x, label.pose.position.y = float(center[0]), float(center[1])
        label.pose.position.z = float(center[2]) + 0.10
        label.pose.orientation.w = 1.0
        label.scale.z = 0.024
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = text

        del_bbox = Marker()
        del_bbox.header.frame_id, del_bbox.header.stamp = "world", stamp
        del_bbox.ns, del_bbox.id, del_bbox.action = "seg_bbox", 3, Marker.DELETE

        ma = MarkerArray()
        ma.markers = [cube, lines, label, del_bbox]
        self._icp_marker_pub.publish(ma)

        tfm = TransformStamped()
        tfm.header.frame_id, tfm.header.stamp, tfm.child_frame_id = "world", stamp, "icp_cube_pose"
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, center)
        tfm.transform.rotation.x, tfm.transform.rotation.y = float(q[0]), float(q[1])
        tfm.transform.rotation.z, tfm.transform.rotation.w = float(q[2]), float(q[3])
        self._tf_broadcaster.sendTransform(tfm)

    def _publish_filtering_viz(self, cloud_pts, text_pos, text, cloud_rgb=(220, 70, 70), bbox_pts=None):
        """Publish target cloud and HUD markers during Stage 2 FOV capture & Instance Segmentation filtering."""
        stamp = self.get_clock().now().to_msg()
        vis_pts = cloud_pts[::2] if len(cloud_pts) > 14000 else cloud_pts
        self._icp_target_pub.publish(self._xyz_to_pc2(vis_pts, cloud_rgb, "world", stamp))
        # Clear source cloud while showing raw FOV or segmented cloud
        self._icp_source_pub.publish(self._xyz_to_pc2(np.empty((0, 3)), (40, 190, 255), "world", stamp))

        del_cube = Marker()
        del_cube.header.frame_id, del_cube.header.stamp = "world", stamp
        del_cube.ns, del_cube.id, del_cube.action = "icp_cad_model", 0, Marker.DELETE

        del_lines = Marker()
        del_lines.header.frame_id, del_lines.header.stamp = "world", stamp
        del_lines.ns, del_lines.id, del_lines.action = "icp_correspondences", 1, Marker.DELETE

        label = Marker()
        label.header.frame_id, label.header.stamp = "world", stamp
        label.ns, label.id, label.type, label.action = "icp_text", 2, Marker.TEXT_VIEW_FACING, Marker.ADD
        label.pose.position.x = float(text_pos[0])
        label.pose.position.y = float(text_pos[1])
        label.pose.position.z = float(text_pos[2]) + 0.12
        label.pose.orientation.w = 1.0
        label.scale.z = 0.026
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = text

        bbox_marker = Marker()
        bbox_marker.header.frame_id, bbox_marker.header.stamp = "world", stamp
        bbox_marker.ns, bbox_marker.id = "seg_bbox", 3
        if bbox_pts is not None and len(bbox_pts) >= 4:
            bbox_marker.type, bbox_marker.action = Marker.LINE_LIST, Marker.ADD
            bbox_marker.pose.orientation.w = 1.0
            bbox_marker.scale.x = 0.0022
            bbox_marker.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.95)
            mn = np.min(bbox_pts, axis=0) - 0.006
            mx = np.max(bbox_pts, axis=0) + 0.006
            c = [
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i, j in edges:
                bbox_marker.points.append(Point(x=float(c[i][0]), y=float(c[i][1]), z=float(c[i][2])))
                bbox_marker.points.append(Point(x=float(c[j][0]), y=float(c[j][1]), z=float(c[j][2])))
        else:
            bbox_marker.action = Marker.DELETE

        ma = MarkerArray()
        ma.markers = [del_cube, del_lines, label, bbox_marker]
        self._icp_marker_pub.publish(ma)

    def estimate_cube_pose_segmented_icp(self):
        """
        Stage 2 (Instance Segmentation Point Cloud Filter + 100% Pure ICP Matching):
          1. Captures camera image and PointCloud2, applies 2D instance segmentation mask,
             and isolates ONLY the 3D points belonging to the red cube (zero table points!).
          2. Directly initializes the CAD model at the segmented point cloud centroid:
             no brute force table sweep needed!
          3. Runs Symmetric Point-to-Plane ICP between the 3D CAD model (red_cube.stl)
             and the segmented cube point cloud for 100% accurate 6D pose estimation (X, Y, Z, Yaw).
        Returns: (x, y, z, yaw) in world frame or None.
        """
        if not self._cad_faces:
            return None

        cube_pts, full_scene, cam_pos, seg_mask = self._capture_segmented_cube_cloud()
        if cube_pts is None or len(cube_pts) < 20:
            self.get_logger().error("Stage 2: Failed to isolate cube point cloud from instance segmentation mask.")
            return None

        t_start = time.time()
        delay = self._icp_step_delay
        tree_target = cKDTree(cube_pts)

        # Direct Centroid Initialization (No Brute-Force Sweep)
        mean_pt = np.mean(cube_pts, axis=0)
        # Table top is at Z=0.80, cube edge is 0.05m -> nominal center Z is 0.825m
        center = np.array([float(mean_pt[0]), float(mean_pt[1]), 0.825])
        yaw = 0.0

        # Quick direct yaw orientation alignment around centroid
        best_yaw = 0.0
        best_init_rmse = float("inf")
        for deg in range(-45, 45, 5):
            cand_yaw = math.radians(deg)
            src_init, _ = self._visible_cad_points(center, cand_yaw, cam_pos)
            if not len(src_init):
                continue
            dists_init, _ = tree_target.query(src_init)
            init_rmse = math.sqrt(float(np.mean(dists_init ** 2))) * 1000.0
            if init_rmse < best_init_rmse:
                best_init_rmse = init_rmse
                best_yaw = cand_yaw

        yaw = best_yaw
        src_init, _ = self._visible_cad_points(center, yaw, cam_pos)
        self.get_logger().info(
            f"[Stage 2 ICP Init] Direct centroid: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
            f"Init Yaw: {math.degrees(yaw):+.1f} deg | Init RMSE: {best_init_rmse:.2f} mm"
        )
        self._publish_icp_viz(
            cube_pts, src_init, None, center, yaw,
            f"[Stage 2 Seg-ICP Init] Direct Centroid Init | RMSE={best_init_rmse:.2f} mm",
            (1.0, 0.75, 0.1),
            src_rgb=(255, 180, 20),
        )
        time.sleep(delay * 0.8)

        # Iterative Closest Point (ICP) directly on segmented cube points
        n_iters = max(self._icp_max_iters, 25)
        prev_rmse = float("inf")
        rmse = best_init_rmse

        for it in range(1, n_iters + 1):
            src, norms = self._visible_cad_points(center, yaw, cam_pos)
            if not len(src):
                break

            # Symmetric correspondences: CAD -> Cube points and Cube -> CAD points
            dists_st, idx_t = tree_target.query(src)
            matched_pts = cube_pts[idx_t]

            # Bidirectional distance check for robust outlier rejection
            tree_src = cKDTree(src)
            dists_ts, _ = tree_src.query(cube_pts)
            sym_rmse = math.sqrt(0.5 * (float(np.mean(dists_st ** 2)) + float(np.mean(dists_ts ** 2)))) * 1000.0
            rmse = sym_rmse

            # Point-to-Plane linear system
            keep = dists_st < max(2.5 * float(np.median(dists_st)), 0.005)
            s_k, t_k, n_k = src[keep], matched_pts[keep], norms[keep]

            rel = s_k - center
            rot_jac = (-rel[:, 1] * n_k[:, 0]) + (rel[:, 0] * n_k[:, 1])
            A_pl = np.column_stack([n_k[:, 0], n_k[:, 1], n_k[:, 2], rot_jac])
            b_pl = np.sum((t_k - s_k) * n_k, axis=1)

            # Center-of-mass anchor to prevent planar sliding
            mu_s = s_k.mean(axis=0)
            mu_t = t_k.mean(axis=0)
            w_com = 0.20
            A_com = np.array([
                [w_com, 0.0, 0.0, 0.0],
                [0.0, w_com, 0.0, 0.0],
                [0.0, 0.0, w_com, 0.0]
            ])
            b_com = (mu_t - mu_s) * w_com

            A = np.vstack([A_pl, A_com])
            b = np.concatenate([b_pl, b_com])

            ATA = A.T @ A + np.diag([1e-4, 1e-4, 1e-4, 1e-5])
            delta = np.linalg.solve(ATA, A.T @ b)

            alpha = 0.65
            center = center + alpha * delta[:3]
            # Keep Z physically resting on table
            center[2] = float(np.clip(center[2], 0.818, 0.835))
            yaw = self._wrap_yaw(yaw + alpha * delta[3])

            self._publish_icp_viz(
                cube_pts, src, matched_pts, center, yaw,
                f"[Stage 2 Seg-ICP {it}/{n_iters}] RMSE={rmse:.2f} mm | Pos=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) | Yaw={math.degrees(yaw):+.1f} deg",
                (0.10, 0.70, 1.0),
                src_rgb=(30, 200, 255),
            )
            time.sleep(delay)

            if it > 8 and abs(prev_rmse - rmse) < 0.005:
                break
            prev_rmse = rmse

        # Final converged state
        src_final, _ = self._visible_cad_points(center, yaw, cam_pos)
        dists_final, idx_final = tree_target.query(src_final)
        tree_src_final = cKDTree(src_final)
        dists_ts_final, _ = tree_src_final.query(cube_pts)
        final_rmse = math.sqrt(0.5 * (float(np.mean(dists_final ** 2)) + float(np.mean(dists_ts_final ** 2)))) * 1000.0
        elapsed_compute = time.time() - t_start

        ok = final_rmse < 5.0
        self._publish_icp_viz(
            cube_pts, src_final, cube_pts[idx_final], center, yaw,
            f"[Stage 2 Seg-ICP {'LOCKED (100% MATCH)' if ok else 'POOR FIT'}] RMSE={final_rmse:.2f} mm | "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Yaw={math.degrees(yaw):+.1f} deg",
            (0.05, 0.95, 0.20) if ok else (1.0, 0.1, 0.1),
            src_rgb=(40, 255, 90) if ok else (255, 40, 40),
        )
        time.sleep(1.2)
        self.get_logger().info(
            f"Stage 2 Segmented ICP finished (compute time {elapsed_compute:.2f}s): "
            f"X={center[0]:.4f} Y={center[1]:.4f} Z={center[2]:.4f} m, "
            f"yaw={math.degrees(yaw):+.2f} deg, RMSE={final_rmse:.2f} mm ({'LOCKED' if ok else 'REJECTED'})"
        )
        if not ok:
            return None
        return float(center[0]), float(center[1]), float(center[2]), float(yaw)

    def estimate_cube_pose_icp(self):
        """
        Phase 1 — Full Unfiltered PointCloud Brute-Force + ICP Pose Estimation:
          A) Takes the ENTIRE camera PointCloud2 (table + cube + scene, NO filtering/masking!).
          B) Slides the 3D CAD cube across a 3D grid over the full scene cloud (Brute-Force search),
             visualizing how the CAD model scans across the table until its 3D faces match the protruding cube.
          C) Runs Point-to-Plane ICP directly against the full unfiltered scene cloud to refine (x, y, z, yaw).
        Returns (x, y, z, yaw) in world frame or None.
        """
        if not self._cad_faces:
            return None
        full_scene, cam_pos = self._capture_full_scene_cloud()
        if full_scene is None:
            return None

        t_start = time.time()
        tree_scene = cKDTree(full_scene)
        delay = self._icp_step_delay

        # Compute 3D search volume directly from the observed full camera cloud percentiles
        p_min = np.percentile(full_scene, 5, axis=0)
        p_max = np.percentile(full_scene, 95, axis=0)
        z_top = float(np.max(full_scene[:, 2]))
        xs = np.arange(p_min[0] + 0.04, p_max[0] - 0.04, 0.035)
        ys = np.arange(p_min[1] + 0.04, p_max[1] - 0.04, 0.035)
        zs = np.unique(np.round(np.linspace(p_min[2] + 0.02, max(p_min[2] + 0.03, z_top - 0.015), 3), 3))

        best = (float("inf"), np.array([float(np.median(full_scene[:, 0])), 0.0, 0.825]), 0.0)
        total_evals = 0
        scan_step = 0
        total_grid = max(1, len(xs) * len(ys) * len(zs))
        viz_stride = max(1, total_grid // 18)  # Show ~18 live brute-force grid steps across the table

        self.get_logger().info(
            f"[Phase 1 Full-Cloud Brute-Force] Scanning {total_grid} 3D grid positions across {len(full_scene)} scene points..."
        )
        for x in xs:
            for y in ys:
                for z in zs:
                    cand_c = np.array([float(x), float(y), float(z)])
                    src, _ = self._visible_cad_points(cand_c, 0.0, cam_pos)
                    if not len(src):
                        continue
                    rmse, _, idx = self._model_to_scene_rmse(src, tree_scene)
                    total_evals += 1
                    scan_step += 1
                    is_new_best = rmse < best[0]
                    if is_new_best:
                        best = (rmse, cand_c.copy(), 0.0)
                    if scan_step % viz_stride == 0 or is_new_best:
                        self._publish_icp_viz(
                            full_scene, src, None, cand_c, 0.0,
                            f"[Full-Cloud Brute-Force {scan_step}/{total_grid}] RMSE={rmse:.1f} mm (best={best[0]:.1f} mm)",
                            (1.0, 0.55, 0.05),
                            src_rgb=(255, 165, 30),
                        )
                        time.sleep(delay * 0.45)

        # Fine local 3D + Yaw brute-force sweep around the best 3D region
        b_center = best[1].copy()
        for dx in (-0.015, 0.0, 0.015):
            for dy in (-0.015, 0.0, 0.015):
                for dz in (-0.008, 0.0, 0.008):
                    cand_c = b_center + np.array([dx, dy, dz])
                    for deg in range(-45, 45, 5):
                        yaw_c = math.radians(deg)
                        src, _ = self._visible_cad_points(cand_c, yaw_c, cam_pos)
                        if not len(src):
                            continue
                        rmse, _, idx = self._model_to_scene_rmse(src, tree_scene)
                        total_evals += 1
                        if rmse < best[0]:
                            best = (rmse, cand_c.copy(), yaw_c)
                            self._publish_icp_viz(
                                full_scene, src, None, cand_c, yaw_c,
                                f"[Brute-Force Yaw {deg:+d} deg] New Best RMSE={rmse:.2f} mm",
                                (1.0, 0.70, 0.10),
                                src_rgb=(255, 190, 40),
                            )
                            time.sleep(delay * 0.45)

        COARSE_THRESHOLD_MM = 11.0
        self.get_logger().info(
            f"Full-Cloud Brute-Force evaluated {total_evals} poses: "
            f"best=({best[1][0]:.3f}, {best[1][1]:.3f}, {best[1][2]:.3f}), yaw={math.degrees(best[2]):+.1f} deg, "
            f"RMSE={best[0]:.2f} mm (coarse threshold={COARSE_THRESHOLD_MM:.1f} mm)"
        )

        if best[0] >= COARSE_THRESHOLD_MM:
            src, _ = self._visible_cad_points(best[1], best[2], cam_pos)
            self._publish_icp_viz(
                full_scene, src, None, best[1], best[2],
                f"[REJECTED - NO CUBE IN FOV] Best RMSE={best[0]:.1f} mm >= {COARSE_THRESHOLD_MM:.0f} mm",
                (1.0, 0.1, 0.1),
                src_rgb=(255, 40, 40),
            )
            time.sleep(1.2)
            self.get_logger().error(
                f"Brute-Force best RMSE ({best[0]:.2f} mm) exceeds coarse threshold ({COARSE_THRESHOLD_MM:.1f} mm) — cube not in FOV!"
            )
            return None

        # --- Stage C: Iterative Closest Point (ICP) directly on the FULL unfiltered scene cloud ---
        center = best[1].copy()
        yaw = best[2]
        n_iters = max(self._icp_max_iters, 20)
        prev_rmse = float("inf")
        rmse = best[0]

        for it in range(1, n_iters + 1):
            src, norms = self._visible_cad_points(center, yaw, cam_pos)
            if not len(src):
                break
            rmse, dists, idx_t = self._model_to_scene_rmse(src, tree_scene)
            matched = full_scene[idx_t]

            keep = dists < max(2.0 * float(np.median(dists)), 0.006)
            s_k, t_k, n_k = src[keep], matched[keep], norms[keep]
            rel = s_k - center
            rot_jac = (-rel[:, 1] * n_k[:, 0]) + (rel[:, 0] * n_k[:, 1])
            A = np.column_stack([n_k[:, 0], n_k[:, 1], n_k[:, 2], rot_jac])
            b = np.sum((t_k - s_k) * n_k, axis=1)
            ATA = A.T @ A + np.diag([1e-3, 1e-3, 1e-3, 1e-4])
            delta = np.linalg.solve(ATA, A.T @ b)

            alpha = 0.50
            center = center + alpha * delta[:3]
            yaw = self._wrap_yaw(yaw + alpha * delta[3])

            self._publish_icp_viz(
                full_scene, src, matched, center, yaw,
                f"[Full-Cloud ICP {it}/{n_iters}] RMSE={rmse:.2f} mm | Pos=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) | Yaw={math.degrees(yaw):+.1f} deg",
                (0.10, 0.65, 1.0),
                src_rgb=(40, 190, 255),
            )
            time.sleep(delay)
            if it > 10 and abs(prev_rmse - rmse) < 0.005:
                break
            prev_rmse = rmse

        src, _ = self._visible_cad_points(center, yaw, cam_pos)
        rmse, _, idx = self._model_to_scene_rmse(src, tree_scene)
        elapsed_compute = time.time() - t_start
        ok = rmse < 8.0
        self._publish_icp_viz(
            full_scene, src, full_scene[idx], center, yaw,
            f"[Full-Cloud ICP {'LOCKED' if ok else 'POOR FIT'}] RMSE={rmse:.2f} mm | "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Yaw={math.degrees(yaw):+.1f} deg",
            (0.05, 0.92, 0.25) if ok else (1.0, 0.1, 0.1),
            src_rgb=(40, 255, 90) if ok else (255, 40, 40),
        )
        time.sleep(1.2)
        self.get_logger().info(
            f"Full-Cloud ICP finished ({total_evals} BF evals, total time {elapsed_compute:.2f}s): "
            f"X={center[0]:.4f} Y={center[1]:.4f} Z={center[2]:.4f} m, "
            f"yaw={math.degrees(yaw):+.2f} deg, RMSE={rmse:.2f} mm ({'OK' if ok else 'REJECTED'})"
        )
        if not ok:
            return None
        return float(center[0]), float(center[1]), float(center[2]), float(yaw)

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
                self.get_logger().warning("Close-up motion failed. Retaining coarse target.")
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
                self.get_logger().warning(f"Failed to save close-up snapshot: {e}")

            return fine_x, fine_y, target_z
        else:
            self.get_logger().warning("Cube not detected in close-up view. Retaining coarse target.")
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
            gz_bin = "/opt/ros/lyrical/opt/gz_tools_vendor/bin/gz"
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

        if self._observe_only:
            self.get_logger().info("[Observe-Only Mode] Moving directly to Angled Observation Pose (OBSERVE_CENTER)...")
            if not self.send_exact_trajectory("OBSERVE_CENTER", POSES["OBSERVE_CENTER"], duration_sec=3.0):
                self.send_pose_goal("OBSERVE_CENTER", POSES["OBSERVE_CENTER"], velocity_scaling=0.30)
            self.get_logger().info("Robot is now holding at OBSERVE_CENTER (~40 deg angled 3D view of the red cube). Ready for RViz2 inspection!")
            return True

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

        obs_delta = max(abs(a - b) for a, b in zip(self._current_joints, POSES["OBSERVE_CENTER"]))
        if obs_delta > 0.08:
            # Move to HOME then to angled observation pose OBSERVE_CENTER
            self.get_logger().info("[Step 1] Moving to HOME")
            if not self.send_pose_goal("HOME", POSES["HOME"], velocity_scaling=0.25):
                return False
            time.sleep(0.3)

            self.get_logger().info("[Step 2] Moving to angled observation pose OBSERVE_CENTER")
            if not self.send_exact_trajectory("OBSERVE_CENTER", POSES["OBSERVE_CENTER"], duration_sec=3.0):
                if not self.send_pose_goal("OBSERVE_CENTER", POSES["OBSERVE_CENTER"], velocity_scaling=0.30):
                    return False
            time.sleep(1.0)
        else:
            self.get_logger().info("[Step 1-2] Robot already at angled observation pose OBSERVE_CENTER; starting ICP scan immediately.")
            time.sleep(0.4)

        self.get_logger().info(f"[Step 3] 3D ICP pose estimation (mode: '{self._icp_mode}')")
        if self._icp_mode == "segmented":
            icp = self.estimate_cube_pose_segmented_icp()
        else:
            icp = self.estimate_cube_pose_icp()
        if icp is None:
            self.get_logger().error("ICP pose estimation failed. Aborting sequence.")
            self.send_pose_goal("HOME", POSES["HOME"], velocity_scaling=0.25)
            return False
        cube_x, cube_y, cube_z, cube_yaw = icp
        cube_z = float(np.clip(cube_z, 0.815, 0.840))

        # Update detected cube pose in MoveIt PlanningScene
        self.update_cube_in_planning_scene(cube_x, cube_y, cube_z, cube_yaw)

        # Compute trajectory waypoints with sequential seeding (gripper rotated by cube yaw)
        self.get_logger().info(
            f"[Step 3b] Computing grasp kinematics for target: X={cube_x:.3f}, Y={cube_y:.3f}, "
            f"Z={cube_z:.3f}, yaw={math.degrees(cube_yaw):+.1f} deg"
        )
        pre_grasp_pos = [cube_x, cube_y, 0.980]
        grasp_pos     = [cube_x, cube_y, cube_z]
        lift_pos      = [cube_x, cube_y, 0.980]

        place_x = cube_x
        place_y = round(cube_y - 0.24, 3) if cube_y >= 0.0 else round(cube_y + 0.24, 3)
        place_approach_pos = [place_x, place_y, 0.980]
        place_pos          = [place_x, place_y, 0.825]

        q_curr = list(self._current_joints)
        q_pre_grasp = self.solve_ik_target(pre_grasp_pos, q_seed=q_curr, yaw=cube_yaw)
        q_grasp     = self.solve_ik_target(grasp_pos, q_seed=q_pre_grasp, yaw=cube_yaw)
        q_lift      = self.solve_ik_target(lift_pos, q_seed=q_grasp, yaw=cube_yaw)
        q_place_app = self.solve_ik_target(place_approach_pos, q_seed=q_lift, yaw=cube_yaw)
        q_place     = self.solve_ik_target(place_pos, q_seed=q_place_app, yaw=cube_yaw)

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
            gz_bin = "/opt/ros/lyrical/opt/gz_tools_vendor/bin/gz"
            if os.path.exists(gz_bin):
                subprocess.run(
                    [gz_bin, "topic", "-t", "/cube/attach", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=2.0, capture_output=True
                )
        except Exception:
            pass
        self.attach_cube_in_planning_scene()
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
            gz_bin = "/opt/ros/lyrical/opt/gz_tools_vendor/bin/gz"
            if os.path.exists(gz_bin):
                subprocess.run(
                    [gz_bin, "topic", "-t", "/cube/detach", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=2.0, capture_output=True
                )
        except Exception:
            pass
        self.detach_cube_in_planning_scene(place_x, place_y, 0.825)
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
