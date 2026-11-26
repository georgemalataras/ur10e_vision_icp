#!/usr/bin/env python3
"""
UR10e Multi-Object FoundationPose Pick-and-Place Node.

Workflow:
1. Starts persistent NVIDIA FoundationPose inference server (CUDA conda env).
2. Moves UR10e to an elevated, angled observation pose with all 6 CAD parts in FOV.
3. Captures RGB-D (/wrist_camera/image, /wrist_camera/depth_image, /wrist_camera/camera_info).
4. Segments the 6 tabletop CAD parts and calls FoundationPose for 6D pose estimation.
5. Publishes 3D Bounding Boxes (/foundationpose/markers) and TF frames for all 6 parts.
6. Opens an interactive UI showing the 2x3 table layout (with 3D CAD images + names)
   asking: "Which part do you want to pick up?"
7. Picks the chosen part, lifts it to Z=0.980m, holds for 2.5 seconds, and places it back
   at its original pose on the table.
"""

import json
import math
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from PIL import Image as PILImage, ImageTk

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory, GripperCommand
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import ColorRGBA, Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


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
    # Exact same camera Z-axis / viewing angle as OBSERVE_ANGLED, elevated to cam Z=1.45m
    # so all 6 grid parts sit cleanly inside the 640x480 FOV
    "OBSERVE_MULTI": [-1.2205, -1.9464, 1.5446, -0.9462, -0.9089, -0.821],
}

FP_PYTHON = "/home/lms/miniforge3/envs/foundationpose/bin/python3"
FP_HOST = "127.0.0.1"
FP_PORT = 5555


class CalibratedKinematics:
    """Forward and inverse kinematics matching ur10e_workcell.urdf.xacro."""

    def __init__(self):
        self.pedestal_z = 0.75
        self.R_target = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)

    def _tf_mat(self, xyz, rpy, axis_rot=0.0):
        T = np.eye(4)
        rot = R.from_euler("xyz", rpy).as_matrix()
        if axis_rot != 0.0:
            rot = rot @ R.from_euler("z", axis_rot).as_matrix()
        T[:3, :3] = rot
        T[:3, 3] = xyz
        return T

    def fk_tool0(self, q):
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
        T = self.fk_tool0(q)
        T = T @ self._tf_mat([0, 0, 0.011], [0, 0, 1.570796])
        T = T @ self._tf_mat([0, 0, 0.14], [0, 0, 0])
        return T

    def fk_cam_opt(self, q):
        T = self.fk_tool0(q)
        T = T @ self._tf_mat([0.095, 0, 0.135], [0, -math.pi / 2.0, 0])
        T = T @ self._tf_mat([0, 0, 0], [-math.pi / 2.0, 0, -math.pi / 2.0])
        return T

    def solve_ik_target(self, target_pos, q_seed, yaw=0.0):
        R_des = R.from_euler("z", yaw).as_matrix() @ self.R_target
        q_seed = np.asarray(q_seed, dtype=float)

        def cost(q):
            T = self.fk_tcp(q)
            pos_err = np.sum((T[:3, 3] - target_pos) ** 2)
            rot_err = np.sum((T[:3, :3] - R_des) ** 2)
            reg = 1e-4 * np.sum((q - q_seed) ** 2)
            return 1000.0 * pos_err + 50.0 * rot_err + reg

        bounds = [(-2 * math.pi, 2 * math.pi)] * 6
        res = minimize(cost, q_seed, method="SLSQP", bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
        return res.x.tolist()


def robotiq_gap_to_knuckle(gap_m: float) -> float:
    """Invert exact Robotiq 2F-85 URDF linkage kinematics: inner pad gap (m) -> knuckle angle (rad)."""
    qs = np.linspace(0.0, 0.79, 2000)
    # From URDF: x_tip(q) = 0.03060114 + 0.0371575*cos(q) - 0.04342168*sin(q)
    # Inner pad surface offset = 0.02526m (gives 0.085m gap at q=0.0, ~0.001m at q=0.79)
    gaps = 2.0 * (0.03060114 + 0.0371575 * np.cos(qs) - 0.04342168 * np.sin(qs) - 0.02526)
    idx = int(np.argmin(np.abs(gaps - float(gap_m))))
    return float(qs[idx])


def robotiq_tip_z_extension(q: float) -> float:
    """Exact downward Z extension (m) of Robotiq 2F-85 fingertips as knuckle closes from 0 to q."""
    return (0.0371575 * math.sin(float(q)) + 0.04342168 * math.cos(float(q))) - 0.04342168


def load_stl_surface_points(stl_path: str, n_samples: int = 3500) -> np.ndarray:
    """Uniformly area-sample 3D surface points from a binary STL (pure numpy)."""
    with open(stl_path, "rb") as f:
        f.read(80)
        n_tris = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n_tris * 50)
    dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(raw, dtype=dt)["v"].astype(np.float64)  # (N, 3, 3)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / np.maximum(areas.sum(), 1e-12)
    rng = np.random.default_rng(42)
    tri_idx = rng.choice(len(tris), size=n_samples, p=probs)
    r1 = np.sqrt(rng.random(n_samples))[:, None]
    r2 = rng.random(n_samples)[:, None]
    pts = (1.0 - r1) * v0[tri_idx] + r1 * (1.0 - r2) * v1[tri_idx] + r1 * r2 * v2[tri_idx]
    return pts


def plan_grasp_from_points(pts_world: np.ndarray) -> dict | None:
    """
    Shape-agnostic 3D cross-section grasp planner (works for both known CAD parts
    and unseen objects via live RGB-D point cloud).
    Finds the optimal yaw, symmetric grasp center, exact cross-section width between
    the Robotiq 2F-85 finger pads, and exact knuckle angle for flush zero-penetration contact.
    """
    if pts_world is None or len(pts_world) < 15:
        return None

    z_min = max(0.800, float(np.percentile(pts_world[:, 2], 2)))
    z_max = float(np.percentile(pts_world[:, 2], 98))
    grasp_z = float(np.clip(0.5 * (z_min + z_max), 0.814, 0.838))

    # Extract 3D cross-section within vertical span of Robotiq finger pads
    z_mask = (pts_world[:, 2] >= grasp_z - 0.020) & (pts_world[:, 2] <= grasp_z + 0.024)
    slice_pts = pts_world[z_mask] if np.sum(z_mask) >= 20 else pts_world
    xy = slice_pts[:, :2]

    best = None
    for deg in range(-90, 90, 5):
        yaw = math.radians(deg)
        # Verified tcp_link X-axis (finger closing axis) and Y-axis (pad width axis)
        u_close = np.array([-math.sin(yaw), math.cos(yaw)])
        u_pad = np.array([math.cos(yaw), math.sin(yaw)])

        p_coords = xy @ u_pad
        p_mid = 0.5 * (float(np.min(p_coords)) + float(np.max(p_coords)))
        in_pad = np.abs(p_coords - p_mid) <= 0.015  # 30mm Robotiq finger pad width
        if np.sum(in_pad) < 10:
            continue

        c_coords = xy[in_pad] @ u_close
        c_min = float(np.min(c_coords))
        c_max = float(np.max(c_coords))
        width = c_max - c_min
        if width < 0.004 or width > 0.078:
            continue

        c_mid = 0.5 * (c_min + c_max)
        grasp_xy = c_mid * u_close + p_mid * u_pad
        score = width + 0.002 * abs(yaw)
        if best is None or score < best["score"]:
            q_close = robotiq_gap_to_knuckle(max(0.0, width - 0.0005))
            # Compensate Robotiq 2F-85 4-bar linkage downward Z extension when closing
            # so fingertips ALWAYS stay >= 6mm above the tabletop (Z=0.800m) even when closed
            z_min_safe = 0.800 + 0.006 + 0.013 + robotiq_tip_z_extension(q_close)
            safe_grasp_z = float(max(grasp_z, z_min_safe))
            best = {
                "score": score,
                "grasp_x": float(grasp_xy[0]),
                "grasp_y": float(grasp_xy[1]),
                "grasp_z": safe_grasp_z,
                "yaw": yaw,
                "width": width,
                "q_close": q_close,
            }
    return best


def fp_rpc(payload: dict, timeout: float = 30.0) -> dict:
    s = socket.create_connection((FP_HOST, FP_PORT), timeout=timeout)
    try:
        data = pickle.dumps(payload, protocol=4)
        s.sendall(struct.pack(">I", len(data)) + data)
        raw_len = bytearray()
        while len(raw_len) < 4:
            chunk = s.recv(4 - len(raw_len))
            if not chunk:
                raise ConnectionError("FoundationPose server closed connection")
            raw_len.extend(chunk)
        (msg_len,) = struct.unpack(">I", bytes(raw_len))
        buf = bytearray()
        while len(buf) < msg_len:
            chunk = s.recv(min(65536, msg_len - len(buf)))
            if not chunk:
                raise ConnectionError("Incomplete response from FoundationPose server")
            buf.extend(chunk)
        return pickle.loads(bytes(buf))
    finally:
        s.close()


class PartSelectionUI:
    """Tkinter UI showing the 2x3 table layout with 3D CAD images and names."""

    def __init__(self, manifest: list, detected_poses: dict, pkg_share: str):
        self.manifest = manifest
        self.detected_poses = detected_poses
        self.pkg_share = pkg_share
        self.selected_part_id = None

    def show(self) -> str | None:
        root = tk.Tk()
        root.title("UR10e FoundationPose — Interactive Pick & Place")
        root.configure(bg="#181b20")
        root.attributes("-topmost", True)

        # Header
        header = tk.Label(
            root,
            text="Which part do you want to pick up?",
            font=("DejaVu Sans", 16, "bold"),
            fg="#ffffff",
            bg="#181b20",
            pady=14,
        )
        header.pack()

        sub = tk.Label(
            root,
            text="2 × 3 Table Layout (Estimated via NVIDIA FoundationPose RGB-D)",
            font=("DejaVu Sans", 10),
            fg="#9aa5b4",
            bg="#181b20",
        )
        sub.pack(pady=(0, 10))

        grid_frame = tk.Frame(root, bg="#232832", padx=16, pady=16, relief=tk.RIDGE, bd=2)
        grid_frame.pack(padx=20, pady=8)

        self._tk_images = []

        def on_select(pid: str):
            self.selected_part_id = pid
            root.destroy()

        def on_exit():
            self.selected_part_id = None
            root.destroy()

        # Display Back row (row=1, further from robot) on top, Front row (row=0) on bottom
        # Left (col=0, +Y) on left, Right (col=2, -Y) on right
        for item in self.manifest:
            pid = item["id"]
            ui_row = 1 - item["row"]
            ui_col = item["col"]

            card = tk.Frame(grid_frame, bg="#2c323f", padx=10, pady=10, relief=tk.RAISED, bd=2)
            card.grid(row=ui_row, column=ui_col, padx=10, pady=10, sticky="nsew")

            thumb_path = os.path.join(self.pkg_share, item["thumbnail"])
            pil_img = PILImage.open(thumb_path).resize((150, 150), PILImage.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            self._tk_images.append(tk_img)

            pose_info = self.detected_poses.get(pid)
            if pose_info is not None:
                px, py, pz = pose_info["world_pos"]
                status_txt = f"6D Pose: ({px:.3f}, {py:.3f}, {pz:.3f}) m"
                status_fg = "#5eead4"
                state = tk.NORMAL
            else:
                status_txt = "Not detected in view"
                status_fg = "#f87171"
                state = tk.DISABLED

            btn = tk.Button(
                card,
                image=tk_img,
                text=f"\n{item['display_name']}",
                compound=tk.TOP,
                font=("DejaVu Sans", 11, "bold"),
                fg="#ffffff",
                bg="#353d4d",
                activebackground="#4c5870",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                padx=8,
                pady=8,
                cursor="hand2",
                state=state,
                command=lambda p=pid: on_select(p),
            )
            btn.pack(fill=tk.BOTH, expand=True)

            lbl = tk.Label(
                card,
                text=status_txt,
                font=("DejaVu Sans Mono", 8),
                fg=status_fg,
                bg="#2c323f",
                pady=4,
            )
            lbl.pack()

        exit_btn = tk.Button(
            root,
            text="Finish & Return Home",
            font=("DejaVu Sans", 10, "bold"),
            fg="#ffffff",
            bg="#475569",
            activebackground="#64748b",
            padx=16,
            pady=6,
            command=on_exit,
        )
        exit_btn.pack(pady=(8, 16))

        root.protocol("WM_DELETE_WINDOW", on_exit)
        root.mainloop()
        return self.selected_part_id


class FoundationPosePickPlaceNode(Node):
    def __init__(self):
        super().__init__("ur10e_foundationpose_pick_place_node")
        self.declare_parameter("observe_only", False)
        self.observe_only = bool(self.get_parameter("observe_only").value)

        self.pkg_share = get_package_share_directory("ur10e_perception_workcell")
        manifest_path = os.path.join(self.pkg_share, "models", "parts_manifest.json")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.parts_by_id = {p["id"]: p for p in self.manifest}

        self.kin = CalibratedKinematics()
        self.bridge = CvBridge()

        self._current_joints = None
        self._gripper_joint_pos = 0.0
        self._latest_rgb = None
        self._latest_depth = None
        self._camera_info = None
        self._detected_poses = {}
        self._part_clouds_world = {}
        self._fp_proc = None
        self._attached_part_id = None
        self._attached_T_tcp_bottom = None
        self._attached_T_tcp_center = None

        # Initialize always-visible world poses for all 6 CAD parts from manifest
        # (so all 6 parts appear in RViz immediately even before entering camera FOV)
        self._part_bottom_poses = {}
        for item in self.manifest:
            T_init = np.eye(4, dtype=float)
            T_init[:3, 3] = [float(item["x"]), float(item["y"]), float(item["z"])]
            self._part_bottom_poses[item["id"]] = T_init

        # Subscribers
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Image, "/wrist_camera/image", self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/wrist_camera/depth_image", self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/wrist_camera/camera_info", self._cam_info_cb, qos_profile_sensor_data)

        # Publishers: Always-visible CAD meshes + 3D Bounding Box MarkerArray + TF Broadcaster + Part Attach/Detach
        self._parts_visual_pub = self.create_publisher(MarkerArray, "/workcell/parts_visual", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/foundationpose/markers", 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._attach_pubs = {}
        self._detach_pubs = {}
        for item in self.manifest:
            pid = item["id"]
            self._attach_pubs[pid] = self.create_publisher(Empty, f"/{pid}/attach", 10)
            self._detach_pubs[pid] = self.create_publisher(Empty, f"/{pid}/detach", 10)

        # Action clients & direct gripper trajectory publisher for adaptive force/stall grasping
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._traj_client = ActionClient(self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory")
        self._gripper_client = ActionClient(self, FollowJointTrajectory, "/robotiq_gripper_controller/follow_joint_trajectory")
        self._gripper_traj_pub = self.create_publisher(JointTrajectory, "/robotiq_gripper_controller/joint_trajectory", 10)

        # Continuous 20Hz timer so all 6 CAD parts, 3D boxes, and TFs update smoothly in RViz2
        self._viz_timer = self.create_timer(0.05, self._publish_viz_and_tf)

    def _joint_cb(self, msg: JointState):
        name_map = dict(zip(msg.name, msg.position))
        if all(j in name_map for j in JOINT_NAMES):
            self._current_joints = [name_map[j] for j in JOINT_NAMES]
        if "robotiq_85_left_knuckle_joint" in name_map:
            self._gripper_joint_pos = float(name_map["robotiq_85_left_knuckle_joint"])

    def _rgb_cb(self, msg: Image):
        self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _depth_cb(self, msg: Image):
        self._latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _cam_info_cb(self, msg: CameraInfo):
        self._camera_info = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def start_foundationpose_server(self) -> bool:
        try:
            resp = fp_rpc({"cmd": "ping"}, timeout=1.0)
            if resp.get("status") == "ok":
                self.get_logger().info("FoundationPose server already running.")
                return True
        except Exception:
            pass

        server_script = os.path.join(self.pkg_share, "scripts", "foundationpose_server.py")
        manifest_path = os.path.join(self.pkg_share, "models", "parts_manifest.json")
        self.get_logger().info("Launching NVIDIA FoundationPose inference server on RTX GPU...")
        self._fp_log = open("/tmp/foundationpose_server.log", "w")
        self._fp_proc = subprocess.Popen(
            [FP_PYTHON, server_script, manifest_path],
            stdout=self._fp_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.time() + 45.0
        while time.time() < deadline:
            try:
                resp = fp_rpc({"cmd": "ping"}, timeout=1.0)
                if resp.get("status") == "ok":
                    self.get_logger().info("FoundationPose inference server is ready.")
                    return True
            except Exception:
                time.sleep(0.5)
        self.get_logger().error("Timed out waiting for FoundationPose server.")
        return False

    def stop_foundationpose_server(self):
        if self._fp_proc is not None:
            try:
                fp_rpc({"cmd": "shutdown"}, timeout=2.0)
            except Exception:
                pass
            self._fp_proc.terminate()
            self._fp_proc = None

    def wait_for_ready(self) -> bool:
        self.get_logger().info("Waiting for joint states, RGB-D camera, and controllers...")
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < 25.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self._current_joints is not None
                and self._latest_rgb is not None
                and self._latest_depth is not None
                and self._camera_info is not None
            ):
                break
        if self._current_joints is None or self._latest_rgb is None or self._latest_depth is None:
            self.get_logger().error("Timed out waiting for sensors/joint_states.")
            return False
        self._traj_client.wait_for_server(timeout_sec=15.0)
        self._gripper_client.wait_for_server(timeout_sec=15.0)
        return True

    def send_exact_trajectory(self, label: str, q_goal: list, duration_sec: float = 3.0) -> bool:
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q_goal]
        pt.velocities = [0.0] * 6
        sec = int(duration_sec)
        nsec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = Duration(seconds=sec, nanoseconds=nsec).to_msg()
        traj.points = [pt]
        goal.trajectory = traj

        self.get_logger().info(f"[{label}] Executing trajectory ({duration_sec:.1f}s)...")
        fut = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f"[{label}] Trajectory goal rejected.")
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=duration_sec + 5.0)
        return True

    def command_gripper(self, position: float, max_effort: float = 35.0, timeout_sec: float = 2.0):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ["robotiq_85_left_knuckle_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.velocities = [0.0]
        pt.time_from_start = Duration(seconds=1, nanoseconds=0).to_msg()
        traj.points = [pt]
        goal.trajectory = traj

        self.get_logger().info(f"[GRIPPER] Moving knuckle joint to {position:.3f} rad...")
        fut = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        gh = fut.result()
        if gh is not None and gh.accepted:
            res_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout_sec)
        else:
            self.get_logger().warning("[GRIPPER] Trajectory goal not accepted.")

    def _send_gripper_step(self, position: float, duration_sec: float = 0.08):
        traj = JointTrajectory()
        traj.joint_names = ["robotiq_85_left_knuckle_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.velocities = [0.0]
        sec = int(duration_sec)
        nsec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = Duration(seconds=sec, nanoseconds=nsec).to_msg()
        traj.points = [pt]
        self._gripper_traj_pub.publish(traj)

    def adaptive_close_gripper(self, max_angle: float = 0.76, step: float = 0.03) -> float:
        """
        Shape-agnostic adaptive grasp: closes the Robotiq 2F-85 incrementally while
        monitoring actual knuckle joint feedback. Stops as soon as physical contact
        stalls the fingers against the part (applicable to any known or unseen object).
        """
        self.get_logger().info("[GRIPPER] Starting adaptive contact-sensing close...")
        q_cmd = max(self._gripper_joint_pos, 0.0)
        prev_actual = self._gripper_joint_pos
        stall_count = 0

        while rclpy.ok() and q_cmd < max_angle:
            q_cmd = min(q_cmd + step, max_angle)
            self._send_gripper_step(q_cmd, duration_sec=0.08)

            t_end = time.time() + 0.09
            while rclpy.ok() and time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.02)

            actual = self._gripper_joint_pos
            lag = q_cmd - actual
            progress = actual - prev_actual
            prev_actual = actual

            # Contact detected once fingers have started closing and physics blocks further motion
            if q_cmd > 0.14:
                if lag > 0.032:
                    stall_count += 1
                elif progress < 0.004:
                    stall_count += 1
                else:
                    stall_count = 0

                if stall_count >= 2:
                    q_hold = min(actual + 0.015, max_angle)
                    self._send_gripper_step(q_hold, duration_sec=0.1)
                    self.get_logger().info(
                        f"[GRIPPER] Contact detected at {actual:.3f} rad (lag={lag:.3f}); "
                        f"holding firm at {q_hold:.3f} rad."
                    )
                    return q_hold

        self.get_logger().info(f"[GRIPPER] Reached max close angle {q_cmd:.3f} rad (actual={self._gripper_joint_pos:.3f}).")
        return self._gripper_joint_pos

    def _gz_attach_detach(self, part_id: str, action: str):
        pub = self._attach_pubs[part_id] if action == "attach" else self._detach_pubs[part_id]
        for _ in range(4):
            pub.publish(Empty())
            time.sleep(0.02)
        gz_bin = "/opt/ros/lyrical/opt/gz_tools_vendor/bin/gz"
        if os.path.exists(gz_bin):
            try:
                subprocess.run(
                    [gz_bin, "topic", "-t", f"/{part_id}/{action}", "-m", "gz.msgs.Empty", "-p", " "],
                    timeout=1.5,
                    capture_output=True,
                )
            except Exception:
                pass

    def _segment_parts_rgbd(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> dict:
        """Segment each of the 6 CAD parts using 3D tabletop filtering + HSV color + connected components."""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid_d = np.isfinite(depth) & (depth > 0.15) & (depth < 2.5)
        z_c = np.where(valid_d, depth, 0.0)
        x_c = (u - K[0, 2]) * z_c / K[0, 0]
        y_c = (v - K[1, 2]) * z_c / K[1, 1]

        pts_cam = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=-1).reshape(-1, 4)
        pts_world = (T_world_cam @ pts_cam.T).T.reshape(H, W, 4)

        # Keep pixels belonging to objects sitting on the table (Z in [0.803, 0.96])
        tabletop_obj_mask = (
            valid_d
            & (pts_world[..., 2] > 0.803)
            & (pts_world[..., 2] < 0.960)
            & (pts_world[..., 0] > 0.55)
            & (pts_world[..., 0] < 1.30)
            & (pts_world[..., 1] > -0.55)
            & (pts_world[..., 1] < 0.55)
        )

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        masks = {}
        kernel = np.ones((3, 3), np.uint8)

        for item in self.manifest:
            pid = item["id"]
            color_mask = np.zeros((H, W), dtype=np.uint8)
            for lo, hi in item["hsv_ranges"]:
                m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
                color_mask = cv2.bitwise_or(color_mask, m)

            combined = (color_mask > 0) & tabletop_obj_mask
            combined_u8 = combined.astype(np.uint8) * 255
            combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_OPEN, kernel)
            combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_CLOSE, kernel)

            # Keep largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_u8)
            if num_labels > 1:
                best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[best_lbl, cv2.CC_STAT_AREA] >= 30:
                    mask_bool = (labels == best_lbl)
                    masks[pid] = mask_bool
                    self._part_clouds_world[pid] = pts_world[mask_bool, :3].copy()
        return masks

    def estimate_all_parts_foundationpose(self) -> bool:
        """Capture fresh stationary RGB-D at observation pose and run FoundationPose on all 6 parts."""
        self._latest_rgb = None
        self._latest_depth = None
        t_wait = time.time()
        while rclpy.ok() and (time.time() - t_wait) < 5.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_rgb is not None and self._latest_depth is not None:
                break
        if self._latest_rgb is None or self._latest_depth is None:
            self.get_logger().error("Did not receive fresh RGB-D frame at observation pose.")
            return False

        rgb = self._latest_rgb.copy()
        depth = self._latest_depth.copy()
        K = self._camera_info.copy()
        q_obs = list(self._current_joints)
        T_world_cam = self.kin.fk_cam_opt(q_obs)

        masks = self._segment_parts_rgbd(rgb, depth, K, T_world_cam)
        self.get_logger().info(f"Segmented {len(masks)}/6 CAD parts in RGB-D view: {list(masks.keys())}")
        if not masks:
            return False

        self.get_logger().info("Running NVIDIA FoundationPose 6D pose estimation...")
        resp = fp_rpc(
            {"cmd": "estimate", "rgb": rgb, "depth": depth, "K": K, "masks": masks, "iteration": 5},
            timeout=120.0,
        )
        if resp.get("status") != "ok":
            self.get_logger().error(f"FoundationPose RPC failed: {resp}")
            return False

        raw_results = resp["results"]
        self._detected_poses = {}
        for pid, res in raw_results.items():
            T_cam_part = res["pose_centered_in_cam"]
            T_world_part = T_world_cam @ T_cam_part
            T_world_bottom = T_world_cam @ res["pose_bottom_in_cam"]
            pos = T_world_part[:3, 3]
            quat = R.from_matrix(T_world_part[:3, :3]).as_quat()
            self._detected_poses[pid] = {
                "T_world_part": T_world_part,
                "T_world_bottom": T_world_bottom,
                "world_pos": pos.tolist(),
                "world_quat": quat.tolist(),
            }
            self._part_bottom_poses[pid] = T_world_bottom
            self.get_logger().info(
                f"  [{pid}] FoundationPose 6D: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}"
            )

        self._publish_viz_and_tf()
        return len(self._detected_poses) > 0

    def _publish_viz_and_tf(self):
        """Publish always-visible 3D CAD meshes, 3D bounding boxes, and TF frames in RViz2."""
        now = self.get_clock().now().to_msg()

        T_tcp_live = None
        if self._attached_part_id is not None and self._current_joints is not None:
            T_tcp_live = self.kin.fk_tcp(self._current_joints)

        # 1. Always-visible 3D CAD mesh markers for ALL 6 parts (even outside FOV or during pick-up)
        mesh_markers = MarkerArray()
        for idx, item in enumerate(self.manifest):
            pid = item["id"]
            if pid == self._attached_part_id and T_tcp_live is not None and self._attached_T_tcp_bottom is not None:
                T_bot = T_tcp_live @ self._attached_T_tcp_bottom
            else:
                T_bot = self._part_bottom_poses[pid]

            q_bot = R.from_matrix(T_bot[:3, :3]).as_quat()
            r, g, b, _ = item["rgba"]
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "workcell_cad_parts"
            m.id = idx
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.mesh_resource = f"package://ur10e_perception_workcell/{item['mesh_stl']}"
            m.mesh_use_embedded_materials = False
            m.pose.position.x = float(T_bot[0, 3])
            m.pose.position.y = float(T_bot[1, 3])
            m.pose.position.z = float(T_bot[2, 3])
            m.pose.orientation.x = float(q_bot[0])
            m.pose.orientation.y = float(q_bot[1])
            m.pose.orientation.z = float(q_bot[2])
            m.pose.orientation.w = float(q_bot[3])
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            m.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            mesh_markers.markers.append(m)

        self._parts_visual_pub.publish(mesh_markers)

        if not self._detected_poses:
            return

        markers = MarkerArray()
        transforms = []

        for idx, (pid, info) in enumerate(self._detected_poses.items()):
            item = self.parts_by_id[pid]
            if pid == self._attached_part_id and T_tcp_live is not None and self._attached_T_tcp_center is not None:
                T = T_tcp_live @ self._attached_T_tcp_center
                pos = T[:3, 3]
                quat = R.from_matrix(T[:3, :3]).as_quat()
            else:
                T = info["T_world_part"]
                pos = info["world_pos"]
                quat = info["world_quat"]
            dx, dy, dz = item["extents"]
            r, g, b, _ = item["rgba"]

            # 1. TF Frame for each object
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = "world"
            tf_msg.child_frame_id = pid
            tf_msg.transform.translation.x = float(pos[0])
            tf_msg.transform.translation.y = float(pos[1])
            tf_msg.transform.translation.z = float(pos[2])
            tf_msg.transform.rotation.x = float(quat[0])
            tf_msg.transform.rotation.y = float(quat[1])
            tf_msg.transform.rotation.z = float(quat[2])
            tf_msg.transform.rotation.w = float(quat[3])
            transforms.append(tf_msg)

            # 2. 3D Wireframe Bounding Box (12 edges in part local frame, transformed to world)
            hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
            corners_local = np.array([
                [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
                [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
            ])
            corners_world = (T[:3, :3] @ corners_local.T).T + T[:3, 3]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            box_marker = Marker()
            box_marker.header.frame_id = "world"
            box_marker.header.stamp = now
            box_marker.ns = "foundationpose_3d_boxes"
            box_marker.id = idx * 2
            box_marker.type = Marker.LINE_LIST
            box_marker.action = Marker.ADD
            box_marker.scale.x = 0.004
            box_marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            for i0, i1 in edges:
                box_marker.points.append(Point(x=float(corners_world[i0, 0]), y=float(corners_world[i0, 1]), z=float(corners_world[i0, 2])))
                box_marker.points.append(Point(x=float(corners_world[i1, 0]), y=float(corners_world[i1, 1]), z=float(corners_world[i1, 2])))
            markers.markers.append(box_marker)

            # 3. Text label above each 3D box
            txt_marker = Marker()
            txt_marker.header.frame_id = "world"
            txt_marker.header.stamp = now
            txt_marker.ns = "foundationpose_labels"
            txt_marker.id = idx * 2 + 1
            txt_marker.type = Marker.TEXT_VIEW_FACING
            txt_marker.action = Marker.ADD
            txt_marker.pose.position.x = float(pos[0])
            txt_marker.pose.position.y = float(pos[1])
            txt_marker.pose.position.z = float(pos[2] + hz + 0.035)
            txt_marker.scale.z = 0.025
            txt_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt_marker.text = item["display_name"]
            markers.markers.append(txt_marker)

        self._tf_broadcaster.sendTransform(transforms)
        self._marker_pub.publish(markers)

    def pick_lift_hold_place_back(self, part_id: str) -> bool:
        """Pick selected part using 3D cross-section grasp planning, lift to Z=0.980m, hold 2.5s, place back."""
        pose_info = self._detected_poses[part_id]
        item = self.parts_by_id[part_id]

        # Build 3D surface points in world frame:
        # Use FoundationPose-transformed CAD surface if available (full 360 deg, zero occlusion),
        # otherwise fall back directly to the live RGB-D point cloud (works for any unseen object).
        pts_world = None
        stl_path = os.path.join(self.pkg_share, item["mesh_stl"])
        if os.path.exists(stl_path) and "T_world_bottom" in pose_info:
            try:
                local_pts = load_stl_surface_points(stl_path)
                T_bot = pose_info["T_world_bottom"]
                pts_world = (T_bot[:3, :3] @ local_pts.T).T + T_bot[:3, 3]
            except Exception as e:
                self.get_logger().warning(f"STL surface load failed ({e}), falling back to live RGB-D cloud.")

        if pts_world is None:
            pts_world = self._part_clouds_world.get(part_id)

        grasp = plan_grasp_from_points(pts_world)
        if grasp is not None:
            px = grasp["grasp_x"]
            py = grasp["grasp_y"]
            grasp_z = grasp["grasp_z"]
            yaw = grasp["yaw"]
            q_close = grasp["q_close"]
            grasp_w_mm = grasp["width"] * 1000.0
        else:
            px, py, pz = pose_info["world_pos"]
            grasp_z = float(np.clip(pz, 0.815, 0.840))
            yaw = 0.0
            grasp_w_mm = min(float(item["extents"][0]), float(item["extents"][1])) * 1000.0
            q_close = robotiq_gap_to_knuckle(grasp_w_mm / 1000.0)

        lift_z = 0.980  # Same lift height as red cube

        self.get_logger().info(
            f"Grasp plan for '{item['display_name']}': pos=({px:.3f}, {py:.3f}, {grasp_z:.3f}), "
            f"yaw={math.degrees(yaw):+.1f} deg, slice_width={grasp_w_mm:.1f} mm -> q_close={q_close:.3f} rad"
        )

        q_curr = list(self._current_joints)
        q_pre_grasp = self.kin.solve_ik_target([px, py, lift_z], q_seed=q_curr, yaw=yaw)
        q_grasp = self.kin.solve_ik_target([px, py, grasp_z], q_seed=q_pre_grasp, yaw=yaw)
        q_lift = self.kin.solve_ik_target([px, py, lift_z], q_seed=q_grasp, yaw=yaw)

        # Open gripper
        self.command_gripper(0.0, max_effort=25.0, timeout_sec=2.5)

        # 1. Pre-grasp
        if not self.send_exact_trajectory("PRE_GRASP", q_pre_grasp, duration_sec=3.0):
            return False
        time.sleep(0.2)

        # 2. Descend to grasp
        if not self.send_exact_trajectory("GRASP", q_grasp, duration_sec=1.5):
            return False
        time.sleep(0.2)

        # 3. Close gripper to exact flush surface contact (zero penetration, zero air gap)
        self.command_gripper(q_close, max_effort=35.0, timeout_sec=2.5)
        t_wait = time.time()
        while rclpy.ok() and (time.time() - t_wait) < 1.5:
            rclpy.spin_once(self, timeout_sec=0.05)
            if abs(self._gripper_joint_pos - q_close) < 0.025:
                break
        time.sleep(0.15)
        self._gz_attach_detach(part_id, "attach")
        T_tcp_grasp = self.kin.fk_tcp(self._current_joints)
        self._attached_T_tcp_bottom = np.linalg.inv(T_tcp_grasp) @ self._part_bottom_poses[part_id]
        self._attached_T_tcp_center = np.linalg.inv(T_tcp_grasp) @ pose_info["T_world_part"]
        self._attached_part_id = part_id
        time.sleep(0.15)

        # 4. Lift to Z = 0.980m
        if not self.send_exact_trajectory("LIFT", q_lift, duration_sec=1.5):
            return False

        # 5. Hold in air for 2.5 seconds
        self.get_logger().info("[HOLD] Holding part aloft for 2.5 seconds...")
        t_hold_start = time.time()
        while rclpy.ok() and (time.time() - t_hold_start) < 2.5:
            rclpy.spin_once(self, timeout_sec=0.05)

        # 6. Lower back to exact original pose on the table
        self.get_logger().info("[PLACE BACK] Returning part to original position...")
        if not self.send_exact_trajectory("PLACE_BACK", q_grasp, duration_sec=1.5):
            return False
        time.sleep(0.2)

        # 7. Open gripper & detach
        self._attached_part_id = None
        self.command_gripper(0.0, max_effort=25.0, timeout_sec=2.0)
        time.sleep(0.15)
        self._gz_attach_detach(part_id, "detach")
        time.sleep(0.2)

        # 8. Retract vertically
        if not self.send_exact_trajectory("RETRACT", q_pre_grasp, duration_sec=1.5):
            return False

        # 9. Return to observation pose
        self.send_exact_trajectory("OBSERVE_MULTI", POSES["OBSERVE_MULTI"], duration_sec=3.0)
        return True

    def run_sequence(self):
        if not self.wait_for_ready():
            return

        # Guarantee all 6 DetachableJoints are detached before any arm motion
        for item in self.manifest:
            self._gz_attach_detach(item["id"], "detach")

        if not self.observe_only:
            if not self.start_foundationpose_server():
                return

        try:
            # Open gripper & move to elevated angled observation pose
            self.command_gripper(0.0, max_effort=25.0, timeout_sec=2.0)
            self.send_exact_trajectory("OBSERVE_MULTI", POSES["OBSERVE_MULTI"], duration_sec=3.5)
            time.sleep(0.5)

            if self.observe_only:
                self.get_logger().info("Observe-only mode active. Holding observation pose.")
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.2)
                return

            # Run FoundationPose 6D estimation on all 6 CAD parts
            if not self.estimate_all_parts_foundationpose():
                self.get_logger().error("FoundationPose estimation failed.")
                return

            # Interactive loop: keep asking user which part to pick up until Finish
            while rclpy.ok():
                # Keep ROS callbacks spinning while UI is open
                stop_spin = threading.Event()

                def bg_spin():
                    while not stop_spin.is_set() and rclpy.ok():
                        rclpy.spin_once(self, timeout_sec=0.1)

                spin_thread = threading.Thread(target=bg_spin, daemon=True)
                spin_thread.start()

                ui = PartSelectionUI(self.manifest, self._detected_poses, self.pkg_share)
                chosen_id = ui.show()

                stop_spin.set()
                spin_thread.join(timeout=1.0)

                if chosen_id is None:
                    self.get_logger().info("User finished session. Returning HOME.")
                    self.send_exact_trajectory("HOME", POSES["HOME"], duration_sec=3.0)
                    break

                self.pick_lift_hold_place_back(chosen_id)
                # Refresh poses after placing back
                self.estimate_all_parts_foundationpose()
        finally:
            self.stop_foundationpose_server()


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPosePickPlaceNode()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.stop_foundationpose_server()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
