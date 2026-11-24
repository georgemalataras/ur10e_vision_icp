#!/usr/bin/env python3
"""
MoveIt 2 Planning Scene Initializer for UR10e Perception Workcell.

Publishes the workcell collision environment into the MoveIt 2 PlanningScene:
  1. 'pedestal' (Robot base pedestal: 0.40 x 0.40 x 0.748 m)
  2. 'table'    (4-legged industrial table at X=0.95, Y=0.0, top surface Z=0.80 m)
  3. 'red_cube' (Target cube: 0.05 x 0.05 x 0.05 m on the table surface at Z=0.825 m)
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene


class WorkcellPlanningScenePublisher(Node):
    def __init__(self):
        super().__init__("workcell_planning_scene_publisher")

        self.declare_parameter("cube_x", 0.850)
        self.declare_parameter("cube_y", 0.000)
        self.declare_parameter("cube_z", 0.825)

        self.cube_x = float(self.get_parameter("cube_x").value)
        self.cube_y = float(self.get_parameter("cube_y").value)
        self.cube_z = float(self.get_parameter("cube_z").value)

        self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self._apply_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

    @staticmethod
    def _make_box_pose(x: float, y: float, z: float) -> Pose:
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        p.orientation.w = 1.0
        return p

    @staticmethod
    def _make_box_primitive(sx: float, sy: float, sz: float) -> SolidPrimitive:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(sx), float(sy), float(sz)]
        return box

    def build_planning_scene_msg(self) -> PlanningScene:
        now = self.get_clock().now().to_msg()

        # 1. Robot Base Pedestal CollisionObject (0.40 x 0.40 x 0.746 m, 4mm clearance below base_link at 0.75m)
        pedestal_obj = CollisionObject()
        pedestal_obj.header.frame_id = "world"
        pedestal_obj.header.stamp = now
        pedestal_obj.id = "pedestal"
        pedestal_obj.operation = CollisionObject.ADD
        pedestal_obj.primitives.append(self._make_box_primitive(0.40, 0.40, 0.746))
        pedestal_obj.primitive_poses.append(self._make_box_pose(0.0, 0.0, 0.373))

        # 2. Industrial Table CollisionObject (Tabletop + 4 Legs matching table_scene.sdf)
        table_obj = CollisionObject()
        table_obj.header.frame_id = "world"
        table_obj.header.stamp = now
        table_obj.id = "table"
        table_obj.operation = CollisionObject.ADD

        table_x = 0.95
        table_y = 0.0
        # Tabletop: 0.80 x 1.20 x 0.05 m, top surface at Z = 0.80 m (center Z = 0.775 m)
        table_obj.primitives.append(self._make_box_primitive(0.80, 1.20, 0.05))
        table_obj.primitive_poses.append(self._make_box_pose(table_x, table_y, 0.775))

        # 4 Table Legs: 0.05 x 0.05 x 0.75 m (center Z = 0.375 m)
        for dx in (0.35, -0.35):
            for dy in (0.55, -0.55):
                table_obj.primitives.append(self._make_box_primitive(0.05, 0.05, 0.75))
                table_obj.primitive_poses.append(self._make_box_pose(table_x + dx, table_y + dy, 0.375))

        # 3. Red Cube CollisionObject (0.05 x 0.05 x 0.05 m on tabletop at Z = 0.825 m)
        cube_obj = CollisionObject()
        cube_obj.header.frame_id = "world"
        cube_obj.header.stamp = now
        cube_obj.id = "red_cube"
        cube_obj.operation = CollisionObject.ADD
        cube_obj.primitives.append(self._make_box_primitive(0.05, 0.05, 0.05))
        cube_obj.primitive_poses.append(self._make_box_pose(self.cube_x, self.cube_y, self.cube_z))

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [pedestal_obj, table_obj, cube_obj]
        return scene

    def apply_scene(self):
        self.get_logger().info("Waiting for MoveIt /apply_planning_scene service...")
        service_ready = self._apply_client.wait_for_service(timeout_sec=25.0)

        scene_msg = self.build_planning_scene_msg()

        # Always publish on /planning_scene topic for any PlanningScene monitors / RViz2
        for _ in range(3):
            self._scene_pub.publish(scene_msg)
            time.sleep(0.15)

        if service_ready:
            req = ApplyPlanningScene.Request()
            req.scene = scene_msg
            future = self._apply_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None and future.result().success:
                self.get_logger().info(
                    f"Applied PlanningScene objects: ['pedestal', 'table', 'red_cube' at ({self.cube_x:.3f}, {self.cube_y:.3f}, {self.cube_z:.3f})]"
                )
            else:
                self.get_logger().warning("ApplyPlanningScene service call did not confirm success, relied on /planning_scene topic.")
        else:
            self.get_logger().warning("/apply_planning_scene service timeout; published objects directly to /planning_scene topic.")


def main(args=None):
    rclpy.init(args=args)
    node = WorkcellPlanningScenePublisher()
    try:
        node.apply_scene()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
