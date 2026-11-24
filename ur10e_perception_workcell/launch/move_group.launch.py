import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except OSError:
        return {}


def load_file(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except OSError:
        return ""


def generate_launch_description():
    pkg_ur10e_workcell = get_package_share_directory("ur10e_perception_workcell")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )

    # Robot Description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            os.path.join(pkg_ur10e_workcell, "urdf", "ur10e_workcell.urdf.xacro"),
            " ",
            "sim_gazebo:=false",
            " ",
            "include_table:=false",
            " ",
            "include_cube:=false",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Semantic Description (SRDF)
    srdf_path = os.path.join(pkg_ur10e_workcell, "srdf", "ur10e_workcell.srdf")
    robot_description_semantic = {
        "robot_description_semantic": load_file(srdf_path)
    }

    # Kinematics
    kinematics_yaml = load_yaml(os.path.join(pkg_ur10e_workcell, "config", "kinematics.yaml"))
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # Joint limits (must be wrapped under robot_description_planning for MoveIt)
    joint_limits_yaml = load_yaml(os.path.join(pkg_ur10e_workcell, "config", "joint_limits.yaml"))
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

    # Planning Pipeline (OMPL)
    ompl_planning_yaml = load_yaml(os.path.join(pkg_ur10e_workcell, "config", "ompl_planning.yaml"))
    ompl_planning_pipeline_config = {
        "planning_pipelines": ["ompl"],
        "ompl": ompl_planning_yaml,
        "default_planning_pipeline": "ompl",
    }

    # MoveIt Controllers
    controllers_yaml = load_yaml(os.path.join(pkg_ur10e_workcell, "config", "moveit_controllers.yaml"))

    # Planning Scene Monitor parameters
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # MoveGroup Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            ompl_planning_pipeline_config,
            robot_description_planning,
            controllers_yaml,
            planning_scene_monitor_parameters,
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
        ],
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            move_group_node,
        ]
    )
