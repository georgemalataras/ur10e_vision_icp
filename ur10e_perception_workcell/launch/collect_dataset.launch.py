import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    RegisterEventHandler,
    EmitEvent,
)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_ur10e_workcell = get_package_share_directory("ur10e_perception_workcell")

    # Launch Arguments
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    script_delay = LaunchConfiguration("script_delay")

    declare_gazebo_gui = DeclareLaunchArgument(
        "gazebo_gui",
        default_value="true",
        description="Launch Gazebo with GUI",
    )

    declare_script_delay = DeclareLaunchArgument(
        "script_delay",
        default_value="15.0",
        description="Delay in seconds before starting dataset collection script to let Gazebo and MoveIt initialize",
    )

    # 1. Gazebo Simulation with UR10e, table, cube and controllers
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ur10e_workcell, "launch", "sim.launch.py")
        ),
        launch_arguments={
            "gazebo_gui": gazebo_gui,
        }.items(),
    )

    # 2. MoveIt move_group node
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ur10e_workcell, "launch", "move_group.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
        }.items(),
    )

    # 3. Dataset Collection Node
    collect_dataset_node = Node(
        package="ur10e_perception_workcell",
        executable="collect_dataset.py",
        name="ur10e_dataset_collector_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    delayed_collection_execution = TimerAction(
        period=script_delay,
        actions=[collect_dataset_node],
    )

    shutdown_on_completion = RegisterEventHandler(
        OnProcessExit(
            target_action=collect_dataset_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription(
        [
            declare_gazebo_gui,
            declare_script_delay,
            sim_launch,
            move_group_launch,
            delayed_collection_execution,
            shutdown_on_completion,
        ]
    )
