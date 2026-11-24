import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_ur10e_workcell = get_package_share_directory("ur10e_perception_workcell")

    # Launch Arguments
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    rviz = LaunchConfiguration("rviz")
    run_script = LaunchConfiguration("run_script")
    script_delay = LaunchConfiguration("script_delay")

    declare_gazebo_gui = DeclareLaunchArgument(
        "gazebo_gui",
        default_value="true",
        description="Launch Gazebo with GUI",
    )

    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 for 3D PointCloud2 and camera visualization",
    )

    declare_run_script = DeclareLaunchArgument(
        "run_script",
        default_value="true",
        description="Run pick and place motion planning script automatically",
    )

    declare_script_delay = DeclareLaunchArgument(
        "script_delay",
        default_value="12.0",
        description="Delay in seconds before starting motion sequence to let Gazebo and MoveIt initialize",
    )

    cube_x = LaunchConfiguration("cube_x")
    cube_y = LaunchConfiguration("cube_y")

    declare_cube_x = DeclareLaunchArgument(
        "cube_x",
        default_value="0.850",
        description="Known spawn X coordinate of red cube on table",
    )

    declare_cube_y = DeclareLaunchArgument(
        "cube_y",
        default_value="0.000",
        description="Known spawn Y coordinate of red cube on table",
    )

    # 1. Gazebo Simulation with UR10e, table, cube and controllers
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ur10e_workcell, "launch", "sim.launch.py")
        ),
        launch_arguments={
            "gazebo_gui": gazebo_gui,
            "rviz": rviz,
            "cube_x": cube_x,
            "cube_y": cube_y,
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

    # 3. Vision-Guided Motion Planning Sequence Node
    perception_pick_and_place_node = Node(
        package="ur10e_perception_workcell",
        executable="perception_pick_and_place.py",
        name="ur10e_perception_pick_place_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    delayed_script_execution = TimerAction(
        period=script_delay,
        actions=[perception_pick_and_place_node],
        condition=IfCondition(run_script),
    )

    return LaunchDescription(
        [
            declare_gazebo_gui,
            declare_rviz,
            declare_run_script,
            declare_script_delay,
            declare_cube_x,
            declare_cube_y,
            sim_launch,
            move_group_launch,
            delayed_script_execution,
        ]
    )
