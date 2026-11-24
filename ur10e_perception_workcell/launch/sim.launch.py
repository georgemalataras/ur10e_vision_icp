import os
import random
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, IfElseSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_ur10e_workcell = get_package_share_directory("ur10e_perception_workcell")
    pkg_ur_simulation_gz = get_package_share_directory("ur_simulation_gz")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    # Launch configuration variables
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")
    controllers_file = LaunchConfiguration("controllers_file")

    declare_gazebo_gui = DeclareLaunchArgument(
        "gazebo_gui",
        default_value="true",
        description="Launch Gazebo with GUI",
    )

    declare_world_file = DeclareLaunchArgument(
        "world_file",
        default_value=os.path.join(pkg_ur10e_workcell, "worlds", "table_scene.sdf"),
        description="Path to the Gazebo world SDF file",
    )

    declare_controllers_file = DeclareLaunchArgument(
        "controllers_file",
        default_value=os.path.join(pkg_ur10e_workcell, "config", "workcell_controllers.yaml"),
        description="Path to controllers YAML configuration",
    )

    gui_config_file = LaunchConfiguration("gui_config_file")
    rviz = LaunchConfiguration("rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")

    declare_gui_config_file = DeclareLaunchArgument(
        "gui_config_file",
        default_value=os.path.join(pkg_ur10e_workcell, "config", "gui.config"),
        description="Path to the Gazebo GUI configuration file",
    )

    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Launch RViz2 for 3D PointCloud2 and camera visualization",
    )

    declare_rviz_config_file = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=os.path.join(pkg_ur10e_workcell, "config", "workcell.rviz"),
        description="Path to the RViz2 configuration file",
    )

    # Random Tabletop Pose for Red Cube (Z=0.825m surface, fixed orientation Yaw=0.0)
    rand_x = round(random.uniform(0.78, 0.92), 3)
    rand_y = round(random.uniform(-0.16, 0.16), 3)

    cube_x = LaunchConfiguration("cube_x")
    cube_y = LaunchConfiguration("cube_y")
    declare_cube_x = DeclareLaunchArgument(
        "cube_x",
        default_value=str(rand_x),
        description="Random spawn X coordinate of red cube on table",
    )
    declare_cube_y = DeclareLaunchArgument(
        "cube_y",
        default_value=str(rand_y),
        description="Random spawn Y coordinate of red cube on table",
    )



    # Robot Description (URDF / Xacro)
    # When simulating in Gazebo, table and cube are loaded by the world file,
    # so we disable their duplicates in the URDF (include_table:=false include_cube:=false).
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            os.path.join(pkg_ur10e_workcell, "urdf", "ur10e_workcell.urdf.xacro"),
            " ",
            "sim_gazebo:=true",
            " ",
            "simulation_controllers:=",
            controllers_file,
            " ",
            "include_table:=false",
            " ",
            "include_cube:=false",
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    # Gazebo Sim Launch
    gz_launch_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": IfElseSubstitution(
                gazebo_gui,
                if_value=[
                    " -r -v 4 --gui-config ",
                    gui_config_file,
                    " --physics-engine gz-physics-dartsim-plugin ",
                    world_file,
                ],
                else_value=[
                    " -s -r --headless-rendering -v 4 --physics-engine gz-physics-dartsim-plugin ",
                    world_file,
                ],
            )
        }.items(),
    )

    # Spawn Robot Entity in Gazebo
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string",
            robot_description_content,
            "-name",
            "ur10e_workcell",
            "-allow_renaming",
            "true",
        ],
    )

    # Spawn Red Cube Entity at Random Table Pose (Fixed orientation Yaw=0.0)
    cube_model_path = os.path.join(pkg_ur10e_workcell, "models", "red_cube", "model.sdf")
    gz_spawn_cube = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file",
            cube_model_path,
            "-name",
            "red_cube",
            "-x",
            cube_x,
            "-y",
            cube_y,
            "-z",
            "0.825",
            "-Y",
            "0.0",
        ],
    )

    # ROS-GZ Parameter Bridge (/clock, /cube/attach, /cube/detach, RGB-D camera topics)
    gz_sim_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cube/attach@std_msgs/msg/Empty]gz.msgs.Empty",
            "/cube/detach@std_msgs/msg/Empty]gz.msgs.Empty",
            "/wrist_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/wrist_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/wrist_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/wrist_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        output="screen",
    )

    # Static TF alias from wrist_camera_link to Gazebo's scoped sensor frame
    # (ur10e_workcell/tool0/wrist_camera) so PointCloud2 renders in RViz2 seamlessly
    camera_gz_frame_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="wrist_camera_gz_frame_broadcaster",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "wrist_camera_link",
            "--child-frame-id", "ur10e_workcell/tool0/wrist_camera",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # Optional RViz2 Node for visualizing RobotModel, RGB, Depth, and 3D PointCloud2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
    )

    # Controller Spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )

    robotiq_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robotiq_gripper_controller", "--controller-manager", "/controller_manager"],
    )

    delay_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gz_spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    delay_joint_trajectory_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_trajectory_controller_spawner],
        )
    )

    delay_robotiq_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robotiq_gripper_controller_spawner],
        )
    )

    # Ensure DetachableJoint is detached so cube rests freely on table under gravity
    gz_initial_detach = ExecuteProcess(
        cmd=[
            FindExecutable(name="gz"),
            "topic",
            "-t",
            "/cube/detach",
            "-m",
            "gz.msgs.Empty",
            "-p",
            " ",
        ],
        output="screen",
    )

    delayed_gz_initial_detach = TimerAction(
        period=4.0,
        actions=[gz_initial_detach],
    )

    return LaunchDescription(
        [
            declare_gazebo_gui,
            declare_world_file,
            declare_controllers_file,
            declare_gui_config_file,
            declare_rviz,
            declare_rviz_config_file,
            declare_cube_x,
            declare_cube_y,
            gz_launch_description,
            robot_state_publisher_node,
            camera_gz_frame_tf,
            gz_spawn_entity,
            gz_spawn_cube,
            gz_sim_bridge,
            rviz_node,
            delayed_gz_initial_detach,
            delay_joint_state_broadcaster,
            delay_joint_trajectory_controller,
            delay_robotiq_gripper_controller,
        ]
    )
