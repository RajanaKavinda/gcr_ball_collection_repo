import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import  PathJoinSubstitution, PythonExpression, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import xacro
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('gcr_diff_robot'))
    xacro_file = os.path.join(pkg_path,'model','robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    gcr_description_dir = get_package_share_directory("gcr_diff_robot")

    params = {'robot_description': robot_description_config.toxml(),'use_sim_time':True}

    world_name_arg = DeclareLaunchArgument(name="world_name", default_value="empty")

    world_path = PathJoinSubstitution([
            gcr_description_dir,
            "worlds",
            PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
        ]
    )   

    gazebo_rosPackageLaunch=PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'),'launch','gazebo.launch.py'))

    gazeboLaunch=IncludeLaunchDescription(
        gazebo_rosPackageLaunch,
        launch_arguments={'world':world_path}.items()
    )


    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    spawnModelNode = Node(
            package='gazebo_ros', 
            executable='spawn_entity.py', 
            arguments=['-topic', 'robot_description', '-entity', 'differential_drive_robot'], 
            output='screen'
        )
    

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(gcr_description_dir, "rviz", "display.rviz")],
    )

    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_camera_joint_trajectory = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'camera_joint_controller'],
        output='screen'
    )

    load_collector_joint_trajectory = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'ball_collector_joint_controller'],
        output='screen'
    )

    return LaunchDescription([
        world_name_arg,
        gazeboLaunch,
        nodeRobotStatePublisher,
        spawnModelNode,
        rviz_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawnModelNode,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_camera_joint_trajectory, load_collector_joint_trajectory],
            )
        )
    ])




