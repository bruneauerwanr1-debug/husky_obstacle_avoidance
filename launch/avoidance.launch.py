"""
avoidance.launch.py — ROS 2 Launch File for Husky Obstacle Avoidance Nodes
==========================================================================
Usage:
  # Launch Farneback Optical Flow (CPU)
  ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=farneback

  # Launch RAFT Deep Optical Flow (GPU)
  ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=raft

  # Launch YOLOv8 Predictive Avoidance (GPU)
  ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=yolo

  # Launch Homography Calibration Tool
  ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=homography
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='farneback',
        description='Avoidance method to run: [farneback, raft, yolo, homography]'
    )

    method = LaunchConfiguration('method')

    # 1. Farneback Optical Flow Node
    farneback_node = Node(
        package='husky_obstacle_avoidance',
        executable='optical_flow_farneback.py',
        name='farneback_avoidance_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", method, "' == 'farneback'"]))
    )

    # 2. RAFT Deep Optical Flow Node
    raft_node = Node(
        package='husky_obstacle_avoidance',
        executable='raft_avoidance_node.py',
        name='raft_avoidance_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", method, "' == 'raft'"]))
    )

    # 3. YOLOv8 Predictive Avoidance Node
    yolo_node = Node(
        package='husky_obstacle_avoidance',
        executable='predictive_yolo.py',
        name='predictive_yolo_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", method, "' == 'yolo'"]))
    )

    # 4. Homography Calibration Node
    homography_node = Node(
        package='husky_obstacle_avoidance',
        executable='homography_node.py',
        name='homography_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", method, "' == 'homography'"]))
    )

    return LaunchDescription([
        method_arg,
        farneback_node,
        raft_node,
        yolo_node,
        homography_node,
    ])
