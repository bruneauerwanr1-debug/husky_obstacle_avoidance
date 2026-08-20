#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base.launch-collision.py — Clearpath Husky Base Bringup with Collision Multiplexing
==================================================================================
Launches the Clearpath Husky robot base with twist_mux configured for reactive
vision-based obstacle avoidance and emergency stop locks.

Multiplexer Configuration:
  - Priority 255 (Lock):     /emergency_stop (Vision E-Stop)
  - Priority 50  (Override): /avoidance_cmd_vel (Avoidance Nodes)
  - Priority 10  (Default):  /joy_teleop/cmd_vel (Manual Joystick Teleop)
  - Priority 8   (Marker):   /twist_marker_server/cmd_vel (Interactive Marker)

Robot Execution Command:
  cd SSM-Nav2/ssm-nav2
  source install/setup.bash
  ros2 launch husky_base base.launch-collision.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    husky_base_dir = get_package_share_directory('husky_base')
    husky_control_dir = get_package_share_directory('husky_control')

    # Path to twist_mux configuration with collision priorities
    config_twist_mux = LaunchConfiguration(
        'config_twist_mux',
        default=os.path.join(husky_base_dir, 'config', 'twist_mux.yaml')
    )

    declare_twist_mux_arg = DeclareLaunchArgument(
        'config_twist_mux',
        default_value=config_twist_mux,
        description='Path to twist_mux configuration file for collision avoidance'
    )

    # 1. Base control launch (husky_control / hardware drivers)
    husky_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(husky_control_dir, 'launch', 'control.launch.py')
        )
    )

    # 2. Twist Mux node with collision avoidance topic and e-stop lock
    twist_mux_node = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        remappings=[('cmd_vel_out', 'cmd_vel_in')],
        parameters=[config_twist_mux]
    )

    # 3. RealSense RGB-D Camera Node (with depth-to-color alignment enabled)
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={'align_depth.enable': 'true'}.items()
    )

    return LaunchDescription([
        declare_twist_mux_arg,
        husky_control_launch,
        twist_mux_node,
        realsense_node,
    ])
