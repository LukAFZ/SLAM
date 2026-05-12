import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():


    slam_node = Node(
        package='imglistener', 
        executable='imglistener',
        name='imglistener',
        output='screen'
    )

    csv_node = Node(
        package='imglistener', 
        executable='odometry_exporter',
        name='odometry_exporter',
        output='screen'
    )

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/media/sf_SLAM/20260428_Bag_Without_TF/rosbag2_2026_04_27-19_59_23_0.mcap','--clock'],
        output='screen'
    )

    return LaunchDescription([
        slam_node,
        csv_node,
        rosbag_play
    ])