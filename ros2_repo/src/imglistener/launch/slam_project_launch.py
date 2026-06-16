import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():


    slam_node = Node(
        package='imglistener', 
        executable='slam_node',
        name='imglistener',
        output='screen'
    )

    #csv_node = Node(
    #    package='imglistener', 
    #    executable='odometry_exporter',
    #    name='odometry_exporter',
    #    output='screen'
    #)

    bag_file = '/media/sf_Projekt/rosplay/Around_The_Desks_bag/Around_The_Desks_bag'

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_file,'--clock','-r', '0.5', '--topics',
             '/serf01/nav_rgbd_1/rgb/image_raw',
             '/serf01/nav_rgbd_1/depth/image_raw',
             '/tf_static','/serf01/odometry/wheel',
             '/serf01/odometry/filtered',
             '/serf01/odometry/imu'],
        output='screen'
    )

    return LaunchDescription([
        slam_node,
        #csv_node,
        rosbag_play
    ])