import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    imglistener_share = get_package_share_directory('imglistener')
    rviz_config = os.path.join(imglistener_share, '../../../../src/imglistener/rviz', 'rviz2_config.rviz')
    #rosbag_path = os.path.join(imglistener_share, '../../../../src/imglistener/bagfiles', '20260616_Around_The_Desks_bag')
    #rosbag_path = os.path.join(imglistener_share, '../../../../src/imglistener/bagfiles', '20260519_TransXY_bag')
    rosbag_path = os.path.join(imglistener_share, '../../../../src/imglistener/bagfiles', '20260428_Bag_Without_TF')
    

    slam_node = Node(
        package='imglistener', 
        executable='slam_node',
        name='slam_node',
        output='screen'
    )

    """
    csv_node = Node(
        package='imglistener', 
        executable='odometry_exporter',
        name='odometry_exporter',
        output='screen'
    )
    """
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path,'-r', '0.4','--clock','--topics','/serf01/nav_rgbd_1/rgb/image_raw',
        '/serf01/nav_rgbd_1/depth/image_raw','/tf_static','/serf01/odometry/wheel','/serf01/odometry/filtered','/serf01/odometry/imu'],
        output='screen'
    )
    rviz_2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )


    return LaunchDescription([
        slam_node,
        #csv_node,
        rosbag_play,
        rviz_2
    ])