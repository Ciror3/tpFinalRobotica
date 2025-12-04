import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'tpFinalA'
    sim_time_config = {'use_sim_time': True}

    odometry_node = Node(
        package=package_name,
        executable='my_odometry', # Nombre definido en entry_points de setup.py
        name='odom_node',
        output='screen',
        parameters=[sim_time_config]
    )

    # Nodo de FastSLAM
    slam_node = Node(
        package=package_name,
        executable='my_fastslam',     # Nombre definido en entry_points de setup.py
        name='fastslam_node',
        output='screen',
        parameters=[
            sim_time_config,
            {'num_particles': 10}
        ]
    )

    rviz_config_file = os.path.join(
        get_package_share_directory(package_name),
        'rviz', 
        'fastslam_config.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[sim_time_config],
        output='screen'
    )

    return LaunchDescription([
        odometry_node,
        slam_node,
        rviz_node
    ])