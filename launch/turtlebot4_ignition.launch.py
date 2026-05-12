# Copyright 2023 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
import os

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='raph',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='sim_world',
                          description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
]

pose_defaults = {
    'x': '-1.5',
    'y': '-3.5',
    'z': '0.25',
    'yaw': '0.0',
}

# pose_defaults = {
#     'x': '0.0',
#     'y': '0.0',
#     'z': '0.0',
#     'yaw': '0.0',
# }

for key, default in pose_defaults.items():
    ARGUMENTS.append(
        DeclareLaunchArgument(
            key,
            default_value=default,
            description=f'{key} component of the robot pose.'
        )
    )


def generate_launch_description():
    
    # Directories
    pkg_talking_turtle = get_package_share_directory('talking-turtle')

    ign_gazebo_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            PathJoinSubstitution([pkg_talking_turtle, 'world']),
            ':',
            os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
        ]
    )
    # Paths
    ignition_launch = PathJoinSubstitution(
        [pkg_talking_turtle, 'launch', 'ignition.launch.py'])
    robot_spawn_launch = PathJoinSubstitution(
        [pkg_talking_turtle, 'launch', 'turtlebot4_spawn.launch.py'])

    ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    
    # Robot Spawn launch
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw'))]
    )

    
    # Gazebo-ROS Bridge node
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/ids_overhead/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/ids_overhead/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
        ],
        output='screen'
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(ign_gazebo_resource_path)
    ld.add_action(ignition)
    ld.add_action(robot_spawn)
    ld.add_action(gz_bridge_node)
    return ld
