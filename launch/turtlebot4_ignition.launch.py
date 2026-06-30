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
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess,
                            TimerAction)
from launch.event_handlers import OnShutdown
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
    'x': '-1.75',
    'y': '-3.25', # adjusted to match old warehouse layout
    # 'x': '-1.5',
    # 'y': '-0.5',   # adjusted to match new house layout
    'z': '0.25',
    'yaw': '0.0',
}

for key, default in pose_defaults.items():
    ARGUMENTS.append(
        DeclareLaunchArgument(
            key,
            default_value=default,
            description=f'{key} component of the robot pose.'
        )
    )

# Second robot (donnie) — blue-hatted, spawned alongside raph. Pose defaults to open
# floor; adjust x2/y2 if it lands on an obstacle.
ARGUMENTS += [
    DeclareLaunchArgument('namespace2', default_value='donnie',
                          description='Second robot namespace'),
    DeclareLaunchArgument('x2', default_value='0.93', description='x of robot 2 (~1 m +x of the table)'),
    DeclareLaunchArgument('y2', default_value='2.96', description='y of robot 2 (table row)'),
    DeclareLaunchArgument('z2', default_value='0.25', description='z of robot 2'),
    DeclareLaunchArgument('yaw2', default_value='0.0', description='yaw of robot 2'),
    DeclareLaunchArgument('hat_color', default_value='0 0 1 1',
                          description='RGBA hat color of robot 2'),
]


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
    # Filtered spawn: strips the world-singleton Sensors/Contact plugins from each robot's
    # description (they now live in sim_world.sdf) so a 2nd robot can spawn without the
    # render-scene rebuild crash (turtlebot4_simulator#60). Used for BOTH robots.
    robot_spawn_launch = PathJoinSubstitution(
        [pkg_talking_turtle, 'launch', 'turtlebot4_spawn_filtered.launch.py'])
    robot_spawn_hat_launch = PathJoinSubstitution(
        [pkg_talking_turtle, 'launch', 'turtlebot4_spawn_hat.launch.py'])

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

    # Second robot (donnie) — uses the SAME filtered spawn as raph (no hat for now).
    # To re-enable the blue hat later, give the filtered spawn a description arg and pass
    # turtlebot4_hat.urdf.xacro (+ hat_color) here; see the deferred notes in the plan.
    robot2_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace2')),
            ('x', LaunchConfiguration('x2')),
            ('y', LaunchConfiguration('y2')),
            ('z', LaunchConfiguration('z2')),
            ('yaw', LaunchConfiguration('yaw2'))]
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

    # On shutdown, force-kill lingering Ignition Gazebo + spawn processes so the next
    # launch starts from a clean world (Ignition v6 often orphans its server on Ctrl-C).
    # Graceful SIGINT first, then SIGKILL any stragglers.
    cleanup = ExecuteProcess(
        cmd=['bash', '-c',
             'pkill -INT -f "ign gazebo"; sleep 2; '
             'pkill -9 -f "ign gazebo"; '
             'pkill -9 -f "ros_gz_sim/create"; '
             'pkill -9 -f "ros_gz_bridge/parameter_bridge"'],
        output='screen',
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(ign_gazebo_resource_path)
    ld.add_action(ignition)
    ld.add_action(robot_spawn)
    # Stagger donnie so raph's ign_ros2_control controller_manager fully initializes and loads
    # its controllers BEFORE donnie's control plugin starts. Both controller_managers run in the
    # one Gazebo process; launching them simultaneously makes their executors contend and the
    # controller spawners time out ("waiting for /…/controller_manager/list_controllers").
    # Sequential startup mirrors the working "separate terminals" multi-robot workaround
    # (turtlebot4_simulator#60). NOTE: this is for controller init, NOT the render race (that's
    # fixed by the world-scope Sensors/Contact systems). Tune the period up if donnie still stalls.
    #ld.add_action(TimerAction(period=20.0, actions=[robot2_spawn]))
    ld.add_action(gz_bridge_node)
    ld.add_action(RegisterEventHandler(OnShutdown(on_shutdown=[cleanup])))
    return ld
