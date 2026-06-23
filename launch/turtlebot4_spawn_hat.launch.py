# Based on Clearpath's turtlebot4_spawn.launch.py (Apache-2.0). Identical to
# turtlebot4_spawn_filtered.launch.py except the robot_description is built from
# talking-turtle's turtlebot4_hat.urdf.xacro so the spawned robot carries a colored "hat"
# disk for overhead identification (so the VLM can tell two robots apart).
#
# As in the filtered launch, robot_description is produced by gen_robot_description.py, which
# runs the xacro and then strips the world-singleton `Sensors` + `Contact` system plugins from
# the output. The hat xacro <xacro:include>s the STOCK turtlebot4.urdf.xacro, so it inherits
# those same singleton plugins and needs the identical strip — without it, adding the 2nd robot
# triggers the multi-robot "Visual already exists" crash (turtlebot4_simulator#60). Those
# systems are declared once at world scope in sim_world.sdf instead. Nothing in /opt/ros is
# modified; the strip is a pure runtime xacro->stdout transform.
# Everything else (bridges, create3 nodes, static TFs, spawn-from-topic) is identical to stock.

from ament_index_python.packages import get_package_share_directory

from irobot_create_common_bringup.namespace import GetNamespacedName
from irobot_create_common_bringup.offset import OffsetParser

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue


ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='use_sim_time'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('namespace', default_value='raph2',
                          description='Robot namespace'),
    DeclareLaunchArgument('hat_color', default_value='0 0 1 1',
                          description='RGBA color of the overhead marker disk.'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():

    # Directories
    pkg_turtlebot4_ignition_bringup = get_package_share_directory(
        'turtlebot4_ignition_bringup')
    pkg_irobot_create_common_bringup = get_package_share_directory(
        'irobot_create_common_bringup')
    pkg_irobot_create_ignition_bringup = get_package_share_directory(
        'irobot_create_ignition_bringup')
    pkg_talking_turtle = get_package_share_directory('talking-turtle')

    # Paths
    turtlebot4_ros_ign_bridge_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'ros_ign_bridge.launch.py'])
    turtlebot4_node_launch = PathJoinSubstitution(
        [pkg_turtlebot4_ignition_bringup, 'launch', 'turtlebot4_nodes.launch.py'])
    create3_nodes_launch = PathJoinSubstitution(
        [pkg_irobot_create_common_bringup, 'launch', 'create3_nodes.launch.py'])
    create3_ignition_nodes_launch = PathJoinSubstitution(
        [pkg_irobot_create_ignition_bringup, 'launch', 'create3_ignition_nodes.launch.py'])
    # Hat description (standard TurtleBot 4 + colored hat) + our generator/strip script.
    hat_xacro = PathJoinSubstitution(
        [pkg_talking_turtle, 'urdf', 'turtlebot4_hat.urdf.xacro'])
    gen_script = PathJoinSubstitution(
        [pkg_talking_turtle, 'scripts', 'gen_robot_description.py'])

    # Parameters
    param_file_cmd = DeclareLaunchArgument(
        'param_file',
        default_value=PathJoinSubstitution(
            [pkg_turtlebot4_ignition_bringup, 'config', 'turtlebot4_node.yaml']),
        description='Turtlebot4 Robot param file')

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    x, y, z = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')
    hat_color = LaunchConfiguration('hat_color')
    turtlebot4_node_yaml_file = LaunchConfiguration('param_file')

    robot_name = GetNamespacedName(namespace, 'turtlebot4')

    # Spawn robot slightly closer to the floor to reduce the drop
    z_robot = OffsetParser(z, 0.0)

    # NOTE: controllers (joint_state_broadcaster, diffdrive_controller) are NOT spawned here.
    # The ign_ros2_control controller_manager advertises its services before its resource manager
    # is ready to *configure* controllers, and a spawner only configures once (no retry) — so any
    # in-launch spawner fires too early and fails. They're loaded instead by scripts/
    # spawn_second_robot.sh, which waits a fixed delay after this launch so the CM is fully ready
    # (replicating the manual `ros2 run controller_manager spawner ...` timing that always works).

    spawn_robot_group_action = GroupAction([
        PushRosNamespace(namespace),

        # Robot description: hat xacro (standard TurtleBot 4 + colored hat), run through
        # gen_robot_description.py which strips the world-singleton Sensors + Contact system
        # plugins (now declared in sim_world.sdf).
        # `python3 <gen_script> <xacro> gazebo:=ignition namespace:=<ns> hat_color:="<rgba>"`.
        # hat_color is quoted so its spaces survive shlex tokenization of the xacro command.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_description': ParameterValue(Command([
                    'python3', ' ', gen_script, ' ',
                    hat_xacro, ' ',
                    'gazebo:=ignition', ' ',
                    'namespace:=', namespace, ' ',
                    'hat_color:="', hat_color, '"']), value_type=str)},
            ],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),

        # Spawn TurtleBot 4
        Node(
            package='ros_ign_gazebo',
            executable='create',
            arguments=['-name', robot_name,
                       '-x', x,
                       '-y', y,
                       '-z', z_robot,
                       '-Y', yaw,
                       '-topic', 'robot_description'],
            output='screen'
        ),

        # ROS IGN bridge
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([turtlebot4_ros_ign_bridge_launch]),
            launch_arguments=[
                ('model', LaunchConfiguration('model')),
                ('robot_name', robot_name),
                ('namespace', namespace)]
        ),

        # TurtleBot 4 nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([turtlebot4_node_launch]),
            launch_arguments=[('model', LaunchConfiguration('model')),
                              ('param_file', turtlebot4_node_yaml_file)]
        ),

        # Create 3 nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_nodes_launch]),
            launch_arguments=[
                ('namespace', namespace)
            ]
        ),

        # Create 3 Ignition nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_ignition_nodes_launch]),
            launch_arguments=[
                ('robot_name', robot_name),
            ]
        ),

        # RPLIDAR static transforms
        Node(
            name='rplidar_stf',
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=[
                '0', '0', '0', '0', '0', '0.0',
                'rplidar_link', [robot_name, '/rplidar_link/rplidar']],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ]
        ),

        # OAKD static transform
        # Required for pointcloud. See https://github.com/gazebosim/gz-sensors/issues/239
        Node(
            name='camera_stf',
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=[
                '0', '0', '0',
                '1.5707', '-1.5707', '0',
                'oakd_rgb_camera_optical_frame',
                [robot_name, '/oakd_rgb_camera_frame/rgbd_camera']
            ],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ]
        ),

    ])

    # Define LaunchDescription variable
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(param_file_cmd)
    ld.add_action(spawn_robot_group_action)
    return ld
