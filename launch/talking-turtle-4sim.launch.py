import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = str(get_package_share_directory('talking-turtle'))
    grid_csv_path = pkg_dir + '/config/grid_cell_centers.csv'
    # Debug artifacts dir, resolved relative to this package (a `debug/` folder at the package
    # root, alongside launch/ and config/). Same __file__-relative pattern as config/.env below.
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'debug'))

    # Load environment variables from .env
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config/.env')
    load_dotenv(dotenv_path=env_path)

    # Define my API key
    api_key = os.getenv('MY_API_KEY')
    if api_key is None:
        raise RuntimeError(f"MY_API_KEY not found in .env file at {env_path}")
    
    # Define Robot's Name
    bot_name = 'raph'
    bot2_name = 'donnie'   # second robot (blue hat); stubbed control on /donnie/* topics

    # Executive API node, OpenAI pathing
    exec_api_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Executive_API',
            name='node_Executive_API',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'openai_api_key': api_key},
                {'robot_names': [bot_name, bot2_name]},
                {'replan_mode': LaunchConfiguration('replan_mode')},
                {'replan_period': ParameterValue(
                    LaunchConfiguration('replan_period'), value_type=float)},
                {'camera': 'gazebo'},
            ],
            remappings=[
                ('/camera_image', '/ids_overhead/image'),
            ],
        ),
    ])

    # Control node, does low level control of the robot (drive to goal, etc.).
    # Fully namespaced into raph's topics (symmetric with donnie below): the translator
    # publishes raph's plan to /raph/waypoint_path and pose comes from /raph/ned/pose_stamped
    # (published by node_Odometry_To_Pose in sim, or by MoCap directly in the real world).
    control_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Control',
            name='node_Control',
            output='screen',
            emulate_tty=True,
            remappings=[
                ('/cmd_vel', '/' + bot_name + '/cmd_vel'),
                ('/waypoint_path', '/' + bot_name + '/waypoint_path'),
                ('/pose_stamped', '/' + bot_name + '/ned/pose_stamped'),
            ],
        ),
    ])

    # Odometry to Pose node
    odometry_to_pose_node = Node(
            package='talking-turtle',
            executable='node_Odometry_To_Pose',
            name='node_Odometry_To_Pose',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'input_topic': '/' + bot_name + '/sim_ground_truth_pose'},
            ],
            remappings=[
                ('/pose_stamped', '/' + bot_name + '/ned/pose_stamped'),
            ],
        )

    # --- Robot 2 (donnie): the same control/odom nodes, remapped into the donnie namespace.
    # The executive now plans for both robots and the translator routes donnie's plan to
    # /donnie/waypoint_path (it can still be driven manually by publishing there directly).
    control_node_2 = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Control',
            name='node_Control_2',
            output='screen',
            emulate_tty=True,
            remappings=[
                ('/cmd_vel', '/' + bot2_name + '/cmd_vel'),
                ('/waypoint_path', '/' + bot2_name + '/waypoint_path'),
                ('/pose_stamped', '/' + bot2_name + '/ned/pose_stamped'),
            ],
        ),
    ])

    odometry_to_pose_node_2 = Node(
            package='talking-turtle',
            executable='node_Odometry_To_Pose',
            name='node_Odometry_To_Pose_2',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'input_topic': '/' + bot2_name + '/sim_ground_truth_pose'},
            ],
            remappings=[
                ('/pose_stamped', '/' + bot2_name + '/ned/pose_stamped'),
            ],
        )

    # Translator node, grid coords to MoCap coords
    path_translator_node = Node(
        package="talking-turtle",
        executable="node_Path_Translator",
        name="node_Path_Translator",
        output='screen',
        emulate_tty=True,
        parameters=[
            {'grid_csv': grid_csv_path},
            {'debug_dir': debug_dir},
            {'save_debug': True},
            {'robot_names': [bot_name, bot2_name]},
            {'camera': 'gazebo'},   # overhead camera calibration for pixel<->world
        ]
    )
    
    # joy_node disabled for now — only the (now-removed) audio push-to-talk node consumed /joy,
    # and the unconditional joy_node was leaking orphaned processes.
    # joy_node = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node'
    # )

    # Data recording node (optional)
    # Start only when pressed button 2 on joystick
    logger_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Trajectory_Logger',
            name='node_Trajectory_Logger',
            output='screen',
            emulate_tty=True,
        ),
    ])

    # Path visualizer node
    path_visualizer_node = Node(
        package='talking-turtle',
        executable='node_Path_Visualizer',
        name='node_Path_Visualizer',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'grid_csv': grid_csv_path},
            {'map_path': pkg_dir + '/map_raw.png'},
            {'save_overlays': True},
            {'line_thickness': 8},
            {'circle_radius': 16},
            {'world_path_color': [200, 200, 200]},  # Light gray for reference path
            {'robot_color': [255, 0, 255]},  # Magenta for robot
            {'robot_names': [bot_name, bot2_name]},
            {'camera': 'gazebo'},   # overhead camera calibration for world<->pixel
        ],
        remappings=[
            ('/camera_image', '/ids_overhead/image'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'replan_mode', default_value='static',
            description='Executive replanning strategy: "static" (plan once per prompt) or '
                        '"dynamic" (re-plan the same prompt every replan_period seconds).'),
        DeclareLaunchArgument(
            'replan_period', default_value='15.0',
            description='Seconds between dynamic replans (used only when replan_mode:=dynamic).'),
        exec_api_node,
        control_node,
        odometry_to_pose_node,
        control_node_2,
        odometry_to_pose_node_2,
        path_translator_node,
        # joy_node,   # disabled — see commented joy_node definition above
        path_visualizer_node,
        # logger_node,
    ])
