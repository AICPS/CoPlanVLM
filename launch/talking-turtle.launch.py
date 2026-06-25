import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = str(get_package_share_directory('talking-turtle'))
    map_file = pkg_dir + '/map.png'
    grid_csv_path = pkg_dir + '/config/grid_cell_centers.csv'
    # Debug artifacts dir (hardcoded; workspace-level, outside the git repo & install tree).
    debug_dir = os.path.expanduser('~/projects/turtle4_ws/debug')

    # Load environment variables from .env
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config/.env')
    load_dotenv(dotenv_path=env_path)

    # Define my API key
    api_key = os.getenv('MY_API_KEY')
    if api_key is None:
        raise RuntimeError(f"MY_API_KEY not found in .env file at {env_path}")
    
    # Define Robot's Name
    bot_name = 'raph'
    bot2_name = 'raph2'   # second robot (blue hat); stubbed control on /raph2/* topics

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
                {'map_path': map_file},
                {'robot_names': [bot_name, bot2_name]},
            ],
        ),
    ])

    # Triage API node, OpenAI interactions
    triage_api_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Triage_API',
            name='node_Triage_API',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'openai_api_key': api_key},
            ],
        ),
    ])

    # Control node, does low level control of the robot (drive to goal, etc.).
    # Fully namespaced into raph's topics (symmetric with raph2 below): the translator
    # publishes raph's plan to /raph/waypoint_path and odom publishes /raph/pose_stamped.
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
                ('/pose_stamped', '/' + bot_name + '/pose_stamped'),
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
                ('/pose_stamped', '/' + bot_name + '/pose_stamped'),
            ],
        )

    # --- Robot 2 (raph2): the same control/odom nodes, remapped into the raph2 namespace.
    # The executive now plans for both robots and the translator routes raph2's plan to
    # /raph2/waypoint_path (it can still be driven manually by publishing there directly).
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
                ('/pose_stamped', '/' + bot2_name + '/pose_stamped'),
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
                ('/pose_stamped', '/' + bot2_name + '/pose_stamped'),
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
        ]
    )
    
    # Audio nodes (TTS output + STT input) — only launched when use_audio:=true.
    # They require the `sounddevice` package; off by default so the stack runs quietly.
    use_audio = IfCondition(LaunchConfiguration('use_audio'))

    speech_gen_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Speech_Gen',
            name='node_Speech_Gen',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'openai_api_key': api_key},
            ],
        ),
    ], condition=use_audio)

    listener_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Listener',
            name='node_Listener',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'openai_api_key': api_key},
            ],
        ),
    ], condition=use_audio)

    # joy_node disabled for now — only the audio push-to-talk node (node_Listener, use_audio)
    # consumes /joy, and the unconditional joy_node was leaking orphaned processes.
    # joy_node = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node'
    # )

    mapper_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Map_Gen',
            name='node_Map_Gen',
            output='screen',
            emulate_tty=True,
              remappings=[
                ('/raw_map', '/ids_overhead/image'),
            ],
        ),
    ])

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
        ],
        remappings=[
            ('/raw_map', '/ids_overhead/image'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_audio', default_value='false',
            description='Launch the audio nodes (node_Speech_Gen TTS + node_Listener STT). '
                        'Requires the sounddevice package; default false.'),
        exec_api_node,
        triage_api_node,
        control_node,
        odometry_to_pose_node,
        control_node_2,
        odometry_to_pose_node_2,
        path_translator_node,
        speech_gen_node,
        listener_node,
        # joy_node,   # disabled — see commented joy_node definition above
        mapper_node,
        path_visualizer_node,
        # logger_node,
    ])
