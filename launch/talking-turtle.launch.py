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

    # Control node, bridges grid to coords
    control_node = GroupAction([
        Node(
            package='talking-turtle',
            executable='node_Control',
            name='node_Control',
            output='screen',
            emulate_tty=True,
            remappings=[
                ('/cmd_vel', '/' + bot_name + '/cmd_vel')
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

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node'
    )

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
            {'world_origin_x': 0.0},
            {'world_origin_y': 5.5},
            {'metres_per_pixel_x': 1.0 / 138.0},
            {'metres_per_pixel_y': -1.0 / 152.0},
        ],
        remappings=[
            ('/raw_map', '/ids_overhead/image'),
            ('/raph/sim_ground_truth_pose', '/' + bot_name + '/sim_ground_truth_pose'),
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
        path_translator_node,
        speech_gen_node,
        listener_node,
        joy_node,
        mapper_node,
        path_visualizer_node,
        # logger_node,
    ])
