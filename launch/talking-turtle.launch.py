import os
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = str(get_package_share_directory('talking-turtle'))
    map_file = pkg_dir + '/map.png'
    grid_csv_path = pkg_dir + '/config/grid_cell_centers.csv'

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
        parameters=[
            {'grid_csv': grid_csv_path}
        ]
    )
    
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
    ])

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
    ])

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


    return LaunchDescription([
        exec_api_node,
        triage_api_node,
        control_node,
        odometry_to_pose_node,
        path_translator_node,
        speech_gen_node,
        listener_node,
        joy_node,
        mapper_node,
        # logger_node,
    ])
