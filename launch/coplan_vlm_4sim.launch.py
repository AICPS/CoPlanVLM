import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = str(get_package_share_directory('coplan_vlm'))
    grid_csv_path = pkg_dir + '/config/grid_cell_centers.csv'
    # Debug artifacts go to the WORKSPACE debug/ dir (not the install share tree), one directory
    # per environment so a sim run and a lab run never overwrite each other. Four levels up from
    # <ws>/install/coplan_vlm/share/coplan_vlm is the workspace root — same traversal map_gen uses
    # for the shared occupancy snapshot.
    ws_root = os.path.normpath(os.path.join(pkg_dir, '..', '..', '..', '..'))
    debug_dir = os.path.join(ws_root, 'debug', 'gazebo_sim')

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
    # Every robot_names roster below lists donnie FIRST, to match the key order of
    # test_data/poses.json used by scripts/test_pipeline.py. The order sets which robot is
    # %ROBOT_A% in the VLM prompt and which marker colour each robot gets (index 0 = magenta), so
    # keeping it identical makes live prompts byte-comparable with the offline harness.

    # Executive API node, OpenAI pathing
    exec_api_node = GroupAction([
        Node(
            package='coplan_vlm',
            executable='node_Executive_API',
            name='node_Executive_API',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'openai_api_key': api_key},
                {'robot_names': [bot2_name, bot_name]},
                {'replan_mode': LaunchConfiguration('replan_mode')},
                {'replan_period': ParameterValue(
                    LaunchConfiguration('replan_period'), value_type=float)},
                {'camera': 'gazebo'},
                # Same dir the translator writes its images to, so the VLM prompt/response land
                # beside that run's artifacts.
                {'debug_dir': debug_dir},
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
            package='coplan_vlm',
            executable='node_Control',
            name='node_Control',
            output='screen',
            emulate_tty=True,
            parameters=[
                # robot_name says WHICH robot this instance drives; robot_names is the same roster
                # the other three nodes take. The CBF safety filter derives its peers as
                # roster-minus-self, so a mismatch here disarms peer avoidance (and is logged).
                {'robot_name': bot_name},
                {'robot_names': [bot2_name, bot_name]},
                {'enable_safety_filter': LaunchConfiguration('safety_filter')},
            ],
            remappings=[
                ('/cmd_vel', '/' + bot_name + '/cmd_vel'),
                ('/waypoint_path', '/' + bot_name + '/waypoint_path'),
                ('/pose_stamped', '/' + bot_name + '/ned/pose_stamped'),
            ],
        ),
    ])

    # Odometry to Pose node
    odometry_to_pose_node = Node(
            package='coplan_vlm',
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
            package='coplan_vlm',
            executable='node_Control',
            name='node_Control_2',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'robot_name': bot2_name},
                {'robot_names': [bot2_name, bot_name]},
                {'enable_safety_filter': LaunchConfiguration('safety_filter')},
            ],
            remappings=[
                ('/cmd_vel', '/' + bot2_name + '/cmd_vel'),
                ('/waypoint_path', '/' + bot2_name + '/waypoint_path'),
                ('/pose_stamped', '/' + bot2_name + '/ned/pose_stamped'),
            ],
        ),
    ])

    odometry_to_pose_node_2 = Node(
            package='coplan_vlm',
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
        package="coplan_vlm",
        executable="node_Path_Translator",
        name="node_Path_Translator",
        output='screen',
        emulate_tty=True,
        parameters=[
            {'grid_csv': grid_csv_path},
            {'debug_dir': debug_dir},
            {'save_debug': True},
            {'robot_names': [bot2_name, bot_name]},
            {'camera': 'gazebo'},   # overhead camera calibration for pixel<->world
        ]
    )
    
    # Path visualizer node
    path_visualizer_node = Node(
        package='coplan_vlm',
        executable='node_Path_Visualizer',
        name='node_Path_Visualizer',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'grid_csv': grid_csv_path},
            {'circle_radius': 16},
            {'robot_names': [bot2_name, bot_name]},
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
        DeclareLaunchArgument(
            'safety_filter', default_value='true', choices=['true', 'false'],
            description='CBF-QP collision-avoidance filter on the controller output. Set false to '
                        'restore the pre-filter behaviour exactly (the filter is bypassed, not '
                        'merely inactive).'),
        exec_api_node,
        control_node,
        odometry_to_pose_node,
        control_node_2,
        odometry_to_pose_node_2,
        path_translator_node,
        path_visualizer_node,
    ])
