import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory


# Real-world deployment variant of coplan_vlm-4sim.launch.py.
#
# Identical to the sim launch EXCEPT it omits the two node_Odometry_To_Pose converters.
# Those only existed to turn the sim's /<robot>/sim_ground_truth_pose (nav_msgs/Odometry)
# into the /<robot>/ned/pose_stamped (PoseStamped) that control/translator/visualizer
# consume. In the real world the MoCap system already publishes /<robot>/ned/pose_stamped
# directly, so no conversion is needed. Everything downstream is unchanged.
def generate_launch_description():
    pkg_dir = str(get_package_share_directory('coplan_vlm'))
    grid_csv_path = pkg_dir + '/config/grid_cell_centers.csv'
    # Debug artifacts go to the WORKSPACE debug/ dir (not the install share tree), one directory
    # per environment so a lab run and a sim run never overwrite each other. Four levels up from
    # <ws>/install/coplan_vlm/share/coplan_vlm is the workspace root — same traversal map_gen uses
    # for the shared occupancy snapshot.
    ws_root = os.path.normpath(os.path.join(pkg_dir, '..', '..', '..', '..'))
    debug_dir = os.path.join(ws_root, 'debug', 'deploy_real')

    # Load environment variables from .env
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config/.env')
    load_dotenv(dotenv_path=env_path)

    # Define my API key
    api_key = os.getenv('MY_API_KEY')
    if api_key is None:
        raise RuntimeError(f"MY_API_KEY not found in .env file at {env_path}")

    # Define Robot's Name
    bot_name = 'raph'
    bot2_name = 'donnie'   # second robot stubbed control on /donnie/* topics

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
                {'robot_names': [bot_name, bot2_name]},
                {'replan_mode': LaunchConfiguration('replan_mode')},
                {'replan_period': ParameterValue(
                    LaunchConfiguration('replan_period'), value_type=float)},
                {'camera_info_topic': '/ueye/test/camera_info'},
                {'camera': 'lab_test'},
                # Same dir the translator writes its images to, so the VLM prompt/response land
                # beside that run's artifacts.
                {'debug_dir': debug_dir},
            ],
            remappings=[
                ('/camera_image', '/ueye/test/image_raw'),
            ],
        ),
    ])

    # Control node, does low level control of the robot (drive to goal, etc.).
    # Fully namespaced into raph's topics (symmetric with donnie below): the translator
    # publishes raph's plan to /raph/waypoint_path and pose comes from the MoCap topic
    # /raph/ned/pose_stamped (remapped onto the node's /pose_stamped input).
    control_node = GroupAction([
        Node(
            package='coplan_vlm',
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

    # --- Robot 2 (donnie): the same control node, remapped into the donnie namespace.
    # The executive now plans for both robots and the translator routes donnie's plan to
    # /donnie/waypoint_path (it can still be driven manually by publishing there directly).
    control_node_2 = GroupAction([
        Node(
            package='coplan_vlm',
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
            {'robot_names': [bot_name, bot2_name]},
            {'camera': 'lab_test'},   # overhead camera calibration for pixel<->world
        ]
    )

    # joy_node disabled for now — only the (now-removed) audio push-to-talk node consumed /joy,
    # and the unconditional joy_node was leaking orphaned processes.
    # joy_node = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node'
    # )

    # Path visualizer node
    path_visualizer_node = Node(
        package='coplan_vlm',
        executable='node_Path_Visualizer',
        name='node_Path_Visualizer',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'grid_csv': grid_csv_path},
            {'save_overlays': True},
            {'line_thickness': 8},
            {'circle_radius': 16},
            {'world_path_color': [200, 200, 200]},  # Light gray for reference path
            {'robot_color': [255, 0, 255]},  # Magenta for robot
            {'robot_names': [bot_name, bot2_name]},
            {'camera': 'lab_test'},   # overhead camera calibration for world<->pixel
            # Real camera intrinsics — the SAME topic exec uses, so both nodes undistort the frame
            # identically. Without this the node keeps the sim default (/ids_overhead/camera_info),
            # never receives intrinsics, and cv2.undistort is silently skipped: overlays would be
            # projected onto a distorted frame (error is <2 px near the image centre but ~175 px,
            # roughly 0.8 m, at the corners) while exec plans on an undistorted one.
            {'camera_info_topic': '/ueye/test/camera_info'},
        ],
        remappings=[
            ('/camera_image', '/ueye/test/image_raw'),
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
        control_node_2,
        path_translator_node,
        # joy_node,   # disabled — see commented joy_node definition above
        path_visualizer_node,
    ])
