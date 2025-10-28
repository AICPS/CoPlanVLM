from setuptools import find_packages, setup

package_name = 'talking-turtle'
node_path = 'talking-turtle.nodes.'

setup(
    name=package_name,
    version='0.0.0',
    package_dir={'': 'nodes'},
    packages=find_packages(where='nodes'),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config/', ['config/.env', 'config/grid_cell_centers.csv', 'config/transparent_grid.png']),
        (f'share/{package_name}/launch', ['launch/talking-turtle.launch.py', 'launch/turtlebot4_ignition.launch.py', 'launch/turtlebot4_spawn.launch.py', 'launch/ignition.launch.py']),
        (f'share/{package_name}/world', ['world/sim_world.sdf']),

    ],
    install_requires=['setuptools', 'openai', 'python-dotenv', 'ament_index_python', 'pyaudio', 'rclpy', 'keyboard'],
    zip_safe=True,
    maintainer='Caleb Craddock',
    maintainer_email='c27caleb.craddock@afacademy.af.edu',
    description='2-node TurtleBot control w/ OpenAI',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_Executive_API = node_Executive_API.exec:main',
            'node_Triage_API = node_Triage_API.triage:main',
            'node_Control = node_Control.control:main',
            'node_Path_Translator = node_Path_Translator.translate:main',
            'node_Speech_Gen = node_Speech_Gen.speech:main',
            'node_Listener = node_Listener.ear:main',
            'node_Map_Gen = node_Map_Gen.mapper:main',
            'node_Trajectory_Logger = node_Path_Translator.logger_node:main',
            'node_Odometry_To_Pose = node_Path_Translator.odom_to_pose:main',
        ],
    },
)
