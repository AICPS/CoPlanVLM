from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'coplan_vlm'

def files_only(pattern):
    """glob() that skips directories (e.g. a stray __pycache__ picked up by scripts/*)."""
    return [f for f in glob(pattern) if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.0.0',
    package_dir={'': 'nodes'},
    packages=find_packages(where='nodes'),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # config/.env is gitignored and may not exist yet; files_only() globs it so a fresh clone
        # still builds. The launch files read it from this installed location.
        ('share/' + package_name + '/config/',
            files_only('config/.env') + ['config/.env.example', 'config/grid_cell_centers.csv']),
        (f'share/{package_name}/launch', files_only('launch/*.py')),
        (f'share/{package_name}/scripts', files_only('scripts/*')),
        (f'share/{package_name}/world', ['world/sim_world.sdf']),
    ],

    install_requires=['setuptools', 'openai', 'python-dotenv', 'ament_index_python', 'rclpy'],
    zip_safe=True,
    maintainer='David',
    maintainer_email='davidrm3@uci.edu',
    description='TurtleBot control w/ OpenAI (adaptive Executive path planner)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'node_Executive_API = node_Executive_API.exec:main',
            'node_Control = node_Control.control:main',
            'node_Path_Translator = node_Path_Translator.translator_node:main',
            'node_Path_Visualizer = node_Path_Visualizer.path_visualizer:main',
            'node_Odometry_To_Pose = node_Odometry_To_Pose.odom_to_pose:main',
            'obs_seg_cli = obs_seg.cli:main',
        ],
    },
)
