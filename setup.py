from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'coplan_vlm'

def files_only(pattern):
    """glob() that skips directories (e.g. a stray __pycache__ picked up by scripts/*)."""
    return [f for f in glob(pattern) if os.path.isfile(f)]

def get_data_files(source_dir, dest_prefix):
    data_files = []
    for root, dirs, files in os.walk(source_dir):
        if files:
            dest = os.path.join(dest_prefix, os.path.relpath(root, os.path.dirname(source_dir)))
            data_files.append((dest, [os.path.join(root, f) for f in files]))
    return data_files

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
        (f'share/{package_name}/launch', files_only('launch/*.py')),
        (f'share/{package_name}/scripts', files_only('scripts/*')),
        (f'share/{package_name}/world', ['world/house.sdf', 'world/sim_world.sdf']),
        (f'share/{package_name}/world/materials/script', glob('world/materials/script/*')),
        (f'share/{package_name}/world/materials/textures', glob('world/materials/textures/*')),
    ] + get_data_files('world/meshes', f'share/{package_name}/world'),

    install_requires=['setuptools', 'openai', 'python-dotenv', 'ament_index_python', 'rclpy', 'keyboard'],
    zip_safe=True,
    maintainer='David',
    maintainer_email='davidrm3@uci.edu',
    description='TurtleBot control w/ OpenAI (adaptive Executive path planner)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_Executive_API = node_Executive_API.exec:main',
            'node_Control = node_Control.control:main',
            'node_Path_Translator = node_Path_Translator.translate:main',
            'node_Path_Visualizer = node_Path_Visualizer.path_visualizer:main',
            'node_Trajectory_Logger = node_Path_Translator.logger_node:main',
            'node_Odometry_To_Pose = node_Path_Translator.odom_to_pose:main',
            'obs_seg_cli = obs_seg.cli:main',
        ],
    },
)
