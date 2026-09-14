import os
from glob import glob

from setuptools import setup

package_name = 'pennair_shapes'

setup(
    name=package_name,
    version='0.1.0',
    # shape_detector is a symlink to the top-level algorithm package, so the
    # ROS nodes run the exact same code as the plain python scripts
    packages=[package_name, 'shape_detector'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anoushka',
    maintainer_email='sudeshsn@gmail.com',
    description='Video streamer + shape detector nodes for the PennAir application challenge.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'video_publisher = pennair_shapes.video_publisher:main',
            'detector_node = pennair_shapes.detector_node:main',
        ],
    },
)
