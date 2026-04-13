from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'floor_object_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dhia',
    maintainer_email='dhia@example.com',
    description='Depth-first floor object detection and temporal tracking for RGB-D cameras.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'floor_object_detector = floor_object_detection.floor_object_detector_node:main',
        ],
    },
)
