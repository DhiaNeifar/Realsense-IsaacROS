from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'realsense_benchmark'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dhianeifar',
    maintainer_email='dhianeifar@todo.todo',
    description='Benchmark utilities for RealSense RGB-D stream throughput and inference load testing.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "live_benchmark_node = realsense_benchmark.live_benchmark_node:main",
            "phase_benchmark_node = realsense_benchmark.phase_benchmark_node:main",
            "detection_benchmark_node = realsense_benchmark.detection_benchmark_node:main",
        ],
    },
)
