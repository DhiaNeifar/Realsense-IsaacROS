from setuptools import setup

package_name = 'floor_box_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/box_floor_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dhia',
    maintainer_email='dhia@example.com',
    description='Simple RGB+Depth box detector using OpenCV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'box_floor_detector = floor_box_perception.box_floor_detector_node:main',
        ],
    },
)
