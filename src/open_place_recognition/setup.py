import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'open_place_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker_opr_ros2',
    maintainer_email='amelekhin96@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'place_recognition = open_place_recognition.place_recognition:main',
            'visualizer = open_place_recognition.visualizer:main',
            'localization = open_place_recognition.localization:main',
        ],
    },
)
