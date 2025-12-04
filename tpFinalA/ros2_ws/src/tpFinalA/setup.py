from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'tpFinalA'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ciror',
    maintainer_email='crussi@udesa.esc.ar',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	"my_fastslam = tpFinalA.slam:main",
            "my_odometry = tpFinalA.odometry:main",
            "save_map = tpFinalA.save_map:main",
            "plan_path = tpFinalA.path_planning:main",
            "controller = tpFinalA.navigation:main",
            'localization = tpFinalA.localization:main',
        ],
    },
)
