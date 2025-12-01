from setuptools import find_packages, setup

package_name = 'tpFinalA'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "my_navigation = tpFinalA.navigation:main"
        ],
    },
)
