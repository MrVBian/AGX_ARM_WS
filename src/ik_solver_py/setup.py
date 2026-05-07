from setuptools import find_packages, setup

package_name = 'ik_solver_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pinocchio',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='zhenwei.bian@zhangmen.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_example = ik_solver_py.main:main',
        ],
    },
)
