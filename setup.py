from setuptools import setup, find_packages

#
# setup(
#     name='a2perf',
#     version='0.1.0',
#     packages=find_packages(),
#     install_requires=[
#         'psutil',
#         'gin-config',
#         'gymnasium',
#         'minari',
#         'absl-py'
#     ],
# )

# Within the package setup, we need to make sure to include the web_navigation_env_config.gin file
setup(
    name='a2perf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'psutil',
        'gin-config',
        'gymnasium',
        'minari',
        'absl-py'
    ],
    package_data={
        'a2perf': [
            'domains/web_navigation/configs/web_navigation_env_config.gin',
            'domains/quadruped_locomotion/configs/quadruped_locomotion_env_config.gin',
            'domains/circuit_training/configs/circuit_training_env_config.gin'
        ]
    }
)
