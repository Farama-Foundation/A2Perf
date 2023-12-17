from setuptools import setup, find_packages

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

            # Include default gin config files for each domain
            'domains/web_navigation/configs/web_navigation_env_config.gin',
            'domains/quadruped_locomotion/configs/quadruped_locomotion_env_config.gin',
            'domains/circuit_training/configs/circuit_training_env_config.gin'

            # Include designs for creating web navigation environments
            'domains/web_navigation/environment_generation/data/difficulty_levels.zip',
        ]
    }
)
