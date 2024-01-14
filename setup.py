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
            'domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin',
            'domains/circuit_training/circuit_training/configs/envdesign.gin',

            # Include designs for creating web navigation environments
            'domains/web_navigation/environment_generation/data/difficulty_levels.zip',

            # Include base gminiwob directory with html and css files
            'domains/web_navigation/gwob/gminiwob/*',
            'domains/web_navigation/gwob/gminiwob/images/*',
            
            # Include all the motion files for the quadruped locomotion domain
            'domains/quadruped_locomotion/motion_imitation/data/motions/*.txt',

            # Include package data from codecarbon's setup.py
            "metrics/system/codecarbon/codecarbon/data/cloud/impact.csv",
            "metrics/system/codecarbon/codecarbon/data/hardware/cpu_power.csv",
            "metrics/system/codecarbon/codecarbon/data/private_infra/2016/usa_emissions.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/2016/canada_energy_mix.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/carbon_intensity_per_source.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/global_energy_mix.json",
            "metrics/system/codecarbon/codecarbon/viz/assets/*.png",
        ]
    }
)
