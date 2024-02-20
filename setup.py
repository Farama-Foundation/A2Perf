import os
import shutil
import site
import sys
import urllib.request

from absl import logging
from setuptools import Command
from setuptools import find_packages
from setuptools import setup


class DreamplaceInstall(Command):
  """Custom command to download and extract Dreamplace."""
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Dynamically determine the dreamplace version based on the Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dreamplace_version = f"dreamplace_python{python_version}.tar.gz"
    logging.info(f"Installing Dreamplace version {dreamplace_version}")

    dreamplace_url = f"https://storage.googleapis.com/rl-infra-public/circuit-training/dreamplace/{dreamplace_version}"
    logging.info(f"Downloading Dreamplace from {dreamplace_url}")

    site_packages_path = site.getsitepackages()[0]
    dreamplace_dir = os.path.join(site_packages_path, 'dreamplace')
    logging.info(f"Installing Dreamplace to {dreamplace_dir}")

    os.makedirs(dreamplace_dir, exist_ok=True)
    tar_path = os.path.join(dreamplace_dir, "dreamplace.tar.gz")

    logging.info(f"Downloading Dreamplace to {tar_path}")
    urllib.request.urlretrieve(dreamplace_url, tar_path)

    logging.info('Extracting Dreamplace')
    shutil.unpack_archive(tar_path, dreamplace_dir)

    logging.info('Dreamplace installed')

    # Create an __init__.py file in the top-level Dreamplace directory
    init_path = os.path.join(dreamplace_dir, '__init__.py')
    with open(init_path, 'w') as init_file:
      init_file.write(
          "from .dreamplace import ("
          "PlaceDB, Params, NonLinearPlace, Placer, "
          "PlaceObj, EvalMetrics, BasicPlace, "
          "NesterovAcceleratedGradientOptimizer)\n"
      )


setup(
    name='a2perf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'psutil',
        'gin-config',
        'gymnasium',
        'gym',
        'minari',
        'absl-py'
    ],
    cmdclass={
        'dreamplace_install': DreamplaceInstall,
    },
    package_data={
        'a2perf': [
            # Include the default gin config files for running the becnhmark
            'submission/**/*.gin',

            # Include default gin config files for each domain
            'domains/web_navigation/configs/web_navigation_env_config.gin',
            'domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin',
            'domains/circuit_training/circuit_training/configs/envdesign.gin',

            # Include designs for creating web navigation environments
            'domains/web_navigation/environment_generation/data/difficulty_levels.zip',

            # Include gminiwob files for displaying web navigation environments
            'domains/web_navigation/**/*.html',
            'domains/web_navigation/**/*.css',
            'domains/web_navigation/**/*.js',
            'domains/web_navigation/**/*.png',

            # Include all the motion files for the quadruped locomotion domain
            'domains/quadruped_locomotion/motion_imitation/data/motions/*.txt',

            # Include the netlist and initial placement files for the circuit training domain
            'domains/circuit_training/**/*.pb.txt',
            'domains/circuit_training/**/*.plc',

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
