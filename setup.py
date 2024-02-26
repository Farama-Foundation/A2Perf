import logging
import os
import shutil
import site
import sys
import urllib.request

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install


def install_dreamplace():
    # Dynamically determine the dreamplace version based on the Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dreamplace_version = f"dreamplace_python{python_version}.tar.gz"
    logging.info(f"Installing Dreamplace version {dreamplace_version}")

    dreamplace_url = f"https://storage.googleapis.com/rl-infra-public/circuit-training/dreamplace/{dreamplace_version}"
    logging.info(f"Downloading Dreamplace from {dreamplace_url}")

    site_packages_path = site.getsitepackages()[0]
    dreamplace_dir = os.path.join(site_packages_path, "dreamplace")
    logging.info(f"Installing Dreamplace to {dreamplace_dir}")

    os.makedirs(dreamplace_dir, exist_ok=True)
    tar_path = os.path.join(dreamplace_dir, "dreamplace.tar.gz")

    logging.info(f"Downloading Dreamplace to {tar_path}")
    urllib.request.urlretrieve(dreamplace_url, tar_path)

    logging.info("Extracting Dreamplace")
    shutil.unpack_archive(tar_path, dreamplace_dir)

    logging.info("Dreamplace installed")

    # Create a .pth file in the site-packages directory
    dreamplace_inner_dir = os.path.join(dreamplace_dir, "dreamplace")
    dreamplace_outer_pth = os.path.join(site_packages_path, "dreamplace.pth")
    with open(dreamplace_outer_pth, "w") as file:
        file.write(dreamplace_dir + "\n")

    dreamplace_inner_pth = os.path.join(site_packages_path, "dreamplace_dreamplace.pth")
    with open(dreamplace_inner_pth, "w") as file:
        file.write(dreamplace_inner_dir + "\n")


def set_executable_permissions():
    """Set the executable permissions for the plc_wrapper_main binary and raise an error if it fails."""
    binary_path = os.path.join(sys.prefix, "bin", "plc_wrapper_main")

    if os.path.exists(binary_path):
        try:
            os.chmod(binary_path, 0o755)
            logging.info("Executable permissions set for plc_wrapper_main.")
        except Exception as e:
            # Raise an exception to halt the installation
            raise RuntimeError(
                f"Failed to set executable permissions for plc_wrapper_main: {e}"
            )
    else:
        # Raise an exception if the file is not found
        raise FileNotFoundError("plc_wrapper_main not found at expected path.")


class CustomInstall(install):
    """Custom installation script to include Dreamplace and plc_wrapper_main installation."""

    def run(self):
        install.run(self)
        install_dreamplace()
        set_executable_permissions()


setup(
    name="a2perf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "psutil",
        "gin-config",
        "gymnasium",
        "gym",
        "minari",
        "absl-py",
    ],
    extras_require={
        "circuit-training": [
            "torch==1.13.1",
            "tensorflow",
            "tf-agents",
            "timeout-decorator",
            "matplotlib",
            "cairocffi",
            "shapely",
        ],
        "web-navigation": ["selenium", "webdriver-manager", "pillow", "regex"],
        "quadruped-locomotion": ["pybullet", "mpi4py", "tensorflow", "scipy", "attrs"],
        "all": [
            "torch==1.13.1",
            "selenium",
            "webdriver-manager",
            "pybullet",
            "tensorflow",
        ],
    },
    cmdclass={
        "install": CustomInstall,
    },
    data_files=[
        ("bin", ["bin/plc_wrapper_main"]),
    ],
    package_data={
        "a2perf": [
            # Include the default gin config files for running the benchmark
            "submission/**/*.gin",
            "domains/**/*.gin",
            # Include designs for creating web navigation environments
            "domains/web_navigation/environment_generation/data/difficulty_levels.zip",
            # Include gminiwob files for displaying web navigation environments
            "domains/web_navigation/**/*.html",
            "domains/web_navigation/**/*.css",
            "domains/web_navigation/**/*.js",
            "domains/web_navigation/**/*.png",
            # Include all the motion files for the quadruped locomotion domain
            "domains/quadruped_locomotion/motion_imitation/data/motions/*.txt",
            # Include the netlist and initial placement files for the circuit training domain
            "domains/circuit_training/**/*.pb.txt",
            "domains/circuit_training/**/*.plc",
            # Include package data from codecarbon's setup.py
            "metrics/system/codecarbon/codecarbon/data/cloud/impact.csv",
            "metrics/system/codecarbon/codecarbon/data/hardware/cpu_power.csv",
            "metrics/system/codecarbon/codecarbon/data/private_infra/2016/usa_emissions.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/2016/canada_energy_mix.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/carbon_intensity_per_source.json",
            "metrics/system/codecarbon/codecarbon/data/private_infra/global_energy_mix.json",
            "metrics/system/codecarbon/codecarbon/viz/assets/*.png",
        ]
    },
)
