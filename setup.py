"""Sets up the project."""

import importlib.util
import logging
import os
import pathlib
import shutil
import site
import sys
import urllib.request
from distutils.cmd import Command

from setuptools import setup

logging.basicConfig(level=logging.INFO)


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the a2perf version."""
    path = CWD / "a2perf" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description


def install_dreamplace():
    # Dynamically determine the dreamplace version based on the Python version
    python_version = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
    dreamplace_version = "dreamplace_python%s.tar.gz" % python_version
    logging.info("Installing Dreamplace version %s", dreamplace_version)

    dreamplace_url = (
        "https://storage.googleapis.com/rl-infra-public/circuit-training/dreamplace/%s"
        % dreamplace_version
    )
    logging.info("Downloading Dreamplace from %s", dreamplace_url)

    site_packages_path = site.getsitepackages()[0]
    dreamplace_dir = os.path.join(site_packages_path, "dreamplace")
    logging.info("Installing Dreamplace to %s", dreamplace_dir)

    os.makedirs(dreamplace_dir, exist_ok=True)
    tar_path = os.path.join(dreamplace_dir, "dreamplace.tar.gz")

    logging.info("Downloading Dreamplace to %s", tar_path)
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
    """Set the executable permissions for the plc_wrapper_main binary
    and raise an error if it fails."""
    package_dir = os.path.dirname(importlib.util.find_spec("a2perf").origin)
    binary_path = os.path.join(
        package_dir, "domains", "circuit_training", "bin", "plc_wrapper_main"
    )
    logging.info("Location of plc_wrapper_main: %s", binary_path)
    if os.path.exists(binary_path):
        try:
            os.chmod(binary_path, 0o755)
            logging.info("Executable permissions set for plc_wrapper_main.")
        except Exception as e:
            raise RuntimeError(
                "Failed to set executable permissions for plc_wrapper_main: %s" % e
            )
    else:
        # Raise an exception if the file is not found
        raise FileNotFoundError("plc_wrapper_main not found at expected path.")


class InstallCircuitTraining(Command):
    description = "Install circuit training dependencies and setup."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        install_dreamplace()
        set_executable_permissions()


setup(
    name="a2perf",
    version=get_version(),
    long_description=get_description(),
    cmdclass={
        "circuit_training": InstallCircuitTraining,
    },
)
