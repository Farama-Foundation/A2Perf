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


setup(
    name="a2perf",
    version=get_version(),
    long_description=get_description(),
)
