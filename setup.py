from setuptools import setup, find_packages

setup(
        name='rl_perf',
        version='0.1',
        packages=find_packages(),
        install_requires=[
                'numpy',
                'psutil',
                'gin-config',
                ],
        )
