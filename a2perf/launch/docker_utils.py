import os

from absl import logging
from xmanager import xm

from a2perf.constants import ENV_NAMES, BenchmarkDomain

GENERIC_GIN_CONFIG_NAME = "submission_config.gin"
DOCKER_EXPERIMENT_DIR = "/experiment_dir"
DOCKER_PARTICIPANT_DIR = "/participant_code"
DOCKER_DATASETS_PATH = "/workdir/datasets"


def _get_common_setup(uid: str, user: str):
    return [
        """
            ARG APT_COMMAND="apt-get -o Acquire::Retries=3 \
              --no-install-recommends -y"
            """,
        "ENV DEBIAN_FRONTEND=noninteractive",
        "ENV TZ=America/New_York",
        """
            RUN ${APT_COMMAND} update --allow-releaseinfo-change && \
              ${APT_COMMAND} install sudo \
              wget \
              software-properties-common \
              curl \
              tmux \
              telnet \
              net-tools \
              vim \
              less \
              unzip && \
              rm -rf /var/lib/apt/lists/*
              """,
        """
            RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
              ${APT_COMMAND} install -y g++-11
            """,
        f"""
        RUN if ! getent passwd {uid}; then \
              useradd -m -u {uid} {user}; \
            else \
              existing_user=$(getent passwd {uid} | cut -d: -f1); \
              if [ "{user}" != "$existing_user" ]; then \
                usermod -l {user} $existing_user; \
                usermod -d /home/{user} -m {user}; \
              fi; \
            fi
        """,
        f"""
        RUN echo "{user} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
        """,
    ]


def get_entrypoint(env_name: str, user: str) -> xm.CommandList:
    # flake8: noqa

    if env_name in ENV_NAMES[BenchmarkDomain.QUADRUPED_LOCOMOTION]:
        return xm.CommandList(
            [
                "echo $@",
                f"""
su {user} -c /bin/bash <<EOF
source /opt/conda/etc/profile.d/conda.sh &&
conda activate py39 &&
pip install -r {DOCKER_PARTICIPANT_DIR}/requirements.txt &&
a2perf {DOCKER_PARTICIPANT_DIR} --verbosity={logging.get_verbosity()} $@
EOF
""",
                # Waste the trailing "$@" argument
                "echo",
            ]
        )
    elif env_name in ENV_NAMES[BenchmarkDomain.WEB_NAVIGATION]:
        return xm.CommandList(
            [
                "echo $@",
                "service dbus start",
                f"""
su {user} -c /bin/bash <<EOF
source /opt/conda/etc/profile.d/conda.sh &&
conda activate py310 &&
pip install -r {DOCKER_PARTICIPANT_DIR}/requirements.txt &&
a2perf {DOCKER_PARTICIPANT_DIR} --verbosity={logging.get_verbosity()} $@
EOF
                    """,
                # Waste the trailing "$@" argument
                "echo",
            ]
        )
    elif env_name in ENV_NAMES[BenchmarkDomain.CIRCUIT_TRAINING]:
        return xm.CommandList(
            [
                "echo $@",
                f"""
su {user} -c /bin/bash <<EOF
source /opt/conda/etc/profile.d/conda.sh &&
conda activate py310 &&
pip install -r {DOCKER_PARTICIPANT_DIR}/requirements.txt &&
a2perf {DOCKER_PARTICIPANT_DIR} --verbosity={logging.get_verbosity()} $@
EOF
""",
                # Waste the trailing "$@" argument
                "echo",
            ]
        )
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    # flake8: qa


def get_docker_instructions(uid: str, user: str, env_name: str):
    repo_dir = os.path.basename(
        os.path.abspath(
            os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir)
        )
    )
    common_setup = _get_common_setup(uid, user)

    if env_name in ENV_NAMES[BenchmarkDomain.QUADRUPED_LOCOMOTION]:
        return common_setup + [
            "RUN mkdir -p /workdir",
            "WORKDIR /workdir",
            f"COPY {repo_dir}/quadruped_locomotion_environment.yml .",
            """
                    RUN conda update -n base -c conda-forge conda -y && \
                      conda env create -f /workdir/quadruped_locomotion_environment.yml --name py39 -y
                    """,
            f"COPY {repo_dir} .",
            f"""
            RUN chown -R {uid}:root /workdir && \
             /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                conda activate py39 && \
                pip install -e /workdir[all] seaborn matplotlib minari==0.4.3 && \
                python /workdir/setup.py install && \
                pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"
            """,
        ]
    elif env_name in ENV_NAMES[BenchmarkDomain.WEB_NAVIGATION]:
        return common_setup + [
            'ARG CHROME_VERSION="120.0.6099.109-1"',
            'ARG CHROMEDRIVER_VERSION="120.0.6099.109"',
            """
                    RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
                      ${APT_COMMAND} update && \
                      ${APT_COMMAND} --fix-broken install && \
                      ${APT_COMMAND} install /tmp/chrome.deb xvfb && \
                      rm /tmp/chrome.deb && \
                      rm -rf /var/lib/apt/lists/*
                    """,
            "RUN mkdir -p /workdir",
            "WORKDIR /workdir",
            f"COPY {repo_dir}/web_navigation_environment.yml .",
            """
                    RUN conda update -n base -c conda-forge conda -y && \
                      conda env create -f /workdir/web_navigation_environment.yml --name py310 -y
                    """,
            f"COPY {repo_dir} .",
            f"""
            RUN chown -R {uid}:root /workdir && \
             /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                conda activate py310 && \
                pip install -e /workdir[all] seaborn matplotlib chromedriver-py==$CHROMEDRIVER_VERSION minari==0.4.3 && \
                python /workdir/setup.py install && \
                pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"
            """,
            f"RUN mkdir -p /var/run/dbus && chown -R {uid}:root /var/run/dbus",
            "ENV CONDA_DEFAULT_ENV=py310",
        ]
    elif env_name in ENV_NAMES[BenchmarkDomain.CIRCUIT_TRAINING]:
        return common_setup + [
            """
                    RUN ${APT_COMMAND} update --allow-releaseinfo-change && \
                      ${APT_COMMAND} install flex \
                      libcairo2-dev \
                      libboost-all-dev && \
                      rm -rf /var/lib/apt/lists/*
                    """,
            "RUN mkdir -p /workdir",
            "WORKDIR /workdir",
            f"COPY {repo_dir}/circuit_training_environment.yml .",
            """
                    RUN conda update -n base -c conda-forge conda -y && \
                      conda env create -f /workdir/circuit_training_environment.yml --name py310 -y
                    """,
            f"COPY {repo_dir} .",
            """
                    RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                        conda activate py310 && \
                        pip install -e /workdir[all] seaborn codecarbon matplotlib minari==0.4.3 && \
                        python /workdir/setup.py install && \
                        pip uninstall -y nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11"
                    """,
            "ENV CONDA_DEFAULT_ENV=py310",
        ]

    else:
        raise ValueError(f"Unsupported environment: {env_name}")
