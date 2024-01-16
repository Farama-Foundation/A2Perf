# Base Image
FROM gcr.io/deeplearning-platform-release/base-gpu:latest


# Arguments and Environment Variables
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 --no-install-recommends -y"
ENV DEBIAN_FRONTEND=noninteractive
ARG REPO_DIR="."

# User Setup
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN echo "clouduser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Conda Environment Setup
RUN conda create -y --name py39 python=3.9
ENV CONDA_DEFAULT_ENV=py39
ENV PATH="/opt/conda/envs/py39/bin:${PATH}"
RUN /opt/conda/envs/py39/bin/pip install --upgrade pip setuptools
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
      conda activate py39 && \
      conda install cuda -c  nvidia -y"

# Set Working Directory
WORKDIR /workdir

# Copying Requirements
COPY ${REPO_DIR}/a2perf/metrics/reliability/requirements.txt ./a2perf/metrics/reliability/requirements.txt
COPY ${REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt ./a2perf/metrics/system/codecarbon/
COPY ${REPO_DIR}/a2perf/domains/quadruped_locomotion/requirements.txt ./a2perf/domains/quadruped_locomotion/requirements.txt
COPY ${REPO_DIR}/a2perf/a2perf_benchmark_submission/requirements.txt ./a2perf/a2perf_benchmark_submission/requirements.txt
COPY ${REPO_DIR}/requirements.txt ./requirements.txt

# Installing Requirements
RUN /opt/conda/envs/py39/bin/pip install -r ./requirements.txt
RUN /opt/conda/envs/py39/bin/pip install -r ./a2perf/domains/quadruped_locomotion/requirements.txt
RUN /opt/conda/envs/py39/bin/pip install -r ./a2perf/a2perf_benchmark_submission/requirements.txt

# Copy the Repository
COPY ${REPO_DIR} .

# Set Permissions and Install
RUN chmod -R 777 /workdir/a2perf /workdir/setup.py
RUN /opt/conda/envs/py39/bin/pip install /workdir

# Change to user 1000
USER 1000

ENTRYPOINT ["/opt/conda/envs/py39/bin/python /workdir/launch/entrypoint.py"]
