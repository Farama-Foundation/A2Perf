# Base Image
FROM gcr.io/deeplearning-platform-release/base-gpu:latest


# Arguments and Environment Variables
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 --no-install-recommends -y"
ENV DEBIAN_FRONTEND=noninteractive
ARG REPO_DIR="."

# Set up clouduser
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN echo "clouduser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Chrome Installation
ARG CHROME_VERSION="114.0.5735.90-1"
ARG CHROMEDRIVER_VERSION="114.0.5735.90"

RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb && \
    ${APT_COMMAND} update && \
    ${APT_COMMAND} --fix-broken install && \
    ${APT_COMMAND} install /tmp/chrome.deb xvfb && \
    rm /tmp/chrome.deb

RUN TODAYS_DATE=$(date +%Y-%m-%d) && \
    wget --no-verbose -O /tmp/chromedriver_linux64.zip https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip && \
    unzip -o /tmp/chromedriver_linux64.zip -d /tmp/ && \
    mkdir -p /home/clouduser/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION} && \
    mv /tmp/chromedriver /home/clouduser/.wdm/drivers/chromedriver/linux64/${CHROMEDRIVER_VERSION}/ && \
    rm /tmp/chromedriver_linux64.zip && \
    printf '{"linux64_chromedriver_%s_for_%s": {"timestamp": "%s", "binary_path": "/home/clouduser/.wdm/drivers/chromedriver/linux64/%s/chromedriver"}}' "${CHROMEDRIVER_VERSION}" "${CHROME_VERSION}" "${TODAYS_DATE}" "${CHROMEDRIVER_VERSION}" > /home/clouduser/.wdm/drivers.json && \
    chmod -R 777 /home/clouduser/.wdm && cp -r /home/clouduser/.wdm /root/

# Set up custom conda environment
RUN conda create -y --name py310 python=3.10
ENV CONDA_DEFAULT_ENV=py310
ENV PATH="/opt/conda/envs/py310/bin:${PATH}"
RUN /opt/conda/envs/py310/bin/pip install --upgrade pip setuptools

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate py310 && \
    conda install cuda -c nvidia -y"



# Install sudo and xvfb
RUN ${APT_COMMAND} update && \
    ${APT_COMMAND} install sudo xvfb -y \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /workdir

# Copying Requirements
COPY ${REPO_DIR}/a2perf/metrics/reliability/requirements.txt \
    ./a2perf/metrics/reliability/requirements.txt
COPY ${REPO_DIR}/a2perf/metrics/system/codecarbon/requirements*.txt \
    ./a2perf/metrics/system/codecarbon/
COPY ${REPO_DIR}/a2perf/a2perf_benchmark_submission/requirements.txt \
    ./a2perf/a2perf_benchmark_submission/requirements.txt
COPY ${REPO_DIR}/a2perf/domains/web_navigation/requirements.txt \
    ./a2perf/domains/web/navigation/requirements.txt


# Installing Requirements
RUN /opt/conda/envs/py310/bin/pip install -r ./a2perf/metrics/reliability/requirements.txt
RUN /opt/conda/envs/py310/bin/pip install -r ./a2perf/metrics/system/codecarbon/requirements.txt
RUN /opt/conda/envs/py310/bin/pip install -r ./a2perf/a2perf_benchmark_submission/requirements.txt
RUN /opt/conda/envs/py310/bin/pip install -r ./a2perf/domains/web/navigation/requirements.txt

# Copy the Repository
COPY ${REPO_DIR} .

# Set Permissions and Install
RUN chmod -R 777 /workdir/a2perf /workdir/setup.py
RUN /opt/conda/envs/py310/bin/pip install /workdir

# Change to user 1000
USER 1000

ENTRYPOINT ["sudo service dbus start","sudo service xvfb start","/opt/conda/envs/py310/bin/python /workdir/launch/entrypoint.py"]
