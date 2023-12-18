#!/bin/bash
set -e

sudo rm -rf ~/workspace/rl-perf

DEBIAN_FRONTEND=noninteractive sudo apt-get -yq update &&
  sudo apt-get -yq upgrade &&
  sudo apt-get install -yq \
    nfs-common \
    telnet \
    software-properties-common \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    wget \
    libbz2-dev \
    git \
    expect \
    tmux

# Download, extract, and install Python 3.10.12
cd ~ && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz &&
  tar -xf Python-3.10.*.tgz && cd Python-3.10.*/ && ./configure \
  --prefix=/usr/local \
  --enable-optimizations \
  --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib" && make -j $(nproc) && sudo make altinstall

git config --global credential.helper 'store'

git clone https://uchendui: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@github.com/uchendui/dotfiles.git /home/ikechukwuu/workspace/dotfiles

cd ~/workspace/dotfiles && {
  echo "Y"
  sleep 10
  echo -e "\004"
} | bash setup.sh

git clone https://uchendui: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@github.com/harvard-edge/rl-perf.git /home/ikechukwuu/workspace/rl-perf

cd ~/workspace/rl-perf/ &&
  git checkout ct_integration &&
  git submodule update --init --recursive

# Create a Python 3.10 virtual environment
cd ~/workspace/rl-perf/ &&
  python3.10 -m venv env &&
  source env/bin/activate &&
  pip install --upgrade pip -r launch_requirements.txt

# Mount the filestore
mkdir -p ~/workspace/gcs && sudo mount -t nfs -o hard,timeo=600,retrans=3,rsize=262144,wsize=1048576,resvport,async,tcp 10.69.127.130:/a2perf /home/ikechukwuu/workspace/gcs && sudo chown -R ikechukwuu:docker /home/ikechukwuu/workspace/gcs
