#!/bin/bash
set -e

GITHUB_TOKEN=$1

# Removing directory and setting password
sudo rm -rf ~/workspace/rl-perf &&
  echo -e "password\npassword" | sudo passwd ikechukwuu &&
  echo "ikechukwuu ALL=(ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers

# Updating and installing required packages
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get -yq update &&
  sudo -E apt-get -yq upgrade &&
  sudo -E apt-get install -yq nfs-common telnet htop openssh-server ssh software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev git expect ca-certificates curl gnupg

# Installing Python 3.11
sudo -E add-apt-repository -y ppa:deadsnakes/ppa &&
  sudo -E apt-get -yq update &&
  sudo -E apt-get install -yq python3.11 python3.11-dev python3.11-venv

# Cloning dotfiles and running setup.sh
git clone https://uchendui:${GITHUB_TOKEN}@github.com/uchendui/dotfiles.git /home/ikechukwuu/workspace/dotfiles &&
  yes | bash /home/ikechukwuu/workspace/dotfiles/setup.sh

# Installing Docker
sudo -E apt-get update &&
  sudo -E install -m 0755 -d /etc/apt/keyrings &&
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -E gpg --dearmor -o /etc/apt/keyrings/docker.gpg &&
  sudo -E chmod a+r /etc/apt/keyrings/docker.gpg &&
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null &&
  sudo -E apt-get update &&
  sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to Docker group
sudo usermod -aG docker ikechukwuu

# Mounting the filestore
mkdir -p ~/workspace/gcs &&
  sudo mount -t nfs -o rw,hard,timeo=600,retrans=3,rsize=524288,wsize=2097152,nconnect=16,tcp 10.69.127.130:/a2perf /home/ikechukwuu/workspace/gcs

# Cloning the rl-perf repository
git clone https://uchendui:${GITHUB_TOKEN}@github.com/harvard-edge/rl-perf.git /home/ikechukwuu/workspace/rl-perf &&
  cd /home/ikechukwuu/workspace/rl-perf/ &&
  git checkout quad_integration &&
  git config --global credential.helper cache &&
  git config --global credential.helper 'cache --timeout=3600' &&
  echo -e "uchendui\n${GITHUB_TOKEN}" | git submodule update --init --recursive

# Setting up Python virtual environment
cd ~/workspace/rl-perf/ &&
  python3.11 -m venv env &&
  source env/bin/activate &&
  pip install --upgrade pip &&
  pip install -r launch_requirements.txt
