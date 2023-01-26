#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sudo apt -y update &&
  sudo apt-get -y update &&
  sudo apt -y upgrade &&
  sudo apt-get -y upgrade

sudo apt-get install -y git expect tmux
git config --global credential.helper 'cache --timeout=43200'
mkdir ~/workspace

USERNAME="uchendui"
TOKEN="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" # replace with your actual token

expect <<EOF
set timeout 20
spawn git clone https://github.com/uchendui/dotfiles.git /home/ikechukwuu/workspace/dotfiles
expect "Username for 'https://github.com':"
send "$USERNAME\r"
expect "Password for 'https://$USERNAME@github.com':"
send "$TOKEN\r"
expect eof
EOF

cd ~/workspace/dotfiles && {
  echo "Y"
  sleep 10
  echo -e "\004"
} | bash setup.sh

# Install dependencies
sudo apt install -y \
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
  libbz2-dev

# Download, extract, and install Python 3.10.12
cd ~ && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz &&
  tar -xf Python-3.10.*.tgz
cd Python-3.10.*/ && ./configure \
  --prefix=/usr/local \
  --enable-optimizations \
  --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"
make -j $(nproc)
sudo make altinstall

# Clone the rl-perf repository
expect <<EOF
set timeout 20
spawn git clone https://github.com/harvard-edge/rl-perf.git /home/ikechukwuu/workspace/rl-perf
expect "Username for 'https://github.com':"
send "$USERNAME\r"
expect "Password for 'https://$USERNAME@github.com':"
send "$TOKEN\r"
expect eof
EOF

cd ~/workspace/rl-perf/ &&
  git checkout ct_integration &&
  git submodule update --init --recursive

# Create a Python 3.10 virtual environment
cd ~/workspace/rl-perf/ &&
  python3.10 -m venv env &&
  source env/bin/activate &&
  pip install --upgrade pip -r launch_requirements.txt
