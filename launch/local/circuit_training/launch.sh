#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

# These arguments most likely do not need to chagne
CT_VERSION=0.0.3
PYTHON_VERSION=python3.9
TF_AGENTS_PIP_VERSION="tf-agents[reverb]"
DM_REVERB_PIP_VERSION=""
CIRCUIT_TRAINING_DIR=../../rl_perf/domains/circuit_training
DOCKER_IMAGE_NAME="rlperf/circuit_training"
DOCKER_CONTAINER_NAME="circuit_training_container"
DOCKERFILE_PATH="./rl_perf/domains/circuit_training/tools/docker/ubuntu_circuit_training"
REQUIREMENTS_PATH="./requirements.txt"

# New Environment Variables
SEED=0
ROOT_DIR=../logs/web_nav
GIN_CONFIG=""
PARTICIPANT_MODULE_PATH=""
REVERB_PORT=8000
REVERB_SERVER_IP=""
NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc
RUN_OFFLINE_METRICS_ONLY=""
NUM_COLLECT_JOBS=0

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --run_offline_metrics_only=*)
    RUN_OFFLINE_METRICS_ONLY="${arg#*=}"
    shift
    ;;
  --root_dir=*)
    ROOT_DIR="${arg#*=}"
    shift
    ;;
  --train_logs_dirs=*)
    TRAIN_LOGS_DIRS="${arg#*=}"
    shift
    ;;
  --gin_config=*)
    GIN_CONFIG="${arg#*=}"
    shift
    ;;
  --participant_module_path=*)
    PARTICIPANT_MODULE_PATH="${arg#*=}"
    shift
    ;;
  --circuit_training_dir=*)
    CIRCUIT_TRAINING_DIR="${arg#*=}"
    shift
    ;;
  --docker_image_name=*)
    DOCKER_IMAGE_NAME="${arg#*=}"
    shift
    ;;
  --docker_container_name=*)
    DOCKER_CONTAINER_NAME="${arg#*=}"
    shift
    ;;
  --ssh_key_path=*)
    SSH_KEY_PATH="${arg#*=}"
    shift
    ;;
  --requirements_path=*)
    REQUIREMENTS_PATH="${arg#*=}"
    shift
    ;;
  --dockerfile_path=*)
    DOCKERFILE_PATH="${arg#*=}"
    shift
    ;;
  --reverb_port=*)
    REVERB_PORT="${arg#*=}"
    shift
    ;;
  --reverb_server_ip=*)
    REVERB_SERVER_IP="${arg#*=}"
    shift
    ;;
  --netlist_file=*)
    NETLIST_FILE="${arg#*=}"
    shift
    ;;
  --init_placement=*)
    INIT_PLACEMENT="${arg#*=}"
    shift
    ;;
  --num_collect_jobs=*)
    NUM_COLLECT_JOBS="${arg#*=}"
    shift
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

SSH_KEY_PATH=$CIRCUIT_TRAINING_DIR/tools/docker/.ssh/id_rsa
echo "CT_VERSION: $CT_VERSION"
echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "TF_AGENTS_PIP_VERSION: $TF_AGENTS_PIP_VERSION"
#echo "TENSORFLOW_PROBABILITY_PIP_VERSION: $TENSORFLOW_PROBABILITY_PIP_VERSION"
#echo "TF_AGENTS_PIP_VERSION: $TF_AGENTS_PIP_VERSION"
#echo "TENSORFLOW_PROBABILITY_PIP_VERSION: $TFP_NIGHTLY"
echo "DM_REVERB_PIP_VERSION: $DM_REVERB_PIP_VERSION"
echo "CIRCUIT_TRAINING_DIR: $CIRCUIT_TRAINING_DIR"
echo "DOCKER_IMAGE_NAME: $DOCKER_IMAGE_NAME"
echo "DOCKER_CONTAINER_NAME: $DOCKER_CONTAINER_NAME"
echo "DOCKERFILE_PATH: $DOCKERFILE_PATH"
echo "REQUIREMENTS_PATH: $REQUIREMENTS_PATH"
echo "SEED: $SEED"
echo "ROOT_DIR: $ROOT_DIR"
echo "GIN_CONFIG: $GIN_CONFIG"
echo "PARTICIPANT_MODULE_PATH: $PARTICIPANT_MODULE_PATH"
echo "REVERB_PORT: $REVERB_PORT"
echo "REVERB_SERVER_IP: $REVERB_SERVER_IP"
echo "NETLIST_FILE: $NETLIST_FILE"
echo "INIT_PLACEMENT: $INIT_PLACEMENT"
echo "RUN_OFFLINE_METRICS_ONLY: $RUN_OFFLINE_METRICS_ONLY"
echo "NUM_COLLECT_JOBS: $NUM_COLLECT_JOBS"
#echo "TRAIN_LOGS_DIRS: $TRAIN_LOGS_DIRS"
#echo "SSH_KEY_PATH: $SSH_KEY_PATH"

# create ssh-key in CIRCUIT_TRAINING_DIR without password
mkdir -p "$CIRCUIT_TRAINING_DIR/tools/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "circuit_training" -f "$SSH_KEY_PATH" -N ""

echo "Successfully parsed command-line arguments."

docker build \
  --rm \
  --pull \
  --build-arg base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
  -f "${DOCKERFILE_PATH}" \
  -t "$DOCKER_IMAGE_NAME" rl_perf/domains/circuit_training/tools/docker

echo "Successfully built docker image."

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  docker run -itd \
    --rm \
    -p 2022:22 \
    --gpus all \
    -v "$(pwd)":/rl-perf \
    -v /sys/class/powercap:/sys/class/powercap \
    --workdir /rl-perf \
    --name "$DOCKER_CONTAINER_NAME" \
    "$DOCKER_IMAGE_NAME"
fi

exit 0
#
# Install required packages inside the container
docker exec --interactive "$DOCKER_CONTAINER_NAME" bash <<EOF
# Install requirements for the rl-perf repo
pip install  --no-cache-dir -r "$REQUIREMENTS_PATH"

# Install RLPerf as a package
pip install --no-cache-dir -e .

export PYTHONPATH=/rl-perf:\$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
#export TF_GPU_ALLOCATOR=cuda_malloc_async
export ROOT_DIR=$ROOT_DIR
export GLOBAL_SEED=$SEED
export REVERB_PORT=$REVERB_PORT
export REVERB_SERVER_IP=$REVERB_SERVER_IP
export NETLIST_FILE=$NETLIST_FILE
export INIT_PLACEMENT=$INIT_PLACEMENT
export NUM_COLLECT_JOBS=$NUM_COLLECT_JOBS
export WRAPT_DISABLE_EXTENSIONS=true

$PYTHON_VERSION rl_perf/submission/main_submission.py \
  --gin_file=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY
EOF

exit 0

# Run these commands for the "smoke-test"
docker run \
  -it \
  --rm \
  --gpus=all \
  -v /home/ike2030/workspace/rl-perf:/rl-perf \
  --workdir /rl-perf/rl_perf/domains/circuit_training \
  circuit_training:core

docker run \
  -it \
  --rm \
  -v "$(pwd)":/rl-perf \
  --workdir /rl-perf/rl_perf/domains/circuit_training \
  circuit_training:core
pip install -e ../../../
mkdir -p ./logs
cd ../../../
rl_perf/domains/circuit_training/tools/e2e_smoke_test.sh --root_dir ./logs
# tools/e2e_smoke_test.sh --root_dir ./logs

exit 0

# Run these commands for another unit test
docker run \
  -it \
  --rm \
  -v /home/ike2030/workspace/rl-perf:/rl-perf \
  --workdir /rl-perf/rl_perf/domains/circuit_training \
  circuit_training:core

cd /rl-perf/rl_perf/domains/circuit_training/ || exit
tox -e py39-stable -- circuit_training/grouping/grouping_test.py

cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf

# Install requirements for the rl-perf repo
pip install --no-deps --ignore-installed -r $REQUIREMENTS_PATH

# Install RLPerf as a package
pip install --no-deps --ignore-installed -e .

# Install packages specific to the user's training code
EOF

export PYTHONPATH=/rl-perf:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export WRAPT_DISABLE_EXTENSIONS=true
export ROOT_DIR=--root_dir=/rl-perf/logs/circuit_training/debug
export GLOBAL_SEED=0
export REVERB_PORT=8000
export REVERB_SERVER_IP=127.0.0.1
export NETLIST_FILE=/rl-perf/rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc
export INIT_PLACEMENT=/rl-perf/rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt
export NUM_COLLECT_JOBS=4
