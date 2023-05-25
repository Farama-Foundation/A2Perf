#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

CT_VERSION=0.0.3
PYTHON_VERSION=python3.9
DREAMPLACE_PATTERN=dreamplace_20230414_2835324_${PYTHON_VERSION}.tar.gz
TF_AGENTS_PIP_VERSION="tf-agents[reverb]"

SEED=0
ROOT_DIR=../logs/web_nav
GIN_CONFIG=""
PARTICIPANT_MODULE_PATH=""

# set circuit training dir to be relative path to circuit training directory
CIRCUIT_TRAINING_DIR=../../rl_perf/domains/circuit_training
DOCKER_IMAGE_NAME="circuit_training:core"
DOCKER_CONTAINER_NAME="circuit_training_container"
DOCKERFILE_PATH="/home/ike2030/workspace/rl-perf/rl_perf/domains/circuit_training/tools/docker/ubuntu_circuit_training"
REQUIREMENTS_PATH="./requirements.txt"
RUN_OFFLINE_METRICS_ONLY=""

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
  --train_logs_dir=*)
    TRAIN_LOGS_DIR="${arg#*=}"
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
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

SSH_KEY_PATH=$CIRCUIT_TRAINING_DIR/.ssh/id_rsa

echo "Env Batch Size: $ENV_BATCH_SIZE"
echo "Difficulty Level: $DIFFICULTY_LEVEL"
echo "Seed value: $SEED"
echo "Root directory: $ROOT_DIR"
echo "Gin config: $GIN_CONFIG"
echo "Participant module path: $PARTICIPANT_MODULE_PATH"
echo "Circuit training directory: $CIRCUIT_TRAINING_DIR"
echo "Docker image name: $DOCKER_IMAGE_NAME"
echo "Docker container name: $DOCKER_CONTAINER_NAME"
echo "Dockerfile path: $DOCKERFILE_PATH"
echo "SSH key path: $SSH_KEY_PATH"
echo "Requirements path: $REQUIREMENTS_PATH"

# create ssh-key in CIRCUIT_TRAINING_DIR without password
mkdir -p "$CIRCUIT_TRAINING_DIR/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "circuit_training" -f "$SSH_KEY_PATH" -N ""

echo "Successfully parsed command-line arguments."

docker build --rm \
  --pull \
  --build-arg base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
  --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
  --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
  --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
  -f "${DOCKERFILE_PATH}" \
  -t "$DOCKER_IMAGE_NAME" rl_perf/domains/circuit_training

echo "Successfully built docker image."

mkdir -p "${CIRCUIT_TRAINING_DIR}"/logs
docker run --rm \
  --gpus=all \
  -p 2022:22 \
  -v "${CIRCUIT_TRAINING_DIR}":/workspace \
  --workdir /workspace \
  --name "$DOCKER_CONTAINER_NAME" \
  "$DOCKER_IMAGE_NAME" bash tools/e2e_smoke_test.sh --root_dir /workspace/logs
