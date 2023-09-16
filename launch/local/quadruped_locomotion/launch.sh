#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

SEED=0
ENV_BATCH_SIZE=1
TOTAL_ENV_STEPS=200000000
ROOT_DIR="/tmp/locomotion"
GIN_CONFIG=""
PARTICIPANT_MODULE_PATH=""
QUAD_LOCO_DIR="$(pwd)/rl_perf/domains/quadruped_locomotion"
DOCKER_IMAGE_NAME="rlperf/quadruped_locomotion:latest"
DOCKER_CONTAINER_NAME="quadruped_locomotion_container"
DOCKERFILE_PATH="$(pwd)/rl_perf/domains/quadruped_locomotion/docker/Dockerfile"
REQUIREMENTS_PATH="./requirements.txt"
RUN_OFFLINE_METRICS_ONLY=false
PARALLEL_MODE=true
PARALLEL_CORES=0
MODE='train'
VISUALIZE=false
MODEL_FILE_PATH=""
#INT_SAVE_FREQ=10000000
INT_SAVE_FREQ=100000
EXTRA_GIN_BINDINGS='--extra_gin_bindings="track_emissions.default_cpu_tdp=240"'

#INT_SAVE_FREQ=10
SETUP_PATH='setup_model_env.py'

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --model_file_path=*)
    MODEL_FILE_PATH="${arg#*=}"
    shift
    ;;
  --run_offline_metrics_only=*)
    RUN_OFFLINE_METRICS_ONLY="${arg#*=}"
    shift
    ;;
  --difficulty_level=*)
    DIFFICULTY_LEVEL="${arg#*=}"
    shift
    ;;
  --env_batch_size=*)
    ENV_BATCH_SIZE="${arg#*=}"
    shift
    ;;
  --total_env_steps=*)
    TOTAL_ENV_STEPS="${arg#*=}"
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
  --quad_loco_dir=*)
    QUAD_LOCO_DIR="${arg#*=}"
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
  --parallel_mode=*)
    PARALLEL_MODE="${arg#*=}"
    shift
    ;;
  --parallel_cores=*)
    PARALLEL_CORES="${arg#*=}"
    shift
    ;;
  --mode=*)
    MODE="${arg#*=}"
    shift
    ;;
  --visualize=*)
    VISUALIZE="${arg#*=}"
    shift
    ;;
  --int_save_freq=*)
    INT_SAVE_FREQ="${arg#*=}"
    shift
    ;;
  --setup_path=*)
    SETUP_PATH="${arg#*=}"
    shift
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

SSH_KEY_PATH=$QUAD_LOCO_DIR/docker/.ssh/id_rsa

echo "Env Batch Size: $ENV_BATCH_SIZE"
echo "Difficulty Level: $DIFFICULTY_LEVEL"
echo "Seed value: $SEED"
echo "Root directory: $ROOT_DIR"
echo "Gin config: $GIN_CONFIG"
echo "Participant module path: $PARTICIPANT_MODULE_PATH"
echo "Quadruped locomotion directory: $QUAD_LOCO_DIR"
echo "Docker image name: $DOCKER_IMAGE_NAME"
echo "Docker container name: $DOCKER_CONTAINER_NAME"
echo "Dockerfile path: $DOCKERFILE_PATH"
echo "SSH key path: $SSH_KEY_PATH"
echo "Requirements path: $REQUIREMENTS_PATH"
echo "Parallel Mode: $PARALLEL_MODE"
echo "Parallel Cores: $PARALLEL_CORES"
echo "Mode: $MODE"
echo "Visualize: $VISUALIZE"
echo "Int Save Freq: $INT_SAVE_FREQ"
echo "Setup Path: $SETUP_PATH"

# create ssh-key in WEB_NAV_DIR without password
mkdir -p "$QUAD_LOCO_DIR/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "quadruped_locomotion" -f "$SSH_KEY_PATH" -N ""

# install xhost
sudo apt-get install x11-xserver-utils

docker build \
  --rm \
  --pull \
  -f "${DOCKERFILE_PATH}" \
  --build-arg REQUIREMENTS_PATH="$REQUIREMENTS_PATH" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg USER_GROUP_ID="$(id -g)" \
  -t "$DOCKER_IMAGE_NAME" \
  rl_perf/domains/quadruped_locomotion/docker

echo "Successfully built docker image."
#exit 0

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  # initial command
  docker_run_command="docker run -itd --rm -p 2020:22 --privileged"
  #  docker_run_command="docker run -itd --rm  --privileged"

  # check to see if /sys/class/powercap exists. if so, mount it
  if [ -d "/sys/class/powercap" ]; then
    docker_run_command+=" -v /sys/class/powercap:/sys/class/powercap"
  else
    echo "No powercap directory found. Will not mount it."
  fi

  # check for GPU and add the necessary flag if found
  if command -v nvidia-smi &>/dev/null; then
    docker_run_command+=" --gpus all"
    # give two GPUs depending on the docker_container_name last digit
    #    docker_run_command+=" --gpus \"device=$WORK_UNIT_ID\""
  fi

  # append the rest of the flags
  docker_run_command+=" -v $(pwd):/rl-perf"
  docker_run_command+=" -v /dev/shm:/dev/shm"
  docker_run_command+=" -v /home/ikechukwuu/workspace/gcs:/mnt/gcs/"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi

#exit 0
# Install packages inside the container
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf
pip install -r requirements.txt
pip install -e .
EOF

# pip install -r rl_perf/rlperf_benchmark_submission/quadruped_locomotion/requirements.txt

# Run the benchmarking code
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
export SEED=$SEED
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export ROOT_DIR=$ROOT_DIR
export TRAIN_LOGS_DIRS=$TRAIN_LOGS_DIRS
export PARALLEL_MODE="$PARALLEL_MODE"
export PARALLEL_CORES="$PARALLEL_CORES"
export MODE="$MODE"
export MODEL_FILE_PATH="$MODEL_FILE_PATH"
export VISUALIZE="$VISUALIZE"
export INT_SAVE_FREQ="$INT_SAVE_FREQ"
export SETUP_PATH="$SETUP_PATH"

cd /rl-perf/rl_perf/submission
export DISPLAY=:0
python3.7 -u main_submission.py \
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY \
  $EXTRA_GIN_BINDINGS
EOF
