#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

SEED=0
ENV_BATCH_SIZE=1
TOTAL_ENV_STEPS=200000000
ROOT_DIR="../logs/quadruped_locomotion"
GIN_CONFIG=""
DIFFICULTY_LEVEL=-1
PARTICIPANT_MODULE_PATH=""
QUAD_LOCO_DIR=""
DOCKER_IMAGE_NAME="rlperf/quadruped_locomotion:latest"
DOCKER_CONTAINER_NAME="quadruped_locomotion_container"
DOCKERFILE_PATH="/home/jasonlinux22/Jason/A2Perf/rl-perf/rl_perf/domains/quadruped_locomotion/docker/Dockerfile"
REQUIREMENTS_PATH="./requirements.txt"
RUN_OFFLINE_METRICS_ONLY=false
PARALLEL_MODE=true
PARALLEL_CORES=1
MODE='train'
VISUALIZE=true
INT_SAVE_FREQ=10000000
SETUP_PATH='setup_model_env.py'

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

SSH_KEY_PATH=$QUAD_LOCO_DIR/.ssh/id_rsa

echo "Env Batch Size: $ENV_BATCH_SIZE"
echo "Difficulty Level: $DIFFICULTY_LEVEL"
echo "Seed value: $SEED"
echo "Root directory: $ROOT_DIR"
echo "Gin config: $GIN_CONFIG"
echo "Participant module path: $PARTICIPANT_MODULE_PATH"
echo "Web Nav directory: $WEB_NAV_DIR"
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
mkdir -p "$QUAD_LOCO_DIR/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "web_nav" -f "$SSH_KEY_PATH" -N ""

# --build-arg WEB_NAV_DIR="$WEB_NAV_DIR" \

# install xhost
sudo apt-get install x11-xserver-utils

# Build the Docker image
docker build --rm --build-arg REQUIREMENTS_PATH="$REQUIREMENTS_PATH" \
  -f "$DOCKERFILE_PATH" \
  -t "$DOCKER_IMAGE_NAME" rl_perf/domains/quadruped_locomotion

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  # if it is running, do nothing
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  xhost + local: # Allow docker to access the display
  docker run -itd \
    --rm \
    --name "$DOCKER_CONTAINER_NAME" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    -v "${HOME}/.Xauthority:/user/.Xauthority:rw" \
    -v /dev/shm:/dev/shm \
    --privileged \
    -v "$(pwd)":/rl-perf \
    -p 2022:22 \
    "$DOCKER_IMAGE_NAME"
fi
# --gpus=all \

# Install packages inside the container
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf
pip install -r requirements.txt
pip install -e .
EOF

# pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt

# Run the benchmarking code
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
export SEED=$SEED
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export ROOT_DIR=$ROOT_DIR
export TRAIN_LOGS_DIR=$TRAIN_LOGS_DIR
export DIFFICULTY_LEVEL=$DIFFICULTY_LEVEL
export PARALLEL_MODE="$PARALLEL_MODE"
export PARALLEL_CORES="$PARALLEL_CORES"
export MODE="$MODE"
export VISUALIZE="$VISUALIZE"
export INT_SAVE_FREQ="$INT_SAVE_FREQ"
export SETUP_PATH="$SETUP_PATH"

cd /rl-perf/rl_perf/submission
export DISPLAY=:0
python main_submission.py \
  --gin_file=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY
EOF
