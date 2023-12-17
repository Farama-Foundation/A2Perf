#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

SEED=0
ENV_BATCH_SIZE=16 # Adjusted default value
BATCH_SIZE=0
TOTAL_ENV_STEPS=1000000
ROOT_DIR=../logs/quadruped_locomotion
GIN_CONFIG=""
DIFFICULTY_LEVEL=-1
TIMESTEPS_PER_ACTORBATCH=0
PARTICIPANT_MODULE_PATH=""
WEB_NAV_DIR="$(pwd)/a2perf/domains/web_nav"
DOCKER_IMAGE_NAME="rlperf/web_nav:latest"
DOCKER_CONTAINER_NAME="web_nav_container"
DOCKERFILE_PATH="$(pwd)/a2perf/domains/web_navigation/docker/Dockerfile"
REQUIREMENTS_PATH="./requirements.txt"
RUN_OFFLINE_METRICS_ONLY=""
LOG_INTERVAL=0
WORK_UNIT_ID=0
LEARNING_RATE=0.001               # Default value, adjust if needed
EVAL_INTERVAL=50000               # Adjusted default value
TRAIN_CHECKPOINT_INTERVAL=100000  # Default value, adjust if needed
POLICY_CHECKPOINT_INTERVAL=100000 # Default value, adjust if needed
RB_CAPACITY=100000                # Default replay buffer capacity
SUMMARY_INTERVAL=10000            # Default value, adjust if needed
ALGORITHM=""
DEBUG=""

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --algo=*)
    ALGORITHM="${arg#*=}"
    shift
    ;;
  --debug=*)
    DEBUG="${arg#*=}"
    shift
    ;;
  --run_offline_metrics_only=*)
    RUN_OFFLINE_METRICS_ONLY="${arg#*=}"
    ;;
  --summary_interval=*)
    SUMMARY_INTERVAL="${arg#*=}"
    shift
    ;;
  --difficulty_level=*)
    DIFFICULTY_LEVEL="${arg#*=}"
    shift
    ;;
  --timesteps_per_actorbatch=*)
    TIMESTEPS_PER_ACTORBATCH="${arg#*=}"
    shift
    ;;
  --work_unit_id=*)
    WORK_UNIT_ID="${arg#*=}"
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
  --batch_size=*)
    BATCH_SIZE="${arg#*=}"
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
  --web_nav_dir=*)
    WEB_NAV_DIR="${arg#*=}"
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
  --learning_rate=*)
    LEARNING_RATE="${arg#*=}"
    shift
    ;;
  --eval_interval=*)
    EVAL_INTERVAL="${arg#*=}"
    shift
    ;;
  --train_checkpoint_interval=*)
    TRAIN_CHECKPOINT_INTERVAL="${arg#*=}"
    shift
    ;;
  --log_interval=*)
    LOG_INTERVAL="${arg#*=}"
    shift
    ;;
  --policy_checkpoint_interval=*)
    POLICY_CHECKPOINT_INTERVAL="${arg#*=}"
    shift
    ;;
  --rb_capacity=*)
    RB_CAPACITY="${arg#*=}"
    shift
    ;;
  --rb_checkpoint_interval=*)
    RB_CHECKPOINT_INTERVAL="${arg#*=}"
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

SSH_KEY_PATH=$WEB_NAV_DIR/docker/.ssh/id_rsa

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

# create ssh-key in WEB_NAV_DIR without password
mkdir -p "$WEB_NAV_DIR/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "web_nav" -f "$SSH_KEY_PATH" -N ""

# Build the Docker image
docker build --rm --network=host \
  --build-arg REQUIREMENTS_PATH="$REQUIREMENTS_PATH" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg USER_GROUP_ID="$(id -g)" \
  -t "$DOCKER_IMAGE_NAME" \
  --build-arg WEB_NAV_DIR="$WEB_NAV_DIR" \
  -f "$DOCKERFILE_PATH" \
  -t "$DOCKER_IMAGE_NAME" a2perf/domains/web_navigation/docker

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  docker_run_command="docker run -itd --rm --privileged --network=host"

  # check to see if /sys/class/powercap exists. if so, mount it
  if [ -d "/sys/class/powercap" ]; then
    docker_run_command+=" -v /sys/class/powercap:/sys/class/powercap"
  else
    echo "No powercap directory found. Will not mount it."
  fi

  # check for GPU and add the necessary flag if found
  if command -v nvidia-smi &>/dev/null; then
    docker_run_command+=" --gpus all"
  fi

  # append the rest of the flags
  docker_run_command+=" -v $(pwd):/rl-perf"
  docker_run_command+=" -v /home/ikechukwuu/workspace/gcs:/mnt/gcs/"
  docker_run_command+=" -v /dev/shm:/dev/shm"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""
  docker_run_command+=" --privileged"

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi

cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf
pip install -U -r requirements.txt

if [ "$DEBUG" = "true" ]; then
  pip install -r /rl-perf/a2perf/a2perf_benchmark_submission/web_navigation/${ALGORITHM}/debug/requirements.txt
else
  pip install -r /rl-perf/a2perf/a2perf_benchmark_submission/web_navigation/${ALGORITHM}/requirements.txt
fi
EOF

exit 0
# Run the benchmarking code
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash

export TF_FORCE_GPU_ALLOW_GROWTH=true

export SEED=$SEED
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export ROOT_DIR=$ROOT_DIR
export TRAIN_LOGS_DIRS=$TRAIN_LOGS_DIRS
export DIFFICULTY_LEVEL=$DIFFICULTY_LEVEL
export LEARNING_RATE=$LEARNING_RATE
export EVAL_INTERVAL=$EVAL_INTERVAL
export TRAIN_CHECKPOINT_INTERVAL=$TRAIN_CHECKPOINT_INTERVAL
export POLICY_CHECKPOINT_INTERVAL=$POLICY_CHECKPOINT_INTERVAL
export RB_CAPACITY=$RB_CAPACITY
export SUMMARY_INTERVAL=$SUMMARY_INTERVAL
export BATCH_SIZE=$BATCH_SIZE
export LOG_INTERVAL=$LOG_INTERVAL
export TIMESTEPS_PER_ACTORBATCH=$TIMESTEPS_PER_ACTORBATCH
export RB_CHECKPOINT_INTERVAL=$RB_CHECKPOINT_INTERVAL
cd /rl-perf/a2perf/submission
export DISPLAY=:0
python3 main_submission.py \
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY \
  --verbosity=1
EOF
