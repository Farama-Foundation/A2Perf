#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit


ALGORITHM=""
BATCH_SIZE=0
DATASET_ID=""
DEBUG=""
DIFFICULTY_LEVEL=-1
DOCKERFILE_PATH="$(pwd)/a2perf/domains/web_navigation/docker/Dockerfile"
DOCKER_CONTAINER_NAME="web_nav_container"
DOCKER_IMAGE_NAME="rlperf/web_nav:latest"
ENTROPY_REGULARIZATION=0.0
ENV_BATCH_SIZE=0
EPSILON_GREEDY=-1
EVAL_INTERVAL=50000
GIN_CONFIG=""
LEARNING_RATE=0.001
LOG_INTERVAL=0
NUM_WEBSITES=0
PARTICIPANT_MODULE_PATH=""
POLICY_CHECKPOINT_INTERVAL=100000
RB_CAPACITY=100000
REQUIREMENTS_PATH="./requirements.txt"
ROOT_DIR=../logs/quadruped_locomotion
RUN_OFFLINE_METRICS_ONLY=""
SEED=0
SKILL_LEVEL=0
SUMMARY_INTERVAL=10000
TIMESTEPS_PER_ACTORBATCH=0
TOTAL_ENV_STEPS=1000000
TRAIN_CHECKPOINT_INTERVAL=100000
WEB_NAV_DIR="$(pwd)/a2perf/domains/web_nav"

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --num_websites=*)
    NUM_WEBSITES="${arg#*=}"
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
  --entropy_regularization=*)
    ENTROPY_REGULARIZATION="${arg#*=}"
    shift
    ;;
  --skill_level=*)
    SKILL_LEVEL="${arg#*=}"
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
  --env_batch_size=*)
    ENV_BATCH_SIZE="${arg#*=}"
    shift
    ;;
  --epsilon_greedy=*)
    EPSILON_GREEDY="${arg#*=}"
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
  --dataset_id=*)
    DATASET_ID="${arg#*=}"
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
echo "Num Websites: $NUM_WEBSITES"
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


# Build the Docker image and let it use the display
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

  USER_HOME_DIR=$(eval echo "~$USER")

  # append the rest of the flags, along with display setup
  docker_run_command+=" -v $(pwd):/rl-perf"
  docker_run_command+=" -v ${USER_HOME_DIR}/workspace/gcs:/mnt/gcs/"
  docker_run_command+=" -v /dev/shm:/dev/shm"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""
  docker_run_command+=" -v /tmp/.X11-unix:/tmp/.X11-unix"
  docker_run_command+=" -e DISPLAY=$DISPLAY"
  docker_run_command+=" --privileged"

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi

cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf
pip  install -r requirements.txt

# Web Navigation specific requirements
pip install -r /rl-perf/a2perf/domains/web_navigation/requirements.txt

# Install the a2perf package
pip install -e /rl-perf

# Install the participant's package
pip  install -r /rl-perf/a2perf/a2perf_benchmark_submission/web_navigation/${ALGORITHM}/requirements.txt
EOF


verbosity_level=$( [ -z "$DEBUG" ] && echo "-2" || echo "2" )
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
export BATCH_SIZE=$BATCH_SIZE
export DATASET_ID=$DATASET_ID
export DIFFICULTY_LEVEL=$DIFFICULTY_LEVEL
export DISPLAY=$DISPLAY
export ENTROPY_REGULARIZATION=$ENTROPY_REGULARIZATION
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export EPSILON_GREEDY=$EPSILON_GREEDY
export EVAL_INTERVAL=$EVAL_INTERVAL
export LEARNING_RATE=$LEARNING_RATE
export LOG_INTERVAL=$LOG_INTERVAL
export NUM_WEBSITES=$NUM_WEBSITES
export POLICY_CHECKPOINT_INTERVAL=$POLICY_CHECKPOINT_INTERVAL
export RB_CAPACITY=$RB_CAPACITY
export RB_CHECKPOINT_INTERVAL=$RB_CHECKPOINT_INTERVAL
export ROOT_DIR=$ROOT_DIR
export SEED=$SEED
export SKILL_LEVEL=$SKILL_LEVEL
export SUMMARY_INTERVAL=$SUMMARY_INTERVAL
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TIMESTEPS_PER_ACTORBATCH=$TIMESTEPS_PER_ACTORBATCH
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export TRAIN_CHECKPOINT_INTERVAL=$TRAIN_CHECKPOINT_INTERVAL
export TRAIN_LOGS_DIRS=$TRAIN_LOGS_DIRS
export WRAPT_DISABLE_EXTENSIONS=true
cd /rl-perf/a2perf/submission

# Use verbosity 2 if we are debugging, otherwise verbosity -2
python3.10 main_submission.py \
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY \
  --verbosity=$verbosity_level
EOF
