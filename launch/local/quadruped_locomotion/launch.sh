#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

ALGORITHM=""
BATCH_SIZE=0
DATASET_ID=""
DEBUG=""
DOCKERFILE_PATH="$(pwd)/a2perf/domains/quadruped_locomotion/docker/Dockerfile"
DOCKER_CONTAINER_NAME="quadruped_locomotion_container"
DOCKER_IMAGE_NAME="rlperf/quadruped_locomotion:latest"
ENTROPY_REGULARIZATION=0.0
EXTRA_GIN_BINDINGS='--extra_gin_bindings="track_emissions.default_cpu_tdp=240"'
GIN_CONFIG=""
INT_EVAL_FREQ=0
INT_SAVE_FREQ=0
LEARNING_RATE=0.0
MODE=""
MOTION_FILE_PATH=""
NUM_EPOCHS=0
PARALLEL_CORES=0
PARALLEL_MODE=true
PARTICIPANT_MODULE_PATH=""
QUAD_LOCO_DIR="$(pwd)/a2perf/domains/quadruped_locomotion"
REQUIREMENTS_PATH="./requirements.txt"
ROOT_DIR="/tmp/locomotion"
RUN_OFFLINE_METRICS_ONLY=false
SEED=0
SETUP_PATH=""
SKILL_LEVEL=0
TOTAL_ENV_STEPS=0
VISUALIZE=false

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --motion_file_path=*)
    MOTION_FILE_PATH="${arg#*=}"
    shift
    ;;
  --run_offline_metrics_only=*)
    RUN_OFFLINE_METRICS_ONLY="${arg#*=}"
    shift
    ;;
  --learning_rate=*)
    LEARNING_RATE="${arg#*=}"
    shift
    ;;
  --algo=*)
    ALGORITHM="${arg#*=}"
    shift
    ;;
  --dataset_id=*)
    DATASET_ID="${arg#*=}"
    shift
    ;;
  --extra_gin_bindings=*)
    EXTRA_GIN_BINDINGS="${arg#*=}"
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
  --batch_size=*)
    BATCH_SIZE="${arg#*=}"
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
  --num_epochs=*)
    NUM_EPOCHS="${arg#*=}"
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
  --int_eval_freq=*)
    INT_EVAL_FREQ="${arg#*=}"
    shift
    ;;
  --setup_path=*)
    SETUP_PATH="${arg#*=}"
    shift
    ;;
  --debug=*)
    DEBUG="${arg#*=}"
    shift
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

echo $EXTRA_GIN_BINDINGS

# remove stray single quotes
EXTRA_GIN_BINDINGS=$(echo "$EXTRA_GIN_BINDINGS" | tr -d "'")
echo $EXTRA_GIN_BINDINGS

# Now split into array
IFS=',' read -ra EXTRA_GIN_BINDINGS_ARRAY <<<"$EXTRA_GIN_BINDINGS"

EXTRA_GIN_BINDINGS_ARG=""
for binding in "${EXTRA_GIN_BINDINGS_ARRAY[@]}"; do
  EXTRA_GIN_BINDINGS_ARG+="--extra_gin_bindings='$binding' "
done
echo $EXTRA_GIN_BINDINGS_ARG

SSH_KEY_PATH=$QUAD_LOCO_DIR/docker/.ssh/id_rsa
# change the setup path depending on the algorithm
if [ "$ALGORITHM" = "ppo" ]; then
  SETUP_PATH="ppo_actor.py"
elif [ "$ALGORITHM" = "ddpg" ]; then
  SETUP_PATH="ddpg_actor.py"
elif [ "$ALGORITHM" = "bc" ]; then
  SETUP_PATH=""
else
  echo "Invalid algorithm: $ALGORITHM"
  exit 1
fi

mkdir -p "$QUAD_LOCO_DIR/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "quadruped_locomotion" -f "$SSH_KEY_PATH" -N ""

# install xhost
sudo apt-get install x11-xserver-utils
xhost + local:
docker build --network=host \
  --rm \
  --pull \
  -f "${DOCKERFILE_PATH}" \
  --build-arg REQUIREMENTS_PATH="$REQUIREMENTS_PATH" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg USER_GROUP_ID="$(id -g)" \
  -t "$DOCKER_IMAGE_NAME" \
  a2perf/domains/quadruped_locomotion/docker

echo "Successfully built docker image."
#exit 0

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  # initial command
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

  # append the rest of the flags
  docker_run_command+=" -v $(pwd):/rl-perf"
  docker_run_command+=" -v ${USER_HOME_DIR}/workspace/gcs:/mnt/gcs/"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""
  docker_run_command+=" -e MINARI_DATASETS_PATH=$MINARI_DATASETS_PATH"

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi

cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf
pip install -r requirements.txt
pip install -e .

if [ "$DEBUG" = "true" ]; then
  pip install -r /rl-perf/a2perf/a2perf_benchmark_submission/quadruped_locomotion/${ALGORITHM}/debug/requirements.txt
else
  pip install -r /rl-perf/a2perf/a2perf_benchmark_submission/quadruped_locomotion/${ALGORITHM}/requirements.txt
fi

EOF

#exit 0
# Remove stray single quotes first
TRAIN_LOGS_DIRS=$(echo "$TRAIN_LOGS_DIRS" | tr -d "'")

# Now split into array
IFS=',' read -ra TRAIN_LOGS_ARRAY <<<"$TRAIN_LOGS_DIRS"

# Construct the train_logs_dirs arguments
TRAIN_LOGS_ARGS=""
for dir in "${TRAIN_LOGS_ARRAY[@]}"; do
  TRAIN_LOGS_ARGS+="--train_logs_dirs=$dir "
done

verbosity_level=$( [ -z "$DEBUG" ] && echo "-2" || echo "2" )
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
export BATCH_SIZE="$BATCH_SIZE"
export DATASET_ID="$DATASET_ID"
export ENTROPY_REGULARIZATION="$ENTROPY_REGULARIZATION"
export INT_EVAL_FREQ="$INT_EVAL_FREQ"
export INT_SAVE_FREQ="$INT_SAVE_FREQ"
export LEARNING_RATE="$LEARNING_RATE"
export MINARI_DATASETS_PATH="$MINARI_DATASETS_PATH"
export MODE="$MODE"
export MOTION_FILE_PATH="$MOTION_FILE_PATH"
export NUM_EPOCHS="$NUM_EPOCHS"
export PARALLEL_CORES="$PARALLEL_CORES"
export PARALLEL_MODE="$PARALLEL_MODE"
export ROOT_DIR=$ROOT_DIR
export SEED=$SEED
export SETUP_PATH="$SETUP_PATH"
export SKILL_LEVEL="$SKILL_LEVEL"
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export TRAIN_LOGS_DIRS=$TRAIN_LOGS_DIRS
export VISUALIZE="$VISUALIZE"

cd /rl-perf/a2perf/submission

python3.9 -u main_submission.py \
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  $TRAIN_LOGS_ARGS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY \
  --verbosity=$verbosity_level \
  $EXTRA_GIN_BINDINGS_ARG
EOF
