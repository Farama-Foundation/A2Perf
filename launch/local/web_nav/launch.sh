#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

echo $(pwd)
SEED=0
ENV_BATCH_SIZE=1
TOTAL_ENV_STEPS=1000000
ROOT_DIR=../logs/web_nav
GIN_CONFIG=""
DIFFICULTY_LEVEL=-1
PARTICIPANT_MODULE_PATH=""
WORK_UNIT_ID=0
WEB_NAV_DIR="$(pwd)/rl_perf/domains/web_nav"
DOCKER_IMAGE_NAME="rlperf/web_nav:latest"
DOCKER_CONTAINER_NAME="web_nav_container"
DOCKERFILE_PATH="$(pwd)/rl_perf/domains/web_nav/docker/Dockerfile"
REQUIREMENTS_PATH="./requirements.txt"
RUN_OFFLINE_METRICS_ONLY="False"
BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"

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
echo "Work unit id: $WORK_UNIT_ID"

# create ssh-key in WEB_NAV_DIR without password
mkdir -p "$WEB_NAV_DIR/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "web_nav" -f "$SSH_KEY_PATH" -N ""

docker build \
  --rm \
  --pull \
  --build-arg base_image=$BASE_IMAGE \
  -f "${DOCKERFILE_PATH}" \
  --build-arg WEB_NAV_DIR="$WEB_NAV_DIR" \
  -t "$DOCKER_IMAGE_NAME" \
  "$WEB_NAV_DIR"/docker

echo "Successfully built docker image."

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  # initial command
  #  docker_run_command="docker run -itd --rm -p 2022:22 --privileged"
  docker_run_command="docker run -itd --rm  --privileged"

  # check to see if /sys/class/powercap exists. if so, mount it
  if [ -d "/sys/class/powercap" ]; then
    docker_run_command+=" -v /sys/class/powercap:/sys/class/powercap"
  else
    echo "No powercap directory found. Will not mount it."
  fi

  # check for GPU and add the necessary flag if found
  if command -v nvidia-smi &>/dev/null; then
    #    docker_run_command+=" --gpus all"
    # give two GPUs depending on the docker_container_name last digit
    docker_run_command+=" --gpus \"device=$WORK_UNIT_ID\""
  fi

  # append the rest of the flags
  docker_run_command+=" -v $(pwd):/rl-perf"
  docker_run_command+=" -v /dev/shm:/dev/shm"
  docker_run_command+=" -v /home/ikechukwuu/workspace/gcs:/mnt/gcs/"
  #  docker_run_command+=" -v /var/run/:/var/run/"
  #  docker_run_command+=" -v /tmp/:/tmp/"
  #  docker_run_command+=" -v /var/tmp/:/var/tmp/"
  docker_run_command+=" -v /sys/fs/cgroup:/sys/fs/cgroup:rw"
  #  docker_run_command+=" -e DISPLAY=$DISPLAY"
  docker_run_command+=" -v $HOME/.Xauthority:/user/.Xauthority:rw"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi

#exit 0

EXTRA_GIN_BINDINGS=$(
  cat <<EOF
  --extra_gin_bindings='track_emissions.default_cpu_tdp=125' \
  --extra_gin_bindings='track_emissions.gpu_ids=[$WORK_UNIT_ID]' 
EOF
)

#exit 0
# Install packages inside the container
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
cd /rl-perf

# Install requirements for the rl-perf repo
python3 -m pip install -r requirements.txt

# Install RLPerf as a packages
pip install -e .


# Install packages specific to the user's training code
pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt
EOF

# Run the benchmarking code
cat <<EOF | docker exec --interactive "$DOCKER_CONTAINER_NAME" bash
export SEED=$SEED
export CUDA_VISIBLE_DEVICES=$WORK_UNIT_ID
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CUDA_MALLOC_ASYNC=1
export ROOT_DIR=$ROOT_DIR
export TRAIN_LOGS_DIRS=$TRAIN_LOGS_DIRS
export DIFFICULTY_LEVEL=$DIFFICULTY_LEVEL
export PYTHONPATH=\$(pwd):\$PYTHONPATH
cd /rl-perf/rl_perf/submission
export DISPLAY=:0
python3 -u main_submission.py \
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY \
  $EXTRA_GIN_BINDINGS
EO
