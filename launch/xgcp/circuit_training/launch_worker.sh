#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

export CIRCUIT_TRAINING_DIR=../../rl_perf/domains/circuit_training

mkdir -p "$CIRCUIT_TRAINING_DIR/tools/docker/.ssh"
yes | ssh-keygen -t rsa -b 4096 -C "circuit_training" -f "$SSH_KEY_PATH" -N ""

echo "Successfully parsed command-line arguments."

# check for gpu with nvidia-smi. if gpu doesn't exist use cpu base image
if nvidia-smi; then
  echo "GPU exists. Will use gpu base image."
  BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
else
  echo "GPU does not exist. Will use cpu base image."
  BASE_IMAGE="ubuntu:20.04"
fi

# print the current working directory
echo "Current working directory: $(pwd)"

docker build \
  --rm \
  --pull \
  --build-arg base_image=$BASE_IMAGE -f "${DOCKERFILE_PATH}" \
  -t "$DOCKER_IMAGE_NAME" rl_perf/domains/circuit_training/tools/docker

echo "Successfully built docker image."

if [ "$(docker ps -q -f name="$DOCKER_CONTAINER_NAME" --format "{{.Names}}")" ]; then
  echo "$DOCKER_CONTAINER_NAME is already running. Run 'docker stop $DOCKER_CONTAINER_NAME' to stop it. Will use the running container."
else
  echo "$DOCKER_CONTAINER_NAME is not running. Will start a new container."
  # initial command
  docker_run_command="docker run -itd --rm -p 2022:22"

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
  docker_run_command+=" -v \"$(pwd)\":/rl-perf"
  docker_run_command+=" -v /mnt/gcs:/mnt/gcs"
  docker_run_command+=" --workdir /rl-perf"
  docker_run_command+=" --name \"$DOCKER_CONTAINER_NAME\""
  docker_run_command+=" \"$DOCKER_IMAGE_NAME\""

  echo "Running command: $docker_run_command"
  eval "$docker_run_command"
fi
echo "Successfully started docker container."

# Install required packages inside the container
docker exec --interactive "$DOCKER_CONTAINER_NAME" bash <<EOF
# Install requirements for the rl-perf repo
pip install  --no-cache-dir -r "$REQUIREMENTS_PATH"

# Install RLPerf as a package
pip install --no-cache-dir -e .
EOF

docker exec --interactive "$DOCKER_CONTAINER_NAME" bash <<EOF
export PYTHONPATH=/rl-perf:\$PYTHONPATH
export PYTHONPATH=/dreamplace:/dreamplace/dreamplace:\$PYTHONPATH
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
  --gin_config=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --metric_values_dir=$METRIC_VALUES_DIR \
  --train_logs_dirs=$TRAIN_LOGS_DIRS \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY
EOF

#exit 0

# Run these commands for the "smoke-test"
docker run \
  -it \
  --rm \
  --gpus=all \
  -v /home/ikechukwuu/workspace/rl-perf:/rl-perf \
  -v /mnt:/mnt \
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
