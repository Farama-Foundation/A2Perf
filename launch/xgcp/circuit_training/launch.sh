#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../../.. || exit

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
NUM_COLLECT_SERVERS=4
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

ENV_SETTINGS_BASE=$(
  cat <<EOF
export CT_VERSION='$CT_VERSION' && \
export PYTHON_VERSION='$PYTHON_VERSION' && \
export SSH_KEY_PATH='$SSH_KEY_PATH' && \
export CIRCUIT_TRAINING_DIR='$CIRCUIT_TRAINING_DIR' && \
export DOCKER_IMAGE_NAME='$DOCKER_IMAGE_NAME' && \
export DOCKER_CONTAINER_NAME='$DOCKER_CONTAINER_NAME' && \
export DOCKERFILE_PATH='$DOCKERFILE_PATH' && \
export REQUIREMENTS_PATH='$REQUIREMENTS_PATH' && \
export SEED='$SEED' && \
export ROOT_DIR='$ROOT_DIR' && \
export GIN_CONFIG='$GIN_CONFIG' && \
export PARTICIPANT_MODULE_PATH='$PARTICIPANT_MODULE_PATH' && \
export REVERB_PORT='$REVERB_PORT' && \
export REVERB_SERVER_IP='$REVERB_SERVER_IP' && \
export NETLIST_FILE='$NETLIST_FILE' && \
export INIT_PLACEMENT='$INIT_PLACEMENT' && \
export RUN_OFFLINE_METRICS_ONLY='$RUN_OFFLINE_METRICS_ONLY' && \
export NUM_COLLECT_JOBS='$NUM_COLLECT_JOBS' &&
EOF
)

ENV_SETTINGS=$(
  cat <<EOF
$ENV_SETTINGS_BASE export JOB_TYPE='reverb' && \
  export METRIC_VALUES_DIR='$ROOT_DIR/reverb-eval' && \
  tmux new-session -d -s 'launch_ct' 'bash /home/ikechukwuu/workspace/rl-perf/launch/xgcp/circuit_training/launch_worker.sh && zsh' && echo 'Done'
EOF
)
echo 'ENV_SETTINGS REVERB: ' "$ENV_SETTINGS"

# Start the reverb server using expect
/usr/bin/expect <<EOD
set timeout 20
spawn ssh ikechukwuu@reverb-eval-server -i /home/ikechukwuu/.ssh/google_compute_engine $ENV_SETTINGS
expect {
    "Enter passphrase for key '/home/ikechukwuu/.ssh/google_compute_engine':" {
        send "password\r"
    }
}
expect eof
EOD

# Start all of the collect jobs
for ((i = 0; i < NUM_COLLECT_SERVERS; i++)); do
  #instance name needs two digits
  if [ $i -lt 10 ]; then
    instance_name="ct-collect-0$i"
  else
    instance_name="ct-collect-$i"
  fi

  ENV_SETTINGS=$(
    cat <<EOF
$ENV_SETTINGS_BASE
  export JOB_TYPE='collect' && \
  export METRIC_VALUES_DIR='$ROOT_DIR/$instance_name' && \
  tmux new-session -d -s 'launch_ct'  echo 'testing'
EOF
  )

  echo 'ENV_SETTINGS COLLECT: ' "$ENV_SETTINGS"

  /usr/bin/expect <<EOD
set timeout 20
spawn ssh  ikechukwuu@$instance_name -i /home/ikechukwuu/.ssh/google_compute_engine "echo hi"

expect {
    "Are you sure you want to continue connecting (yes/no)?" {
        send "yes\r"
        exp_continue
    }
    "Enter passphrase for key '/home/ikechukwuu/.ssh/google_compute_engine':" {
        send "password\r"
    }
}
expect eof
EOD

done
