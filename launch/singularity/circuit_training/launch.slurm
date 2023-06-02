#!/bin/bash
#SBATCH -n 80                  # Number of cores requested
#SBATCH -t 1-00:00:00          # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_gpu            # Partition to submit to
#SBATCH --mem=64000            # Memory per core in MB
#SBATCH --gres=gpu:2           # Number of GPUs to use
#SBATCH -o ../logs/circuit_training/%j.log  # File to which output and errors will be written, %j inserts jobid

# These arguments most likely do not need to change
CT_VERSION=0.0.3
PYTHON_VERSION=python3.9
DREAMPLACE_PATTERN=dreamplace_20230414_2835324_${PYTHON_VERSION}.tar.gz
TF_AGENTS_PIP_VERSION="tf-agents[reverb]==0.16.0"
DM_REVERB_PIP_VERSION=""
CIRCUIT_TRAINING_DIR=../../rl_perf/domains/circuit_training
DOCKER_IMAGE_NAME="circuit_training:core"
DOCKER_CONTAINER_NAME="circuit_training_container"
DOCKERFILE_PATH="./rl_perf/domains/circuit_training/tools/docker/ubuntu_circuit_training"
REQUIREMENTS_PATH="./requirements.txt"

# New Environment Variables
SEED=""
ROOT_DIR=""
GIN_CONFIG=""
PARTICIPANT_MODULE_PATH=""
REVERB_PORT=""
REVERB_SERVER=""
NETLIST_FILE=""
INIT_PLACEMENT=""
RUN_OFFLINE_METRICS_ONLY=""
NUM_COLLECT_JOBS=""

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
  --reverb_port=*)
    REVERB_PORT="${arg#*=}"
    shift
    ;;
  --reverb_server=*)
    REVERB_SERVER="${arg#*=}"
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

SSH_KEY_PATH=$CIRCUIT_TRAINING_DIR/.ssh/id_rsa
REVERB_SERVER=$REVERB_SERVER:$REVERB_PORT

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
echo "Reverb Port: $REVERB_PORT"
echo "Reverb Server: $REVERB_SERVER"
echo "Netlist File: $NETLIST_FILE"
echo "Initial Placement: $INIT_PLACEMENT"

# Move the top level directory
cd ..

# Start a singularity instance. The instance should be named with the job id
singularity instance start \
  --nv \
  --bind /dev/shm:/dev/shm \
  --bind /var/run/:/var/run/ \
  --bind /sys/class/powercap/:/sys/class/powercap/ \
  launch/singularity/circuit_training/circuit_training.sif circuit_training_"$SLURM_JOB_ID"

singularity exec instance://circuit_training_"$SLURM_JOB_ID" bash <<EOF

source ~/.circuit_training/bin/activate

# Pass in the command line arguments
export ROOT_DIR=$ROOT_DIR
export GLOBAL_SEED=$SEED
export REVERB_PORT=$REVERB_PORT
export REVERB_SERVER=$REVERB_SERVER
export NETLIST_FILE=$NETLIST_FILE
export INIT_PLACEMENT=$INIT_PLACEMENT
export NUM_COLLECT_JOBS=$NUM_COLLECT_JOBS
export WRAPT_DISABLE_EXTENSIONS=true


$PYTHON_VERSION rl_perf/submission/main_submission.py \
  --gin_file=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dir=$TRAIN_LOGS_DIR \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY
EOF

exit 0

# Set the python path to a hidden directory in the home directory.
# When we run multiple jobs on the same machine, we do not want them to interfere with each other.
# This also helps us avoid installing packages repeatedly.

#IMPORTANT: Run these commands first to get the environment set up
exit 0

singularity instance start \
  --nv --bind /dev/shm:/dev/shm \
  --bind /var/run/:/var/run/ \
  launch/singularity/circuit_training/circuit_training.sif circuit_training_"$SLURM_JOB_ID"

singularity exec instance://circuit_training_"$SLURM_JOB_ID" bash <<EOF
python3.9 -m venv  --system-site-packages  ~/.circuit_training
source ~/.circuit_training/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
pip install -e .
pip install -r rl_perf/rlperf_benchmark_submission/circuit_training/requirements.txt
EOF

# Run a python shell if you need...
singularity shell instance://circuit_training_"$SLURM_JOB_ID"
source ~/.circuit_training/bin/activate
