#!/bin/bash
#SBATCH -n 32                  # Number of cores requested
#SBATCH -t 0-12:00:00          # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_gpu            # Partition to submit to
#SBATCH --mem=64000            # Memory per core in MB
#SBATCH --gres=gpu:1           # Number of GPUs to use
#SBATCH -o ../logs/web_nav/%j.log  # File to which output and errors will be written, %j inserts jobid

SEED=0
ENV_BATCH_SIZE=1
DIFFICULTY_LEVEL=-1
TOTAL_ENV_STEPS=1000000
ROOT_DIR=../logs/web_nav
GIN_CONFIG=""

PARTICIPANT_MODULE_PATH=""
WEB_NAV_DIR=""
DOCKER_IMAGE_NAME="rlperf/web_nav:latest"
DOCKER_CONTAINER_NAME="web_nav_container"
DOCKERFILE_PATH="/home/ike2030/workspace/rl-perf/rl_perf/domains/web_nav/docker/Dockerfile"
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

SSH_KEY_PATH=$WEB_NAV_DIR/.ssh/id_rsa
echo "Difficulty Level: $DIFFICULTY_LEVEL"
echo "Env Batch Size: $ENV_BATCH_SIZE"
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

# Move the top level directory
cd ..

# Start a singularity instance in the background and bind /dev/shm and /var/run/
# to the host machine's /dev/shm and /var/run/ directories
# the instance should be named with the job id

singularity instance start \
  --nv --bind /dev/shm:/dev/shm \
  --bind /var/run/:/var/run/ \
  launch/singularity/web_nav/web_nav.sif web_nav_"$SLURM_JOB_ID"

singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
source ~/.web_nav/bin/activate

# Pass in the command-line arguments
export SEED=$SEED
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export TOTAL_ENV_STEPS=$TOTAL_ENV_STEPS
export ROOT_DIR=$ROOT_DIR
export TRAIN_LOGS_DIR=$TRAIN_LOGS_DIR
export DIFFICULTY_LEVEL=$DIFFICULTY_LEVEL

python3 rl_perf/submission/main_submission.py \
  --gin_file=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR \
  --train_logs_dir=$TRAIN_LOGS_DIR \
  --run_offline_metrics_only=$RUN_OFFLINE_METRICS_ONLY
EOF

# Set the python path to a hidden directory in the home directory.
# When we run multiple jobs on the same machine, we do not want them to interfere with each other.
# This also helps us avoid installing packages repeatedly.

#IMPORTANT: Run these commands first to get the environment set up
exit 0

singularity instance start \
  --nv --bind /dev/shm:/dev/shm \
  --bind /var/run/:/var/run/ \
  launch/singularity/web_nav/web_nav.sif web_nav_"$SLURM_JOB_ID"

singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
python3 -m venv  --system-site-packages  ~/.web_nav
source ~/.web_nav/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
pip install -e .
pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt
EOF

# Run a python shell if you need...
singularity shell instance://web_nav_"$SLURM_JOB_ID"
source ~/.web_nav/bin/activate
