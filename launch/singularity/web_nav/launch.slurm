#!/bin/bash
#SBATCH -n 32                  # Number of cores requested
#SBATCH -t 0-12:00:00          # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_gpu            # Partition to submit to
#SBATCH --mem=64000            # Memory per core in MB
#SBATCH --gres=gpu:1           # Number of GPUs to use
#SBATCH -o ../logs/web_nav/%j.log  # File to which output and errors will be written, %j inserts jobid

# define variables
SEED=0
ENV_BATCH_SIZE=1
ROOT_DIR=../logs/web_nav

# parse command-line arguments
for arg in "$@"; do
  case "$arg" in
  --seed=*)
    SEED="${arg#*=}"
    shift
    ;;
  --env_batch_size=*)
    ENV_BATCH_SIZE="${arg#*=}"
    shift
    ;;

  --root_dir=*)
    ROOT_DIR="${arg#*=}"
    shift
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done

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
export ROOT_DIR=$ROOT_DIR

python rl_perf/submission/main_submission.py --gin_file=configs/web_nav_train.gin
EOF

# Set the python path to a hidden directory in the home directory.
# When we run multiple jobs on the same machine, we do not want them to interfere with each other.
# This also helps us avoid installing packages repeatedly.

#IMPORTANT: Run these commands first to get the environment set up
#singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
#python3 -m venv ~/.web_nav
#source ~/.web_nav/bin/activate
#pip install --upgrade pip
#pip install --upgrade setuptools
#pip install -r requirements.txt
#pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt
# pip install -e .
#EOF
