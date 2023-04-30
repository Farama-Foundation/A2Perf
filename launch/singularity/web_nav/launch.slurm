#!/bin/bash
#SBATCH -n 64 # Number of cores requested
#SBATCH -t 1-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_gpu # Partition to submit to
#SBATCH --mem=64000 # Memory per core in MB
#SBATCH --gres=gpu:1 # Number of GPUs to use
#SBATCH -o ../logs/web_nav/out_%j.log  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../logs/web_nav/err_%j.log  # File to which STDERR will be written, %j inserts jobid

# Move the top level directory
cd ..

# Start a singularity instance in the background and bind /dev/shm and /var/run/
# to the host machine's /dev/shm and /var/run/ directories
# the instance should be named with the job id

singularity instance start \
  --nv --bind /dev/shm:/dev/shm \
  --bind /var/run/:/var/run/ \
  launch/singularity/web_nav/web_nav.sif web_nav_"$SLURM_JOB_ID"

# Set the python path to a hidden directory in the home directory.
# When we run multiple jobs on the same machine, we do not want them to interfere with each other.
# This also helps us avoid installing packages repeatedly.

#singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
#python3 -m venv ~/.web_nav
#EOF

singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
source ~/.web_nav/bin/activate
pip install -e .
#pip install -r requirements.txt
pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt
EOF

singularity exec instance://web_nav_"$SLURM_JOB_ID" bash <<EOF
source ~/.web_nav/bin/activate
python rl_perf/submission/main_submission.py --gin_file=configs/web_nav_train.gin
EOF
