cd "$(dirname "$0")" || exit
cd ../../.. || exit

# Build the Docker image
docker build --build-arg REQUIREMENTS_PATH=./requirements.txt \
  -f rl_perf/domains/web_nav/docker/Dockerfile \
  -t rlperf/web_nav:latest .

if [ "$(docker ps -q -f name=web_nav_container --format "{{.Names}}")" ]; then
  # if it is running, do nothing
  echo "web_nav_container is already running. Run 'docker stop web_nav_container' to stop it. Will use the running container."
else
  docker run -itd --gpus=all --name web_nav_container -v /dev/shm:/dev/shm -v "$(pwd)":/rl-perf rlperf/web_nav:latest
fi

# Install packages inside the container
cat <<EOF | docker exec --interactive web_nav_container bash
cd /rl-perf
pip install -e .
pip install -r rl_perf/rlperf_benchmark_submission/requirements.txt
EOF

# TODO: Parse hyperparameter arguments

# Run the benchmarking code
cat <<EOF | docker exec --interactive web_nav_container bash
cd /rl-perf/rl_perf/submission
python3 main_submission.py --gin_file=configs/web_nav_train.gin
EOF
