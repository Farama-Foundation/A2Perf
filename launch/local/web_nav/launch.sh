cd "$(dirname "$0")" || exit
cd ../../.. || exit

SEED=0
ENV_BATCH_SIZE=1
ROOT_DIR=../logs/web_nav
GIN_CONFIG=""
PARTICIPANT_MODULE_PATH=""

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
  --gin_config=*)
    GIN_CONFIG="${arg#*=}"
    shift
    ;;
  --participant_module_path=*)
    PARTICIPANT_MODULE_PATH="${arg#*=}"
    shift
    ;;
  *)
    echo "Invalid option: $arg"
    exit 1
    ;;
  esac
done
echo "Env Batch Size: $ENV_BATCH_SIZE"
echo "Seed value: $SEED"
# create ssh-key in WEB_NAV_DIR without password
mkdir -p "$WEB_NAV_DIR/.ssh"
ssh-keygen -t rsa -b 4096 -C "web_nav" -f "$WEB_NAV_DIR/.ssh/id_rsa" -N ""

# Build the Docker image
docker build --rm --build-arg REQUIREMENTS_PATH=./requirements.txt \
  --build-arg WEB_NAV_DIR="$WEB_NAV_DIR" \
  -f rl_perf/domains/web_nav/docker/Dockerfile \
  -t rlperf/web_nav:latest rl_perf/domains/web_nav

if [ "$(docker ps -q -f name=web_nav_container --format "{{.Names}}")" ]; then
  # if it is running, do nothing
  echo "web_nav_container is already running. Run 'docker stop web_nav_container' to stop it. Will use the running container."
else
  docker run -itd \
    --gpus=all \
    --name web_nav_container \
    -v /dev/shm:/dev/shm \
    -v "$(pwd)":/rl-perf -p 2022:22 \
    rlperf/web_nav:latest
fi

# Install packages inside the container
cat <<EOF | docker exec --interactive web_nav_container bash
cd /rl-perf
pip install -r requirements.txt
pip install -e .
pip install -r rl_perf/rlperf_benchmark_submission/web_nav/requirements.txt
EOF

# TODO: Parse hyperparameter arguments

# Run the benchmarking code
cat <<EOF | docker exec --interactive web_nav_container bash
export SEED=$SEED
export ENV_BATCH_SIZE=$ENV_BATCH_SIZE
export ROOT_DIR=$ROOT_DIR
cd /rl-perf/rl_perf/submission
python3 main_submission.py \
  --gin_file=$GIN_CONFIG \
  --participant_module_path=$PARTICIPANT_MODULE_PATH \
  --root_dir=$ROOT_DIR
EOF
