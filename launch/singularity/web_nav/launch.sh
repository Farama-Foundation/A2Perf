# set working directory to the directory of this script
cd "$(dirname "$0")" || exit
cd ../../.. || exit

# By launch, we mean build the singularity script
docker build rl_perf/domains/web_nav/docker/Dockerfile -t rlperf/web_nav:latest .
docker run --rm \
  -it \
  -v "$(pwd)":/rl-perf \
  rlperf/web_nav:latest bash -c "pip install -r rl_perf/rlperf_benchmark_submission/requirements.txt"
