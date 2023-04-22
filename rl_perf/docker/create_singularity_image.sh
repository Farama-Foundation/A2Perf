# Build the docker image for the web navigation domain
WEB_NAV_DIR=./rl_perf/domains/web_nav/docker
docker build --no-cache -f $WEB_NAV_DIR/Dockerfile_singularity -t rlperf/web_nav_singularity $WEB_NAV_DIR

# Convert the docker image to a singularity image
docker save rlperf/web_nav_singularity -o $WEB_NAV_DIR/web_nav.tar
#singularity build web_nav.sif docker-archive://$WEB_NAV_DIR/web_nav.tar

# Build the singularity image for the circuit training domain

# Build the image for the quadruped locomotion domain
