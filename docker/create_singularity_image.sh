# Build the docker image for the web navigation domain
WEB_NAV_DIR=./rl_perf/domains/web_nav/docker
docker build -f $WEB_NAV_DIR/Dockerfile_singularity -t rlperf/web_nav:singularity_w_user $WEB_NAV_DIR

# Convert the docker image to a singularity image
docker save rlperf/web_nav:singularity_w_user -o $WEB_NAV_DIR/web_nav.tar
#singularity build web_nav.sif docker-archive://$WEB_NAV_DIR/web_nav.tar
