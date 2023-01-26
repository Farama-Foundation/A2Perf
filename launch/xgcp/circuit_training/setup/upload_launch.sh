#!/bin/bash
#
#for i in $(seq -w 00 19); do
#  INSTANCE_NAME="ct-collect-$i"
#
#  expect <<EOD
#        set timeout 10
#        spawn gcloud compute scp --tunnel-through-iap ../launch_worker.sh $INSTANCE_NAME:/home/ikechukwuu/ --zone us-central1-a
#        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
#        send "password\r"
#        expect eof
#EOD
#
#  echo "Done with $INSTANCE_NAME"
#done

#exit 0
#
#INSTANCE_NAME=reverb-eval-server
#expect <<EOD
#set timeout 10
#spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "mkdir -p /home/ikechukwuu/workspace/rl-perf/launch/xgcp/circuit_training"
#expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
#send "password\r"
#expect eof
#EOD
#
#INSTANCE_NAME="train-server"
#expect <<EOD
#set timeout 10
#spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "mkdir -p /home/ikechukwuu/workspace/rl-perf/launch/xgcp/circuit_training"
#expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
#send "password\r"
#expect eof
#EOD

#exit 0
INSTANCE_NAME="reverb-eval-server"

expect <<EOD
        set timeout 10
        spawn gcloud compute scp --tunnel-through-iap ../launch_worker.sh $INSTANCE_NAME:/home/ikechukwuu/workspace/rl-perf/launch/xgcp/circuit_training --zone us-central1-a
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\r"
        expect eof
EOD

INSTANCE_NAME="train-server"

expect <<EOD
        set timeout 10
        spawn gcloud compute scp --tunnel-through-iap ../launch_worker.sh $INSTANCE_NAME:/home/ikechukwuu/workspace/rl-perf/launch/xgcp/circuit_training --zone us-central1-a
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\r"
        expect eof
EOD
