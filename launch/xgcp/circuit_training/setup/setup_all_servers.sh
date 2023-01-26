#INSTANCE_NAME="reverb-eval-server"
#
#expect <<EOD
#        set timeout 10
#        spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "tmux new-session -d -s 'startup' bash ~/setup.sh; bash"
#        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
#        send "password\r"
#        expect eof
#EOD
#
#INSTANCE_NAME="train-server"
#
#expect <<EOD
#        set timeout 10
#        spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "tmux new-session -d -s 'startup' bash ~/setup.sh; bash"
#        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
#        send "password\r"
#        expect eof
#EOD


# Run the setup file on all servers
for i in $(seq -w 00 04); do
  INSTANCE_NAME="ct-collect-$i"

  expect <<EOD
        set timeout 10
        spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "tmux new-session -d -s 'startup' bash ~/setup.sh; bash"
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\r"
        expect eof
EOD

done

exit 0

# Setup the cloud bucket
for i in $(seq -w 00 19); do
  INSTANCE_NAME="ct-collect-$i"

  expect <<EOD
        set timeout 10
        spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "sudo mkdir -p /mnt/gcs/a2perf && sudo chown $USER /mnt/gcs/a2perf && gcsfuse a2perf /mnt/gcs/a2perf; bash"
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\r"
        expect eof
EOD

done
