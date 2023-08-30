INSTANCE_NAME="web-nav-0"

expect <<EOD
        set timeout 10
        spawn gcloud compute ssh --tunnel-through-iap $INSTANCE_NAME --zone us-central1-a --command "tmux new-session -d -s 'startup' bash ~/workspace/rl-perf/launch/xgcp/web_nav/setup/setup.sh; bash"
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\r"
        expect eof
EOD
