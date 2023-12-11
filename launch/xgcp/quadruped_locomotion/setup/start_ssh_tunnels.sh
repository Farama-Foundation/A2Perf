#!/bin/bash

declare -a servers=("locomotion-0" "locomotion-1" "locomotion-2" "locomotion-3" "locomotion-4" "locomotion-5")
declare -a zones=("us-central1-a" "europe-west4-a" "us-east4-a" "us-east4-a" "us-east7-a" "us-east1-b")

local_port=10999

# Create a new detached tmux session with a dummy command
tmux new-session -d -s tunnels_locomotion "echo 'Tunnels'; zsh"

# Loop through each server using its index and establish the SSH tunnel
for index in "${!servers[@]}"; do

  server="${servers[$index]}"
  zone="${zones[$index]}"

  source /usr/local/google/home/ikechukwuu/workspace/a2perf/a2perf-repo/env/bin/activate
  echo "Establishing tunnel for $server in zone $zone on local port $local_port..."

  # Run expect within tmux, also adding the zone parameter
  # Source the virtual environment for better ssh tunnel performance
  tmux new-window -t tunnels_locomotion -n "$server" "/usr/bin/expect -c \"
        set timeout -1
        spawn gcloud compute ssh $server --zone $zone --tunnel-through-iap -- -L $local_port:localhost:22
        expect \\\"Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':\\\"
        send \\\"password\\r\\\"
        interact
    \""

  # Increment the local port
  ((local_port++))

  # Add a sleep to allow some breathing room between establishing connections
  sleep 2
done
