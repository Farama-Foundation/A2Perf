#!/bin/bash

declare -a servers=("web-nav-0")

# Start port number
local_port=10021

# Create a new detached tmux session with a dummy command
tmux new-session -d -s tunnels_web_nav "echo 'Tunnels'; zsh"

# Loop through each server and establish the SSH tunnel
for server in "${servers[@]}"; do
  echo "Establishing tunnel for $server on local port $local_port..."

  # Run expect within tmux
  tmux new-window -t tunnels_web_nav -n "$server" "/usr/bin/expect -c \"
        set timeout -1
        spawn gcloud compute ssh $server --tunnel-through-iap -- -L $local_port:localhost:22
        expect \\\"Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':\\\"
        send \\\"password\\r\\\"
        interact
    \""

  # Increment the local port
  ((local_port++))

  # Add a sleep to allow some breathing room between establishing connections
  sleep 2
done
