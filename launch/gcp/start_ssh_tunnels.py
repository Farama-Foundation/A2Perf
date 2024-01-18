import subprocess
import time

# Equivalent to your Bash arrays
servers = {
    "web-nav-0": "us-central1-a",
    "web-nav-0-collect": "us-central1-a",
    "web-nav-2": "us-central1-a",
    "web-nav-2-collect": "europe-west4-a",
    "web-nav-4": "us-central1-a",
    "web-nav-4-collect": "us-east4-a",
    "web-nav-6": "us-central1-a",
    "web-nav-6-collect": "us-east4-a",
}

# Start port number
local_port = 10021

# Kill the existing tunnels session
subprocess.run(["tmux", "kill-session", "-t", "tunnels"])

# Create a new detached tmux session with a dummy command
subprocess.run(["tmux", "new-session", "-d", "-s", "tunnels",
                "echo 'Tunnels'; zsh"])

# Iterate  over the servers sorted
servers_list = sorted(servers.keys())

# Loop through each server and establish the SSH tunnel
for i, server in enumerate(servers_list):
  # Get the location of the server
  location = servers[server]
  print(f"Establishing tunnel for {server} on local port {local_port}...")

  # Run expect within tmux
  expect_script = f"""
        set timeout -1
        spawn gcloud compute ssh {server} --project=brain-complex-systems-xgcp --zone={location} --tunnel-through-iap -- -L {local_port}:localhost:22
        expect "Enter passphrase for key '/usr/local/google/home/ikechukwuu/.ssh/google_compute_engine':"
        send "password\\r"
        interact
    """
  subprocess.run([
      "tmux", "new-window", "-t", "tunnels", "-n", server,
      "/usr/bin/expect", "-c", expect_script
  ])

  # Increment the local port
  local_port += 1

  # Add a sleep to allow some breathing room between establishing connections
  time.sleep(2)
