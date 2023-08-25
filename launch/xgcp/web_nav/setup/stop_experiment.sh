#!/bin/bash

function run_in_tmux {
  local WINDOW_NAME=$1
  local COMMAND=$2
  tmux send-keys -t tunnels:"$WINDOW_NAME" "$COMMAND" C-m
}

COMMAND=$(
  cat <<EOF
pkill -f 'tmux.*launch_web_nav'; docker rm -f circuit_training_container; zsh
EOF
)

# Run the command in the appropriate tmux windows
run_in_tmux "reverb-eval-server" "$COMMAND"
run_in_tmux "train-server" "$COMMAND"

# Run the command on all windows
for i in $(seq -w 00 19); do
  run_in_tmux "ct-collect-$i" "$COMMAND"
done

exit 0
