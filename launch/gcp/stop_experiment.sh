#!/bin/bash

function run_in_tmux {
  local WINDOW_NAME=$1
  local COMMAND=$2
  tmux send-keys -t tunnels_web_nav:"$WINDOW_NAME" "$COMMAND" C-m
}

COMMAND=$(
  cat <<EOF
pkill -f 'tmux.*launch_web_nav'; docker rm -f web_nav_container; zsh
EOF
)

# Run the command in the appropriate tmux windows
run_in_tmux "web-nav" "$COMMAND"

exit 0
