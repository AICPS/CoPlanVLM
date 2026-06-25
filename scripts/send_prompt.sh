#!/usr/bin/env bash
# Publish a navigation prompt straight to the Executive node (/nav/prompt), bypassing triage
# so the planner gets your exact wording (no "telephone" rephrasing).
#
# Usage:
#   ./send_prompt.sh Send raph to the door and raph2 to the window
#   ./send_prompt.sh "Send a robot to each person."
#   ./send_prompt.sh                      # uses the default prompt below
#
# Everything after the script name is taken as the prompt, so quotes are optional (but use
# them if your text contains shell metacharacters like ' or ! ).

set -euo pipefail

TOPIC="/nav/prompt"
DEFAULT_PROMPT="Send a robot to each person."

# Join all args into one prompt string; fall back to the default if none given.
if [ "$#" -gt 0 ]; then
    PROMPT="$*"
else
    PROMPT="$DEFAULT_PROMPT"
fi

# ros2 topic pub YAML uses single quotes around the string, so escape any single quotes in
# the prompt as the YAML-safe '\'' sequence.
ESCAPED=${PROMPT//\'/\'\\\'\'}

echo "Publishing to ${TOPIC}:"
echo "  ${PROMPT}"
ros2 topic pub --once "${TOPIC}" std_msgs/msg/String "{data: '${ESCAPED}'}"

echo "Sent. Watch results with:"
echo "  ros2 topic echo /grid_path     # per-robot grid plan"
echo "  ros2 topic echo /nav/status    # success / analysis / error reason"
