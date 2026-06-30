#!/usr/bin/env bash
# Publish a constant velocity to /<robot>/cmd_vel until interrupted (Ctrl-C).
#
# Usage:
#   ./drive_raph.sh                      # raph, 0.2 m/s forward, no rotation
#   ./drive_raph.sh donnie                # donnie, 0.2 m/s forward
#   ./drive_raph.sh donnie 0.3            # donnie, 0.3 m/s forward
#   ./drive_raph.sh donnie 0.3 0.5        # donnie, 0.3 m/s forward + 0.5 rad/s rotation
#
# ros2 topic pub repeats at -r Hz, which keeps the diff-drive watchdog fed so
# the robot keeps moving. On Ctrl-C the script sends one zero Twist to stop it.

set -euo pipefail

ROBOT="${1:-raph}"    # robot namespace, prepended before /cmd_vel
LINEAR="${2:-0.2}"    # m/s along x
ANGULAR="${3:-0.0}"   # rad/s about z
RATE="${4:-10}"       # publish rate in Hz

TOPIC="/${ROBOT}/cmd_vel"

stop() {
    echo
    echo "Stopping ${TOPIC}..."
    ros2 topic pub --once "${TOPIC}" geometry_msgs/msg/Twist \
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
    exit 0
}
trap stop INT TERM

echo "Publishing to ${TOPIC} at ${RATE} Hz: linear.x=${LINEAR}, angular.z=${ANGULAR}"
echo "Press Ctrl-C to stop."
ros2 topic pub -r "${RATE}" "${TOPIC}" geometry_msgs/msg/Twist \
    "{linear: {x: ${LINEAR}, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: ${ANGULAR}}}"
