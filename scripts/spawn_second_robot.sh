#!/usr/bin/env bash
# Spawn a 2nd (or Nth) TurtleBot 4 into an ALREADY-RUNNING sim and bring up its controllers.
#
# Why a script instead of doing this in the launch file: the in-Gazebo ign_ros2_control
# controller_manager advertises its services BEFORE its resource manager is ready to *configure*
# controllers, and a `spawner` only configures once (no retry). So an in-launch spawner fires too
# early and fails. Waiting a fixed delay after the spawn launch (so the CM is fully ready) and
# THEN running the spawners is exactly the manual timing that reliably works.
#
# Prereqs: the world + robot 1 are already up (ros2 launch talking-turtle turtlebot4_ignition.launch.py).
#
# Usage:
#   ./spawn_second_robot.sh [namespace] [x] [y] [z] [yaw]
#   ./spawn_second_robot.sh                 # raph2 at 0.93 2.96 0.25 0.0
#   ./spawn_second_robot.sh raph3 1.5 2.0   # different robot/pose

# NOTE: no `set -u` — ROS 2's setup.bash references unset vars (e.g. AMENT_TRACE_SETUP_FILES)
# and isn't `set -u`-safe; our own vars below all have ${:-default} fallbacks anyway.

NS=${1:-raph2}
X=${2:-0.93}
Y=${3:-2.96}
Z=${4:-0.25}
YAW=${5:-0.0}

# Delays (seconds): time for the CM to be fully ready, then a small gap between controllers.
JSB_DELAY=${JSB_DELAY:-10}
DIFFDRIVE_DELAY=${DIFFDRIVE_DELAY:-3}

# Must match the running sim/bridges, or this terminal's nodes won't see the graph.
export ROS_LOCALHOST_ONLY=1

source /opt/ros/humble/setup.bash
source ~/projects/turtle4_ws/install/setup.bash

CM="/${NS}/controller_manager"

echo "[spawn_second_robot] launching ${NS} at (x=${X}, y=${Y}, z=${Z}, yaw=${YAW})"
ros2 launch talking-turtle turtlebot4_spawn_hat.launch.py \
    namespace:="${NS}" x:="${X}" y:="${Y}" z:="${Z}" yaw:="${YAW}" &
LAUNCH_PID=$!

# On Ctrl-C / exit, tear down the backgrounded launch.
cleanup() {
    echo "[spawn_second_robot] shutting down ${NS} spawn (pid ${LAUNCH_PID})"
    kill "${LAUNCH_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[spawn_second_robot] waiting ${JSB_DELAY}s for ${CM} to be ready..."
sleep "${JSB_DELAY}"

echo "[spawn_second_robot] loading joint_state_broadcaster"
ros2 run controller_manager spawner joint_state_broadcaster -c "${CM}" || \
    echo "[spawn_second_robot] WARN: joint_state_broadcaster spawn returned non-zero"

sleep "${DIFFDRIVE_DELAY}"

echo "[spawn_second_robot] loading diffdrive_controller"
ros2 run controller_manager spawner diffdrive_controller -c "${CM}" || \
    echo "[spawn_second_robot] WARN: diffdrive_controller spawn returned non-zero"

echo "[spawn_second_robot] done. Drive ${NS} with:"
echo "  ros2 topic pub -r 10 /${NS}/diffdrive_controller/cmd_vel_unstamped geometry_msgs/msg/Twist \"{linear: {x: 0.2}}\""
echo "[spawn_second_robot] (Ctrl-C here tears down ${NS}.)"

# Keep the script alive so the backgrounded launch (and trap) stay active until Ctrl-C.
wait "${LAUNCH_PID}"
