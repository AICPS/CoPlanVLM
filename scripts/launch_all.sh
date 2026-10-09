#!/usr/bin/env bash
# One-shot bring-up of the whole two-robot sim:
#   1. launch turtlebot4_ignition.launch.py (world + raph), backgrounded
#   2. wait IGNITION_DELAY seconds for it to come up
#   3. run spawn_second_robot.sh (donnie + its controllers)
# Ctrl-C tears down both. Everything shares this one terminal (output interleaves).
#
# NOTE: no `set -u` — ROS 2's setup.bash isn't `set -u`-safe (see spawn_second_robot.sh).
#
# Usage:
#   ./launch_all.sh                         # donnie at 0.93 2.96 0.25 0.0
#   ./launch_all.sh raph3 1.5 2.0           # args pass through to spawn_second_robot.sh
#   IGNITION_DELAY=20 ./launch_all.sh       # give the world longer to start

IGNITION_DELAY=${IGNITION_DELAY:-15} # seconds to wait for the world + raph to come up before spawning donnie

# Must match the sim/bridges or nothing shares the ROS graph.
export ROS_LOCALHOST_ONLY=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> src/CoPlanVLM -> src -> workspace root. Derived rather than hardcoded so the script
# works from whatever the workspace is called; override with COPLAN_WS=/path/to/ws if it differs.
COPLAN_WS="${COPLAN_WS:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

source /opt/ros/humble/setup.bash
if [[ ! -f "${COPLAN_WS}/install/setup.bash" ]]; then
  echo "No build found at ${COPLAN_WS}/install/setup.bash — run 'colcon build --symlink-install' from the workspace root first." >&2
  exit 1
fi
source "${COPLAN_WS}/install/setup.bash"

# Kill anything left over from a previous run so we start from a clean slate. Order matters:
# kill the orchestrator/respawner (spawn_second_robot.sh) FIRST, otherwise it relaunches
# controllers right after we kill them. We deliberately do NOT match "launch_all.sh" here — that
# would kill THIS script. None of these patterns match this script's own command line
# (bash .../launch_all.sh), so this is self-safe.
# The actual SIGKILL sweep — used at BOTH startup (clean slate) and shutdown (reap orphans
# that ros2 launch didn't kill, esp. spawners stuck in wait_for_service). Self-safe: none of
# these patterns match this script's own command line (bash .../launch_all.sh).
sweep_sim_procs() {
    pkill -9 -f "spawn_second_robot.sh"             2>/dev/null
    pkill -9 -f "ign gazebo"                        2>/dev/null
    pkill -9 -f "ros_gz_sim/create"                 2>/dev/null
    pkill -9 -f "ros_gz_bridge/parameter_bridge"    2>/dev/null
    pkill -9 -f "turtlebot4"                        2>/dev/null
    pkill -9 -f "coplan_vlm"                    2>/dev/null
    pkill -9 -f "spawner.*raph"                     2>/dev/null  # orphaned controller spawners
}

kill_leftovers() {
    echo "[launch_all] clearing leftover sim processes (clean slate)..."
    sweep_sim_procs
    sleep 2   # give them a moment to die before we launch
}
kill_leftovers

IGNITION_PID=""
SPAWN_PID=""

cleanup() {
    echo "[launch_all] shutting down..."
    [ -n "${SPAWN_PID}" ] && kill "${SPAWN_PID}" 2>/dev/null
    [ -n "${IGNITION_PID}" ] && kill "${IGNITION_PID}" 2>/dev/null
    sleep 1
    # Reap orphans the launches didn't kill (e.g. stock spawners stuck in wait_for_service,
    # which ros2 launch leaves behind on teardown).
    sweep_sim_procs
}
trap cleanup EXIT INT TERM

echo "[launch_all] starting world + raph (turtlebot4_ignition.launch.py)"
ros2 launch coplan_vlm turtlebot4_ignition.launch.py &
IGNITION_PID=$!

echo "[launch_all] waiting ${IGNITION_DELAY}s for the world + raph to come up..."
sleep "${IGNITION_DELAY}"

echo "[launch_all] spawning donnie via spawn_second_robot.sh"
"${SCRIPT_DIR}/spawn_second_robot.sh" "$@" &
SPAWN_PID=$!

# Stay alive until both background jobs exit (or Ctrl-C triggers cleanup).
wait
