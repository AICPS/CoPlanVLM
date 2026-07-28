# CoPlanVLM – VLM‑Powered Multi‑Robot Mission Planning

> **ROS 2 workspace that turns a natural‑language instruction into coordinated motion for two TurtleBot 4s — in Ignition Gazebo Fortress or in the REEF Autonomous Vehile Laboratory.**

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Node Reference](#node-reference)
5. [Topics & Interfaces](#topics--interfaces)
6. [Prerequisites](#prerequisites)
7. [Workspace Setup](#workspace-setup)
8. [Configuration](#configuration)
9. [Running in Simulation](#running-in-simulation)
10. [Running in the Lab](#running-in-the-lab)
11. [Testing & Debugging](#testing--debugging)
12. [Development Guidelines](#development-guidelines)
13. [License](#license)

---

## Introduction

CoPlanVLM plans missions for **two robots — `raph` and `donnie`** — from a single operator
sentence. One instruction covers both; the planner decides what each robot does.

A prompt runs through two LLM calls:

1. **Classifier** (text) tags the instruction as one of three task types — `nav2point`, `maneuver`, or `coverage`.
2. **Planner** (vision) sees an overhead camera frame marked with a grid of candidate points and returns a route per robot. Chain‑of‑thought is used to produce a `reasoning` field first, so the model reasons before it commits to a route.

The task type determines the low‑level planner that turns those cell picks into an obstacle‑free route: A* for nav2point and maneuver tasks, or a TSP‑ordered sweep for coverage.

Everything was developed on Python 3.10+ on ROS 2 Humble. The same code runs in sim and on hardware — only the camera calibration and the pose source differ.

---

## Features

* **One instruction, two robots** – the planner assigns work to `raph` and `donnie` together.
* **Three task types** – `nav2point` (go there), `maneuver` (go there a particular way), `coverage`
  (sweep a region), each with its own prompt and low‑level planner.
* **Vision grounding** – CLIPSeg segments traversable floor into an occupancy grid; the VLM picks cells off a set‑of‑marks overlay built from that same grid.
* **Sim ↔ hardware parity** – identical nodes and topics; a `camera` parameter selects the calibration. Poses always arrive as NED `PoseStamped` on `/<robot>/ned/pose_stamped` — from MoCap in the lab, or from `node_Odometry_To_Pose` converting Gazebo odometry in sim to NED pose.
* **Debugging Figures** – every run writes the exact prompt, the exact image the model saw, and the resulting routes to a per‑environment `debug/` directory.

---

## Project Architecture

```
operator sentence
        │  /nav/prompt  (std_msgs/String)
        ▼
┌───────────────────────────────────────┐
│ node_Executive_API                    │
│  1. classifier LLM  → task type       │   overhead camera
│  2. CLIPSeg         → occupancy grid  │◄──────────────────
│  3. planner VLM     → cells per robot │
└───────────────┬───────────────────────┘
                │  /vlm_plan
                │  {"planner": "astar"|"coverage",
                │   "routes": {"raph": [...], "donnie": [...]}}
                ▼
┌───────────────────────────────────────┐
│ node_Path_Translator                  │
│  cells → world metres, then A* or     │
│  coverage‑TSP around obstacles        │
└───────┬───────────────────────┬───────┘
        │ /raph/waypoint_path   │ /donnie/waypoint_path
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ node_Control  │       │ node_Control  │
│    (raph)     │       │   (donnie)    │
└───────┬───────┘       └───────┬───────┘
        │ /raph/cmd_vel         │ /donnie/cmd_vel
        ▼                       ▼
     TurtleBot 4             TurtleBot 4
```

`node_Path_Visualizer` subscribes alongside this chain and draws both robots' plans and live poses
on the overhead frame.

The Executive shares its occupancy grid **and the camera frame it was computed from** with the
Translator through `debug/coplan_vlm_occupancy.npz`, so CLIPSeg runs once per plan and every debug
figure shows the same instant the VLM saw.

---

## Node Reference

| Executable | Module | Role |
| --- | --- | --- |
| `node_Executive_API` | `node_Executive_API.exec` | Classifier + planner LLM calls, CLIPSeg, publishes `/vlm_plan` |
| `node_Path_Translator` | `node_Path_Translator.translator_node` | Cell labels → metric waypoints via A* / coverage controllers|
| `node_Control` | `node_Control.control` | Path follower → `/cmd_vel`. **One instance per robot** |
| `node_Path_Visualizer` | `node_Path_Visualizer.path_visualizer` | Live overlay of plans + poses |
| `node_Odometry_To_Pose` | `node_Odometry_To_Pose.odom_to_pose` | **Sim only.** Gazebo odom → NED pose. **One per robot** |
| `obs_seg_cli` | `obs_seg.cli` | Standalone segmentation / occupancy CLI |

**How two robots work:** `node_Control` and `node_Odometry_To_Pose` are each launched **twice**,
remapped into `/raph/…` and `/donnie/…`. The Executive, Translator and Visualizer are single
instances that handle both robots via their `robot_names` parameter.

---

## Topics & Interfaces

| Topic | Type | Publisher → Subscriber |
| --- | --- | --- |
| `/nav/prompt` | `std_msgs/String` | operator → Executive |
| `/vlm_plan` | `std_msgs/String` (JSON) | Executive → Translator, Visualizer |
| `/<robot>/waypoint_path` | `std_msgs/Float32MultiArray` `[x1,y1,x2,y2,…]` | Translator → Control |
| `/<robot>/cmd_vel` | `geometry_msgs/Twist` | Control → base |
| `/<robot>/ned/pose_stamped` | `geometry_msgs/PoseStamped` (BEST_EFFORT) | Odom shim *(sim)* / MoCap *(lab)* → everyone |
| `/path_visualization` | `sensor_msgs/Image` | Visualizer → RViz / `showimage` |
| overhead image | `sensor_msgs/Image` | `/ids_overhead/image` *(sim)* · `/ueye/test/image_raw` *(lab)* |
| overhead info | `sensor_msgs/CameraInfo` | `/ids_overhead/camera_info` *(sim)* · `/ueye/test/camera_info` *(lab)* |

`/vlm_plan` payload:

```json
{"planner": "astar", "routes": {"raph": ["G7", "H8"], "donnie": ["C3"]}}
```

`planner` is `astar` for `nav2point`/`maneuver` and `coverage` for `coverage`.

---

## Prerequisites

* **OS / runtime** – Ubuntu 22.04, ROS 2 Humble, Python 3.10+.
* **Simulation** – Ignition Gazebo **Fortress** plus the TurtleBot 4 sim packages. Follow the
  [TurtleBot 4 simulator install guide](https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_simulator.html#installation).
* **Python** – see [`requirements.txt`](requirements.txt). Includes `torch` / `transformers` for
  CLIPSeg; a GPU is recommended but not required.
* **OpenAI API key** – see [Configuration](#configuration).

---

## Workspace Setup

```bash
# 1. create a ROS 2 overlay workspace (if you don't have one)
mkdir -p ~/turtle4_ws/src && cd ~/turtle4_ws

# 2. clone the repo
git clone <your-repo-url> src/CoPlanVLM

# 3. install Python deps
python3 -m pip install -r src/CoPlanVLM/requirements.txt

# 4. resolve ROS 2 deps & build — FROM THE WORKSPACE ROOT
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

> **Always run `colcon build` from the workspace root.** colcon builds relative to your working
> directory and will silently adopt any directory containing a `package.xml` as the workspace
> root. Running it from inside `src/CoPlanVLM/` produces a green "Finished" while writing a nested
> `build/`, `install/`, `log/` tree that the launch files never look at.

> **Tip:** add `source ~/turtle4_ws/install/setup.bash` to your `~/.bashrc`.

---

## Configuration

### OpenAI credentials

Both launch files load `config/.env` and read **`MY_API_KEY`**:

```bash
# src/CoPlanVLM/config/.env
MY_API_KEY=sk-...
```

Launch fails immediately with `MY_API_KEY not found in .env file` if it is missing.

### Camera calibration

Every node that converts pixels ↔ metres takes a `camera` parameter — `gazebo` or `lab_test` —
selecting an entry in `coord_transform.CAMERAS` (focal lengths, principal point, mounting height).
The launch files set it; you should not need to.

### Other parameters

```bash
ros2 param list /node_Executive_API      # after startup
ros2 launch coplan_vlm coplan_vlm_4sim.launch.py --show-args
```

---

## Running in Simulation

Three terminals from the **workspace root**.

### 1. Start the simulator and both robots

```bash
./src/CoPlanVLM/scripts/launch_all.sh
```

This brings up the Ignition world with `raph`, waits `IGNITION_DELAY` seconds (default 15) for the
world to settle, then runs `spawn_second_robot.sh` to add `donnie` and its controllers. Both
robots come from this one script.

```bash
IGNITION_DELAY=25 ./src/CoPlanVLM/scripts/launch_all.sh    # slow machine
./src/CoPlanVLM/scripts/launch_all.sh donnie 1.5 2.0       # override donnie's spawn pose
```

Wait until both robots are spawned and their controllers are up. Ctrl‑C tears everything down and
sweeps orphaned sim processes.

### 2. Start the planning stack

```bash
ros2 launch coplan_vlm coplan_vlm_4sim.launch.py
```

Starts the Executive, Translator, Visualizer, and one Control + Odometry‑to‑Pose pair per robot.

Optional replanning arguments:

```bash
ros2 launch coplan_vlm coplan_vlm_4sim.launch.py replan_mode:=dynamic replan_period:=30.0
```

`static` (the default) plans once per prompt; `dynamic` re‑plans the same prompt on a timer.

### 3. Send a mission

```bash
./src/CoPlanVLM/scripts/send_prompt.sh "Send raph to the chair and donnie to the table"
./src/CoPlanVLM/scripts/send_prompt.sh "Sweep the open floor with both robots"
```

Watch it work:

```bash
ros2 topic echo /vlm_plan               # task type, planner, cells per robot
ros2 topic echo /raph/waypoint_path     # metric waypoints
ros2 topic echo /donnie/cmd_vel         # velocity commands
```

The Executive logs the chosen task type, e.g.
`[exec] Planned [task=coverage -> planner=coverage] routes: {...}`.

---

## Running in the Lab

Same stack, minus the simulator. The lab replaces two things: MoCap publishes robot poses
directly, and the ueye overhead camera replaces the sim camera. Because MoCap already publishes
NED `PoseStamped`, the deploy launch **omits both `node_Odometry_To_Pose` converters** — they exist
only to translate Gazebo odometry.

### 1. Bring up the room

<!-- TODO: MoCap and ueye camera startup instructions go here. -->

> **Placeholder.** The commands that start the motion‑capture system and the ueye camera driver
> live outside this repo and have not been documented yet.

Before launching, these four topics must be publishing:

| Topic | Source |
| --- | --- |
| `/raph/ned/pose_stamped` | MoCap |
| `/donnie/ned/pose_stamped` | MoCap |
| `/ueye/test/image_raw` | ueye overhead camera |
| `/ueye/test/camera_info` | ueye overhead camera |

Verify each is live and has a publisher:

```bash
ros2 topic list | grep -E "ned/pose_stamped|ueye"
ros2 topic hz /raph/ned/pose_stamped
ros2 topic hz /donnie/ned/pose_stamped
ros2 topic hz /ueye/test/image_raw
ros2 topic echo --once /ueye/test/camera_info
```

A missing `camera_info` is the quiet failure mode: the stack skips lens undistortion rather than
erroring, and overlays drift from reality toward the edges of the frame.

### 2. Start the planning stack

```bash
ros2 launch coplan_vlm coplan_vlm_deploy.launch.py
```

Starts the Executive, Translator, Visualizer, and one Control node per robot — all with
`camera:=lab_test`.

### 3. Send a mission

Identical to sim:

```bash
./src/CoPlanVLM/scripts/send_prompt.sh "Send raph to the door and donnie to the window"
```

---

## Testing & Debugging

### Debug artifacts

Each run writes to its own directory under `debug/`, so runs never overwrite each other:

| Directory | Written by |
| --- | --- |
| `debug/gazebo_sim` | live sim run |
| `debug/deploy_real` | live lab run |
| `debug/offline_test_sim` | `test_pipeline.py` |
| `debug/offline_test_real` | `test_pipeline_real.py` |

Each contains:

| File | Contents |
| --- | --- |
| `marks_overlay.png` | **byte‑exact copy of the image sent to the VLM** |
| `vlm_prompt.txt` / `vlm_response.txt` | the exact exchange |
| `raw_overhead.png` | the undistorted camera frame the plan was built from |
| `inflation_overlay.png` | occupancy: red = obstacle, yellow = inflation margin |
| `robot_paths_waypoints.png` | both robots' final routes on one image |
| `<robot>/` | per‑robot `vlm_selections.png`, `route_centroids.png`, `route_planned.png` |

Every figure in a directory shows the same camera frame, so they can be compared directly.

### Offline harnesses

These run the full planning pipeline on a saved image — no sim, no robots. **Each run spends
OpenAI API credits.**

| Script | Purpose |
| --- | --- |
| `scripts/test_pipeline.py` | The reference pipeline on a sim overhead image |
| `scripts/test_pipeline_real.py` | Same, on a real lab image with `lab_test` calibration |
| `scripts/test_battleship_baseline.py` | Baseline: plain battleship grid instead of set‑of‑marks |
| `scripts/test_regression_only.py` | Baseline: raw pixel coordinates, no grid |

```bash
python3 src/CoPlanVLM/scripts/test_pipeline.py \
  --prompt "Send raph to the chair and donnie to the table"
```

`scripts/test_map_gen.py` renders overlays with **no** API calls — useful for checking the image
before spending a request.

### Live inspection

```bash
ros2 node list                                       # every node up?
ros2 topic echo /vlm_plan
ros2 run image_tools showimage --ros-args -r image:=/path_visualization
rqt_graph                                            # topic wiring
```

---

## Development Guidelines

1. **Shared logic is ROS‑free.** Libraries under `nodes/` (`coord_transform`, `obs_seg`,
   `debug_io`) import no `rclpy`, so live nodes and offline harnesses run identical code. Anything
   validated offline is what the robots execute.
2. **One source of truth per contract.** Prompts, schemas and task→planner routing all live in
   `node_Executive_API/prompt_gen.py`; both the Executive and the harnesses import from it.
3. **Debug artifacts go through `debug_io`** — never an ad‑hoc `cv2.imwrite`. One writer means one
   filename, one failure mode, and no drift between environments.
4. **Keep nodes single‑responsibility** and avoid blocking work in callbacks.
5. **No secrets in code.** Use `config/.env` or ROS 2 parameters.
6. **Rebuild after changing entry points**, `setup.py`, or adding a package directory.
7. **Keep this README in sync** with the code.

---

## License

Distributed under the Apache 2.0 License (see [`LICENSE`](LICENSE)).
