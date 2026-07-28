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
| `/<robot>/ned/pose_stamped` | `geometry_msgs/PoseStamped` | Odom shim *(sim)* / MoCap *(lab)* → everyone |
| `/path_visualization` | `sensor_msgs/Image` | Visualizer → the node opens its own OpenCV window |
| overhead image | `sensor_msgs/Image` | `/ids_overhead/image` *(sim)* · `/ueye/test/image_raw` *(lab)* |
| overhead info | `sensor_msgs/CameraInfo` | `/ids_overhead/camera_info` *(sim)* · `/ueye/test/camera_info` *(lab)* |

`/vlm_plan` example:

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
  CLIPSeg, which uses CUDA when available and falls back to CPU automatically.
* **OpenAI API key** – see [Configuration](#configuration).

---

## Workspace Setup

```bash
# 1. create a ROS 2 overlay workspace (if you don't have one)
mkdir -p ~/turtle4_ws/src && cd ~/turtle4_ws

# 2. clone the repo
git clone https://github.com/AICPS/CoPlanVLM.git src/CoPlanVLM

# 3. install Python deps
python3 -m pip install -r src/CoPlanVLM/requirements.txt

# 4. resolve ROS 2 deps & build — FROM THE WORKSPACE ROOT
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

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

Wait until both robots are spawned and visible in the sim. Ctrl‑C tears everything down and sweeps orphaned sim processes.

### 2. Start the planning stack

```bash
ros2 launch coplan_vlm coplan_vlm_4sim.launch.py
```

Starts the Executive, Translator, Visualizer, and one Control + Odometry‑to‑Pose pair per robot.

### 3. Send a mission

```bash
./src/CoPlanVLM/scripts/send_prompt.sh "Send raph to the chair and donnie to the table"
./src/CoPlanVLM/scripts/send_prompt.sh "Patrol the perimeter of the map"
```

Watch it work in the `node_Path_Visualizer` window.

---

## Running in the Lab

Same stack, minus the simulator. The lab replaces two things: MoCap publishes robot poses
directly, and the ueye overhead camera replaces the sim camera. Because MoCap already publishes
NED `PoseStamped`, the deploy launch **omits both `node_Odometry_To_Pose` converters** — they exist
only to translate Gazebo odometry to the NED frame.

### 1. Bring up the Mocap room

ADD ANY ADDITIONAL STEPS for making both publish without errors

```bash
ros2 run ros_vrpn_client ros_vrpn_client --ros-args -r __node:=donnie -p vrpn_ip:="192.168.1.104" -p my_int:=3883
```

```bash
ros2 run ros_vrpn_client ros_vrpn_client --ros-args -r __node:=raph -p vrpn_ip:="192.168.1.104" -p my_int:=3883
```



### 2. Launch the ueye client:
```bash
ros2 launch ueye_cam standalone.launch.py
```

Before launching, these four topics must be publishing:

| Topic | Source |
| --- | --- |
| `/raph/ned/pose_stamped` | MoCap |
| `/donnie/ned/pose_stamped` | MoCap |
| `/ueye/test/image_raw` | ueye overhead camera |
| `/ueye/test/camera_info` | ueye overhead camera |

Verify each is live and has a publisher.

### 3. Start the planning stack

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

```bash
python3 src/CoPlanVLM/scripts/test_pipeline.py \
  --prompt "Send raph to the chair and donnie to the table"
```

`scripts/test_map_gen.py` renders overlays with **no** API calls — useful for checking the image
before spending a request.

---

## License

Distributed under the Apache 2.0 License (see [`LICENSE`](LICENSE)).
