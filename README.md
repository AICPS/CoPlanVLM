# CoPlanVLM – VLM‑Powered Multi‑Robot Mission Planning

> **ROS 2 workspace that turns a natural‑language instruction into coordinated motion for two TurtleBot 4s — in Ignition Gazebo Fortress or in the REEF Autonomous Vehile Laboratory.**

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Node Reference](#node-reference)
5. [Topics & Interfaces](#topics--interfaces)
6. [Safety Filter (CBF‑QP)](#safety-filter-cbfqp)
7. [Prerequisites](#prerequisites)
8. [Workspace Setup](#workspace-setup)
9. [Configuration](#configuration)
10. [Running in Simulation](#running-in-simulation)
11. [Running in the Lab](#running-in-the-lab)
12. [Testing & Debugging](#testing--debugging)
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
│ node_Control  │◄─────►│ node_Control  │  each sees the other's pose
│    (raph)     │ poses │   (donnie)    │
│  P controller │       │  P controller │
│  + CBF filter │       │  + CBF filter │
└───────┬───────┘       └───────┬───────┘
        │ /raph/cmd_vel         │ /donnie/cmd_vel
        ▼                       ▼
     TurtleBot 4             TurtleBot 4
```

Each `node_Control` runs a [CBF‑QP safety filter](#safety-filter-cbfqp) on its own output, using the
other robot's pose and the shared occupancy grid — so collisions are avoided even when the plan
itself is stale.

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
| `node_Control` | `node_Control.control` | Path follower + [CBF‑QP safety filter](#safety-filter-cbfqp) → `/cmd_vel`. **One instance per robot** |
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
| overhead image | `sensor_msgs/Image` | `/ids_overhead/image` *(sim)* · `/ueye/test/image_raw` *(lab)* |
| overhead info | `sensor_msgs/CameraInfo` | `/ids_overhead/camera_info` *(sim)* · `/ueye/test/camera_info` *(lab)* |

`/vlm_plan` example:

```json
{"planner": "astar", "routes": {"raph": ["G7", "H8"], "donnie": ["C3"]}}
```

`planner` is `astar` for `nav2point`/`maneuver` and `coverage` for `coverage`.

---

<!-- ## Safety Filter (CBF‑QP)

The VLM plan is obstacle‑aware only at planning time. Anything that happens afterwards — the other
robot crossing the path, tracking error, a stale plan — is unmodelled. `node_Control` therefore runs
a **decentralised Control‑Barrier‑Function QP** between the nominal P controller and the `/cmd_vel`
publisher. Each robot runs its own, using only its own pose, the other robot's pose, and the shared
occupancy grid.

```
nominal P controller → [v, ω] → clamp → CBF‑QP filter → safe [v, ω] → /cmd_vel
```

The nominal controller is **not** modified. With no constraint active the filter returns the nominal
command byte‑identical (it short‑circuits rather than solving a trivial QP).

### Method

Because `[v, ω]` cannot move a differential‑drive robot sideways, the barrier is enforced on a
**look‑ahead point** ℓ metres ahead, whose velocity is a full‑rank function of the command:

```
p_ℓ  = [x + ℓcosθ, y + ℓsinθ]        G(θ) = [[cosθ, −ℓsinθ],      ṗ_ℓ = G(θ)u
                                             [sinθ,  ℓcosθ]]
```

For each obstacle/peer centre `p_k`, with `Δp = p_ℓ − p_k`:

```
h_k = |Δp|² − R_k²          ḣ_k = 2Δpᵀ G(θ) u          ZCBF:  ḣ_k ≥ −γ_k h_k
QP row:  −2Δpᵀ G(θ) u  ≤  max(γ_k h_k, 0)                       (A u ≤ b)
```

and the filter solves, subject to those rows and the actuator box:

```
u* = argmin ½(u − u_nom)ᵀ W (u − u_nom),     W = diag(4, 1)
```

**Why the `max(…, 0)` clamp.** Without it, a robot starting inside the unsafe set (`h < 0`) faces a
row demanding `ḣ > 0` — "actively move away" — which is infeasible with an obstacle dead ahead and
no reverse, since the `ω` coefficient is then exactly zero. The QP would fail, the fallback would
stop the robot, and stopping never restores `h`: a permanent freeze. The clamp degrades a violated
row to "do not get worse", which `u = 0` always satisfies. Consequently **the QP is feasible
unconditionally**, no slack variables are needed, and the robot escapes by rotating until forward
motion is permitted again.

**Inflation convention.** Map obstacles are read from the `infl` layer of
`debug/coplan_vlm_occupancy.npz` — the same pre‑inflated grid A\* plans on, already dilated by
`INFLATION_RADIUS` (0.45 m). The filter therefore adds only a margin and does **not** re‑add the
robot radius; using the planner's own grid also means the filter never fights a correctly‑tracked
path. The peer arrives as a raw pose, so it needs the full footprint plus ℓ, because the barrier
protects the look‑ahead point while the body extends behind it:

```
R_obstacle = obstacle_safety_margin
R_robot    = ℓ + radius_self + radius_other + robot_robot_safety_margin
```

**Fallback.** On solver failure, a non‑finite result, or a post‑solve check of `A u ≤ b + ε` failing,
the filter publishes **zero linear and angular velocity** and increments a failure counter. It never
falls back to the unfiltered nominal command.

**Cost.** The QP is only solved on ticks where the nominal command actually violates a row — when it
is already feasible it *is* the optimum, so it is returned unchanged without invoking the solver.
Measured against a real occupancy snapshot (20 496 blocked cells): the filter runs in **0.6 ms mean,
2.1 ms worst** against the 100 ms control period, with 0 solver fallbacks in 1500 poses. -->

<!-- ### Parameters

All on `node_Control`; the launch files set `robot_name` / `robot_names` per instance.

**Defaults are not listed here.** Every filter knob is defined once, in `CBFConfig`
([`nodes/node_Control/cbf_filter.py`](nodes/node_Control/cbf_filter.py)), next to the maths that
justifies its value; `control.py` seeds each parameter declaration from that dataclass. Read the
current defaults there, or from a running node with `ros2 param get`. A table of numbers here would
be a third copy, and third copies go stale.

| Parameter | Meaning |
| --- | --- |
| `enable_safety_filter` | `false` bypasses the filter entirely (`safety_filter:=false`) |
| `robot_name` | which robot this instance drives — **must** be in `robot_names` |
| `robot_names` | roster; peers are derived as roster‑minus‑self |
| `lookahead_distance` | ℓ, metres |
| `cbf_gamma_robot` / `cbf_gamma_obstacle` | ZCBF gains (γ·dt ≪ 1 at 10 Hz) |
| `robot_radius_self` / `robot_radius_other` | TurtleBot 4 body radius |
| `robot_robot_safety_margin` | ≥ (v_self + v_peer) × reaction latency + tracking error |
| `obstacle_safety_margin` | added to the **already inflated** grid; must stay below `RESOLUTION/2` |
| `obstacle_query_radius` | map obstacles beyond this are ignored |
| `max_obstacle_constraints` | cap on map rows, nearest‑first |
| `obstacle_downsample_resolution` | bucket size when thinning obstacle cells |
| `solver_tolerance` | post‑solve feasibility tolerance (catches gross solver failure) |
| `linear_velocity_min` | no reverse — blind backing is worse than stopping |

The QP weights are hardcoded `W = diag(4, 1)` in `cbf_filter.py`; only their ratio matters, and it is
set by normalising each deviation against the actuation available to it.

### Limitations

* The peer is treated as **stationary** within each solve. The error is bounded by its own top speed
  and absorbed by `robot_robot_safety_margin`; re‑derive that margin if `max_linear_vel` is raised.
* Protection **assumes both robots run the filter**. A peer with it disabled, driving at full speed,
  cannot be avoided from one side alone.
* **Poses are assumed fresh** — there is no staleness detection. If a pose source dies mid‑run the
  filter keeps computing barriers from a frozen position. Verify the pose topics are live first.
* ℓ makes `ω` a `1/ℓ` weaker lever than `v`, so the filter **brakes rather than swerves**.
* Starting *outside* the safe set gives non‑worsening only, not a recovery guarantee.
* It enforces safety, not progress: two robots meeting head‑on both brake and can deadlock.
* Safety further depends on pose, map, model and timing accuracy; discrete 10 Hz updates and
  actuator tracking error are what the conservative margins above pay for.

--- -->

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
ros2 launch ros_vrpn_client test.launch name:=donnie
```

```bash
ros2 launch ros_vrpn_client test.launch name:=raph
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

### Unit tests

The safety filter's maths and its closed-loop behaviour are covered by ROS-free tests — no sim, no
API calls:

```bash
source install/setup.bash
python3 -m pytest src/CoPlanVLM/test/ -v
```

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
