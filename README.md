# CoPlanVLM: Coordinated Path Planning for a Robot Team using Vision Language Models

> **ROS 2 workspace that turns a natural‑language instruction into coordinated motion for two TurtleBot 4s — in Ignition Gazebo Fortress or in the REEF Autonomous Vehicle Laboratory.**

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
12. [Test Prompts and Success Criteria](#test-prompts-and-success-criteria)
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

Each `node_Control` runs a CBF‑QP safety filter on its own output, using the
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
| `node_Control` | `node_Control.control` | Path follower + CBF‑QP safety filter → `/cmd_vel`. **One instance per robot** |
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

# 4. add your OpenAI key (see Configuration) — do this BEFORE building
cp src/CoPlanVLM/config/.env.example src/CoPlanVLM/config/.env
$EDITOR src/CoPlanVLM/config/.env

# 5. resolve ROS 2 deps & build — FROM THE WORKSPACE ROOT
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

> **Tip:** add `source ~/turtle4_ws/install/setup.bash` to your `~/.bashrc`.

---

## Configuration

### OpenAI credentials

Both launch files load `config/.env` and read **`MY_API_KEY`**. Copy the template and insert your
own key:

```bash
cp src/CoPlanVLM/config/.env.example src/CoPlanVLM/config/.env
$EDITOR src/CoPlanVLM/config/.env        # set MY_API_KEY=sk-...
```

`config/.env` is gitignored — keep your key out of version control. Launch fails immediately with
`MY_API_KEY not found in .env file` if it is missing.

> Create the file **before** `colcon build`, or re-run the build afterwards: the launch files read
> the copy under `install/`, which the build places there.

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
./src/CoPlanVLM/scripts/send_prompt.sh "Surround the person in the purple shirt by sending robots to the left and right sides of the person"
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

> **Lab-only dependencies.** `ros_vrpn_client` and `ueye_cam` are site-specific packages for the
> REEF MoCap rig and the ueye overhead camera. They are not public ROS 2 packages and are
> deliberately **not** declared in `package.xml`, so this section is not reproducible outside the
> lab. Everything in [Running in Simulation](#running-in-simulation) and
> [Testing & Debugging](#testing--debugging) runs anywhere.

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

### 4. Send a mission

Identical to sim:

```bash
./src/CoPlanVLM/scripts/send_prompt.sh "Send raph to the door and donnie to the window"
```

---

## Testing & Debugging

### The bundled `test_data/` fixture

The offline harnesses need no simulator, no robots and no capture step — the fixture ships with the
repo at `test_data/`:

| File | Scene |
| --- | --- |
| `overhead.png` + `poses.json` | the Gazebo warehouse frame used by every `--mode sim` run |
| `AVL_1.png`, `AVL_2.png`, `AVL_3.png` | the three lab scenes used by `--mode real` / `test_pipeline_real.py` |

Every harness resolves this directory from its own location, so the commands below work from any
working directory. Pass `--data <dir>` to point at a different capture.

To regenerate the sim fixture against a different scene, run this once with the simulator up:

```bash
python3 src/CoPlanVLM/scripts/save_overhead.py \
  --out src/CoPlanVLM/test_data --robots raph donnie
```

> The lab scenes' robot **identities** are inferred, not measured: `test_pipeline_real.py` assumes
> the easternmost robot is `raph`. If a run's markers look swapped, swap the two pose tuples in
> `SCENES` at the top of that file.

### Offline harnesses

These run the full planning pipeline on a saved image. **Each run spends OpenAI API credits.**

| Script | Purpose |
| --- | --- |
| `scripts/test_pipeline.py` | The reference pipeline on the sim overhead image |
| `scripts/test_pipeline_real.py` | Same, on a lab image with `lab_test` calibration |

```bash
# sim scene — test_data/overhead.png (gazebo calibration)
python3 src/CoPlanVLM/scripts/test_pipeline.py \
  --prompt "Surround the person in the purple shirt by sending robots to the left and right sides of the person"

# lab scene — test_data/AVL_2.png (lab_test calibration)
python3 src/CoPlanVLM/scripts/test_pipeline_real.py --scene AVL_2 \
  --prompt "Move one robot to the north of the boxes and one to the south of the boxes."
```

`test_pipeline_real.py` takes `--scene AVL_1`, `AVL_2` or `AVL_3`; it defaults to `AVL_3`. Each
scene carries its own hardcoded robot poses, so the scene and the prompt have to match — the boxes
above exist in `AVL_2`. See [Test Prompts and Success Criteria](#test-prompts-and-success-criteria)
for a prompt that has been validated against each scene.

### Ablations

Three configurations isolate the contribution of each component. Only the obstacle-marking arm
needs its own script; the other two are flags on the reference harness.

| Configuration | Command |
| --- | --- |
| CoPlanVLM (full) | `test_pipeline.py --prompt "…"` |
| CoPlanVLM w/o Marked Obs. | `ablation_test_no_obs_markers.py --mode sim --prompt "…"` |
| CoPlanVLM Maneuver Only | `test_pipeline.py --planner maneuver --out debug/ablation_maneuver_only_sim --prompt "…"` |
| CoPlanVLM w/o CoT | `test_pipeline.py --no-cot --out debug/ablation_no_cot_sim --prompt "…"` |

Each arm takes `--mode real` (for the wrapper) or `test_pipeline_real.py --scene AVL_1|AVL_2|AVL_3`
to run the same configuration on the lab scenes.

**How each one works:**

* **w/o Marked Obs.** needs a dedicated script because the red-X cue reaches the model through
  *four* channels: the rendered image, the overlay description, the impassable-label listing, and
  the per-step descriptions in the reply JSON schema. The wrapper swaps in `generate_prompt_nomark`
  and `route_schema_nomark`, which strip all four. `--map-overlay points` removes only the first
  three and leaves the red-marker wording in the schema — it is a **half-ablation and not the
  ablation arm**.
* **Maneuver Only** sets the controller directly, so the classifier call is skipped entirely rather
  than overridden.
* **w/o CoT** drops both the chain-of-thought scaffold from the prompt and the per-step reasoning
  fields from the reply schema, so the model's first generated tokens are the waypoints.

> **Pass `--out` for the last two.** The no-markers wrapper forces its own output directory, but
> `--planner maneuver` and `--no-cot` default to `debug/offline_test_sim` — the same directory as
> the full pipeline — and will otherwise overwrite the baseline's artifacts.

> **Ablations are offline-only.** The live `node_Executive_API` hardcodes the production overlay
> and chain-of-thought, and exposes no parameter for either, so these configurations cannot be run
> through the launch files.

### Baselines

Three alternative marking schemes, each a single VLM call with no classifier and no
chain-of-thought:

| Baseline | Command |
| --- | --- |
| Grid Overlay | `test_grid_overlay.py --mode sim --prompt "…"` |
| Pixel Selection | `test_pixel_selection.py --mode sim --prompt "…"` |
| CoNVOI Marking | `test_convoi_prompting.py --mode sim --prompt "…"` |

`scripts/plot_success_rates.py` regenerates the comparison figure. Its `SUCCESS_RATES` table is
entered by hand — there is no automated path from harness output into the chart.

### Debug artifacts

Every run writes its prompt, the exact image the VLM saw, the occupancy overlay and the planned
routes to `debug/` at the workspace root — run the offline harnesses from there so their output
lands alongside the live runs'.

Each run writes to its own directory under `debug/`, so runs never overwrite each other:

| Directory | Written by |
| --- | --- |
| `debug/gazebo_sim` | live sim run |
| `debug/deploy_real` | live lab run |
| `debug/offline_test_sim` / `debug/offline_test_real` | `test_pipeline.py` / `test_pipeline_real.py` |
| `debug/ablation_no_obs_markers_{sim,real}` | `ablation_test_no_obs_markers.py` |
| `debug/grid_overlay_{sim,real}` | `test_grid_overlay.py` |
| `debug/pixel_selection_{sim,real}` | `test_pixel_selection.py` |
| `debug/convoi_{sim,real}` | `test_convoi_prompting.py` |

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

---

## Test Prompts and Success Criteria

| Prompt | Classification | Environment | Prompt | Success Criteria |
|--------|---------------|-------------|--------|-----------------|
| 1 | nav2point | Sim | "Surround the person in the purple shirt by sending robots to the left and right sides of the person" | Both robots end within one grid point of the person in purple, with one robot on the left of the person and one robot on the right of the person. |
| 2 | nav2point | Sim | "Have each robot hide on left side of the nearest obstacle" | Robots are positioned directly left of the nearest obstacle, within 1 grid point of the obstacle. |
| 3 | nav2point | Sim | "Visit each of the four colored boxes" | For each box, a robot passes within one grid point of the box. |
| 4 | nav2point | AVL 1 | "Send both robots to the white X marked in tape" | Both robots end within 1 grid point of the white tape X. |
| 5 | nav2point | AVL 2 | "Move one robot to the north of the boxes and one to the south of the boxes." | One robot ends within 1 grid point of the north side of the rack of boxes; the other ends within 1 grid point of the south side. |
| 6 | nav2point | AVL 3 | "Visit each object in the scene." | At least one robot passes within 1 grid point of each of the 3 objects (2 boxes, 1 chair). |
| 7 | Coverage | Sim | "Patrol the perimeter of the map" | Every point of the edge of the map that is not blocked by an obstacle is visited by at least one robot. Success is 95% of points visited, meaning 2 points total can be missed. |
| 8 | Coverage | Sim | "Patrol around the rack of boxes" | At least one robot passes within one grid point of each grid point on the perimeter of the rack. |
| 9 | Coverage | Sim | "Survey the area inside the black and yellow striped tape." | All open points within the square marked by the tape are visited by a robot. (At least 95% coverage means no points can be missed.) |
| 10 | Coverage | AVL 1 | "Have the robots patrol the area around the chair." | At least one robot passes within 1 grid point of all 4 sides (North, South, East, West) of the chair. Success = all 4 sides visited. |
| 11 | Coverage | AVL 2 | "Have the robots survey the large square marked in tape on the left side." | At least one robot visits every free grid point inside the taped square. Success = 95% of free points visited (max 1 point missed). |
| 12 | Coverage | AVL 3 | "Patrol the left half of the map." | At least one robot visits every free grid point in the left half of the map. Success = ≥95% of free points visited (max 1 point missed). |
| 13 | maneuver | Sim | "Have donnie do a loop around the red and blue boxes and then meet up with raph" | Donnie completes a full loop around the red and blue boxes, then ends within 1 grid point of Raph. |
| 14 | maneuver | Sim | "Have raph approach the person in white from the left while donnie passes around the right of the rack of boxes before approaching the same person from the other side" | Raph's final approach vector reaches the person from their left; Donnie's path passes around the right side of the rack and his final approach reaches the person from their right. Both robots end within 1 grid point of the person in white. |
| 15 | maneuver | Sim | "Have one robot move along the north edge of the map to the person in purple while the other robot travels along the south edge until it is directly south of the same person, then move north to the person" | Robot A stays within 1 grid point of the north edge for its entire traverse until reaching the person. Robot B stays within 1 grid point of the south edge until it is directly south of the person (same column ±1 grid point), then moves north. Both robots end within 1 grid point of the person in purple. |
| 16 | maneuver | AVL 1 | "Have the robots switch places. Have raph move around the south of the chair and have Donnie move around the north of the chair." | Each robot ends within 1 grid point of the other robot's starting position. Raph's path passes to the south of the chair; Donnie's path passes to the north of the chair. |
| 17 | maneuver | AVL 2 | "Send both robots to the person. Have one approach around the north of the boxes, and one approach around the south of the boxes." | Both robots end within 1 grid point of the person. One robot's path passes around the north side of the boxes; the other's passes around the south side. |
| 18 | maneuver | AVL 3 | "Have Donnie circle around the box on the left and then go to the chair." | Donnie completes a full loop around the left box, then ends within 1 grid point of the chair. |

## License

Distributed under the Apache 2.0 License (see [`LICENSE`](LICENSE)).
