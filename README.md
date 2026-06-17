# Talking Turtle – LLM‑Powered Navigation Stack

> **Multi‑node ROS 2 workspace for natural‑language control, path‑planning, and autonomous execution in Gazebo Ignition Fortress using Turtlebot4**

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Node Reference](#node-reference)
5. [Topics & Interfaces](#topics--interfaces)
6. [Prerequisites](#prerequisites)
7. [Workspace Setup](#workspace-setup)
8. [Configuration](#configuration)
9. [Running the Stack](#running-the-stack)
10. [Testing & Debugging](#testing--debugging)
11. [Development Guidelines](#development-guidelines)
12. [License](#license)

---

## Introduction

Talking Turtle turns high‑level human instructions into safe, interpretable robot motion.  The stack couples OpenAI GPT models with ROS 2 Humble nodes to:

* parse natural‑language commands,
* plan collision‑free world‑frame paths, and
* drive a TurtleBot 4 along those paths under velocity control.

Everything is written in **Python 3.10+** for quick iteration and leverages standard ROS 2 patterns (publish/subscribe, parameters, launch files).

---

## Features

* **Natural‑Language Interface → Motion** – Speak or type intents such as “Park between rows three and four” and receive `/cmd_vel` messages.
* **Modular Nodes** – Separate nodes for language understanding, path translation, execution monitoring, and low‑level following.
* **OpenAI Integration** – Clean separation between cloud calls (Executive node) and robot runtime; API key handled via environment or parameter.
* **Grid & Pixel Support** – Path Translator converts grid IDs or pixel coordinates (CSV) to real‑world metres.
* **ROS 2 Launch Ready** – Each node ships with example launch files; combine the full stack or run modules individually.
* **Ignition Gazebo Support** – Seamless simulation workflow using Ignition Fortress and official TurtleBot 4 packages.
* **Extensive Logging** – All non‑user messages are promoted to `WARN` for simpler debugging; otherwise concise `INFO` output.

---

## Project Architecture

```
[User / CLI / Voice]
        │ natural‑language           ┌───────────────────────┐
        ▼                            │  TriageApiNode        │
┌─────────────────┐  prompt + chat   │  • gate conversation  │
│ TriageApiNode   ├─────────────────►│  • forward prompts    │
└─────────────────┘                  └───────────────────────┘
                                               │ prompt JSON
                                               ▼
                                   ┌────────────────────────┐
                                   │ ExecutiveApiNode       │
                                   │ • OpenAI call (o4‑mini)│
                                   │ • publishes /path      │
                                   │   & /nav/status        │
                                   └─────────┬──────────────┘
                                 world path  │ Float32MultiArray
                                             ▼
                                   ┌────────────────────────┐
                                   │ Path_Translator        │
                                   │ • grid→world metres    │
                                   └─────────┬──────────────┘
                                world path   │ Float32MultiArray
                                             ▼
                                   ┌────────────────────────┐
                                   │ Controller             │
                                   │ • path follower        │
                                   │ • publishes /cmd_vel   │
                                   └────────────────────────┘
```

Each block may be launched stand‑alone for unit testing.

---

## Node Reference

| Node (exec)                                 | Purpose                                                                                  | Key Parameters                             |
| ------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| **`talking-turtle/basic_LLM_control_node`** | One‑shot language→Twist mapper (demo)                                                    | `openai_api_key`, `temperature` (optional) |
| **`node_Triage_API`**                       | Conversational front‑end; throttles prompts while waiting on the planner                 | `chat_timeout`, `openai_api_key`           |
| **`node_Executive_API`**                    | Sends world snapshot & operator prompt to GPT; returns `/path` list & `/nav/status` JSON | `openai_api_key`                           |
| **`node_Path_Translator`**                  | Converts grid labels or pixel (u,v) coords to world metres and publishes `/world_path`   | `csv_file`, `origin_label`, `metres_per_pixel_x`, `metres_per_pixel_y` |
| **`node_Control`**                       | Subscribes `/world_path`, drives TurtleBot 4 with `/cmd_vel`                             | `lookahead_dist`, `kp`, `max_speed`        |

---

## Topics & Interfaces

| Topic         | Type                          | Publisher → Subscriber              |
| ------------- | ----------------------------- | ----------------------------------- |
| `/path`       | `std_msgs/String` (JSON list) | ExecutiveApiNode → Path\_Translator |
| `/world_path` | `std_msgs/Float32MultiArray`  | Path\_Translator → Control    |
| `/cmd_vel`    | `geometry_msgs/Twist`         | Control → TurtleBot 4 base    |
| `/nav/status` | `std_msgs/String` (JSON)      | ExecutiveApiNode → TriageApiNode    |
| `/openai/log` | `std_msgs/String` (debug)     | *optional*                          |

---

## Prerequisites

* **Robot** – TurtleBot 4 running ROS 2 Humble (tested) or newer.
* **Workstation / Dev PC** – Ubuntu 22.04 / Python 3.10+.  (macOS/Windows WSL work too for development.)
* **ROS 2 Packages** – `rclpy`, `geometry_msgs`, `std_msgs`, `tf_transformations`, `python-csv`, `turtlebot4-simulator`, `turtlebot4-description`, `turtlebot4-msgs`, `turtlebot4-navigation`, `turtlebot4-node`.
* **Python Packages** –

  * `openai>=1.15.0`
  * `python-dotenv` *(optional)*
  * `numpy`, `pandas` (only for tooling, not runtime)

---

## Workspace Setup


Follow the instructions in this [link](https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_simulator.html#installation) for installation of turtlebot4 dependencies and gazebo ignition fortress.


```bash
# 1. create a ROS 2 overlay workspace (if you don’t have one)
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws

# 2. clone the repo
git clone <your-repo-url> src/talking-turtle

# 3. install Python deps
python3 -m pip install -r src/talking-turtle/requirements.txt

# 4. resolve ROS 2 deps & build
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

> **Tip:** add the `source install/setup.bash` line to your `~/.bashrc` for convenience.

---

## Configuration

### OpenAI Credentials

The stack requires an **OpenAI API key**.

```bash
export OPENAI_API_KEY="sk-..."   # shell
```

Alternatively supply `-p openai_api_key:=...` to individual nodes.

You may also store keys in a `.env` file when using `python-dotenv`.

### Launch Parameters

All nodes expose ROS 2 parameters.  See `launch/` directory or run:

```bash
ros2 param dump /executive_api_node   # after startup
```

---

## Running the Stack

### 1. Start Ignition Gazebo

Open a terminal and start the simulator first:

```bash
ros2 launch talking-turtle turtlebot4_ignition.launch.py
```

#### Set the robot namespace

Use `raph` as the robot namespace.

### 2. Wait for startup

Wait until the robot is spawned and you can see the TurtleBot come up with the expected status lights. Wait until you see four lights on, except Wi-Fi.

### 3. Start the VLM stack

Open a second terminal and launch the planning and control stack:

```bash
ros2 launch talking-turtle talking-turtle.launch.py openai_api_key:=$OPENAI_API_KEY
```

This starts **Triage → Executive → Translator → Controller** and binds the CLI prompt for the operator.

### 4. Talk to the robot

Hold `A` to talk, then release it when you are done speaking.

Alternatively, to send a text command from the terminal:
```bash
ros2 topic pub --once /user_text std_msgs/msg/String "{data: 'Find the green block and drive to it'}"
```

### 5. Recovery steps

If you run into an image-type issue or the controller is already stuck from a previous run, restart the computer to clear the existing processes and try again.

### 6. Individual Nodes (Optional)

Run any module on its own for unit tests, e.g.:

```bash
ros2 run talking-turtle node_Path_Translator \
  --ros-args -p csv_file:=maps/grid_lookup.csv -p pixel_sign_x:=-1 -p pixel_sign_y:=1
```

---

## Testing & Debugging

* **ROS 2 CLI :** 
  * `/world_path` to verify translation.
  * `/path` -> The grid path generated by the VLM agent
  * `/world_path` -> The waypoints generated by converted the grid path centers
  * `/raph/cmd_vel` -> The velocity topic for the turtlebot  
* To visualize the camera feed
  * `ros2 run image_tools showimage image:=/ids_overhead/image`
* **rqt\_console / rqt\_graph** – inspect logs and topic flow.
* **Simulation** – Use Gazebo‑classic or Ignition with TurtleBot 4 model to test without hardware.
* **Unit Tests** – See `tests/` for pytest cases.

---

## Development Guidelines

1. **Keep nodes single‑responsibility.**
2. **No secrets in code.**  Use env vars or ROS 2 parameters.
3. **No blocking calls in callbacks.**  Use threads or `rclpy.executors.MultiThreadedExecutor` where required.
4. **Write doc‑strings** and keep this README in sync with code changes.
5. **Run `ruff` & `black`.**  Lint before pushing.

---

## License

Distributed under the Apache 2.0 License (see `LICENSE`).
