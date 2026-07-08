#!/usr/bin/env python3
"""
Executive API Node — Adaptive Path‑Planning Edition (Responses API)
===================================================================
Receives an operator goal prompt on `/nav/prompt`, and plans in two LLM
calls:

1. **Classifier** — a text‑only call tags the instruction with a task
   type (`nav2point` / `manouver` / `coverage`; see prompt_gen.py).
2. **Planner** — a vision call that, given the selected per‑task system
   prompt and an overhead grid image, returns the route(s).

Publishes:

* **`/grid_path`** – the selected route(s) (JSON object of per-robot grid-cell labels).

The task type selects both the system prompt (`prompt_gen.PROMPT_BUILDERS`)
and the map overlay (`map_gen.MAP_BUILDERS`).

No chat history is stored – each call is stateless.
"""

from __future__ import annotations

import json
import re

import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from openai import OpenAI

from node_Executive_API.prompt_gen import (
    EXECUTIVE_SYSTEM_PROMPT,
    PROMPT_BUILDERS,
    CLASSIFIER_SYSTEM_PROMPT,
)
from node_Executive_API.map_gen import MAP_BUILDERS, run_segmentation

# ----------------------------------------------------------------------
# Node definition
# ----------------------------------------------------------------------

class ExecutiveApiNode(Node):
    """TurtleBot high‑level path planner publishing to /grid_path."""

    def __init__(self) -> None:
        super().__init__('executive_api_node')

        # ---------- ROS parameters ----------
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model', 'gpt-4o')                   # vision-capable planner model
        self.declare_parameter('nav_prompt_topic', '/nav/prompt')  # ⇐ input
        self.declare_parameter('exec_path_topic',  '/grid_path')     # ⇐ output
        self.declare_parameter('camera_image_topic', '/camera_image')
        self.declare_parameter('camera_info_topic', '/ids_overhead/camera_info')
        self.declare_parameter('robot_names', ['raph', 'donnie'])  # roster the planner must cover
        self.declare_parameter('temperature', 0.0)               # 0 = deterministic
        # Replanning strategy: "static" plans once per prompt; "dynamic" re-runs the same prompt
        # at replan_period seconds until a new prompt arrives. Future modes (event-driven, …) add
        # their own trigger that also calls _run_plan() — the VLM call body is never duplicated.
        self.declare_parameter('replan_mode', 'static')          # "static" | "dynamic"
        self.declare_parameter('replan_period', 120)            # seconds between dynamic replans
        # Camera calibration key and occupancy grid resolution (passed to map builders via self).
        self.declare_parameter('camera', 'gazebo')              # "gazebo" | "lab_test"
        self.declare_parameter('resolution', 0.05)              # m / occupancy cell

        self.robot_names: list[str] = list(self.get_parameter('robot_names').value)
        self.api_key: str   = self.get_parameter('openai_api_key').value
        self.model: str     = self.get_parameter('model').value
        self.temperature: float = self.get_parameter('temperature').value
        self.nav_prompt_topic: str = self.get_parameter('nav_prompt_topic').value
        self.exec_path_topic: str   = self.get_parameter('exec_path_topic').value
        self.camera_image_topic: str = self.get_parameter('camera_image_topic').value
        self.camera_info_topic: str = self.get_parameter('camera_info_topic').value
        self.replan_mode: str = self.get_parameter('replan_mode').value
        self.replan_period: float = float(self.get_parameter('replan_period').value)
        self.camera_name: str = self.get_parameter('camera').get_parameter_value().string_value
        self.resolution: float = float(self.get_parameter('resolution').value)

        # ---------- OpenAI client ----------
        self.client = OpenAI(api_key=self.api_key or None)
        if not self.api_key:
            self.get_logger().warn('OpenAI API key not set — planner disabled.')

        # ---------- Camera state ----------
        # Live overhead camera inputs read by the map builders in map_gen.py.
        self.bridge = CvBridge()
        self.camera_image: Image | None = None
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        # ---------- ROS pubs/subs ----------
        self.path_pub   = self.create_publisher(String, self.exec_path_topic, 1)
        self.prompt_sub = self.create_subscription(
            String, self.nav_prompt_topic, self.prompt_callback, 1
        )
        self.create_subscription(Image, self.camera_image_topic, self._camera_image_cb, 1)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)

        # ---------- Replan trigger state ----------
        # current_prompt holds the latest operator goal (already suffixed with '\n\njson'); the
        # dynamic-mode timer re-plans against whatever this currently holds, so "replan unless the
        # prompt changed" needs no extra bookkeeping — a new prompt simply overwrites it.
        self.current_prompt: str | None = None
        # Task type chosen by the classifier for the current prompt; drives both the system prompt
        # (PROMPT_BUILDERS) and the map overlay (MAP_BUILDERS).
        self.current_task_type: str | None = None
        self.pix_labels: np.ndarray | None = None
        self._timer = None
        if self.replan_mode not in ('static', 'dynamic'):
            self.get_logger().warn(
                f"Unknown replan_mode '{self.replan_mode}'; defaulting to 'static'.")
            self.replan_mode = 'static'
        if self.replan_mode == 'dynamic':
            self._timer = self.create_timer(self.replan_period, self._on_timer)
            self.get_logger().info(
                f"[exec] dynamic replanning every {self.replan_period:.1f}s")
        else:
            self.get_logger().info("[exec] static planning (one plan per prompt)")


    # ------------------------------------------------------------------
    # Callback: handle incoming prompt and plan
    # ------------------------------------------------------------------
    def prompt_callback(self, msg: String) -> None:
        """Store the new goal and plan once immediately.

        Thin trigger: the actual VLM planning lives in _run_plan(). In dynamic mode the timer
        re-runs _run_plan() on the stored prompt at a fixed period; here we (re)start that timer so
        the period is measured from the latest prompt.
        """
        self.get_logger().info(f"[exec] received /nav/prompt: {msg.data!r}")
        self.current_prompt = msg.data + '\n\njson'

        # Classify once per new prompt (not per replan tick): the task type doesn't change between
        # dynamic replans of the same instruction, so we stash it and reuse it in _run_plan().
        self.current_task_type = self._classify_task(msg.data)
        if self.current_task_type is None:
            self.get_logger().error("[exec] could not classify task type — skipping plan.")
            return
        self.get_logger().info(f"[exec] task type: {self.current_task_type}")

        if self.replan_mode == 'dynamic' and self._timer is not None:
            self._timer.reset()  # restart the period from this prompt

        self._run_plan()

    # ------------------------------------------------------------------
    # Timer callback: dynamic-mode periodic replan
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        """Re-plan against the current prompt. No-op until a prompt has been received."""
        if self.current_prompt is None:
            return
        self.get_logger().info("[exec] dynamic replan tick")
        self._run_plan()

    # ------------------------------------------------------------------
    # Core: build snapshot, call the VLM, publish /grid_path
    # ------------------------------------------------------------------
    def _run_plan(self) -> None:
        if self.current_prompt is None:
            return

        if self.current_task_type is None:
            # Classification failed (or hasn't happened). No task type → no prompt/map to pick.
            self.get_logger().error('No task type classified — cannot plan.')
            return

        if not self.api_key:
            self.get_logger().error('API key missing — cannot plan.')
            return

        # NOTE: single-threaded executor — the VLM call below blocks this node's callbacks
        # (incl. /nav/prompt) until it returns. A timer cannot re-enter while a plan is in
        # flight, so replans never overlap. Acceptable for this use.
        try:
            # Run CLIPSeg once per replan tick: stores node.pix_labels and writes _OCC_FILE so
            # translate.py can read the occupancy without running its own CLIPSeg instance.
            run_segmentation(self)

            # Resolve prompt + map from the classifier's task type. Both are scaffolding for now:
            # an empty per-task prompt falls back to EXECUTIVE_SYSTEM_PROMPT, and every map builder
            # currently returns the Battleship grid — so behavior is unchanged until they're filled.
            system_prompt = PROMPT_BUILDERS[self.current_task_type]()
            map_b64 = MAP_BUILDERS[self.current_task_type](self)

            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": self.current_prompt },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{map_b64}",
                            },
                        ],
                    }
                ],
            )
            reply_json: str = response.output_text.strip()

            # ------------- safe JSON parse -------------
            # Tolerate ```json fences / surrounding prose the model sometimes adds.
            result = self._parse_json_reply(reply_json)
            if result is None:
                self.get_logger().error(
                    f'Malformed JSON from model - cannot parse. Raw reply: {reply_json!r}')
                return

            # Validate + publish paths — must be a dict of {robot_name: [grid labels]} (the
            # translator routes each robot's list to /<robot>/waypoint_path). Fail loudly if the
            # model returns another shape.
            paths = result.get('paths', {})
            if not isinstance(paths, dict):
                self.get_logger().error(
                    f"Model 'paths' is not an object (got {type(paths).__name__}: {paths!r}); "
                    "not publishing.")
                return
            bad = [n for n, p in paths.items() if not isinstance(p, list)]
            if bad:
                self.get_logger().error(
                    f"Model 'paths' has non-list routes for {bad}; not publishing.")
                return
            missing = [n for n in self.robot_names if n not in paths]
            if missing:
                self.get_logger().warn(
                    f"Model omitted a route for {missing}; those robots will not move.")
            path_msg = String()
            path_msg.data = json.dumps(paths)
            self.path_pub.publish(path_msg)

            # Log through ROS logger (INFO)
            self.get_logger().info(f"[exec] Planned paths: {path_msg.data}")
            analysis = result.get('analysis', '')
            if analysis:
                self.get_logger().info(f"[exec] analysis: {analysis}")

        except Exception as exc:
            self.get_logger().error(f'Path‑planning failed: {exc}')

    # ------------------------------------------------------------------
    # Camera callbacks
    # ------------------------------------------------------------------
    def _camera_image_cb(self, msg: Image) -> None:
        self.camera_image = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    # ------------------------------------------------------------------
    # Helper: classify the instruction into a task type (first, text-only LLM call)
    # ------------------------------------------------------------------
    def _classify_task(self, instruction: str) -> str | None:
        """Return one of PROMPT_BUILDERS' keys, or None if the classifier can't produce a valid one.

        A None result means "don't plan this prompt" — there is no default task type to fall back on.
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=CLASSIFIER_SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": instruction}],
                    }
                ],
            )
            result = self._parse_json_reply(response.output_text.strip())
            task_type = (result or {}).get('task_type')
            if task_type in PROMPT_BUILDERS:
                return task_type
            self.get_logger().warn(f"Classifier returned unknown task_type {task_type!r}.")
        except Exception as exc:
            self.get_logger().warn(f"Classifier call failed ({exc}).")
        return None

    # ------------------------------------------------------------------
    # Helper: parse the model reply as JSON, tolerating fences / extra prose
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_reply(text: str) -> dict | None:
        """Return the JSON object from the model reply, or None if unparseable.

        Tries a direct parse first, then strips a ```json ... ``` code fence, then falls
        back to the first {...} block — covers the common ways models wrap strict JSON.
        """
        # 1) direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2) ```json ... ``` or ``` ... ``` fenced block
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        # 3) first {...} block anywhere in the text
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
        return None


# ----------------------------------------------------------------------
# Entry‑point with single‑init guard
# ----------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:  # pragma: no cover
    """Spin the node, guarding against duplicate `rclpy.init()` calls."""
    already_init = rclpy.ok()
    if not already_init:
        rclpy.init(args=args)

    node = ExecutiveApiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if not already_init:
            rclpy.shutdown()

if __name__ == '__main__':
    main()
