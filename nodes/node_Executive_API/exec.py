#!/usr/bin/env python3
"""
Executive API Node — Adaptive Path‑Planning Edition (Responses API)
===================================================================
Receives an operator goal prompt on `/nav/prompt`, and plans in two LLM
calls, both constrained by strict JSON schemas (structured outputs) so the
reply shape is guaranteed by decoding rather than requested in prose:

1. **Classifier** — a text‑only call tags the instruction with a task
   type (`nav2point` / `maneuver` / `coverage`; see prompt_gen.py). The
   `classifier_schema()` enum makes any other answer impossible.
2. **Planner** — a vision call that, given the selected per‑task system
   prompt and the `marked_obs` overlay image, returns the route(s) under
   `route_schema()`. Chain‑of‑thought is always on: the schema declares one
   required field per reasoning step (prompt_gen.COT_FIELDS for the task
   type) ahead of the routes, so the model works through every step, in
   order, BEFORE emitting routes — and cannot silently skip one.

Publishes:

* **`/vlm_plan`** – the selected plan: a JSON wrapper
  ``{"planner": "astar"|"coverage", "routes": {robot_name: [labels]}}`` naming the downstream
  planner and each robot's label list.

The task type (controller) drives `prompt_gen.generate_prompt`, which assembles the system prompt and
renders the matching overlay image together, and selects the downstream planner (`TASK_ROUTING`).

This mirrors scripts/test_pipeline.py exactly — both import the prompts, schemas and routing from
prompt_gen, so what is validated offline is what the robots execute.

No chat history is stored – each call is stateless.
"""

from __future__ import annotations

import json
import re

import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from openai import OpenAI

import debug_io

from node_Executive_API.prompt_gen import (
    generate_prompt,
    classifier_schema,
    route_schema,
    CONTROLLERS,
    TASK_ROUTING,
    PRODUCTION_MAP_OVERLAY,
    classifier_prompt,
    reasoning_text,
)
from node_Executive_API.map_gen import run_segmentation, _node_to_pil

# ----------------------------------------------------------------------
# Node definition
# ----------------------------------------------------------------------

class ExecutiveApiNode(Node):
    """TurtleBot high‑level path planner publishing to /vlm_plan."""

    def __init__(self) -> None:
        super().__init__('executive_api_node')

        # ---------- ROS parameters ----------
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model', 'gpt-4o')                   # vision-capable planner model
        self.declare_parameter('nav_prompt_topic', '/nav/prompt')  # ⇐ input
        self.declare_parameter('vlm_plan_topic',  '/vlm_plan')       # ⇐ output
        self.declare_parameter('camera_image_topic', '/camera_image')
        self.declare_parameter('camera_info_topic', '/ids_overhead/camera_info')
        self.declare_parameter('robot_names', ['raph', 'donnie'])  # roster the planner must cover
        self.declare_parameter('temperature', 0.0)               # 0 = deterministic
        # Replanning strategy: "static" plans once per prompt; "dynamic" re-runs the same prompt
        # at replan_period seconds until a new prompt arrives. Future modes (event-driven, …) add
        # their own trigger that also calls _run_plan() — the VLM call body is never duplicated.
        self.declare_parameter('replan_mode', 'static')          # "static" | "dynamic"
        self.declare_parameter('replan_period', 120.0)          # seconds between dynamic replans
        # Camera calibration key (passed to map builders via self).
        self.declare_parameter('camera', 'gazebo')              # "gazebo" | "lab_test"
        # Where to persist the VLM exchange (vlm_prompt.txt / vlm_response.txt). Empty = disabled,
        # the same convention node_Path_Translator uses for its debug_dir. Launch files point this
        # at the same per-environment directory the translator writes its images to.
        self.declare_parameter('debug_dir', '')

        self.robot_names: list[str] = list(self.get_parameter('robot_names').value)
        self.api_key: str   = self.get_parameter('openai_api_key').value
        self.model: str     = self.get_parameter('model').value
        self.temperature: float = self.get_parameter('temperature').value
        self.nav_prompt_topic: str = self.get_parameter('nav_prompt_topic').value
        self.vlm_plan_topic: str   = self.get_parameter('vlm_plan_topic').value
        self.camera_image_topic: str = self.get_parameter('camera_image_topic').value
        self.camera_info_topic: str = self.get_parameter('camera_info_topic').value
        self.replan_mode: str = self.get_parameter('replan_mode').value
        self.replan_period: float = float(self.get_parameter('replan_period').value)
        self.camera_name: str = self.get_parameter('camera').get_parameter_value().string_value
        self.debug_dir: str = self.get_parameter('debug_dir').get_parameter_value().string_value

        # ---------- OpenAI client ----------
        self.client = OpenAI(api_key=self.api_key or None)
        if not self.api_key:
            self.get_logger().warn('OpenAI API key not set — planner disabled.')

        # ---------- Camera state ----------
        # Live overhead camera inputs read by the map builders in map_gen.py.
        self.bridge = CvBridge()
        self.camera_image: Image | None = None
        # Snapshot of robot_poses taken when camera_image was stored, so occupancy clearing frees the
        # robots' footprints at their positions IN that frame (not a later, moved pose).
        self.camera_image_poses: dict[str, tuple[float, float] | None] = {}
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        # ---------- Robot pose state ----------
        # Latest NED (x, y) per robot, passed to map builders to overlay robot markers.
        self.robot_poses: dict[str, tuple[float, float] | None] = {
            name: None for name in self.robot_names}

        # ---------- ROS pubs/subs ----------
        self.path_pub   = self.create_publisher(String, self.vlm_plan_topic, 1)
        self.prompt_sub = self.create_subscription(
            String, self.nav_prompt_topic, self.prompt_callback, 1
        )
        self.create_subscription(Image, self.camera_image_topic, self._camera_image_cb, 1)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)
        pose_qos = QoSProfile(depth=1)
        pose_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        for name in self.robot_names:
            self.create_subscription(
                PoseStamped, f'/{name}/ned/pose_stamped',
                self._make_pose_cb(name), pose_qos)

        # ---------- Replan trigger state ----------
        # current_prompt holds the latest operator goal (already suffixed with '\n\njson'); the
        # dynamic-mode timer re-plans against whatever this currently holds, so "replan unless the
        # prompt changed" needs no extra bookkeeping — a new prompt simply overwrites it.
        self.current_prompt: str | None = None
        # Task type (controller) chosen by the classifier for the current prompt; passed to
        # generate_prompt, which builds the system prompt + overlay image and picks the planner.
        self.current_task_type: str | None = None
        # Occupancy state written by map_gen.run_segmentation each replan tick: pix_labels (raw
        # segmentation), plus the single inflated planning grid + meta the nav2point overlay filters
        # against. None until the first segmentation runs.
        self.pix_labels: np.ndarray | None = None
        self.occ_grid: np.ndarray | None = None
        self.occ_meta: dict | None = None
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
        self.current_prompt = msg.data

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
    # Core: build snapshot, call the VLM, publish /vlm_plan
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

        # No overhead frame yet -> nothing to segment or render. Bail out explicitly: without this
        # run_segmentation() would no-op (leaving occ_grid None, so the marked_obs overlay would
        # report "(none)" blocked cells) and _node_to_pil() would then raise on the None image.
        if self.camera_image is None:
            self.get_logger().warn(
                f"No overhead frame received on '{self.camera_image_topic}' yet — cannot plan. "
                "Is the camera (or the sim) publishing?")
            return

        # The classifier's task type selects both the model output key and the downstream planner.
        # Resolved BEFORE the call because result_key is part of the reply schema below.
        result_key, planner = TASK_ROUTING[self.current_task_type]

        # NOTE: single-threaded executor — the VLM call below blocks this node's callbacks
        # (incl. /nav/prompt) until it returns. A timer cannot re-enter while a plan is in
        # flight, so replans never overlap. Acceptable for this use.
        try:
            # Run CLIPSeg once per replan tick: stores node.pix_labels and writes _OCC_FILE so
            # translator_node.py can read the occupancy without running its own CLIPSeg instance.
            run_segmentation(self)

            # generate_prompt assembles the system prompt AND renders the matching overlay image,
            # so the two always agree. cot=True appends the controller's chain-of-thought scaffold;
            # the schema below makes the model fill in its "reasoning" before the routes.
            # robot_names is the SAME roster passed to route_schema below, so the prompt names
            # exactly the robots the reply schema will accept.
            system_prompt, map_b64 = generate_prompt(
                self.current_prompt, self.current_task_type, PRODUCTION_MAP_OVERLAY,
                pil_img=_node_to_pil(self), occ_grid=self.occ_grid, occ_meta=self.occ_meta,
                camera=self.camera_name, robot_poses=self.robot_poses, cot=True,
                robot_names=self.robot_names)

            # Structured outputs: the reply can only be the fixed {reasoning, result_key{robots…}}
            # shape, so invented keys / missing robots are impossible at decode time. Waypoints are
            # left unconstrained (no allowed_labels enum) — the planner projects picks onto free
            # cells downstream.
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{map_b64}",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "route_plan",
                        "strict": True,
                        "schema": route_schema(result_key, self.robot_names, cot=True,
                                               controller=self.current_task_type),
                    }
                },
            )
            reply_json: str = response.output_text.strip()

            # Persist the exact exchange so a live run can be reconstructed afterwards (the offline
            # harnesses save the same files). Never raises — failures go to the ROS log.
            # marks_overlay.png is written from map_b64, so it is byte-for-byte the image the model
            # saw; this node is the only place that image exists in the live pipeline.
            if self.debug_dir:
                warn = self.get_logger().warn
                debug_io.save_vlm_exchange(self.debug_dir, system_prompt, reply_json, on_error=warn)
                debug_io.save_marks_overlay(self.debug_dir, map_b64, on_error=warn)

            # ------------- safe JSON parse -------------
            # Strict mode already guarantees valid JSON; this stays as a cheap guard against an
            # empty reply or a refusal, and tolerates fences if the format is ever relaxed.
            result = self._parse_json_reply(reply_json)
            if result is None:
                self.get_logger().error(
                    f'Malformed JSON from model - cannot parse. Raw reply: {reply_json!r}')
                return

            # Routes must be a dict of {robot_name: [labels]}; the translator routes each robot's
            # list to /<robot>/waypoint_path via the named planner. The schema enforces this, so the
            # checks below only fire if structured output was bypassed. Fail loudly on a bad shape.
            routes = result.get(result_key, {})
            if not isinstance(routes, dict):
                self.get_logger().error(
                    f"Model '{result_key}' is not an object (got {type(routes).__name__}: "
                    f"{routes!r}); not publishing.")
                return
            bad = [n for n, p in routes.items() if not isinstance(p, list)]
            if bad:
                self.get_logger().error(
                    f"Model '{result_key}' has non-list routes for {bad}; not publishing.")
                return
            missing = [n for n in self.robot_names if n not in routes]
            if missing:
                self.get_logger().warn(
                    f"Model omitted a route for {missing}; those robots will not move.")
            path_msg = String()
            path_msg.data = json.dumps({"planner": planner, "routes": routes})
            self.path_pub.publish(path_msg)

            # Log through ROS logger (INFO). Reasoning first — it is what the routes were derived
            # from, and the schema makes the model generate it in that order. The reply carries one
            # field per reasoning step (COT_FIELDS for this task type); reasoning_text joins them.
            reasoning = reasoning_text(result, self.current_task_type)
            if reasoning:
                self.get_logger().info(f"[exec] reasoning:\n{reasoning}")
            # Report the task type as well as the planner: nav2point and maneuver both route to
            # "astar", so the planner name alone does not identify the chosen category — and in
            # dynamic mode the classification line only prints once per prompt, not per replan.
            self.get_logger().info(
                f"[exec] Planned [task={self.current_task_type} -> planner={planner}] "
                f"routes: {path_msg.data}")

        except Exception as exc:
            self.get_logger().error(f'Path‑planning failed: {exc}')

    # ------------------------------------------------------------------
    # Camera callbacks
    # ------------------------------------------------------------------
    def _camera_image_cb(self, msg: Image) -> None:
        self.camera_image = msg
        # Freeze the poses that go with this frame (see camera_image_poses).
        self.camera_image_poses = dict(self.robot_poses)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _make_pose_cb(self, name: str):
        def _cb(msg: PoseStamped) -> None:
            self.robot_poses[name] = (msg.pose.position.x, msg.pose.position.y)
        return _cb

    # ------------------------------------------------------------------
    # Helper: classify the instruction into a task type (first, text-only LLM call)
    # ------------------------------------------------------------------
    def _classify_task(self, instruction: str) -> str | None:
        """Return one of CONTROLLERS, or None if the classifier can't produce a valid one.

        The reply is constrained by classifier_schema(), whose enum over CONTROLLERS makes an
        out-of-vocabulary task type impossible at decode time. The membership check below is kept
        as a cheap guard against an empty reply or a refusal.

        A None result means "don't plan this prompt" — there is no default task type to fall back on.
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=classifier_prompt(self.robot_names),
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": instruction}],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task_classification",
                        "strict": True,
                        "schema": classifier_schema(),
                    }
                },
            )
            result = self._parse_json_reply(response.output_text.strip())
            task_type = (result or {}).get('task_type')
            if task_type in CONTROLLERS:
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
