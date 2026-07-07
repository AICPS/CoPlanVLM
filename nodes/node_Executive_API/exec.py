#!/usr/bin/env python3
"""
Executive API Node — Path‑Planning Edition (Responses API + o4‑mini)
===================================================================
Receives a JSON snapshot of the TurtleBot4's Battleship‑grid world plus
an operator goal prompt, forwards it to OpenAI's **Responses API**
(`o4‑mini`), and publishes:

* **`/grid_path`** – the selected route(s) (JSON object of per-robot grid-cell labels).
* **`/nav/status`** – a compact JSON object consumed by the triage node.

Unlike earlier versions, **all task instructions now live on the
OpenAI server**.  The node simply forwards the snapshot (as `input=`)
and expects a strict‑JSON reply containing `path` and either
`analysis` *(on success)* or an error description *(on failure)*.

No chat history is stored – each call is stateless.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from openai import OpenAI
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

from node_Executive_API.prompt import EXECUTIVE_SYSTEM_PROMPT

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
        self.declare_parameter('exec_status_topic', '/nav/status')   # ⇐ output
        self.declare_parameter('camera_image_topic', '/camera_image')
        self.declare_parameter('camera_info_topic', '/ids_overhead/camera_info')
        self.declare_parameter('robot_names', ['raph', 'donnie'])  # roster the planner must cover
        self.declare_parameter('temperature', 0.0)               # 0 = deterministic
        # Replanning strategy: "static" plans once per prompt; "dynamic" re-runs the same prompt
        # at replan_period seconds until a new prompt arrives. Future modes (event-driven, …) add
        # their own trigger that also calls _run_plan() — the VLM call body is never duplicated.
        self.declare_parameter('replan_mode', 'static')          # "static" | "dynamic"
        self.declare_parameter('replan_period', 15.0)            # seconds between dynamic replans


        self.robot_names: list[str] = list(self.get_parameter('robot_names').value)
        self.api_key: str   = self.get_parameter('openai_api_key').value
        self.model: str     = self.get_parameter('model').value
        self.temperature: float = self.get_parameter('temperature').value
        self.nav_prompt_topic: str = self.get_parameter('nav_prompt_topic').value
        self.exec_path_topic: str   = self.get_parameter('exec_path_topic').value
        self.exec_status_topic: str = self.get_parameter('exec_status_topic').value
        self.camera_image_topic: str = self.get_parameter('camera_image_topic').value
        self.camera_info_topic: str = self.get_parameter('camera_info_topic').value
        self.replan_mode: str = self.get_parameter('replan_mode').value
        self.replan_period: float = float(self.get_parameter('replan_period').value)

        # ---------- OpenAI client ----------
        self.client = OpenAI(api_key=self.api_key or None)
        if not self.api_key:
            self.get_logger().warn('OpenAI API key not set — planner disabled.')

        # ---------- Camera / map state ----------
        self.bridge = CvBridge()
        self.camera_image: Image | None = None
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        pkg_dir = get_package_share_directory('talking-turtle')
        grid_path = os.path.join(pkg_dir, 'config', 'transparent_grid.png')
        self.grid_img = PILImage.open(grid_path).convert("RGBA")

        # ---------- ROS pubs/subs ----------
        self.path_pub   = self.create_publisher(String, self.exec_path_topic, 10)
        self.status_pub = self.create_publisher(String, self.exec_status_topic, 10)
        self.prompt_sub = self.create_subscription(
            String, self.nav_prompt_topic, self.prompt_callback, 10
        )
        self.create_subscription(Image, self.camera_image_topic, self._camera_image_cb, 1)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)

        # ---------- Replan trigger state ----------
        # current_prompt holds the latest operator goal (already suffixed with '\n\njson'); the
        # dynamic-mode timer re-plans against whatever this currently holds, so "replan unless the
        # prompt changed" needs no extra bookkeeping — a new prompt simply overwrites it.
        self.current_prompt: str | None = None
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

        self.get_logger().info("✓")


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
    # Core: build snapshot, call the VLM, publish /grid_path + status
    # ------------------------------------------------------------------
    def _run_plan(self) -> None:
        if self.current_prompt is None:
            return

        if not self.api_key:
            self._publish_status(False, 'API key missing')
            return

        # NOTE: single-threaded executor — the VLM call below blocks this node's callbacks
        # (incl. /nav/prompt) until it returns. A timer cannot re-enter while a plan is in
        # flight, so replans never overlap. Acceptable for this use.
        try:
            map_b64 = self._generate_map()

            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=EXECUTIVE_SYSTEM_PROMPT,
                # --- Switched to the in-repo prompt (node_Executive_API/prompt.py). ---
                # PREVIOUSLY ACTIVE server-stored prompt (uncomment to restore exactly):
                # prompt={"id": "pmpt_685963df1d0081958a7bbfdd74bdae590a18ad364ec2d535", "version": "5"},
                # Other (already-inactive) stored-prompt alternates that were here before:
                # prompt={"id": "pmpt_68d6bfb538708195a919d8d93d58e9b20c3d5460618192f7", "version": "11"},
                # prompt={"id": "pmpt_6a0261312ee881939341b343263b23280651a298466c79a0", "version": "4"},
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
                self._publish_status(False, 'malformed JSON from model')
                return

            # Validate + publish paths — must be a dict of {robot_name: [grid labels]} (the
            # translator routes each robot's list to /<robot>/waypoint_path). Fail loudly if the
            # model returns another shape.
            paths = result.get('paths', {})
            if not isinstance(paths, dict):
                self.get_logger().error(
                    f"Model 'paths' is not an object (got {type(paths).__name__}: {paths!r}); "
                    "not publishing.")
                self._publish_status(False, 'planner returned a non-object paths')
                return
            bad = [n for n, p in paths.items() if not isinstance(p, list)]
            if bad:
                self.get_logger().error(
                    f"Model 'paths' has non-list routes for {bad}; not publishing.")
                self._publish_status(False, 'planner returned a non-list route')
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

            # Inform triage of success with analysis
            self._publish_status(True, result.get('analysis', ''))

        except Exception as exc:
            self.get_logger().error(f'Path‑planning failed: {exc}')
            self._publish_status(False, str(exc))

    # ------------------------------------------------------------------
    # Camera callbacks
    # ------------------------------------------------------------------
    def _camera_image_cb(self, msg: Image) -> None:
        self.camera_image = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    # ------------------------------------------------------------------
    # Helper: generate grid-overlay map as base64 PNG (in memory)
    # ------------------------------------------------------------------
    def _generate_map(self) -> str:
        if self.camera_image is None:
            raise RuntimeError("No camera image received yet — is the camera publishing?")
        cv_img = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding='rgba8')
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            cv_img = cv2.undistort(cv_img, self.camera_matrix, self.dist_coeffs)
        pil_img = PILImage.fromarray(cv_img).convert("RGBA")
        grid_resized = self.grid_img.resize(pil_img.size)
        combined = PILImage.alpha_composite(pil_img, grid_resized)
        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # ------------------------------------------------------------------
    # Helper: publish status
    # ------------------------------------------------------------------
    def _publish_status(self, success: bool, detail: str | None = None) -> None:
        status_obj = {'source': 'exec', 'success': success}
        if success:
            status_obj['analysis'] = detail or ''
        else:
            status_obj['reason'] = detail or 'unknown error'

        status_msg = String()
        status_msg.data = json.dumps(status_obj)
        self.status_pub.publish(status_msg)

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
