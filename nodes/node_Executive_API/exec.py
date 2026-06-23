#!/usr/bin/env python3
"""
Executive API Node — Path‑Planning Edition (Responses API + o4‑mini)
===================================================================
Receives a JSON snapshot of the TurtleBot4’s Battleship‑grid world plus
an operator goal prompt, forwards it to OpenAI’s **Responses API**
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

import json
import re
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from std_msgs.msg import String, Bool
from openai import OpenAI
import base64
from time import sleep

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
        self.declare_parameter('need_map_topic', '/need_map')   # ⇐ output
        self.declare_parameter('map_path', '')
        self.declare_parameter('robot_names', ['raph', 'raph2'])  # roster the planner must cover


        self.robot_names: list[str] = list(self.get_parameter('robot_names').value)
        self.api_key: str   = self.get_parameter('openai_api_key').value
        self.model: str     = self.get_parameter('model').value
        self.nav_prompt_topic: str = self.get_parameter('nav_prompt_topic').value
        self.exec_path_topic: str   = self.get_parameter('exec_path_topic').value
        self.exec_status_topic: str = self.get_parameter('exec_status_topic').value
        self.need_map_topic: str = self.get_parameter('need_map_topic').value
        self.map_path: str   = self.get_parameter('map_path').value


        # ---------- OpenAI client ----------
        self.client = OpenAI(api_key=self.api_key or None)
        if not self.api_key:
            self.get_logger().warn('OpenAI API key not set — planner disabled.')

        # ---------- ROS pubs/subs ----------
        self.path_pub   = self.create_publisher(String, self.exec_path_topic, 10)
        self.need_map_pub   = self.create_publisher(Bool, self.need_map_topic, 10)
        self.status_pub = self.create_publisher(String, self.exec_status_topic, 10)
        self.prompt_sub = self.create_subscription(
            String, self.nav_prompt_topic, self.prompt_callback, 10
        )

        self.get_logger().info("✓")

        
    # ------------------------------------------------------------------
    # Callback: handle incoming prompt and plan
    # ------------------------------------------------------------------
    def prompt_callback(self, msg: String) -> None:
        self.get_logger().info(f"[exec] received /nav/prompt: {msg.data!r}")
        self.prompt = msg.data + '\n\njson'

        if not self.api_key:
            self._publish_status(False, 'API key missing')
            return

        try:
            self._need_map_pub(True)
            sleep(5)
            self.map = self._encode_image(self.map_path)

            response = self.client.responses.create(
                model=self.model,
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
                            { "type": "input_text", "text": self.prompt },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{self.map}",
                            },
                        ],
                    }
                ],
            )
            self._need_map_pub(False)
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


    def _need_map_pub(self, need: bool) -> None:
        need_msg = Bool()
        need_msg.data = need
        self.need_map_pub.publish(need_msg)

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

    # ------------------------------------------------------------------
    # Function to encode the image
    # ------------------------------------------------------------------
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


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
