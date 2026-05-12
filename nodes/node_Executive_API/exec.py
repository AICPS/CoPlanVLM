#!/usr/bin/env python3
"""
Executive API Node — Path‑Planning Edition (Responses API + o4‑mini)
===================================================================
Receives a JSON snapshot of the TurtleBot4’s Battleship‑grid world plus
an operator goal prompt, forwards it to OpenAI’s **Responses API**
(`o4‑mini`), and publishes:

* **`/path`** – the selected path (JSON list of grid coordinates).
* **`/nav/status`** – a compact JSON object consumed by the triage node.

Unlike earlier versions, **all task instructions now live on the
OpenAI server**.  The node simply forwards the snapshot (as `input=`)
and expects a strict‑JSON reply containing `path` and either
`analysis` *(on success)* or an error description *(on failure)*.

No chat history is stored – each call is stateless.
"""

from __future__ import annotations

import json
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from std_msgs.msg import String, Bool
from openai import OpenAI
import base64
from time import sleep

# ----------------------------------------------------------------------
# Node definition
# ----------------------------------------------------------------------

class ExecutiveApiNode(Node):
    """TurtleBot high‑level path planner publishing to /path."""

    def __init__(self) -> None:
        super().__init__('executive_api_node')

        # ---------- ROS parameters ----------
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('nav_prompt_topic', '/nav/prompt')  # ⇐ input
        self.declare_parameter('exec_path_topic',  '/path')     # ⇐ output
        self.declare_parameter('exec_status_topic', '/nav/status')   # ⇐ output
        self.declare_parameter('need_map_topic', '/need_map')   # ⇐ output
        self.declare_parameter('map_path', '')


        self.api_key: str   = self.get_parameter('openai_api_key').value
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
        self.prompt = msg.data + '\n\njson'

        if not self.api_key:
            self._publish_status(False, 'API key missing')
            return

        try:
            self._need_map_pub(True)
            sleep(5)
            self.map = self._encode_image(self.map_path)

            response = self.client.responses.create(
                prompt={
                    "id": "pmpt_685963df1d0081958a7bbfdd74bdae590a18ad364ec2d535",
                    "version": "5"
                },
                # prompt={
                #     "id": "pmpt_68d6bfb538708195a919d8d93d58e9b20c3d5460618192f7",
                #     "version": "11"
                # },
                # prompt={
                #     "id": "pmpt_6a0261312ee881939341b343263b23280651a298466c79a0",
                #     "version": "4"
                # },
                # model="gpt-5"
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
            try:
                result = json.loads(reply_json)
            except json.JSONDecodeError:
                self.get_logger().error('Malformed JSON from model - cannot parse path.')
                self._publish_status(False, 'malformed JSON from model')
                return

            # Publish path
            path_msg = String()
            path_msg.data = json.dumps(result.get('path', []))            
            self.path_pub.publish(path_msg)

            # Log through ROS logger (INFO)
            self.get_logger().info(f"[exec] Planned path: {path_msg.data}")

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
