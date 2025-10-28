#!/usr/bin/env python3
"""
response_tts_node.py – ROS 2 subscriber that speaks /response using OpenAI TTS (.wav stream)
==========================================================================================
• Subscribes to **/response** (std_msgs/String) and voices each message.
• Streams **.wav** audio via `with_streaming_response` for near‑instant playback.
• Accepts **openai_api_key** as a ROS 2 launch/parameter (preferred) with an environment‑variable fallback.

Quick Install
-------------
```bash
pip install --upgrade "openai>=1.9.0" sounddevice rclpy
```
(`sounddevice` is used by *LocalAudioPlayer* for cross‑platform playback.)

Example Launch File
-------------------
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="<your_package>",
            executable="response_tts_node",
            name="response_tts_node",
            parameters=[{"openai_api_key": "sk‑...replace‑me..."}],
        )
    ])
```

Notes
-----
• **openai_api_key** parameter overrides `OPENAI_API_KEY` env var.  Either is fine—you can swap keys without restarting the node by using `ros2 param set`.
• Audio is streamed at 24 kHz / 16‑bit mono **WAV**; header arrives first, then chunks.
• One worker thread per utterance ensures ROS callbacks stay responsive.
• Adjust *voice*, *model*, or *instructions* to suit your robot’s persona.
"""

import asyncio
import os
import threading
from typing import Optional

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ResponseTTSNode(Node):
    """Convert text on /response into speech and play it locally."""

    def __init__(self):
        super().__init__("response_tts_node")

        # Declare parameter; empty default means "check environment instead"
        self.declare_parameter("openai_api_key", "")
        self.api_key: Optional[str] = (
            self.get_parameter("openai_api_key").get_parameter_value().string_value
            or os.getenv("OPENAI_API_KEY", "")
        )
        if not self.api_key:
            self.get_logger().fatal(
                "API key missing – supply via launch parameter 'openai_api_key' or env var OPENAI_API_KEY."
            )
            raise SystemExit(1)

        # Instantiate async client once
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Voice style (tweak or expose as params if needed)
        self.instructions = (
            """Affect/personality: A cheerful guide 
            Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.
            Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.
            Pause: Brief, purposeful pauses after key instructions (e.g., "cross the street" and "turn right") to allow time for the listener to process the information and follow along.
            Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey."""
        )

        # Subscribe to topic
        self.create_subscription(String, "/response", self._on_text, 10)
        
        
        self.get_logger().info("✓")

    # ------------------------------------------------------------------
    # ROS 2 callback – spawn a thread so we never block executor
    # ------------------------------------------------------------------
    def _on_text(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().debug(f"Queued {len(text)} chars for TTS → .wav")
        threading.Thread(target=lambda: asyncio.run(self._speak_async(text)), daemon=True).start()

    # ------------------------------------------------------------------
    # Async TTS
    # ------------------------------------------------------------------
    async def _speak_async(self, text: str) -> None:
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",  # or newer/larger as available
                voice="echo",
                input=text,
                instructions=self.instructions,
                response_format="wav",   # stream .wav (header + chunks)
            ) as response:
                await LocalAudioPlayer().play(response)
        except Exception as exc:
            self.get_logger().error(f"TTS error: {exc}")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = ResponseTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
