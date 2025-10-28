#!/usr/bin/env python3
"""
Speech‑to‑Text node – **Joystick Push‑to‑Talk**
=============================================

Hold a button on any ROS‑published **/joy** joystick to start recording
and release it to stop. The node transcribes with Whisper and publishes
text on **/user_text** (`std_msgs/String`).

* Default PTT button index: **0** (usually the “A” button on Xbox/PS
  controllers). Change with `-p ptt_button:=2`, etc.
* Audio stream runs continuously; frames are saved only while the
  button is held.

---
### Quick install
```bash
pip install sounddevice numpy openai rclpy
```

### Run
```bash
ros2 run talking_turtle node_Listener               # button 0 as PTT
ros2 run talking_turtle node_Listener --ros-args -p ptt_button:=4
```

Ensure another node publishes `sensor_msgs/msg/Joy` messages on
**/joy** (e.g., `joy_node`, SDL Joystick ROS2 driver, or a gamepad
bridge).
"""

import queue
import tempfile
import threading
import wave
from pathlib import Path
from typing import List

import numpy as np
import rclpy
import sounddevice as sd
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16_000
CHANNELS = 1
MODEL_NAME = 'whisper-1'
DEFAULT_PTT_BUTTON = 0     # index in Joy.buttons[]
# ----------------------------------------

class SpeechToTextJoy(Node):
    """ROS 2 node: joystick‑controlled push‑to‑talk → Whisper."""

    def __init__(self):
        super().__init__('speech_to_text_joy')

        # PTT button index parameter
        self.ptt_button = self.declare_parameter('ptt_button', DEFAULT_PTT_BUTTON).value
        api_key = self.declare_parameter('openai_api_key', '').value

        # ROS publisher and subscriber
        self.pub  = self.create_publisher(String, '/user_text', 10)
        self.sub  = self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        # OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Audio buffers
        self.recording = False
        self.frames: List[np.ndarray] = []

        # Thread‑safe queue from audio callback
        self.q: queue.Queue[np.ndarray] = queue.Queue()

        # Start audio stream (runs continuously)
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE,
                                     channels=CHANNELS,
                                     dtype='float32',
                                     callback=self._audio_cb)
        self.stream.start()

        # Worker to accumulate frames
        threading.Thread(target=self._worker, daemon=True).start()
        
        self.get_logger().info("✓")
        

    # ---------- joystick callback ----------
    def _joy_cb(self, msg: Joy):
        pressed = 0 <= self.ptt_button < len(msg.buttons) and msg.buttons[self.ptt_button] != 0
        if pressed and not self.recording:
            self._start_recording()
        elif not pressed and self.recording:
            self._stop_recording()

    # ---------- audio callback ----------
    def _audio_cb(self, indata, frames, time, status):
        if status:
            self.get_logger().warning(f'Stream status: {status}')
        if self.recording:
            self.q.put(indata[:, 0].copy())

    # ---------- worker ----------
    def _worker(self):
        while rclpy.ok():
            frame = self.q.get()
            if frame is None:
                break
            self.frames.append(frame)

    # ---------- control helpers ----------
    def _start_recording(self):
        self.frames.clear()
        self.recording = True
        self.get_logger().info('🎙️  Recording… (release button to end)')

    def _stop_recording(self):
        self.recording = False
        self.get_logger().info('⏹️  Processing…')
        if not self.frames:
            self.get_logger().info('🕳️  (no audio captured)')
            return
        audio_np = np.concatenate(self.frames)
        self.frames = []
        threading.Thread(target=self._transcribe, args=(audio_np,)).start()

    # ---------- Whisper ----------
    def _transcribe(self, audio_np: np.ndarray):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
        try:
            with open(wav_path, 'rb') as f:
                resp = self.client.audio.transcriptions.create(model=MODEL_NAME, file=f)
            text = resp.text.strip()
            if text:
                msg = String(); msg.data = text; self.pub.publish(msg)
                self.get_logger().info(f'📤  /user_text → {text}')
            else:
                self.get_logger().info('🕳️  (silence)')
        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
        finally:
            Path(wav_path).unlink(missing_ok=True)

    # ---------- cleanup ----------
    def destroy_node(self):
        if self.stream:
            self.stream.stop(); self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextJoy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
