#!/usr/bin/env python3
"""
Triage API Node — Conversational front‑end (Responses API)
=========================================================
"""
from __future__ import annotations

import json
from collections import deque
from typing import Deque, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI

from node_Triage_API.prompt import TRIAGE_SYSTEM_PROMPT


class TriageApiNode(Node):

    # ---------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__('triage_api_node')

        # Parameters
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model', 'gpt-4o')
        self.declare_parameter('history_size', 20)
        self.declare_parameter('nav_status_topic', '/nav/status')
        self.declare_parameter('nav_prompt_topic', '/nav/prompt')
        self.declare_parameter('response_topic', '/response')
        self.declare_parameter('user_text_topic', '/user_text')

        # Read parameters
        self.api_key = self.get_parameter('openai_api_key').value
        self.model = self.get_parameter('model').value
        history_sz = self.get_parameter('history_size').value
        self.nav_status_topic = self.get_parameter('nav_status_topic').value
        self.nav_prompt_topic = self.get_parameter('nav_prompt_topic').value
        self.response_topic = self.get_parameter('response_topic').value
        self.user_text_topic = self.get_parameter('user_text_topic').value

        # OpenAI client
        self.client = OpenAI(api_key=self.api_key or None)
        if not self.api_key:
            self.get_logger().warn('OpenAI API key not set — triage disabled.')

        # Rolling chat history
        self.chat_history: Deque[dict[str, str]] = deque(maxlen=history_sz)

        # ROS wiring
        self.create_subscription(String, self.nav_status_topic, self._nav_status_cb, 10)
        self.nav_pmt_pub = self.create_publisher(String, self.nav_prompt_topic, 10)
        self.response_pub = self.create_publisher(String, self.response_topic, 10)
        self.create_subscription(String, self.user_text_topic, self._user_text_cb, 10)

        # Internal flags
        self._awaiting_exec_response = False
        self._pending_exec_prompt: Optional[str] = None
        self.get_logger().info("✓")
    # ---------------------------------------------------------------------
    # User processing
    # ---------------------------------------------------------------------
    def process_user_text(self, user_text: str) -> None:
        if self._awaiting_exec_response:
            action_obj = self._call_responses_api(user_text, enforce_reply_only=True)
            if action_obj and action_obj.get('action') == 'reply':
                self._log_reply(action_obj.get('text', ''))
            else:
                self.get_logger().warn('Model attempted exec while planner busy; ignoring.')
            return

        action_obj = self._call_responses_api(user_text)
        if action_obj is None:
            self.get_logger().warn('Failed to get response from model.')
            return

        match action_obj.get('action'):
            case 'reply':
                self._log_reply(action_obj.get('text', ''))
            case 'exec':
                prompt = action_obj.get('prompt', '')
                ack_text = action_obj.get('text', '')
                if ack_text:
                    self._log_reply(ack_text)
                self._send_exec_request(prompt)
            case _:
                self.get_logger().warn('Model returned unknown action type.')


    def _user_text_cb(self, msg: String) -> None:
        """Handle incoming user text from the /user_text topic."""
        text = msg.data.strip()
        if text:
            self.process_user_text(text)

    # ---------------------------------------------------------------------
    # /nav/status callback — store but don’t print planner output
    # ---------------------------------------------------------------------
    def _nav_status_cb(self, msg: String) -> None:
        self._awaiting_exec_response = False

        # Stash planner text into history so the model can reference it later
        self.chat_history.append({'role': 'assistant', 'content': msg.data})

        self._pending_exec_prompt = None

    # ---------------------------------------------------------------------
    # OpenAI helper
    # ---------------------------------------------------------------------
    def _call_responses_api(self, user_text: str, *, enforce_reply_only: bool = False) -> Optional[dict]:
        if not self.api_key:
            return None

        chat = list(self.chat_history)
        chat.append({'role': 'user', 'content': user_text + '\n\njson'})
        if enforce_reply_only:
            chat.insert(0, {
                'role': 'system',
                'content': 'Planner busy – respond with {"action":"reply",...} only.'
            })
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=TRIAGE_SYSTEM_PROMPT,
                # In-repo prompt now (see node_Triage_API/prompt.py). Old server-stored
                # prompt kept as a fallback — uncomment to switch back:
                # prompt={"id": "pmpt_685c3b930ee48196b1dc9866c1f3452906bc8a963b8044a2", "version": "4"},
                input=chat,
            )
            output = response.output_text.strip()
            result = json.loads(output)
            self.chat_history.append({'role': 'user', 'content': user_text})
            self.chat_history.append({'role': 'assistant', 'content': output})
            return result
        except Exception as exc:
            self.get_logger().warn(f'Responses API error: {exc}')
            return None

    # ---------------------------------------------------------------------
    # Exec helpers
    # ---------------------------------------------------------------------
    def _send_exec_request(self, prompt: str) -> None:
        self.nav_pmt_pub.publish(String(data=prompt))
        self._awaiting_exec_response = True
        self._pending_exec_prompt = prompt

    # ---------------------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------------------
    def _log_reply(self, text: str) -> None:
        self.response_pub.publish(String(data=text))
        self.get_logger().info(f'{text}')

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:  # pragma: no cover
    rclpy.init(args=args)
    node = TriageApiNode()
    try:
        rclpy.spin(node)  # spin forever; callbacks handle work
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
