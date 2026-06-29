from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

def is_debug() -> bool:
    return os.getenv("DEBUG", "").lower() == "true"

def debug_log(**kwargs):
    if not is_debug():
        return

    if not kwargs:
        return

    session_round = kwargs.pop("session_round", None)
    if session_round is not None:
        print(f"\n\n===== Assistant (Session Round: {session_round}) =====")

    tool_name = kwargs.pop("tool_name", None)
    if tool_name:
        print(f"\n\nTool: {tool_name}")

    assistant_response = kwargs.pop("assistant_response", None)
    if assistant_response:
        print(f"\n\nAssistant Response: \n{assistant_response}")

    for key, value in kwargs.items():
            print(f"{key}: {value}")
