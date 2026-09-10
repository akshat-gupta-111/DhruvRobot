"""
DhruvRobot - Root Runner
========================
Dispatches to the lightweight Kaggle-GPU / Orin single-script pipeline.
Zero LangGraph | Zero Local Ollama | 100% Native Python

Usage:
  python main.py --chat       # Interactive text chat (keyboard input + spoken voice output)
  python main.py --trigger    # Auto-trigger Kaggle GPU and infer Ngrok URL
"""

import sys
from pathlib import Path

# Add DhruvOrin to path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / "DhruvOrin"))

from orin_main import run_dhruv
import asyncio
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DhruvRobot Pipeline")
    parser.add_argument("--chat", action="store_true", default=True, help="Run in text chat mode with voice output (Keyboard input + Spoken speech)")
    parser.add_argument("--trigger", action="store_true", help="Trigger Kaggle GPU instance and automatically obtain active Ngrok URL")
    args = parser.parse_args()

    trigger_flag = args.trigger or os.getenv("AUTO_TRIGGER_KAGGLE", "false").lower() in ("true", "1", "yes")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_dhruv(trigger_requested=trigger_flag, chat_mode=args.chat))
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting.")