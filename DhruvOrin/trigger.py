"""
Kaggle Moondream Trigger Automation
====================================
Packages llava.py into automated_run.ipynb, pushes it to Kaggle GPU,
and tracks the remote status to provide the active Ngrok tunnel.
"""

import os
import json
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
KERNEL_ID = "akshatgupta2006/local-automation-test"
SCRIPT_FILE = BASE_DIR / "llava.py"
NOTEBOOK_FILE = BASE_DIR / "automated_run.ipynb"

# Ensure Kaggle credentials are discoverable by Kaggle CLI
KAGGLE_JSON = BASE_DIR / "kaggle.json"
if KAGGLE_JSON.exists():
    os.environ["KAGGLE_CONFIG_DIR"] = str(BASE_DIR)


def run_command(command: str):
    """Executes a shell command within the script directory."""
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def build_automated_notebook() -> bool:
    """Reads llava.py and wraps it programmatically into a valid Jupyter Notebook."""
    print("🛠️ Generating automated notebook template...")
    if not SCRIPT_FILE.exists():
        print(f"❌ Error: {SCRIPT_FILE} not found locally.")
        return False

    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        script_content = f.read()

    notebook_data = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [script_content]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(NOTEBOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)
    print("✅ Created automated_run.ipynb with embedded execution code.")
    return True


def trigger_and_stream(stream_logs: bool = True) -> bool:
    """Pushes the notebook to Kaggle and starts tracking status."""
    if not build_automated_notebook():
        return False

    print("🚀 Triggering Kaggle GPU cloud instance...")
    push_output = run_command("kaggle kernels push")
    if not push_output:
        print("❌ Failed to push to Kaggle. Check credentials or metadata configuration.")
        return False
    print(f"✅ Success: {push_output}")

    if not stream_logs:
        return True

    print("\n⏳ Monitoring remote execution loop (Press Ctrl+C to stop local tracking)...")
    print("🤖 The server will remain RUNNING on Kaggle backend even if you close this local script.")

    try:
        while True:
            status_output = run_command(f"kaggle kernels status {KERNEL_ID}")
            if status_output:
                print(f"ℹ️ Current Remote Status: {status_output}")

                if 'has status "complete"' in status_output.lower():
                    print("🏁 Server run ended unexpectedly or reached execution limit.")
                    break
                elif 'has status "error"' in status_output.lower():
                    print("❌ Server crashed. Fetching crash logs...")
                    logs = run_command(f"kaggle kernels output {KERNEL_ID}")
                    if logs:
                        print(logs)
                    break

            time.sleep(20)
    except KeyboardInterrupt:
        print("\n👋 Local tracking stopped. Note: Your server is still running live on Kaggle.")
    return True


if __name__ == "__main__":
    try:
        trigger_and_stream(stream_logs=True)
    except KeyboardInterrupt:
        print("\n👋 Local tracking stopped.")
