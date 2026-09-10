import os
import json
import subprocess
import time

KERNEL_ID = "akshatgupta2006/local-automation-test"
SCRIPT_FILE = "llava.py"
NOTEBOOK_FILE = "automated_run.ipynb"

def run_command(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def build_automated_notebook():
    """Reads llava.py and wraps it programmatically into a valid Jupyter Notebook."""
    print("🛠️ Generating automated notebook template...")
    if not os.path.exists(SCRIPT_FILE):
        print(f"❌ Error: {SCRIPT_FILE} not found locally.")
        return False
        
    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        script_content = f.read()

    # Structure a flawless Jupyter Notebook configuration block
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

def trigger_and_stream():
    if not build_automated_notebook():
        return

    print("🚀 Triggering Kaggle GPU cloud instance...")
    push_output = run_command("kaggle kernels push")
    if not push_output:
        print("❌ Failed to push to Kaggle. Check credentials or metadata configuration.")
        return
    print(f"✅ Success: {push_output}")

    print("\n⏳ Monitoring remote execution loop (Press Ctrl+C to stop local tracking)...")
    print("🤖 The server will remain RUNNING on Kaggle backend even if you close this local script.")
    
    while True:
        status_output = run_command(f"kaggle kernels status {KERNEL_ID}")
        if status_output:
            print(f"ℹ️ Current Remote Status: {status_output}")
            
            if "has status \"complete\"" in status_output.lower():
                print("🏁 Server run ended unexpectedly or reached execution limit.")
                break
            elif "has status \"error\"" in status_output.lower():
                print("❌ Server crashed. Fetching crash logs...")
                logs = run_command(f"kaggle kernels output {KERNEL_ID}")
                if logs: print(logs)
                break
        
        # Periodically attempt to fetch the output files to look for an established Ngrok URL
        # You can see live terminal prints by viewing the logs at your Kaggle dashboard URL.
        time.sleep(20)

if __name__ == "__main__":
    try:
        trigger_and_stream()
    except KeyboardInterrupt:
        print("\n👋 Local tracking stopped. Note: Your server is still running live on Kaggle.")
