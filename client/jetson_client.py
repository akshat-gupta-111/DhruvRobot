import cv2
import requests
import sys
import subprocess
import urllib.parse
import threading
import time

SERVER_URL = "http://172.16.92.123:8000/interact" # Update with your Mac's IP

class LiveCameraStream:
    """Continuously consumes frames in a background thread to prevent buffer lag."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("FATAL ERROR: Could not map local device to camera.")
            sys.exit(1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


print("\n⚡ Jetson Edge Client Initialized. Connected to Dhruv Core.")
cam = LiveCameraStream(0).start()
time.sleep(1.0)

auto_search_active = False
last_query = ""

import re

# ... (keep your existing setup, cam = LiveCameraStream().start(), etc.)

active_tracking_target = None

try:
    while True:
        # --- MODE 1: TRACKING REFLEX LOOP ---
        if active_tracking_target:
            print(f"\n🔄 [Tracking Mode] Capturing frame for '{active_tracking_target}'...")
            
            # Flush for freshest frame
            frame_grab_status, raw_matrix_frame = cam.read()
            if not frame_grab_status:
                continue
                
            success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
            
            try:
                # Ping the FAST endpoint, NOT the LangGraph interact endpoint
                response = requests.post(
                    f"{SERVER_URL.replace('/interact', '/track')}", 
                    data={"target": active_tracking_target}, 
                    files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                    timeout=10
                )
                
                if response.status_code == 200:
                    action_text = response.json().get("action", "")
                    print(f"Reflex Action: {action_text}")
                    
                    if "[TARGET_REACHED]" in action_text:
                        print("🎉 Target reached! Returning to Chat Mode.")
                        active_tracking_target = None # Break the loop
                        
                # Brief sleep to allow physical actuators to complete their micro-movements
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Tracking network error: {e}")
                active_tracking_target = None
                
            continue # Loop back immediately, skipping the chat input!

        # --- MODE 2: DELIBERATIVE CHAT MODE ---
        query = input("\nYou: ")
        if query.strip().lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue
            
        frame_grab_status, raw_matrix_frame = cam.read()
        success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
        
        try:
            print("Thinking...")
            response = requests.post(
                SERVER_URL, 
                data={"query": query}, 
                files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                stream=True
            )
            
            if response.status_code == 200:
                encoded_text = response.headers.get("X-Agent-Text", "")
                if encoded_text:
                    decoded_text = urllib.parse.unquote(encoded_text)
                    print(f"Dhruv: {decoded_text}")
                    
                    # CHECK FOR STATE TRANSITION TRIGGER
                    match = re.search(r'\[START_TRACKING:\s*(.+?)\]', decoded_text)
                    if match:
                        active_tracking_target = match.group(1).strip()
                        print(f"⚙️ Switching to Reflex Tracking Mode for: {active_tracking_target}")
                
                # Stream audio
                mpv_process = subprocess.Popen(['mpv', '--no-video', '--really-quiet', '-'], stdin=subprocess.PIPE)
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        mpv_process.stdin.write(chunk)
                mpv_process.wait()
        except Exception as e:
             print(f"Network error: {e}")
finally:
    cam.stop()