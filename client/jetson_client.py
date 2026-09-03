import cv2
import requests
import sys
import subprocess
import urllib.parse
import threading
import time
import re

SERVER_URL = "http://172.16.92.123:8000/interact" # Update with your Mac's IP

class LiveCameraStream:
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

active_tracking_target = None

try:
    while True:
        # --- MODE 1: TRACKING REFLEX LOOP ---
        if active_tracking_target:
            print(f"\n🔄 [Tracking Mode] Capturing frame for '{active_tracking_target}'...")
            
            frame_grab_status, raw_matrix_frame = cam.read()
            if not frame_grab_status:
                continue
                
            success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
            
            try:
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
                        active_tracking_target = None 
                        
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Tracking network error: {e}")
                active_tracking_target = None
                
            continue 

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
                stream=True,
                timeout=60 # Prevent hanging if server dies
            )
            
            if response.status_code == 200:
                encoded_text = response.headers.get("X-Agent-Text", "")
                if encoded_text:
                    decoded_text = urllib.parse.unquote(encoded_text)
                    print(f"Dhruv: {decoded_text}")
                    
                    match = re.search(r'\[START_TRACKING:\s*(.+?)\]', decoded_text)
                    if match:
                        active_tracking_target = match.group(1).strip()
                        print(f"⚙️ Switching to Reflex Tracking Mode for: {active_tracking_target}")
                
                # --- ROBUST AUDIO PIPE FIX ---
                mpv_process = subprocess.Popen(
                    ['mpv', '--no-video', '--really-quiet', '-'], 
                    stdin=subprocess.PIPE
                )
                
                # Write chunks
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        mpv_process.stdin.write(chunk)
                        
                # Explicitly close the pipe FIRST so mpv knows the file is done
                mpv_process.stdin.close()
                
                # Wait for playback to finish, with a safety timeout
                try:
                    mpv_process.wait(timeout=10.0) 
                except subprocess.TimeoutExpired:
                    mpv_process.kill()
                    print("[Warning] Audio playback timed out.")
                    
        except Exception as e:
             print(f"Network error: {e}")
             
finally:
    cam.stop()
2. Double-Check the Fast Endpoint (api_server.py)
When the Jetson successfully switches to Tracking Mode, it hits your /track endpoint. Ensure this endpoint does not try to generate edge-tts audio. It must purely return JSON to keep the loop lightning fast.

Python
# Verify your /track endpoint looks like this in api_server.py:

@app.post("/track")
async def track_target(target: str = Form(...), image: UploadFile = File(...)):
    """The High-Frequency Reflex Loop endpoint (Bypasses LangGraph entirely)."""
    image_bytes = await image.read()
    b64_payload = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"\n🎯 [Reflex Tracker] Searching frame for: '{target}'")
    
    vision_state = await get_moondream_tracking_json(b64_payload, target)
    print(f"   ├─ State: {vision_state}")
    
    response_action = ""
    
    if not vision_state.get("is_there"):
        response_action = "Target lost. Spinning to search."
        await ble_manager.send_payload_and_wait({"c": "SPN", "d": "R", "deg": 30, "t": 500}, 500)
        
    elif vision_state.get("location") == "left":
        response_action = "Target left. Nudging left."
        await ble_manager.send_payload_and_wait({"c": "SPN", "d": "L", "deg": 10, "t": 200}, 200)
        
    elif vision_state.get("location") == "right":
        response_action = "Target right. Nudging right."
        await ble_manager.send_payload_and_wait({"c": "SPN", "d": "R", "deg": 10, "t": 200}, 200)
        
    elif vision_state.get("location") == "center":
        if vision_state.get("closeness") == "high":
            response_action = "[TARGET_REACHED] I have reached the target."
            await ble_manager.send_payload_and_wait({"c": "STP"}, 0)
        else:
            response_action = "Target centered. Moving forward."
            await ble_manager.send_payload_and_wait({"c": "DRV", "d": "F", "t": 1000, "s": 200}, 1000)

    print(f"   └─ Action: {response_action}")
    
    # Returns standard JSON - NO AUDIO STREAMING
    return {"action": response_action}