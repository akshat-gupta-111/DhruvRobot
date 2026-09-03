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

try:
    while True:
        if not auto_search_active:
            query = input("\nYou: ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
            last_query = query
        else:
            print("\n🔄 [Autonomous Mode] Capturing fresh frame to evaluate environment...")
            query = "Evaluate the new scene. " + last_query 
            # Give the rover a brief moment to stabilize after moving before capturing the next frame
            time.sleep(0.5)

        # Flush buffer and grab freshest frame
        frame_grab_status, raw_matrix_frame = cam.read()
        if not frame_grab_status:
            continue
            
        success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
        if not success:
            continue
            
        try:
            response = requests.post(
                SERVER_URL, 
                data={"query": query}, 
                files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                encoded_text = response.headers.get("X-Agent-Text", "")
                if encoded_text:
                    decoded_text = urllib.parse.unquote(encoded_text)
                    print(f"Dhruv: {decoded_text}")
                    
                    # TRIGGER AUTONOMOUS LOOP IF TOKEN IS PRESENT
                    if "[SEARCH_CONTINUES]" in decoded_text:
                        auto_search_active = True
                    else:
                        auto_search_active = False
                
                # Stream Audio
                mpv_process = subprocess.Popen(['mpv', '--no-video', '--really-quiet', '-'], stdin=subprocess.PIPE)
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        mpv_process.stdin.write(chunk)
                mpv_process.stdin.close()
                mpv_process.wait()
                
        except Exception as e:
            print(f"Network failure: {e}")
            auto_search_active = False # Break loop on error
            
finally:
    cam.stop()