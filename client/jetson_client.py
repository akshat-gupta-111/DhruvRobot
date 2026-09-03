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
time.sleep(1.0) # Allow camera sensor to warm up

try:
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue
            
        print("Capturing live frame...")
        # Instantly grabs the frame captured milliseconds ago by the background thread
        frame_grab_status, raw_matrix_frame = cam.read()
            
        if not frame_grab_status:
            print("Camera error.")
            continue
            
        success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
        if not success:
            continue
            
        print("Transmitting context to Dhruv Brain...")
        try:
            response = requests.post(
                SERVER_URL, 
                data={"query": query}, 
                files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                img_status = response.headers.get("X-Image-Status", "")
                if img_status:
                    print(f"[Server Vision Status]: {urllib.parse.unquote(img_status)}")

                encoded_text = response.headers.get("X-Agent-Text", "")
                if encoded_text:
                    print(f"Dhruv: {urllib.parse.unquote(encoded_text)}")
                
                mpv_process = subprocess.Popen(
                    ['mpv', '--no-video', '--really-quiet', '-'],
                    stdin=subprocess.PIPE
                )
                
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        mpv_process.stdin.write(chunk)
                
                mpv_process.stdin.close()
                mpv_process.wait()
            else:
                print(f"Server Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Network failure: {e}")
            
finally:
    cam.stop()
    print("Jetson offline.")