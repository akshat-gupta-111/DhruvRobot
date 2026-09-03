# sudo apt update && sudo apt install mpv -y

import cv2
import requests
import sys
import subprocess
import urllib.parse
import time

# Change this to your Mac's IP address on the local network
SERVER_URL = "http://192.168.X.X:8000/interact"

video_capture_object = cv2.VideoCapture(0)
if not video_capture_object.isOpened():
    print("FATAL ERROR: Could not map local device to camera.")
    sys.exit(1)

print("\n⚡ Jetson Edge Client Initialized. Connected to Dhruv Core.")

try:
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue
            
        print("Capturing live frame...")
        # Flush the buffer to get the freshest frame
        for _ in range(5):
            frame_grab_status, raw_matrix_frame = video_capture_object.read()
            
        if not frame_grab_status:
            print("Camera error.")
            continue
            
        # Encode to tiny JPEG bytes
        success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
        if not success:
            continue
            
        print("Transmitting context to Dhruv Brain...")
        try:
            # We use stream=True to process the audio chunk-by-chunk as it arrives
            response = requests.post(
                SERVER_URL, 
                data={"query": query}, 
                files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                # 1. Print the text response embedded in the HTTP headers instantly
                encoded_text = response.headers.get("X-Agent-Text", "")
                if encoded_text:
                    print(f"Dhruv: {urllib.parse.unquote(encoded_text)}")
                
                # 2. Pipe the streaming audio payload directly into the Jetson's speakers via mpv
                mpv_process = subprocess.Popen(
                    ['mpv', '--no-video', '--really-quiet', '-'],
                    stdin=subprocess.PIPE
                )
                
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        mpv_process.stdin.write(chunk)
                
                # Close the pipe and wait for the audio to finish playing
                mpv_process.stdin.close()
                mpv_process.wait()
                
            else:
                print(f"Server Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Network failure: {e}")
            
finally:
    video_capture_object.release()
    print("Jetson offline.")