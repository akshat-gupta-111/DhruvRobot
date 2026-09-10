import cv2
import requests
import sys
import subprocess
import urllib.parse
import threading
import time
import queue
import platform
import os
import tempfile
import numpy as np
import face_recognition

SERVER_URL = "http://127.0.0.1:8000/autonomous" # Running onboard the Jetson Nano

# Load Known Faces
known_faces_dir = "known_faces"
known_face_encodings = []
known_face_names = []

if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

print("\n[VISION] Loading known faces for autonomous mode...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].capitalize()
            known_face_names.append(name)
            print(f"         -> Loaded face profile: {name}")

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

# Voice Queue (Loop C)
audio_queue = queue.Queue()

def voice_worker():
    """Loop C: The Voice & Personality Loop. Plays audio without blocking vision."""
    print("🗣️ Voice Loop (Loop C) Started.")
    while True:
        try:
            audio_content = audio_queue.get()
            if audio_content is None:
                break
                
            if platform.system() == "Darwin":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_filename = fp.name
                    fp.write(audio_content)
                subprocess.run(['afplay', temp_filename])
                os.remove(temp_filename)
            else:
                mpv_process = subprocess.Popen(['mpv', '--no-video', '--really-quiet', '-'], stdin=subprocess.PIPE)
                mpv_process.stdin.write(audio_content)
                mpv_process.stdin.close()
                mpv_process.wait()
                
            audio_queue.task_done()
        except Exception as e:
            print(f"Voice playback error: {e}")

# Start Loop C in background
threading.Thread(target=voice_worker, daemon=True).start()

print("\n⚡ Jetson Edge Autonomous Client Initialized. Connected to Dhruv Core.")
cam = LiveCameraStream(0).start()
time.sleep(1.0)

last_frame_gray = None
last_send_time = time.time()

try:
    print("👁️ Vision Loop (Loop A) Started.")
    while True:
        time.sleep(1.0) # 1 FPS
        
        frame_grab_status, raw_matrix_frame = cam.read()
        if not frame_grab_status:
            continue
            
        # Calculate Interestingness
        gray = cv2.cvtColor(raw_matrix_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        is_interesting = False
        current_time = time.time()
        
        if last_frame_gray is None:
            is_interesting = True
            last_frame_gray = gray
        else:
            frame_delta = cv2.absdiff(last_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            diff_score = np.sum(thresh) / 255
            
            if diff_score > 5000: # Threshold for significant scene change
                is_interesting = True
                last_frame_gray = gray
                print(f"\n👀 [Vision] Interesting scene detected! (Score: {diff_score})")
            elif current_time - last_send_time > 5.0:
                # Mandatory safety frame so the server knows if we hit a dead-end wall
                is_interesting = True 
                print(f"\n👀 [Vision] Safety frame (checking for dead-ends).")

        if is_interesting:
            # Face Recognition (Run ONLY when sending an interesting frame to save CPU)
            detected_names = []
            if known_face_encodings:
                rgb_frame = cv2.cvtColor(raw_matrix_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    print(f"\n[DEBUG] Found {len(face_locations)} face(s) in frame. Attempting recognition...")
                    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encs:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match = np.argmin(face_distances)
                            print(f"[DEBUG] Closest match distance: {face_distances[best_match]:.3f} (Tolerance is 0.55)")
                            
                            if matches[best_match]:
                                name = known_face_names[best_match]
                                print(f"[DEBUG] Successfully matched: {name}!")
                                if name not in detected_names:
                                    detected_names.append(name)
                            else:
                                print(f"[DEBUG] Face rejected (Too far from known faces).")

            success, compression_buffer = cv2.imencode('.jpg', raw_matrix_frame)
            if not success:
                continue
                
            try:
                payload_data = {}
                if detected_names:
                    payload_data["names"] = ",".join(detected_names)
                    print(f"\n👀 [Vision] Recognized known person(s): {payload_data['names']}")

                # Send to Mac Brain (Loop B)
                response = requests.post(
                    SERVER_URL, 
                    data=payload_data,
                    files={"image": ("frame.jpg", compression_buffer.tobytes(), "image/jpeg")},
                    timeout=60
                )
                
                if response.status_code == 200:
                    last_send_time = time.time()
                    has_audio = response.headers.get("X-Has-Audio", "false")
                    
                    if has_audio == "true":
                        encoded_text = response.headers.get("X-Agent-Text", "")
                        if encoded_text:
                            decoded_text = urllib.parse.unquote(encoded_text)
                            print(f"🤖 Brain: {decoded_text}")
                        
                        # Queue audio to play in background (Loop C)
                        audio_queue.put(response.content)
                    else:
                        print("🤖 Brain processed frame. Exploring...")
                    
            except requests.exceptions.Timeout:
                print("Network failure: Request timed out")
            except Exception as e:
                print(f"Network failure: {e}")
                
finally:
    cam.stop()
    audio_queue.put(None)
