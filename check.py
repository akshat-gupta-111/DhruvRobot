# ========================================================
# LOCAL DEVICE CODE: RUN DIRECTLY ON YOUR JETSON ORIN
# ========================================================
import cv2
import requests
import base64
import time
import sys

# ⚠️ PASTE YOUR COPIED NGROK URL FROM KAGGLE CELL 2 DIRECTLY HERE:
# Example: "https://ngrok-free.app"
PUBLIC_NGROK_URL = "https://choreographic-zelda-tangly.ngrok-free.dev/"
TARGET_ENDPOINT = f"{PUBLIC_NGROK_URL}/api/generate"

def image_to_base64_payload(opencv_frame):
    """Encodes raw matrix frame array data to a standard Base64 string payload."""
    # Compress the frame to JPEG to radically minimize network upload time
    success, compression_buffer = cv2.imencode('.jpg', opencv_frame)
    if not success:
        return None
    return base64.b64encode(compression_buffer).decode('utf-8')

def transmit_payload(encoded_frame, query_string):
    """Handles HTTP session request execution to the cloud server framework."""
    payload = {
        "model": "moondream",
        "prompt": query_string,
        "stream": False,
        "images": [encoded_frame]
    }
    
    # Clean headers block—Ngrok handles whitelisting natively via your auth token
    network_headers = {
        "Content-Type": "application/json",
        "User-Agent": "JetsonOrinClient/1.0"
    }
    
    try:
        execution_start = time.time()
        network_response = requests.post(TARGET_ENDPOINT, json=payload, headers=network_headers, timeout=30)
        roundtrip_latency = time.time() - execution_start
        
        if network_response.status_code == 200:
            parsed_text = network_response.json().get("response")
            print(f"\n💡 [AI Insights] (Roundtrip Latency: {roundtrip_latency:.2f}s):\n{parsed_text}")
        else:
            print(f"\n❌ Server returned error status: {network_response.status_code}")
            print(f"Server Response Text: {network_response.text}")
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out. The model took too long to respond.")
    except Exception as network_error:
        print(f"\n⚠️ Remote socket communication breakdown: {network_error}")

# Mount local edge media hardware layer (0 indicates default USB system hardware index)
video_capture_object = cv2.VideoCapture(0)

if not video_capture_object.isOpened():
    print("FATAL ERROR: Could not map local device to an operational camera index.")
    sys.exit(1)

print("\n" + "="*60)
print("⚡ Hardware initialization complete. System listening...")
print("Input your text prompt queries below (Type 'exit' to terminate loop)")
print("="*60 + "\n")

try:
    while True:
        input_query = input("Ask about the live video frame context: ")
        if input_query.strip().lower() == 'exit':
            break
            
        if not input_query.strip():
            continue

        print("Capturing absolute freshest frame...")
        # Clear out old frames stacked up in the OS video driver buffer queue
        # This guarantees the model sees exactly what is happening *now*
        for _ in range(5):
            frame_grab_status, raw_matrix_frame = video_capture_object.read()
            
        if not frame_grab_status:
            print("Failed to pull frame matrix from camera pipeline.")
            continue
            
        print("Sending payload data over Ngrok tunnel to Kaggle GPU...")
        base64_payload_string = image_to_base64_payload(raw_matrix_frame)
        
        if base64_payload_string:
            transmit_payload(base64_payload_string, input_query)
        else:
            print("Compression encoding failure encountered.")

finally:
    # Safely release camera hardware interface hook references
    video_capture_object.release()
    print("\nLocal system components cleanly unmounted. Exiting.")
