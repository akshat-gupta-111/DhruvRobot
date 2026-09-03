import cv2
import asyncio
import httpx
import base64
import os
import pytesseract # Added for local OCR
from dotenv import load_dotenv

load_dotenv()

# Moondream Cloud API config
MOONDREAM_API_URL = "https://api.moondream.ai/v1/caption"
MOONDREAM_MODEL   = "moondream3.1-9B-A2B"

def image_to_base64_payload(opencv_frame):
    success, compression_buffer = cv2.imencode('.jpg', opencv_frame)
    if not success:
        return None
    return base64.b64encode(compression_buffer).decode('utf-8')

def extract_text_locally(frame):
    """Uses local CPU to extract text from the frame."""
    try:
        # Convert to grayscale for better OCR accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else "No legible text found."
    except Exception as e:
        return f"OCR Error: {e}"

async def capture_and_analyze(shared_memory: dict, stop_event=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        shared_memory["current_scene"] = "Vision Error: Camera not accessible."
        return

    api_key = os.getenv("MOONDREAM_API_KEY", "")

    network_headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
        "User-Agent": "DhruvVisionAgent/2.0"
    }

    print("👁️ Dhruv's dual-pipeline vision (Scene + OCR) activated.")
    print(f"   └─ Using Moondream Cloud API → {MOONDREAM_API_URL}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                # 1. Check for shutdown signal at the start of the loop
                if stop_event and stop_event.is_set():
                    break

                for _ in range(5):
                    frame_grab_status, raw_matrix_frame = cap.read()
                    
                if not frame_grab_status:
                    await asyncio.sleep(1)
                    continue

                base64_payload_string = image_to_base64_payload(raw_matrix_frame)
                if not base64_payload_string:
                    continue

                # Moondream Cloud API expects a data-URI for the image
                image_data_uri = f"data:image/jpeg;base64,{base64_payload_string}"

                payload = {
                    "model": MOONDREAM_MODEL,
                    "image_url": image_data_uri,
                    "stream": False
                }

                try:
                    ocr_task = asyncio.to_thread(extract_text_locally, raw_matrix_frame)
                    api_task = client.post(MOONDREAM_API_URL, json=payload, headers=network_headers)

                    ocr_text, response = await asyncio.gather(ocr_task, api_task)
                    
                    scene_description = "Scene unavailable."
                    if response.status_code == 200:
                        # Moondream Cloud API returns {"caption": "..."}
                        scene_description = response.json().get("caption", "").strip()
                    else:
                        scene_description = f"API Error {response.status_code}: {response.text[:120]}"
                    
                    combined_context = f"""
                    SCENE DESCRIPTION: {scene_description}
                    VISIBLE TEXT DETECTED: {ocr_text}
                    """
                    
                    shared_memory["current_scene"] = combined_context.strip()
                        
                except httpx.TimeoutException:
                    shared_memory["current_scene"] = "Vision lagging: Moondream Cloud API timed out."
                except Exception as e:
                    shared_memory["current_scene"] = f"Vision offline: {str(e)}"

                # 2. Break the 5-second sleep into 0.1s chunks. 
                # This makes the thread respond instantly to Ctrl+C instead of hanging.
                for _ in range(50):
                    if stop_event and stop_event.is_set():
                        break
                    await asyncio.sleep(0.1)

    finally:
        # 3. Cleanly release the hardware when the loop terminates
        cap.release()


# ── Quick API test ────────────────────────────────────────────
if __name__ == "__main__":
    import httpx, base64, os
    from dotenv import load_dotenv
    load_dotenv()

    print("🧪 Moondream Cloud API — single-frame test")

    # 1. Grab one frame from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible."); exit(1)

    for _ in range(5):           # flush stale buffer frames
        ok, frame = cap.read()
    cap.release()

    if not ok:
        print("❌ Failed to read frame."); exit(1)

    print("📸 Frame captured — sending to Moondream API...")

    # 2. Encode to base64 data URI
    _, buf = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    image_data_uri = f"data:image/jpeg;base64,{b64}"

    # 3. Call the API
    api_key = os.getenv("MOONDREAM_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
    }
    payload = {
        "model": MOONDREAM_MODEL,
        "image_url": image_data_uri,
        "stream": False,
    }

    try:
        response = httpx.post(MOONDREAM_API_URL, json=payload, headers=headers, timeout=30.0)
        if response.status_code == 200:
            caption = response.json().get("caption", "").strip()
            print(f"\n✅ API Response:\n   {caption}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except httpx.TimeoutException:
        print("❌ Request timed out.")
    except Exception as e:
        print(f"❌ Exception: {e}")