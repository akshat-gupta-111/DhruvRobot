import cv2
import asyncio
import httpx
import base64
import os
import pytesseract # Added for local OCR
from dotenv import load_dotenv

load_dotenv()

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

    base_url = os.getenv("NGROK_BASE_URL", "").rstrip('/')
    target_endpoint = f"{base_url}/api/generate"
    
    vision_prompt = "Describe the current scene, objects, and people. Do not attempt to read text."
    
    network_headers = {
        "Content-Type": "application/json",
        "User-Agent": "DhruvVisionAgent/1.1"
    }

    print("👁️ Dhruv's dual-pipeline vision (Scene + OCR) activated.")

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

                payload = {
                    "model": "moondream",
                    "prompt": vision_prompt,
                    "stream": False,
                    "images": [base64_payload_string]
                }

                try:
                    ocr_task = asyncio.to_thread(extract_text_locally, raw_matrix_frame)
                    api_task = client.post(target_endpoint, json=payload, headers=network_headers)

                    ocr_text, response = await asyncio.gather(ocr_task, api_task)
                    
                    scene_description = "Scene unavailable."
                    if response.status_code == 200:
                        scene_description = response.json().get("response", "").strip()
                    
                    combined_context = f"""
                    SCENE DESCRIPTION: {scene_description}
                    VISIBLE TEXT DETECTED: {ocr_text}
                    """
                    
                    shared_memory["current_scene"] = combined_context.strip()
                        
                except httpx.TimeoutException:
                    shared_memory["current_scene"] = "Vision lagging: Ngrok/Kaggle endpoint timed out."
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