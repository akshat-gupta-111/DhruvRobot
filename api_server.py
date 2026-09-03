import os
import cv2
import numpy as np
import base64
import asyncio
import httpx
import pytesseract
import edge_tts
import urllib.parse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from core.graph import dhruv_brain
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

def extract_text_locally(frame):
    """Runs Tesseract locally on the already-decoded OpenCV frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else "No legible text found."
    except Exception as e:
        return f"OCR Error: {e}"

async def get_scene_description(base64_image: str):
    """Pings the remote Moondream API asynchronously."""
    base_url = os.getenv("NGROK_BASE_URL", "").rstrip('/')
    target_endpoint = f"{base_url}/api/generate"
    
    payload = {
        "model": "moondream",
        "prompt": "Describe the current scene, objects, and people. Do not attempt to read text.",
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(target_endpoint, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"Scene unavailable (HTTP {response.status_code})."
    except Exception as e:
        return f"Vision offline: {str(e)}"

@app.post("/interact")
async def interact(query: str = Form(...), image: UploadFile = File(None)):
    print("\n" + "─" * 55)
    print(f"📥 Request Received | Query: \"{query}\"")

    image_received = False
    image_status_msg = "No Image Received"
    combined_context = "No visual input provided."

    # 1. Inspect and validate incoming image payload
    if image is not None:
        image_bytes = await image.read()
        size_kb = len(image_bytes) / 1024

        if size_kb > 0:
            # Decode byte buffer into OpenCV matrix
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                height, width, channels = frame.shape
                image_received = True
                image_status_msg = f"OK: {width}x{height} ({size_kb:.1f} KB)"
                print(f"📸 Image Frame: ✅ RECEIVED | {width}x{height} px | {size_kb:.1f} KB | Name: '{image.filename}'")
            else:
                image_status_msg = "Error: Decode Failed"
                print(f"📸 Image Frame: ❌ FAILED TO DECODE ({size_kb:.1f} KB buffer corrupted)")
        else:
            image_status_msg = "Error: Empty File (0 KB)"
            print("📸 Image Frame: ⚠️ EMPTY PAYLOAD (0 bytes)")
    else:
        print("📸 Image Frame: ⚠️ NONE (Client sent null/empty file field)")

    # 2. Process scene & OCR if image is valid
    if image_received:
        base64_payload = base64.b64encode(image_bytes).decode('utf-8')
        
        # Run OCR and remote Moondream concurrently
        ocr_task = asyncio.to_thread(extract_text_locally, frame)
        scene_task = get_scene_description(base64_payload)
        
        ocr_text, scene_description = await asyncio.gather(ocr_task, scene_task)
        print(f"🔍 OCR Output  : {ocr_text[:60]}{'...' if len(ocr_text) > 60 else ''}")
        print(f"🧠 Moondream   : {scene_description[:60]}{'...' if len(scene_description) > 60 else ''}")

        combined_context = f"SCENE DESCRIPTION: {scene_description}\nVISIBLE TEXT: {ocr_text}"

    # 3. Feed to LangGraph Brain
    state_input = {
        "messages": [HumanMessage(content=query)],
        "current_scene": combined_context
    }
    
    response_text = ""
    for event in dhruv_brain.stream(state_input, config={"configurable": {"thread_id": "1"}}):
        for value in event.values():
            response_text = value['messages'][-1].content

    clean_text = response_text.replace("*", "").replace("#", "")
    print(f"🗣️ Response    : {response_text}")
    print("─" * 55)

    # 4. Attach telemetry headers for the Jetson
    headers = {
        "X-Agent-Text": urllib.parse.quote(response_text),
        "X-Image-Status": urllib.parse.quote(image_status_msg)
    }
    
    # 5. Stream audio
    async def audio_generator():
        communicate = edge_tts.Communicate(clean_text, "en-IN-PrabhatNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
                
    return StreamingResponse(audio_generator(), media_type="audio/mpeg", headers=headers)