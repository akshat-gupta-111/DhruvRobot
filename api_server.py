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

def extract_text_locally(image_bytes):
    """Runs Tesseract locally on the server to keep Jetson lightweight."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else "No legible text found."
    except Exception as e:
        return f"OCR Error: {e}"

async def get_scene_description(base64_image):
    """Pings your Ngrok Moondream API asynchronously."""
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
            return "Scene unavailable."
    except Exception as e:
        return f"Vision offline: {str(e)}"

@app.post("/interact")
async def interact(query: str = Form(...), image: UploadFile = File(...)):
    """Receives query + frame from Jetson, returns Text (Headers) + Audio stream (Body)."""
    
    # 1. Read binary image from Jetson
    image_bytes = await image.read()
    base64_payload = base64.b64encode(image_bytes).decode('utf-8')
    
    # 2. Run Scene Analysis and OCR concurrently
    ocr_task = asyncio.to_thread(extract_text_locally, image_bytes)
    scene_task = get_scene_description(base64_payload)
    ocr_text, scene_description = await asyncio.gather(ocr_task, scene_task)
    
    combined_context = f"SCENE DESCRIPTION: {scene_description}\nVISIBLE TEXT: {ocr_text}"
    
    # 3. Feed to LangGraph Brain (using thread_id="1" to remember the Jetson's conversation)
    state_input = {
        "messages": [HumanMessage(content=query)],
        "current_scene": combined_context
    }
    
    response_text = ""
    for event in dhruv_brain.stream(state_input, config={"configurable": {"thread_id": "1"}}):
        for value in event.values():
            response_text = value['messages'][-1].content
            
    # 4. Clean text and put it in a custom header so Jetson can print it immediately
    clean_text = response_text.replace("*", "").replace("#", "")
    headers = {"X-Agent-Text": urllib.parse.quote(response_text)}
    
    # 5. Stream the generated audio back to Jetson piece-by-piece
    async def audio_generator():
        communicate = edge_tts.Communicate(clean_text, "en-IN-PrabhatNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
                
    return StreamingResponse(audio_generator(), media_type="audio/mpeg", headers=headers)