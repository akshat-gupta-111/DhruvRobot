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
from tools.diagnostics import run_system_diagnostics
from contextlib import asynccontextmanager

load_dotenv()
app = FastAPI()

VOICE = "en-IN-NeerjaNeural"
SPEED_RATE = "+25%"

api_key = os.getenv("MOONDREAM_API_KEY", "")
MOONDREAM_API_URL = "https://api.moondream.ai/v1/caption"
MOONDREAM_MODEL   = "moondream3.1-9B-A2B"

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
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    headers = {
            "Content-Type": "application/json",
            "X-Moondream-Auth": api_key,
        }
    payload = {
            "model": MOONDREAM_MODEL,
            "image_url": image_data_uri,
            "stream": False,
        }
    # print("iside the function")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(MOONDREAM_API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                caption = response.json().get("caption", "").strip()
                return caption
            return f"Scene unavailable (HTTP {response.status_code})."
        
    except Exception as e:
        return f"Vision offline: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs the pre-flight connection tests before the server starts accepting requests
    await run_system_diagnostics()
    yield
    print("\n🛑 Shutting down Dhruv Core Server...")

app = FastAPI(lifespan=lifespan)

@app.post("/interact")
async def interact(query: str = Form(...), image: UploadFile = File(None)):
    print("\n" + "─" * 55)
    print(f"📥 Request Received | Query: \"{query}\"")

    b64_payload = None
    image_status = "No Image"

    if image is not None:
        image_bytes = await image.read()
        if len(image_bytes) > 0:
            b64_payload = base64.b64encode(image_bytes).decode('utf-8')
            image_status = f"Attached ({len(image_bytes)/1024:.1f} KB)"
            print(f"📸 Image Payload: ✅ {image_status}")

    # Build input state for the agent graph
    state_input = {
        "messages": [HumanMessage(content=query)],
        "raw_image_b64": b64_payload,
        "current_scene": ""
    }

    # Execute graph asynchronously
    response_text = ""
    async for event in dhruv_brain.astream(state_input, config={"configurable": {"thread_id": "1"}}):
        for node_name, value in event.items():
            if "messages" in value and value["messages"]:
                last_msg = value["messages"][-1]
                # If a tool was executed, log the hardware action
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"🔧 Tool Triggered: {tc['name']}({tc['args']})")
                elif hasattr(last_msg, "content") and last_msg.content:
                    response_text = last_msg.content

    if not response_text:
        response_text = "Movement command executed."

    clean_text = response_text.replace("*", "").replace("#", "")
    print(f"🗣️ Response    : {clean_text}")
    print("─" * 55)

    headers = {
        "X-Agent-Text": urllib.parse.quote(response_text),
        "X-Image-Status": urllib.parse.quote(image_status)
    }

    async def audio_generator():
        communicate = edge_tts.Communicate(clean_text, VOICE, rate=SPEED_RATE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(audio_generator(), media_type="audio/mpeg", headers=headers)
