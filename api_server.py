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
import time
import random
from enum import Enum
from fastapi.responses import StreamingResponse, Response
from core.graph import dhruv_brain
from memory.history_buffer import VisionHistory
from v2.autonomous_logic import evaluate_scene_agentic, RobotAction
from tools.actuators import drive_vehicle, control_motor, emergency_stop, spin_vehicle, diagonal_movement
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

async def get_scene_description_kaggle(base64_image: str):
    """Pings the Kaggle-deployed Moondream API asynchronously via Ngrok."""
    base_url = os.getenv("NGROK_BASE_URL", "").rstrip('/')
    if not base_url:
        return "Kaggle vision offline: NGROK_BASE_URL not set."
        
    target_endpoint = f"{base_url}/api/generate"
    
    payload = {
        "model": "moondream",
        "prompt": "Describe the current scene, objects, and people. Do not attempt to read text.",
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Kaggle/Ollama style endpoints don't strictly require the auth header
            response = await client.post(target_endpoint, json=payload)
            if response.status_code == 200:
                # The kaggle script uses 'response' key instead of 'caption'
                return response.json().get("response", "").strip()
            return f"Scene unavailable (HTTP {response.status_code})."
    except Exception as e:
        return f"Kaggle Vision offline: {str(e)}"

class RobotState(str, Enum):
    WANDERING = "WANDERING"
    OBSERVING = "OBSERVING"

class RobotStateMachine:
    def __init__(self):
        self.state = RobotState.WANDERING
        self.current_action = RobotAction.continue_forward
        self.last_turn_time = time.time()
        self.observing_start_time = 0.0

robot_sm = RobotStateMachine()

async def agentic_body_loop():
    """Loop B: Agentic Control Loop (The Body)"""
    print("🤖 Agentic Body Loop (Loop B) Started.")
    while True:
        try:
            current_time = time.time()
            
            if robot_sm.state == RobotState.WANDERING:
                # Boredom Timer check
                if current_time - robot_sm.last_turn_time > 10.0:
                    print("🤖 [Boredom Timer] Sniffing around...")
                    robot_sm.current_action = random.choice([RobotAction.turn_left, RobotAction.turn_right])
                    robot_sm.last_turn_time = current_time
                    
                # Execute action
                if robot_sm.current_action == RobotAction.continue_forward:
                    await drive_vehicle.ainvoke({"direction": 'F', "duration_ms": 500, "speed": 150})
                elif robot_sm.current_action == RobotAction.turn_left:
                    await spin_vehicle.ainvoke({"degrees": 90, "direction": 'L'})
                    robot_sm.current_action = RobotAction.continue_forward
                elif robot_sm.current_action == RobotAction.turn_right:
                    await spin_vehicle.ainvoke({"degrees": 90, "direction": 'R'})
                    robot_sm.current_action = RobotAction.continue_forward
                    
            elif robot_sm.state == RobotState.OBSERVING:
                # If observing for more than 5s, go back to wandering
                if current_time - robot_sm.observing_start_time > 5.0:
                    robot_sm.state = RobotState.WANDERING
                    robot_sm.current_action = RobotAction.continue_forward
                    robot_sm.last_turn_time = current_time
                    continue
                    
                # Execute observing action
                if robot_sm.current_action == RobotAction.stop:
                    await emergency_stop.ainvoke({})
                elif robot_sm.current_action == RobotAction.orbit:
                    await spin_vehicle.ainvoke({"degrees": 45, "direction": 'L'})
                elif robot_sm.current_action == RobotAction.explore_closer:
                    await drive_vehicle.ainvoke({"direction": 'F', "duration_ms": 300, "speed": 100})
                elif robot_sm.current_action in [RobotAction.turn_left, RobotAction.turn_right]:
                    await spin_vehicle.ainvoke({"degrees": 90, "direction": 'L' if robot_sm.current_action == RobotAction.turn_left else 'R'})
                    robot_sm.state = RobotState.WANDERING
                    robot_sm.last_turn_time = current_time
                else:
                    await emergency_stop.ainvoke({})
                    
            await asyncio.sleep(0.5)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Body loop error: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs the pre-flight connection tests before the server starts accepting requests
    # await run_system_diagnostics()
    
    # Start Loop B
    body_task = asyncio.create_task(agentic_body_loop())
    
    yield
    print("\n🛑 Shutting down Dhruv Core Server...")
    body_task.cancel()

app = FastAPI(lifespan=lifespan)

# Global history for autonomous mode (Increased to 15 seconds so it remembers what it recently said)
global_history = VisionHistory(max_seconds=15)

@app.post("/autonomous")
async def autonomous_mode(image: UploadFile = File(None)):
    if image is None:
        return Response(status_code=400)
        
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        return Response(status_code=400)
        
    b64_payload = base64.b64encode(image_bytes).decode('utf-8')
    
    # Try Kaggle endpoint first for fast autonomous vision
    scene_description = await get_scene_description_kaggle(b64_payload)
    
    # If Kaggle is offline, seamlessly fallback to the official Moondream Cloud API
    if "Kaggle vision offline" in scene_description or "Kaggle Vision offline" in scene_description:
        print("⚠️ Kaggle endpoint offline. Falling back to Moondream Cloud API...")
        scene_description = await get_scene_description(b64_payload)
    
    combined_context = f"SCENE DESCRIPTION: {scene_description}"
    global_history.add(combined_context)
    history_text = global_history.get_recent_history()
    
    # Set to OBSERVING state since we got an interesting frame
    robot_sm.state = RobotState.OBSERVING
    robot_sm.observing_start_time = time.time()
    
    # Evaluate using the new unified Agentic LLM
    agentic_resp = await evaluate_scene_agentic(history_text)
    
    if agentic_resp:
        robot_sm.current_action = agentic_resp.action
        
        if agentic_resp.speech:
            # Inject what the robot just said and did back into its memory so it doesn't repeat itself!
            global_history.add(f"ROBOT ACTION TAKEN: {agentic_resp.action.value}")
            global_history.add(f"ROBOT SAID: {agentic_resp.speech}")
            
            print(f"\n🤔 [Agentic Mode] Dhruv: {agentic_resp.speech} | Action: {agentic_resp.action.value}")
            clean_text = agentic_resp.speech
            
            headers = {
                "X-Has-Audio": "true",
                "X-Agent-Text": urllib.parse.quote(clean_text)
            }
            
            async def audio_generator():
                communicate = edge_tts.Communicate(clean_text, VOICE, rate=SPEED_RATE)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
                        
            return StreamingResponse(audio_generator(), media_type="audio/mpeg", headers=headers)
                
    headers = {"X-Has-Audio": "false"}
    return Response(content="ok", status_code=200, headers=headers)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)