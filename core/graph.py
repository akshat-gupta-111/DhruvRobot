import os
import cv2
import numpy as np
import base64
import asyncio
import httpx
import pytesseract
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from core.state import DhruvState
from tools.actuators import drive_vehicle, control_motor, emergency_stop, spin_vehicle, diagonal_movement
from dotenv import load_dotenv

load_dotenv()
# 1. Models & Tools
tools = [drive_vehicle, control_motor, emergency_stop, spin_vehicle, diagonal_movement]


llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature = 0
)
llm_with_tools = llm.bind_tools(tools)

# 2. Vision Helper Functions
def run_ocr(image_bytes: bytes) -> str:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else "No legible text found."
    except Exception as e:
        return f"OCR Error: {e}"

async def run_moondream(base64_image: str) -> str:
    """Fetches scene description using the official Moondream Cloud API."""
    api_url = os.getenv("MOONDREAM_API_URL", "https://api.moondream.ai/v1/caption")
    api_key = os.getenv("MOONDREAM_API_KEY", "")
    model_name = os.getenv("MOONDREAM_MODEL", "moondream3.1-9B-A2B")
    
    # Ensure the payload includes the required data URI prefix
    image_data_uri = base64_image
    if not image_data_uri.startswith("data:image"):
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"
        
    headers = {
        "Content-Type": "application/json",
        "X-Moondream-Auth": api_key,
    }
    
    payload = {
        "model": model_name,
        "image_url": image_data_uri,
        "stream": False,
    }
    
    try:
        # 30-second timeout is plenty for a dedicated cloud API
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(api_url, json=payload, headers=headers)
            
            if res.status_code == 200:
                caption = res.json().get("caption", "").strip()
                if caption:
                    return caption
                return "Moondream returned an empty description."
            else:
                return f"Moondream API Error {res.status_code}: {res.text}"
                
    except httpx.TimeoutException:
        return "Moondream API timed out."
    except Exception as e:
        return f"Network error reaching Moondream: {e}"

# 3. Graph Nodes

class RouteQuery(BaseModel):
    intent: Literal["movement", "perception", "hybrid"] = Field(
        description=(
            "Categorize as 'movement' if purely asking to drive/move motors. "
            "Categorize as 'perception' if purely asking about what is visible. "
            "Categorize as 'hybrid' if requiring both (e.g., 'drive toward the desk')."
        )
    )

def intent_router(state: DhruvState):
    query = state["messages"][-1].content
    router_llm = llm.with_structured_output(RouteQuery)
    decision = router_llm.invoke(query)
    print(f"\n[Supervisor] Query routed to: {decision.intent.upper()} cortex.")
    return {"intent": decision.intent}

async def vision_processor(state: DhruvState):
    """Executes Moondream + OCR ONLY when routed through perception or hybrid."""
    b64_img = state.get("raw_image_b64")
    if not b64_img:
        return {"current_scene": "No visual frame provided."}

    print("🧠 [Perception Node] Executing OCR and Moondream VLM...")
    img_bytes = base64.b64decode(b64_img)
    ocr_task = asyncio.to_thread(run_ocr, img_bytes)
    vlm_task = run_moondream(b64_img)

    ocr_text, scene = await asyncio.gather(ocr_task, vlm_task)
    context = f"SCENE DESCRIPTION: {scene}\nVISIBLE TEXT: {ocr_text}"
    return {"current_scene": context}

def movement_worker(state: DhruvState):
    """Calls actuator tools immediately without vision overhead."""
    sys_msg = SystemMessage(content=(
        "You are Dhruv's actuator control cortex. "
        "Translate the user request into an explicit tool call (drive_vehicle or control_motor). "
        "Always specify direction, speed, and finite duration_ms (max 5000ms)."
    ))
    # LLM will invoke drive_vehicle or control_motor
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def perception_worker(state: DhruvState):
    sys_msg = SystemMessage(content=(
        f"You are Dhruv's perception cortex.\n"
        f"CURRENT VISUAL CONTEXT:\n{state.get('current_scene', 'No visual data.')}\n"
        "Answer the user based strictly on this visual environment."
    ))
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def hybrid_worker(state: DhruvState):
    sys_msg = SystemMessage(content=(
        f"You are Dhruv's embodied cognitive brain.\n"
        f"CURRENT VISUAL CONTEXT:\n{state.get('current_scene', 'No visual data.')}\n"
        "Evaluate the environment before issuing any movement tool call. "
        "AUTONOMOUS SEARCH RULE: If the user asks you to 'move until' a target is found, check the visual context. "
        "1. If the target is NOT seen, issue a short movement tool call (e.g. 1000ms), AND end your text response exactly with: [SEARCH_CONTINUES]\n"
        "2. If the target IS seen, do not move. End your text response exactly with: [TARGET_FOUND]"
    ))
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# 4. Construct the Workflow
workflow = StateGraph(DhruvState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("vision_processor", vision_processor)
workflow.add_node("movement", movement_worker)
workflow.add_node("perception", perception_worker)
workflow.add_node("hybrid", hybrid_worker)
workflow.add_node("tools", ToolNode(tools))

# Conditional branching from router
workflow.add_edge(START, "intent_router")

def route_decision(state: DhruvState):
    if state["intent"] == "movement":
        return "movement"  # Skips vision entirely!
    return "vision_processor"

workflow.add_conditional_edges("intent_router", route_decision, {
    "movement": "movement",
    "vision_processor": "vision_processor"
})

def route_after_vision(state: DhruvState):
    return "perception" if state["intent"] == "perception" else "hybrid"

workflow.add_conditional_edges("vision_processor", route_after_vision, {
    "perception": "perception",
    "hybrid": "hybrid"
})

# Tool calling logic for movement and hybrid nodes
workflow.add_conditional_edges("movement", tools_condition, {"tools": "tools", END: END})
workflow.add_conditional_edges("hybrid", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("perception", END)

def route_after_tools(state: DhruvState):
    """Routes the tool execution results back to the cortex that requested them."""
    if state["intent"] == "movement":
        return "movement"
    return "hybrid"

workflow.add_conditional_edges("tools", route_after_tools, {
    "movement": "movement",
    "hybrid": "hybrid"
})

dhruv_brain = workflow.compile()