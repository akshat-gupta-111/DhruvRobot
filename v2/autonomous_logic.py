import os
from enum import Enum
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

class RobotAction(str, Enum):
    explore_closer = "explore_closer"
    turn_left = "turn_left"
    turn_right = "turn_right"
    continue_forward = "continue_forward"
    orbit = "orbit"
    stop = "stop"

class AgenticResponse(BaseModel):
    speech: str = Field(description="What the robot says out loud. Keep it short, conversational, and highly curious.")
    action: RobotAction = Field(description="The physical motor action the robot should take next.")

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature=0.7 # Higher temperature for more creative/curious comments
)

agentic_llm = llm.with_structured_output(AgenticResponse)

async def evaluate_scene_agentic(history_text: str) -> AgenticResponse:
    """Invoked when the vision loop sends an interesting frame."""
    sys_msg = SystemMessage(content=(
        "You are the brain of a curious, friendly explorer robot. "
        "You are looking at this camera frame sequence (history). "
        "Respond with two keys: speech (what you say out loud) and action "
        "(what you want your wheels to do next: 'explore_closer', 'turn_left', 'turn_right', 'continue_forward', 'orbit', 'stop'). "
        "If you see an interesting object, speak about it and choose an appropriate action like 'explore_closer' or 'orbit'. "
        "If you see a dead-end or wall, say something about it and choose 'turn_left' or 'turn_right'. "
        "If KNOWN PEOPLE IN SCENE are listed in your context, use their names and contextual details to greet them personally and tailor your conversation to them! "
        "Keep speech short and conversational. Do not use asterisks or hashtags."
    ))
    msg = HumanMessage(content=f"VISUAL HISTORY:\n{history_text}")
    
    try:
        response = await agentic_llm.ainvoke([sys_msg, msg])
        # Clean up speech
        response.speech = response.speech.replace("*", "").replace("#", "")
        return response
    except Exception as e:
        print(f"Agentic evaluation failed: {e}")
        return AgenticResponse(speech="", action=RobotAction.continue_forward)
