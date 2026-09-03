from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI # Swap with your preferred LLM provider
from core.state import DhruvState
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature = 0
)


def reasoning_node(state: DhruvState):
    """Processes user input alongside the real-time visual context."""
    
    # In core/graph.py, update the system_prompt in reasoning_node:

    system_prompt = f"""You are Dhruv, an intelligent, living entity. 
You can 'see' your environment through a dual-pipeline vision system that provides both a scene description and raw OCR text.

CURRENT VISUAL CONTEXT:
{state.get('current_scene', 'No visual data available.')}

Guidelines for responding:
1. If the user asks what you see, synthesize the SCENE DESCRIPTION and VISIBLE TEXT DETECTED naturally.
2. The VISIBLE TEXT DETECTED might contain typos or fragmented words (it is raw OCR data). Use your intelligence to infer what the text actually says based on the scene context.
3. If the user asks you to read something, quote the text from the VISIBLE TEXT section.
4. Dont use emojis.
"""
    
    # We dynamically prepend the system prompt so the LLM always sees the freshest vision data,
    # without permanently saving the system prompt into the message history array.
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build the state machine
workflow = StateGraph(DhruvState)
workflow.add_node("reason", reasoning_node)
workflow.add_edge(START, "reason")
workflow.add_edge("reason", END)

# <-- NEW: Initialize memory and compile with the checkpointer
memory = MemorySaver()
dhruv_brain = workflow.compile(checkpointer=memory)