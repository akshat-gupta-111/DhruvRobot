import asyncio
import sys
from aioconsole import ainput
from core.graph import dhruv_brain
from modalities.vision.scene_analyzer import capture_and_analyze
from modalities.audio.tts import speak_text  # <-- Added TTS import
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

live_context = {"current_scene": "Initializing vision..."}

async def chat_loop():
    print("\nDhruv is waking up... (Type 'exit' to quit)\n")
    thread_config = {"configurable": {"thread_id": "1"}} 
    
    while True:
        try:
            # ainput handles terminal input gracefully in async without blocking threads
            user_input = await ainput("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nForce quit detected.")
            break 

        if not user_input.strip():
            continue

        if user_input.lower() in ['exit', 'quit']:
            break
            
        state_input = {
            "messages": [HumanMessage(content=user_input)],
            "current_scene": live_context.get("current_scene", "No vision data")
        }
        
        try:
            for event in dhruv_brain.stream(state_input, config=thread_config):
                for value in event.values():
                    # Extract the response text
                    response_text = value['messages'][-1].content
                    print(f"Dhruv: {response_text}")
                    
                    # <-- Added: Trigger the audio asynchronously
                    await speak_text(response_text)
                    
        except Exception as e:
            print(f"Graph execution error: {e}")

async def main():
    stop_event = asyncio.Event()
    vision_task = asyncio.create_task(capture_and_analyze(live_context, stop_event))
    
    try:
        await chat_loop()
    finally:
        print("\nInitiating system shutdown... (releasing hardware)")
        stop_event.set()
        await asyncio.sleep(1)
        vision_task.cancel()
        print("Dhruv: Offline.")

if __name__ == "__main__":
    try:
        # aioconsole works best when setting the event loop policy on Windows/Mac
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass # Caught gracefully at the highest level