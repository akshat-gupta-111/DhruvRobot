import edge_tts
import asyncio
import os
import subprocess

# You can change the voice. Some good options:
# Indian English male: "en-IN-PrabhatNeural"
# Indian English female: "en-IN-NeerjaNeural"
# US English male: "en-US-GuyNeural" 
VOICE = "en-IN-NeerjaNeural" 
SPEED_RATE = "+25%" 

async def speak_text(text: str):
    """Generates audio from text using Microsoft Azure Neural TTS and plays it."""
    
    # Remove any markdown formatting (like ** or *) that the LLM might output
    clean_text = text.replace("*", "").replace("#", "")
    
    if not clean_text.strip():
        return

    output_file = "dhruv_response.mp3"
    
    try:
        # Generate the audio file
        communicate = edge_tts.Communicate(clean_text, VOICE, rate=SPEED_RATE)
        await communicate.save(output_file)
        
        # Play the audio. 
        # On Mac, 'afplay' is built-in. If you installed mpv, replace 'afplay' with 'mpv'
        subprocess.run(["afplay", output_file], check=True)
        
    except Exception as e:
        print(f"\n[TTS Error] Could not speak text: {e}")
    finally:
        # Clean up the audio file after playing
        if os.path.exists(output_file):
            os.remove(output_file)