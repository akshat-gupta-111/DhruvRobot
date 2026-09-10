"""
DhruvOrin - Unified Single Script Architecture
==============================================
Zero LangGraph | Zero LangChain | 100% Native Python

Runs directly on NVIDIA Jetson Orin Nano:
- Optional: Can automatically trigger Kaggle Moondream GPU (--trigger)
- Live Real-time Camera Stream (zero buffer lag)
- Dual-Vision: Local CPU OCR (Tesseract) + Moondream VLM (Local / Kaggle / Cloud)
- Pure Python Conversational Brain (Sliding-window memory, direct LLM calls)
- Native Linux Audio Synthesizer (Edge-TTS via mpv/aplay)
"""

import os
import sys
import cv2
import time
import base64
import asyncio
import platform
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Dict

import httpx
from dotenv import load_dotenv

# Load local environment settings
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

# Ensure Kaggle CLI discovers kaggle.json if in same directory
if (BASE_DIR / "kaggle.json").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(BASE_DIR))

# =====================================================================
# CONFIGURATION
# =====================================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")

# LLM Reasoning Backend: "ollama" (recommended for Jetson) or "azure"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")

# Azure OpenAI Fallback Settings (if LLM_BACKEND="azure")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Moondream VLM Backend: "ollama" (local or Kaggle Ngrok) or "cloud"
MOONDREAM_BACKEND = os.getenv("MOONDREAM_BACKEND", "ollama").lower()
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip("/")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY", "")
MOONDREAM_CLOUD_URL = "https://api.moondream.ai/v1/caption"
MOONDREAM_CLOUD_MODEL = "moondream3.1-9B-A2B"


# =====================================================================
# 1. HARDWARE LAYER: REAL-TIME ZERO-LAG CAMERA STREAM
# =====================================================================
class LiveCameraStream:
    """
    Dedicated daemon thread consuming OpenCV frames continuously.
    Crucial on Jetson Linux (V4L2) to prevent hardware buffer queuing lag.
    """
    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.stopped = False

        if not self.stream.isOpened():
            print(f"⚠️ [Camera]: Warning - Could not open video device index {src}.")
            self.grabbed = False
            self.frame = None
        else:
            self.grabbed, self.frame = self.stream.read()

    def start(self):
        if self.stream.isOpened():
            threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()


# =====================================================================
# 2. VISION SUBSYSTEM: LOCAL OCR + MOONDREAM CAPTIONING
# =====================================================================
def extract_text_locally(frame) -> str:
    """Runs local Tesseract OCR on CPU frame."""
    if frame is None:
        return "No frame available."
    try:
        import pytesseract
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else "No legible text detected."
    except ImportError:
        return "Local OCR skipped (pytesseract not installed)."
    except Exception as e:
        return f"OCR Error: {e}"


async def get_scene_caption(client: httpx.AsyncClient, frame) -> str:
    """Sends image frame to Moondream (Local Ollama, Kaggle Ngrok, or Cloud)."""
    if frame is None:
        return "Camera frame unavailable."

    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        return "Failed to compress frame."
    b64_img = base64.b64encode(buffer).decode("utf-8")

    # Mode A: Ollama API (Local Jetson or Kaggle Ngrok)
    if MOONDREAM_BACKEND == "ollama":
        base_url = NGROK_BASE_URL if NGROK_BASE_URL else OLLAMA_HOST
        endpoint = f"{base_url}/api/generate"
        payload = {
            "model": "moondream",
            "prompt": "Describe the current scene, objects, and people succinctly. Do not attempt to read text.",
            "stream": False,
            "images": [b64_img]
        }
        try:
            res = await client.post(endpoint, json=payload, timeout=25.0)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            return f"Moondream Ollama Error: HTTP {res.status_code}"
        except Exception as e:
            return f"Moondream connection error ({endpoint}): {e}"

    # Mode B: Moondream Cloud API
    else:
        headers = {
            "Content-Type": "application/json",
            "X-Moondream-Auth": MOONDREAM_API_KEY,
            "User-Agent": "DhruvOrin/2.0"
        }
        payload = {
            "model": MOONDREAM_CLOUD_MODEL,
            "image_url": f"data:image/jpeg;base64,{b64_img}",
            "stream": False
        }
        try:
            res = await client.post(MOONDREAM_CLOUD_URL, json=payload, headers=headers, timeout=25.0)
            if res.status_code == 200:
                return res.json().get("caption", "").strip()
            return f"Moondream Cloud Error: HTTP {res.status_code}"
        except Exception as e:
            return f"Moondream Cloud connection error: {e}"


# =====================================================================
# 3. REASONING ENGINE: DHRUV BRAIN (ZERO LANGGRAPH)
# =====================================================================
class DhruvBrain:
    """
    Lean, high-performance conversational brain.
    Replaces LangGraph with pure Python in-memory state and direct LLM calls.
    """
    def __init__(self, max_history_turns: int = 4):
        self.history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns

    def build_system_prompt(self, visual_context: str) -> str:
        return (
            "You are Dhruv, an intelligent, living robotic entity powered by NVIDIA Jetson.\n"
            "You see the environment through a dual-pipeline vision system (visual scene + OCR text).\n\n"
            f"CURRENT VISUAL CONTEXT:\n{visual_context}\n\n"
            "Guidelines for responding:\n"
            "1. Synthesize both scene description and visible text naturally when asked what you see.\n"
            "2. Raw OCR data may contain fragments; infer the actual text based on context.\n"
            "3. If asked to read something, quote the exact detected text.\n"
            "4. Keep responses concise, direct, and natural for spoken voice. Do not use markdown emojis."
        )

    async def generate_response(self, client: httpx.AsyncClient, user_query: str, visual_context: str) -> str:
        system_prompt = self.build_system_prompt(visual_context)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-(self.max_history_turns * 2):])
        messages.append({"role": "user", "content": user_query})

        reply = ""
        # 1. Local Jetson Ollama (Default)
        if LLM_BACKEND == "ollama":
            endpoint = f"{OLLAMA_HOST}/api/chat"
            payload = {
                "model": OLLAMA_LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            try:
                res = await client.post(endpoint, json=payload, timeout=45.0)
                if res.status_code == 200:
                    reply = res.json().get("message", {}).get("content", "").strip()
                else:
                    reply = f"Local LLM Error: HTTP {res.status_code} - {res.text}"
            except Exception as e:
                reply = f"Failed to connect to local Ollama at {endpoint}: {e}"

        # 2. Azure OpenAI Fallback
        elif LLM_BACKEND == "azure":
            endpoint = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
            headers = {
                "api-key": AZURE_OPENAI_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "messages": messages,
                "temperature": 0.2
            }
            try:
                res = await client.post(endpoint, json=payload, headers=headers, timeout=30.0)
                if res.status_code == 200:
                    reply = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    reply = f"Azure OpenAI Error: HTTP {res.status_code} - {res.text}"
            except Exception as e:
                reply = f"Azure connection error: {e}"

        else:
            reply = f"Unknown LLM_BACKEND: {LLM_BACKEND}."

        if reply and not reply.startswith("Failed") and not reply.startswith("Local LLM Error"):
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": reply})
            if len(self.history) > self.max_history_turns * 4:
                self.history = self.history[-(self.max_history_turns * 2):]

        return reply


# =====================================================================
# 4. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Speaks response using edge-tts and local Linux/Jetson audio player."""
    if not AUDIO_ENABLED or not text.strip():
        return

    clean_text = text.replace("*", "").replace("#", "").strip()
    temp_audio = str(BASE_DIR / "temp_dhruv_speech.mp3")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, TTS_VOICE)
        await communicate.save(temp_audio)

        current_os = platform.system()
        player = None

        if current_os == "Linux":
            for candidate in ["mpv", "ffplay", "aplay"]:
                if subprocess.run(["which", candidate], capture_output=True).returncode == 0:
                    player = candidate
                    break
        elif current_os == "Darwin":
            player = "afplay"
        elif current_os == "Windows":
            player = "mpv"

        if player == "mpv":
            cmd = ["mpv", "--no-video", "--really-quiet", temp_audio]
        elif player == "afplay":
            cmd = ["afplay", temp_audio]
        elif player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", temp_audio]
        else:
            cmd = None

        if cmd:
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()
    except Exception as e:
        print(f"⚠️ [TTS Warning]: {e}")
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


# =====================================================================
# 5. KAGGLE TRIGGER AUTOMATION HELPER
# =====================================================================
def maybe_trigger_kaggle():
    """Triggers Kaggle GPU instance if requested via --trigger."""
    try:
        from trigger import trigger_and_stream
        print("🚀 Invoking Kaggle Moondream Trigger Automation...")
        trigger_and_stream(stream_logs=False)
    except Exception as e:
        print(f"⚠️ [Trigger Warning]: Could not auto-trigger Kaggle: {e}")


# =====================================================================
# 6. MAIN INTERACTIVE EXECUTION LOOP
# =====================================================================
async def run_dhruv():
    print("=" * 65)
    print("⚡ DHRUV ROBOT: JETSON ORIN NATIVE STANDALONE PIPELINE")
    print("=" * 65)
    print(f"🧠 LLM Backend      : {LLM_BACKEND.upper()} ({OLLAMA_LLM_MODEL if LLM_BACKEND == 'ollama' else AZURE_DEPLOYMENT})")
    print(f"👁️ Vision Backend   : {MOONDREAM_BACKEND.upper()} ({NGROK_BASE_URL if NGROK_BASE_URL else OLLAMA_HOST})")
    print(f"🔊 Audio Synthesizer: {'ENABLED (' + TTS_VOICE + ')' if AUDIO_ENABLED else 'DISABLED'}")
    print("─" * 65)

    print(f"📷 Initializing camera index {CAMERA_INDEX}...")
    cam = LiveCameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(1.0)

    brain = DhruvBrain(max_history_turns=4)

    async with httpx.AsyncClient(timeout=45.0) as client:
        print("\n✅ Dhruv is online! Type your query below, or 'exit' to quit.\n")

        try:
            while True:
                try:
                    import aioconsole
                    user_query = await aioconsole.ainput("\nYou: ")
                except ImportError:
                    user_query = await asyncio.to_thread(input, "\nYou: ")

                user_query = user_query.strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    break

                # 1. Grab fresh camera frame
                grabbed, frame = cam.read()
                if not grabbed or frame is None:
                    print("⚠️ [Camera]: Frame grab failed, continuing without visual input.")
                    visual_context = "No camera frame available."
                else:
                    print("👁️ [Vision]: Capturing frame & running concurrent OCR + Moondream...")
                    t0 = time.time()

                    ocr_future = asyncio.to_thread(extract_text_locally, frame)
                    caption_future = get_scene_caption(client, frame)

                    ocr_text, scene_caption = await asyncio.gather(ocr_future, caption_future)
                    dt = time.time() - t0

                    print(f"   ├─ OCR Text : {ocr_text[:60]}{'...' if len(ocr_text) > 60 else ''}")
                    print(f"   ├─ Scene    : {scene_caption[:60]}{'...' if len(scene_caption) > 60 else ''}")
                    print(f"   └─ Vision Latency: {dt:.2f}s")

                    visual_context = (
                        f"SCENE DESCRIPTION: {scene_caption}\n"
                        f"VISIBLE TEXT DETECTED: {ocr_text}"
                    )

                # 2. Reasoning via DhruvBrain (Pure Python)
                print("🧠 [Brain]: Reasoning...")
                t_brain = time.time()
                response = await brain.generate_response(client, user_query, visual_context)
                dt_brain = time.time() - t_brain

                print(f"\nDhruv: {response}")
                print(f"⏱️ (Inference Time: {dt_brain:.2f}s)")

                # 3. Audio output
                if AUDIO_ENABLED:
                    await speak_text(response)

        finally:
            print("\nShutting down hardware interfaces...")
            cam.stop()
            print("Dhruv: Offline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DhruvOrin Standalone Pipeline")
    parser.add_argument("--trigger", action="store_true", help="Automatically trigger the Kaggle Moondream server before starting")
    args = parser.parse_args()

    if args.trigger or os.getenv("AUTO_TRIGGER_KAGGLE", "false").lower() in ("true", "1", "yes"):
        maybe_trigger_kaggle()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_dhruv())
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting.")
