"""
DhruvOrin - Pure Cloud-GPU / Edge-Capture Pipeline
==================================================
Zero Local Ollama | Zero LangGraph | Zero LangChain | 100% Native Python

Architecture:
  - Local Hardware: USB/CSI Camera + Local CPU OCR + Audio Speaker
  - Remote Cloud: Ollama + Moondream running 100% on Kaggle GPU via Ngrok tunnel
  - Communication: Direct HTTP POST to Kaggle Ngrok endpoint (/api/generate)
  - Modes:
      --chat   : Interactive terminal text chat (Keyboard Input -> Kaggle GPU -> Spoken Voice Output)
      --trigger: Auto-triggers Kaggle GPU instance and captures Ngrok URL
"""

import os
import sys
import re
import cv2
import time
import base64
import asyncio
import platform
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# Safe UTF-8 console output for Windows / Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env", override=True)

# Ensure Kaggle CLI discovers kaggle.json if in directory
if (BASE_DIR / "kaggle.json").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(BASE_DIR))
elif (BASE_DIR.parent / "Trigger" / "kaggle.json").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(BASE_DIR.parent / "Trigger"))

# =====================================================================
# CONFIGURATION
# =====================================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_SPEED = os.getenv("TTS_SPEED", "+20%")

# Kaggle Ollama Moondream Settings (NO LOCAL OLLAMA)
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip("/")
MOONDREAM_MODEL = "moondream"


# =====================================================================
# 1. HARDWARE LAYER: REAL-TIME ZERO-LAG CAMERA STREAM
# =====================================================================
class LiveCameraStream:
    """
    Dedicated daemon thread consuming OpenCV frames continuously.
    Crucial on Jetson Linux (V4L2) to eliminate video driver queue lag.
    """
    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.stopped = False

        if not self.stream.isOpened():
            print(f"[Camera] Notice: Camera index {src} not detected. Text chat will run without camera.")
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


def get_fresh_frame(cam: LiveCameraStream) -> Tuple[bool, Optional[object]]:
    """Flush stale buffer frames and return the true current live frame."""
    for _ in range(5):
        cam.read()
    return cam.read()


# =====================================================================
# 2. LOCAL VISION LAYER: INTENT CLASSIFIER & FAST CPU OCR
# =====================================================================
_VISION_PATTERNS = [
    r"\b(see|look|watch|view|visual|camera)\b",
    r"\bdescribe\s+(the\s+)?(scene|room|surroundings|view|what|image|picture)\b",
    r"\bwhat\s+(do\s+you|can\s+you)\s+see\b",
    r"\bwhat\s+is\s+(in\s+front|around|holding|visible|this|that|there)\b",
    r"\bwhat('s|\s+is)\s+this\b",
    r"\bwho\s+is\s+(in\s+front|there|this|that)\b",
    r"\bread\s+(the\s+)?(text|sign|label|card|book|screen|writing|words)\b",
    r"\bwhat\s+does\s+it\s+say\b",
    r"\bwhat\s+is\s+written\b",
    r"\bwhat\s+color\b",
]

def is_vision_query(query: str) -> bool:
    """Returns True ONLY when user explicitly asks about their surroundings / vision."""
    q = query.lower().strip()
    return any(re.search(pat, q) for pat in _VISION_PATTERNS)


def extract_text_locally(frame) -> str:
    """Runs local Tesseract OCR on CPU frame to catch written text."""
    if frame is None:
        return ""
    try:
        import pytesseract
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text if text else ""
    except ImportError:
        return ""
    except Exception:
        return ""


# =====================================================================
# 3. RESPONSE FORMATTER & SANITIZER
# =====================================================================
def clean_for_speech_and_display(raw_text: str) -> str:
    """
    Sanitizes raw LLM output into clean, natural conversational spoken English.
    Removes robotic artifacts like '1. Hi', lists, speaker prefixes, and markdown.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Remove speaker labels (e.g., "Dhruv:", "AI:", "Assistant:", "Answer:")
    text = re.sub(r'^(?:Dhruv|AI|Assistant|Bot|Answer|Response)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    # 2. Fix numbered list artifacts (e.g., "1. Hi", "1.Hello", "1) ...")
    # Strip leading numbers like "1. ", "1.Hi", "1) "
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    # Convert middle list numbers into smooth sentence transitions
    text = re.sub(r'[\r\n]+\s*\d+[\.\)]\s*', '. ', text)
    text = re.sub(r'\.\s*\d+[\.\)]\s*', '. ', text)

    # 3. Remove bullet points (-, *, •)
    text = re.sub(r'[\r\n]+\s*[\-\*•]\s*', '. ', text)
    text = re.sub(r'^\s*[\-\*•]\s*', '', text)

    # 4. Remove markdown artifacts (*bold*, _italic_, # headers, `code`, quotes)
    text = re.sub(r'[*#`_~>\[\]]', '', text)

    # 5. Remove any leftover HTML or angle brackets
    text = re.sub(r'<[^>]+>', '', text)

    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Clean up weird duplicate punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    return text


# =====================================================================
# ORIGINAL DHRUV PERSONA & PROMPT (From core/graph.py & api_server.py)
# =====================================================================
ORIGINAL_SYSTEM_PROMPT_TEMPLATE = """You are Dhruv, an intelligent, living entity. 
You can 'see' your environment through a dual-pipeline vision system that provides both a scene description and raw OCR text.

CURRENT VISUAL CONTEXT:
{visual_context}

Guidelines for responding:
1. If the user asks what you see, synthesize the SCENE DESCRIPTION and VISIBLE TEXT DETECTED naturally.
2. The VISIBLE TEXT DETECTED might contain typos or fragmented words (it is raw OCR data). Use your intelligence to infer what the text actually says based on the scene context.
3. If the user asks you to read something, quote the text from the VISIBLE TEXT section.
4. Dont use emojis.
"""

ORIGINAL_MOONDREAM_SCENE_PROMPT = "Describe the current scene, objects, and people. Do not attempt to read text."


# =====================================================================
# 4. KAGGLE MOONDREAM REASONING ENGINE (RUNS 100% ON KAGGLE GPU)
# =====================================================================
class KaggleMoondreamBrain:
    """
    Directly queries the remote Moondream instance running inside Ollama
    on Kaggle GPU via the Ngrok tunnel.
    Preserves dialogue memory and scene context using the exact original
    Dhruv LangGraph architecture and persona.
    """
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.endpoint = f"{self.ngrok_url}/api/generate"
        self.history: List[Tuple[str, str]] = []  # List of (user_query, dhruv_reply)
        self.last_visual_context: str = "No visual data available."

    def update_url(self, new_url: str):
        self.ngrok_url = new_url.rstrip("/")
        self.endpoint = f"{self.ngrok_url}/api/generate"

    @staticmethod
    def _looks_like_garbage(text: str) -> bool:
        """Detect raw tensor/weight output or internal model tokens."""
        if not text or len(text.strip()) < 2:
            return True
        if re.search(r'\[\s*-?\d+\.\d+\s*,', text):
            return True
        if re.fullmatch(r'[!?.\-_\s]+', text.strip()):
            return True
        return False

    async def _query_ollama(self, client: httpx.AsyncClient, prompt: str, b64_img: Optional[str] = None, options: Optional[dict] = None) -> str:
        """Helper to stream response from Kaggle Ollama."""
        payload = {
            "model": MOONDREAM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": options or {"temperature": 0.5, "num_predict": 150}
        }
        if b64_img:
            payload["images"] = [b64_img]

        full_response = ""
        import json as _json
        async with client.stream("POST", self.endpoint, json=payload, timeout=45.0) as res:
            if res.status_code != 200:
                body = await res.aread()
                return f"Kaggle Server Error: HTTP {res.status_code} - {body.decode(errors='replace')}"

            async for raw_line in res.aiter_lines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    chunk = _json.loads(raw_line)
                    token = chunk.get("response", "")
                    full_response += token
                    if chunk.get("done", False):
                        break
                except _json.JSONDecodeError:
                    continue

        return full_response.strip().strip('"').strip("'")

    async def query(self, client: httpx.AsyncClient, frame, user_query: str, ocr_text: str) -> str:
        b64_img = None
        if frame is not None:
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                b64_img = base64.b64encode(buffer).decode("utf-8")

        try:
            # ── Mode A: Visual Query (Camera frame captured) ──────────────
            if b64_img:
                # Use original api_server.py Moondream prompt for scene description
                is_general_scene = any(kw in user_query.lower() for kw in ["see", "scene", "look", "around", "view", "describe"])

                if is_general_scene:
                    scene_prompt = ORIGINAL_MOONDREAM_SCENE_PROMPT
                else:
                    scene_prompt = user_query

                scene_raw = await self._query_ollama(
                    client,
                    prompt=scene_prompt,
                    b64_img=b64_img,
                    options={"temperature": 0.2, "num_predict": 120}
                )

                if self._looks_like_garbage(scene_raw):
                    return "I had trouble recognizing the visual scene. Please try asking again."

                scene_description = clean_for_speech_and_display(scene_raw)

                # Combine context exactly like original api_server.py line 94
                combined_context = (
                    f"SCENE DESCRIPTION: {scene_description}\n"
                    f"VISIBLE TEXT: {ocr_text if ocr_text else 'No legible text found.'}"
                )
                self.last_visual_context = combined_context

                # Synthesize response
                if is_general_scene:
                    if ocr_text and ocr_text.lower() != "no legible text found.":
                        reply = f"I see {scene_description}. Visible text detected is {ocr_text}."
                    else:
                        reply = f"I see {scene_description}."
                else:
                    reply = scene_description

                cleaned_reply = clean_for_speech_and_display(reply)
                self.history.append((user_query, cleaned_reply))
                if len(self.history) > 6:
                    self.history.pop(0)
                return cleaned_reply

            # ── Mode B: Conversational Reasoning (Original core/graph.py) ─
            else:
                system_prompt = ORIGINAL_SYSTEM_PROMPT_TEMPLATE.format(
                    visual_context=self.last_visual_context
                )

                dialogue = [system_prompt, ""]
                for past_q, past_a in self.history[-2:]:
                    dialogue.append(f"User: {past_q}")
                    dialogue.append(f"Dhruv: {past_a}")

                dialogue.append(f"User: {user_query}")
                dialogue.append("Dhruv:")
                full_prompt = "\n".join(dialogue)

                raw_reply = await self._query_ollama(
                    client,
                    prompt=full_prompt,
                    options={
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "num_predict": 120,
                        "stop": ["User:", "\nUser", "Human:", "\n\nUser"]
                    }
                )

                if self._looks_like_garbage(raw_reply):
                    return "I received an unclear signal. Please ask again."

                cleaned_reply = clean_for_speech_and_display(raw_reply)
                self.history.append((user_query, cleaned_reply))
                if len(self.history) > 6:
                    self.history.pop(0)
                return cleaned_reply

        except httpx.ConnectError:
            return f"Cannot connect to Kaggle Ngrok tunnel at {self.endpoint}. Is the Kaggle instance running?"
        except httpx.TimeoutException:
            return "Request to Kaggle Moondream timed out. Kaggle GPU might be busy."
        except Exception as e:
            return f"Error communicating with Kaggle: {e}"


# =====================================================================
# 5. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Synthesizes natural spoken response and plays it on device speakers."""
    skip_prefixes = ("Cannot connect", "Error communicating", "Kaggle Moondream Server Error", "Request to Kaggle")
    if not AUDIO_ENABLED or not text or not text.strip():
        return
    if any(text.startswith(p) for p in skip_prefixes):
        return

    clean_text = clean_for_speech_and_display(text)
    if not clean_text:
        return

    temp_audio = str(BASE_DIR / f"temp_dhruv_{int(time.time() * 1000)}.mp3")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, TTS_VOICE, rate=TTS_SPEED)
        await communicate.save(temp_audio)

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            return

        current_os = platform.system()
        cmd = None

        if current_os == "Linux":
            # Priority: mpv > ffplay > mpg123
            for candidate in ["mpv", "ffplay", "mpg123"]:
                if subprocess.run(["which", candidate], capture_output=True).returncode == 0:
                    if candidate == "mpv":
                        cmd = ["mpv", "--no-video", "--really-quiet", temp_audio]
                    elif candidate == "ffplay":
                        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", temp_audio]
                    elif candidate == "mpg123":
                        cmd = ["mpg123", "-q", temp_audio]
                    break
        elif current_os == "Darwin":
            cmd = ["afplay", temp_audio]
        elif current_os == "Windows":
            if subprocess.run(["where", "mpv"], capture_output=True).returncode == 0:
                cmd = ["mpv", "--no-video", "--really-quiet", temp_audio]
            else:
                cmd = ["powershell", "-c",
                       f"(New-Object Media.SoundPlayer '{temp_audio}').PlaySync()"]

        if cmd:
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()
        else:
            print("[TTS Warning]: No audio player found. Install mpv, ffplay, or mpg123.")

    except Exception as e:
        print(f"[TTS Warning]: {e}")
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


# =====================================================================
# 6. KAGGLE TRIGGER & NGROK RESOLUTION
# =====================================================================
def get_or_trigger_ngrok_url(force_trigger: bool = False) -> str:
    """Resolves active Ngrok URL from .env or queries Kaggle live logs."""
    load_dotenv(BASE_DIR / ".env", override=True)
    active_url = os.getenv("NGROK_BASE_URL", "").rstrip("/")

    if active_url and not force_trigger:
        print(f"[*] Using active Kaggle Ngrok URL from .env: {active_url}")
        return active_url

    try:
        from trigger import trigger_and_get_url
        print("[*] Contacting Kaggle to obtain active Ngrok tunnel...")
        url = trigger_and_get_url(timeout_seconds=60)
        if url:
            return url
    except Exception as e:
        print(f"[!] Trigger Notice: {e}")

    return active_url


# =====================================================================
# 7. MAIN INTERACTIVE EXECUTION LOOP
# =====================================================================
async def run_dhruv(trigger_requested: bool = False, chat_mode: bool = True):
    print("=" * 65)
    print("⚡ DHRUV ROBOT: KAGGLE-GPU CLOUD ARCHITECTURE")
    print("=" * 65)

    ngrok_url = get_or_trigger_ngrok_url(force_trigger=trigger_requested)
    if not ngrok_url:
        print("[!] FATAL: No Ngrok URL available.")
        print("    Please run with '--trigger' or add your Kaggle Ngrok URL to .env as NGROK_BASE_URL.")
        return

    print(f"🌐 Kaggle Ollama URL : {ngrok_url}")
    print(f"🧠 Remote VLM Model  : {MOONDREAM_MODEL} (Running 100% on Kaggle GPU)")
    print(f"💬 Interaction Mode  : {'TEXT CHAT (--chat)' if chat_mode else 'DEFAULT'}")
    print(f"🔊 Spoken Voice      : {'ENABLED (' + TTS_VOICE + ' @ ' + TTS_SPEED + ')' if AUDIO_ENABLED else 'DISABLED'}")
    print("─" * 65)

    # Initialize camera
    cam = LiveCameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(0.5)

    brain = KaggleMoondreamBrain(ngrok_url)

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            ping = await client.get(f"{ngrok_url}/api/tags", timeout=6.0)
            if ping.status_code == 200:
                print("[+] Verified connection to Kaggle Ollama server.")
        except Exception:
            pass

        print("\n💬 [Chat Mode Active]: Type your questions below. Dhruv will respond and speak out loud!")
        print("   (Type 'exit' or 'quit' to close)\n")

        try:
            while True:
                try:
                    import aioconsole
                    user_query = await aioconsole.ainput("You: ")
                except ImportError:
                    user_query = await asyncio.to_thread(input, "You: ")

                user_query = user_query.strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    break

                # 1. Inspect user intent: Only capture camera frame when visual context is requested
                frame = None
                ocr_text = ""
                if is_vision_query(user_query):
                    grabbed, raw_frame = get_fresh_frame(cam)
                    if grabbed and raw_frame is not None:
                        frame = raw_frame
                        ocr_text = extract_text_locally(frame)
                        if ocr_text:
                            print(f"   [Visible Text]: {ocr_text[:60]}{'...' if len(ocr_text) > 60 else ''}")
                        print("   [📷 Fresh camera frame captured]")
                    else:
                        print("   [Camera]: No frame available.")

                # 2. Query Kaggle Moondream over Ngrok
                print("🧠 Dhruv is thinking on Kaggle GPU...")
                t0 = time.time()
                response = await brain.query(client, frame, user_query, ocr_text)
                dt = time.time() - t0

                print(f"\nDhruv: {response}")
                print(f"⏱️ (Response Time: {dt:.2f}s)\n")

                # 3. Speak response out loud via TTS
                if AUDIO_ENABLED:
                    await speak_text(response)

        finally:
            print("\nShutting down hardware interfaces...")
            cam.stop()
            print("Dhruv: Offline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DhruvOrin Standalone Pipeline")
    parser.add_argument("--chat", action="store_true", default=True, help="Run in text chat mode with voice output (Keyboard input + Spoken speech)")
    parser.add_argument("--trigger", action="store_true", help="Trigger Kaggle GPU instance and automatically obtain active Ngrok URL")
    args = parser.parse_args()

    trigger_flag = args.trigger or os.getenv("AUTO_TRIGGER_KAGGLE", "false").lower() in ("true", "1", "yes")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_dhruv(trigger_requested=trigger_flag, chat_mode=args.chat))
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting.")
