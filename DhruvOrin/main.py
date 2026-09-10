"""
DhruvOrin - Pure Cloud-GPU / Edge-Capture Pipeline
==================================================
Zero Local Ollama | Zero LangGraph | Zero LangChain | 100% Native Python

Architecture:
  - Local Edge (Jetson Orin Nano): USB/CSI Camera + CPU OCR + Audio Speaker
  - Remote Cloud (Kaggle GPU via Ngrok):
      * Vision Engine: Moondream (Scene understanding & visual captioning)
      * Brain Engine : Llama-3.2 3B (Reasoning, general knowledge, LangGraph persona)
  - Communication: Direct async HTTP to Kaggle Ngrok tunnel (/api/generate & /api/chat)
  - Modes:
      --chat   : Interactive terminal text chat (Keyboard Input -> Kaggle GPU -> Spoken Voice Output)
      --trigger: Auto-triggers Kaggle GPU instance and captures Ngrok URL
"""

import os
import sys
import re
import cv2
import time
import json
import base64
import asyncio
import platform
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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

# Kaggle Cloud Models (Running 100% on Kaggle GPU)
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip("/")
BRAIN_MODEL = os.getenv("BRAIN_MODEL", "llama3.2:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")


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
        self.grabbed = False
        self.frame = None

        if not self.stream.isOpened():
            # Try alternate indices
            for alt_idx in [1, 2]:
                alt_stream = cv2.VideoCapture(alt_idx)
                if alt_stream.isOpened():
                    self.stream = alt_stream
                    src = alt_idx
                    break

        if not self.stream.isOpened():
            print(f"[Camera] Notice: Camera index {src} not detected. Running without camera.")
        else:
            self.grabbed, self.frame = self.stream.read()
            if self.grabbed:
                h, w = self.frame.shape[:2]
                print(f"[Camera] Initialized successfully on index {src} ({w}x{h} px).")

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

    def read(self) -> Tuple[bool, Optional[object]]:
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()


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
    """Returns True ONLY when user asks about visual surroundings / reading."""
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
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
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
# 4. KAGGLE DUAL-ENGINE: MOONDREAM (VISION) + LLAMA (BRAIN)
# =====================================================================
class KaggleDhruvBrain:
    """
    Directly mirrors the original LangGraph dual-pipeline architecture:
      - Vision: Moondream Cloud GPU generates scene descriptions from live camera
      - OCR: Local CPU Tesseract extracts visible text
      - Brain: Llama-3.2 / Qwen running on Kaggle GPU synthesizes knowledge,
               persona, conversation, and visual context via /api/chat.
    """
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.chat_endpoint = f"{self.ngrok_url}/api/chat"
        self.generate_endpoint = f"{self.ngrok_url}/api/generate"
        self.history: List[Dict[str, str]] = []  # List of {"role": "...", "content": "..."}
        self.last_visual_context: str = "No visual data available."

    def update_url(self, new_url: str):
        self.ngrok_url = new_url.rstrip("/")
        self.chat_endpoint = f"{self.ngrok_url}/api/chat"
        self.generate_endpoint = f"{self.ngrok_url}/api/generate"

    async def ensure_models(self, client: httpx.AsyncClient):
        """Verifies required models exist on Kaggle Ollama, pulling if needed."""
        try:
            r = await client.get(f"{self.ngrok_url}/api/tags", timeout=10.0)
            if r.status_code == 200:
                installed = [m.get("name", "") for m in r.json().get("models", [])]
                
                # Check Brain Model
                if not any(BRAIN_MODEL in m for m in installed):
                    print(f"[*] Brain model '{BRAIN_MODEL}' not found on Kaggle GPU. Pulling now (~20s)...")
                    await client.post(f"{self.ngrok_url}/api/pull", json={"model": BRAIN_MODEL, "stream": False}, timeout=180.0)
                    print(f"[+] Brain model '{BRAIN_MODEL}' ready on Kaggle.")
                
                # Check Vision Model
                if not any(VISION_MODEL in m for m in installed):
                    print(f"[*] Vision model '{VISION_MODEL}' not found on Kaggle GPU. Pulling now...")
                    await client.post(f"{self.ngrok_url}/api/pull", json={"model": VISION_MODEL, "stream": False}, timeout=180.0)
                    print(f"[+] Vision model '{VISION_MODEL}' ready on Kaggle.")
        except Exception as e:
            print(f"[Model Check Note]: {e}")

    async def update_scene_from_frame(self, client: httpx.AsyncClient, frame, ocr_text: str) -> str:
        """Sends frame to Moondream on Kaggle GPU to produce scene description."""
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return ""

        b64_img = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "model": VISION_MODEL,
            "prompt": "Describe the current scene, objects, and people. Do not attempt to read text.",
            "stream": False,
            "images": [b64_img]
        }

        try:
            res = await client.post(self.generate_endpoint, json=payload, timeout=35.0)
            if res.status_code == 200:
                scene_desc = res.json().get("response", "").strip()
                self.last_visual_context = (
                    f"SCENE DESCRIPTION: {scene_desc}\n"
                    f"VISIBLE TEXT DETECTED: {ocr_text if ocr_text else 'No legible text found.'}"
                )
                return scene_desc
        except Exception as e:
            print(f"[Vision Engine Warning]: {e}")
        return ""

    async def query(self, client: httpx.AsyncClient, frame, user_query: str, ocr_text: str) -> str:
        # ── Step 1: Dual-Pipeline Vision (if camera frame captured) ────
        if frame is not None:
            scene_desc = await self.update_scene_from_frame(client, frame, ocr_text)
            if scene_desc:
                print("   [👁️ Scene Analyzed by Moondream]")

        # ── Step 2: Reasoning Brain (Original LangGraph System Prompt) ──
        system_prompt = f"""You are Dhruv, an intelligent, living robotic companion with broad knowledge and vision.
You can 'see' your environment through a dual-pipeline vision system that provides both a scene description and raw OCR text.

CURRENT VISUAL CONTEXT:
{self.last_visual_context}

Guidelines for responding:
1. For general knowledge, science, people, history, and conversation, answer directly and intelligently.
2. If the user asks what you see or about your environment, synthesize the SCENE DESCRIPTION and VISIBLE TEXT DETECTED naturally.
3. If the user asks you to read something, quote the text from the VISIBLE TEXT section.
4. Keep answers conversational, natural, and concise (1-3 sentences).
5. Do not use emojis."""

        # Build ChatML message list (exact LangGraph state machine structure)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-6:])  # Include last 3 conversation turns for memory
        messages.append({"role": "user", "content": user_query})

        payload = {
            "model": BRAIN_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        try:
            full_response = ""
            async with client.stream("POST", self.chat_endpoint, json=payload, timeout=45.0) as res:
                if res.status_code != 200:
                    body = await res.aread()
                    return f"Kaggle Brain Server Error: HTTP {res.status_code} - {body.decode(errors='replace')}"

                async for raw_line in res.aiter_lines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                        token = chunk.get("message", {}).get("content", "")
                        full_response += token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            reply = full_response.strip().strip('"').strip("'")
            cleaned_reply = clean_for_speech_and_display(reply)

            # Record history for multi-turn dialogue memory
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": cleaned_reply})
            if len(self.history) > 12:
                self.history = self.history[-12:]

            return cleaned_reply

        except httpx.ConnectError:
            return f"Cannot connect to Kaggle Ngrok tunnel at {self.chat_endpoint}. Is Kaggle running?"
        except httpx.TimeoutException:
            return "Request to Kaggle GPU timed out. Kaggle GPU might be busy."
        except Exception as e:
            return f"Error communicating with Kaggle: {e}"


# =====================================================================
# 5. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Synthesizes natural spoken response and plays it on device speakers."""
    skip_prefixes = ("Cannot connect", "Error communicating", "Kaggle Brain Server Error", "Request to Kaggle")
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
    print("⚡ DHRUV ROBOT: DUAL-PIPELINE KAGGLE-GPU ARCHITECTURE")
    print("=" * 65)

    ngrok_url = get_or_trigger_ngrok_url(force_trigger=trigger_requested)
    if not ngrok_url:
        print("[!] FATAL: No Ngrok URL available.")
        print("    Please run with '--trigger' or add your Kaggle Ngrok URL to .env as NGROK_BASE_URL.")
        return

    print(f"🌐 Kaggle Ollama URL : {ngrok_url}")
    print(f"🧠 Brain Model (LLM) : {BRAIN_MODEL} (Running 100% on Kaggle GPU)")
    print(f"👁️ Vision Model (VLM): {VISION_MODEL} (Running 100% on Kaggle GPU)")
    print(f"💬 Interaction Mode  : {'TEXT CHAT (--chat)' if chat_mode else 'DEFAULT'}")
    print(f"🔊 Spoken Voice      : {'ENABLED (' + TTS_VOICE + ' @ ' + TTS_SPEED + ')' if AUDIO_ENABLED else 'DISABLED'}")
    print("─" * 65)

    # Initialize camera hardware
    cam = LiveCameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(0.5)

    brain = KaggleDhruvBrain(ngrok_url)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Verify and ensure Kaggle has both models ready
        await brain.ensure_models(client)

        print("\n💬 [Dhruv Online]: Type your questions below. Dhruv will respond and speak out loud!")
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

                # 1. Dual-Pipeline Vision Check:
                # Capture camera frame when the user asks about their environment/surroundings
                frame = None
                ocr_text = ""
                if is_vision_query(user_query):
                    grabbed, raw_frame = cam.read()
                    if grabbed and raw_frame is not None:
                        frame = raw_frame
                        ocr_text = extract_text_locally(frame)
                        if ocr_text:
                            print(f"   [Visible Text]: {ocr_text[:60]}{'...' if len(ocr_text) > 60 else ''}")
                        print("   [📷 Fresh camera frame captured]")
                    else:
                        print("   [Camera]: Frame not available from video device.")

                # 2. Query Kaggle Brain over Ngrok
                print("🧠 Dhruv is reasoning on Kaggle GPU...")
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
