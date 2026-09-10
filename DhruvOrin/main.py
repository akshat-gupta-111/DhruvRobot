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
import cv2
import time
import base64
import asyncio
import platform
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Optional

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

# =====================================================================
# CONFIGURATION
# =====================================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")

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


# =====================================================================
# 2. LOCAL VISION LAYER: FAST CPU OCR
# =====================================================================
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
# 3. KAGGLE MOONDREAM REASONING ENGINE (RUNS 100% ON KAGGLE GPU)
# =====================================================================
class KaggleMoondreamBrain:
    """
    Directly queries the remote Moondream instance running inside Ollama
    on Kaggle GPU via the Ngrok tunnel.
    """
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.endpoint = f"{self.ngrok_url}/api/generate"
        self.context: Optional[List[int]] = None

    def update_url(self, new_url: str):
        self.ngrok_url = new_url.rstrip("/")
        self.endpoint = f"{self.ngrok_url}/api/generate"

    @staticmethod
    def _looks_like_garbage(text: str) -> bool:
        """Detect raw tensor/weight output instead of natural language."""
        if not text or len(text.strip()) < 2:
            return True
        # Raw weights look like lists of floats: [0.0, 0.12, 0.38, ...]
        import re
        if re.search(r'\[\s*-?\d+\.\d+\s*,', text):
            return True
        # Pure punctuation / tokens like !!!
        if re.fullmatch(r'[!?.\-_\s]+', text.strip()):
            return True
        return False

    async def query(self, client: httpx.AsyncClient, frame, user_query: str, ocr_text: str) -> str:
        b64_img = None
        if frame is not None:
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                b64_img = base64.b64encode(buffer).decode("utf-8")

        # Build a clear, concise prompt Moondream can follow
        if b64_img:
            # Vision query — keep it short and direct for Moondream
            full_prompt = (
                f"You are Dhruv, a helpful robot. "
                f"{('Text visible in scene: ' + ocr_text + '. ') if ocr_text else ''}"
                f"User asked: {user_query}\n"
                f"Describe what you see and answer in 1-3 clear sentences."
            )
        else:
            # Text-only chat — conversational assistant
            full_prompt = (
                f"You are Dhruv, a friendly and intelligent robotic companion. "
                f"Answer the following in 1-3 natural spoken sentences without markdown or bullet points.\n"
                f"User: {user_query}\nDhruv:"
            )

        # Use stream=True so we can read NDJSON line-by-line (more reliable with Ollama)
        payload = {
            "model": MOONDREAM_MODEL,
            "prompt": full_prompt,
            "stream": True,
            "options": {"temperature": 0.7, "num_predict": 200}
        }
        if b64_img:
            payload["images"] = [b64_img]
        if self.context:
            payload["context"] = self.context

        try:
            import json as _json
            full_response = ""
            async with client.stream("POST", self.endpoint, json=payload, timeout=45.0) as res:
                if res.status_code != 200:
                    body = await res.aread()
                    return f"Kaggle Moondream Server Error: HTTP {res.status_code} - {body.decode(errors='replace')}"
                async for raw_line in res.aiter_lines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        chunk = _json.loads(raw_line)
                        token = chunk.get("response", "")
                        full_response += token
                        if chunk.get("done", False):
                            ctx = chunk.get("context")
                            if ctx:
                                self.context = ctx
                            break
                    except _json.JSONDecodeError:
                        continue

            reply = full_response.strip().strip('"').strip("'")

            if self._looks_like_garbage(reply):
                return "I received an unclear signal from the AI. Please try asking again."

            return reply

        except httpx.ConnectError:
            return f"Cannot connect to Kaggle Ngrok tunnel at {self.endpoint}. Is the Kaggle instance running?"
        except httpx.TimeoutException:
            return "Request to Kaggle Moondream timed out. Kaggle GPU might be busy."
        except Exception as e:
            return f"Error communicating with Kaggle: {e}"


# =====================================================================
# 4. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Synthesizes voice response and plays it on device speakers."""
    skip_prefixes = ("Cannot connect", "Error communicating", "Kaggle Moondream Server Error", "Request to Kaggle")
    if not AUDIO_ENABLED or not text or not text.strip():
        return
    if any(text.startswith(p) for p in skip_prefixes):
        return

    # Strip markdown artifacts
    import re
    clean_text = re.sub(r'[*#`_~]', '', text).strip()
    if not clean_text:
        return

    temp_audio = str(BASE_DIR / "temp_dhruv_speech.mp3")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, TTS_VOICE)
        await communicate.save(temp_audio)

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            print("[TTS Warning]: Audio file was empty — check that the text is valid.")
            return

        current_os = platform.system()
        cmd = None

        if current_os == "Linux":
            # Priority: mpv > ffplay > mpg123 (mpg123 natively decodes mp3, aplay cannot)
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
            # Use powershell media player as fallback if mpv not installed
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
# 5. KAGGLE TRIGGER & NGROK RESOLUTION
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
# 6. MAIN INTERACTIVE EXECUTION LOOP
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
    print(f"🔊 Spoken Voice      : {'ENABLED (' + TTS_VOICE + ')' if AUDIO_ENABLED else 'DISABLED'}")
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

                # 1. Capture camera frame (if camera is active)
                frame = None
                ocr_text = ""
                grabbed, raw_frame = cam.read()
                if grabbed and raw_frame is not None:
                    frame = raw_frame
                    ocr_text = extract_text_locally(frame)
                    if ocr_text:
                        print(f"   [Visible Text]: {ocr_text[:60]}{'...' if len(ocr_text) > 60 else ''}")

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
