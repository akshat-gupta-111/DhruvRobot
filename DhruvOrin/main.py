"""
DhruvOrin - Edge-Capture / Cloud-GPU Pipeline
=============================================
Zero Local Ollama | Zero LangGraph | Zero LangChain | 100% Native Python

Architecture:
  - Local Hardware (Jetson Orin Nano): USB/CSI Camera + CPU OCR + Audio Speaker
  - Vision Engine : Moondream running on Kaggle Cloud GPU via Ngrok tunnel
  - Brain Engine  : Llama running on Azure AI (fetched via API) with Kaggle fallback
  - Communication : Direct async HTTP requests (No LangChain/LangGraph overhead)
  - Modes:
      --chat   : Interactive terminal text chat (Keyboard Input -> Cloud AI -> Spoken Voice)
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
# Load .env from DhruvOrin first, then parent repo root
load_dotenv(BASE_DIR / ".env", override=True)
if (BASE_DIR.parent / ".env").exists():
    load_dotenv(BASE_DIR.parent / ".env", override=False)

# Ensure Kaggle CLI discovers kaggle.json
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

# ── Vision Backend: Moondream on Kaggle GPU via Ngrok ────────────────
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip("/")
MOONDREAM_MODEL = os.getenv("MOONDREAM_MODEL", "moondream")

# ── Brain Backend: Llama on Azure AI / Azure OpenAI ──────────────────
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or ""
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_MODEL") or "llama"
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Fallback brain model if Azure credentials are not set
KAGGLE_FALLBACK_MODEL = os.getenv("BRAIN_MODEL", "llama3.2:3b")


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
            for alt_idx in [1, 2]:
                alt_stream = cv2.VideoCapture(alt_idx)
                if alt_stream.isOpened():
                    self.stream = alt_stream
                    src = alt_idx
                    break

        if not self.stream.isOpened():
            print(f"[Camera] Notice: Camera index {src} not detected. Chat will run without live video.")
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
    """Returns True ONLY when user asks about visual surroundings or reading."""
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
# 4. DHRUV DUAL-ENGINE: MOONDREAM (VISION) + AZURE LLAMA (BRAIN)
# =====================================================================
class DhruvBrainEngine:
    """
    Implements the original LangGraph dual-pipeline architecture:
      - Vision: Moondream on Kaggle GPU describes camera frames
      - OCR: Local CPU Tesseract reads visible text
      - Brain: Llama on Azure AI (via API) reasons, engages, and synthesizes
               the scene with deep general knowledge.
    """
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.vision_endpoint = f"{self.ngrok_url}/api/generate"
        self.history: List[Dict[str, str]] = []  # List of {"role": "...", "content": "..."}
        self.last_visual_context: str = "No visual data available."

        # Check whether Azure Llama is configured
        self.use_azure = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)

    def update_ngrok_url(self, new_url: str):
        self.ngrok_url = new_url.rstrip("/")
        self.vision_endpoint = f"{self.ngrok_url}/api/generate"

    async def get_scene_caption(self, client: httpx.AsyncClient, frame, ocr_text: str) -> str:
        """Calls Moondream on Kaggle GPU with the exact original caption prompt."""
        if frame is None or not self.ngrok_url:
            return ""

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return ""

        b64_img = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "model": MOONDREAM_MODEL,
            "prompt": "Describe the current scene, objects, and people. Do not attempt to read text.",
            "stream": False,
            "images": [b64_img]
        }

        try:
            res = await client.post(self.vision_endpoint, json=payload, timeout=30.0)
            if res.status_code == 200:
                scene_desc = res.json().get("response", "").strip()
                self.last_visual_context = (
                    f"SCENE DESCRIPTION: {scene_desc}\n"
                    f"VISIBLE TEXT DETECTED: {ocr_text if ocr_text else 'No legible text found.'}"
                )
                return scene_desc
            else:
                print(f"[Vision Warning]: Moondream returned HTTP {res.status_code}")
        except Exception as e:
            print(f"[Vision Warning]: Could not reach Kaggle Moondream: {e}")
        return ""

    async def query_azure_llama(self, client: httpx.AsyncClient, messages: List[Dict[str, str]]) -> str:
        """Queries Llama deployed on Azure AI / Azure OpenAI."""
        # Support Azure OpenAI, Azure AI Studio, or Serverless API URLs
        if "/chat/completions" in AZURE_OPENAI_ENDPOINT:
            url = AZURE_OPENAI_ENDPOINT
        elif "/deployments/" in AZURE_OPENAI_ENDPOINT:
            url = f"{AZURE_OPENAI_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"
        else:
            url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
            "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}"
        }

        payload = {
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 300
        }

        try:
            res = await client.post(url, json=payload, headers=headers, timeout=40.0)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return "Azure returned an empty response."
            else:
                return f"Azure Llama API Error: HTTP {res.status_code} - {res.text[:150]}"
        except Exception as e:
            return f"Azure Connection Error: {e}"

    async def query_kaggle_brain_fallback(self, client: httpx.AsyncClient, messages: List[Dict[str, str]]) -> str:
        """Fallback to Kaggle Ollama if Azure is not configured."""
        url = f"{self.ngrok_url}/api/chat"
        payload = {
            "model": KAGGLE_FALLBACK_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        try:
            res = await client.post(url, json=payload, timeout=45.0)
            if res.status_code == 200:
                return res.json().get("message", {}).get("content", "").strip()
            return f"Kaggle Brain Error: HTTP {res.status_code}"
        except Exception as e:
            return f"Kaggle Brain Connection Error: {e}"

    async def query(self, client: httpx.AsyncClient, frame, user_query: str, ocr_text: str) -> str:
        # ── Step 1: Update visual context if camera frame is provided ──
        if frame is not None:
            scene_desc = await self.get_scene_caption(client, frame, ocr_text)
            if scene_desc:
                print("   [👁️ Scene Analyzed by Moondream on Kaggle GPU]")

        # ── Step 2: Formulate the exact original LangGraph system prompt ─
        system_prompt = f"""You are Dhruv, an intelligent, living entity. 
You can 'see' your environment through a dual-pipeline vision system that provides both a scene description and raw OCR text.

CURRENT VISUAL CONTEXT:
{self.last_visual_context}

Guidelines for responding:
1. If the user asks what you see, synthesize the SCENE DESCRIPTION and VISIBLE TEXT DETECTED naturally.
2. The VISIBLE TEXT DETECTED might contain typos or fragmented words (it is raw OCR data). Use your intelligence to infer what the text actually says based on the scene context.
3. If the user asks you to read something, quote the text from the VISIBLE TEXT section.
4. If the user asks a general question, answer it directly and intelligently using your broad knowledge.
5. Respond conversationally in 1-3 spoken sentences. Dont use emojis."""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-6:])  # last 3 turns
        messages.append({"role": "user", "content": user_query})

        # ── Step 3: Query Brain Engine (Llama on Azure AI) ────────────
        if self.use_azure:
            raw_reply = await self.query_azure_llama(client, messages)
        else:
            raw_reply = (
                "My Azure Llama Brain API is not configured. "
                "Please add AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to your .env file."
            )
            print("\n⚠️ [Azure Notice]: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is missing in your .env file.")
            print("   Please configure your Azure Llama API credentials in DhruvOrin/.env\n")

        cleaned_reply = clean_for_speech_and_display(raw_reply)

        # Update dialogue history
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": cleaned_reply})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        return cleaned_reply


# =====================================================================
# 5. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Synthesizes natural spoken response and plays it on device speakers."""
    skip_prefixes = ("Cannot connect", "Error communicating", "Azure Llama API Error", "Azure Connection Error")
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
    print("⚡ DHRUV ROBOT: DUAL-PIPELINE ARCHITECTURE")
    print("=" * 65)

    ngrok_url = get_or_trigger_ngrok_url(force_trigger=trigger_requested)
    brain = DhruvBrainEngine(ngrok_url)

    brain_source = f"AZURE AI ({AZURE_DEPLOYMENT})" if brain.use_azure else f"KAGGLE GPU ({KAGGLE_FALLBACK_MODEL})"

    print(f"🧠 Brain Engine (LLM) : {brain_source}")
    print(f"👁️ Vision Engine (VLM): MOONDREAM on Kaggle GPU ({ngrok_url if ngrok_url else 'Not connected'})")
    print(f"💬 Interaction Mode   : {'TEXT CHAT (--chat)' if chat_mode else 'DEFAULT'}")
    print(f"🔊 Spoken Voice       : {'ENABLED (' + TTS_VOICE + ' @ ' + TTS_SPEED + ')' if AUDIO_ENABLED else 'DISABLED'}")
    print("─" * 65)

    # Initialize camera hardware
    cam = LiveCameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(0.5)

    async with httpx.AsyncClient(timeout=45.0) as client:
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

                # 2. Query Dhruv Brain
                print(f"🧠 Dhruv is reasoning via {brain_source}...")
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
