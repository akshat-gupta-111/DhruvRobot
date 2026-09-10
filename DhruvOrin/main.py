"""
DhruvOrin - Autonomous Exploration & Voice Command Pipeline
============================================================
Zero Local Ollama | Zero LangGraph | Zero LangChain | 100% Native Python

Modes & Architecture:
  1. 🔭 CONTINUOUS EXPLORATION MODE (Default):
     - Continuously analyzes surroundings using Kaggle Moondream (Vision) + Local CPU OCR.
     - Generates natural, engaging observations using Azure GPT-4o.
     - Speaks observations out loud through speakers.
  2. ⚡ COMMAND ACCEPTING MODE:
     - Triggered anytime Dhruv hears: "Listen Dhruv !" (or "Hey Dhruv" / typing in console).
     - Pauses exploration, responds: "Yes, I am listening!"
     - Accepts user query/action (via voice or keyboard), executes with full GPT-4o knowledge.
     - Speaks the response, then automatically returns to Exploration Mode!
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
load_dotenv(BASE_DIR / ".env", override=True)
if (BASE_DIR.parent / ".env").exists():
    load_dotenv(BASE_DIR.parent / ".env", override=False)

# Ensure Kaggle CLI discovers kaggle.json
if (BASE_DIR / "kaggle.json").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(BASE_DIR))
elif (BASE_DIR.parent / "Trigger" / "kaggle.json").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(BASE_DIR.parent / "Trigger"))

# Known Figures & Facial Recognition Engine with MediaPipe Hands & JSON Profile Lookup
try:
    from test_known import JarvisVisionEngine
except ImportError:
    try:
        from DhruvOrin.test_known import JarvisVisionEngine
    except ImportError:
        JarvisVisionEngine = None

# =====================================================================
# CONFIGURATION
# =====================================================================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() in ("true", "1", "yes")
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_SPEED = os.getenv("TTS_SPEED", "+20%")

# ── Known Figures & People Recognition ────────────────────────────────
KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", str(BASE_DIR / "known_faces"))
PEOPLE_CONTEXT_PATH = os.getenv("PEOPLE_CONTEXT_PATH", str(BASE_DIR / "people_context.json"))

# ── Vision Backend: Moondream on Kaggle GPU via Ngrok ────────────────
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip("/")
MOONDREAM_MODEL = os.getenv("MOONDREAM_MODEL", "moondream")

# ── Brain Backend: Azure OpenAI / AI Foundry (gpt-4o or llama) ───────
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or ""
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_MODEL") or "gpt-4o"
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# Global lock to prevent microphone from picking up Dhruv's own speech
is_dhruv_speaking = False


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
            print(f"[Camera] Notice: Camera index {src} not detected. Running without live video.")
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
# 2. LOCAL VISION LAYER: FAST CPU OCR & INTENT DETECTION
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
    text = re.sub(r'^(?:Dhruv|AI|Assistant|Bot|Answer|Response)\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    text = re.sub(r'[\r\n]+\s*\d+[\.\)]\s*', '. ', text)
    text = re.sub(r'\.\s*\d+[\.\)]\s*', '. ', text)
    text = re.sub(r'[\r\n]+\s*[\-\*•]\s*', '. ', text)
    text = re.sub(r'^\s*[\-\*•]\s*', '', text)
    text = re.sub(r'[*#`_~>\[\]]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text


# =====================================================================
# 4. AUDIO SUBSYSTEM: JETSON-NATIVE TTS
# =====================================================================
async def speak_text(text: str):
    """Synthesizes natural spoken response and plays it on device speakers."""
    global is_dhruv_speaking
    skip_prefixes = ("Cannot connect", "Error communicating", "Azure API Error", "Azure Connection Error")
    if not AUDIO_ENABLED or not text or not text.strip():
        return
    if any(text.startswith(p) for p in skip_prefixes):
        return

    clean_text = clean_for_speech_and_display(text)
    if not clean_text:
        return

    temp_audio = str(BASE_DIR / f"temp_dhruv_{int(time.time() * 1000)}.mp3")

    try:
        is_dhruv_speaking = True  # Mute microphone listener during self-speech
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
        is_dhruv_speaking = False  # Re-enable microphone listener
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


# =====================================================================
# 5. VOICE TRIGGER & MICROPHONE LISTENER ("Listen Dhruv !")
# =====================================================================
class WakeWordListener:
    """
    Background microphone listener monitoring for the wake phrase:
    'Listen Dhruv !' (or 'Hey Dhruv', 'Dhruv').
    Ignores sound while Dhruv is actively speaking.
    """
    WAKE_TRIGGERS = [
        "listen dhruv", "listen through", "listen dhrub",
        "listen to dhruv", "hey dhruv", "hi dhruv", "ok dhruv", "okay dhruv"
    ]

    def __init__(self, on_wake_callback):
        self.callback = on_wake_callback
        self.stop_listening = None
        self.active = False
        self.recognizer = None
        self.mic = None

    def start(self):
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 280
            self.recognizer.dynamic_energy_threshold = True
            self.mic = sr.Microphone()

            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)

            def audio_callback(recognizer, audio):
                global is_dhruv_speaking
                if is_dhruv_speaking or not self.active:
                    return

                try:
                    text = recognizer.recognize_google(audio).lower().strip()
                    # Check for wake word matches
                    if any(t in text for t in self.WAKE_TRIGGERS):
                        print(f"\n⚡ [Wake Word Detected via Mic]: \"{text}\"")
                        self.callback()
                except (sr.UnknownValueError, sr.RequestError):
                    pass
                except Exception:
                    pass

            self.active = True
            self.stop_listening = self.recognizer.listen_in_background(
                self.mic, audio_callback, phrase_time_limit=4
            )
            print("🎙️ [Microphone Listener]: Active. Say 'Listen Dhruv!' to give a command.")
        except Exception as e:
            print(f"[Notice]: Microphone listener not active ({e}). Type 'Listen Dhruv' in terminal.")

    def listen_for_command_voice(self, timeout_sec: int = 6) -> Optional[str]:
        """Listens directly for a user command after the wake word is triggered."""
        global is_dhruv_speaking
        try:
            import speech_recognition as sr
            if not self.recognizer or not self.mic:
                return None

            print("🎤 [Listening for your command...]")
            with self.mic as source:
                audio = self.recognizer.listen(source, timeout=timeout_sec, phrase_time_limit=8)
            text = self.recognizer.recognize_google(audio).strip()
            print(f"🗣️ [You said]: \"{text}\"")
            return text
        except Exception:
            return None

    def stop(self):
        self.active = False
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)


# =====================================================================
# 6. DHRUV DUAL-ENGINE: MOONDREAM (VISION) + AZURE (BRAIN)
# =====================================================================
class DhruvBrainEngine:
    """
    Dual-pipeline reasoning engine:
      - Vision: Moondream on Kaggle GPU describes visual surroundings
      - OCR: Local CPU Tesseract extracts visible text
      - Brain: Azure OpenAI / AI Foundry (gpt-4o) reasons & answers
    """
    def __init__(self, ngrok_url: str):
        self.ngrok_url = ngrok_url.rstrip("/")
        self.vision_endpoint = f"{self.ngrok_url}/api/generate"
        self.history: List[Dict[str, str]] = []
        self.last_visual_context: str = "No visual data available."
        self.past_observations: List[str] = []

    def update_ngrok_url(self, new_url: str):
        self.ngrok_url = new_url.rstrip("/")
        self.vision_endpoint = f"{self.ngrok_url}/api/generate"

    async def get_scene_caption(self, client: httpx.AsyncClient, frame, ocr_text: str = "", figure_context_str: str = "") -> str:
        """Calls Moondream on Kaggle GPU to describe the current video frame."""
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
                    f"PEOPLE & FIGURES IN SCENE: {figure_context_str if figure_context_str else 'No recognized figures.'}\n"
                    f"VISIBLE TEXT DETECTED: {ocr_text if ocr_text else 'No legible text found.'}"
                )
                return scene_desc
        except Exception as e:
            print(f"[Vision Warning]: {e}")
        return ""

    async def generate_exploration_observation(self, client: httpx.AsyncClient, scene_desc: str, ocr_text: str = "", figure_context_str: str = "") -> str:
        """Generates a lively, natural observation about the surroundings in exploration mode."""
        past_str = "\n".join([f"- {obs}" for obs in self.past_observations[-3:]]) if self.past_observations else "None yet."

        prompt = f"""You are Dhruv, an intelligent living robotic entity exploring your physical environment.

CURRENT SENSORY INPUT:
Scene: {scene_desc}
Recognized People / Figures: {figure_context_str if figure_context_str else 'None'}
Visible Text: {ocr_text if ocr_text else 'None'}

Previous observations you already shared:
{past_str}

Guidelines for this observation:
1. Speak a single, natural 1-2 sentence spoken observation about what you notice right now.
2. If a RECOGNIZED PERSON or FIGURE is in view, prioritize acknowledging them personally using their biographical context! Tailor your interaction (e.g., greet your creator/mentor warmly and mention their work or passion).
3. CRITICAL: If you already greeted this person in recent observations, DO NOT keep repeating the greeting; instead, comment on what they are doing, their hand gestures (e.g. fingers held up), or the surrounding scene.
4. Focus on an interesting detail, object, person, activity, or subtle change.
5. Sound lively, observant, and curious.
6. Do NOT repeat the exact sentences or ideas from previous observations.
7. Do not use emojis or bullet points."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "What do you notice around you right now?"}
        ]

        obs = await self.query_azure_llm(client, messages, temperature=0.7, max_tokens=150)
        obs = clean_for_speech_and_display(obs)
        if obs:
            self.past_observations.append(obs)
            if len(self.past_observations) > 8:
                self.past_observations.pop(0)
        return obs

    async def query_azure_llm(self, client: httpx.AsyncClient, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 300) -> str:
        """Queries Azure OpenAI / AI Foundry endpoint with messages."""
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
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            res = await client.post(url, json=payload, headers=headers, timeout=35.0)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return "Azure returned an empty response."
            else:
                return f"Azure API Error: HTTP {res.status_code} - {res.text[:120]}"
        except Exception as e:
            return f"Azure Connection Error: {e}"

    async def execute_command(self, client: httpx.AsyncClient, frame, user_query: str, ocr_text: str = "", figure_context_str: str = "") -> str:
        """Executes user command with full LangGraph persona, visual awareness, and person context."""
        if frame is not None:
            await self.get_scene_caption(client, frame, ocr_text, figure_context_str)

        human_ctx = f"\nPEOPLE / FIGURES PRESENT:\n{figure_context_str}" if figure_context_str else ""

        system_prompt = f"""You are Dhruv, an intelligent, living robotic companion.
You can 'see' your environment through a multi-modal vision system providing scene descriptions, facial recognition with known identity profiles, hand gesture tracking, and raw OCR text.

CURRENT VISUAL & HUMAN CONTEXT:
{self.last_visual_context}{human_ctx}

Guidelines for responding:
1. If a known person is detected in front of you, you know exactly who you are speaking to! Use their profile and background from the context to personalize your response, greeting, and relationship.
2. If the user asks what you see, who is in front of you, or asks about hand gestures/fingers, synthesize the SCENE DESCRIPTION, RECOGNIZED PEOPLE, and HAND GESTURES naturally.
3. The VISIBLE TEXT DETECTED might contain typos or fragmented words (raw OCR data). Use your intelligence to infer what the text actually says based on the scene context.
4. If the user asks you to read something, quote the text from the VISIBLE TEXT section.
5. If the user asks a general question, answer it directly and intelligently using your broad knowledge.
6. Respond conversationally in 1-3 spoken sentences. Dont use emojis."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": user_query})

        raw_reply = await self.query_azure_llm(client, messages, temperature=0.3, max_tokens=300)
        cleaned_reply = clean_for_speech_and_display(raw_reply)

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": cleaned_reply})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        return cleaned_reply


# =====================================================================
# 7. KAGGLE TRIGGER & NGROK RESOLUTION
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
# 8. MAIN INTERACTIVE EXECUTION LOOP (EXPLORATION + COMMAND MODES)
# =====================================================================
async def run_dhruv(trigger_requested: bool = False):
    print("=" * 68)
    print("⚡ DHRUV ROBOT: AUTONOMOUS EXPLORATION & COMMAND PIPELINE")
    print("=" * 68)

    ngrok_url = get_or_trigger_ngrok_url(force_trigger=trigger_requested)
    brain = DhruvBrainEngine(ngrok_url)

    # Initialize Known Figures & Face Recognition Engine
    vision_engine = None
    if JarvisVisionEngine is not None:
        try:
            vision_engine = JarvisVisionEngine(
                known_faces_dir=KNOWN_FACES_DIR,
                context_json_path=PEOPLE_CONTEXT_PATH
            )
        except Exception as e:
            print(f"[Vision Engine Notice]: Could not initialize face engine: {e}")

    active_profiles = len(vision_engine.known_face_names) if vision_engine else 0
    print(f"🧠 Brain Engine (LLM)   : AZURE AI ({AZURE_DEPLOYMENT})")
    print(f"👁️ Vision Engine (VLM)  : MOONDREAM on Kaggle ({ngrok_url if ngrok_url else 'Not connected'})")
    print(f"👤 Known Figures Engine : {f'ACTIVE ({active_profiles} profiles loaded)' if active_profiles > 0 else 'Active (0 reference faces)' if vision_engine else 'DISABLED'}")
    print(f"🔊 Spoken Voice         : {'ENABLED (' + TTS_VOICE + ' @ ' + TTS_SPEED + ')' if AUDIO_ENABLED else 'DISABLED'}")
    print("─" * 68)
    print("Mode 1: 🔭 CONTINUOUS EXPLORATION (Observing & speaking scene details)")
    print("Mode 2: ⚡ COMMAND ACCEPTING (Say 'Listen Dhruv !' to interrupt)")
    print("=" * 68 + "\n")

    # Initialize camera hardware
    cam = LiveCameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(0.5)

    # Event signaling wake word was heard
    wake_event = asyncio.Event()

    def on_wake_triggered():
        wake_event.set()

    # Start microphone wake word listener
    voice_listener = WakeWordListener(on_wake_triggered)
    voice_listener.start()

    async with httpx.AsyncClient(timeout=45.0) as client:
        # Initial greeting
        start_msg = "Dhruv online. Entering exploration mode. Say 'Listen Dhruv!' anytime to give me a command."
        print(f"Dhruv: {start_msg}\n")
        if AUDIO_ENABLED:
            await speak_text(start_msg)

        try:
            while True:
                # =========================================================
                # STATE A: EXPLORATION MODE (Active until wake word triggered)
                # =========================================================
                if not wake_event.is_set():
                    print("🔭 [Exploration Mode]: Analyzing surroundings...")
                    frame = None
                    ocr_text = ""
                    figure_context_str = ""

                    grabbed, raw_frame = cam.read()
                    if grabbed and raw_frame is not None:
                        frame = raw_frame
                        ocr_text = extract_text_locally(frame)
                        if vision_engine is not None:
                            try:
                                fingers, names, contexts, figure_context_str, _ = vision_engine.process_frame(frame)
                                recognized_known = [n for n in names if n != "Unknown"]
                                if recognized_known:
                                    print(f"👤 [Spotted Known Figure]: {', '.join(recognized_known)}")
                                    for rk in recognized_known:
                                        if rk in contexts:
                                            print(f"   📖 [Bio]: {contexts[rk]}")
                                if fingers > 0:
                                    print(f"   🖐️ [Gestures]: {fingers} fingers held up")
                            except Exception as e:
                                print(f"[Face Recognition Warning]: {e}")

                    # Get scene description from Moondream on Kaggle GPU
                    scene_desc = await brain.get_scene_caption(client, frame, ocr_text, figure_context_str)

                    if scene_desc and not wake_event.is_set():
                        # Generate a fresh 1-2 sentence lively observation
                        observation = await brain.generate_exploration_observation(client, scene_desc, ocr_text, figure_context_str)
                        if observation and not wake_event.is_set():
                            print(f"\n🔭 Dhruv Observes: \"{observation}\"\n")
                            if AUDIO_ENABLED:
                                await speak_text(observation)

                    # Pause between exploration cycles (8 seconds), but wake immediately if user speaks
                    try:
                        await asyncio.wait_for(wake_event.wait(), timeout=8.0)
                    except asyncio.TimeoutError:
                        pass

                # =========================================================
                # STATE B: COMMAND ACCEPTING MODE (Triggered by 'Listen Dhruv !')
                # =========================================================
                if wake_event.is_set():
                    wake_event.clear()
                    print("\n" + "⚡" * 30)
                    print("⚡ COMMAND ACCEPTING MODE ACTIVATED")
                    print("⚡" * 30)

                    ack = "Yes, I am listening!"
                    print(f"Dhruv: {ack}\n")
                    if AUDIO_ENABLED:
                        await speak_text(ack)

                    # Prompt user query: check voice microphone first, fallback to keyboard
                    user_command = None
                    if voice_listener.active:
                        user_command = await asyncio.to_thread(voice_listener.listen_for_command_voice, 6)

                    # If voice wasn't heard or no mic, prompt console
                    if not user_command:
                        print("💬 [Listening]: Speak into mic or type your command below:")
                        try:
                            import aioconsole
                            user_command = await asyncio.wait_for(aioconsole.ainput("You (Command): "), timeout=15.0)
                        except asyncio.TimeoutError:
                            print("[Timeout]: No command received. Returning to exploration mode.")
                        except Exception:
                            user_command = await asyncio.to_thread(input, "You (Command): ")

                    user_command = (user_command or "").strip()

                    if user_command:
                        if user_command.lower() in ("exit", "quit", "q"):
                            print("Exit command received.")
                            break

                        # Grab fresh live frame for visual commands
                        frame = None
                        ocr_text = ""
                        figure_context_str = ""
                        grabbed, raw_frame = cam.read()
                        if grabbed and raw_frame is not None:
                            frame = raw_frame
                            ocr_text = extract_text_locally(frame)
                            if vision_engine is not None:
                                try:
                                    fingers, names, contexts, figure_context_str, _ = vision_engine.process_frame(frame)
                                    recognized_known = [n for n in names if n != "Unknown"]
                                    if recognized_known:
                                        print(f"👤 [Command from]: {', '.join(recognized_known)}")
                                except Exception as e:
                                    print(f"[Face Recognition Warning]: {e}")

                        print("🧠 Dhruv is reasoning on your command...")
                        t0 = time.time()
                        response = await brain.execute_command(client, frame, user_command, ocr_text, figure_context_str)
                        dt = time.time() - t0

                        print(f"\nDhruv: {response}")
                        print(f"⏱️ (Response Time: {dt:.2f}s)\n")

                        if AUDIO_ENABLED:
                            await speak_text(response)

                    # Resume exploration
                    resume_msg = "Resuming exploration."
                    print(f"\n[🔭 {resume_msg}]\n")
                    if AUDIO_ENABLED:
                        await speak_text(resume_msg)
                    await asyncio.sleep(1.0)

        finally:
            print("\nShutting down hardware interfaces...")
            voice_listener.stop()
            cam.stop()
            if vision_engine is not None:
                vision_engine.release()
            print("Dhruv: Offline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DhruvOrin Autonomous Pipeline")
    parser.add_argument("--trigger", action="store_true", help="Trigger Kaggle GPU instance and automatically obtain active Ngrok URL")
    args = parser.parse_args()

    trigger_flag = args.trigger or os.getenv("AUTO_TRIGGER_KAGGLE", "false").lower() in ("true", "1", "yes")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_dhruv(trigger_requested=trigger_flag))
    except KeyboardInterrupt:
        print("\nShutdown signal received. Exiting.")
