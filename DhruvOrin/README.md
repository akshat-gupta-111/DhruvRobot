# DhruvOrin: Standalone Jetson Edge + Kaggle GPU Architecture
Zero Local Ollama | Zero LangGraph | Zero LangChain | 100% Pure Python

This folder is **completely self-contained**.

### How it works:
- **Local Device (Jetson / PC)**:
  - Captures frames from USB / CSI camera (zero buffer lag).
  - Extracts visible text via fast local CPU OCR (`pytesseract`).
  - Plays spoken responses via speaker (Edge-TTS via `mpv`/`aplay`).
- **Remote Cloud (Kaggle GPU via Ngrok)**:
  - Runs **Ollama** and **Moondream** on a free Kaggle GPU.
  - Exposes port 11434 through a public Ngrok tunnel.
  - Receives the camera frame + OCR text + user query over HTTP and runs high-speed VLM inference.

There is **NO local Ollama** and **NO LangGraph**.

---

## 📁 What's in this folder?

| File | Purpose |
| :--- | :--- |
| `main.py` | **Master runner**: Captures camera, local OCR, queries Kaggle Moondream, and speaks response. |
| `trigger.py` | Kaggle GPU automation: Pushes `llava.py` to Kaggle and extracts the active Ngrok tunnel URL. |
| `llava.py` | Kaggle remote worker: Installs Ollama, pulls Moondream, starts server, and establishes Ngrok tunnel. |
| `kernel-metadata.json` | Kaggle notebook configuration for GPU cloud runs. |
| `kaggle.json` | Kaggle API credentials. |
| `requirements.txt` | Clean, minimal dependencies (no LangGraph, no LangChain, ~60MB footprint). |
| `.env` | Active configuration file with `NGROK_BASE_URL`. |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
# On Jetson Linux:
sudo apt-get install -y tesseract-ocr mpv

# Python packages:
cd DhruvOrin
pip install -r requirements.txt
```

### 2. Start Dhruv

```bash
# If Kaggle is already running:
python main.py

# If you want to automatically trigger Kaggle and discover the URL:
python main.py --trigger
```
