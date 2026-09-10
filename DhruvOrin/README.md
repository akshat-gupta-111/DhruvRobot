# DhruvOrin: Standalone Jetson Orin Nano Architecture
Zero LangGraph | Zero LangChain | 100% Pure Python

This folder is **completely self-contained**. You can copy this entire `DhruvOrin` directory directly to your NVIDIA Jetson Orin Nano (or any Linux/Windows/Mac machine) and run the full robot intelligence pipeline with a single command.

---

## 📁 What's in this folder?

| File | Purpose |
| :--- | :--- |
| `main.py` | **Master single-script runner**: Controls the camera, local OCR, Moondream VLM, pure-Python brain, and audio TTS. |
| `trigger.py` | Kaggle GPU automation: Wraps `llava.py` into a notebook, pushes it to Kaggle, and monitors the Ngrok tunnel. |
| `llava.py` | Remote Kaggle worker script: Installs Ollama, pulls Moondream, and starts the Ngrok tunnel. |
| `kernel-metadata.json` | Kaggle notebook configuration for GPU cloud runs. |
| `kaggle.json` | Kaggle API credentials. |
| `requirements.txt` | Clean, minimal dependencies (no LangGraph, no LangChain). |
| `.env` | Active configuration file. |
| `.env.example` | Template for environment variables and model selection. |

---

## 🚀 Quick Start Guide

### 1. System Packages (on Jetson Linux)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr mpv
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running Dhruv

#### Mode A: 100% On-Device (Jetson Local Ollama)
Run Moondream and the LLM locally on your Jetson Orin Nano:
```bash
# Pull models in Ollama (once)
ollama pull moondream
ollama pull qwen2.5:1.5b

# Start Dhruv
python main.py
```

#### Mode B: With Kaggle GPU Moondream Server
Offload vision to a free Kaggle GPU instance while running reasoning on your Jetson:
```bash
# Option 1: Trigger Kaggle automatically
python main.py --trigger

# Option 2: Run trigger manually in a separate terminal
python trigger.py
# (Copy the resulting Ngrok URL into .env as NGROK_BASE_URL)
python main.py
```

---

## ⚙️ Configuration (`.env`)

- `LLM_BACKEND`: `ollama` (default for local Jetson) or `azure`.
- `OLLAMA_LLM_MODEL`: `qwen2.5:1.5b` (or `llama3.2:1b`).
- `MOONDREAM_BACKEND`: `ollama` (local or Kaggle Ngrok) or `cloud`.
- `NGROK_BASE_URL`: Paste your active Ngrok URL if using Kaggle.
- `AUDIO_ENABLED`: `true` or `false`.
