FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - PortAudio and mpg123 for audio processing
# - cmake and build-essential for compiling dlib (face_recognition)
# - libbluetooth-dev for bleak BLE communication
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libbluetooth-dev \
    mpg123 \
    alsa-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy all files into container
COPY . /app

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Run both Server and Client
CMD ["./entrypoint.sh"]
