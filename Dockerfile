# voxcpm2_worker/Dockerfile
# Pristine VoxCPM 2 Serverless Worker with Julia Voice Baked-in

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install VoxCPM2, RunPod SDK, and dependencies
RUN pip install --no-cache-dir \
    runpod \
    voxcpm \
    soundfile \
    torchaudio \
    fastapi \
    pydantic

# Pre-download openbmb/VoxCPM2 model during build so workers start instantly
RUN python3 -c "from voxcpm import VoxCPM; print('Pre-downloading model...'); VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False); print('Pre-download complete!')"

# Copy audio reference and handler
COPY julia_ref_16k.wav /app/julia_ref_16k.wav
COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENV REF_WAV=/app/julia_ref_16k.wav

CMD ["python3", "-u", "/app/handler.py"]
