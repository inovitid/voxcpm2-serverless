# voxcpm2_worker/Dockerfile
# VoxCPM 2 Serverless Worker with Julia Voice Baked-in
#
# Clean CUDA base on purpose: the runpod/pytorch image ships preinstalled packages
# that made pip silently fall back to voxcpm 1.5.0 (wrong 16 kHz decoder, ghost voice).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade pip

# Matched torch stack (CUDA 12.4). torchcodec 0.2.x is the release built for torch 2.6.
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Constraints keep pip from swapping torch out while resolving voxcpm's dependencies.
RUN printf "torch==2.6.0\ntorchaudio==2.6.0\n" > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt \
    voxcpm==2.0.3 \
    torchcodec==0.2.1 \
    runpod \
    soundfile \
    huggingface_hub

# Fail the build instead of shipping a mismatched stack.
RUN python3 -c "import importlib.metadata as m; v = {p: m.version(p) for p in ['voxcpm', 'torch', 'torchaudio', 'torchcodec']}; print(v); assert v['voxcpm'] == '2.0.3', v; assert v['torch'].startswith('2.6.0'), v; assert v['torchaudio'].startswith('2.6.0'), v"

# Pre-download openbmb/VoxCPM2 weights so cold starts skip the download.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

COPY julia_ref.wav /app/julia_ref.wav
COPY julia_ref_16k.wav /app/julia_ref_16k.wav
COPY handler.py /app/handler.py

ENV REF_WAV=/app/julia_ref.wav

CMD ["python3", "-u", "/app/handler.py"]
