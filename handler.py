# voxcpm2_worker/handler.py
"""
RunPod Serverless Worker: VoxCPM 2 (OpenBMB) - Julia Laura voice clone.

Defaults mirror runpod_server.py from the pod that produced the good-sounding voice:
continuation mode (prompt_wav_path + prompt_text), cfg 2.0, 6 timesteps, seed 42.
"""

import base64
import importlib.metadata
import io
import os

import runpod
import soundfile as sf
import torch
from voxcpm import VoxCPM

VOXCPM_VERSION = importlib.metadata.version("voxcpm")

REF_WAV = os.environ.get("REF_WAV", "/app/julia_ref.wav")
if not os.path.exists(REF_WAV):
    for candidate in ("/app/julia_ref.wav", "/app/julia_ref_16k.wav", "julia_ref.wav"):
        if os.path.exists(candidate):
            REF_WAV = candidate
            break

JULIA_TRANSCRIPT = (
    "Halo guys perkenalin nama aku Julia Laura buat kalian yang mau jualan melalui "
    "media live streaming di aplikasi TikTok, Shopee dan marketplace lainnya. "
    "Saya Julia Laura bisa menjadi host live kalian"
)

VALID_MODES = ("continuation", "combined", "reference")

print(f"Loading openbmb/VoxCPM2 with voxcpm=={VOXCPM_VERSION}, ref={REF_WAV}")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SAMPLE_RATE = int(model.tts_model.sample_rate)
print(f"VoxCPM2 ready. Output sample rate: {SAMPLE_RATE} Hz")


def handler(job):
    job_input = job.get("input", {}) or {}
    text = str(job_input.get("text", "")).strip()
    if not text:
        return {"error": "Text parameter is required"}

    mode = job_input.get("mode", "continuation")
    if mode not in VALID_MODES:
        return {"error": f"mode must be one of {VALID_MODES}"}

    cfg_value = float(job_input.get("cfg_value", 2.0))
    inference_timesteps = int(job_input.get("inference_timesteps", 6))
    seed = int(job_input.get("seed", 42))

    kwargs = {"text": text, "cfg_value": cfg_value, "inference_timesteps": inference_timesteps}
    if mode in ("continuation", "combined"):
        kwargs["prompt_wav_path"] = REF_WAV
        kwargs["prompt_text"] = JULIA_TRANSCRIPT
    if mode in ("reference", "combined"):
        kwargs["reference_wav_path"] = REF_WAV

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        wav = model.generate(**kwargs)

        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="WAV")
        audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "status": "COMPLETED",
            "audio_base64": audio_base64,
            "format": "wav",
            "sample_rate": SAMPLE_RATE,
            "duration_sec": round(len(wav) / SAMPLE_RATE, 2),
            "mode": mode,
            "voxcpm_version": VOXCPM_VERSION,
        }
    except Exception as e:
        print(f"Error generating voice: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
