# voxcpm2_worker/handler.py
"""
🚀 RunPod Serverless Worker: VoxCPM 2 (OpenBMB)
Pristine Indonesian Voice Cloning for Julia Laura
Inference timesteps: 8, CFG: 2.0 (Studio Quality)
"""

import os
import sys
import io
import base64
import torch
import soundfile as sf
import runpod
from voxcpm import VoxCPM

REF_WAV = os.environ.get("REF_WAV", "/app/julia_ref_16k.wav")
if not os.path.exists(REF_WAV):
    REF_WAV = "julia_ref_16k.wav"

JULIA_TRANSCRIPT = (
    "Halo guys perkenalin nama aku Julia Laura buat kalian yang mau jualan melalui "
    "media live streaming di aplikasi TikTok, Shopee dan marketplace lainnya. "
    "Saya Julia Laura bisa menjadi host live kalian"
)

print("⏳ Memuat model VoxCPM2 (openbmb/VoxCPM2) ke GPU...")
try:
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("✅ Model VoxCPM2 BERHASIL DIMUAT KE GPU!")
except Exception as e:
    print(f"❌ Error loading VoxCPM2: {e}")
    model = None

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "").strip()

    if not text:
        return {"error": "Text parameter is required"}

    if model is None:
        return {"error": "VoxCPM2 model failed to initialize"}

    cfg_value = float(job_input.get("cfg_value", 2.0))
    inference_timesteps = int(job_input.get("inference_timesteps", 8))

    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Generate audio using exact VoxCPM 2 engine
        wav = model.generate(
            text=text,
            prompt_wav_path=REF_WAV,
            prompt_text=JULIA_TRANSCRIPT,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps
        )

        buf = io.BytesIO()
        sample_rate = getattr(model.tts_model, "sample_rate", 24000)
        sf.write(buf, wav, sample_rate, format="WAV")
        buf.seek(0)

        audio_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "status": "COMPLETED",
            "audio_base64": audio_base64,
            "format": "wav",
            "sample_rate": sample_rate
        }

    except Exception as e:
        print(f"❌ Error generating voice: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless VoxCPM2 Loop...")
    runpod.serverless.start({"handler": handler})
