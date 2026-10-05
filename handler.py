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
import torchaudio

# Ensure torchaudio doesn't fail if torchcodec is missing
_orig_load = getattr(torchaudio, "load", None)
def safe_load(filepath, *args, **kwargs):
    if _orig_load is not None:
        try:
            return _orig_load(filepath, *args, **kwargs)
        except Exception:
            pass
    data, sr = sf.read(filepath)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = safe_load

from voxcpm import VoxCPM

REF_WAV = os.environ.get("REF_WAV", "/app/julia_ref_16k.wav")
if not os.path.exists(REF_WAV):
    if os.path.exists("/app/julia_ref_16k.wav"):
        REF_WAV = "/app/julia_ref_16k.wav"
    elif os.path.exists("/app/julia_ref.wav"):
        REF_WAV = "/app/julia_ref.wav"
    elif os.path.exists("julia_ref_16k.wav"):
        REF_WAV = "julia_ref_16k.wav"
    elif os.path.exists("julia_ref.wav"):
        REF_WAV = "julia_ref.wav"

JULIA_TRANSCRIPT = (
    "Halo guys perkenalin nama aku Julia Laura buat kalian yang mau jualan melalui "
    "media live streaming di aplikasi TikTok, Shopee dan marketplace lainnya. "
    "Saya Julia Laura bisa menjadi host live kalian"
)

STYLE_PROMPT = "(A young, lively and energetic Indonesian female live commerce host speaking fast, cheerful, and enthusiastic tone)"

print(f"⏳ Memuat model VoxCPM2 (openbmb/VoxCPM2) ke GPU... (Ref: {REF_WAV})")
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

    cfg_value = float(job_input.get("cfg_value", 2.5))
    inference_timesteps = int(job_input.get("inference_timesteps", 8))
    use_style = job_input.get("use_style", True)

    clean_text = text.strip()
    if use_style and not clean_text.startswith("("):
        styled_text = f"{STYLE_PROMPT} {clean_text}"
    else:
        styled_text = clean_text

    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # 1. First attempt: Isolated Voice Cloning via reference_wav_path
        # This clones Julia's timbre without continuation concatenation (no extra 15s prompt prepended)
        wav = None
        try:
            print(f"🎙️ Generating with reference_wav_path ({REF_WAV})...")
            wav = model.generate(
                text=styled_text,
                reference_wav_path=REF_WAV,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps
            )
        except Exception as ref_err:
            print(f"⚠️ reference_wav_path failed, trying prompt continuation: {ref_err}")
            wav = model.generate(
                text=clean_text,
                prompt_wav_path=REF_WAV,
                prompt_text=JULIA_TRANSCRIPT,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps
            )

        # 2. Output is 48kHz audio from VoxCPM2 vocoder
        sample_rate = 48000
        buf = io.BytesIO()
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
