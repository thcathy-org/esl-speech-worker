import os
import logging
import subprocess
import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from kokoro_onnx import Kokoro
import whisperx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ESL Speech Worker")

kokoro: Kokoro = None
align_model = None
align_metadata = None
whisper_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Kokoro assets are not shipped via PyPI; you must provide them.
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices-v1.0.bin")
DEFAULT_VOICE = "af_sarah"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny.en")
WHISPER_VAD_METHOD = os.getenv("WHISPER_VAD_METHOD", "silero")

@app.get("/healthz")
async def healthz():
    """
    Liveness probe: process is up.
    """
    return {
        "ok": True,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }

@app.get("/readyz")
async def readyz():
    """
    Readiness probe: models are loaded and service can handle requests.
    """
    ready = bool(kokoro) and bool(align_model) and bool(whisper_model)
    if not ready:
        raise HTTPException(status_code=503, detail="Not ready: models not loaded")
    return {"ok": True}

@app.on_event("startup")
async def startup_event():
    global kokoro, align_model, align_metadata, whisper_model
    
    logger.info(f"Starting up on device: {device}")
    
    model_candidates = [
        KOKORO_MODEL_PATH,
        "kokoro-v1.0.onnx",
        "kokoro-v0_19.onnx",
    ]
    voices_candidates = [
        KOKORO_VOICES_PATH,
        "voices-v1.0.bin",
        "voices.json",
    ]

    model_path = next((p for p in model_candidates if p and os.path.exists(p)), None)
    voices_path = next((p for p in voices_candidates if p and os.path.exists(p)), None)

    if model_path and voices_path:
        kokoro = Kokoro(model_path, voices_path)
        logger.info(f"Kokoro TTS loaded. model={model_path} voices={voices_path}")
    else:
        logger.warning(
            "Kokoro model/voices files not found. TTS will fail until you provide them. "
            f"Looked for model in {model_candidates} and voices in {voices_candidates}."
        )

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en", 
            device=device
        )
        logger.info("WhisperX Alignment model loaded.")
    except Exception as e:
        logger.error(f"Failed to load WhisperX: {e}")

    try:
        compute_type = "float16" if device == "cuda" else "int8"
        whisper_model = whisperx.load_model(
            WHISPER_MODEL_NAME,
            device,
            compute_type=compute_type,
            vad_method=WHISPER_VAD_METHOD,
        )
        logger.info(
            f"Whisper model loaded: {WHISPER_MODEL_NAME} "
            f"(compute_type={compute_type}, vad_method={WHISPER_VAD_METHOD})"
        )
    except Exception as e:
        logger.error(f"Failed to load Whisper model {WHISPER_MODEL_NAME}: {e}")

class GenerateRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    speak_punctuation: Optional[bool] = False
    speed: Optional[float] = 1.0
    audio_format: Optional[str] = "mp3"
    mp3_quality: Optional[int] = 4

@app.post("/generate")
async def generate_audio(
    request: GenerateRequest,
):
    if not kokoro or not align_model or not whisper_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    processed_text = request.text
    logger.info(f"Generating audio for: {processed_text[:50]}...")

    try:
        speed = request.speed if request.speed is not None else 1.0
        speed = float(speed)
        speed = max(0.5, min(1.5, speed))

        samples, sample_rate = kokoro.create(
            processed_text,
            voice=request.voice,
            speed=speed,
            lang="en-us"
        )
    except Exception as e:
        logger.error(f"TTS Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        sf.write(tmp_audio.name, samples, sample_rate)
        tmp_path = tmp_audio.name
    tmp_mp3_path = None

    try:
        # WhisperX alignment expects segments; we transcribe first, then align.
        batch_size = 16 

        audio = whisperx.load_audio(tmp_path)
        
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        
        result_aligned = whisperx.align(
            result["segments"], 
            align_model, 
            align_metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        timestamps = result_aligned["word_segments"]
        
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Alignment failed: {str(e)}")

    audio_format = (request.audio_format or "mp3").lower().strip()
    if audio_format not in {"mp3", "wav"}:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="audio_format must be 'mp3' or 'wav'")

    if audio_format == "wav":
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        mime_type = "audio/wav"
    else:
        try:
            q = int(request.mp3_quality) if request.mp3_quality is not None else 4
            q = max(0, min(9, q))
        except Exception:
            q = 4

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    tmp_path,
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    str(q),
                    tmp_mp3_path,
                ],
                check=True,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="ffmpeg not found (required for mp3 output)")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"ffmpeg mp3 encode failed: {e}")

        with open(tmp_mp3_path, "rb") as f:
            audio_bytes = f.read()
        mime_type = "audio/mpeg"
    
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    if os.path.exists(tmp_path): os.remove(tmp_path)
    if tmp_mp3_path and os.path.exists(tmp_mp3_path): os.remove(tmp_mp3_path)

    return {
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "mime_type": mime_type,
        "sample_rate": sample_rate,
        "word_timestamps": timestamps,
        "original_text": request.text,
        "processed_text": processed_text
    }
