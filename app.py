import os
import logging
import hmac
import subprocess
import asyncio
import base64
import tempfile
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from kokoro_onnx import Kokoro
import whisperx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ESL Speech Worker")

kokoro: Kokoro = None
align_model = None
align_metadata = None
whisper_model = None
models_loaded = False
models_lock = asyncio.Lock()

API_KEY = os.getenv("ESL_SPEECH_WORKER_API_KEY", "")
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices-v1.0.bin")
DEFAULT_VOICE = "af_sarah"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny.en")
WHISPER_VAD_METHOD = os.getenv("WHISPER_VAD_METHOD", "silero")

def _resolve_device() -> str:
    forced = os.getenv("ESL_SPEECH_WORKER_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"

device = _resolve_device()

def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def _safe_remove(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)

def _require_api_key(request_obj: Request) -> None:
    if not API_KEY:
        return
    provided = request_obj.headers.get("x-api-key", "")
    if not provided or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

def _encode_audio(
    audio_format: str,
    wav_path: str,
    mp3_quality: Optional[int],
) -> tuple[bytes, str]:
    audio_format = (audio_format or "mp3").lower().strip()
    if audio_format not in {"mp3", "wav"}:
        raise HTTPException(status_code=400, detail="audio_format must be 'mp3' or 'wav'")

    if audio_format == "wav":
        with open(wav_path, "rb") as f:
            return f.read(), "audio/wav"

    try:
        q = int(mp3_quality) if mp3_quality is not None else 4
        q = _clamp(q, 0, 9)
    except Exception:
        q = 4

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        mp3_path = tmp_mp3.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-q:a",
                str(q),
                mp3_path,
            ],
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found (required for mp3 output)")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg mp3 encode failed: {e}")

    try:
        with open(mp3_path, "rb") as f:
            return f.read(), "audio/mpeg"
    finally:
        _safe_remove(mp3_path)

def _align_timestamps(audio, batch_size: int = 16) -> list:
    result = whisper_model.transcribe(audio, batch_size=batch_size)
    segments = result.get("segments") or []
    if not segments:
        logger.warning("No speech detected in audio; skipping alignment.")
        return []

    try:
        result_aligned = whisperx.align(
            segments,
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False
        )
        return result_aligned.get("word_segments") or []
    except Exception as e:
        if "list index out of range" in str(e):
            logger.warning("Alignment failed on short audio; returning empty timestamps.")
            return []
        raise

@app.get("/healthz")
async def healthz():
    """
    Liveness probe: process is up.
    """
    return {
        "ok": True,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": models_loaded,
    }

@app.get("/readyz")
async def readyz():
    """
    Readiness probe: models are loaded and service can handle requests.
    """
    ready = bool(kokoro) and bool(align_model) and bool(whisper_model)
    if not ready:
        raise HTTPException(status_code=503, detail="Not ready: models not loaded")
    return {"ok": True, "models_loaded": ready}

def _load_models():
    global kokoro, align_model, align_metadata, whisper_model, models_loaded
    if models_loaded:
        return
    logger.info(f"Starting up on device: {device}")
    if not API_KEY:
        logger.warning("ESL_SPEECH_WORKER_API_KEY is not set")

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
            device=device,
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

    models_loaded = bool(kokoro) and bool(align_model) and bool(whisper_model)

async def _ensure_models_loaded():
    if models_loaded:
        return
    async with models_lock:
        if models_loaded:
            return
        await asyncio.to_thread(_load_models)

@app.on_event("startup")
async def startup_event():
    await _ensure_models_loaded()

class GenerateRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    speed: Optional[float] = 1.0
    audio_format: Optional[str] = "mp3"
    mp3_quality: Optional[int] = 4

@app.post("/generate")
async def generate_audio(
    request_obj: Request,
    request: GenerateRequest,
):
    if not kokoro or not align_model or not whisper_model:
        await _ensure_models_loaded()
    if not kokoro or not align_model or not whisper_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    _require_api_key(request_obj)

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")

    voice = request.voice or DEFAULT_VOICE

    processed_text = request.text.strip()
    logger.info(f"Generating audio for: {processed_text[:50]}...")

    try:
        speed = request.speed if request.speed is not None else 1.0
        speed = _clamp(float(speed), 0.5, 1.5)

        samples, sample_rate = kokoro.create(
            processed_text,
            voice=request.voice,
            speed=speed,
            lang="en-us"
        )
    except Exception as e:
        logger.error(f"TTS Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        sf.write(tmp_audio.name, samples, sample_rate)
        tmp_path = tmp_audio.name
    try:
        # WhisperX alignment expects segments; we transcribe first, then align.
        audio = whisperx.load_audio(tmp_path)
        
        timestamps = _align_timestamps(audio)
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        _safe_remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Alignment failed: {str(e)}")

    try:
        audio_bytes, mime_type = _encode_audio(
            request.audio_format,
            tmp_path,
            request.mp3_quality,
        )
        audio_format = (request.audio_format or "mp3").lower().strip()
    finally:
        _safe_remove(tmp_path)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "mime_type": mime_type,
        "sample_rate": sample_rate,
        "word_timestamps": timestamps,
        "original_text": request.text,
        "processed_text": processed_text,
    }
