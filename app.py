import os
import logging
import hmac
import subprocess
import asyncio
import base64
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ESL Speech Worker")

kokoro: Kokoro = None
models_loaded = False
models_lock = asyncio.Lock()

API_KEY = os.getenv("ESL_SPEECH_WORKER_API_KEY", "")
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices-v1.0.bin")
DEFAULT_VOICE = "af_sarah"
MAX_TTS_WORDS_PER_CHUNK = int(os.getenv("MAX_TTS_WORDS_PER_CHUNK", "50"))

def _models_ready() -> bool:
    return bool(kokoro)

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

@app.get("/healthz")
async def healthz():
    """
    Liveness probe: process is up.
    """
    return {
        "ok": True,
        "models_loaded": models_loaded,
    }

@app.get("/readyz")
async def readyz():
    """
    Readiness probe: models are loaded and service can handle requests.
    """
    ready = _models_ready()
    if not ready:
        raise HTTPException(status_code=503, detail="Not ready: models not loaded")
    return {"ok": True, "models_loaded": ready}

def _load_models():
    global kokoro, models_loaded
    if models_loaded:
        return
    logger.info("Starting up (CPU-only mode)")
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

    models_loaded = _models_ready()

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

def _parse_speed(value: Optional[float]) -> float:
    try:
        raw = value if value is not None else 1.0
        return _clamp(float(raw), 0.5, 1.5)
    except Exception:
        raise HTTPException(status_code=400, detail="speed must be a number")

def _validate_and_normalize_text(text: str) -> str:
    processed_text = (text or "").strip()
    if not processed_text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    return processed_text

def _split_text_for_tts(text: str) -> list[str]:
    words = (text or "").split()
    if not words:
        return []

    chunk_size = max(1, MAX_TTS_WORDS_PER_CHUNK)
    word_chunks = [
        words[i:i + chunk_size]
        for i in range(0, len(words), chunk_size)
    ]

    if len(word_chunks) > 1 and len(word_chunks[-1]) < 10:
        word_chunks[-2].extend(word_chunks[-1])
        word_chunks.pop()

    return [" ".join(chunk) for chunk in word_chunks]

def _resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples

    if samples.ndim == 1:
        source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0])
        target_length = max(1, int(round(samples.shape[0] * target_rate / source_rate)))
        target_positions = np.linspace(0, samples.shape[0] - 1, num=target_length)
        return np.interp(target_positions, source_positions, samples).astype(np.float32, copy=False)

    if samples.ndim == 2:
        source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0])
        target_length = max(1, int(round(samples.shape[0] * target_rate / source_rate)))
        target_positions = np.linspace(0, samples.shape[0] - 1, num=target_length)
        channels = [
            np.interp(target_positions, source_positions, samples[:, channel_idx])
            for channel_idx in range(samples.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32, copy=False)

    return samples.astype(np.float32, copy=False)

async def _require_models_loaded() -> None:
    if not _models_ready():
        await _ensure_models_loaded()
    if not _models_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")

def _generate_audio_sync(
    processed_text: str,
    voice: str,
    speed: float,
    audio_format: str,
    mp3_quality: Optional[int],
) -> tuple[bytes, str, int, str]:
    text_chunks = _split_text_for_tts(processed_text)

    logger.info(f"Splitting TTS request into {len(text_chunks)} chunks to avoid phoneme truncation")

    all_samples: list[np.ndarray] = []
    sample_rate: Optional[int] = None

    try:
        for chunk in text_chunks:
            chunk_samples, chunk_sample_rate = kokoro.create(
                chunk,
                voice=voice,
                speed=speed,
                lang="en-us"
            )
            chunk_samples_np = np.asarray(chunk_samples, dtype=np.float32)
            if sample_rate is None:
                sample_rate = int(chunk_sample_rate)
            elif sample_rate != chunk_sample_rate:
                logger.warning(
                    f"Resampling chunk from {chunk_sample_rate} Hz to {sample_rate} Hz"
                )
                chunk_samples_np = _resample_audio(
                    chunk_samples_np,
                    int(chunk_sample_rate),
                    sample_rate,
                )
            all_samples.append(chunk_samples_np)

        if not all_samples:
            raise HTTPException(status_code=500, detail="No audio generated")

        samples = all_samples[0] if len(all_samples) == 1 else np.concatenate(all_samples)
    except Exception as e:
        logger.error(f"TTS Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        sf.write(tmp_audio.name, samples, sample_rate)
        tmp_path = tmp_audio.name

    try:
        audio_bytes, mime_type = _encode_audio(
            audio_format,
            tmp_path,
            mp3_quality,
        )
        normalized_audio_format = (audio_format or "mp3").lower().strip()
    except Exception as e:
        logger.error(f"Audio post-processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
    finally:
        _safe_remove(tmp_path)

    return audio_bytes, mime_type, sample_rate, normalized_audio_format

@app.post("/generate")
async def generate_audio(
    request_obj: Request,
    request: GenerateRequest,
):
    await _require_models_loaded()
    _require_api_key(request_obj)

    voice = request.voice or DEFAULT_VOICE
    processed_text = _validate_and_normalize_text(request.text)
    logger.info(f"Generating audio for: {processed_text[:50]}...")

    speed = _parse_speed(request.speed)

    audio_bytes, mime_type, sample_rate, audio_format = await asyncio.to_thread(
        _generate_audio_sync,
        processed_text,
        voice,
        speed,
        request.audio_format,
        request.mp3_quality,
    )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "mime_type": mime_type,
        "sample_rate": sample_rate,
        "original_text": request.text,
        "processed_text": processed_text,
    }
