# ESL Speech Worker

FastAPI service that generates TTS audio with **Kokoro ONNX** and produces **word-level timestamps** via **WhisperX alignment**.

## What it does

- **Input**: text (+ options like `speed`, `voice`). If you need “speak punctuation”, transform the text before calling this API.
- **Output**: base64 audio (`mp3` by default, or `wav`) + `word_timestamps`

## Requirements

- Python 3.10+ (tested with 3.12)
- `ffmpeg` available on PATH (**required** for `audio_format="mp3"`)
- Enough RAM/disk for Whisper/WhisperX model caches
- Optional: NVIDIA GPU + CUDA for faster alignment (WhisperX)

## Model assets (required)

This repo does **not** include Kokoro model assets. Place these files next to `app.py` (or point to them via env vars):

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

Download (from `thewh1teagle/kokoro-onnx` model releases):

```bash
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Local run (venv)

```bash
cd esl-speech-worker
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt
pip install "whisperx==3.7.4"

uvicorn app:app --host 0.0.0.0 --port 8000
```

Notes:
- On first start, WhisperX may download models into your cache (can take time).
- The worker preloads WhisperX models at startup so requests are not blocked by downloads.

## API

### `POST /generate`

Request JSON fields:

```json
{
  "text": "Hello world.",
  "voice": "af_sarah",
  "speed": 0.9,
  "audio_format": "mp3",
  "mp3_quality": 4
}
```

- `speed`: 1.0 default, <1.0 slower, >1.0 faster (clamped to 0.5–1.5)
- `audio_format`: `"mp3"` (default) or `"wav"`
- `mp3_quality`: 0..9 (0 best, 9 worst), used only when `audio_format="mp3"`
 
Note: the service no longer transforms punctuation. If you need a “speak punctuation” variant, pre-transform the input text on the caller side.

Response JSON fields (subset):

```json
{
  "audio_base64": "....",
  "audio_format": "mp3",
  "mime_type": "audio/mpeg",
  "sample_rate": 24000,
  "word_timestamps": [
    { "word": "Hello", "start": 0.12, "end": 0.42, "score": 0.87 }
  ]
}
```

## Save audio + timestamps (manual testing)

### Save MP3 + timestamps

```bash
curl -sS -f http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello world. This is an mp3 test.","speed":0.9,"audio_format":"mp3","mp3_quality":4}' \
| python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.mp3","wb").write(base64.b64decode(d["audio_base64"])); open("word_timestamps.json","w",encoding="utf-8").write(json.dumps(d["word_timestamps"], indent=2)); print("wrote out.mp3 and word_timestamps.json")'
```

### Save WAV + timestamps

```bash
curl -sS -f http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello world. This is a wav test.","speed":0.9,"audio_format":"wav"}' \
| python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.wav","wb").write(base64.b64decode(d["audio_base64"])); open("word_timestamps.json","w",encoding="utf-8").write(json.dumps(d["word_timestamps"], indent=2)); print("wrote out.wav and word_timestamps.json")'
```

## Configuration (env vars)

- `KOKORO_MODEL_PATH`: path to `kokoro-*.onnx` (default `kokoro-v1.0.onnx`)
- `KOKORO_VOICES_PATH`: path to `voices-*.bin` (default `voices-v1.0.bin`)
- `WHISPER_MODEL_NAME`: Whisper model name (default `tiny.en`)
- `WHISPER_VAD_METHOD`: WhisperX VAD method (default `silero`)

Example:

```bash
KOKORO_MODEL_PATH=/models/kokoro-v1.0.onnx \
KOKORO_VOICES_PATH=/models/voices-v1.0.bin \
WHISPER_MODEL_NAME=tiny.en \
WHISPER_VAD_METHOD=silero \
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

This folder includes a `Dockerfile`. It installs `ffmpeg` and runs uvicorn.

By default the image **does not COPY** Kokoro assets, so either:

- Mount them at runtime:

```bash
docker build -t esl-speech-worker .
docker run --rm -p 8000:8000 -v "$PWD:/app" esl-speech-worker
```

- Or bake the assets into the image (edit `Dockerfile` and `COPY` them).

## Troubleshooting

- **`503 Models not loaded`**: Kokoro assets missing or WhisperX failed to load. Check logs and ensure the two Kokoro files exist.
- **MP3 output errors**: ensure `ffmpeg` is installed and on PATH.
- **Pyannote / torch weights_only errors**: keep `WHISPER_VAD_METHOD=silero` (default).

