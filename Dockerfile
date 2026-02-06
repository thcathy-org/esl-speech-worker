FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Prevent tzdata from prompting during apt installs
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"

COPY app.py .
# Kokoro assets must be provided (volume mount or baked into image):
# COPY kokoro-v1.0.onnx .
# COPY voices-v1.0.bin .

RUN mkdir -p /cache/huggingface /cache/huggingface/transformers

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
