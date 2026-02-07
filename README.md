# ESL Speech Worker

FastAPI service that generates TTS audio and word-level timestamps (Kokoro + WhisperX).

This service is supporting https://www.funfunspell.com - a free platform for English dictation and learning vocabulary

## Quick call example

```bash
curl -sS -f https://<your-host>/esl-speech-worker/generate \
  -H 'content-type: application/json' \
  -H 'X-API-KEY: <key>' \
  -d '{"text":"Hello world.","voice":"af_sarah","speed":0.9,"audio_format":"mp3"}'
```

## API (POST /generate)

Request JSON:

- `text` (required): input text
- `voice` (optional): voice ID (e.g. `af_sarah`)
- `speed` (optional): 0.5–1.5, default 1.0
- `audio_format` (optional): `mp3` (default) or `wav`
- `mp3_quality` (optional): 0..9 (0 best, 9 worst), used only for `mp3`

If `ESL_SPEECH_WORKER_API_KEY` is set, include header `X-API-KEY: <key>`.

Minimal steps to build and run the service in Kubernetes.

## 1) Build the image (local, for k3s/containerd)

```bash
cd esl-speech-worker
chmod +x scripts/build-and-push.sh
./scripts/build-and-push.sh
```

This builds `localhost/esl-speech-worker:<VERSION>` and imports it into k3s containerd.

## 2) Create PVCs (namespace: esl-speech-worker)

```bash
kubectl apply -f k8s/k8s-esl-rest-pvc.yaml
```

## 3) Copy model assets into the PVC

```bash
kubectl run model-uploader -n esl-speech-worker --image=alpine --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "model-uploader",
      "image": "alpine",
      "command": ["sleep","3600"],
      "volumeMounts": [{"name":"models","mountPath":"/models"}]
    }],
    "volumes": [{"name":"models","persistentVolumeClaim":{"claimName":"esl-speech-worker-models"}}]
  }
}'

kubectl cp -n esl-speech-worker kokoro-v1.0.onnx model-uploader:/models/
kubectl cp -n esl-speech-worker voices-v1.0.bin model-uploader:/models/
kubectl delete pod -n esl-speech-worker model-uploader
```

## 4) Create API key secret (optional but recommended)

```bash
kubectl create secret generic esl-speech-worker-api-key \
  -n esl-speech-worker \
  --from-literal=api-key='<your-secret>'
```

## 5) Deploy (Deployment + Service)

```bash
kubectl apply -f k8s/k8s-esl-rest-deploy.yaml -n esl-speech-worker
```

## 6) Ingress (Traefik)

```bash
kubectl apply -f k8s/k8s-esl-rest-ingress.yaml -n esl-speech-worker
```

Test:

```bash
curl -sS https://homeserver.funfunspell.com/esl-speech-worker/healthz
```

## Notes

- Image tag used in `k8s/k8s-esl-rest-deploy.yaml` must match the local image tag (e.g. `localhost/esl-speech-worker:1.0.1`).
- If `ESL_SPEECH_WORKER_API_KEY` is set, `/generate` requires header `X-API-KEY: <key>`.
- If you rebuild the image with the same tag, Kubernetes will keep the old pod. Either bump the tag and `kubectl apply`, or run `kubectl rollout restart deploy/esl-speech-worker -n esl-speech-worker`.

