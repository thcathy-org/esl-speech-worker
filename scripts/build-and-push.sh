#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [tag]"
  echo ""
  echo "Examples:"
  echo "  $0"
  echo "  $0 1.0.3"
  echo "  IMAGE_NAME=ghcr.io/thcathy-org/esl-speech-worker $0"
  echo "  IMAGE_NAME=ghcr.io/thcathy-org/esl-speech-worker $0 1.0.3"
  echo ""
  echo "Required env for login (or login separately beforehand):"
  echo "  GHCR_USER, GHCR_TOKEN"
}

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  if [[ ! -f VERSION ]]; then
    echo "VERSION file not found. Provide an explicit tag or create VERSION." >&2
    exit 1
  fi
  TAG="$(tr -d ' \t\r\n' < VERSION)"
  if [[ -z "$TAG" ]]; then
    echo "VERSION file is empty. Provide an explicit tag." >&2
    exit 1
  fi
fi

IMAGE_NAME="${IMAGE_NAME:-esl-speech-worker}"
TAR_PATH="${TAR_PATH:-/tmp/esl-speech-worker.tar}"
ENGINE="podman"
if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found. Install podman before running this script." >&2
  exit 1
fi

echo "Building image ${IMAGE_NAME}:${TAG}"
${ENGINE} build -t "${IMAGE_NAME}:${TAG}" .

echo "Saving image to ${TAR_PATH}"
if [[ -f "$TAR_PATH" ]]; then
  rm -f "$TAR_PATH"
fi
${ENGINE} save "${IMAGE_NAME}:${TAG}" -o "${TAR_PATH}"

echo "Importing image into k3s containerd"
sudo k3s ctr images import "${TAR_PATH}"

echo "Done. Image imported into k3s: ${IMAGE_NAME}:${TAG}"
