#!/usr/bin/env bash
# Build the VietASR server image for amd64 + arm64 and push to Docker Hub.
#
# Usage:
#   docker login                      # once
#   scripts/publish-docker.sh [version]
#
# version defaults to __version__ in server/vietasr_server/__init__.py.
# Override the repo with VIETASR_IMAGE, the arches with VIETASR_PLATFORMS.
#
# Note: building linux/arm64 on an amd64 host uses QEMU emulation and is slow.
# If binfmt/QEMU is not set up, run once:
#   docker run --privileged --rm tonistiigi/binfmt --install all
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${VIETASR_IMAGE:-dangvansam/viet-asr}"
PLATFORMS="${VIETASR_PLATFORMS:-linux/amd64,linux/arm64}"
VERSION="${1:-$(sed -n 's/^__version__ = "\(.*\)"/\1/p' server/vietasr_server/__init__.py)}"
VERSION="${VERSION:-dev}"

echo "==> image     : $IMAGE"
echo "==> version   : $VERSION"
echo "==> platforms : $PLATFORMS"

# Multi-platform --push needs the docker-container buildx driver.
if ! docker buildx inspect vietasr-builder >/dev/null 2>&1; then
    docker buildx create --name vietasr-builder --driver docker-container >/dev/null
fi

docker buildx build \
    --builder vietasr-builder \
    --platform "$PLATFORMS" \
    --file server/Dockerfile \
    --tag "$IMAGE:$VERSION" \
    --tag "$IMAGE:latest" \
    --push \
    .

echo "==> pushed $IMAGE:$VERSION and $IMAGE:latest"
docker buildx imagetools inspect "$IMAGE:$VERSION"
