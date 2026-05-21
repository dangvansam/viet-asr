#!/usr/bin/env bash
# Download + extract the iOS ONNX Runtime C/C++ package — an onnxruntime.xcframework
# with device + simulator slices, CoreML execution provider included. Mirrors the
# Android build's .ort-android step. Prints the extraction directory on stdout.
#
# Requires: curl, unzip.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# onnxruntime-c iOS pods are published as pod-archive zips on download.onnxruntime.ai.
ORT_VERSION="${ORT_VERSION:-1.20.1}"
ort="$HERE/.ort-ios"

if [ ! -d "$ort/onnxruntime.xcframework" ]; then
    mkdir -p "$ort"
    url="https://download.onnxruntime.ai/pod-archive-onnxruntime-c-${ORT_VERSION}.zip"
    echo "fetching $url" >&2
    curl -fSL --retry 3 -o "$ort/ort.zip" "$url"
    ( cd "$ort" && unzip -oq ort.zip && rm ort.zip )
fi

[ -d "$ort/onnxruntime.xcframework" ] || {
    echo "fetch-ort: onnxruntime.xcframework missing after extract" >&2
    exit 1
}
echo "$ort"
