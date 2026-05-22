#!/usr/bin/env bash
# Build the Apple binding artifact: CVietASR.xcframework.
#
# CVietASR.xcframework — CVietASR.framework (dynamic) for iOS device, iOS
# simulator and macOS, carrying the C ABI header (vietasr.h) and a Clang
# module map. It is fully self-contained: the ~66 MB ASR model AND ONNX
# Runtime are baked in — the onnxruntime-c CocoaPods archive ships a *static*
# framework, so it links straight into CVietASR.framework.
#
# libvietasr ships as a *dynamic* framework — not a static archive — because
# the C++ core registers its presets/modules through static initialisers,
# which a static library would dead-strip unless force-loaded.
#
# The xcframework lands in bindings/apple/artifacts/, which Package.swift
# references as a local binary target. Re-run after any change to the C++ core.
#
# Usage:  bash bindings/apple/build-xcframework.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
# onnxruntime ships an iOS xcframework only via the CocoaPods archive. 1.20.0
# is the newest version published there (1.20.1 is desktop-only) and matches
# the version the Android binding already uses.
ORT_VERSION="1.20.0"
ORT_URL="https://download.onnxruntime.ai/pod-archive-onnxruntime-c-${ORT_VERSION}.zip"

# Deployment targets = the *minimum* OS supported. One build runs on that
# version and every newer one (iOS 13 … latest, macOS 11 … latest). The floors
# below are hard limits — the onnxruntime 1.20 slices are built for iOS 13.0 /
# macOS 11.0, and the C++ core needs std::filesystem (iOS 13 / macOS 10.15).
# Override to raise them, e.g. IOS_DEPLOYMENT_TARGET=15.0 bash build-xcframework.sh
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-13.0}"
MACOS_DEPLOYMENT_TARGET="${MACOS_DEPLOYMENT_TARGET:-11.0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
BUILD_DIR="$HERE/build"
ARTIFACTS="$HERE/artifacts"
ORT_XCFW="$BUILD_DIR/onnxruntime/onnxruntime.xcframework"

echo "==> Apple binding build"
echo "    repo root  : $REPO_ROOT"
echo "    onnxruntime: $ORT_VERSION"

mkdir -p "$BUILD_DIR" "$ARTIFACTS"

# ── 1. ONNX Runtime xcframework (download once, cache the zip) ─────────────
ORT_ZIP="$BUILD_DIR/onnxruntime-c-${ORT_VERSION}.zip"
if [[ ! -d "$ORT_XCFW" ]]; then
    if [[ ! -f "$ORT_ZIP" ]]; then
        echo "==> downloading onnxruntime iOS pod archive"
        curl -fSL --retry 3 -o "$ORT_ZIP" "$ORT_URL"
    fi
    echo "==> extracting onnxruntime.xcframework"
    rm -rf "$BUILD_DIR/onnxruntime"
    mkdir -p "$BUILD_DIR/onnxruntime"
    unzip -q "$ORT_ZIP" -d "$BUILD_DIR/onnxruntime"
fi
[[ -d "$ORT_XCFW" ]] || { echo "ERROR: onnxruntime.xcframework not found after extract"; exit 1; }

# ── Stage the C ABI header + Clang module map into a built framework ───────
# iOS frameworks are flat; macOS frameworks are versioned (Versions/A). CMake
# does not populate a framework's Headers/ from a $<TARGET_OBJECTS> target, so
# place vietasr.h and module.modulemap explicitly.
stage_framework_api() {
    local fw="$1"
    local hdr="$REPO_ROOT/core/include/vietasr.h"
    local mod="$HERE/Headers/module.modulemap"
    if [[ -d "$fw/Versions/A" ]]; then
        mkdir -p "$fw/Versions/A/Headers" "$fw/Versions/A/Modules"
        cp "$hdr" "$fw/Versions/A/Headers/"
        cp "$mod" "$fw/Versions/A/Modules/"
        ln -sfn Versions/Current/Headers "$fw/Headers"
        ln -sfn Versions/Current/Modules "$fw/Modules"
    else
        mkdir -p "$fw/Headers" "$fw/Modules"
        cp "$hdr" "$fw/Headers/"
        cp "$mod" "$fw/Modules/"
    fi
}

# ── 2. Build CVietASR.framework per Apple slice ────────────────────────────
# $1 tag  $2 onnxruntime.framework dir  $3.. extra cmake args
build_slice() {
    local tag="$1"; local ort_fw="$2"; shift 2
    local out="$BUILD_DIR/$tag"
    [[ -d "$ort_fw" ]] || { echo "ERROR: missing onnxruntime slice $ort_fw"; exit 1; }
    echo "==> building CVietASR.framework [$tag]"
    rm -rf "$out"
    cmake -S "$REPO_ROOT/core" -B "$out" \
        -DCMAKE_BUILD_TYPE=Release \
        -DVIETASR_BUILD_SHARED=ON \
        -DVIETASR_APPLE_FRAMEWORK=ON \
        -DVIETASR_BUILD_TESTS=OFF \
        -DVIETASR_BUILD_EXAMPLES=OFF \
        -DVIETASR_INSTALL=OFF \
        -DVIETASR_EMBED_MODEL=ON \
        -DVIETASR_WITH_CURL=OFF \
        -DONNXRUNTIME_FRAMEWORK_DIR="$ort_fw" \
        "$@" >/dev/null
    cmake --build "$out" --config Release -j >/dev/null
    local fw="$out/CVietASR.framework"
    [[ -d "$fw" ]] || { echo "ERROR: $fw missing"; exit 1; }
    stage_framework_api "$fw"
    echo "    -> $fw ($(du -sh "$fw" | cut -f1))"
}

build_slice ios-device "$ORT_XCFW/ios-arm64/onnxruntime.framework" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"

build_slice ios-sim "$ORT_XCFW/ios-arm64_x86_64-simulator/onnxruntime.framework" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"

build_slice macos "$ORT_XCFW/macos-arm64_x86_64/onnxruntime.framework" \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"

# ── 3. Assemble CVietASR.xcframework ──────────────────────────────────────
echo "==> creating CVietASR.xcframework"
rm -rf "$ARTIFACTS/CVietASR.xcframework"
xcodebuild -create-xcframework \
    -framework "$BUILD_DIR/ios-device/CVietASR.framework" \
    -framework "$BUILD_DIR/ios-sim/CVietASR.framework" \
    -framework "$BUILD_DIR/macos/CVietASR.framework" \
    -output "$ARTIFACTS/CVietASR.xcframework" >/dev/null

echo
echo "==> done. artifact:"
du -sh "$ARTIFACTS/CVietASR.xcframework"
echo
echo "Package.swift consumes this via a local binary target."
echo "For a release, zip the xcframework and run 'swift package compute-checksum'."
