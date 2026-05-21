#!/usr/bin/env bash
# Build VietASR.xcframework for iOS — device (arm64) and simulator (arm64/x86_64).
#
# Each slice is ONE self-contained static library: libvietasr.a merged with the
# onnxruntime-c static library via libtool, so the xcframework has no external
# native dependency. The headers folder carries the C ABI plus a Clang modulemap
# so the SwiftPM `VietASR` target can `import CVietASR`.
#
# Requires: macOS with Xcode (xcodebuild, libtool) and CMake. Run by
# .github/workflows/release.yml on a macOS runner.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE="$HERE/../../core"
OUT="$HERE/VietASR.xcframework"
work="$HERE/.build-ios"

ort="$("$HERE/fetch-ort.sh")"
ort_xcfw="$ort/onnxruntime.xcframework"
# onnxruntime-c ships the C API headers once; they are slice-independent.
ort_headers="$(dirname "$(find "$ort" -name onnxruntime_c_api.h -print -quit)")"
[ -n "$ort_headers" ] || { echo "onnxruntime_c_api.h not found" >&2; exit 1; }

rm -rf "$work" "$OUT"
mkdir -p "$work"

# build_slice <name> <sysroot> <archs> <onnxruntime xcframework slice glob>
build_slice() {
    local name="$1" sysroot="$2" archs="$3" ort_glob="$4"
    local b="$work/$name"

    cmake -S "$HERE" -B "$b" -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_ARCHITECTURES="$archs" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
        -DORT_INCLUDE_DIR="$ort_headers"
    cmake --build "$b" --config Release

    local vietasr_a ort_a
    vietasr_a="$(find "$b" -name 'libvietasr.a' -print -quit)"
    # The onnxruntime-c slice ships either a bare .a or a static .framework.
    ort_a="$(find "$ort_xcfw"/$ort_glob -name '*.a' -print -quit || true)"
    [ -n "$ort_a" ] || ort_a="$(find "$ort_xcfw"/$ort_glob -type f -name onnxruntime -print -quit || true)"
    [ -n "$vietasr_a" ] && [ -n "$ort_a" ] || {
        echo "slice $name: missing libvietasr.a or onnxruntime lib" >&2
        exit 1
    }

    mkdir -p "$b/merged"
    libtool -static -o "$b/merged/libvietasr.a" "$vietasr_a" "$ort_a"
    echo "slice $name -> $b/merged/libvietasr.a"
}

build_slice device    iphoneos        "arm64"        "ios-arm64*"
build_slice simulator iphonesimulator "arm64;x86_64" "ios-*simulator*"

# Headers folder embedded in the xcframework: the C ABI + the Clang modulemap.
hdr="$work/Headers"
mkdir -p "$hdr"
cp "$CORE/include/vietasr.h" "$hdr/"
cp "$HERE/module.modulemap" "$hdr/"

xcodebuild -create-xcframework \
    -library "$work/device/merged/libvietasr.a"    -headers "$hdr" \
    -library "$work/simulator/merged/libvietasr.a" -headers "$hdr" \
    -output "$OUT"

echo "built $OUT"
