#!/usr/bin/env bash
# Cross-compile libvietasr.so for every Android ABI and stage it — together with
# the matching libonnxruntime.so — into lib/src/main/jniLibs/, which the Gradle
# AAR build packages. Run by .github/workflows/release.yml before :lib:publish.
#
# Requires: Android NDK (ANDROID_NDK_HOME), cmake, curl, unzip.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDK="${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the Android NDK path}"
ORT_VERSION="1.20.1"
ABIS=(arm64-v8a armeabi-v7a x86_64)

# onnxruntime-android AAR ships per-ABI libonnxruntime.so + the C API headers.
ort="$HERE/.ort-android"
if [ ! -f "$ort/headers/onnxruntime_c_api.h" ]; then
    mkdir -p "$ort"
    curl -fSL --retry 3 -o "$ort/ort.aar" \
        "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"
    ( cd "$ort" && unzip -oq ort.aar && rm ort.aar )
fi

for abi in "${ABIS[@]}"; do
    out="$HERE/lib/src/main/jniLibs/$abi"
    build="$HERE/.build-$abi"
    mkdir -p "$out"
    cmake -S "$HERE/lib/src/main/cpp" -B "$build" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT="$ort"
    cmake --build "$build" -j
    cp "$build/libvietasr.so" "$out/"
    cp "$ort/jni/$abi/libonnxruntime.so" "$out/"
    echo "staged jniLibs/$abi"
done
