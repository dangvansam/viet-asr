#!/usr/bin/env bash
# Package a freshly built libvietasr + onnxruntime into the release tarball that
# the binding native-resolvers download at install time.
#
#   scripts/package-natives.sh <platform> <build-dir> <out-dir>
#
# platform (CPU):  linux-x64 | linux-arm64 | darwin-universal2 | win-x64
# platform (GPU):  linux-x64-gpu | win-x64-gpu | win-x64-directml
# Run by .github/workflows/release.yml on each native runner.
set -euo pipefail

platform="${1:?usage: package-natives.sh <platform> <build-dir> <out-dir>}"
build_dir="${2:?build dir}"
out_dir="${3:?out dir}"

stage="$(mktemp -d)"
mkdir -p "$out_dir"

# Resolve the ONNX Runtime lib directory from the CMake cache so this works for
# every variant: the CPU/CUDA packages extract under _deps/onnxruntime-src/lib,
# the DirectML NuGet package lays out under runtimes/win-x64/native.
ort_lib="$(sed -n 's/^ONNXRUNTIME_LIB_DIR:[^=]*=//p' "$build_dir/CMakeCache.txt" 2>/dev/null || true)"
if [ -z "$ort_lib" ] || [ ! -d "$ort_lib" ]; then
  ort_lib="$build_dir/_deps/onnxruntime-src/lib"
fi

case "$platform" in
  win-*)
    cp "$(find "$build_dir" -name 'vietasr.dll' -print -quit)" "$stage/"
    # Import library — Rust links against libvietasr at build time on MSVC.
    lib="$(find "$build_dir" -name 'vietasr.lib' -print -quit || true)"
    [ -n "$lib" ] && cp "$lib" "$stage/"
    find "$ort_lib" -name '*.dll' -exec cp {} "$stage/" \;
    ;;
  darwin-*)
    cp "$(find "$build_dir" -name 'libvietasr.dylib' -print -quit)" "$stage/"
    find "$ort_lib" -name 'libonnxruntime*.dylib' -exec cp -a {} "$stage/" \;
    install_name_tool -add_rpath '@loader_path' "$stage/libvietasr.dylib" || true
    ;;
  linux-*)
    cp "$(find "$build_dir" -name 'libvietasr.so*' -print -quit)" "$stage/libvietasr.so"
    find "$ort_lib" -name 'libonnxruntime.so*' -exec cp -a {} "$stage/" \;
    # GPU variants ship the EP libraries as separate sidecar .so files
    # (libonnxruntime_providers_cuda/_tensorrt/_shared) loaded at runtime.
    find "$ort_lib" -name 'libonnxruntime_providers_*.so*' -exec cp -a {} "$stage/" \;
    if command -v patchelf >/dev/null 2>&1; then
      patchelf --set-rpath '$ORIGIN' "$stage/libvietasr.so"
    fi
    ;;
  *)
    echo "package-natives: unknown platform '$platform'" >&2
    exit 1
    ;;
esac

tar -czf "$out_dir/viet-asr-native-$platform.tar.gz" -C "$stage" .
echo "packaged -> $out_dir/viet-asr-native-$platform.tar.gz"
ls -la "$stage"
