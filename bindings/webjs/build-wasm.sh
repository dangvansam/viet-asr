#!/usr/bin/env bash
set -euo pipefail

# Builds the vietasr DSP core to WebAssembly.
# The ONNX encoder runs separately via onnxruntime-web (see src/index.js).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE="$HERE/../../core"
OUT="$HERE/dist"
mkdir -p "$OUT"

if ! command -v emcc >/dev/null 2>&1; then
    echo "emcc not found. Activate emsdk first:"
    echo "  source /path/to/emsdk/emsdk_env.sh"
    exit 1
fi

SOURCES=(
    "$HERE/src/wasm_api.cc"
    "$CORE/src/preprocess/feature_pipeline.cc"
    "$CORE/src/preprocess/audio_resampler.cc"
    "$CORE/src/modules/vietasr/ctc_beam_search.cc"
    "$CORE/src/modules/vietasr/post_processor.cc"
    "$CORE/src/modules/vietasr/units.cc"
)

EXPORTED_FUNCS='["_vietasr_wasm_init","_vietasr_wasm_vocab_size","_vietasr_wasm_reset","_vietasr_wasm_accept_pcm","_vietasr_wasm_frames_ready","_vietasr_wasm_pop_features","_vietasr_wasm_feature_dim","_vietasr_wasm_decode_logits","_vietasr_wasm_transcript","_malloc","_free"]'

emcc "${SOURCES[@]}" \
    -I"$CORE/include" \
    -I"$CORE/src" \
    -O3 \
    -std=c++17 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createVietasrModule \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCS" \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAP32","UTF8ToString","stringToUTF8","lengthBytesUTF8"]' \
    -o "$OUT/vietasr-core.js"

echo "built: $OUT/vietasr-core.js + $OUT/vietasr-core.wasm"
ls -lh "$OUT"/vietasr-core.*

# Bundle the chunked vietasr model into the package (dist/ ships via npm).
MODEL_SRC="$HERE/../../models/vietasr"
MODEL_OUT="$OUT/model"
mkdir -p "$MODEL_OUT"
cp "$MODEL_SRC"/chunks.json "$MODEL_SRC"/vocab.txt "$MODEL_SRC"/model.onnx.part* "$MODEL_OUT/"
echo "bundled model -> $MODEL_OUT"
ls -lh "$MODEL_OUT"
