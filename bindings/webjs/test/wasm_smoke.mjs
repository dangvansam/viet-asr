import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

import createVietasrModule from "../dist/vietasr-core.js";

const here = dirname(fileURLToPath(import.meta.url));
const unitsPath = process.argv[2]
    || join(here, "../dist/model/vocab.txt");

async function main() {
    const wasm = await createVietasrModule();
    console.log("WASM module loaded");

    const unitsText = readFileSync(unitsPath, "utf-8");
    const initStatus = wasm.ccall("vietasr_wasm_init", "number", ["string"], [unitsText]);
    if (initStatus !== 0) {
        console.error(`FAIL: wasm init returned ${initStatus}`);
        process.exit(1);
    }
    const vocab = wasm.ccall("vietasr_wasm_vocab_size", "number", [], []);
    console.log(`vocab size: ${vocab}`);
    if (vocab !== 4972) {
        console.error(`FAIL: expected 4972 tokens, got ${vocab}`);
        process.exit(1);
    }

    wasm.ccall("vietasr_wasm_reset", null, [], []);

    // 1 second of 16 kHz silence-ish ramp — exercises the fbank front-end
    const sampleRate = 16000;
    const pcm = new Float32Array(sampleRate);
    for (let i = 0; i < pcm.length; ++i) {
        pcm[i] = 0.05 * Math.sin((2 * Math.PI * 220 * i) / sampleRate);
    }

    const ptr = wasm._malloc(pcm.length * 4);
    wasm.HEAPF32.set(pcm, ptr >> 2);
    wasm.ccall("vietasr_wasm_accept_pcm", "number",
        ["number", "number", "number"], [ptr, pcm.length, sampleRate]);
    wasm._free(ptr);

    const framesReady = wasm.ccall("vietasr_wasm_frames_ready", "number", [], []);
    console.log(`feature frames ready: ${framesReady}`);
    if (framesReady < 90) {
        console.error(`FAIL: expected ~98 frames for 1s audio, got ${framesReady}`);
        process.exit(1);
    }

    const dim = wasm.ccall("vietasr_wasm_feature_dim", "number", [], []);
    console.log(`feature dim: ${dim}`);
    if (dim !== 80) {
        console.error(`FAIL: expected 80-bin features, got ${dim}`);
        process.exit(1);
    }

    const featPtr = wasm.ccall("vietasr_wasm_pop_features", "number", ["number"], [43]);
    const feats = wasm.HEAPF32.subarray(featPtr >> 2, (featPtr >> 2) + 43 * 80);
    const finite = feats.every((v) => Number.isFinite(v));
    console.log(`popped 43x80 features, all finite: ${finite}`);
    if (!finite) {
        console.error("FAIL: features contain non-finite values");
        process.exit(1);
    }

    console.log("\nWASM DSP core smoke test passed");
}

main().catch((err) => {
    console.error("ERROR:", err);
    process.exit(1);
});
