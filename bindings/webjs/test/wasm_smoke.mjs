import { readFileSync } from "fs";

import createVietasrModule from "../dist/vietasr-core.js";

const tokensPath = process.argv[2];
if (!tokensPath) {
    console.error("usage: node test/wasm_smoke.mjs <tokens.txt>  (from the Hugging Face model, e.g. ~/.cache/vietasr/vietasr/<rev>/tokens.txt)");
    process.exit(2);
}

class SmokeTest {
    constructor(wasm) {
        this.wasm = wasm;
    }

    fail(message) {
        console.error(`FAIL: ${message}`);
        process.exit(1);
    }

    run() {
        const wasm = this.wasm;
        const status = wasm.ccall("vietasr_wasm_init", "number", ["string", "number"], [readFileSync(tokensPath, "utf-8"), 4]);
        if (status !== 0) this.fail(`init returned ${status}`);
        const vocab = wasm.ccall("vietasr_wasm_vocab_size", "number", [], []);
        if (vocab !== 8192) this.fail(`expected 8192 tokens, got ${vocab}`);

        wasm.ccall("vietasr_wasm_reset", null, [], []);
        const pcm = new Float32Array(16000);
        for (let i = 0; i < pcm.length; ++i) pcm[i] = 0.05 * Math.sin((2 * Math.PI * 220 * i) / 16000);
        const ptr = wasm._malloc(pcm.length * 4);
        wasm.HEAPF32.set(pcm, ptr >> 2);
        wasm.ccall("vietasr_wasm_accept_pcm", "number", ["number", "number", "number"], [ptr, pcm.length, 16000]);
        wasm._free(ptr);

        const frames = wasm.ccall("vietasr_wasm_compute_features", "number", ["number", "number"], [0, -1]);
        if (frames !== 100) this.fail(`expected 100 frames for 1 s, got ${frames}`);
        const fptr = wasm.ccall("vietasr_wasm_features_ptr", "number", [], []);
        const feats = wasm.HEAPF32.subarray(fptr >> 2, (fptr >> 2) + frames * 80);
        if (!feats.every((v) => Number.isFinite(v))) this.fail("non-finite features");

        wasm.ccall("vietasr_wasm_search_begin", null, [], []);
        const pending = wasm.ccall("vietasr_wasm_search_pending", "number", [], []);
        if (pending !== 1) this.fail(`expected 1 pending decoder context, got ${pending}`);
        console.log(`vocab=${vocab} frames=${frames} pending=${pending}\nWASM core smoke test passed`);
    }
}

createVietasrModule().then((wasm) => new SmokeTest(wasm).run()).catch((err) => {
    console.error("ERROR:", err);
    process.exit(1);
});
