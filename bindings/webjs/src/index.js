import * as ort from "onnxruntime-web";
import createVietasrModule from "../dist/vietasr-core.js";

const INPUT_FRAMES = 43;
const FEATURE_DIM = 80;
const OUTPUT_FRAMES = 10;
const NUM_LAYERS = 12;
const ATT_HEADS = 4;
const ATT_CACHE_LEN = 40;
const ATT_CACHE_DIM = 224;
const CNN_CACHE_W = 448;
const CNN_CACHE_H = 30;

// The vietasr model ships inside this package as <50 MB chunks under
// dist/model/, reassembled at load time. No network fetch.
const MODEL_DIR_URL = new URL("../dist/model/", import.meta.url);
const IS_NODE = typeof process !== "undefined" && !!process?.versions?.node;

async function loadAsset(name) {
    const url = new URL(name, MODEL_DIR_URL);
    if (IS_NODE) {
        const { readFileSync } = await import("fs");
        const { fileURLToPath } = await import("url");
        return new Uint8Array(readFileSync(fileURLToPath(url)));
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
}

async function sha256Hex(bytes) {
    if (!globalThis.crypto?.subtle) return null;
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(digest)]
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
}

/** Reassemble the chunked model + vocab bundled with this package. */
async function loadBundledModel() {
    const manifest = JSON.parse(
        new TextDecoder().decode(await loadAsset("chunks.json")));

    const model = new Uint8Array(manifest.total_size);
    let offset = 0;
    for (const chunk of manifest.chunks) {
        const part = await loadAsset(chunk.name);
        if (part.length !== chunk.size) {
            throw new Error(
                `chunk ${chunk.name}: got ${part.length}, expected ${chunk.size}`);
        }
        model.set(part, offset);
        offset += part.length;
    }
    if (offset !== manifest.total_size) {
        throw new Error(
            `reassembled model size ${offset}, expected ${manifest.total_size}`);
    }
    const got = await sha256Hex(model);
    if (got && manifest.sha256 && got !== manifest.sha256) {
        throw new Error(`model sha256 mismatch: ${got} != ${manifest.sha256}`);
    }

    const unitsText = new TextDecoder().decode(await loadAsset("vocab.txt"));
    return { encoderModel: model, unitsText };
}

/**
 * Browser Vietnamese ASR pipeline.
 *
 * Architecture: the DSP front-end (fbank, CTC prefix beam search,
 * BPE detokenize) runs in WebAssembly compiled from the shared C++ core.
 * The streaming Conformer encoder runs via onnxruntime-web. JS orchestrates
 * the two and carries the att/cnn caches between chunks.
 */
export class Pipeline {
    constructor(wasm, session, unitsText) {
        this.wasm = wasm;
        this.session = session;
        this.unitsText = unitsText;
        this.encoderOffset = 0;
        this.attCache = new Float32Array(
            NUM_LAYERS * ATT_HEADS * ATT_CACHE_LEN * ATT_CACHE_DIM);
        this.cnnCache = new Float32Array(NUM_LAYERS * CNN_CACHE_W * CNN_CACHE_H);
    }

    /**
     * @param {object} [options]
     * @param {string} [options.unitsText] pre-loaded vocab (skips bundled load)
     * @param {ArrayBuffer|Uint8Array} [options.encoderModel] pre-loaded ONNX (skips bundled load)
     */
    static async create(options = {}) {
        const wasm = await createVietasrModule();

        let unitsText = options.unitsText;
        let encoderBuffer = options.encoderModel;
        if (!unitsText || !encoderBuffer) {
            const bundled = await loadBundledModel();
            unitsText = unitsText || bundled.unitsText;
            encoderBuffer = encoderBuffer || bundled.encoderModel;
        }

        const initStatus = wasm.ccall(
            "vietasr_wasm_init", "number", ["string"], [unitsText]);
        if (initStatus !== 0) {
            throw new Error(`wasm init failed: ${initStatus}`);
        }

        const modelBytes = encoderBuffer instanceof Uint8Array
            ? encoderBuffer : new Uint8Array(encoderBuffer);
        const session = await ort.InferenceSession.create(
            modelBytes,
            { executionProviders: ["wasm"] });

        return new Pipeline(wasm, session, unitsText);
    }

    reset() {
        this.wasm.ccall("vietasr_wasm_reset", null, [], []);
        this.encoderOffset = 0;
        this.attCache.fill(0);
        this.cnnCache.fill(0);
    }

    /**
     * Transcribe a complete audio clip.
     * @param {Float32Array} pcm  audio samples in [-1, 1]
     * @param {number} sampleRate
     * @returns {Promise<string>}
     */
    async transcribe(pcm, sampleRate = 16000) {
        this.reset();
        await this._feed(pcm, sampleRate);
        await this._drain();
        return this._transcript();
    }

    async _feed(pcm, sampleRate) {
        const wasm = this.wasm;
        const bytes = pcm.length * 4;
        const ptr = wasm._malloc(bytes);
        wasm.HEAPF32.set(pcm, ptr >> 2);
        wasm.ccall("vietasr_wasm_accept_pcm", "number",
            ["number", "number", "number"], [ptr, pcm.length, sampleRate]);
        wasm._free(ptr);
    }

    async _drain() {
        const wasm = this.wasm;
        while (true) {
            const ready = wasm.ccall("vietasr_wasm_frames_ready", "number", [], []);
            if (ready < INPUT_FRAMES) break;
            await this._runChunk(INPUT_FRAMES);
        }
        const tail = wasm.ccall("vietasr_wasm_frames_ready", "number", [], []);
        if (tail > 0) {
            await this._runChunk(tail);
        }
    }

    async _runChunk(nFrames) {
        const wasm = this.wasm;
        const featPtr = wasm.ccall("vietasr_wasm_pop_features", "number",
            ["number"], [nFrames]);
        const features = new Float32Array(INPUT_FRAMES * FEATURE_DIM);
        const popped = wasm.HEAPF32.subarray(
            featPtr >> 2, (featPtr >> 2) + nFrames * FEATURE_DIM);
        features.set(popped);

        const feeds = {
            chunk: new ort.Tensor("float32", features, [1, INPUT_FRAMES, FEATURE_DIM]),
            offset: new ort.Tensor("int64",
                BigInt64Array.from([BigInt(this.encoderOffset)]), [1, 1]),
            att_cache: new ort.Tensor("float32", this.attCache,
                [1, NUM_LAYERS, ATT_HEADS, ATT_CACHE_LEN, ATT_CACHE_DIM]),
            cnn_cache: new ort.Tensor("float32", this.cnnCache,
                [1, NUM_LAYERS, CNN_CACHE_W, CNN_CACHE_H]),
        };

        const out = await this.session.run(feeds);
        const logits = out.output;
        const vocab = logits.dims[logits.dims.length - 1];
        const outFrames = logits.dims[logits.dims.length - 2];

        const logitsData = logits.data;
        const lbytes = logitsData.length * 4;
        const lptr = wasm._malloc(lbytes);
        wasm.HEAPF32.set(logitsData, lptr >> 2);
        wasm.ccall("vietasr_wasm_decode_logits", null,
            ["number", "number", "number"], [lptr, outFrames, vocab]);
        wasm._free(lptr);

        this.attCache = out.r_att_cache.data;
        this.cnnCache = out.r_cnn_cache.data;
        this.encoderOffset += OUTPUT_FRAMES;
    }

    _transcript() {
        return this.wasm.ccall("vietasr_wasm_transcript", "string", [], []);
    }
}

export default { Pipeline };
