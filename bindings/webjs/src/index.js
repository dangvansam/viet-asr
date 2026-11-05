import createVietasrModule from "../dist/vietasr-core.js";
import { MODEL_BASE_URL, MODEL_FILES } from "../dist/model-config.js";

const SAMPLE_RATE = 16000;
const MAX_WINDOW_SECONDS = 30;
const SPLIT_SEARCH_SECONDS = 3;
const ENERGY_HOP = 512;
const PAUSE_SECONDS = 1.2;
const PAUSE_RMS = 0.01;
const PARTIAL_SECONDS = 0.64;

let ort = null;

class OrtLoader {
    static async load(options) {
        if (!ort) {
            const mod = options.ortModuleUrl
                ? await import(options.ortModuleUrl)
                : await import("onnxruntime-web");
            ort = mod.InferenceSession ? mod : (mod.default ?? mod);
        }
        if (options.ortWasmPaths) ort.env.wasm.wasmPaths = options.ortWasmPaths;
        return ort;
    }
}

class AssetFetcher {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async bytes(name) {
        const res = await fetch(new URL(name, this.baseUrl));
        if (!res.ok) throw new Error(`fetch ${name}: ${res.status}`);
        return new Uint8Array(await res.arrayBuffer());
    }

    async text(name) {
        return new TextDecoder().decode(await this.bytes(name));
    }

    static async sha256Hex(bytes) {
        if (!globalThis.crypto?.subtle) return null;
        const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
        return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
    }

    async verified(entry) {
        const bytes = await this.bytes(entry.name);
        if (entry.size && bytes.length !== entry.size) {
            throw new Error(`${entry.name}: got ${bytes.length} bytes, expected ${entry.size}`);
        }
        const got = await AssetFetcher.sha256Hex(bytes);
        if (got && entry.sha256 && got !== entry.sha256) {
            throw new Error(`${entry.name} sha256 mismatch`);
        }
        return bytes;
    }

    async loadModel(files = MODEL_FILES) {
        const out = {};
        await Promise.all(files.map(async (entry) => {
            out[entry.role] = await this.verified(entry);
        }));
        return {
            encoder: out.encoder,
            decoder: out.decoder,
            joiner: out.joiner,
            tokens: new TextDecoder().decode(out.tokens),
        };
    }
}

class SegmentPlanner {
    static plan(pcm) {
        const maxLen = MAX_WINDOW_SECONDS * SAMPLE_RATE;
        const searchLen = SPLIT_SEARCH_SECONDS * SAMPLE_RATE;
        const windows = [];
        let start = 0;
        while (pcm.length - start > maxLen) {
            const end = start + maxLen;
            let cut = end;
            let best = Infinity;
            for (let pos = end - searchLen; pos + ENERGY_HOP <= end; pos += ENERGY_HOP) {
                let energy = 0;
                for (let i = pos; i < pos + ENERGY_HOP; ++i) energy += pcm[i] * pcm[i];
                if (energy < best) {
                    best = energy;
                    cut = pos + ENERGY_HOP / 2;
                }
            }
            windows.push([start, cut - start]);
            start = cut;
        }
        if (pcm.length > start) windows.push([start, pcm.length - start]);
        return windows;
    }
}

class StreamState {
    constructor() {
        this.committed = [];
        this.commitOffset = 0;
        this.partial = "";
        this.partialDecodedLength = 0;
    }

    text() {
        return [...this.committed, this.partial].filter((t) => t).join(" ");
    }
}

class PauseDetector {
    static trailingSilence(pcm, from) {
        const need = Math.floor(PAUSE_SECONDS * SAMPLE_RATE);
        if (pcm.length - from < need + SAMPLE_RATE / 2) return false;
        let energy = 0;
        for (let i = pcm.length - need; i < pcm.length; ++i) energy += pcm[i] * pcm[i];
        return Math.sqrt(energy / need) < PAUSE_RMS;
    }
}

class WasmHeap {
    constructor(wasm) {
        this.wasm = wasm;
    }

    floats(ptr, count) {
        return this.wasm.HEAPF32.slice(ptr >> 2, (ptr >> 2) + count);
    }

    ints(ptr, count) {
        return this.wasm.HEAP32.slice(ptr >> 2, (ptr >> 2) + count);
    }

    withFloats(data, fn) {
        const ptr = this.wasm._malloc(data.length * 4);
        try {
            this.wasm.HEAPF32.set(data, ptr >> 2);
            return fn(ptr);
        } finally {
            this.wasm._free(ptr);
        }
    }
}

export class Pipeline {
    constructor(wasm, sessions, options) {
        this.wasm = wasm;
        this.heap = new WasmHeap(wasm);
        this.encoder = sessions.encoder;
        this.decoder = sessions.decoder;
        this.joiner = sessions.joiner;
        this.options = options;
        this.decoderDim = 0;
        this.stream = null;
    }

    static async create(options = {}) {
        await OrtLoader.load(options);
        const wasm = await createVietasrModule();
        const model = options.model || await new AssetFetcher(options.modelUrl || MODEL_BASE_URL).loadModel();
        const maxActivePaths = options.maxActivePaths ?? 4;
        const status = wasm.ccall("vietasr_wasm_init", "number", ["string", "number"], [model.tokens, maxActivePaths]);
        if (status !== 0) throw new Error(`wasm init failed: ${status}`);
        const encoderProviders = options.executionProviders || ["wasm"];
        const [encoder, decoder, joiner] = await Promise.all([
            ort.InferenceSession.create(model.encoder, { executionProviders: encoderProviders, graphOptimizationLevel: "all" }),
            ort.InferenceSession.create(model.decoder, { executionProviders: ["wasm"] }),
            ort.InferenceSession.create(model.joiner, { executionProviders: ["wasm"] }),
        ]);
        return new Pipeline(wasm, { encoder, decoder, joiner }, { maxActivePaths });
    }

    startStream() {
        this.wasm.ccall("vietasr_wasm_reset", null, [], []);
        this.stream = new StreamState();
    }

    async pushStream(pcm, sampleRate = SAMPLE_RATE) {
        if (!this.stream) this.startStream();
        this.appendPcm(pcm, sampleRate);
        const all = this.currentPcm();
        const st = this.stream;
        const pending = all.length - st.commitOffset;
        if (pending > MAX_WINDOW_SECONDS * SAMPLE_RATE) {
            const [start, len] = SegmentPlanner.plan(all.subarray(st.commitOffset))[0];
            await this.commit(st.commitOffset + start, len);
        } else if (PauseDetector.trailingSilence(all, st.commitOffset)) {
            await this.commit(st.commitOffset, pending);
        } else {
            const growth = Math.max(PARTIAL_SECONDS * SAMPLE_RATE, st.partialDecodedLength / 4);
            if (pending >= st.partialDecodedLength + growth) {
                st.partial = await this.decodeWindow(st.commitOffset, pending);
                st.partialDecodedLength = pending;
            }
        }
        return st.text();
    }

    async finishStream() {
        if (!this.stream) return "";
        const st = this.stream;
        const length = this.wasm.ccall("vietasr_wasm_pcm_length", "number", [], []);
        if (length > st.commitOffset) await this.commit(st.commitOffset, length - st.commitOffset);
        const text = st.text();
        this.stream = null;
        return text;
    }

    async commit(start, len) {
        const st = this.stream;
        const text = await this.decodeWindow(start, len);
        if (text) st.committed.push(text);
        st.commitOffset = start + len;
        st.partial = "";
        st.partialDecodedLength = 0;
    }

    appendPcm(pcm, sampleRate) {
        const wasm = this.wasm;
        this.heap.withFloats(pcm, (ptr) =>
            wasm.ccall("vietasr_wasm_accept_pcm", "number", ["number", "number", "number"], [ptr, pcm.length, sampleRate]));
    }

    currentPcm() {
        const wasm = this.wasm;
        const length = wasm.ccall("vietasr_wasm_pcm_length", "number", [], []);
        return this.heap.floats(wasm.ccall("vietasr_wasm_pcm_ptr", "number", [], []), length);
    }

    async transcribe(pcm, sampleRate = SAMPLE_RATE) {
        const wasm = this.wasm;
        wasm.ccall("vietasr_wasm_reset", null, [], []);
        this.appendPcm(pcm, sampleRate);
        const resampled = this.currentPcm();
        const texts = [];
        for (const [start, len] of SegmentPlanner.plan(resampled)) {
            const text = await this.decodeWindow(start, len);
            if (text) texts.push(text);
        }
        return texts.join(" ");
    }

    async decodeWindow(start, len) {
        const wasm = this.wasm;
        const frames = wasm.ccall("vietasr_wasm_compute_features", "number", ["number", "number"], [start, len]);
        if (frames < 10) return "";
        const dim = wasm.ccall("vietasr_wasm_feature_dim", "number", [], []);
        const features = this.heap.floats(wasm.ccall("vietasr_wasm_features_ptr", "number", [], []), frames * dim);
        const encoded = await this.encoder.run({
            x: new ort.Tensor("float32", features, [1, frames, dim]),
            x_lens: new ort.Tensor("int64", BigInt64Array.from([BigInt(frames)]), [1]),
        });
        const encoderOut = encoded.encoder_out;
        const outFrames = Number(encoded.encoder_out_lens.data[0]);
        const encDim = encoderOut.dims[2];
        return this.search(encoderOut.data, outFrames, encDim);
    }

    async search(encoderData, frames, encDim) {
        const wasm = this.wasm;
        wasm.ccall("vietasr_wasm_search_begin", null, [], []);
        for (let t = 0; t < frames; ++t) {
            const pending = wasm.ccall("vietasr_wasm_search_pending", "number", [], []);
            if (pending > 0) {
                const ctx = this.heap.ints(wasm.ccall("vietasr_wasm_search_pending_ptr", "number", [], []), pending * 2);
                const out = await this.decoder.run({
                    y: new ort.Tensor("int64", BigInt64Array.from(ctx, (v) => BigInt(v)), [pending, 2]),
                });
                const decOut = out.decoder_out;
                this.decoderDim = decOut.dims[1];
                this.heap.withFloats(decOut.data, (ptr) =>
                    wasm.ccall("vietasr_wasm_search_accept_decoder", null, ["number", "number", "number"],
                        [ptr, pending, decOut.dims[1]]));
            }
            const frame = encoderData.subarray(t * encDim, (t + 1) * encDim);
            const batch = this.heap.withFloats(frame, (ptr) =>
                wasm.ccall("vietasr_wasm_search_joiner_inputs", "number", ["number", "number"], [ptr, encDim]));
            const encRows = this.heap.floats(wasm.ccall("vietasr_wasm_joiner_encoder_ptr", "number", [], []), batch * encDim);
            const decRows = this.heap.floats(wasm.ccall("vietasr_wasm_joiner_decoder_ptr", "number", [], []), batch * this.decoderDim);
            const joined = await this.joiner.run({
                encoder_out: new ort.Tensor("float32", encRows, [batch, encDim]),
                decoder_out: new ort.Tensor("float32", decRows, [batch, this.decoderDim]),
            });
            const logits = joined.logit;
            this.heap.withFloats(logits.data, (ptr) =>
                wasm.ccall("vietasr_wasm_search_accept_logits", null, ["number", "number", "number"],
                    [ptr, batch, logits.dims[1]]));
        }
        return wasm.ccall("vietasr_wasm_search_text", "string", [], []);
    }
}

export default { Pipeline };
