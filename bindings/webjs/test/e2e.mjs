import { readFileSync } from "fs";

import { Pipeline } from "../src/index.js";

const wavPath = process.argv[2];

function readWavMono(path) {
    const buf = readFileSync(path);
    if (buf.slice(0, 4).toString() !== "RIFF" || buf.slice(8, 12).toString() !== "WAVE") {
        throw new Error("not a RIFF/WAVE file");
    }
    let offset = 12;
    let sampleRate = 16000;
    let channels = 1;
    let bits = 16;
    while (offset + 8 <= buf.length) {
        const id = buf.slice(offset, offset + 4).toString();
        const size = buf.readUInt32LE(offset + 4);
        const body = buf.slice(offset + 8, offset + 8 + size);
        if (id === "fmt ") {
            channels = body.readUInt16LE(2);
            sampleRate = body.readUInt32LE(4);
            bits = body.readUInt16LE(14);
        }
        if (id === "data") {
            if (bits !== 16) throw new Error("only 16-bit PCM supported");
            const view = new Int16Array(body.buffer, body.byteOffset, body.byteLength / 2);
            const frames = view.length / channels;
            const pcm = new Float32Array(frames);
            for (let i = 0; i < frames; ++i) {
                let mixed = 0;
                for (let c = 0; c < channels; ++c) {
                    mixed += view[i * channels + c];
                }
                pcm[i] = mixed / channels / 32768;
            }
            return { pcm, sampleRate };
        }
        offset += 8 + size + (size & 1);
    }
    throw new Error("no data chunk");
}

async function main() {
    if (!wavPath) {
        console.error("usage: node test/e2e.mjs <wav-file>");
        process.exit(1);
    }

    const pipe = await Pipeline.create();
    console.log("pipeline ready (WASM DSP + onnxruntime-web encoder, bundled model)");

    const { pcm, sampleRate } = readWavMono(wavPath);
    console.log(`audio: ${(pcm.length / sampleRate).toFixed(2)}s @ ${sampleRate} Hz`);

    const started = Date.now();
    const text = await pipe.transcribe(pcm, sampleRate);
    const elapsed = (Date.now() - started) / 1000;

    console.log();
    console.log(`transcript (${elapsed.toFixed(2)}s):`);
    console.log(text);

    const expectedPrefix = "sao lại không liên quan các anh";
    if (text.startsWith(expectedPrefix)) {
        console.log("\nE2E test passed — transcript matches expected prefix");
    } else {
        console.error(`\nFAIL: expected prefix "${expectedPrefix}"`);
        process.exit(1);
    }
}

main().catch((err) => {
    console.error("ERROR:", err);
    process.exit(1);
});
