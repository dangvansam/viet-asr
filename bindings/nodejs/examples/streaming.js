const fs = require("fs");
const path = require("path");

const vietasr = require("..");

function readWavInt16Mono(wavPath) {
    const buf = fs.readFileSync(wavPath);
    if (buf.slice(0, 4).toString() !== "RIFF" || buf.slice(8, 12).toString() !== "WAVE") {
        throw new Error(`not a RIFF/WAVE file: ${wavPath}`);
    }
    let offset = 12;
    let sampleRate = 16000;
    let channels = 1;
    let bitsPerSample = 16;
    while (offset + 8 <= buf.length) {
        const chunkId = buf.slice(offset, offset + 4).toString();
        const chunkSize = buf.readUInt32LE(offset + 4);
        const body = buf.slice(offset + 8, offset + 8 + chunkSize);
        if (chunkId === "fmt ") {
            channels = body.readUInt16LE(2);
            sampleRate = body.readUInt32LE(4);
            bitsPerSample = body.readUInt16LE(14);
        }
        if (chunkId === "data") {
            if (bitsPerSample !== 16) throw new Error("only 16-bit PCM supported");
            const view = new Int16Array(
                body.buffer, body.byteOffset, body.byteLength / 2);
            if (channels === 1) {
                return { pcm: view, sampleRate };
            }
            const mono = new Int16Array(view.length / channels);
            for (let i = 0, j = 0; j < mono.length; i += channels, ++j) {
                let mixed = 0;
                for (let c = 0; c < channels; ++c) mixed += view[i + c];
                mono[j] = mixed / channels;
            }
            return { pcm: mono, sampleRate };
        }
        offset += 8 + chunkSize + (chunkSize & 1);
    }
    throw new Error("no data chunk");
}

function main() {
    const wavPath = process.argv[2];
    if (!wavPath) {
        console.error(`usage: node ${path.basename(__filename)} <wav-file>`);
        process.exit(1);
    }

    const { pcm, sampleRate } = readWavInt16Mono(wavPath);
    const durationS = pcm.length / sampleRate;
    console.log(`audio: ${durationS.toFixed(2)}s @ ${sampleRate} Hz`);

    const pipe = vietasr.Pipeline.preset("transcribe");
    const chunkSize = Math.floor(sampleRate * 0.32);

    const startedAt = Date.now();
    const session = pipe.stream(sampleRate);
    let lastPartial = "";
    try {
        for (let offset = 0; offset < pcm.length; offset += chunkSize) {
            const chunk = pcm.subarray(offset, offset + chunkSize);
            session.accept(chunk);
            const partial = session.partial().text;
            if (partial && partial !== lastPartial) {
                const t = (offset + chunk.length) / sampleRate;
                console.log(`  [${t.toFixed(2)}s] ${partial}`);
                lastPartial = partial;
            }
        }
        const final = session.final().text;
        const elapsedS = (Date.now() - startedAt) / 1000;
        console.log();
        console.log(`FINAL (${elapsedS.toFixed(2)}s wall, RTF ${(elapsedS / durationS).toFixed(2)}):`);
        console.log(final);
    } finally {
        session.close();
        pipe.close();
    }
}

main();
