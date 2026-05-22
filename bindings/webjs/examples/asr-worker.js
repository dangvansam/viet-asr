// Web Worker hosting the VietASR Pipeline. Running the ONNX encoder here keeps
// the UI thread free, so live transcription never janks the page.
//
// Messages in : {type:"init", options?} {type:"start"} {type:"pcm", pcm, sampleRate}
//               {type:"stop"} {type:"transcribe", pcm, sampleRate}
// Messages out: {type:"ready"} {type:"started"} {type:"partial", text}
//               {type:"final", text} {type:"error", message}

import { Pipeline } from "../src/index.js";

// onnxruntime-web is a bare specifier a Worker cannot resolve (no import map).
// Point Pipeline.create() at the concrete ESM file shipped in node_modules.
const ortDir = new URL(
    "../node_modules/onnxruntime-web/dist/", import.meta.url).href;
const ortModuleUrl = ortDir + "ort.wasm.min.mjs";

let pipeline = null;

async function handle(msg) {
    try {
        switch (msg.type) {
            case "init":
                pipeline = await Pipeline.create({
                    ...(msg.options || {}),
                    ortModuleUrl,
                    ortWasmPaths: ortDir,
                });
                self.postMessage({ type: "ready" });
                break;

            case "start":
                pipeline.startStream();
                self.postMessage({ type: "started" });
                break;

            case "pcm": {
                const text = await pipeline.pushStream(
                    new Float32Array(msg.pcm), msg.sampleRate);
                self.postMessage({ type: "partial", text });
                break;
            }

            case "stop": {
                const text = await pipeline.finishStream();
                self.postMessage({ type: "final", text });
                break;
            }

            case "transcribe": {
                const text = await pipeline.transcribe(
                    new Float32Array(msg.pcm), msg.sampleRate);
                self.postMessage({ type: "final", text });
                break;
            }

            default:
                break;
        }
    } catch (err) {
        self.postMessage({
            type: "error",
            message: String((err && err.message) || err),
        });
    }
}

// Serialize message handling: pushStream/finishStream touch shared WASM state,
// so they must never run concurrently even though each handler is async.
let chain = Promise.resolve();
self.onmessage = (e) => {
    chain = chain.then(() => handle(e.data || {}));
};
