"""Self-contained browser test console, served at GET /."""

INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VietASR — Test Console</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 720px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0; font-size: 22px; }
  .sub { color: #888; margin: 2px 0 20px; }
  section { border: 1px solid #8884; border-radius: 10px; padding: 16px; margin: 14px 0; }
  h2 { margin: 0 0 4px; font-size: 16px; }
  code { background: #8882; padding: 1px 6px; border-radius: 4px; font-size: 13px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0; }
  button { font: inherit; padding: 7px 14px; border-radius: 7px; border: 1px solid #8886;
           background: #7773; cursor: pointer; }
  button:hover:not(:disabled) { background: #7775; }
  button:disabled { opacity: .5; cursor: default; }
  button.rec { background: #d33; color: #fff; border-color: #d33; }
  input[type=file] { font: inherit; max-width: 100%; }
  .label { color: #888; font-size: 12px; text-transform: uppercase;
           letter-spacing: .5px; margin: 10px 0 3px; }
  .out { background: #8881; border-radius: 7px; padding: 10px 12px; min-height: 42px;
         white-space: pre-wrap; word-break: break-word; font-size: 15px; }
  .out.partial { color: #888; }
  .status { color: #888; font-size: 13px; margin-top: 16px; }
</style>
</head>
<body>
<h1>VietASR</h1>
<div class="sub">Offline Vietnamese speech-to-text — test console</div>

<section>
  <h2>Batch</h2>
  <code>POST /v1/audio/transcriptions</code>
  <div class="row">
    <input type="file" id="batchFile" accept="audio/*">
    <button id="batchBtn">Transcribe</button>
  </div>
  <div class="out" id="batchOut">—</div>
</section>

<section>
  <h2>Streaming</h2>
  <code>WebSocket /v1/stream</code>
  <div class="row">
    <button id="micBtn">● Record mic</button>
    <span style="color:#888">or</span>
    <input type="file" id="streamFile" accept="audio/*">
    <button id="streamBtn">Stream file</button>
  </div>
  <div class="label">partial</div>
  <div class="out partial" id="partialOut">—</div>
  <div class="label">final</div>
  <div class="out" id="finalOut">—</div>
</section>

<div class="status" id="statusBar">ready</div>

<script>
const $ = (id) => document.getElementById(id);
const statusBar = $("statusBar");
const setStatus = (t) => { statusBar.textContent = t; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function wsUrl(rate, encoding) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/v1/stream?sample_rate=${rate}&encoding=${encoding}`;
}

// ---- Batch (REST) --------------------------------------------------------
$("batchBtn").onclick = async () => {
  const file = $("batchFile").files[0];
  if (!file) { setStatus("pick an audio file first"); return; }
  $("batchOut").textContent = "…";
  setStatus("uploading " + file.name + " …");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/v1/audio/transcriptions", { method: "POST", body: fd });
    const data = await res.json();
    if (res.ok) {
      $("batchOut").textContent = data.text || "(empty)";
      setStatus("done");
    } else {
      $("batchOut").textContent = "error: " + (data.detail || res.status);
      setStatus("failed");
    }
  } catch (err) {
    $("batchOut").textContent = "error: " + err;
    setStatus("failed");
  }
};

// ---- Streaming (WebSocket) ----------------------------------------------
function openStream(rate) {
  const ws = new WebSocket(wsUrl(rate, "pcm_f32le"));
  ws.binaryType = "arraybuffer";
  $("partialOut").textContent = "—";
  $("finalOut").textContent = "—";
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "partial") $("partialOut").textContent = msg.text || "—";
    else if (msg.type === "final") { $("finalOut").textContent = msg.text || "—";
                                     setStatus("final received"); }
    else if (msg.type === "error") setStatus("server error: " + msg.message);
  };
  return ws;
}

let mic = null;

async function startMic() {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    await ctx.close();
    setStatus("microphone denied: " + err);
    return;
  }
  const ws = openStream(ctx.sampleRate);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = (e) => {
    if (ws.readyState === 1)
      ws.send(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(node);
  node.connect(ctx.destination);  // outputs silence; keeps the node alive

  mic = { ctx, stream, src, node, ws };
  $("micBtn").textContent = "■ Stop";
  $("micBtn").classList.add("rec");
  setStatus("recording @ " + Math.round(ctx.sampleRate) + " Hz — speak…");
}

async function stopMic() {
  if (!mic) return;
  const { ctx, stream, src, node, ws } = mic;
  mic = null;
  $("micBtn").textContent = "● Record mic";
  $("micBtn").classList.remove("rec");
  node.disconnect();
  src.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  await ctx.close();
  if (ws.readyState === 1) ws.send(JSON.stringify({ type: "eof" }));
  setStatus("finalizing…");
}

$("micBtn").onclick = () => (mic ? stopMic() : startMic());

$("streamBtn").onclick = async () => {
  const file = $("streamFile").files[0];
  if (!file) { setStatus("pick an audio file to stream"); return; }
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let pcm, rate;
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    pcm = buf.getChannelData(0);
    rate = buf.sampleRate;
  } catch (err) {
    await ctx.close();
    setStatus("cannot decode audio: " + err);
    return;
  }
  const ws = openStream(rate);
  try {
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  } catch {
    setStatus("websocket failed"); await ctx.close(); return;
  }
  $("streamBtn").disabled = true;
  setStatus("streaming " + file.name + " …");
  const frame = Math.floor(rate / 5);  // ~200 ms per message
  for (let i = 0; i < pcm.length; i += frame) {
    if (ws.readyState !== 1) break;
    ws.send(new Float32Array(pcm.subarray(i, i + frame)));
    await sleep(40);
  }
  if (ws.readyState === 1) ws.send(JSON.stringify({ type: "eof" }));
  await ctx.close();
  $("streamBtn").disabled = false;
};
</script>
</body>
</html>
"""
