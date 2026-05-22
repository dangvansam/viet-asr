"""Self-contained browser test console, served at GET /.

Lets you pick any pipeline — a preset or a custom set of modules (to exercise a
single model on its own) — and run it in both batch and streaming modes, with
the full module result JSON shown for every run.
"""

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
         max-width: 760px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0; font-size: 22px; }
  .sub { color: #888; margin: 2px 0 20px; }
  section { border: 1px solid #8884; border-radius: 10px; padding: 16px; margin: 14px 0; }
  h2 { margin: 0 0 4px; font-size: 16px; }
  code { background: #8882; padding: 1px 6px; border-radius: 4px; font-size: 13px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0; }
  button, select { font: inherit; padding: 7px 12px; border-radius: 7px;
           border: 1px solid #8886; background: #7773; cursor: pointer; }
  button:hover:not(:disabled) { background: #7775; }
  button:disabled { opacity: .5; cursor: default; }
  button.rec { background: #d33; color: #fff; border-color: #d33; }
  button.go { background: #2a7; color: #fff; border-color: #2a7; }
  input[type=file] { font: inherit; max-width: 100%; }
  .chk { border: 1px solid #8884; border-radius: 6px; padding: 4px 9px;
         cursor: pointer; user-select: none; font-size: 14px; }
  .chk input { vertical-align: -1px; margin-right: 4px; }
  .label { color: #888; font-size: 12px; text-transform: uppercase;
           letter-spacing: .5px; margin: 12px 0 3px; }
  .hint { color: #888; font-size: 13px; margin-top: 6px; }
  .out { background: #8881; border-radius: 7px; padding: 10px 12px; min-height: 40px;
         white-space: pre-wrap; word-break: break-word; font-size: 13px;
         font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         max-height: 340px; overflow: auto; margin: 0; }
  .out.partial { color: #888; }
  .text { font-size: 16px; margin: 6px 0; min-height: 22px;
          font-family: system-ui, sans-serif; }
  .status { color: #888; font-size: 13px; margin-top: 16px; }
</style>
</head>
<body>
<h1>VietASR</h1>
<div class="sub">Offline Vietnamese speech AI — test console</div>

<!-- ====================== Pipeline picker ====================== -->
<section>
  <h2>Pipeline</h2>
  <div class="row">
    <select id="modeSel">
      <option value="auto">Auto (per mode)</option>
      <option value="preset">Preset</option>
      <option value="modules">Custom modules</option>
    </select>
    <span id="presetWrap"><select id="presetSel"></select></span>
  </div>
  <div class="row" id="modulesWrap" style="display:none"></div>
  <div class="hint" id="pipelineHint">—</div>
</section>

<!-- ====================== Batch ====================== -->
<section>
  <h2>Batch — non-streaming</h2>
  <code>POST /v1/audio/transcriptions</code>
  <div class="row">
    <input type="file" id="batchFile" accept="audio/*">
    <button id="batchBtn" class="go">Transcribe</button>
  </div>
  <div class="row">
    <label>Long audio:
      <select id="vadSegment">
        <option value="auto">auto — VAD-segment if &gt; 3 min</option>
        <option value="on">on — always VAD-segment</option>
        <option value="off">off — whole file at once</option>
      </select>
    </label>
  </div>
  <div class="label">transcript</div>
  <div class="text" id="batchText">—</div>
  <div class="label">full result</div>
  <pre class="out" id="batchOut">—</pre>
</section>

<!-- ====================== Streaming ====================== -->
<section>
  <h2>Streaming</h2>
  <code>WebSocket /v1/stream</code>
  <div class="row">
    <button id="micBtn">● Record mic</button>
    <span style="color:#888">or</span>
    <input type="file" id="streamFile" accept="audio/*">
    <button id="streamBtn">Stream file</button>
  </div>
  <div class="label">partial transcript</div>
  <div class="text partial" id="partialText">—</div>
  <div class="label">final result</div>
  <pre class="out" id="streamOut">—</pre>
</section>

<div class="status" id="statusBar">loading…</div>

<script>
const $ = (id) => document.getElementById(id);
const statusBar = $("statusBar");
const setStatus = (t) => { statusBar.textContent = t; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pretty = (o) => JSON.stringify(o, null, 2);

// ---- Pipeline selection -------------------------------------------------
let PIPELINES = { presets: [], modules: [], default: ["preset", "transcribe"] };

async function loadPipelines() {
  try {
    PIPELINES = await (await fetch("/v1/pipelines")).json();
  } catch (err) { setStatus("could not load pipelines: " + err); return; }

  const ps = $("presetSel");
  ps.innerHTML = "";
  PIPELINES.presets.slice().sort().forEach((name) => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    ps.appendChild(o);
  });

  const mw = $("modulesWrap");
  mw.innerHTML = "";
  PIPELINES.modules.forEach((name) => {
    const lbl = document.createElement("label");
    lbl.className = "chk";
    lbl.innerHTML = '<input type="checkbox" id="mod_' + name +
                    '" value="' + name + '"> ' + name;
    mw.appendChild(lbl);
  });

  // Default: Auto — batch runs ASR + ITN (whole-text ITN), streaming runs the
  // stream preset (ITN per sentence). Pre-tick vietasr + itn so switching to
  // Custom is ready to go.
  $("modeSel").value = "auto";
  ["vietasr", "itn"].forEach((m) => {
    const c = $("mod_" + m);
    if (c) c.checked = true;
  });
  if (PIPELINES.presets.includes("transcribe")) ps.value = "transcribe";
  syncMode();
  health();
}

function syncMode() {
  const mode = $("modeSel").value;
  $("presetWrap").style.display = mode === "preset" ? "" : "none";
  $("modulesWrap").style.display = mode === "modules" ? "flex" : "none";
  updateHint();
}

// The pipeline for a run — runMode is "batch" or "stream". In Auto mode,
// batch uses ASR + ITN (whole-text ITN) and streaming uses the stream preset
// (ITN per sentence). Returns {pipeline} / {modules}, or null when a custom
// selection is empty.
function pipelineSel(runMode) {
  const mode = $("modeSel").value;
  if (mode === "auto") {
    return runMode === "stream"
      ? { pipeline: "stream" }
      : { pipeline: "batch" };
  }
  if (mode === "modules") {
    const mods = [...document.querySelectorAll("#modulesWrap input:checked")]
      .map((c) => c.value);
    return mods.length ? { modules: mods.join(",") } : null;
  }
  return { pipeline: $("presetSel").value };
}

function updateHint() {
  if ($("modeSel").value === "auto") {
    $("pipelineHint").textContent =
      "auto · batch → batch preset (VAD + ASR + ITN + segments) · " +
      "streaming → stream preset (ITN per sentence)";
    return;
  }
  const sel = pipelineSel("batch");
  if (!sel) { $("pipelineHint").textContent =
                "select at least one module to run"; return; }
  $("pipelineHint").textContent = sel.pipeline
    ? "preset · " + sel.pipeline
    : "custom · " + sel.modules.split(",").join(" → ");
}

$("modeSel").onchange = syncMode;
$("presetSel").onchange = updateHint;
$("modulesWrap").onchange = updateHint;

async function health() {
  try {
    const h = await (await fetch("/health")).json();
    setStatus("ready · server default: " + h.pipeline);
  } catch { setStatus("ready"); }
}

// ---- Batch (REST) -------------------------------------------------------
$("batchBtn").onclick = async () => {
  const file = $("batchFile").files[0];
  if (!file) { setStatus("pick an audio file first"); return; }
  const sel = pipelineSel("batch");
  if (!sel) { setStatus("select at least one module"); return; }

  $("batchText").textContent = "…";
  $("batchOut").textContent = "…";
  setStatus("transcribing " + file.name + " …");

  const fd = new FormData();
  fd.append("file", file);
  fd.append("response_format", "verbose_json");
  fd.append("vad_segment", $("vadSegment").value);
  if (sel.pipeline) fd.append("pipeline", sel.pipeline);
  if (sel.modules) fd.append("modules", sel.modules);

  try {
    const res = await fetch("/v1/audio/transcriptions",
                            { method: "POST", body: fd });
    const data = await res.json();
    if (res.ok) {
      $("batchText").textContent = data.text || "(no transcript)";
      $("batchOut").textContent = pretty(data);
      setStatus("done · " + (data.pipeline || ""));
    } else {
      $("batchText").textContent = "—";
      $("batchOut").textContent = "error: " + (data.detail || res.status);
      setStatus("failed");
    }
  } catch (err) {
    $("batchOut").textContent = "error: " + err;
    setStatus("failed");
  }
};

// ---- Streaming (WebSocket) ---------------------------------------------
function wsUrl(rate, encoding) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const q = new URLSearchParams({ sample_rate: rate, encoding });
  const sel = pipelineSel("stream") || {};
  if (sel.pipeline) q.set("pipeline", sel.pipeline);
  if (sel.modules) q.set("modules", sel.modules);
  return proto + "//" + location.host + "/v1/stream?" + q.toString();
}

function openStream(rate) {
  const ws = new WebSocket(wsUrl(rate, "pcm_f32le"));
  ws.binaryType = "arraybuffer";
  $("partialText").textContent = "—";
  $("streamOut").textContent = "—";
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "ready") {
      setStatus("streaming · " + (msg.pipeline || ""));
    } else if (msg.type === "partial") {
      $("partialText").textContent = msg.text || "—";
      $("streamOut").textContent = pretty(msg.result || { text: msg.text });
    } else if (msg.type === "final") {
      $("partialText").textContent = msg.text || "—";
      $("streamOut").textContent = pretty(msg.result || { text: msg.text });
      setStatus("final received");
    } else if (msg.type === "error") {
      $("streamOut").textContent = "server error: " + msg.message;
      setStatus("server error");
    }
  };
  return ws;
}

let mic = null;

async function startMic() {
  if (!pipelineSel("stream")) { setStatus("select at least one module"); return; }
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
  try {
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  } catch {
    setStatus("websocket failed");
    stream.getTracks().forEach((t) => t.stop());
    await ctx.close();
    return;
  }

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
  if (!pipelineSel("stream")) { setStatus("select at least one module"); return; }

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

loadPipelines();
</script>
</body>
</html>
"""
