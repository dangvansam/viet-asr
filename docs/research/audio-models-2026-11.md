# Candidate audio models — integration study (2026-11-04)

ASR = the bundled int8 Zipformer-vi transducer. WER is `spoken_canonical` unless noted.

## Speech enhancement — NVIDIA NemotronLabs RE-USE

| | |
|---|---|
| License | OpenMDW-1.1 (permissive, commercial use allowed, keep notice) |
| Size | 9.6M params, 38.6 MB safetensors, PyTorch + `mamba_ssm` CUDA kernels, no ONNX |
| Architecture | STFT (5 ms hop) → dense conv encoder → 30 bidirectional time/frequency Mamba blocks → mag/phase decoders |
| Vendor support | GPU only ("CPU mode is not supported"), Linux, clips < 10 s (chunked inference for longer) |

Measured with a pure-PyTorch port of the published weights (no CUDA kernels):

- Port check: white noise at 0 dB SNR → SI-SDR 0.0 → 14.6 dB. Equal-level competing
  speaker → no improvement (the model keeps one "main" speaker by design).
- CPU speed: RTF ≈ 10.6 (8 threads), i.e. ~250× the cost of the ASR itself (RTF 0.04).
- ASR impact, 80 utterances per set:

| Set | Original | RE-USE → ASR |
|---|---|---|
| tongdai-hard-reviewed | 21.48 | **19.84** |
| meeting-hard-reviewed | 30.19 | 30.00 |

Conclusion: real gain on telephone audio, none on meetings, and far too heavy for
on-device use. Integrate only as an optional **server/GPU** pre-processing stage
(`denoise` module behind the CUDA build), not in the default pipelines. An ONNX
export needs the selective scan rewritten as an ONNX Loop/Scan; expect it to stay
much slower than the ASR on CPU.

## VAD — FSMN-VAD vs Silero vs TEN-VAD

FSMN-VAD (FunASR): Apache-2.0, 0.5 MB int8 ONNX, but needs Hamming-window fbank +
LFR(5,1) + CMVN and a ~1k-line endpoint state machine; trained on Mandarin.

Long-audio test: 60 hard utterances concatenated with 0.2–2.0 s gaps, segmented by
each VAD, then decoded by the same ASR.

| Segmentation | tongdai (341 s) | meeting (226 s) |
|---|---|---|
| Oracle (true utterances) | 21.98 | 28.60 |
| Silero, old defaults (pause 1.2 s) | 27.13 | 29.64 |
| TEN-VAD, old defaults | 26.33 | 31.37 |
| FSMN-VAD (FunASR defaults) | 23.83 | 30.68 |
| **Silero, adaptive pauses (new default)** | **21.58** | **27.73** |

The gap was the pause rule, not the detector. The new default ends an utterance on a
1.2 s pause while it is shorter than 4 s, on 500 ms after 4 s and on 300 ms after
10 s. On 900 single-utterance clips it keeps the old accuracy except vlsp2021-T1
(long lectures, +0.27 spoken / +1.35 pnc). Conclusion: keep Silero (default) and
TEN-VAD; FSMN-VAD adds no accuracy over the tuned Silero and costs a new frontend.

Side finding: the Zipformer encoder fails on inputs longer than 4008 fbank frames
(≈ 40 s). The SDK now splits windows above 39 s at the quietest point and caps
`max_utterance_ms` at 38 s.

## Word timestamps — Desert Ant Labs Align

- License: Desert Ant Labs Source-Available (commercial license required at scale) —
  not compatible with redistribution in an Apache-2.0 SDK.
- Languages: de, en, es, fr, it, ja, ko, pt, zh — **no Vietnamese**; unsupported
  languages are returned unchanged.
- Runtimes: Core ML / LiteRT only; no Android, no browser.

Conclusion: not usable. Implemented instead: word timestamps from the transducer's
own emission frames (`emit_words`), identical to sherpa-onnx token timestamps
(1824 words, median Δ 0 ms). Absolute accuracy against hand-marked boundaries is
not yet measured — that needs a labelled Vietnamese set.
