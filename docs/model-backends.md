# Model Backends

Tham chiếu ngắn gọn các model backend của pipeline. Mỗi nhóm theo pattern **registry + factory**: `build_<nhóm>_backend(name, **kwargs)` tra trong `<NHÓM>_REGISTRY`. Mọi backend có `load(device)` + một hàm suy luận, trả về dataclass kết quả chuẩn.

Kiến trúc service hoá (REST, URL-only, docker-compose, chạy 2 GPU) xem [data-pipeline.md §11](data-pipeline.md). Mỗi nhóm có một REST contract; client `service` (URL-only) dùng chung cho nhiều server.

---

**Chuẩn hoá theo OpenAI Audio API.** Mọi model local (trừ Google) chạy dưới dạng REST
service nói **cùng format OpenAI** — ASR `POST /v1/audio/transcriptions` (verbose_json:
text, language, segments/words, hỗ trợ word/segment timestamps + pass-through diarization,
streaming, logprobs); VAD `POST /v1/audio/vad`; align `POST /v1/audio/alignments` (chung
envelope `{task, duration, segments[], words[]}`). Pipeline chỉ giữ HTTP client + URL — đổi
backend = đổi URL. Serving wrapper: `serving/app_factory.py`
(`create_transcription_service` / `create_vad_service` / `create_alignment_service`).

## VAD — `vad_backends/` · `config.vad.backend`

`build_vad_backend(name)` → `BaseVADBackend.detect(audio, sr) -> VADResult{segments[start,end], speech_ratio}`.

| key | Model / cách chạy | Ghi chú |
|---|---|---|
| `silero` | silero-vad | mặc định, nhẹ (chạy trong service container) |
| `pyannote_seg` | pyannote/segmentation-3.0 | gated → `HF_TOKEN`, hoặc offline (`HF_HUB_OFFLINE=1` + `PYANNOTE_CACHE`); pyannote.audio **4.x** |
| `ten` | ten-vad | cần `libc++1` |
| `fsmn` | FunASR FSMN-VAD | |
| `consensus` | hợp nhất nhiều VAD theo frame-vote | `backend_kwargs: {providers, strategy: majority\|intersection\|union}` |
| `service` | client → `POST /v1/audio/vad` | URL-only, server `scripts/serve_vad.py` |

## ASR — `asr_backends/` · `multi_transcribe.backends[].name`

`build_asr_backend(name)` → `BaseASRBackend.transcribe(audio, sr, language) -> ASRResult{text, confidence, language, word_timings}`.

| key | Model / cách chạy | Loại | Ghi chú |
|---|---|---|---|
| `openai_transcription` (alias `openai`) | client `POST /v1/audio/transcriptions` chung | service | **client thống nhất** cho mọi ASR service; chỉ đổi `base_url` (+`model`) |
| `google_speech` | Google Speech-to-Text V2 (chirp) | cloud | backend cloud duy nhất; cần GCP creds |
| `funasr` / `nemotron` / `vietasr` | loader in-venv | (server-side) | KHÔNG dùng trong pipeline; chỉ chạy *bên trong* service container qua `scripts/serve_asr.py` |

**Service nào nói `/v1/audio/transcriptions`:** Qwen3-ASR (vLLM, native) · Fun-ASR-MLT
(vLLM, modelscope/FunASR `serve_vllm.py`) · nemotron & vietasr (in-venv, bọc bởi
`create_transcription_service`). Tất cả → cùng 1 client `openai_transcription`.

Ensemble (gộp nhiều ASR): `ensemble_strategy = single \| vote \| rover \| llm_judge` (xem `asr_backends/ensemble.py`).

## Word Alignment — `align_backends/` · `config.align.backend`

`build_align_backend(name)` → `BaseAlignBackend.align(audio_path, text, language) -> AlignResult{words[text,start,end], score}`.

| key | Model | Ghi chú |
|---|---|---|
| `mms_fa` | torchaudio MMS_FA | in-venv, không cần transformers — mặc định an toàn |
| `nemo_nfa` | NeMo forced aligner (CTC) | in-venv |
| `qwen3` | Qwen3-ForcedAligner (in-venv) | xung đột transformers/NeMo → ưu tiên dùng dạng service |
| `funasr_align` / `funasr_nano_align` | FunASR aligner | trọng số Mandarin, thuật toán language-agnostic |
| `qwen3_service` | client → Qwen3 aligner | `POST /v1/audio/alignments`, server `scripts/serve_qwen3_aligner.py` (venv riêng) |
| `service` | client `/v1/audio/alignments` chung | server `scripts/serve_align.py` (mms_fa/nemo_nfa) |

## Speaker & Gender — services (REST, OpenAI-style)

Cùng convention OpenAI như ASR (multipart `file` in; JSON `{...}` out; flags qua form).

| Nhóm | Client | Endpoint | Flags | Ghi chú |
|---|---|---|---|---|
| Speaker embedding | `SpeakerEmbedder` (`speaker/embedder.py`) | `POST /v1/audio/embeddings` | `embedding=true/false`, `verify=true/false` | ECAPA-TDNN → `{embedding, dim, score, confidence}`; `speaker_verify.url` (base), cluster_threshold 0.45 |
| Gender | stage `gender_classify` | `POST /v1/audio/classifications` | `model`, `probs=true/false` | ensemble wav2vec2+ECAPA → `{label, confidence, probs}`; `gender.url` (base) |

Server-side: `serving/app_factory.create_embedding_service` / `create_classification_service`
(dùng trong repo speaker-recognition / gender-classification-service). Client tự suy base
URL (chấp nhận cả legacy `/embed`,`/predict`).

---

## Endpoints & ports (verified live 2026-06-14)

Mọi model service nói **cùng OpenAI Audio API** (`POST /v1/audio/*`, multipart `file`) +
`GET /health`. Một client cho mỗi nhóm; đổi backend = đổi URL.

| Service | Host port | Route | Engine | Status |
|---|---|---|---|---|
| vad | 9401 | `/v1/audio/vad` | pyannote_seg (silero opt) | ✅ |
| asr-qwen3 | 8101 | `/v1/audio/transcriptions` | Qwen3-ASR-1.7B (vLLM) | ✅ (json; verbose_json ⇒ 400) |
| asr-nemotron | 9404 | `/v1/audio/transcriptions` | nemotron-3.5 (NeMo) | ✅ verbose_json |
| asr-vietasr | 9104 | `/v1/audio/transcriptions` | viet-asr (ONNX) | ✅ verbose_json |
| asr-funasr | 9402 | `/v1/audio/transcriptions` | Fun-ASR-MLT (vLLM) | ✅ (no `/health`; TCP healthcheck) |
| align-mms | 9403 | `/v1/audio/alignments` | torchaudio MMS_FA | ✅ |
| align-qwen3 | 8103 | `/v1/audio/alignments` | Qwen3-ForcedAligner | ✅ |
| speaker | 2010 | `/v1/audio/embeddings` | ECAPA-TDNN (stateless, no DB) | ✅ |
| gender | 8000 | `/v1/audio/classifications` | wav2vec2+ECAPA ensemble | ✅ |

All run via this repo's `docker-compose.yml` (speaker/gender build from their repo Dockerfiles
— no external compose). Every service mounts the host HF cache `${HF_CACHE_HOST:-/data/samdv/.cache/huggingface}`
at `/root/.cache/huggingface` and runs **HF-offline** (`HF_OFFLINE=1`) — no ModelScope, no
named volumes. Speaker is **stateless** (`DISABLE_DB=true`, no MySQL).

ASR/word-timestamps: `verbose_json` cho timestamps; vLLM Qwen3-ASR chỉ hỗ trợ `json`/`text`
→ client tự fallback (không có word timings, dùng align service bù).

## Service mode (URL-only, docker-compose, dual-GPU)

- Config production chỉ truyền URL: `backend: openai_transcription` + `kwargs.base_url`, hoặc `speaker_verify.url` / `gender.url`. Image `pipeline` không có torch/nemo.
- Mọi image model dùng base **`pytorch/pytorch:2.10.0-cuda12.8`** → torch 2.10+cu128 chạy được **cả RTX 4090 (sm_86) và Blackwell (sm_120)**. Đặt GPU qua env CDI `GPU_DEVICE`/`NEMOTRON_GPU`/`QWEN_GPU`.
- Chạy: `cp .env.example .env` → `docker compose up vad asr-funasr align-mms` → `docker compose --profile full up` (nemotron/vietasr/qwen3).
- **Offline**: tải sẵn model vào HF cache, set `HF_CACHE_HOST` (bind-mount) + `HF_OFFLINE=1`; funasr trỏ `FUNASR_MODEL` vào snapshot dir; pyannote cần `PYANNOTE_CACHE=/root/.cache/huggingface/hub`.

## Thêm backend mới

1. Tạo class kế thừa `Base<Nhóm>Backend` trong thư mục `*_backends/`, implement `load()` + hàm suy luận.
2. Đăng ký vào `<NHÓM>_REGISTRY` (trong `*_backends/__init__.py`).
3. Dùng qua config (`backend: <key>` + `backend_kwargs`). Nếu là model nặng → viết `scripts/serve_*.py` (dựa `serving/app_factory.py`) và gọi qua client `service`.
