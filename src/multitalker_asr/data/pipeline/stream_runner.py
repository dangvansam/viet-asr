"""
StreamPipeline: on-the-fly, parallel processing of social-video-crawl tasks.

Source of truth is the crawl DB (API): tasks stream in carrying platform/tags/
channel/url, are downloaded to a staging dir keyed by task_id, processed through
the existing pipeline stages by a pool of GPU workers, written to a growing
manifest, then the raw + intermediate audio is deleted — only `segments/` +
manifest + per-task meta survive. Disk stays bounded by the ready-queue.

Layers (threads):
  (A) Lister   — iterate DB tasks, resume-filter by task_id → item_queue
  (B) Prefetch — download audio(+vtt) → staging/{task_id}.* → ready_queue (bounded)
  (C) GPU pool — each worker owns a DataPipeline on its device, runs batches
  (D) Writer   — lock-guarded manifest accumulation + checkpoint + cleanup
"""

import copy
import json
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from ..crawl import CrawlAPIClient, CrawlDBClient, CrawlStreamSource, MinIOFetcher
from ..crawl.tag_taxonomy import resolve_tag_family
from .checkpoint import PipelineCheckpoint
from .config import PipelineConfig
from .pipeline import DataPipeline
from .stages.crawl_seed import CrawlSeedStage
from .stages.write_manifest import DatasetMetadataWriter, WriteManifestStage

_STREAM_STEP = "s3_stream"
_INNER_EXCLUDE = {"crawl_seed", "write_manifest"}
_CRAWL_ENV = "/home/samdv/social-video-crawl/.env"


class StreamPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._stream = config.stream
        self._src_cfg = config.crawl_source

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._out = out
        self._staging = Path(self._stream.staging_dir or (out / "_staging"))
        self._meta_dir = out / "meta"
        self._extracted = out / "extracted"
        self._seed_path = out / "_seed" / "seed.jsonl"
        self._seed_path.parent.mkdir(parents=True, exist_ok=True)

        self._ck = PipelineCheckpoint(str(out / "_ingest_ck"))
        self._source = self._build_source()

        self._wm = WriteManifestStage()
        self._seed_builder = CrawlSeedStage()
        self._entries: List[Dict] = []
        self._seen: Set[str] = set()
        self._kept_records: List[Dict] = []
        self._load_existing_manifest()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._item_q: "queue.Queue" = queue.Queue(maxsize=self._stream.queue_size * 2)
        self._ready_q: "queue.Queue" = queue.Queue(maxsize=self._stream.queue_size)
        self._batch_count = 0
        self._stats = {
            "tasks": 0, "segments": 0,
            "no_audio": 0, "download_fail": 0, "skipped_resume": 0,
        }
        self._worker_pipelines: List[DataPipeline] = []

    # ---- setup -----------------------------------------------------------

    def _build_source(self) -> CrawlStreamSource:
        c = self._src_cfg
        minio = self._build_minio()
        if (c.backend or "api").lower() == "db":
            client = CrawlDBClient.from_config(c, env_path=_CRAWL_ENV)
            logger.info("Crawl source: direct Postgres (backend=db)")
        else:
            client = CrawlAPIClient(
                base_url=c.api_url or None,
                token=c.api_token or self._resolve_token(),
            )
            logger.info("Crawl source: HTTP API (backend=api)")
        return CrawlStreamSource(client, minio_fetcher=minio, language=c.language)

    def _build_minio(self) -> Optional[MinIOFetcher]:
        """Creds from config else the crawl .env; bucket from config wins (the
        .env MINIO_BUCKET points at an empty bucket — data is in another)."""
        c = self._src_cfg
        env = MinIOFetcher._load_env(_CRAWL_ENV)
        endpoint = c.minio_endpoint or env.get("MINIO_ENDPOINT")
        access = c.minio_access_key or env.get("MINIO_ACCESS_KEY")
        secret = c.minio_secret_key or env.get("MINIO_SECRET_KEY")
        bucket = c.minio_bucket or env.get("MINIO_BUCKET")
        region = c.minio_region or env.get("MINIO_REGION")
        if endpoint and access and secret and bucket:
            return MinIOFetcher.from_params(
                endpoint=endpoint,
                access_key=access,
                secret_key=secret,
                bucket=bucket,
                region=region,
                secure=c.minio_secure,
            )
        logger.warning("MinIO creds incomplete — downloads will fail")
        return None

    def _resolve_token(self) -> Optional[str]:
        path = Path(_CRAWL_ENV)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("API_TOKEN="):
                    return line.split("=", 1)[1].strip()
        return None

    def _resolve_tag_ids(self) -> Optional[List[int]]:
        c = self._src_cfg
        if c.tag_ids:
            return list(c.tag_ids)
        if c.tag_family:
            return resolve_tag_family(c.tag_family)
        return None

    def _make_worker_pipeline(self, idx: int) -> DataPipeline:
        cfg = copy.deepcopy(self._cfg)
        cfg.stages = [s for s in cfg.stages if s not in _INNER_EXCLUDE]
        device = self._stream.devices[idx % len(self._stream.devices)]
        cfg.device = device
        cfg.diarize.device = device
        cfg.transcribe.device = device
        cfg.align.device = device
        cfg.checkpoint_dir = str(self._staging / f".ck_w{idx}")
        return DataPipeline(cfg)

    def _load_existing_manifest(self) -> None:
        path = self._out / "manifest_all.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fp = entry.get("audio_filepath")
            if fp and fp not in self._seen:
                self._seen.add(fp)
                self._entries.append(entry)
        logger.info(f"Resuming manifest with {len(self._entries)} existing entries")

    # ---- run -------------------------------------------------------------

    def run(self) -> Dict:
        start = time.time()
        self._install_sigint()
        self._worker_pipelines = [
            self._make_worker_pipeline(i) for i in range(self._stream.gpu_workers)
        ]

        lister = threading.Thread(target=self._lister, name="lister", daemon=True)
        dls = [
            threading.Thread(target=self._downloader, name=f"dl{i}", daemon=True)
            for i in range(self._stream.download_workers)
        ]
        gpus = [
            threading.Thread(target=self._gpu_worker, args=(i,), name=f"gpu{i}", daemon=True)
            for i in range(self._stream.gpu_workers)
        ]
        for t in gpus + dls:
            t.start()
        lister.start()

        lister.join()
        for _ in dls:
            self._item_q.put(None)
        for t in dls:
            t.join()
        for _ in gpus:
            self._ready_q.put(None)
        for t in gpus:
            t.join()

        self._flush_manifest()
        self._write_ledger()
        self._ck.save_state()

        elapsed = round(time.time() - start, 1)
        summary = {
            **self._stats,
            "manifest_entries": len(self._entries),
            "elapsed_s": elapsed,
            "throughput_tasks_per_min": round(self._stats["tasks"] / (elapsed / 60.0), 2) if elapsed else 0,
            "manifest_path": str(self._out / "manifest_all.jsonl"),
        }
        logger.success(f"Stream complete: {summary}")
        return summary

    def _install_sigint(self) -> None:
        def handler(signum, frame):
            logger.warning("SIGINT — draining queues and flushing, run again to resume")
            self._stop.set()
        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            pass  # not in main thread (e.g. tests)

    # ---- (A) lister ------------------------------------------------------

    def _lister(self) -> None:
        skip = set(self._ck.get_processed_files(_STREAM_STEP))
        tag_ids = self._resolve_tag_ids()
        platforms = self._src_cfg.platforms or None
        try:
            for task in self._source.iter_tasks(
                platforms=platforms,
                tag_ids=tag_ids,
                status=self._src_cfg.status,
                page_limit=self._src_cfg.page_limit,
                max_items=self._src_cfg.max_items,
                skip_ids=skip,
            ):
                if self._stop.is_set():
                    break
                self._item_q.put(task)
        except Exception as exc:
            logger.error(f"Lister failed: {exc}")
        if skip:
            self._stats["skipped_resume"] = len(skip)

    # ---- (B) prefetch ----------------------------------------------------

    def _downloader(self) -> None:
        while True:
            task = self._item_q.get()
            if task is None:
                self._item_q.task_done()
                break
            try:
                staged = self._source.download(task, str(self._staging))
                if self._src_cfg.keep_meta:
                    self._source.write_meta(task, str(self._meta_dir))
                self._ready_q.put((task, staged))
            except ValueError as exc:
                # No audio object in the folder (stale/mismatched preview) — skip.
                logger.debug(f"No audio for {self._source.task_id(task)}: {exc}")
                with self._lock:
                    self._stats["no_audio"] += 1
            except Exception as exc:
                logger.warning(f"Download failed for {self._source.task_id(task)}: {exc}")
                with self._lock:
                    self._stats["download_fail"] += 1
            finally:
                self._item_q.task_done()

    # ---- (C) GPU worker --------------------------------------------------

    def _gpu_worker(self, idx: int) -> None:
        pipeline = self._worker_pipelines[idx]
        batch: List = []
        while True:
            item = self._ready_q.get()
            if item is None:
                self._ready_q.task_done()
                if batch:
                    self._process_batch(pipeline, batch)
                break
            batch.append(item)
            self._ready_q.task_done()
            if len(batch) >= self._stream.batch_size:
                self._process_batch(pipeline, batch)
                batch = []

    def _process_batch(self, pipeline: DataPipeline, batch: List) -> None:
        tasks = [t for t, _ in batch]
        seeds = [self._source.build_seed(t, s) for t, s in batch]
        records = [self._seed_builder._build_record(sd) for sd in seeds]
        try:
            processed = pipeline.run_records(records)
        except Exception as exc:
            logger.error(f"Batch failed ({len(tasks)} tasks): {exc}")
            self._cleanup(tasks)  # drop intermediates; tasks stay unmarked → retried
            return
        self._writer(tasks, seeds, processed)

    # ---- (D) writer (lock-guarded) ---------------------------------------

    def _writer(self, tasks: List[Dict], seeds: List[Dict], processed: List[Dict]) -> None:
        with self._lock:
            self._kept_records.extend(processed)
            for r in processed:
                entry = self._wm._build_manifest_entry(r, self._cfg.manifest)
                if entry is None:
                    continue
                fp = entry["audio_filepath"]
                if fp in self._seen:
                    continue
                self._seen.add(fp)
                self._entries.append(entry)

            with open(self._seed_path, "a", encoding="utf-8") as f:
                for sd in seeds:
                    f.write(json.dumps(sd, ensure_ascii=False) + "\n")

            segs_by_task: Dict[str, int] = {}
            for r in processed:
                tid = str(r.get("extra", {}).get("task_id") or "")
                segs_by_task[tid] = segs_by_task.get(tid, 0) + 1

            for task in tasks:
                tid = self._source.task_id(task)
                n = segs_by_task.get(tid, 0)
                self._stats["segments"] += n
                self._stats["tasks"] += 1
                if self._src_cfg.keep_meta:
                    self._source.write_meta(
                        task, str(self._meta_dir),
                        {"n_segments": n, "status": "ok" if n else "no_segments"},
                    )
                self._ck.mark_processed(tid, _STREAM_STEP)

            self._batch_count += 1
            if self._batch_count % max(1, self._stream.manifest_flush_every) == 0:
                self._flush_manifest()
                self._ck.save_state()

            self._cleanup(tasks)

    def _flush_manifest(self) -> None:
        self._wm._write_manifests(self._entries, self._out, self._cfg.manifest)

    def _write_ledger(self) -> None:
        try:
            DatasetMetadataWriter(self._out, self._cfg.manifest).write(
                self._kept_records, str(self._seed_path)
            )
        except Exception as exc:
            logger.warning(f"Ledger write skipped: {exc}")

    def _cleanup(self, tasks: List[Dict]) -> None:
        for task in tasks:
            tid = self._source.task_id(task)
            if not self._stream.keep_raw:
                for f in self._staging.glob(f"{tid}.*"):   # wav / mp4 / vtt
                    self._unlink(f)
            if not self._stream.keep_extracted:
                self._unlink(self._extracted / f"{tid}.wav")

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
