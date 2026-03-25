"""
Speaker Diarization Evaluation Pipeline

Synthesize multi-speaker test audio, run Sortformer diarization inference
(streaming or non-streaming), compute DER metrics, and generate reports
with charts and Audacity labels.

Usage examples:

  # Full pipeline: synthesize + infer + evaluate
  uv run scripts/eval_diarization.py \
      --source_manifest data/val_single_speaker.json \
      --num_samples 200 \
      --max_speakers 4 \
      --streaming \
      --eval_mode all

  # Non-streaming mode
  uv run scripts/eval_diarization.py \
      --source_manifest data/val_single_speaker.json \
      --no_streaming

  # Evaluate on pre-existing data (skip synthesis)
  uv run scripts/eval_diarization.py \
      --audio_dir data/eval_audio/ \
      --rttm_dir data/eval_rttm/ \
      --streaming

  # Re-run metrics only (skip synthesis and inference)
  uv run scripts/eval_diarization.py \
      --skip_synthesis --skip_inference \
      --eval_data_dir data/eval_diarization \
      --output_dir data/eval_results
"""

import argparse
import json
import os
import sys

from loguru import logger

from multitalker_asr.config import EvalConfig
from multitalker_asr.eval.synthesize_eval_data import (
    synthesize_eval_set,
    load_eval_manifest_from_dirs,
)
from multitalker_asr.eval.diar_inference import run_diar_inference
from multitalker_asr.eval.metrics import calculate_metrics, save_metrics
from multitalker_asr.eval.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    source = parser.add_argument_group("Data Source")
    source.add_argument(
        "--source_manifest",
        type=str,
        default=None,
        help="Single-speaker manifest (JSONL) for synthesizing eval data",
    )
    source.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory of pre-mixed WAV files (skip synthesis)",
    )
    source.add_argument(
        "--rttm_dir",
        type=str,
        default=None,
        help="Directory of ground truth RTTM files (skip synthesis)",
    )

    # Synthesis
    synth = parser.add_argument_group("Synthesis")
    synth.add_argument("--num_samples", type=int, default=200)
    synth.add_argument("--max_speakers", type=int, default=4)
    synth.add_argument("--min_speakers", type=int, default=2)
    synth.add_argument(
        "--eval_data_dir",
        type=str,
        default="data/eval_diarization",
        help="Directory to store synthesized eval data",
    )

    # Model
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--diar_model_path",
        type=str,
        default="models/diar_streaming_sortformer_4spk-v2.1.nemo",
    )
    model.add_argument("--device", type=str, default="cuda")
    model.add_argument("--cuda_id", type=int, default=0)

    # Inference mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming inference (default)",
    )
    mode.add_argument(
        "--no_streaming",
        action="store_true",
        help="Use non-streaming (offline) inference",
    )
    parser.add_argument("--batch_size", type=int, default=1)

    # Evaluation
    evl = parser.add_argument_group("Evaluation")
    evl.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["full", "fair", "forgiving", "all"],
        help="DER evaluation mode(s)",
    )
    evl.add_argument("--collar", type=float, default=0.25)
    evl.add_argument(
        "--ignore_overlap", action="store_true", default=False
    )

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output_dir", type=str, default="data/eval_results"
    )
    out.add_argument(
        "--no_charts", action="store_true", help="Skip chart generation"
    )
    out.add_argument(
        "--no_audacity_labels",
        action="store_true",
        help="Skip Audacity label generation",
    )

    # Pipeline control
    pipe = parser.add_argument_group("Pipeline Control")
    pipe.add_argument(
        "--skip_synthesis",
        action="store_true",
        help="Skip synthesis, use existing data in eval_data_dir",
    )
    pipe.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference, only recompute metrics from existing RTTMs",
    )

    return parser.parse_args()


def build_config(args):
    """Build EvalConfig from parsed arguments."""
    return EvalConfig(
        diar_model_path=args.diar_model_path,
        device=args.device,
        cuda_id=args.cuda_id,
        source_manifest=args.source_manifest,
        eval_data_dir=args.eval_data_dir,
        num_samples=args.num_samples,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        audio_dir=args.audio_dir,
        rttm_dir=args.rttm_dir,
        streaming=not args.no_streaming,
        batch_size=args.batch_size,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
        eval_mode=args.eval_mode,
        output_dir=args.output_dir,
        generate_charts=not args.no_charts,
        generate_audacity_labels=not args.no_audacity_labels,
    )


def _load_existing_manifest(eval_data_dir, output_dir):
    """Load eval manifest from a previous synthesis run or inference run."""
    # Check output_dir first (has hyp_rttm paths from previous inference)
    for search_dir in [output_dir, eval_data_dir]:
        manifest_path = os.path.join(search_dir, "eval_manifest.json")
        if os.path.exists(manifest_path):
            manifest = []
            with open(manifest_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        manifest.append(json.loads(line))
            logger.info(f"Loaded existing manifest: {manifest_path} ({len(manifest)} entries)")
            return manifest
    return None


def main():
    args = parse_args()
    cfg = build_config(args)

    logger.info("=" * 60)
    logger.info("Speaker Diarization Evaluation Pipeline")
    logger.info(f"Mode: {'Streaming' if cfg.streaming else 'Non-streaming'}")
    logger.info("=" * 60)

    # ── Step 1: Prepare eval data ──────────────────────────────────
    eval_manifest = None

    if args.skip_synthesis:
        logger.info("Skipping synthesis, loading existing manifest...")
        if cfg.audio_dir and cfg.rttm_dir:
            eval_manifest = load_eval_manifest_from_dirs(cfg.audio_dir, cfg.rttm_dir)
        else:
            eval_manifest = _load_existing_manifest(cfg.eval_data_dir, cfg.output_dir)
    elif cfg.audio_dir and cfg.rttm_dir:
        logger.info("Using pre-existing audio and RTTM directories...")
        eval_manifest = load_eval_manifest_from_dirs(cfg.audio_dir, cfg.rttm_dir)
    elif cfg.source_manifest:
        logger.info("Synthesizing evaluation data...")
        eval_manifest = synthesize_eval_set(cfg)
    else:
        logger.error(
            "Must provide either --source_manifest or --audio_dir + --rttm_dir"
        )
        sys.exit(1)

    if not eval_manifest:
        logger.error("No evaluation data available")
        sys.exit(1)

    logger.info(f"Eval set: {len(eval_manifest)} files")

    # ── Step 2: Run inference ──────────────────────────────────────
    if args.skip_inference:
        logger.info("Skipping inference, loading existing results...")
        # Ensure hyp_rttm paths exist in manifest
        hyp_rttm_dir = os.path.join(cfg.output_dir, "hyp_rttm")
        for entry in eval_manifest:
            if "hyp_rttm_filepath" not in entry:
                hyp_path = os.path.join(
                    hyp_rttm_dir, f"{entry['sample_id']}.rttm"
                )
                if os.path.exists(hyp_path):
                    entry["hyp_rttm_filepath"] = os.path.abspath(hyp_path)
                else:
                    logger.warning(
                        f"Hypothesis RTTM not found for {entry['sample_id']}"
                    )
    else:
        logger.info(
            f"Running {'streaming' if cfg.streaming else 'offline'} inference..."
        )
        eval_manifest = run_diar_inference(cfg, eval_manifest)

    # Save manifest with inference results
    os.makedirs(cfg.output_dir, exist_ok=True)
    manifest_out = os.path.join(cfg.output_dir, "eval_manifest.json")
    with open(manifest_out, "w") as f:
        for entry in eval_manifest:
            # Remove large chunk latency arrays from saved manifest
            entry_save = dict(entry)
            if "latency" in entry_save:
                lat = dict(entry_save["latency"])
                lat.pop("chunk_latencies_ms", None)
                entry_save["latency"] = lat
            f.write(json.dumps(entry_save) + "\n")

    # ── Step 3: Compute metrics ────────────────────────────────────
    # Check all entries have hypothesis RTTMs
    valid_entries = [
        e for e in eval_manifest if e.get("hyp_rttm_filepath")
    ]
    if not valid_entries:
        logger.error("No valid entries with hypothesis RTTMs found")
        sys.exit(1)

    if len(valid_entries) < len(eval_manifest):
        logger.warning(
            f"Only {len(valid_entries)}/{len(eval_manifest)} files have "
            f"hypothesis RTTMs"
        )

    logger.info("Computing DER metrics...")
    all_results = calculate_metrics(valid_entries, cfg)

    # Save raw metrics
    save_metrics(all_results, cfg.output_dir)

    # ── Step 4: Generate report ────────────────────────────────────
    logger.info("Generating report...")
    generate_report(all_results, cfg)

    # ── Summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    for mode_name, result in all_results.items():
        agg = result["aggregate"]
        logger.info(
            f"  [{mode_name:>10}] DER={agg['DER'] * 100:.2f}% "
            f"(FA={agg['FA'] * 100:.2f}%, "
            f"Miss={agg['Miss'] * 100:.2f}%, "
            f"Conf={agg['Confusion'] * 100:.2f}%)"
        )

    logger.info("")
    logger.info(f"Results: {cfg.output_dir}")
    logger.info(f"  Report:  {os.path.join(cfg.output_dir, 'report.txt')}")
    logger.info(f"  Metrics: {os.path.join(cfg.output_dir, 'metrics.json')}")
    if cfg.generate_charts:
        logger.info(f"  Charts:  {os.path.join(cfg.output_dir, 'charts/')}")
    if cfg.generate_audacity_labels:
        logger.info(f"  Labels:  {os.path.join(cfg.output_dir, 'audacity_labels/')}")


if __name__ == "__main__":
    main()
