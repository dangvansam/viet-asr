import argparse
import json
import os
import sys

from loguru import logger

from multitalker_asr.configs import EvalConfig
from multitalker_asr.eval import (
    DiarizationEvaluator,
    EvalDataSynthesizer,
    EvaluationPipeline,
)
from multitalker_asr.eval.metrics import DERMetric
from multitalker_asr.eval.reporters import ChartReporter, TextReporter


def parse_args():
    parser = argparse.ArgumentParser(description="Speaker Diarization Evaluation Pipeline")

    source = parser.add_argument_group("Data Source")
    source.add_argument("--source_manifest", type=str, default=None)
    source.add_argument("--audio_dir", type=str, default=None)
    source.add_argument("--rttm_dir", type=str, default=None)

    synth = parser.add_argument_group("Synthesis")
    synth.add_argument("--num_samples", type=int, default=200)
    synth.add_argument("--max_speakers", type=int, default=4)
    synth.add_argument("--min_speakers", type=int, default=2)
    synth.add_argument("--eval_data_dir", type=str, default="data/eval_diarization")

    model = parser.add_argument_group("Model")
    model.add_argument("--diar_model_path", type=str, default="models/diar_streaming_sortformer_4spk-v2.1.nemo")
    model.add_argument("--device", type=str, default="cuda")
    model.add_argument("--cuda_id", type=int, default=0)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--streaming", action="store_true", default=True)
    mode.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)

    evl = parser.add_argument_group("Evaluation")
    evl.add_argument("--eval_mode", type=str, default="all", choices=["full", "fair", "forgiving", "all"])
    evl.add_argument("--collar", type=float, default=0.25)
    evl.add_argument("--ignore_overlap", action="store_true", default=False)

    out = parser.add_argument_group("Output")
    out.add_argument("--output_dir", type=str, default="data/eval_results")
    out.add_argument("--no_charts", action="store_true")
    out.add_argument("--no_audacity_labels", action="store_true")

    pipe = parser.add_argument_group("Pipeline Control")
    pipe.add_argument("--skip_synthesis", action="store_true")
    pipe.add_argument("--skip_inference", action="store_true")

    return parser.parse_args()


def build_config(args):
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


def load_existing_manifest(eval_data_dir, output_dir):
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

    synthesizer = EvalDataSynthesizer()
    eval_manifest = None

    if args.skip_synthesis:
        logger.info("Skipping synthesis, loading existing manifest...")
        if cfg.audio_dir and cfg.rttm_dir:
            eval_manifest = synthesizer.load_from_dirs(cfg.audio_dir, cfg.rttm_dir)
        else:
            eval_manifest = load_existing_manifest(cfg.eval_data_dir, cfg.output_dir)
    elif cfg.audio_dir and cfg.rttm_dir:
        logger.info("Using pre-existing audio and RTTM directories...")
        eval_manifest = synthesizer.load_from_dirs(cfg.audio_dir, cfg.rttm_dir)
    elif cfg.source_manifest:
        logger.info("Synthesizing evaluation data...")
        eval_manifest = synthesizer.synthesize(cfg)
    else:
        logger.error("Must provide either --source_manifest or --audio_dir + --rttm_dir")
        sys.exit(1)

    if not eval_manifest:
        logger.error("No evaluation data available")
        sys.exit(1)

    logger.info(f"Eval set: {len(eval_manifest)} files")

    if args.skip_inference:
        logger.info("Skipping inference, loading existing results...")
        hyp_rttm_dir = os.path.join(cfg.output_dir, "hyp_rttm")
        for entry in eval_manifest:
            if "hyp_rttm_filepath" not in entry:
                hyp_path = os.path.join(hyp_rttm_dir, f"{entry['sample_id']}.rttm")
                if os.path.exists(hyp_path):
                    entry["hyp_rttm_filepath"] = os.path.abspath(hyp_path)
                else:
                    logger.warning(f"Hypothesis RTTM not found for {entry['sample_id']}")
    else:
        logger.info(f"Running {'streaming' if cfg.streaming else 'offline'} inference...")
        evaluator = DiarizationEvaluator(cfg)
        eval_manifest = evaluator.evaluate(eval_manifest)

    os.makedirs(cfg.output_dir, exist_ok=True)
    manifest_out = os.path.join(cfg.output_dir, "eval_manifest.json")
    with open(manifest_out, "w") as f:
        for entry in eval_manifest:
            entry_save = dict(entry)
            if "latency" in entry_save:
                lat = dict(entry_save["latency"])
                lat.pop("chunk_latencies_ms", None)
                entry_save["latency"] = lat
            f.write(json.dumps(entry_save) + "\n")

    valid_entries = [e for e in eval_manifest if e.get("hyp_rttm_filepath")]
    if not valid_entries:
        logger.error("No valid entries with hypothesis RTTMs found")
        sys.exit(1)

    if len(valid_entries) < len(eval_manifest):
        logger.warning(f"Only {len(valid_entries)}/{len(eval_manifest)} files have hypothesis RTTMs")

    logger.info("Computing DER metrics...")
    der_metric = DERMetric(eval_mode=cfg.eval_mode)
    all_results = der_metric.compute_all_modes(valid_entries, streaming=cfg.streaming)

    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Generating report...")
    text_reporter = TextReporter(model_path=cfg.diar_model_path, streaming=cfg.streaming)
    text_reporter.generate(all_results, cfg.output_dir)

    if cfg.generate_charts:
        chart_reporter = ChartReporter()
        chart_reporter.generate(all_results, cfg.output_dir)

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


if __name__ == "__main__":
    main()
