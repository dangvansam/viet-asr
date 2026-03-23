import argparse
import os
import datetime
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel


def rttm_to_srt(rttm_lines, srt_path):
    """Converts RTTM text lines to SubRip (SRT) format file."""
    srt_lines = []

    # Parse RTTM lines
    for idx, line in enumerate(rttm_lines):
        parts = line.strip().split()
        if len(parts) < 8:
            continue

        start_time = float(parts[3])
        duration = float(parts[4])
        end_time = start_time + duration
        speaker = parts[7]

        def format_time(seconds):
            td = datetime.timedelta(seconds=seconds)
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            milliseconds = int(td.microseconds / 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        srt_lines.append(f"{idx + 1}")
        srt_lines.append(
            f"{format_time(start_time)} --> {format_time(end_time)}")
        srt_lines.append(f"[{speaker}]")
        srt_lines.append("")

    with open(srt_path, 'w') as f:
        f.write('\n'.join(srt_lines))

    logger.success(f"Successfully wrote SRT: {srt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Sortformer Diarization Inference")
    parser.add_argument(
        "--model", type=str, default="models/diar_streaming_sortformer_4spk-v2.1.nemo")
    parser.add_argument("--audio", type=str, required=True,
                        help="Input WAV file")
    parser.add_argument("--output_dir", type=str,
                        default="data/diarization_output")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        logger.error(f"Input audio not found: {args.audio}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.audio))[0]
    rttm_path = os.path.join(args.output_dir, f"{basename}.rttm")
    srt_path = os.path.join(args.output_dir, f"{basename}.srt")

    logger.info(f"Loading diarization model from {args.model}...")
    # NeMo prints messy warnings, but we just load the model
    diar_model = SortformerEncLabelModel.restore_from(args.model)
    diar_model.eval()

    logger.info(
        f"Executing Sortformer Diarization inference on {args.audio}...")
    try:
        # Pass the audio directly
        # Format returned: List of Lists of strings [ [ 'SPEAKER...', ... ] ]
        outputs = diar_model.diarize(audio=[args.audio])

        # Some NeMo versions return tuple (text, tensors), some return just text if include_tensor_outputs=False
        if isinstance(outputs, tuple):
            rttm_lines_list = outputs[0]
        else:
            rttm_lines_list = outputs

        rttm_lines = rttm_lines_list[0]

    except Exception as e:
        logger.error(f"Diarization inference failed: {e}")
        return

    # Write RTTM
    with open(rttm_path, 'w') as f:
        for line in rttm_lines:
            f.write(line + "\n")
    logger.success(f"Successfully wrote RTTM: {rttm_path}")

    # Write SRT
    rttm_to_srt(rttm_lines, srt_path)


if __name__ == "__main__":
    main()
