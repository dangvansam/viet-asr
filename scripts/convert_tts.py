import os
import json
import soundfile as sf
import argparse
from tqdm import tqdm
from loguru import logger


def convert_metadata_to_manifest(data_dirs, output_json):
    manifest_data = []

    for ddir in data_dirs:
        meta_path = os.path.join(ddir, "metadata.txt")
        if not os.path.exists(meta_path):
            logger.warning(f"No metadata.txt found in {ddir}")
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        logger.info(f"Processing {ddir}...")
        for line in tqdm(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 3:
                speaker_id, old_audio_path, text = parts
                # Extract filename and construct correct local path
                filename = os.path.basename(old_audio_path)
                real_audio_path = os.path.join(ddir, "wavs", filename)

                if os.path.exists(real_audio_path):
                    try:
                        info = sf.info(real_audio_path)
                        duration = info.frames / info.samplerate

                        manifest_data.append({
                            "audio_filepath": real_audio_path,
                            "offset": 0.0,
                            "duration": round(duration, 3),
                            "label": speaker_id,
                            "text": text
                        })
                    except Exception as e:
                        pass

    # Save the manifest
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        for item in manifest_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Total isolated speech files processed: {len(manifest_data)}")
    logger.info(f"Saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs='+', required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    convert_metadata_to_manifest(args.data_dirs, args.output)
