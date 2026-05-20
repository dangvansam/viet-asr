"""Reassemble a model file from chunks produced by split_model.py.

Reads chunks.json, concatenates the chunks in order, and verifies every
per-chunk hash plus the full-model hash before writing the output.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def join(chunk_dir: Path) -> bytes:
    manifest = json.loads((chunk_dir / "chunks.json").read_text())
    out = bytearray()
    for chunk in manifest["chunks"]:
        piece = (chunk_dir / chunk["name"]).read_bytes()
        if len(piece) != chunk["size"]:
            sys.exit(f"size mismatch for {chunk['name']}: "
                     f"got {len(piece)}, expected {chunk['size']}")
        if sha256_bytes(piece) != chunk["sha256"]:
            sys.exit(f"sha256 mismatch for {chunk['name']}")
        out += piece

    if len(out) != manifest["total_size"]:
        sys.exit(f"total size mismatch: got {len(out)}, expected {manifest['total_size']}")
    if sha256_bytes(bytes(out)) != manifest["sha256"]:
        sys.exit("sha256 mismatch for the reassembled model")
    return bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", required=True,
                        help="directory containing chunks.json + chunk files")
    parser.add_argument("--out", required=True, help="path to write the reassembled model")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir).resolve()
    if not (chunk_dir / "chunks.json").is_file():
        parser.error(f"chunks.json not found in {chunk_dir}")

    data = join(chunk_dir)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"[ok] reassembled {len(data)/1024/1024:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
