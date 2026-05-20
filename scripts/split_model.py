"""Split a model file into <50 MB chunks committable to git.

Produces model.onnx.partNN files plus a chunks.json manifest that
join_model.py and the build use to reassemble and verify the model.
"""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_CHUNK_SIZE = 45 * 1024 * 1024  # 45 MiB — under GitHub's 50 MB warning


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the model file to split")
    parser.add_argument("--out-dir", required=True, help="directory to write chunks + chunks.json")
    parser.add_argument("--name", default="model.onnx",
                        help="logical model filename recorded in the manifest")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"chunk size in bytes (default: {DEFAULT_CHUNK_SIZE})")
    args = parser.parse_args()

    if args.chunk_size > 50 * 1024 * 1024:
        parser.error("chunk-size must stay under 50 MB so chunks commit cleanly to git")

    src = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not src.is_file():
        parser.error(f"model file not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = src.read_bytes()
    total_size = len(data)

    chunks = []
    for index, offset in enumerate(range(0, total_size, args.chunk_size)):
        piece = data[offset:offset + args.chunk_size]
        chunk_name = f"{args.name}.part{index:02d}"
        (out_dir / chunk_name).write_bytes(piece)
        chunks.append({
            "name": chunk_name,
            "size": len(piece),
            "sha256": sha256_bytes(piece),
        })
        print(f"[chunk] {chunk_name}  {len(piece)/1024/1024:.1f} MB")

    manifest = {
        "model": args.name,
        "total_size": total_size,
        "sha256": sha256_bytes(data),
        "chunk_size": args.chunk_size,
        "chunks": chunks,
    }
    manifest_path = out_dir / "chunks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[done] {len(chunks)} chunk(s), {total_size/1024/1024:.1f} MB total")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
