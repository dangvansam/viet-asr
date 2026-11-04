import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict


class ChunkJoiner:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self.manifest = json.loads((chunk_dir / "chunks.json").read_text())

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def join_file(self, entry: Dict[str, object]) -> bytes:
        out = bytearray()
        for chunk in entry["chunks"]:
            piece = (self.chunk_dir / chunk["name"]).read_bytes()
            if len(piece) != chunk["size"] or self.digest(piece) != chunk["sha256"]:
                sys.exit(f"corrupt chunk {chunk['name']}")
            out += piece
        if len(out) != entry["total_size"] or self.digest(bytes(out)) != entry["sha256"]:
            sys.exit(f"corrupt reassembled file {entry['name']}")
        return bytes(out)

    def write_all(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.manifest["files"]:
            data = self.join_file(entry)
            (out_dir / entry["name"]).write_bytes(data)
            print(f"[ok] {entry['name']} {len(data) / 1024 / 1024:.1f} MB")
        tokens = self.manifest.get("tokens", "tokens.txt")
        (out_dir / tokens).write_bytes((self.chunk_dir / tokens).read_bytes())
        print(f"[ok] {tokens}")


class JoinCommand:
    def run(self) -> None:
        parser = argparse.ArgumentParser(
            description="Reassemble and verify the chunked transducer model described by chunks.json.")
        parser.add_argument("--chunk-dir", required=True)
        parser.add_argument("--out-dir", required=True)
        args = parser.parse_args()
        chunk_dir = Path(args.chunk_dir).resolve()
        if not (chunk_dir / "chunks.json").is_file():
            parser.error(f"chunks.json not found in {chunk_dir}")
        ChunkJoiner(chunk_dir).write_all(Path(args.out_dir).resolve())


if __name__ == "__main__":
    JoinCommand().run()
