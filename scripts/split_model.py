import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DEFAULT_CHUNK_SIZE = 45 * 1024 * 1024
ROLES = ("encoder", "decoder", "joiner")


@dataclass(frozen=True)
class ChunkRecord:
    name: str
    size: int
    sha256: str

    def to_json(self) -> Dict[str, object]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


class ModelFileLocator:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def find(self, role: str) -> Path:
        for candidate in (f"{role}.int8.onnx", f"{role}.onnx"):
            path = self.model_dir / candidate
            if path.is_file():
                return path
        raise FileNotFoundError(f"no {role}.int8.onnx or {role}.onnx in {self.model_dir}")


class ModelSplitter:
    def __init__(self, out_dir: Path, chunk_size: int):
        self.out_dir = out_dir
        self.chunk_size = chunk_size

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def clean(self) -> None:
        for stale in self.out_dir.glob("*.part*"):
            stale.unlink()

    def split(self, role: str, source: Path) -> Dict[str, object]:
        data = source.read_bytes()
        chunks: List[ChunkRecord] = []
        for index, offset in enumerate(range(0, len(data), self.chunk_size)):
            piece = data[offset:offset + self.chunk_size]
            name = f"{source.name}.part{index:02d}"
            (self.out_dir / name).write_bytes(piece)
            chunks.append(ChunkRecord(name, len(piece), self.digest(piece)))
            print(f"[chunk] {name}  {len(piece) / 1024 / 1024:.1f} MB")
        return {
            "role": role,
            "name": source.name,
            "total_size": len(data),
            "sha256": self.digest(data),
            "chunks": [c.to_json() for c in chunks],
        }


class ManifestWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def write(self, model_name: str, files: List[Dict[str, object]], chunk_size: int) -> Path:
        manifest = {
            "manifest_version": 2,
            "model": model_name,
            "architecture": "zipformer2-transducer",
            "chunk_size": chunk_size,
            "files": files,
            "tokens": "tokens.txt",
        }
        path = self.out_dir / "chunks.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path


class SplitCommand:
    def parse(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Split a transducer model (encoder/decoder/joiner + tokens.txt) into <50 MB git-committable chunks.")
        parser.add_argument("--model-dir", required=True)
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--name", default="vietasr-zipformer-vi")
        parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        args = parser.parse_args()
        if args.chunk_size > 50 * 1024 * 1024:
            parser.error("chunk-size must stay under 50 MB")
        return args

    def run(self) -> None:
        args = self.parse()
        model_dir = Path(args.model_dir).resolve()
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        locator = ModelFileLocator(model_dir)
        splitter = ModelSplitter(out_dir, args.chunk_size)
        splitter.clean()
        files = [splitter.split(role, locator.find(role)) for role in ROLES]
        tokens = (model_dir / "tokens.txt").read_bytes()
        (out_dir / "tokens.txt").write_bytes(tokens)
        path = ManifestWriter(out_dir).write(args.name, files, args.chunk_size)
        total = sum(int(f["total_size"]) for f in files)
        print(f"[done] {total / 1024 / 1024:.1f} MB across {len(files)} files -> {path}")


if __name__ == "__main__":
    SplitCommand().run()
