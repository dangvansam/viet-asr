#!/usr/bin/env python3
"""Stamp a release version into every VietASR binding manifest.

Usage: scripts/stamp-version.py <version>

Run by .github/workflows/release.yml before each publish job so all bindings
ship the same version as the git tag. The Python binding keeps its own stamper
in pypi-publish.yml and is intentionally not touched here.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stamp(rel_path: str, pattern: str, version: str, label: str) -> None:
    path = ROOT / rel_path
    text = path.read_text()
    new_text, n = re.subn(
        pattern, lambda m: m.group(1) + version + m.group(2), text, count=1
    )
    if n != 1:
        raise SystemExit(f"stamp-version: no {label} match in {rel_path}")
    path.write_text(new_text)
    print(f"{rel_path}: {label} -> {version}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stamp-version.py <version>")
    version = sys.argv[1].lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+([.-][0-9A-Za-z.]+)?", version):
        raise SystemExit(f"stamp-version: invalid version {version!r}")

    npm_pat = r'("version"\s*:\s*")[^"]+(")'
    stamp("bindings/nodejs/package.json", npm_pat, version, "npm version")
    stamp("bindings/webjs/package.json", npm_pat, version, "npm version")

    stamp(
        "bindings/rust/Cargo.toml",
        r'(?s)(\[package\].*?\nversion\s*=\s*")[^"]+(")',
        version,
        "cargo version",
    )
    stamp(
        "bindings/java/pom.xml",
        r"(<artifactId>viet-asr</artifactId>\s*<version>)[^<]+(</version>)",
        version,
        "pom version",
    )
    stamp(
        "bindings/csharp/Vietasr/Vietasr.csproj",
        r"(<Version>)[^<]+(</Version>)",
        version,
        "csproj version",
    )
    stamp(
        "bindings/android/lib/build.gradle.kts",
        r'(version\s*=\s*")[^"]+(")',
        version,
        "gradle version",
    )


if __name__ == "__main__":
    main()
