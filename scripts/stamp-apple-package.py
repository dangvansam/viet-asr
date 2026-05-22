#!/usr/bin/env python3
"""Stamp the GitHub Release URL + checksum into bindings/apple/Package.swift.

The Swift package distributes CVietASR.xcframework as a binary target. A local
checkout links artifacts/CVietASR.xcframework directly; a released version tag
links the GitHub Release asset instead, which needs a concrete url + checksum.
This script writes those into the manifest's release-mode binary target.

Usage:
    stamp-apple-package.py <version> <checksum>

    version   release version without the leading 'v' (e.g. 0.2.0 or 0.2.0-dev.7)
    checksum  output of `swift package compute-checksum CVietASR.xcframework.zip`
"""
import pathlib
import re
import sys

REPO = "dangvansam/viet-asr"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    version, checksum = sys.argv[1], sys.argv[2].strip()

    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        sys.exit(f"error: checksum must be 64 hex chars, got: {checksum!r}")

    manifest = (pathlib.Path(__file__).resolve().parent.parent
                / "bindings" / "apple" / "Package.swift")
    text = manifest.read_text()

    url = (f"https://github.com/{REPO}/releases/download/"
           f"v{version}/CVietASR.xcframework.zip")

    text, n_url = re.subn(r'url: "[^"]*CVietASR\.xcframework\.zip"',
                          f'url: "{url}"', text)
    text, n_sum = re.subn(r'checksum: "[0-9a-f]*"',
                          f'checksum: "{checksum}"', text)

    if n_url != 1 or n_sum != 1:
        sys.exit(f"error: expected one url + one checksum line, "
                 f"matched url={n_url} checksum={n_sum}")

    manifest.write_text(text)
    print(f"stamped {manifest.name}: v{version}, checksum {checksum[:12]}…")


if __name__ == "__main__":
    main()
