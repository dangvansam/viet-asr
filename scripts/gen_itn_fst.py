#!/usr/bin/env python3
"""Regenerate the Vietnamese ITN grammar FSTs embedded by the `itn` module.

The `itn` module loads two plain OpenFst archives:

    core/src/modules/itn/models/vi_itn_tagger.fst
    core/src/modules/itn/models/vi_itn_verbalizer.fst

They are compiled from FunASR's `fun_text_processing` Vietnamese grammars.
FunASR's own export_models.py writes FAR archives, but the C++ runtime reads a
plain VectorFst, so this script writes the FSTs directly.

Prerequisites
-------------
    python3 -m venv venv && . venv/bin/activate
    pip install pynini regex joblib
    git clone https://github.com/modelscope/FunASR

Usage
-----
    python scripts/gen_itn_fst.py --funasr /path/to/FunASR

The committed FSTs only need regenerating when FunASR's `vi` grammar changes.
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "core", "src", "modules", "itn", "models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--funasr",
        required=True,
        help="path to a FunASR checkout (provides fun_text_processing)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.funasr, "fun_text_processing")):
        sys.exit(f"error: {args.funasr} has no fun_text_processing/ package")
    sys.path.insert(0, os.path.abspath(args.funasr))

    from fun_text_processing.inverse_text_normalization.vi.taggers.tokenize_and_classify import (
        ClassifyFst,
    )
    from fun_text_processing.inverse_text_normalization.vi.verbalizers.verbalize_final import (
        VerbalizeFinalFst,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.perf_counter()
    tagger = ClassifyFst().fst.optimize()
    print(f"tagger:     {tagger.num_states():>6} states "
          f"({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    verbalizer = VerbalizeFinalFst().fst.optimize()
    print(f"verbalizer: {verbalizer.num_states():>6} states "
          f"({time.perf_counter() - t1:.1f}s)")

    tagger_path = os.path.join(args.out_dir, "vi_itn_tagger.fst")
    verbalizer_path = os.path.join(args.out_dir, "vi_itn_verbalizer.fst")
    tagger.write(tagger_path)
    verbalizer.write(verbalizer_path)
    print(f"wrote {tagger_path}")
    print(f"wrote {verbalizer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
