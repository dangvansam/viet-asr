import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    DefaultWeightOnlyQuantConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--is-symmetric", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[matmul-4bit] {src} -> {dst}")
    print(f"             block_size={args.block_size}, symmetric={args.is_symmetric}")

    model = onnx.load(str(src))
    config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=args.is_symmetric,
    )
    quantizer = MatMulNBitsQuantizer(model=model, algo_config=config)
    quantizer.process()
    quantized = quantizer.model.model
    onnx.save_model(quantized, str(dst))

    fp_size = src.stat().st_size
    q_size = dst.stat().st_size
    print(f"[size] {fp_size/1024/1024:.1f} MB -> {q_size/1024/1024:.1f} MB"
          f"  ({100*q_size/fp_size:.0f}%)")


if __name__ == "__main__":
    main()
