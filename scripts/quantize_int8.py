import argparse
import tempfile
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--weight-type", choices=["int8", "uint8"], default="int8")
    parser.add_argument("--skip-ops", nargs="*", default=["Conv"],
                        help="Op types to exclude from quantization "
                             "(default: Conv, due to pre-existing QDQ wrappers in the model)")
    args = parser.parse_args()

    weight_type = QuantType.QInt8 if args.weight_type == "int8" else QuantType.QUInt8

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path = Path(tmpdir) / "preprocessed.onnx"
        print(f"[quant_pre_process] {src} -> {pre_path}")
        quant_pre_process(
            input_model=str(src),
            output_model_path=str(pre_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            int_max=2**31 - 1,
            guess_output_rank=False,
            verbose=0,
            save_as_external_data=False,
            all_tensors_to_one_file=False,
            external_data_location=None,
            external_data_size_threshold=1024,
        )

        print(f"[quantize] {pre_path} -> {dst}")
        print(f"           weight_type={args.weight_type}, per_channel={args.per_channel}, skip={args.skip_ops}")
        all_op_types = {"MatMul", "Gemm", "Conv", "Attention", "EmbedLayerNormalization",
                        "LSTM", "GRU", "RNN"}
        op_types_to_quantize = sorted(all_op_types - set(args.skip_ops))
        print(f"[quantize] op_types_to_quantize={op_types_to_quantize}")

        quantize_dynamic(
            model_input=str(pre_path),
            model_output=str(dst),
            weight_type=weight_type,
            per_channel=args.per_channel,
            reduce_range=False,
            op_types_to_quantize=op_types_to_quantize,
            extra_options={"DefaultTensorType": 1},
            use_external_data_format=False,
        )

    fp_size = src.stat().st_size
    q_size = dst.stat().st_size
    print(f"[size] {fp_size/1024/1024:.1f} MB -> {q_size/1024/1024:.1f} MB"
          f"  ({100*q_size/fp_size:.0f}%)")


if __name__ == "__main__":
    main()
