import argparse
import sys

import onnx
from onnx import AttributeProto, TensorProto, helper


FLOAT_ATTRS_BY_OP = {
    "ThresholdedRelu": {"alpha"},
    "LeakyRelu": {"alpha"},
    "Elu": {"alpha"},
    "Selu": {"alpha", "gamma"},
    "HardSigmoid": {"alpha", "beta"},
}

FLOAT_ONLY_OPS = {"ThresholdedRelu", "Relu", "LeakyRelu", "Elu",
                  "Selu", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
                  "Softplus", "Softsign", "HardSigmoid"}

INT_DTYPES = {TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
              TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64}


def coerce_float_attrs(model: onnx.ModelProto) -> int:
    fixed = 0
    for node in model.graph.node:
        targets = FLOAT_ATTRS_BY_OP.get(node.op_type)
        if not targets:
            continue
        for attr in node.attribute:
            if attr.name not in targets:
                continue
            if attr.type == AttributeProto.FLOAT:
                continue
            if attr.type == AttributeProto.INT:
                replacement = helper.make_attribute(attr.name, float(attr.i))
            elif attr.type == AttributeProto.STRING:
                try:
                    replacement = helper.make_attribute(attr.name, float(attr.s.decode()))
                except Exception:
                    continue
            else:
                continue
            attr.Clear()
            attr.CopyFrom(replacement)
            fixed += 1
            print(f"  fixed {node.op_type}/{attr.name} on node '{node.name}'")
    return fixed


def propagate_dtypes(model: onnx.ModelProto) -> dict:
    dtypes = {}
    for vi in model.graph.input:
        dtypes[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type

    type_promoting = {"Add", "Sub", "Mul", "Div", "Mod", "Min", "Max",
                      "Pow", "Sum", "Mean", "Neg", "Abs", "Sign",
                      "Concat", "Reshape", "Transpose", "Squeeze",
                      "Unsqueeze", "Slice", "Gather", "GatherND", "ScatterND",
                      "Identity", "Pad", "Expand", "Tile", "Where",
                      "ReduceSum", "ReduceMin", "ReduceMax", "ReduceMean"}

    progress = True
    while progress:
        progress = False
        for node in model.graph.node:
            if all(out in dtypes for out in node.output if out):
                continue
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to":
                        for out in node.output:
                            if out and out not in dtypes:
                                dtypes[out] = attr.i
                                progress = True
                continue
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name in ("value", "value_float", "value_floats"):
                        if attr.type == AttributeProto.TENSOR:
                            for out in node.output:
                                if out and out not in dtypes:
                                    dtypes[out] = attr.t.data_type
                                    progress = True
                        elif attr.type in (AttributeProto.FLOAT, AttributeProto.FLOATS):
                            for out in node.output:
                                if out and out not in dtypes:
                                    dtypes[out] = TensorProto.FLOAT
                                    progress = True
                    elif attr.name in ("value_int", "value_ints"):
                        for out in node.output:
                            if out and out not in dtypes:
                                dtypes[out] = TensorProto.INT64
                                progress = True
                continue
            if node.op_type in ("Shape", "Size", "NonZero"):
                for out in node.output:
                    if out and out not in dtypes:
                        dtypes[out] = TensorProto.INT64
                        progress = True
                continue
            if node.op_type in ("ArgMax", "ArgMin"):
                for out in node.output:
                    if out and out not in dtypes:
                        dtypes[out] = TensorProto.INT64
                        progress = True
                continue
            input_dtypes = [dtypes.get(inp) for inp in node.input if inp]
            if any(dt is None for dt in input_dtypes):
                continue
            if node.op_type in type_promoting or input_dtypes:
                first = input_dtypes[0]
                for out in node.output:
                    if out and out not in dtypes:
                        dtypes[out] = first
                        progress = True
    return dtypes


def cast_int_inputs_for_float_only_ops(model: onnx.ModelProto) -> int:
    dtypes = propagate_dtypes(model)
    fixed = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type not in FLOAT_ONLY_OPS:
            new_nodes.append(node)
            continue
        original_int_dt = None
        for idx, inp in enumerate(node.input):
            dt = dtypes.get(inp)
            if dt is None or dt not in INT_DTYPES:
                continue
            original_int_dt = dt
            base = node.name or f"{node.op_type}_{len(new_nodes)}"
            cast_in_name = f"{base}__cast_in_{idx}"
            cast_in = helper.make_node(
                "Cast",
                inputs=[inp],
                outputs=[cast_in_name + "_f"],
                to=TensorProto.FLOAT,
                name=cast_in_name,
            )
            new_nodes.append(cast_in)
            node.input[idx] = cast_in_name + "_f"

        new_nodes.append(node)

        if original_int_dt is not None:
            base = node.name or f"{node.op_type}_{len(new_nodes)}"
            cast_out_name = f"{base}__cast_out"
            original_out = node.output[0]
            relabeled = f"{cast_out_name}_f"
            node.output[0] = relabeled
            cast_out = helper.make_node(
                "Cast",
                inputs=[relabeled],
                outputs=[original_out],
                to=original_int_dt,
                name=cast_out_name,
            )
            new_nodes.append(cast_out)
            fixed += 1
            dtype_name = {TensorProto.INT8: "int8", TensorProto.INT16: "int16",
                          TensorProto.INT32: "int32", TensorProto.INT64: "int64",
                          TensorProto.UINT8: "uint8"}.get(original_int_dt, str(original_int_dt))
            print(f"  wrapped {node.op_type} '{node.name}': {dtype_name} -> float -> {dtype_name}")

    if fixed > 0:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    args = parser.parse_args()

    model = onnx.load(args.src)
    fixed_attrs = coerce_float_attrs(model)
    fixed_casts = cast_int_inputs_for_float_only_ops(model)
    print(f"[patch] fixed {fixed_attrs} attribute(s), inserted {fixed_casts} cast(s)")

    try:
        onnx.checker.check_model(model)
        print("[ok] onnx.checker passed")
    except Exception as exc:
        print(f"[warn] checker reports: {exc}", file=sys.stderr)

    onnx.save(model, args.dst)
    print(f"[saved] {args.dst}")


if __name__ == "__main__":
    main()
