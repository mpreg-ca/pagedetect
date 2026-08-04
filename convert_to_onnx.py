import argparse

import onnx
import torch

from page_sequence_classifier import PageSequenceClassifier


def load_model(model_path: str, device: torch.device) -> PageSequenceClassifier:
    """Load the PyTorch model from checkpoint (.pth or .safetensors)."""
    model = PageSequenceClassifier(3, 256).to(device)

    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path, device=str(device))
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def convert_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int,
):
    device = torch.device("cpu")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # Create dummy inputs
    batch_size = 1

    # Input dimensions
    edge_h, edge_w = 256, 64
    spread_h, spread_w = 256, 64

    seq_len = PageSequenceClassifier.MAX_SEQUENCE_LENGTH

    dummy_left_edges = torch.randn(batch_size, seq_len, 1, edge_h, edge_w)
    dummy_right_edges = torch.randn(batch_size, seq_len, 1, edge_h, edge_w)
    dummy_spreads = torch.randn(batch_size, seq_len, 1, spread_h, spread_w)
    dummy_ratios = torch.rand(batch_size, seq_len)

    print(f"Exporting to ONNX with opset version {opset_version}...")
    print(f"  - Reference sequence length: {seq_len}")
    print(f"  - Dynamic sequence length: enabled")
    print(f"  - Edge image size: {edge_h}x{edge_w}")
    print(f"  - Spread image size: {spread_h}x{spread_w}")

    torch.onnx.export(
        model,
        (dummy_left_edges, dummy_right_edges, dummy_spreads, dummy_ratios),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["left_edges", "right_edges", "spreads", "ratios"],
        output_names=["logits"],
        dynamic_axes={
            "left_edges": {1: "seq_len"},
            "right_edges": {1: "seq_len"},
            "spreads": {1: "seq_len"},
            "ratios": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        dynamo=False,
    )

    print(f"Model exported to {output_path}")

    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Print model info
    print("\nModel inputs:")
    for inp in onnx_model.graph.input:
        print(
            f"  - {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}"
        )

    print("\nModel outputs:")
    for out in onnx_model.graph.output:
        print(
            f"  - {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PageSequenceClassifier to ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_comic_classifier.safetensors",
        help="Path to the PyTorch model file (.pth or .safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="page_classifier.onnx",
        help="Path for the output ONNX file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()
    convert_to_onnx(args.model, args.output, args.opset)
