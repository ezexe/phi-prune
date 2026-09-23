"""
Export pruned models to deployment formats.

Supports:
- PyTorch checkpoint (with masks and encoding metadata)
- ONNX (with sparse weight representation)
- Raw bitstream (self-delimiting Fibonacci-encoded weights)
"""

import torch
import json
import os
from typing import Dict, Optional

from zeckendorf_prune.encoding import FibonacciEncoder
from zeckendorf_prune.masks import mask_stats


def save_checkpoint(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    path: str,
    encoder: Optional[FibonacciEncoder] = None,
    scales: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
):
    """
    Save a pruned model as a Zeckendorf checkpoint.

    Includes masks, encoding info, and metadata in a single file.
    Load with: data = torch.load(path); model.load_state_dict(data['state_dict'])
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "masks": masks,
        "format": "zeckendorf-prune",
        "version": "0.2.1",
    }

    if encoder is not None:
        checkpoint["encoding"] = {
            "n_digits": encoder.n_digits,
            "n_levels": encoder.n_levels,
            "max_value": encoder.max_value,
        }

    if scales is not None:
        # Convert to serializable format
        checkpoint["scales"] = {
            name: {"scale": s, "offset": o} for name, (s, o) in scales.items()
        }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Compute and attach stats
    stats = {}
    for name, mask in masks.items():
        stats[name] = mask_stats(mask)
    checkpoint["prune_stats"] = stats

    torch.save(checkpoint, path)
    return path


def export_onnx(
    model: torch.nn.Module,
    path: str,
    input_shape: tuple = (1, 3, 32, 32),
    opset: int = 17,
):
    """
    Export pruned model to ONNX format.

    Zeroed weights are preserved in the ONNX graph — the sparsity
    pattern is baked into the weight values. Compatible with any
    ONNX runtime.

    For sparse-aware runtimes, the mask can be extracted post-export
    by scanning for zero weights.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("pip install onnx onnxruntime")

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model, dummy, path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return path


def export_bitstream(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    encoder: FibonacciEncoder,
    scales: Dict,
    path: str,
):
    """
    Export weights as a self-delimiting Fibonacci bitstream.

    Output format:
    1. JSON header: layer names, shapes, scale/offset per layer
    2. Binary payload: concatenated self-delimiting Fibonacci codewords
       for all active weights, in parameter order

    The bitstream is parseable without knowing tensor shapes —
    each codeword boundary is found by scanning for '11'.
    """
    header = {"layers": [], "encoder": {"n_digits": encoder.n_digits}}
    all_codewords = []

    for name, param in model.named_parameters():
        if name not in masks or name not in scales:
            continue

        mask = masks[name]
        scale, offset = scales[name]
        active = param.data[mask.bool()].cpu().numpy()

        layer_codewords = []
        for val in active:
            grid_val = round((val - offset) * scale)
            grid_val = max(0, min(grid_val, encoder.max_value))
            cw = encoder.to_codeword(int(grid_val))
            layer_codewords.append(cw)

        header["layers"].append({
            "name": name,
            "shape": list(param.shape),
            "n_active": len(layer_codewords),
            "scale": float(scale),
            "offset": float(offset),
        })
        all_codewords.extend(layer_codewords)

    bitstream = encoder.to_bitstream(all_codewords)

    # Write header + payload
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(bitstream)

    bits_per_weight = len(bitstream) / len(all_codewords) if all_codewords else 0
    return {
        "path": path,
        "n_weights": len(all_codewords),
        "bitstream_length": len(bitstream),
        "bits_per_weight": bits_per_weight,
        "header_bytes": len(json.dumps(header)),
    }
