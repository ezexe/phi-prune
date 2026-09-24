"""
Cassini integrity checking.

Detect weight corruption by checking the Zeckendorf adjacency constraint.
A single-bit flip that creates consecutive 1s in any codeword is caught
for free — no parity bits, no CRC, just a scan for '11'.

Detection rate: ~46-48% of random single-bit flips on pretrained ResNets
(10-digit codes), and 41-52% on synthetic weights at 8 or 10 digits,
depending on the weight distribution. Averaged over all grid levels it is
~41-42% and barely moves with codeword width. Only 0 -> 1 flips next to a
set bit are caught. scripts/measure_detection.py reproduces these figures.
"""

import torch
from typing import Dict, Tuple


def adjacency_check(codeword: list) -> bool:
    """
    Check if a binary codeword satisfies the Zeckendorf constraint.

    Args:
        codeword: List of 0s and 1s

    Returns:
        True if no consecutive 1s (valid), False if corrupted
    """
    for i in range(len(codeword) - 1):
        if codeword[i] == 1 and codeword[i + 1] == 1:
            return False
    return True


def check_tensor_integrity(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    encoder,
    scale: float,
    offset: float,
) -> Dict:
    """
    Verify all encoded weights in a tensor pass the adjacency check.

    Decodes each active weight back to its Fibonacci codeword and
    checks for adjacent 1s. Any failure indicates corruption.

    Args:
        tensor: Weight tensor with Fibonacci-encoded values
        mask: Binary pruning mask
        encoder: FibonacciEncoder instance used for encoding
        scale: Scale factor from encoding
        offset: Offset from encoding

    Returns:
        dict with pass_count, fail_count, total, pass_rate
    """
    active = tensor[mask.bool()].cpu().numpy()
    passed = 0
    failed = 0

    for val in active:
        # Map back to grid space
        grid_val = round((val - offset) * scale)
        grid_val = max(0, min(grid_val, encoder.max_value))

        # Get codeword
        cw = encoder.to_codeword(int(grid_val))

        if adjacency_check(cw):
            passed += 1
        else:
            failed += 1

    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
    }


def cassini_check(model, masks: Dict, encoder, scales: Dict) -> Dict:
    """
    Run Cassini integrity check on all masked layers of a model.

    Args:
        model: PyTorch model with Fibonacci-encoded weights
        masks: Dict mapping parameter names to mask tensors
        encoder: FibonacciEncoder instance
        scales: Dict mapping parameter names to (scale, offset) tuples

    Returns:
        dict with per-layer and aggregate results
    """
    results = {}
    total_passed = 0
    total_failed = 0

    for name, param in model.named_parameters():
        if name in masks and name in scales:
            scale, offset = scales[name]
            report = check_tensor_integrity(
                param.data, masks[name], encoder, scale, offset
            )
            results[name] = report
            total_passed += report["passed"]
            total_failed += report["failed"]

    total = total_passed + total_failed
    results["_aggregate"] = {
        "passed": total_passed,
        "failed": total_failed,
        "total": total,
        "pass_rate": total_passed / total if total > 0 else 0,
    }

    return results


def simulate_corruption(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    encoder,
    scale: float,
    offset: float,
    n_flips: int = 1000,
) -> Tuple[int, int]:
    """
    Simulate random single-bit flips and measure detection rate.

    For each trial: pick a random active weight, flip a random bit in
    its Fibonacci codeword, check if the corruption is detected.

    Returns:
        (detected_count, total_flips)
    """
    import random

    active_indices = mask.bool().nonzero(as_tuple=False)
    if len(active_indices) == 0:
        return 0, 0

    detected = 0

    for _ in range(n_flips):
        # Pick random active weight
        idx = tuple(active_indices[random.randrange(len(active_indices))].tolist())
        val = tensor[idx].item()

        # Map to grid and get codeword
        grid_val = round((val - offset) * scale)
        grid_val = max(0, min(grid_val, encoder.max_value))
        cw = encoder.to_codeword(int(grid_val))

        # Flip a random bit
        bit_pos = random.randrange(len(cw))
        cw[bit_pos] = 1 - cw[bit_pos]

        # Check if detectable
        if not adjacency_check(cw):
            detected += 1

    return detected, n_flips
