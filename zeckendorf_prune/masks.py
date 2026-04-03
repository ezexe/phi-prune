"""
Zeckendorf-constrained mask generation.

The adjacency constraint: no two consecutive positions may both be active.
This is the same (1,∞)-RLL constraint that governs Fibonacci coding,
CSD arithmetic, and the golden mean shift. Capacity: log₂(φ) ≈ 0.694
bits per position, guaranteeing ≤50% density.

Mask selection uses dynamic programming to find the maximum-weight
independent set under the adjacency constraint — the optimal subset
of weights to keep, given their magnitudes.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def zeckendorf_dp(scores: np.ndarray) -> np.ndarray:
    """
    Find the maximum-weight independent set with no two adjacent elements.

    Dynamic programming over the 1D score array. At each position i:
      keep[i] = skip[i-1] + scores[i]   (take this one, must have skipped previous)
      skip[i] = max(keep[i-1], skip[i-1]) (skip this one, best of either previous)

    Args:
        scores: 1D array of non-negative importance scores (e.g. weight magnitudes)

    Returns:
        Binary mask array, same length as scores. 1 = keep, 0 = prune.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([1], dtype=np.int32)

    # Forward pass: compute optimal values
    keep = np.zeros(n, dtype=np.float64)
    skip = np.zeros(n, dtype=np.float64)

    keep[0] = scores[0]
    skip[0] = 0.0

    for i in range(1, n):
        keep[i] = skip[i - 1] + scores[i]
        skip[i] = max(keep[i - 1], skip[i - 1])

    # Backward pass: reconstruct the mask
    mask = np.zeros(n, dtype=np.int32)
    i = n - 1
    while i >= 0:
        if i == 0:
            if keep[0] >= skip[0]:
                mask[0] = 1
            break
        if keep[i] >= skip[i]:
            mask[i] = 1
            i -= 2  # must skip the previous
        else:
            i -= 1

    return mask


def zeckendorf_mask(
    weight: torch.Tensor,
    axis: int = 0,
    score_fn: str = "magnitude",
) -> torch.Tensor:
    """
    Generate a Zeckendorf-constrained pruning mask for a weight tensor.

    The mask enforces: no two adjacent positions along `axis` are both active.
    Active positions are chosen to maximize total importance (weight magnitude
    by default).

    For Conv2d weights (shape: [out_c, in_c, kH, kW]), axis=0 constrains
    along output channels. For Linear weights (shape: [out, in]), axis=0
    constrains along output features.

    The mask is broadcast across all other dimensions — if output channel i
    is pruned, all its weights are zeroed.

    Args:
        weight: Parameter tensor (2D or 4D)
        axis: Dimension to apply the adjacency constraint along
        score_fn: Importance scoring method. Currently: "magnitude"

    Returns:
        Binary mask tensor, same shape as weight. Ready to multiply.
    """
    # Compute per-position importance scores along the pruning axis
    if score_fn == "magnitude":
        # Sum of absolute values across all other dimensions
        dims_to_reduce = [d for d in range(weight.dim()) if d != axis]
        scores = weight.abs().sum(dim=dims_to_reduce).cpu().numpy()
    else:
        raise ValueError(f"Unknown score function: {score_fn}")

    # Run DP to find optimal mask
    mask_1d = zeckendorf_dp(scores)

    # Broadcast mask to full tensor shape
    shape = [1] * weight.dim()
    shape[axis] = weight.shape[axis]
    mask = torch.tensor(mask_1d, dtype=weight.dtype, device=weight.device)
    mask = mask.reshape(shape).expand_as(weight)

    return mask


def verify_mask(mask_1d) -> bool:
    """
    Verify that a 1D binary pattern satisfies the Zeckendorf constraint:
    no two consecutive 1s.

    Args:
        mask_1d: Iterable of 0s and 1s

    Returns:
        True if valid (no adjacent 1s), False otherwise
    """
    prev = 0
    for val in mask_1d:
        if val == 1 and prev == 1:
            return False
        prev = val
    return True


def mask_stats(mask: torch.Tensor, axis: int = 0) -> dict:
    """
    Compute statistics for a Zeckendorf mask.

    Returns:
        dict with density, sparsity, active_count, total_count,
        capacity_utilization (actual density / theoretical max of 0.5)
    """
    # Extract 1D pattern along axis
    idx = [0] * mask.dim()
    idx[axis] = slice(None)
    pattern = mask[tuple(idx)]
    while pattern.dim() > 1:
        pattern = pattern[0]

    active = pattern.sum().item()
    total = pattern.numel()
    density = active / total if total > 0 else 0

    return {
        "density": density,
        "sparsity": 1.0 - density,
        "active_count": int(active),
        "total_count": total,
        "capacity_utilization": density / 0.5 if density > 0 else 0,
        "valid": verify_mask(pattern.cpu().numpy()),
    }
