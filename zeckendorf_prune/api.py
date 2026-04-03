"""
High-level API for Zeckendorf pruning.

    from zeckendorf_prune import prune, finetune, check

    model, masks = prune(model, density=0.5)
    finetune(model, train_loader, epochs=40, masks=masks)
    report = check(model, masks)
"""

import copy
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

from zeckendorf_prune.masks import zeckendorf_mask, mask_stats


# ───────────────────────────────────────────────────────────
# PRUNING
# ───────────────────────────────────────────────────────────

def prune(
    model: nn.Module,
    density: float = 0.5,
    axis: int = 0,
    layer_types: Tuple = (nn.Conv2d, nn.Linear),
    min_dim: int = 8,
    inplace: bool = False,
    score_fn: str = "magnitude",
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Apply Zeckendorf-constrained pruning to a model.

    Generates an adjacency-constrained mask for each eligible layer
    and zeros the pruned weights.

    Args:
        model: Trained PyTorch model
        density: Target density (informational — actual density is
                 determined by the DP algorithm, always ≤0.5)
        axis: Dimension to apply constraint along (0 = output features)
        layer_types: Which layer types to prune
        min_dim: Skip layers smaller than this along the prune axis
        inplace: If False, works on a deep copy
        score_fn: Weight importance metric ("magnitude")

    Returns:
        (pruned_model, masks) where masks maps param names to mask tensors
    """
    if not inplace:
        model = copy.deepcopy(model)

    masks = {}
    stats = {"total_params": 0, "active_params": 0, "pruned_layers": 0}

    for name, module in model.named_modules():
        is_target = isinstance(module, layer_types)
        if not is_target:
            continue

        for pname, param in module.named_parameters(prefix=name):
            if "weight" not in pname:
                continue
            if param.shape[axis] < min_dim:
                stats["total_params"] += param.numel()
                stats["active_params"] += param.numel()
                continue

            mask = zeckendorf_mask(param.data, axis=axis, score_fn=score_fn)
            masks[pname] = mask

            # Apply mask
            param.data.mul_(mask)

            stats["total_params"] += param.numel()
            stats["active_params"] += mask.sum().item()
            stats["pruned_layers"] += 1

    # Attach stats
    stats["density"] = stats["active_params"] / stats["total_params"] if stats["total_params"] > 0 else 0
    stats["sparsity"] = 1.0 - stats["density"]
    model._zeck_prune_stats = stats

    return model, masks


# ───────────────────────────────────────────────────────────
# FINE-TUNING
# ───────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_loader,
    epochs: int = 40,
    masks: Dict[str, torch.Tensor] = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    val_loader=None,
    verbose: bool = True,
    criterion: nn.Module = None,
) -> Dict:
    """
    Fine-tune a pruned model with mask enforcement.

    Zeros gradients on pruned positions after each backward pass,
    ensuring the sparsity pattern is maintained throughout training.

    Args:
        model: Pruned model
        train_loader: Training data
        epochs: Fine-tuning epochs
        masks: Dict of pruning masks (from prune())
        lr: Learning rate
        device: Compute device
        val_loader: Optional validation loader for tracking accuracy
        verbose: Print progress
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        dict with training history and best accuracy
    """
    if device is None:
        device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Enforce masks: zero gradients on pruned positions
            if masks:
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name].to(param.device))

            optimizer.step()

            # Re-apply masks to weights (belt and suspenders)
            if masks:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks:
                            param.mul_(masks[name].to(param.device))

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        # Validation
        val_acc = None
        if val_loader is not None:
            val_acc = _evaluate(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            val_str = f" val={val_acc:.1f}%" if val_acc is not None else ""
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} "
                  f"train={train_acc:.1f}%{val_str}")

    # Restore best checkpoint if we tracked validation
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_acc": best_acc if val_loader else None,
        "final_train_acc": history[-1]["train_acc"],
    }


def _evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return 100.0 * correct / total


# ───────────────────────────────────────────────────────────
# INTEGRITY CHECK
# ───────────────────────────────────────────────────────────

def check(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> Dict:
    """
    Verify pruned model integrity.

    Checks:
    1. All masks satisfy the adjacency constraint
    2. All pruned positions are actually zero
    3. Density statistics

    Args:
        model: Pruned model
        masks: Dict of pruning masks

    Returns:
        dict with per-layer and aggregate results
    """
    results = {}
    all_valid = True
    all_zeros_enforced = True

    for name, param in model.named_parameters():
        if name not in masks:
            continue

        mask = masks[name].to(param.device)
        ms = mask_stats(mask)

        # Check pruned positions are zero
        pruned_vals = param.data[~mask.bool()]
        zeros_ok = (pruned_vals.abs() < 1e-10).all().item()

        results[name] = {
            **ms,
            "zeros_enforced": zeros_ok,
        }

        if not ms["valid"]:
            all_valid = False
        if not zeros_ok:
            all_zeros_enforced = False

    results["_summary"] = {
        "all_masks_valid": all_valid,
        "all_zeros_enforced": all_zeros_enforced,
        "n_layers": len(results) - 1,
    }

    return results
