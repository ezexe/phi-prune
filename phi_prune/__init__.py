"""
phi-prune: Structured sparsity via the adjacency constraint.

    from phi_prune import prune, finetune, check
    from phi_prune.export import export_onnx

    # Prune a model
    pruned_model, masks = prune(model, density=0.5)

    # Fine-tune with mask enforcement
    finetune(pruned_model, train_loader, epochs=40, masks=masks)

    # Verify integrity
    report = check(pruned_model, masks)

    # Export to ONNX (pruned weights stay as zeros in the graph)
    export_onnx(pruned_model, "model_sparse.onnx")
"""

__version__ = "0.3.0"

from phi_prune.api import prune, finetune, check, evaluate
from phi_prune.encoding import FibonacciEncoder
from phi_prune.integrity import cassini_check
from phi_prune.masks import phi_mask, verify_mask

__all__ = [
    "prune",
    "finetune",
    "check",
    "evaluate",
    "FibonacciEncoder",
    "cassini_check",
    "phi_mask",
    "verify_mask",
]
