"""
zeckendorf-prune: Structured sparsity via the adjacency constraint.

    from zeckendorf_prune import prune, finetune, check
    from zeckendorf_prune.export import export_onnx

    # Prune a model
    pruned_model, masks = prune(model, density=0.5)

    # Fine-tune with mask enforcement
    finetune(pruned_model, train_loader, epochs=40, masks=masks)

    # Verify integrity
    report = check(pruned_model, masks)

    # Export to ONNX (pruned weights stay as zeros in the graph)
    export_onnx(pruned_model, "model_sparse.onnx")
"""

__version__ = "0.2.3"

from zeckendorf_prune.api import prune, finetune, check, evaluate
from zeckendorf_prune.encoding import FibonacciEncoder
from zeckendorf_prune.integrity import cassini_check
from zeckendorf_prune.masks import zeckendorf_mask, verify_mask

__all__ = [
    "prune",
    "finetune",
    "check",
    "evaluate",
    "FibonacciEncoder",
    "cassini_check",
    "zeckendorf_mask",
    "verify_mask",
]
