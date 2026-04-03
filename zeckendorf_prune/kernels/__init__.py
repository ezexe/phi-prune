"""
Sparse inference kernels for Zeckendorf-constrained weight matrices.

STATUS: Placeholder. Ship Path B first (convert to CSR, use existing BLAS).
Path A (custom kernels exploiting the adjacency guarantee) comes later.

The adjacency constraint guarantees: between any two nonzero positions,
there is at least one zero. This means:
  - Memory access stride is always ≥2 (prefetch-friendly)
  - No two adjacent MAC operations share a nonzero weight
  - The mask is compressible to 0.694 bits/position (RLL optimal)
"""


def sparse_matmul_csr(weight, mask, input_tensor):
    """
    Path B: Convert Zeckendorf-sparse weight to CSR and use torch.sparse.mm.

    This is the ship-now option — no custom CUDA code, works everywhere.
    Performance is decent but doesn't exploit the adjacency structure.
    """
    import torch

    sparse_w = weight.to_sparse_csr()
    return torch.sparse.mm(sparse_w, input_tensor.t()).t()


# TODO: Path A — custom kernel exploiting adjacency guarantee
# def sparse_matmul_zeck(weight, mask, input_tensor):
#     """
#     Custom kernel: stride-2 minimum access pattern, predictable prefetch.
#     Target: ONNX Runtime custom op, or direct CUDA kernel.
#     """
#     raise NotImplementedError("Path A kernel not yet implemented")
