"""Tests for zeckendorf-prune core functionality."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from zeckendorf_prune.masks import zeckendorf_dp, zeckendorf_mask, verify_mask, mask_stats
from zeckendorf_prune.encoding import FibonacciEncoder
from zeckendorf_prune.integrity import adjacency_check, simulate_corruption
from zeckendorf_prune.api import prune, finetune, check


# ── Mask DP ──

class TestZeckendorfDP:
    def test_no_adjacent_ones(self):
        """Core invariant: output never has consecutive 1s."""
        for _ in range(100):
            scores = np.random.rand(np.random.randint(1, 200))
            mask = zeckendorf_dp(scores)
            assert verify_mask(mask), f"Adjacent 1s found in mask for n={len(scores)}"

    def test_density_at_most_half(self):
        """Density never exceeds 50%."""
        for n in [1, 2, 3, 10, 50, 100, 199]:
            scores = np.ones(n)
            mask = zeckendorf_dp(scores)
            assert mask.sum() <= (n + 1) // 2

    def test_optimal_uniform(self):
        """On uniform scores, DP should pick alternating pattern (maximum independent set)."""
        scores = np.ones(10)
        mask = zeckendorf_dp(scores)
        assert mask.sum() == 5  # ⌊10/2⌋ = 5

    def test_prefers_high_scores(self):
        """DP should keep higher-scored positions."""
        scores = np.array([0.1, 10.0, 0.1, 10.0, 0.1])
        mask = zeckendorf_dp(scores)
        assert mask[1] == 1 and mask[3] == 1

    def test_empty(self):
        mask = zeckendorf_dp(np.array([]))
        assert len(mask) == 0

    def test_single(self):
        mask = zeckendorf_dp(np.array([5.0]))
        assert mask[0] == 1


# ── Mask on tensors ──

class TestZeckendorfMask:
    def test_conv2d_shape(self):
        w = torch.randn(16, 8, 3, 3)
        mask = zeckendorf_mask(w, axis=0)
        assert mask.shape == w.shape

    def test_linear_shape(self):
        w = torch.randn(64, 32)
        mask = zeckendorf_mask(w, axis=0)
        assert mask.shape == w.shape

    def test_broadcast_consistency(self):
        """All elements along non-pruned dims should share the same mask value."""
        w = torch.randn(16, 8, 3, 3)
        mask = zeckendorf_mask(w, axis=0)
        for i in range(16):
            vals = mask[i].unique()
            assert len(vals) == 1  # all 0 or all 1


# ── Fibonacci Encoding ──

class TestFibonacciEncoder:
    def test_grid_size(self):
        enc = FibonacciEncoder(n_digits=8)
        assert enc.n_levels == 55  # F(10) = 55 for 8 digits

    def test_max_value_is_representable(self):
        """max_value must be achievable without adjacent 1s."""
        enc = FibonacciEncoder(n_digits=8)
        cw = enc.to_codeword(enc.max_value)
        assert adjacency_check(cw), f"max_value {enc.max_value} has adjacent 1s"

    def test_no_adjacent_ones_in_grid(self):
        """Every grid value must produce a valid codeword."""
        enc = FibonacciEncoder(n_digits=8)
        for val in enc.grid:
            cw = enc.to_codeword(int(val))
            assert adjacency_check(cw), f"Adjacent 1s in codeword for grid value {int(val)}"

    def test_encode_decode_roundtrip(self):
        enc = FibonacciEncoder(n_digits=8)
        t = torch.randn(32)
        encoded, scale, rmse = enc.encode_tensor(t)
        assert encoded.shape == t.shape
        assert rmse < 1.0  # quantization error scales with value range

    def test_stream_roundtrip(self):
        """Encode values to bitstream and decode back."""
        enc = FibonacciEncoder(n_digits=8)
        values = [0, 1, 5, 12, 33]
        stream = enc.encode_to_stream(values)
        recovered = enc.decode_from_stream(stream)
        assert recovered == values

    def test_stream_all_grid_values(self):
        """Every grid value survives a stream roundtrip."""
        enc = FibonacciEncoder(n_digits=8)
        values = [int(v) for v in enc.grid]
        stream = enc.encode_to_stream(values)
        recovered = enc.decode_from_stream(stream)
        assert recovered == values


# ── Integrity ──

class TestIntegrity:
    def test_clean_codewords_pass(self):
        enc = FibonacciEncoder(n_digits=8)
        for val in enc.grid:
            cw = enc.to_codeword(int(val))
            assert adjacency_check(cw)

    def test_corrupted_detected(self):
        """Flipping a bit next to a 1 should be caught."""
        cw = [1, 0, 1, 0, 1, 0, 0, 0]  # valid
        assert adjacency_check(cw)
        cw_corrupt = [1, 1, 1, 0, 1, 0, 0, 0]  # bit 1 flipped
        assert not adjacency_check(cw_corrupt)


# ── High-level API ──

class TestAPI:
    def _make_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def test_prune_returns_masks(self):
        model = self._make_model()
        pruned, masks = prune(model)
        assert len(masks) > 0
        for name, mask in masks.items():
            assert mask.shape == dict(pruned.named_parameters())[name].shape

    def test_pruned_weights_are_zero(self):
        model = self._make_model()
        pruned, masks = prune(model)
        for name, param in pruned.named_parameters():
            if name in masks:
                pruned_vals = param.data[~masks[name].bool()]
                assert (pruned_vals.abs() < 1e-10).all()

    def test_check_passes(self):
        model = self._make_model()
        pruned, masks = prune(model)
        report = check(pruned, masks)
        assert report["_summary"]["all_masks_valid"]
        assert report["_summary"]["all_zeros_enforced"]

    def test_density_under_half(self):
        model = self._make_model()
        pruned, masks = prune(model, prune_head=True)  # every eligible layer pruned, head included
        assert pruned._zeck_prune_stats["density"] <= 0.51  # ≤50% + float tolerance

    def test_head_left_dense_by_default(self):
        """The last eligible layer (the 10-class head here) keeps every row."""
        model = self._make_model()
        pruned, masks = prune(model)
        assert "6.weight" not in masks
        assert torch.equal(pruned[6].weight, model[6].weight)

    def test_prune_head_opt_in(self):
        model = self._make_model()
        pruned, masks = prune(model, prune_head=True)
        assert "6.weight" in masks

    def test_conv_only_layer_types_prunes_every_conv(self):
        """layer_types=(nn.Conv2d,) already leaves the Linear head out; every conv still gets pruned."""
        model = self._make_model()
        pruned, masks = prune(model, layer_types=(nn.Conv2d,))
        assert set(masks) == {"0.weight", "2.weight"}

    def test_exclude_keeps_named_layers_dense(self):
        model = self._make_model()
        pruned, masks = prune(model, exclude={"0"})
        assert "0.weight" not in masks
        assert "2.weight" in masks

    def test_finetune_keeps_pruned_weights_zero(self):
        """amp=True falls back to fp32 off CUDA, so both settings run on any machine."""
        model = self._make_model()
        pruned, masks = prune(model)
        batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(2)]
        for amp in (False, True):
            result = finetune(pruned, batches, epochs=1, masks=masks, val_loader=batches,
                              verbose=False, amp=amp)
            assert set(result) == {"history", "best_val_acc", "final_train_acc"}
            assert check(pruned, masks)["_summary"]["all_zeros_enforced"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast runs on CUDA only")
    def test_finetune_amp_on_cuda(self):
        model = self._make_model().cuda()
        pruned, masks = prune(model)
        batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(2)]
        result = finetune(pruned, batches, epochs=1, masks=masks, val_loader=batches,
                          verbose=False, amp=True)
        assert result["best_val_acc"] is not None
        assert check(pruned, masks)["_summary"]["all_zeros_enforced"]
