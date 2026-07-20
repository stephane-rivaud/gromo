"""Tests for sqrt(0.3) channel freeze, BN-Inception construction, and M2b FP."""

from __future__ import annotations

import math
import unittest

import torch

from gromo.containers.bn_inception_channels import (
    FULL_INCEPTION_MODULES,
    INCEPTION_WIDTH_SCALE,
    assert_teacher_param_ratio,
    round_inception_channels,
    teacher_inception_modules,
)
from gromo.containers.growing_bn_inception import GrowingBNInception
from tests.torch_unittest import TorchTestCase


FP_ATOL = 1e-4


class TestSqrt03Freeze(unittest.TestCase):
    def test_rounding_rule_reproducible(self) -> None:
        self.assertAlmostEqual(INCEPTION_WIDTH_SCALE, math.sqrt(0.3))
        # 64 * sqrt(0.3) ≈ 35.054 → 35
        self.assertEqual(round_inception_channels(64), 35)
        # 16 * sqrt(0.3) ≈ 8.764 → 9
        self.assertEqual(round_inception_channels(16), 9)
        self.assertEqual(round_inception_channels(1), 1)

    def test_teacher_tables_scale_all_branches(self) -> None:
        teacher = teacher_inception_modules()
        self.assertEqual(len(teacher), len(FULL_INCEPTION_MODULES))
        for full, thin in zip(FULL_INCEPTION_MODULES, teacher, strict=True):
            self.assertEqual(thin.name, full.name)
            self.assertEqual(thin.n1x1, round_inception_channels(full.n1x1))
            self.assertEqual(thin.n3x3, round_inception_channels(full.n3x3))
            self.assertLess(thin.out_channels, full.out_channels)

    def test_param_ratio_assert_band(self) -> None:
        ratio = assert_teacher_param_ratio(30, 100)
        self.assertAlmostEqual(ratio, 0.3)
        with self.assertRaises(AssertionError):
            assert_teacher_param_ratio(5, 100)


class TestGrowingBNInception(TorchTestCase):
    def test_full_forward_shape_224(self) -> None:
        model = GrowingBNInception(
            input_shape=(3, 224, 224),
            out_features=1000,
            width_preset="full",
            assert_param_ratio=False,
            device=torch.device("cpu"),
        )
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 1000))
        self.assertEqual(model._last_inception_out, 1024)

    def test_teacher_param_ratio_and_forward(self) -> None:
        teacher = GrowingBNInception(
            input_shape=(3, 224, 224),
            out_features=1000,
            width_preset="teacher_sqrt_0_3",
            assert_param_ratio=True,
            device=torch.device("cpu"),
        )
        full = GrowingBNInception(
            input_shape=(3, 224, 224),
            out_features=1000,
            width_preset="full",
            assert_param_ratio=False,
            device=torch.device("cpu"),
        )
        ratio = teacher.inception_parameter_count() / full.inception_parameter_count()
        self.assertGreaterEqual(ratio, 0.22)
        self.assertLessEqual(ratio, 0.38)
        teacher.eval()
        y = teacher(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(y.shape), (1, 1000))
        first = next(m for m in teacher.inceptions if hasattr(m, "concat_offsets"))
        offsets = first.concat_offsets
        self.assertEqual(offsets["branch1"], 0)
        self.assertIn("branch_pool", offsets)


class TestFullGraphNet2Wider(TorchTestCase):
    """M2b: full-graph multi-branch remap with shared g + concat offsets."""

    def test_widen_branch1_module0_to_next_fp(self) -> None:
        torch.manual_seed(0)
        model = GrowingBNInception(
            input_shape=(3, 224, 224),
            out_features=10,
            width_preset="teacher_sqrt_0_3",
            assert_param_ratio=False,
            device=torch.device("cpu"),
        )
        model.train()
        for _ in range(2):
            model(torch.randn(1, 3, 224, 224))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y_before = model(x).clone()

        # Widen first inception branch1 into next module's four entry convs.
        n1 = model.inception_modules()[0].branch1.out_channels
        selected = torch.arange(min(2, n1), dtype=torch.long)
        model.net2wider_widen_to_next(
            module_index=0,
            branch="branch1",
            extension_size=int(selected.numel()),
            selected_indices=selected,
            mode="net2wider",
        )
        model.apply_net2wider_widen_to_next()

        with torch.no_grad():
            y_after = model(x)
        self.assertAllClose(y_before, y_after, atol=FP_ATOL)
        # Source branch grew; next entry in_channels grew by the same amount.
        src = model.inception_modules()[0]
        dst = model.inception_modules()[1]
        self.assertEqual(src.branch1.out_channels, n1 + int(selected.numel()))
        self.assertEqual(dst.branch1.in_channels, src.out_features)


if __name__ == "__main__":
    unittest.main()
