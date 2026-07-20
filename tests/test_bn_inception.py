"""Tests for √0.3 channel freeze and GrowingBNInception construction."""

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
        # Concat offsets exist on first inception module
        first = next(m for m in teacher.inceptions if hasattr(m, "concat_offsets"))
        offsets = first.concat_offsets
        self.assertEqual(offsets["branch1"], 0)
        self.assertIn("branch_pool", offsets)


if __name__ == "__main__":
    unittest.main()
