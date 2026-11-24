"""
Unit tests for the growing normalisation layers: GrowingBatchNorm2d,
GrowingBatchNorm1d, GrowingLayerNorm and GrowingGroupNorm.

The behaviour the four classes share runs once per class as a subTest of the
tests of ``TestGrowingNormalisation``; the other test cases hold what is
specific to one class.
"""

import dataclasses
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from gromo.modules.growing_module import SupportsExtendedForward
from gromo.modules.growing_normalisation import (
    GrowingBatchNorm1d,
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)
from tests.torch_unittest import TorchTestCase


INITIAL_SIZE = 32
BATCH_SIZE = 4
# Every growth below is a multiple of 4, the number of groups of the GroupNorm case.
GROWTH = 16

# Default value of the new entries of each grown tensor, and the keyword of
# ``grow`` that sets them.
DEFAULT_VALUE_FN = {
    "weight": torch.ones,
    "bias": torch.zeros,
    "running_mean": torch.zeros,
    "running_var": torch.ones,
}
GROW_KWARG = {
    "weight": "new_weights",
    "bias": "new_biases",
    "running_mean": "new_running_mean",
    "running_var": "new_running_var",
}


@dataclasses.dataclass(frozen=True)
class NormCase:
    """How the shared tests build and inspect one growing normalisation class.

    ``size_attr`` is at once the constructor keyword, the attribute and the
    growth-information key of the normalised size.
    """

    label: str
    module_class: type
    size_attr: str
    affine_kwarg: str
    input_shape: Callable[[int, int], tuple[int, ...]]
    fixed_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    init_flags: dict[str, Any] = dataclasses.field(default_factory=dict)
    buffer_names: tuple[str, ...] = ()

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Names of the tensors a growth extends."""
        return ("weight", "bias", *self.buffer_names)

    def make(self, size: int, **kwargs) -> Any:
        """Build a module of normalised size ``size``."""
        return self.module_class(**{self.size_attr: size}, **self.fixed_kwargs, **kwargs)

    def size_value(self, size: int) -> int | tuple[int]:
        """Expected value of the size attribute for a size ``size``."""
        return (size,) if self.size_attr == "normalized_shape" else size

    def size_of(self, module: torch.nn.Module) -> Any:
        """Current value of the size attribute of ``module``."""
        return getattr(module, self.size_attr)


BATCH_NORM_FLAGS = {"momentum": 0.1, "affine": True, "track_running_stats": True}
RUNNING_STATS = ("running_mean", "running_var")
BATCH_NORM_2D = NormCase(
    "BatchNorm2d",
    GrowingBatchNorm2d,
    "num_features",
    "affine",
    lambda batch, size: (batch, size, 6, 6),
    init_flags=BATCH_NORM_FLAGS,
    buffer_names=RUNNING_STATS,
)
BATCH_NORM_1D_SEQUENCE = NormCase(
    "BatchNorm1d (N, C, L)",
    GrowingBatchNorm1d,
    "num_features",
    "affine",
    lambda batch, size: (batch, size, 10),
    init_flags=BATCH_NORM_FLAGS,
    buffer_names=RUNNING_STATS,
)
BATCH_NORM_1D = dataclasses.replace(
    BATCH_NORM_1D_SEQUENCE,
    label="BatchNorm1d (N, C)",
    input_shape=lambda batch, size: (batch, size),
)
LAYER_NORM = NormCase(
    "LayerNorm",
    GrowingLayerNorm,
    "normalized_shape",
    "elementwise_affine",
    lambda batch, size: (batch, 5, size),
    init_flags={"elementwise_affine": True},
)
GROUP_NORM = NormCase(
    "GroupNorm",
    GrowingGroupNorm,
    "num_channels",
    "affine",
    lambda batch, size: (batch, size, 6, 6),
    fixed_kwargs={"num_groups": 4},
    init_flags={"affine": True},
)
NORM_CASES = (
    BATCH_NORM_2D,
    BATCH_NORM_1D_SEQUENCE,
    BATCH_NORM_1D,
    LAYER_NORM,
    GROUP_NORM,
)


class TestGrowingNormalisation(TorchTestCase):
    """Behaviour shared by the four growing normalisation classes."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_input(self, case: NormCase, size: int) -> torch.Tensor:
        return torch.randn(case.input_shape(BATCH_SIZE, size), device=self.device)

    def test_initialization(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(
                    INITIAL_SIZE,
                    eps=1e-5,
                    device=self.device,
                    name="test_norm",
                    **case.init_flags,
                )
                self.assertEqual(case.size_of(module), case.size_value(INITIAL_SIZE))
                self.assertEqual(module.name, "test_norm")
                self.assertEqual(module.eps, 1e-5)
                for key, value in {**case.fixed_kwargs, **case.init_flags}.items():
                    self.assertEqual(getattr(module, key), value)
                for name in case.tensor_names:
                    self.assertEqual(getattr(module, name).shape[0], INITIAL_SIZE)

    def test_initialization_no_affine(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(
                    INITIAL_SIZE, device=self.device, **{case.affine_kwarg: False}
                )
                self.assertIsNone(module.weight)
                self.assertIsNone(module.bias)
                for name in case.buffer_names:
                    self.assertIsNotNone(getattr(module, name))

    def test_forward_pass(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                x = self.make_input(case, INITIAL_SIZE)
                self.assertEqual(module(x).shape, x.shape)
                module.eval()
                with torch.no_grad():
                    self.assertEqual(module(x).shape, x.shape)

    def test_grow_default_parameters(self):
        """The old entries are kept and the new ones take the default values."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                originals = {
                    name: getattr(module, name).detach().clone()
                    for name in case.tensor_names
                }

                module.grow(GROWTH)

                expected_size = INITIAL_SIZE + GROWTH
                self.assertEqual(case.size_of(module), case.size_value(expected_size))
                for name, original in originals.items():
                    grown = getattr(module, name).detach()
                    self.assertEqual(grown.shape[0], expected_size, name)
                    torch.testing.assert_close(grown[:INITIAL_SIZE], original)
                    torch.testing.assert_close(
                        grown[INITIAL_SIZE:],
                        DEFAULT_VALUE_FN[name](GROWTH, device=self.device),
                    )

    def test_grow_custom_parameters(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                additional = 8
                custom_values = {
                    name: torch.full((additional,), value, device=self.device)
                    for name, value in zip(
                        case.tensor_names, (0.5, -0.1, 0.2, 1.5), strict=False
                    )
                }

                module.grow(
                    additional,
                    **{GROW_KWARG[name]: value for name, value in custom_values.items()},
                )

                for name, value in custom_values.items():
                    torch.testing.assert_close(
                        getattr(module, name).detach()[INITIAL_SIZE:], value
                    )

    def test_grow_multiple_times(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                expected_size = INITIAL_SIZE
                for additional in (8, 4, 12):
                    module.grow(additional)
                    expected_size += additional
                    self.assertEqual(case.size_of(module), case.size_value(expected_size))
                for name in case.tensor_names:
                    self.assertEqual(getattr(module, name).shape[0], expected_size)

    def test_forward_after_growth(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                module.grow(GROWTH)
                x = self.make_input(case, INITIAL_SIZE + GROWTH)
                self.assertEqual(module(x).shape, x.shape)

    def test_grow_no_affine(self):
        """Without affine parameters, a growth extends only the statistics."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(
                    INITIAL_SIZE, device=self.device, **{case.affine_kwarg: False}
                )
                module.grow(8)
                self.assertEqual(case.size_of(module), case.size_value(INITIAL_SIZE + 8))
                self.assertIsNone(module.weight)
                self.assertIsNone(module.bias)
                for name in case.buffer_names:
                    self.assertEqual(getattr(module, name).shape[0], INITIAL_SIZE + 8)

    def test_grow_error_cases(self):
        """A non-positive growth, or custom values of the wrong size, raise."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                for additional in (0, -1, -4):
                    with self.assertRaises(ValueError):
                        module.grow(additional)
                for name in case.tensor_names:
                    for wrong_size in (5, 10):
                        # A fresh module: a failed growth may leave it partly grown.
                        module = case.make(INITIAL_SIZE, device=self.device)
                        with self.assertRaises(ValueError, msg=(name, wrong_size)):
                            module.grow(8, **{GROW_KWARG[name]: torch.ones(wrong_size)})

    def test_get_growth_info(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, name="info_norm")
                info = module.get_growth_info()
                self.assertEqual(info[case.size_attr], case.size_value(INITIAL_SIZE))
                self.assertEqual(info["name"], "info_norm")
                for key, value in case.fixed_kwargs.items():
                    self.assertEqual(info[key], value)

                module.grow(GROWTH)
                info = module.get_growth_info()
                self.assertEqual(
                    info[case.size_attr], case.size_value(INITIAL_SIZE + GROWTH)
                )

    def test_extra_repr(self):
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, eps=1e-4, name="repr_norm")
                repr_str = module.extra_repr()
                self.assertIsInstance(repr_str, str)
                self.assertIn("repr_norm", repr_str)

    def test_dtype_preservation(self):
        for case in NORM_CASES:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(norm=case.label, dtype=dtype):
                    module = case.make(INITIAL_SIZE, dtype=dtype)
                    module.grow(8)
                    for name in case.tensor_names:
                        self.assertEqual(getattr(module, name).dtype, dtype, name)

    def test_extend_parameter_none_param(self):
        """_extend_parameter returns early when the attribute is None."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(
                    INITIAL_SIZE, device=self.device, **{case.affine_kwarg: False}
                )
                module._extend_parameter(
                    "weight", 8, None, torch.ones, torch.device("cpu"), as_parameter=True
                )
                self.assertIsNone(module.weight)

    def test_extend_parameter_as_buffer(self):
        """The as_parameter=False path stores the result as a buffer."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                device = module.weight.device
                # A fresh buffer (not an nn.Parameter) so register_buffer succeeds.
                module.register_buffer(
                    "running_stat", torch.zeros(INITIAL_SIZE, device=device)
                )
                module._extend_parameter(
                    "running_stat", 8, None, torch.zeros, device, as_parameter=False
                )
                buffers = dict(module.named_buffers())
                self.assertIn("running_stat", buffers)
                self.assertEqual(buffers["running_stat"].shape[0], INITIAL_SIZE + 8)

    def test_grow_device_inferred_from_weight(self):
        """grow places the new entries on the device of the weight."""
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=torch.device("cpu"))
                module.grow(8)
                self.assertEqual(module.weight.device.type, "cpu")
                self.assertEqual(module.weight.shape[0], INITIAL_SIZE + 8)

    def test_grow_custom_params_device_transfer(self):
        """Custom values on another device are moved to the device of the module."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=torch.device("cuda"))
                module.grow(8)
                self.assertEqual(module.weight.device.type, "cuda")
                cpu_weights = torch.ones(8)  # intentionally on CPU
                module.grow(8, new_weights=cpu_weights)
                self.assertEqual(module.weight.device.type, "cuda")

    def test_extended_forward(self):
        """extended_forward normalises x and passes x_ext unchanged."""
        extension_size = 8
        for case in NORM_CASES:
            with self.subTest(norm=case.label):
                module = case.make(INITIAL_SIZE, device=self.device)
                module.eval()
                x = self.make_input(case, INITIAL_SIZE)
                x_ext = self.make_input(case, extension_size)

                self.assertIsInstance(module, SupportsExtendedForward)

                processed_x, processed_x_ext = module.extended_forward(x, x_ext)

                self.assertEqual(processed_x.shape, x.shape)
                self.assertEqual(processed_x_ext.shape, x_ext.shape)
                torch.testing.assert_close(processed_x, module(x))
                torch.testing.assert_close(processed_x_ext, x_ext)


class TestGrowingBatchNormRunningStats(TorchTestCase):
    """GrowingBatchNorm without running statistics."""

    def test_without_running_stats(self):
        bn = GrowingBatchNorm2d(
            num_features=INITIAL_SIZE, affine=True, track_running_stats=False
        )
        self.assertIsNotNone(bn.weight)
        self.assertIsNotNone(bn.bias)
        self.assertIsNone(bn.running_mean)
        self.assertIsNone(bn.running_var)

        bn.grow(8)

        self.assertIsNone(bn.running_mean)
        self.assertIsNone(bn.running_var)
        self.assertEqual(bn.weight.shape[0], INITIAL_SIZE + 8)
        self.assertEqual(bn.bias.shape[0], INITIAL_SIZE + 8)

    def test_grow_without_affine_or_running_stats(self):
        bn = GrowingBatchNorm2d(
            num_features=INITIAL_SIZE, track_running_stats=False, affine=False
        )
        bn.grow(8)
        self.assertEqual(bn.num_features, INITIAL_SIZE + 8)


class TestGrowingLayerNorm(TorchTestCase):
    """GrowingLayerNorm without bias, and with a multi-dimensional shape."""

    def test_without_bias(self):
        ln = GrowingLayerNorm(
            normalized_shape=INITIAL_SIZE, elementwise_affine=True, bias=False
        )
        self.assertIsNotNone(ln.weight)
        self.assertIsNone(ln.bias)

        ln.grow(8)

        self.assertEqual(ln.normalized_shape, (INITIAL_SIZE + 8,))
        self.assertEqual(ln.weight.shape[0], INITIAL_SIZE + 8)
        self.assertIsNone(ln.bias)

    def test_grow_multi_dim_normalized_shape(self):
        """grow extends the first (channel) dim of a 2-D normalized_shape."""
        H, W = 8, 16
        ln = GrowingLayerNorm(normalized_shape=[H, W])
        additional = 8

        ln.grow(additional)

        self.assertEqual(ln.normalized_shape, (H + additional, W))
        self.assertEqual(ln.weight.shape, (H + additional, W))
        self.assertEqual(ln.bias.shape, (H + additional, W))

        x = torch.randn(BATCH_SIZE, H + additional, W)
        self.assertEqual(ln(x).shape, x.shape)


class TestGrowingGroupNorm(TorchTestCase):
    """The number of groups of GrowingGroupNorm."""

    def setUp(self):
        super().setUp()
        self.gn = GROUP_NORM.make(INITIAL_SIZE)

    def test_grow_with_new_num_groups(self):
        self.gn.grow(8, new_num_groups=8)  # 32 + 8 = 40; 40 % 8 == 0
        self.assertEqual(self.gn.num_groups, 8)
        self.assertEqual(self.gn.num_channels, INITIAL_SIZE + 8)

    def test_grow_error_not_divisible(self):
        with self.assertRaises(ValueError):
            self.gn.grow(3)  # 32 + 3 = 35; 35 % 4 != 0

    def test_grow_error_wrong_ndim_custom_params(self):
        with self.assertRaises(ValueError):
            self.gn.grow(8, new_weights=torch.ones(8, 2))


# Pruning cases: the case, and the gap between two indices that fall in two
# different groups of the GroupNorm (two groups of four channels).
PRUNE_CASES = (
    (BATCH_NORM_1D, 2),
    (LAYER_NORM, 2),
    (dataclasses.replace(GROUP_NORM, fixed_kwargs={"num_groups": 2}), 4),
)


class TestGrowingNormalisationPruning(TorchTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")

    def test_pruning_validation_and_edge_cases(self):
        for case, step in PRUNE_CASES:
            with self.subTest(norm=case.label):
                module = case.make(8, device=self.device)

                # indices_to_remove is list
                module.prune([1, 1 + step])
                self.assertEqual(case.size_of(module), case.size_value(6))

                # invalid type -> TypeError
                with self.assertRaises(TypeError):
                    module.prune("invalid")

                # non-integer dtype -> TypeError
                with self.assertRaises(TypeError):
                    module.prune(np.array([1.0, 2.0]))

                # empty index -> no-op
                module.prune(np.array([], dtype=np.int64))
                self.assertEqual(case.size_of(module), case.size_value(6))

                # out-of-bounds -> IndexError
                with self.assertRaises(IndexError):
                    module.prune([6])
                with self.assertRaises(IndexError):
                    module.prune([-1])

                # pruning all features -> ValueError
                with self.assertRaises(ValueError):
                    module.prune([0, 1, 2, 3, 4, 5])

    def test_group_norm_prune_unequal_groups(self):
        """Groups losing different numbers of channels raise."""
        gn = GrowingGroupNorm(num_groups=2, num_channels=8, device=self.device)
        gn.prune([1, 5])
        with self.assertRaises(ValueError):
            gn.prune([0])
        with self.assertRaises(ValueError):
            gn.prune([0, 1])

    def test_prune_without_affine(self):
        for case, step in PRUNE_CASES:
            with self.subTest(norm=case.label):
                module = case.make(8, device=self.device, **{case.affine_kwarg: False})
                self.assertIsNone(module.weight)
                module.prune([0, step])
                self.assertEqual(case.size_of(module), case.size_value(6))
                self.assertIsNone(module.weight)

    def test_prune_with_bias_none(self):
        for case, step in PRUNE_CASES:
            with self.subTest(norm=case.label):
                module = case.make(8, device=self.device, **{case.affine_kwarg: True})
                module.bias = None
                module.prune([0, step])
                self.assertEqual(case.size_of(module), case.size_value(6))
                self.assertIsNone(module.bias)

    def test_batch_norm_prune_running_stats(self):
        bn_no_stats = GrowingBatchNorm1d(
            num_features=8, track_running_stats=False, device=self.device
        )
        self.assertIsNone(bn_no_stats.running_mean)
        bn_no_stats.prune([0, 2])
        self.assertEqual(bn_no_stats.num_features, 6)
        self.assertIsNone(bn_no_stats.running_mean)

        # running stats are None with track_running_stats=True
        bn_stats_none = GrowingBatchNorm1d(
            num_features=8, track_running_stats=True, device=self.device
        )
        bn_stats_none.running_mean = None
        bn_stats_none.running_var = None
        bn_stats_none.prune([0, 2])
        self.assertEqual(bn_stats_none.num_features, 6)
        self.assertIsNone(bn_stats_none.running_mean)
        self.assertIsNone(bn_stats_none.running_var)

    def test_extend_parameter_mismatches(self):
        for case, _ in PRUNE_CASES:
            with self.subTest(norm=case.label):
                module = case.make(8, device=self.device)
                with self.assertRaises(ValueError):
                    module.grow(4, new_weights=torch.ones(2))

                # a direct call reaches the shape check of _extend_parameter
                with self.assertRaises(ValueError):
                    module._extend_parameter(
                        "weight",
                        4,
                        torch.ones(2),
                        torch.ones,
                        self.device,
                        as_parameter=True,
                    )

                # custom values of another dtype take the dtype of the module
                module = case.make(8, device=self.device)
                module.grow(4, new_weights=torch.ones(4, dtype=torch.float64))
                self.assertEqual(case.size_of(module), case.size_value(12))
                self.assertEqual(module.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
