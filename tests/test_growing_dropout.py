"""
Unit tests for GrowingDropout variants.

Each test runs once per dropout class, as a subTest, on an input of the shape
that class expects.
"""

import torch

from gromo.modules.growing_dropout import (
    GrowingDropout,
    GrowingDropout1d,
    GrowingDropout2d,
)
from gromo.modules.growing_module import SupportsExtendedForward
from tests.torch_unittest import TorchTestCase


BATCH_SIZE = 8
EXTENSION_SIZE = 4

# Dropout class, and the shape of its input for a number of channels / features.
DROPOUT_CASES = (
    (GrowingDropout, lambda size: (BATCH_SIZE, 5, size)),  # element-wise, sequence
    (GrowingDropout1d, lambda size: (BATCH_SIZE, size, 10)),
    (GrowingDropout2d, lambda size: (BATCH_SIZE, size, 16, 16)),
)


class TestGrowingDropout(TorchTestCase):
    """Test cases for GrowingDropout, GrowingDropout1d and GrowingDropout2d."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = 32

    def test_initialization(self):
        for dropout_class, _ in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class(dropout_rate=0.5, name="test_dropout")
                self.assertEqual(dropout.p, 0.5)
                self.assertEqual(dropout.name, "test_dropout")
                self.assertFalse(dropout.inplace)

    def test_default_initialization(self):
        for dropout_class, _ in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class()
                self.assertEqual(dropout.p, 0.0)
                self.assertEqual(dropout.name, "growing_dropout")
                self.assertFalse(dropout.inplace)

    def test_forward_pass(self):
        for dropout_class, input_shape in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class(dropout_rate=0.5)
                x = torch.randn(input_shape(self.features), device=self.device)
                self.assertEqual(dropout(x).shape, x.shape)

    def test_extended_forward(self):
        """extended_forward applies dropout to x and passes x_ext unchanged."""
        for dropout_class, input_shape in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class(dropout_rate=0.5)
                x = torch.randn(input_shape(self.features), device=self.device)
                x_ext = torch.randn(input_shape(EXTENSION_SIZE), device=self.device)

                self.assertIsInstance(dropout, SupportsExtendedForward)

                processed_x, processed_x_ext = dropout.extended_forward(x, x_ext)

                self.assertEqual(processed_x.shape, x.shape)
                self.assertEqual(processed_x_ext.shape, x_ext.shape)
                self.assertIs(processed_x_ext, x_ext)

                # a None main tensor propagates as None
                processed_none, processed_x_ext = dropout.extended_forward(None, x_ext)
                self.assertIsNone(processed_none)
                self.assertIs(processed_x_ext, x_ext)

    def test_extra_repr(self):
        for dropout_class, _ in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                repr_str = dropout_class(name="test_dropout").extra_repr()
                self.assertIsInstance(repr_str, str)
                self.assertIn("test_dropout", repr_str)

    def test_zero_dropout(self):
        """A dropout rate of 0 leaves the input unchanged."""
        for dropout_class, input_shape in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class(dropout_rate=0.0)
                x = torch.randn(input_shape(self.features), device=self.device)
                output = dropout(x)
                self.assertEqual(output.shape, x.shape)
                torch.testing.assert_close(output, x)

                x_ext = torch.randn(input_shape(EXTENSION_SIZE), device=self.device)
                kept, forwarded_ext = dropout.extended_forward(x, x_ext)
                torch.testing.assert_close(kept, x)
                self.assertIs(forwarded_ext, x_ext)

    def test_full_dropout(self):
        """A dropout rate of 1 zeroes the input."""
        for dropout_class, input_shape in DROPOUT_CASES:
            with self.subTest(dropout=dropout_class.__name__):
                dropout = dropout_class(dropout_rate=1.0)
                x = torch.randn(input_shape(self.features), device=self.device)
                output = dropout(x)
                self.assertEqual(output.shape, x.shape)
                torch.testing.assert_close(output, torch.zeros_like(x))
