"""
Provide unittest class for code using torch.

 - assertShapeEqual: check the shape of a torch tensor
 - assertAllClose: check that two torch tensors are equal up to a tolerance
"""

from unittest import TestCase

import torch


class TorchTestCase(TestCase):
    def assertShapeEqual(
        self, t: torch.Tensor, shape: tuple[int, ...], message: str = ""
    ):
        """
        Check the shape of a torch tensor is equal to the expected shape

        Parameters
        ----------
        t: torch.Tensor
            tensor to check
        shape: tuple
            expected shape, if a dimension is not tested set it to -1
        message: str
            message to display if the test fails
        """
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(
            t.dim(),
            len(shape),
            f"Error: {t.dim()=} should be {len(shape)=}\n" f"{message}",
        )
        for i, s in enumerate(shape):
            if s >= 0:
                self.assertEqual(
                    t.size(i),
                    s,
                    f"Incorrect shape for dim {i}: got {t.size(i)} should be {s=}\n"
                    f"{message}",
                )

    def assertAllClose(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        message: str = "",
    ):
        """
        Check that two torch tensors are close.

        Parameters
        ----------
        a: torch.Tensor
              first tensor
        b: torch.Tensor
              second tensor
        atol: float
              absolute tolerance
        rtol: float
              relative tolerance
        message: str
              message to display if the test fails
        """
        self.assertIsInstance(a, torch.Tensor)
        self.assertIsInstance(b, torch.Tensor)
        self.assertEqual(
            a.shape,
            b.shape,
            f"Error: tensors have different shapes {a.shape=} {b.shape=}\n" f"{message}",
        )
        all_close = torch.allclose(a, b, atol=atol, rtol=rtol)
        max_diff = 0
        error_percentage = 0
        norm_delta_relative_to_a = 0
        if not all_close:
            d = torch.abs(a - b)
            max_diff = torch.max(d)
            error_percentage = 100 * torch.sum(d > atol) / torch.numel(d)
            norm_delta_relative_to_a = torch.norm(d) / torch.norm(a)
        self.assertTrue(
            all_close,
            f"Error: tensors are not close with atol={atol} and rtol={rtol}\n"
            f"Max difference: {max_diff:.2e}\n"
            f"Error percentage: {error_percentage:.2f}%\n"
            f"Norm delta relative to a: {norm_delta_relative_to_a:.2e}\n"
            f"{message}",
        )
