"""Tests of splitting steepest descent in gromo.lite.

The splitting matrices are compared with brute-force computations (a per-sample
Hessian of the unit for Linear sites, an analytic tanh'' over unfolded patches
for convolutions). The loss change of a split is compared with its prediction
``eps**2 / 2 * sum(lambda_min)`` by finite differences.
"""

import torch
from torch import nn

from gromo.lite import (
    Site,
    activation_second_derivative,
    collect,
    split_copy,
    splitting,
    splitting_moment,
)


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


def summed_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum over samples of the squared error summed over outputs."""
    return ((prediction - target) ** 2).sum()


def mean_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean over samples of the summed squared error."""
    with torch.no_grad():
        return summed_squared_error(model(inputs), targets).item() / inputs.shape[0]


def collect_splitting(model: nn.Sequential, inputs: torch.Tensor, targets: torch.Tensor):
    """Collect the splitting matrices of the site ``model[0] -> model[2]`` (tanh)."""
    site = Site(model[0], model[2])
    with collect([site], extra_moments=[splitting_moment(torch.tanh)]) as moments:
        summed_squared_error(model(inputs), targets).backward()
    return moments[site]


def tanh_second_derivative(pre_activation: torch.Tensor) -> torch.Tensor:
    """Analytic second derivative of tanh."""
    value = torch.tanh(pre_activation)
    return -2 * value * (1 - value**2)


class TestSplittingMatrices(TorchTestCase):
    """The accumulated splitting matrices match brute-force computations."""

    def test_activation_second_derivative(self) -> None:
        points = torch.linspace(-3, 3, 41, dtype=torch.float64)
        self.assertAllClose(
            activation_second_derivative(torch.tanh, points),
            tanh_second_derivative(points),
        )
        self.assertEqual(
            activation_second_derivative(torch.relu, points).abs().max().item(), 0.0
        )

    def test_linear_matches_per_sample_hessians(self) -> None:
        inputs = torch.randn(24, 3, dtype=torch.float64)
        targets = torch.randn(24, 2, dtype=torch.float64)
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
        matrices = collect_splitting(model, inputs, targets)["splitting_matrices"]

        hidden = model[1](model[0](inputs)).detach().requires_grad_()
        summed_squared_error(model[2](hidden), targets).backward()
        unit_gradient = hidden.grad
        augmented = torch.cat([inputs, torch.ones(24, 1, dtype=torch.float64)], dim=1)
        theta = torch.cat([model[0].weight, model[0].bias[:, None]], dim=1).detach()
        expected = torch.zeros(4, 4, 4, dtype=torch.float64)
        for unit in range(4):
            for row in range(24):
                hessian = torch.func.hessian(
                    lambda weights, x=augmented[row]: torch.tanh(weights @ x)
                )(theta[unit])
                expected[unit] += unit_gradient[row, unit] * hessian
        self.assertAllClose(matrices, expected / 24, atol=1e-10)

    def test_conv_matches_unfolded_patches(self) -> None:
        inputs = torch.randn(6, 2, 5, 5, dtype=torch.float64)
        targets = torch.randn(6, 2, 5, 5, dtype=torch.float64)
        model = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1), nn.Tanh(), nn.Conv2d(3, 2, 3, padding=1)
        ).double()
        matrices = collect_splitting(model, inputs, targets)["splitting_matrices"]

        pre_activation = model[0](inputs).detach()
        hidden = torch.tanh(pre_activation).requires_grad_()
        summed_squared_error(model[2](hidden), targets).backward()
        patches = nn.functional.unfold(inputs, 3, padding=1)  # (6, 18, 25)
        patches = torch.cat([patches, torch.ones(6, 1, 25, dtype=torch.float64)], dim=1)
        weights = hidden.grad * tanh_second_derivative(pre_activation)  # (6, 3, 5, 5)
        expected = torch.einsum("bup,bip,bjp->uij", weights.flatten(2), patches, patches)
        self.assertAllClose(matrices, expected / 6, atol=1e-10)

    def test_relu_has_no_curvature_to_split(self) -> None:
        inputs, targets = torch.randn(32, 3), torch.randn(32, 2)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        site = Site(model[0], model[2])
        with collect([site], extra_moments=[splitting_moment(torch.relu)]) as moments:
            summed_squared_error(model(inputs), targets).backward()
        self.assertEqual(splitting(moments[site]).units.numel(), 0)


class TestSplitting(TorchTestCase):
    """Splitting preserves the function at eps = 0 and follows its prediction."""

    def setUp(self) -> None:
        super().setUp()
        self.cases = {
            "linear": (
                nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).double(),
                torch.randn(256, 4, dtype=torch.float64),
                torch.randn(256, 3, dtype=torch.float64),
            ),
            "conv": (
                nn.Sequential(
                    nn.Conv2d(2, 4, 3, padding=1),
                    nn.Tanh(),
                    nn.Conv2d(4, 2, 3, padding=1),
                ).double(),
                torch.randn(16, 2, 6, 6, dtype=torch.float64),
                torch.randn(16, 2, 6, 6, dtype=torch.float64),
            ),
        }

    def test_selection_keeps_the_most_negative_indices(self) -> None:
        model, inputs, targets = self.cases["linear"]
        proposal = splitting(collect_splitting(model, inputs, targets), maximum_splits=2)
        self.assertLessEqual(proposal.units.numel(), 2)
        self.assertTrue(bool((proposal.eigenvalues < 0).all()))
        self.assertTrue(
            bool((proposal.eigenvalues[:-1] <= proposal.eigenvalues[1:]).all())
        )
        self.assertAllClose(
            proposal.directions.norm(dim=1),
            torch.ones(proposal.units.numel(), dtype=torch.float64),
        )

    def test_zero_step_preserves_the_function(self) -> None:
        for name, (model, inputs, targets) in self.cases.items():
            with self.subTest(case=name):
                proposal = splitting(collect_splitting(model, inputs, targets))
                self.assertGreater(proposal.units.numel(), 0)
                split = split_copy(model, "0", "2", proposal, epsilon=0.0)
                self.assertEqual(
                    split[0].weight.shape[0],
                    model[0].weight.shape[0] + proposal.units.numel(),
                )
                with torch.no_grad():
                    self.assertAllClose(split(inputs), model(inputs), atol=1e-12)

    def test_loss_change_matches_the_splitting_indices(self) -> None:
        for name, (model, inputs, targets) in self.cases.items():
            with self.subTest(case=name):
                proposal = splitting(collect_splitting(model, inputs, targets))
                predicted = 0.5 * proposal.eigenvalues.sum().item()
                epsilon = 1e-3
                split = split_copy(model, "0", "2", proposal, epsilon)
                measured = (
                    mean_loss(split, inputs, targets) - mean_loss(model, inputs, targets)
                ) / epsilon**2
                self.assertLess(predicted, 0)
                self.assertAlmostEqual(measured / predicted, 1.0, delta=1e-2)
