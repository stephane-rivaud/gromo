"""Tests of the gromo.lite helpers shared by growth methods.

``first_order_decrease`` is checked against finite differences with a GELU
activation, ``random_proposal`` against v1's initialization formulas, and the
extra-moment extension point of ``collect`` against autograd.
"""

import torch
from torch import nn

from gromo.lite import (
    Site,
    collect,
    first_order_decrease,
    grown_copy,
    random_proposal,
    tiny,
)


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


def mean_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean over samples of the summed squared error."""
    with torch.no_grad():
        return ((model(inputs) - targets) ** 2).sum(dim=1).mean().item()


class TestFirstOrderDecrease(TorchTestCase):
    """The predicted first-order decrease is the one the surgery follows."""

    def test_gelu_slope_matches_finite_differences(self) -> None:
        inputs = torch.randn(256, 5, dtype=torch.float64)
        targets = torch.randn(256, 3, dtype=torch.float64)
        model = nn.Sequential(nn.Linear(5, 4), nn.GELU(), nn.Linear(4, 3)).double()
        site = Site(model[0], model[2])
        with collect([site]) as moments:
            ((model(inputs) - targets) ** 2).sum().backward()
        proposal = tiny(moments[site], statistical_threshold=0)
        predicted = first_order_decrease(moments[site], proposal, activation_slope=0.5)

        gamma = 1e-4
        grown = grown_copy(model, "0", "2", proposal, gamma)
        measured = (
            mean_loss(model, inputs, targets) - mean_loss(grown, inputs, targets)
        ) / gamma**2
        self.assertAlmostEqual(measured / predicted.item(), 1.0, delta=1e-3)


class TestRandomProposal(TorchTestCase):
    """Random new units follow v1's extension initializations."""

    def test_mode_b_draws_kaiming_fan_in_and_zero_fan_out(self) -> None:
        producer, consumer = nn.Linear(50, 8), nn.Linear(8, 3)
        proposal = random_proposal(producer, consumer, 400, "kaiming_normal", "zeros")
        self.assertShapeEqual(proposal.alpha, (400, 51))
        self.assertShapeEqual(proposal.omega, (3, 400))
        self.assertEqual(proposal.omega.abs().max().item(), 0.0)
        self.assertEqual(proposal.alpha[:, -1].abs().max().item(), 0.0)
        self.assertAlmostEqual(
            proposal.alpha[:, :-1].std().item(), (2 / 50) ** 0.5, delta=0.01
        )

    def test_mode_e_copies_the_existing_standard_deviations(self) -> None:
        producer, consumer = nn.Linear(50, 8), nn.Linear(8, 30)
        with torch.no_grad():
            producer.weight.normal_(0.0, 0.3)
            consumer.weight.normal_(0.0, 0.05)
        proposal = random_proposal(producer, consumer, 400, "copy_normal", "copy_normal")
        self.assertAlmostEqual(proposal.alpha[:, :-1].std().item(), 0.3, delta=0.02)
        self.assertAlmostEqual(proposal.omega.std().item(), 0.05, delta=0.005)

    def test_kaiming_fan_out_uses_the_grown_consumer_fan_in(self) -> None:
        producer, consumer = nn.Conv2d(3, 4, 3), nn.Conv2d(4, 200, 3)
        proposal = random_proposal(producer, consumer, 100, "zeros", "kaiming_normal")
        self.assertShapeEqual(proposal.alpha, (100, 3 * 9 + 1))
        self.assertShapeEqual(proposal.omega, (200, 100))
        expected = (2 / ((4 + 100) * 9)) ** 0.5
        self.assertAlmostEqual(
            proposal.omega.std().item(), expected, delta=0.1 * expected
        )

    def test_zero_fan_out_preserves_the_function(self) -> None:
        model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
        proposal = random_proposal(model[0], model[2], 6, "kaiming_normal", "zeros")
        grown = grown_copy(model, "0", "2", proposal, gamma=1.0)
        inputs = torch.randn(16, 5)
        self.assertEqual(grown[0].out_features, 10)
        with torch.no_grad():
            self.assertAllClose(grown(inputs), model(inputs), atol=1e-7)

    def test_unknown_initialization_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_proposal(nn.Linear(5, 4), nn.Linear(4, 3), 2, "orthogonal", "zeros")


class TestExtraMoments(TorchTestCase):
    """An extra moment sees the tensors of the site and can derive dL/dz."""

    def test_consumer_input_gradient_matches_autograd(self) -> None:
        inputs, targets = torch.randn(32, 5), torch.randn(32, 3)
        model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
        seen = []

        def record(batch, moments):
            seen.append((batch.producer_output, batch.consumer_input_gradient()))
            moments.add_sum("count", torch.ones(()))

        site = Site(model[0], model[2])
        with collect([site], extra_moments=[record]) as moments:
            ((model(inputs) - targets) ** 2).sum().backward()

        hidden = model[1](model[0](inputs)).detach().requires_grad_()
        ((model[2](hidden) - targets) ** 2).sum().backward()
        producer_output, gradient = seen[0]
        self.assertAllClose(producer_output, model[0](inputs).detach())
        self.assertAllClose(gradient, hidden.grad, atol=1e-6)
        self.assertAllClose(moments[site]["count"], torch.tensor(1 / 32))
