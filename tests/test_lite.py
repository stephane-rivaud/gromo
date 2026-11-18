"""Equivalence with v1 and behaviour checks for :mod:`gromo.lite`."""

import unittest

import torch
from torch import nn

from gromo.lite import (
    Site,
    collect,
    gradmax,
    grown_copy,
    input_patches,
    tiny,
    widen_linear,
)
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


def copy_parameters(source: nn.Module, target: nn.Module) -> None:
    """Copy the weight and bias of a v1 layer into a plain layer."""
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.bias.copy_(source.bias)


def weight_gradient_rows(layer: nn.Module, samples: int) -> torch.Tensor:
    """Return autograd's gradient in the (fan_in + 1, fan_out) layout of tensor_m."""
    gradient = torch.cat([layer.weight.grad.flatten(1), layer.bias.grad[:, None]], 1)
    return gradient.T / samples


class TestLinearMatchesV1(TorchTestCase):
    """Linear -> ReLU -> Linear: every TINY and GradMax tensor matches v1."""

    def setUp(self):
        super().setUp()
        device = global_device()
        self.inputs = torch.randn(256, 5, device=device)
        self.targets = torch.randn(256, 3, device=device)

        self.v1_producer = LinearGrowingModule(
            5, 4, post_layer_function=nn.ReLU(), device=device
        )
        self.v1_consumer = LinearGrowingModule(
            4, 3, previous_module=self.v1_producer, device=device
        )
        self.v1_producer.next_module = self.v1_consumer
        self.model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3)).to(device)
        copy_parameters(self.v1_producer, self.model[0])
        copy_parameters(self.v1_consumer, self.model[2])

        self.v1_producer.init_computation()
        self.v1_consumer.init_computation()
        for inputs, targets in zip(
            self.inputs.split(64), self.targets.split(64), strict=True
        ):
            prediction = self.v1_consumer(self.v1_producer(inputs))
            ((prediction - targets) ** 2).sum().backward()
            self.v1_producer.update_computation()
            self.v1_consumer.update_computation()

        self.site = Site(self.model[0], self.model[2])
        with collect([self.site]) as moments:
            for inputs, targets in zip(
                self.inputs.split(64), self.targets.split(64), strict=True
            ):
                ((self.model(inputs) - targets) ** 2).sum().backward()
        self.moments = moments[self.site]

    def test_moments_match_v1(self):
        expected = {
            "tensor_s": self.v1_consumer.tensor_s(),
            "tensor_m": self.v1_consumer.tensor_m(),
            "tensor_s_growth": self.v1_consumer.tensor_s_growth(),
            "tensor_m_prev": self.v1_consumer.tensor_m_prev(),
            "cross_covariance": self.v1_consumer.cross_covariance(),
        }
        for name, v1_value in expected.items():
            with self.subTest(moment=name):
                self.assertAllClose(self.moments[name], v1_value, atol=1e-6)

    def test_tensor_m_is_the_scaled_weight_gradient(self):
        self.assertAllClose(
            self.moments["tensor_m"],
            weight_gradient_rows(self.model[2], self.inputs.shape[0]),
            atol=1e-6,
        )

    def test_tiny_matches_v1(self):
        self.v1_consumer.compute_optimal_delta()
        alpha_weight, alpha_bias, omega, eigenvalues = (
            self.v1_consumer._compute_optimal_added_parameters(
                statistical_threshold=0, use_projection=True, use_covariance=True
            )
        )
        proposal = tiny(self.moments, statistical_threshold=0)

        self.assertAllClose(proposal.optimal_delta, self.v1_consumer.delta_raw, atol=1e-6)
        self.assertAllClose(
            proposal.parameter_update_decrease,
            self.v1_consumer.parameter_update_decrease,
            atol=1e-6,
        )
        self.assertAllClose(
            proposal.alpha, torch.cat([alpha_weight, alpha_bias[:, None]], 1), atol=1e-5
        )
        self.assertAllClose(proposal.omega, omega, atol=1e-5)
        self.assertAllClose(proposal.eigenvalues, eigenvalues, atol=1e-6)

    def test_gradmax_matches_v1(self):
        alpha_weight, alpha_bias, omega, eigenvalues = (
            self.v1_consumer._compute_optimal_added_parameters(
                statistical_threshold=0,
                use_projection=False,
                use_covariance=False,
                alpha_zero=True,
                ignore_singular_values=True,
            )
        )
        proposal = gradmax(self.moments, statistical_threshold=0)

        self.assertIsNone(proposal.optimal_delta)
        self.assertAllClose(
            proposal.alpha, torch.cat([alpha_weight, alpha_bias[:, None]], 1)
        )
        self.assertAllClose(proposal.omega, omega, atol=1e-5)
        self.assertAllClose(proposal.eigenvalues, eigenvalues, atol=1e-6)


class TestConvMatchesV1(TorchTestCase):
    """Conv2d -> ReLU -> Conv2d: single-layer moments match v1.

    ``tensor_s_growth`` is the producer input covariance, as in
    ``RestrictedConv2dGrowingModule``; ``FullConv2dGrowingModule`` instead
    unfolds the producer input over the composed receptive field.
    """

    def setUp(self):
        super().setUp()
        device = global_device()
        self.inputs = torch.randn(16, 3, 8, 8, device=device)
        self.v1_producer = RestrictedConv2dGrowingModule(
            3, 4, 3, padding=1, post_layer_function=nn.ReLU(), device=device
        )
        self.v1_consumer = RestrictedConv2dGrowingModule(
            4, 5, 3, padding=1, previous_module=self.v1_producer, device=device
        )
        self.v1_producer.next_module = self.v1_consumer
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 5, 3, padding=1)
        ).to(device)
        copy_parameters(self.v1_producer, self.model[0])
        copy_parameters(self.v1_consumer, self.model[2])

        self.v1_producer.init_computation()
        self.v1_consumer.init_computation()
        self.v1_consumer(self.v1_producer(self.inputs)).pow(2).sum().backward()
        self.v1_producer.update_computation()
        self.v1_consumer.update_computation()

        self.site = Site(self.model[0], self.model[2])
        with collect([self.site]) as moments:
            self.model(self.inputs).pow(2).sum().backward()
        self.moments = moments[self.site]

    def test_moments_match_v1(self):
        expected = {
            "tensor_s": self.v1_consumer.tensor_s(),
            "tensor_m": self.v1_consumer.tensor_m(),
            "tensor_s_growth": self.v1_consumer.tensor_s_growth(),
        }
        for name, v1_value in expected.items():
            with self.subTest(moment=name):
                self.assertAllClose(self.moments[name], v1_value, atol=1e-5)

    def test_tensor_m_is_the_scaled_weight_gradient(self):
        self.assertAllClose(
            self.moments["tensor_m"],
            weight_gradient_rows(self.model[2], self.inputs.shape[0]),
            atol=1e-5,
        )

    def test_pair_moments_are_not_collected(self):
        self.assertNotIn("tensor_m_prev", self.moments)
        with self.assertRaises(KeyError):
            tiny(self.moments)


class TestCollect(TorchTestCase):
    """Hooks are temporary and only count passes that reach the backward pass."""

    def setUp(self):
        super().setUp()
        self.model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
        self.inputs = torch.randn(8, 5)

    def test_hooks_are_removed_on_exit(self):
        site = Site(self.model[0], self.model[2])
        with self.assertRaises(ValueError), collect([site]):
            self.model(self.inputs).sum().backward()
            raise ValueError("leave the context early")
        self.assertEqual(len(self.model[0]._forward_hooks), 0)
        self.assertEqual(len(self.model[2]._forward_hooks), 0)

    def test_passes_without_gradient_are_ignored(self):
        site = Site(self.model[0], self.model[2])
        with collect([site]) as moments:
            with torch.no_grad():
                self.model(torch.randn(100, 5))
            self.model(self.inputs).sum().backward()
        self.assertEqual(moments[site].samples, self.inputs.shape[0])

    def test_inverted_site_raises(self):
        site = Site(self.model[2], self.model[0])
        with self.assertRaises(RuntimeError), collect([site]):
            self.model(self.inputs)

    def test_three_dimensional_inputs_count_batch_elements(self):
        site = Site(self.model[0], self.model[2])
        with collect([site]) as moments:
            self.model(torch.randn(2, 7, 5)).sum().backward()
        self.assertEqual(moments[site].samples, 2)
        self.assertShapeEqual(moments[site]["tensor_m_prev"], (6, 3))

    def test_layers_without_bias_grow_without_bias_column(self):
        model = nn.Sequential(
            nn.Linear(5, 4, bias=False), nn.Tanh(), nn.Linear(4, 3, bias=False)
        )
        site = Site(model[0], model[2])
        with collect([site]) as moments:
            model(self.inputs).pow(2).sum().backward()
        proposal = tiny(moments[site], statistical_threshold=0)
        self.assertShapeEqual(proposal.alpha, (-1, 5))
        self.assertShapeEqual(proposal.optimal_delta, (3, 4))
        grown = grown_copy(model, "0", "2", proposal, gamma=0.0)
        self.assertIsNone(grown[0].bias)
        with torch.no_grad():
            self.assertAllClose(grown(self.inputs), model(self.inputs), atol=0)

    def test_unsupported_convolutions_raise(self):
        grouped = nn.Conv2d(4, 4, 3, groups=2)
        with self.assertRaises(NotImplementedError):
            input_patches(grouped, torch.randn(1, 4, 5, 5))
        with self.assertRaises(TypeError):
            input_patches(nn.Conv1d(4, 4, 3), torch.randn(1, 4, 5))


class TestSurgery(TorchTestCase):
    """Growth preserves the function at gamma = 0 and follows the first-order term."""

    def setUp(self):
        super().setUp()
        self.inputs = torch.randn(256, 5, dtype=torch.float64)
        self.targets = torch.randn(256, 3, dtype=torch.float64)
        self.model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3)).double()
        self.site = Site(self.model[0], self.model[2])
        with collect([self.site]) as moments:
            ((self.model(self.inputs) - self.targets) ** 2).sum().backward()
        self.moments = moments[self.site]
        self.proposal = tiny(self.moments, statistical_threshold=0)

    def mean_loss(self, model: nn.Module) -> float:
        with torch.no_grad():
            return ((model(self.inputs) - self.targets) ** 2).sum(1).mean().item()

    def test_zero_gamma_preserves_the_function(self):
        grown = grown_copy(self.model, "0", "2", self.proposal, gamma=0.0)
        added_units = self.proposal.alpha.shape[0]
        self.assertEqual(grown[0].out_features, 4 + added_units)
        self.assertEqual(grown[2].in_features, 4 + added_units)
        with torch.no_grad():
            self.assertAllClose(grown(self.inputs), self.model(self.inputs), atol=0)

    def test_original_model_is_unchanged(self):
        weight_before = self.model[2].weight.detach().clone()
        grown_copy(self.model, "0", "2", self.proposal, gamma=0.5)
        self.assertEqual(self.model[0].out_features, 4)
        self.assertAllClose(self.model[2].weight, weight_before, atol=0)

    def test_loss_decrease_matches_the_first_order_term(self):
        # Tanh has slope 1 at 0, so the new units add gamma^2 * omega @ alpha @ x.
        exact_first_order = self.proposal.parameter_update_decrease - torch.trace(
            self.proposal.omega @ self.proposal.alpha @ self.moments["tensor_m_prev"]
        )
        gamma = 1e-4
        grown = grown_copy(self.model, "0", "2", self.proposal, gamma)
        measured = (self.mean_loss(self.model) - self.mean_loss(grown)) / gamma**2
        self.assertAlmostEqual(measured / exact_first_order.item(), 1.0, delta=1e-4)

    def test_widen_linear_rejects_convolutions(self):
        with self.assertRaises(TypeError):
            widen_linear(nn.Conv2d(3, 4, 3), nn.Linear(4, 2), self.proposal, 0.0)


if __name__ == "__main__":
    unittest.main()
