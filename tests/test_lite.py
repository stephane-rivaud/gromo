"""Equivalence with v1 and behaviour checks for :mod:`gromo.lite`."""

import inspect
import unittest

import torch
from torch import nn

from gromo.containers.growing_block import Conv2dGrowingBlock
from gromo.lite import (
    Moments,
    Proposal,
    Site,
    aligned_producer_patches,
    collect,
    fixed_proportional_budget,
    gradmax,
    gradmax_normalization,
    grown_copy,
    input_patches,
    optimal_updates,
    tensor_n,
    tiny,
    widen,
    widen_conv2d,
    widen_linear,
    widen_norm,
)
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_module import GrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingBatchNorm2d
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import ReLUDerivativeOneAtZero, global_device


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


def copy_parameters(source: nn.Module, target: nn.Module) -> None:
    """Copy the weight and bias of a v1 layer into a plain layer."""
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is not None:
            target.bias.copy_(source.bias)


def weight_gradient_rows(layer: nn.Module, samples: int) -> torch.Tensor:
    """Return autograd's gradient in the (fan_in [+1], fan_out) layout of tensor_m."""
    gradient = layer.weight.grad.flatten(1)
    if layer.bias is not None:
        gradient = torch.cat([gradient, layer.bias.grad[:, None]], 1)
    return gradient.T / samples


def with_bias(
    alpha_weight: torch.Tensor, alpha_bias: torch.Tensor | None
) -> torch.Tensor:
    """Return a v1 fan-in in the lite layout: flattened weight, bias column last."""
    alpha = alpha_weight.flatten(1)
    if alpha_bias is None:
        return alpha
    return torch.cat([alpha, alpha_bias[:, None]], 1)


# (producer kwargs, consumer kwargs) of nn.Conv2d sites; "bias" maps to use_bias in v1.
CONV_GEOMETRIES = {
    "same padding": ({"kernel_size": 3, "padding": 1}, {"kernel_size": 3, "padding": 1}),
    "no padding": ({"kernel_size": 3}, {"kernel_size": 3}),
    "consumer stride 2": (
        {"kernel_size": 3, "padding": 1},
        {"kernel_size": 3, "padding": 1, "stride": 2},
    ),
    "producer stride 2, consumer 5x5": (
        {"kernel_size": 3, "padding": 1, "stride": 2},
        {"kernel_size": 5, "padding": 2},
    ),
    "consumer dilation 2": (
        {"kernel_size": 3, "padding": 1},
        {"kernel_size": 3, "padding": 1, "dilation": 2},
    ),
    "rectangular kernels": (
        {"kernel_size": (3, 1), "padding": (1, 0)},
        {"kernel_size": (1, 3), "padding": (0, 2), "stride": (2, 1)},
    ),
    "no bias": (
        {"kernel_size": 3, "padding": 1, "bias": False},
        {"kernel_size": 3, "padding": 1, "bias": False},
    ),
}


def v1_restricted_conv(
    in_channels: int, out_channels: int, geometry: dict, **kwargs
) -> RestrictedConv2dGrowingModule:
    """Build a v1 Restricted convolution from nn.Conv2d keyword arguments."""
    geometry = dict(geometry)
    use_bias = geometry.pop("bias", True)
    return RestrictedConv2dGrowingModule(
        in_channels, out_channels, use_bias=use_bias, **geometry, **kwargs
    )


def run_linear_site() -> tuple[LinearGrowingModule, nn.Sequential, Moments]:
    """Accumulate v1 and lite statistics of Linear -> ReLU -> Linear on the same data.

    Returns
    -------
    tuple[LinearGrowingModule, nn.Sequential, Moments]
        v1 consumer, plain model and lite moments of the site
    """
    device = global_device()
    all_inputs = torch.randn(256, 5, device=device)
    all_targets = torch.randn(256, 3, device=device)

    v1_producer = LinearGrowingModule(5, 4, post_layer_function=nn.ReLU(), device=device)
    v1_consumer = LinearGrowingModule(4, 3, previous_module=v1_producer, device=device)
    v1_producer.next_module = v1_consumer
    model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3)).to(device)
    copy_parameters(v1_producer, model[0])
    copy_parameters(v1_consumer, model[2])

    v1_producer.init_computation()
    v1_consumer.init_computation()
    for inputs, targets in zip(all_inputs.split(64), all_targets.split(64), strict=True):
        prediction = v1_consumer(v1_producer(inputs))
        ((prediction - targets) ** 2).sum().backward()
        v1_producer.update_computation()
        v1_consumer.update_computation()

    site = Site(model[0], model[2])
    with collect([site]) as moments:
        for inputs, targets in zip(
            all_inputs.split(64), all_targets.split(64), strict=True
        ):
            ((model(inputs) - targets) ** 2).sum().backward()
    return v1_consumer, model, moments[site]


def run_conv_site(
    producer_geometry: dict, consumer_geometry: dict
) -> tuple[RestrictedConv2dGrowingModule, nn.Sequential, Moments, int]:
    """Accumulate v1 and lite statistics of Conv2d -> ReLU -> Conv2d on the same data.

    The loss has random targets: with ``||f(x)||^2`` the output gradient is
    linear in the consumer input, so ``tensor_n`` would be round-off noise.

    Returns
    -------
    tuple[RestrictedConv2dGrowingModule, nn.Sequential, Moments, int]
        v1 consumer, plain model, lite moments of the site and number of samples
    """
    device = global_device()
    inputs = torch.randn(6, 3, 9, 9, device=device)
    v1_producer = v1_restricted_conv(
        3, 4, producer_geometry, post_layer_function=nn.ReLU(), device=device
    )
    v1_consumer = v1_restricted_conv(
        4, 5, consumer_geometry, previous_module=v1_producer, device=device
    )
    v1_producer.next_module = v1_consumer
    model = nn.Sequential(
        nn.Conv2d(3, 4, **producer_geometry),
        nn.ReLU(),
        nn.Conv2d(4, 5, **consumer_geometry),
    ).to(device)
    copy_parameters(v1_producer, model[0])
    copy_parameters(v1_consumer, model[2])

    with torch.no_grad():
        targets = torch.randn_like(model(inputs))

    v1_producer.init_computation()
    v1_consumer.init_computation()
    ((v1_consumer(v1_producer(inputs)) - targets) ** 2).sum().backward()
    v1_producer.update_computation()
    v1_consumer.update_computation()

    site = Site(model[0], model[2])
    with collect([site]) as moments:
        ((model(inputs) - targets) ** 2).sum().backward()
    return v1_consumer, model, moments[site], inputs.shape[0]


# GradMax as configured by experimental_grow (growing/compute_optimal_updates_kwargs/
# gradmax.yaml, with the statistical threshold forced to 0 by the training script).
V1_GRADMAX_KWARGS = {
    "statistical_threshold": 0,
    "compute_delta": False,
    "use_covariance": False,
    "alpha_zero": True,
    "use_projection": False,
    "ignore_singular_values": True,
}


def v1_fan_out(consumer: GrowingModule) -> torch.Tensor:
    """Return the v1 fan-out of the new units in the lite ``(consumer_out, k)`` layout."""
    weight = consumer.extended_input_layer.weight
    if weight.dim() == 2:
        return weight
    centre_height, centre_width = (size // 2 for size in weight.shape[2:])
    return weight[:, :, centre_height, centre_width]


class TestLinearMatchesV1(TorchTestCase):
    """Linear -> ReLU -> Linear: every TINY and GradMax tensor matches v1."""

    def setUp(self):
        super().setUp()
        self.v1_consumer, self.model, self.moments = run_linear_site()

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
            weight_gradient_rows(self.model[2], self.moments.samples),
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

    def test_projection_without_update_matches_v1(self):
        # v1 computes the update to project the gradient, but does not apply it
        self.v1_consumer.compute_optimal_updates(
            statistical_threshold=0, compute_delta=False, use_projection=True
        )
        proposal = optimal_updates(
            self.moments,
            compute_delta=False,
            use_projection=True,
            statistical_threshold=0,
        )
        self.assertIsNone(proposal.optimal_delta)
        self.assertEqual(proposal.parameter_update_decrease.item(), 0.0)
        self.assertAllClose(
            proposal.eigenvalues, self.v1_consumer.eigenvalues_extension, atol=1e-6
        )
        v1_alpha = self.v1_consumer.previous_module.extended_output_layer
        v1_omega = self.v1_consumer.extended_input_layer
        self.assertAllClose(
            proposal.omega @ proposal.alpha,
            v1_omega.weight @ torch.cat([v1_alpha.weight, v1_alpha.bias[:, None]], 1),
            atol=1e-5,
        )

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


class TestRestrictedConvMatchesV1(TorchTestCase):
    """Conv2d -> ReLU -> Conv2d: lite matches RestrictedConv2dGrowingModule.

    Each test runs every geometry of ``CONV_GEOMETRIES`` as a sub-test.
    """

    def assertSameUnits(self, proposal, v1_alpha, v1_omega, atol: float = 1e-4):
        """Compare new units up to the SVD sign: (alpha, omega) ~ (-alpha, -omega)."""
        signs = torch.sign((proposal.alpha * v1_alpha).sum(dim=1))
        self.assertAllClose(proposal.alpha * signs[:, None], v1_alpha, atol=atol)
        self.assertAllClose(proposal.omega * signs[None, :], v1_omega, atol=atol)

    def run_both(self, producer_geometry: dict, consumer_geometry: dict):
        """Accumulate v1 and lite statistics on the same weights and inputs."""
        return run_conv_site(producer_geometry, consumer_geometry)

    def test_moments_match_v1(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            v1_consumer, _, moments, _ = self.run_both(
                producer_geometry, consumer_geometry
            )
            expected = {
                "tensor_s": v1_consumer.tensor_s(),
                "tensor_m": v1_consumer.tensor_m(),
                "tensor_s_growth": v1_consumer.tensor_s_growth(),
                "tensor_m_prev": v1_consumer.tensor_m_prev(),
                "cross_covariance": v1_consumer.cross_covariance(),
            }
            for moment, v1_value in expected.items():
                with self.subTest(geometry=name, moment=moment):
                    self.assertAllClose(moments[moment], v1_value, atol=1e-5)

    def test_tensor_m_is_the_scaled_weight_gradient(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                _, model, moments, samples = self.run_both(
                    producer_geometry, consumer_geometry
                )
                self.assertAllClose(
                    moments["tensor_m"],
                    weight_gradient_rows(model[2], samples),
                    atol=1e-5,
                )

    def test_tiny_matches_v1(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                v1_consumer, _, moments, _ = self.run_both(
                    producer_geometry, consumer_geometry
                )
                v1_consumer.compute_optimal_delta()
                alpha_weight, alpha_bias, omega, eigenvalues = (
                    v1_consumer._compute_optimal_added_parameters(
                        statistical_threshold=0, use_projection=True, use_covariance=True
                    )
                )
                proposal = tiny(moments, statistical_threshold=0)

                self.assertAllClose(
                    proposal.optimal_delta, v1_consumer.delta_raw, atol=1e-5
                )
                self.assertAllClose(
                    tensor_n(moments, proposal.optimal_delta),
                    v1_consumer.tensor_n,
                    atol=1e-5,
                )
                self.assertAllClose(
                    proposal.parameter_update_decrease,
                    v1_consumer.parameter_update_decrease,
                    atol=1e-5,
                )
                self.assertSameUnits(proposal, with_bias(alpha_weight, alpha_bias), omega)
                self.assertAllClose(proposal.eigenvalues, eigenvalues, atol=1e-5)

    def test_gradmax_matches_v1(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                v1_consumer, _, moments, _ = self.run_both(
                    producer_geometry, consumer_geometry
                )
                alpha_weight, alpha_bias, omega, eigenvalues = (
                    v1_consumer._compute_optimal_added_parameters(
                        statistical_threshold=0,
                        use_projection=False,
                        use_covariance=False,
                        alpha_zero=True,
                        ignore_singular_values=True,
                    )
                )
                proposal = gradmax(moments, statistical_threshold=0)
                self.assertEqual(proposal.alpha.abs().max().item(), 0.0)
                self.assertAllClose(with_bias(alpha_weight, alpha_bias), proposal.alpha)
                signs = torch.sign((proposal.omega * omega).sum(dim=0))
                self.assertAllClose(proposal.omega * signs[None, :], omega, atol=1e-4)
                self.assertAllClose(proposal.eigenvalues, eigenvalues, atol=1e-5)


class TestConvSurgery(TorchTestCase):
    """Conv2d -> Tanh -> Conv2d in float64: growth follows the restricted scheme."""

    def grow(self, producer_geometry: dict, consumer_geometry: dict):
        """Build a float64 model, collect its moments and solve TINY."""
        inputs = torch.randn(8, 3, 9, 9, dtype=torch.float64)
        model = nn.Sequential(
            nn.Conv2d(3, 4, **producer_geometry),
            nn.Tanh(),
            nn.Conv2d(4, 5, **consumer_geometry),
        ).double()
        with torch.no_grad():
            targets = torch.randn_like(model(inputs))
        site = Site(model[0], model[2])
        with collect([site]) as moments:
            ((model(inputs) - targets) ** 2).sum().backward()
        proposal = tiny(moments[site], statistical_threshold=0)
        return model, inputs, targets, moments[site], proposal

    @staticmethod
    def mean_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            return ((model(inputs) - targets) ** 2).sum(dim=(1, 2, 3)).mean().item()

    def test_zero_gamma_preserves_the_function(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                model, inputs, _, _, proposal = self.grow(
                    producer_geometry, consumer_geometry
                )
                grown = grown_copy(model, "0", "2", proposal, gamma=0.0)
                added_units = proposal.alpha.shape[0]
                self.assertEqual(grown[0].out_channels, 4 + added_units)
                self.assertEqual(grown[2].in_channels, 4 + added_units)
                with torch.no_grad():
                    self.assertAllClose(grown(inputs), model(inputs), atol=1e-12)

    def test_loss_decrease_matches_the_first_order_term(self):
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                model, inputs, targets, moments, proposal = self.grow(
                    producer_geometry, consumer_geometry
                )
                exact_first_order = proposal.parameter_update_decrease - torch.trace(
                    proposal.omega @ proposal.alpha @ moments["tensor_m_prev"]
                )
                gamma = 1e-4
                grown = grown_copy(model, "0", "2", proposal, gamma)
                measured = (
                    self.mean_loss(model, inputs, targets)
                    - self.mean_loss(grown, inputs, targets)
                ) / gamma**2
                self.assertAlmostEqual(
                    measured / exact_first_order.item(), 1.0, delta=1e-4
                )

    def test_new_channels_only_use_the_centre_tap(self):
        producer_geometry, consumer_geometry = CONV_GEOMETRIES[
            "producer stride 2, consumer 5x5"
        ]
        model, _, _, _, proposal = self.grow(producer_geometry, consumer_geometry)
        grown = grown_copy(model, "0", "2", proposal, gamma=0.5)
        new_fan_out = grown[2].weight[:, 4:].detach().clone()
        self.assertAllClose(new_fan_out[:, :, 2, 2], 0.5 * proposal.omega)
        new_fan_out[:, :, 2, 2] = 0
        self.assertEqual(new_fan_out.abs().max().item(), 0.0)

    def test_widen_rejects_mixed_layer_types(self):
        _, _, _, _, proposal = self.grow(*CONV_GEOMETRIES["same padding"])
        with self.assertRaises(TypeError):
            widen(nn.Conv2d(3, 4, 3), nn.Linear(4, 2), proposal, 0.0)
        with self.assertRaises(TypeError):
            widen_conv2d(nn.Linear(3, 4), nn.Linear(4, 2), proposal, 0.0)


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
            self.assertAllClose(grown(self.inputs), model(self.inputs), atol=0, rtol=0)

    def test_unsupported_convolutions_raise(self):
        grouped = nn.Conv2d(4, 4, 3, groups=2)
        with self.assertRaises(NotImplementedError):
            input_patches(grouped, torch.randn(1, 4, 5, 5))
        with self.assertRaises(TypeError):
            input_patches(nn.Conv1d(4, 4, 3), torch.randn(1, 4, 5))

    def test_full_conv_scheme_raises(self):
        convolutions = nn.Conv2d(3, 4, 3), nn.Conv2d(4, 5, 3)
        with self.assertRaisesRegex(NotImplementedError, "FullConv2dGrowingModule"):
            Site(*convolutions, conv_scheme="full")
        with self.assertRaises(ValueError):
            Site(*convolutions, conv_scheme="diagonal")

    def test_shape_changing_layers_between_producer_and_consumer_raise(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 5, 3, padding=1),
        )
        site = Site(model[0], model[2])
        with self.assertRaisesRegex(ValueError, "preserve the shape"), collect([site]):
            model(torch.randn(2, 3, 8, 8))

    def test_mixed_sites_have_no_pair_moments(self):
        site = Site(nn.Conv2d(3, 4, 3), nn.Linear(4, 2))
        self.assertFalse(site.has_pair_moments)
        with self.assertRaises(TypeError):
            aligned_producer_patches(site, torch.randn(4, 28), torch.randn(4, 4))


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
            # the new units add exact zeros, but a wider matmul may sum in another
            # order: equal up to float64 rounding, not bit for bit
            self.assertAllClose(
                grown(self.inputs), self.model(self.inputs), atol=1e-14, rtol=0
            )

    def test_original_model_is_unchanged(self):
        weight_before = self.model[2].weight.detach().clone()
        grown_copy(self.model, "0", "2", self.proposal, gamma=0.5)
        self.assertEqual(self.model[0].out_features, 4)
        self.assertAllClose(self.model[2].weight, weight_before, atol=0, rtol=0)

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


def assert_same_columns_up_to_sign(
    test: TorchTestCase, actual: torch.Tensor, expected: torch.Tensor, atol: float
) -> None:
    """Compare two ``(features, k)`` fan-outs column by column up to the SVD sign."""
    signs = torch.sign((actual * expected).sum(dim=0))
    test.assertAllClose(actual * signs[None, :], expected, atol=atol)


class TestGradmaxNormalizationMatchesV1(TorchTestCase):
    """gradmax_normalization matches normalize_optimal_updates("gradmax_normalization")."""

    SCALES = (1.0, 0.5, 2.0)

    def normalize_both(
        self, consumer: GrowingModule, omega: torch.Tensor, scale: float
    ) -> tuple[Proposal, torch.Tensor, torch.Tensor]:
        """Normalize the same fan-out and eigenvalues with v1 and with lite."""
        added_units = omega.shape[1]
        eigenvalues = torch.rand(added_units, device=omega.device) + 0.5
        proposal = Proposal(
            optimal_delta=None,
            alpha=torch.zeros(added_units, 1, device=omega.device),
            omega=omega.clone(),
            eigenvalues=eigenvalues.clone(),
            parameter_update_decrease=torch.zeros((), device=omega.device),
        )
        if isinstance(consumer, RestrictedConv2dGrowingModule):
            consumer.extended_input_layer = consumer.linear_layer_of_tensor(omega.clone())
        else:
            consumer.extended_input_layer = consumer.layer_of_tensor(
                omega.clone(), bias=None
            )
        consumer.eigenvalues_extension = eigenvalues.clone()
        consumer.normalize_optimal_updates(
            normalization_type="gradmax_normalization", gradmax_scale=scale
        )
        normalized = gradmax_normalization(proposal, consumer.weight, scale)
        return normalized, v1_fan_out(consumer), consumer.eigenvalues_extension

    def test_linear_consumer(self):
        device = global_device()
        for scale in self.SCALES:
            with self.subTest(scale=scale):
                consumer = LinearGrowingModule(4, 3, use_bias=False, device=device)
                omega = torch.randn(3, 5, device=device)
                omega[:, 2] = 0.0  # a zero column keeps its value and eigenvalue
                normalized, v1_omega, v1_eigenvalues = self.normalize_both(
                    consumer, omega, scale
                )
                self.assertAllClose(normalized.omega, v1_omega, atol=1e-6)
                self.assertAllClose(normalized.eigenvalues, v1_eigenvalues, atol=1e-6)
                self.assertEqual(normalized.omega[:, 2].abs().max().item(), 0.0)

    def test_restricted_conv_consumer(self):
        device = global_device()
        for scale in self.SCALES:
            with self.subTest(scale=scale):
                consumer = RestrictedConv2dGrowingModule(
                    4, 5, kernel_size=3, padding=1, use_bias=False, device=device
                )
                omega = torch.randn(5, 3, device=device)
                normalized, v1_omega, v1_eigenvalues = self.normalize_both(
                    consumer, omega, scale
                )
                self.assertAllClose(normalized.omega, v1_omega, atol=1e-6)
                self.assertAllClose(normalized.eigenvalues, v1_eigenvalues, atol=1e-6)

    def test_columns_reach_the_target_norm(self):
        weight = torch.randn(5, 4, 3, 3)
        omega = torch.randn(5, 3)
        proposal = Proposal(
            None, torch.zeros(3, 1), omega, torch.ones(3), torch.zeros(())
        )
        normalized = gradmax_normalization(proposal, weight, scale=0.5)
        target = 0.5 * weight.pow(2).sum(dim=(0, 2, 3)).sqrt().mean()
        self.assertAllClose(normalized.omega.norm(dim=0), target.expand(3), atol=1e-6)
        self.assertAllClose(
            normalized.eigenvalues, (target / omega.norm(dim=0)).sqrt(), atol=1e-6
        )
        self.assertIs(normalized.alpha, proposal.alpha)

    def test_gradmax_pipeline_matches_v1(self):
        """compute_optimal_updates + sub-selection + normalization, as in experimental_grow."""
        for name, (producer_geometry, consumer_geometry) in CONV_GEOMETRIES.items():
            with self.subTest(geometry=name):
                v1_consumer, model, moments, _ = run_conv_site(
                    producer_geometry, consumer_geometry
                )
                v1_consumer.compute_optimal_updates(
                    maximum_added_neurons=2, **V1_GRADMAX_KWARGS
                )
                v1_consumer.sub_select_optimal_added_parameters(threshold=0)
                v1_consumer.normalize_optimal_updates(
                    normalization_type="gradmax_normalization", gradmax_scale=1.0
                )
                proposal = gradmax_normalization(
                    gradmax(moments, maximum_added_neurons=2, statistical_threshold=0),
                    model[2].weight,
                )
                self.assertEqual(proposal.omega.shape[1], 2)
                assert_same_columns_up_to_sign(
                    self, proposal.omega, v1_fan_out(v1_consumer), atol=1e-5
                )
                self.assertAllClose(
                    proposal.eigenvalues, v1_consumer.eigenvalues_extension, atol=1e-5
                )

    def test_non_positive_scale_raises(self):
        proposal = Proposal(None, torch.zeros(1, 1), torch.ones(2, 1), torch.ones(1), 0)
        with self.assertRaises(ValueError):
            gradmax_normalization(proposal, torch.ones(2, 3), scale=0.0)

    def test_degenerate_consumers_leave_the_proposal_unchanged(self):
        proposal = Proposal(None, torch.zeros(1, 1), torch.ones(2, 1), torch.ones(1), 0)
        for name, weight in {
            "zero weight": torch.zeros(2, 3),
            "no input unit": torch.zeros(2, 0),
            "vector weight": torch.ones(2),
        }.items():
            with self.subTest(case=name):
                with self.assertWarns(UserWarning):
                    normalized = gradmax_normalization(proposal, weight)
                self.assertIs(normalized, proposal)


class TestFixedProportionalBudgetMatchesV1(TorchTestCase):
    """fixed_proportional_budget matches number_of_neurons_to_add("fixed_proportional")."""

    CASES = ((4, 16, 12), (8, 32, 12), (16, 64, 12), (5, 16, 4), (3, 3, 2), (2, 9, 1))

    def test_linear_and_conv_consumers(self):
        for initial, target, steps in self.CASES:
            with self.subTest(initial=initial, target=target, steps=steps):
                linear = LinearGrowingModule(initial, 2, target_in_features=target)
                conv = RestrictedConv2dGrowingModule(
                    initial, 2, kernel_size=3, target_in_channels=target
                )
                budget = fixed_proportional_budget(initial, target, steps)
                self.assertEqual(
                    budget, linear.number_of_neurons_to_add(number_of_growth_steps=steps)
                )
                self.assertEqual(
                    budget, conv.number_of_neurons_to_add(number_of_growth_steps=steps)
                )

    def test_wrn_28_1_from_a_quarter_width(self):
        budgets = [fixed_proportional_budget(w // 4, w, 12) for w in (16, 32, 64)]
        self.assertEqual(budgets, [1, 2, 4])

    def test_non_positive_number_of_steps_raises(self):
        with self.assertRaises(ValueError):
            fixed_proportional_budget(4, 16, 0)


# Batch normalisation configurations compared with GrowingBatchNorm.
NORM_VARIANTS: dict[str, dict[str, bool]] = {
    "affine, running stats": {},
    "no affine": {"affine": False},
    "no running stats": {"track_running_stats": False},
}


class TestWidenNormMatchesV1(TorchTestCase):
    """widen_norm matches GrowingBatchNorm.grow with its default new values."""

    def build_pair(self, plain_class, v1_class, features: int, **kwargs):
        """Build a v1 and a plain normalisation with the same random state."""
        device = global_device()
        v1_norm = v1_class(features, device=device, **kwargs)
        with torch.no_grad():
            if v1_norm.affine:
                v1_norm.weight.uniform_(0.5, 1.5)
                v1_norm.bias.normal_()
            if v1_norm.track_running_stats:
                v1_norm.running_mean.normal_()
                v1_norm.running_var.uniform_(0.5, 2.0)
                v1_norm.num_batches_tracked.fill_(7)
        plain_norm = plain_class(features, device=device, **kwargs)
        plain_norm.load_state_dict(v1_norm.state_dict())
        return v1_norm, plain_norm

    def check_same_state(self, plain_norm: nn.Module, v1_norm: nn.Module) -> None:
        """Compare parameters, buffers and hyper-parameters."""
        plain_state, v1_state = plain_norm.state_dict(), v1_norm.state_dict()
        self.assertEqual(set(plain_state), set(v1_state))
        for key, value in v1_state.items():
            with self.subTest(entry=key):
                self.assertAllClose(plain_state[key], value, atol=0, rtol=0)
        for attribute in ("num_features", "eps", "momentum", "affine"):
            self.assertEqual(getattr(plain_norm, attribute), getattr(v1_norm, attribute))

    def test_batch_norm_2d(self):
        for name, kwargs in NORM_VARIANTS.items():
            with self.subTest(variant=name):
                v1_norm, plain_norm = self.build_pair(
                    nn.BatchNorm2d, GrowingBatchNorm2d, 4, **kwargs
                )
                v1_norm.grow(3)
                widened = widen_norm(plain_norm, 3)
                self.assertIs(type(widened), nn.BatchNorm2d)
                self.assertEqual(plain_norm.num_features, 4)
                self.check_same_state(widened, v1_norm)
                inputs = torch.randn(8, 7, 5, 5, device=global_device())
                for training in (False, True):
                    widened.train(training)
                    v1_norm.train(training)
                    with torch.no_grad():
                        self.assertAllClose(widened(inputs), v1_norm(inputs), atol=1e-6)
                self.check_same_state(widened, v1_norm)

    def test_batch_norm_1d(self):
        v1_norm, plain_norm = self.build_pair(nn.BatchNorm1d, GrowingBatchNorm1d, 4)
        v1_norm.grow(2)
        widened = widen_norm(plain_norm, 2)
        self.assertIs(type(widened), nn.BatchNorm1d)
        self.check_same_state(widened, v1_norm)

    @unittest.skipUnless(
        "bias" in inspect.signature(nn.BatchNorm2d.__init__).parameters,
        "affine batch normalisation without bias needs torch >= 2.13",
    )
    def test_affine_norm_without_bias_stays_without_bias(self):
        norm = nn.BatchNorm2d(4, bias=False)  # type: ignore[call-arg]
        widened = widen_norm(norm, 2)
        self.assertIsNone(widened.bias)
        self.assertAllClose(widened.weight[4:], torch.ones(2), atol=0, rtol=0)

    def test_zero_added_units_copy_the_norm(self):
        _, plain_norm = self.build_pair(nn.BatchNorm2d, GrowingBatchNorm2d, 4)
        widened = widen_norm(plain_norm, 0)
        self.assertIsNot(widened, plain_norm)
        self.check_same_state(widened, plain_norm)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(TypeError):
            widen_norm(nn.GroupNorm(2, 4), 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            widen_norm(nn.BatchNorm2d(4), -1)


class PlainBlock(nn.Module):
    """Pre-activation-free residual block: conv -> BN -> act -> conv, plus a 1x1 shortcut."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.act = ReLUDerivativeOneAtZero()
        self.conv2 = nn.Conv2d(4, 5, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(3, 5, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.conv2(self.act(self.bn(self.conv1(x))))


class TestGradmaxBlockGrowthMatchesV1(TorchTestCase):
    """A GradMax step on a plain block equals the v1 Conv2dGrowingBlock step.

    The v1 side reproduces experimental_grow: statistics in evaluation mode,
    ``compute_optimal_updates`` with the GradMax options, sub-selection at
    threshold 0, ``gradmax_normalization``, fixed scalings (1, 1) and
    ``apply_change``, which also grows the ``GrowingBatchNorm2d``.
    """

    ADDED_UNITS = 2

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def build(self, stride: int) -> tuple[Conv2dGrowingBlock, PlainBlock]:
        """Build a v1 block and its plain twin with the same parameters and BN state."""
        v1_block = Conv2dGrowingBlock(
            in_channels=3,
            out_channels=5,
            hidden_channels=4,
            target_hidden_channels=8,
            pre_activation=nn.Identity(),
            mid_activation=nn.Sequential(
                GrowingBatchNorm2d(4), ReLUDerivativeOneAtZero()
            ),
            kwargs_first_layer={
                "kernel_size": 3,
                "padding": 1,
                "stride": stride,
                "use_bias": False,
            },
            kwargs_second_layer={"kernel_size": 3, "padding": 1, "use_bias": False},
            downsample=nn.Conv2d(3, 5, 1, stride=stride, bias=False),
            device=self.device,
        )
        v1_norm = v1_block.first_layer.post_layer_function[0]
        with torch.no_grad():
            v1_norm.weight.uniform_(0.5, 1.5)
            v1_norm.bias.normal_()
            v1_norm.running_mean.normal_()
            v1_norm.running_var.uniform_(0.5, 2.0)
        plain = PlainBlock(stride).to(self.device)
        copy_parameters(v1_block.first_layer, plain.conv1)
        copy_parameters(v1_block.second_layer, plain.conv2)
        plain.bn.load_state_dict(v1_norm.state_dict())
        plain.shortcut.load_state_dict(v1_block.downsample.state_dict())
        return v1_block, plain

    def grow_both(self, stride: int):
        """Run one GradMax growth step with v1 and with lite on the same batch."""
        v1_block, plain = self.build(stride)
        # Statistics run in evaluation mode, as in compute_statistics: BatchNorm uses
        # (and keeps) its running statistics.
        v1_block.eval()
        plain.eval()
        inputs = torch.randn(6, 3, 8, 8, device=self.device)
        with torch.no_grad():
            targets = torch.randn_like(plain(inputs))

        v1_block.init_computation()
        ((v1_block(inputs) - targets) ** 2).sum().backward()
        v1_block.update_computation()
        v1_block.compute_optimal_updates(
            maximum_added_neurons=self.ADDED_UNITS, **V1_GRADMAX_KWARGS
        )
        v1_block.reset_computation()
        v1_block.set_scaling_factor(1.0)
        v1_block.sub_select_optimal_added_parameters(threshold=0)
        v1_block.normalize_optimal_updates(
            normalization_type="gradmax_normalization", gradmax_scale=1.0
        )
        v1_block.second_layer.optimal_delta_scaling = 0.0
        v1_block.second_layer.input_extension_scaling = 1.0
        v1_block.first_layer.output_extension_scaling = 1.0
        v1_block.apply_change(extension_size=self.ADDED_UNITS)
        v1_block.delete_update()

        site = Site(plain.conv1, plain.conv2)
        with collect([site]) as moments:
            ((plain(inputs) - targets) ** 2).sum().backward()
        proposal = gradmax_normalization(
            gradmax(
                moments[site],
                maximum_added_neurons=self.ADDED_UNITS,
                statistical_threshold=0,
            ),
            plain.conv2.weight,
        )
        grown = grown_copy(
            plain, "conv1", "conv2", proposal, gamma=1.0, norm_names=["bn"]
        )
        return v1_block, plain, grown

    def test_parameters_match_v1(self):
        for stride in (1, 2):
            with self.subTest(stride=stride):
                v1_block, _, grown = self.grow_both(stride)
                self.assertEqual(v1_block.hidden_neurons, 4 + self.ADDED_UNITS)
                self.assertAllClose(
                    grown.conv1.weight, v1_block.first_layer.weight, atol=0
                )
                self.assertAllClose(
                    grown.conv2.weight[:, :4], v1_block.second_layer.weight[:, :4], atol=0
                )
                new_fan_out = grown.conv2.weight[:, 4:]
                v1_new_fan_out = v1_block.second_layer.weight[:, 4:]
                assert_same_columns_up_to_sign(
                    self, new_fan_out[:, :, 1, 1], v1_new_fan_out[:, :, 1, 1], atol=1e-5
                )
                self.assertAllClose(
                    new_fan_out.flatten(2)[:, :, [0, 1, 2, 3, 5, 6, 7, 8]],
                    v1_new_fan_out.flatten(2)[:, :, [0, 1, 2, 3, 5, 6, 7, 8]],
                    atol=0,
                )
                v1_norm = v1_block.first_layer.post_layer_function[0]
                for key, value in v1_norm.state_dict().items():
                    self.assertAllClose(grown.bn.state_dict()[key], value, atol=0, rtol=0)

    def test_grown_block_matches_v1_forward_and_backward(self):
        for stride in (1, 2):
            with self.subTest(stride=stride):
                v1_block, plain, grown = self.grow_both(stride)
                inputs = torch.randn(4, 3, 8, 8, device=self.device)
                for training in (False, True):
                    v1_block.zero_grad()
                    grown.zero_grad()
                    v1_block.train(training)
                    grown.train(training)
                    v1_output = v1_block(inputs)
                    output = grown(inputs)
                    self.assertAllClose(output, v1_output, atol=1e-5)
                    with torch.no_grad():
                        self.assertAllClose(
                            output, plain.train(training)(inputs), atol=1e-5
                        )
                    output.pow(2).sum().backward()
                    v1_output.pow(2).sum().backward()
                    # The new fan-in learns through sigma'(0) = 1, with the sign of omega.
                    sign = torch.sign(
                        (grown.conv2.weight[:, 4:] * v1_block.second_layer.weight[:, 4:])
                        .flatten(2)
                        .sum(dim=(0, 2))
                    )
                    # In training mode BatchNorm divides the zero channels by sqrt(eps),
                    # so the tolerance follows the gradient scale.
                    v1_gradient = v1_block.first_layer.weight.grad[4:]
                    self.assertAllClose(
                        grown.conv1.weight.grad[4:] * sign[:, None, None, None],
                        v1_gradient,
                        atol=1e-5 * v1_gradient.abs().max().item(),
                    )
                    self.assertGreater(
                        grown.conv1.weight.grad[4:].abs().max().item(), 0.0
                    )

    def test_norm_names_must_match_the_hidden_dimension(self):
        _, plain = self.build(stride=1)
        proposal = Proposal(None, torch.zeros(1, 27), torch.ones(5, 1), torch.ones(1), 0)
        with self.assertRaises(ValueError):
            grown_copy(
                plain, "conv1", "conv2", proposal, gamma=1.0, norm_names=["shortcut"]
            )


if __name__ == "__main__":
    unittest.main()
