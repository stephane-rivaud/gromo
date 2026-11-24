"""Edge cases of gromo.lite found by external reviews, one test per fixed behaviour.

Each test pins a case that failed before the fix: zero-width sites with a bias,
gradient hooks outliving ``collect``, in-place activations, keyword calls, empty
batches, mixed precision, aliasing in ``Moments``, zero-width random proposals,
the zero-spread ``copy_normal`` fallback, per-layer dtypes and module state in
surgery, and the validation of splits.
"""

import torch
from torch import nn

from gromo.lite import (
    Moments,
    Site,
    collect,
    gradmax_normalization,
    grown_copy,
    random_proposal,
    split_copy,
    split_units,
    splitting,
    splitting_moment,
    tiny,
    widen_linear,
    widen_norm,
)


try:
    from tests.torch_unittest import TorchTestCase, indicator_batch
except ImportError:
    from torch_unittest import TorchTestCase, indicator_batch


def summed_squared_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error summed over the batch and the outputs."""
    return ((output - target) ** 2).sum()


class ResidualBlock(nn.Module):
    """``x + fc2(fc1(x))`` with optional biases."""

    def __init__(self, features: int, hidden: int, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(features, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual output."""
        return x + self.fc2(self.fc1(x))


class KeywordMLP(nn.Module):
    """An MLP whose layers are called with ``input=`` instead of positionally."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keyword calls of both layers."""
        return self.fc2(input=torch.tanh(self.fc1(input=x)))


class TestCaptureEdgeCases(TorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.network = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
        self.inputs, self.targets = torch.randn(8, 4), torch.randn(8, 2)

    def collect_site(self, network: nn.Module, inputs: torch.Tensor) -> Moments:
        """Moments of the site ``network[0] -> network[2]`` for one batch."""
        site = Site(network[0], network[2])
        with collect([site]) as moments:
            summed_squared_error(network(inputs), self.targets[: len(inputs)]).backward()
        return moments[site]

    def test_zero_width_site_with_bias_keeps_its_bias_statistics(self) -> None:
        block = ResidualBlock(4, 0, bias=True)
        site = Site(block.fc1, block.fc2)
        batch = indicator_batch((4,))
        with collect([site]) as moments:
            ((block(batch) ** 2).sum() / 2).backward()
        # the consumer input is only its bias column of ones
        self.assertAllClose(moments[site]["tensor_s"], torch.ones(1, 1))
        proposal = tiny(moments[site], maximum_added_neurons=4)
        grown = grown_copy(block, "fc1", "fc2", proposal, 1.0)
        # the consumer bias update removes the mean, the new units the rest
        self.assertGreater(grown.fc1.out_features, 0)
        with torch.no_grad():
            self.assertAllClose(grown(batch), torch.zeros_like(batch), atol=1e-4)

    def test_gradient_hooks_are_removed_on_exit(self) -> None:
        site = Site(self.network[0], self.network[2])
        with collect([site]) as moments:
            loss = summed_squared_error(self.network(self.inputs), self.targets)
            loss.backward(retain_graph=True)
        samples = moments[site].samples
        loss.backward()  # after the block: must not be collected
        self.assertEqual(moments[site].samples, samples)

    def test_inplace_activation_does_not_alter_the_cached_output(self) -> None:
        def matrices(inplace: bool) -> torch.Tensor:
            torch.manual_seed(1)
            network = nn.Sequential(
                nn.Linear(4, 3), nn.ELU(inplace=inplace), nn.Linear(3, 2)
            ).double()
            site = Site(network[0], network[2])
            moment = splitting_moment(nn.functional.elu)
            with collect([site], extra_moments=[moment]) as moments:
                summed_squared_error(
                    network(self.inputs.double()), self.targets.double()
                ).backward()
            return moments[site]["splitting_matrices"]

        self.assertAllClose(matrices(True), matrices(False), atol=1e-12)

    def test_keyword_argument_calls_are_collected(self) -> None:
        network = KeywordMLP()
        site = Site(network.fc1, network.fc2)
        with collect([site]) as moments:
            summed_squared_error(network(self.inputs), self.targets).backward()
        positional = nn.Sequential(network.fc1, nn.Tanh(), network.fc2)
        expected = self.collect_site(positional, self.inputs)
        for name in ("tensor_s", "tensor_m", "tensor_m_prev"):
            self.assertAllClose(moments[site][name], expected[name])

    def test_empty_batch_adds_nothing_and_is_not_averaged(self) -> None:
        moments = self.collect_site(self.network, self.inputs[:0])
        self.assertEqual(moments.samples, 0)
        with self.assertRaises(ValueError):
            moments["tensor_s"]
        # one more batch: the empty one did not change the averages
        convolutional = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1), nn.Tanh(), nn.Conv2d(3, 2, 3, padding=1)
        )
        site = Site(convolutional[0], convolutional[2])
        with collect([site]) as conv_moments:
            convolutional(torch.randn(0, 2, 5, 5)).sum().backward()
        self.assertEqual(conv_moments[site].samples, 0)

    def test_add_sum_does_not_alias_the_caller_tensor(self) -> None:
        moments = Moments()
        update = torch.ones(2)
        moments.add_sum("count", update)
        moments.add_sum("count", update)
        self.assertAllClose(update, torch.ones(2))
        moments.samples = 2
        self.assertAllClose(moments["count"], torch.ones(2))

    def test_mixed_precision_is_accumulated_in_float32(self) -> None:
        network = nn.Sequential(
            nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)
        )
        site = Site(network[2], network[4])
        moment = splitting_moment(torch.tanh)
        with collect([site], extra_moments=[moment]) as moments:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = network(self.inputs)
            summed_squared_error(output.float(), self.targets).backward()
        for name in ("tensor_s", "tensor_m_prev", "splitting_matrices"):
            self.assertEqual(moments[site][name].dtype, torch.float32)
        self.assertEqual(tiny(moments[site]).alpha.dtype, torch.float32)

    def test_several_sites_match_separate_collections(self) -> None:
        network = nn.Sequential(
            nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 2)
        )
        first, second = Site(network[0], network[2]), Site(network[2], network[4])
        with collect(iter([first, second])) as together:
            summed_squared_error(network(self.inputs), self.targets).backward()
        for site in (first, second):
            with collect([site]) as alone:
                summed_squared_error(network(self.inputs), self.targets).backward()
            for name in ("tensor_s", "tensor_m", "tensor_s_growth", "tensor_m_prev"):
                self.assertAllClose(together[site][name], alone[site][name])


class TestSolveEdgeCases(TorchTestCase):
    def test_random_proposal_grows_a_zero_width_site(self) -> None:
        producer, consumer = nn.Linear(4, 0), nn.Linear(0, 3)
        proposal = random_proposal(
            producer, consumer, 2, "kaiming_normal", "kaiming_normal"
        )
        self.assertEqual(tuple(proposal.alpha.shape), (2, 5))
        self.assertEqual(tuple(proposal.omega.shape), (3, 2))
        with self.assertRaises(ValueError):
            random_proposal(nn.Linear(4, 3), nn.Linear(3, 2), -1)

    def test_copy_normal_falls_back_to_kaiming_without_spread(self) -> None:
        producer = nn.Linear(50, 4, bias=False)
        consumer = nn.Linear(4, 400, bias=False)
        with torch.no_grad():
            consumer.weight.zero_()
        proposal = random_proposal(producer, consumer, 200, "copy_normal", "copy_normal")
        expected = (2.0 / (4 + 200)) ** 0.5  # v1's Kaiming fallback
        self.assertAlmostEqual(
            proposal.omega.std().item(), expected, delta=0.05 * expected
        )

    def test_gradmax_normalization_needs_a_finite_scale(self) -> None:
        proposal = random_proposal(
            nn.Linear(4, 3), nn.Linear(3, 2), 2, "zeros", "kaiming_normal"
        )
        with self.assertRaises(ValueError):
            gradmax_normalization(proposal, torch.ones(2, 3), scale=float("nan"))


class TestSurgeryEdgeCases(TorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.network = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
        site = Site(self.network[0], self.network[2])
        with collect([site]) as moments:
            summed_squared_error(
                self.network(torch.randn(16, 4)), torch.randn(16, 2)
            ).backward()
        self.proposal = tiny(
            moments[site], maximum_added_neurons=2, statistical_threshold=0
        )

    def test_each_layer_keeps_its_dtype(self) -> None:
        producer, consumer = nn.Linear(4, 3), nn.Linear(3, 2).double()
        new_producer, new_consumer = widen_linear(producer, consumer, self.proposal, 0.5)
        self.assertEqual(new_producer.weight.dtype, torch.float32)
        self.assertEqual(new_consumer.weight.dtype, torch.float64)

    def test_grown_layers_keep_mode_and_gradient_flags(self) -> None:
        network = self.network.eval()
        network[2].weight.requires_grad_(False)
        grown = grown_copy(network, "0", "2", self.proposal, 0.5)
        self.assertFalse(grown[0].training or grown[2].training)
        self.assertFalse(grown[2].weight.requires_grad)
        self.assertTrue(grown[0].weight.requires_grad)
        norm = nn.BatchNorm3d(3).eval()
        norm.weight.requires_grad_(False)
        widened = widen_norm(norm, 2)
        self.assertEqual(widened.num_features, 5)
        self.assertFalse(widened.training or widened.weight.requires_grad)

    def test_a_single_norm_name_is_one_name(self) -> None:
        network = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            nn.Conv2d(3, 2, 3, padding=1),
        )
        site = Site(network[0], network[3])
        with collect([site]) as moments:
            network(torch.randn(4, 2, 5, 5)).pow(2).sum().backward()
        proposal = tiny(moments[site], maximum_added_neurons=1, statistical_threshold=0)
        grown = grown_copy(network, "0", "3", proposal, 0.0, norm_names="1")
        self.assertEqual(grown[1].num_features, 3 + proposal.alpha.shape[0])


class TestSplittingEdgeCases(TorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.network = nn.Sequential(
            nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Tanh(), nn.Linear(6, 3)
        ).double()
        with torch.no_grad():
            self.network[1].running_mean.copy_(torch.randn(6, dtype=torch.float64))
            self.network[1].weight.copy_(torch.randn(6, dtype=torch.float64))
        self.network.eval()
        self.inputs = torch.randn(32, 4, dtype=torch.float64)
        self.targets = torch.randn(32, 3, dtype=torch.float64)

    def matrices(self, norm_key: Site) -> torch.Tensor:
        """Splitting matrices with the norm registered under ``norm_key``."""
        site = Site(self.network[0], self.network[3])
        moment = splitting_moment(torch.tanh, norms={norm_key: self.network[1]})
        with collect([site], extra_moments=[moment]) as moments:
            summed_squared_error(self.network(self.inputs), self.targets).backward()
        return moments[site]["splitting_matrices"]

    def test_norms_are_matched_by_the_site_layers(self) -> None:
        equal_site = Site(self.network[0], self.network[3])
        other_layers = Site(nn.Linear(4, 6), nn.Linear(6, 3))
        folded, unfolded = self.matrices(equal_site), self.matrices(other_layers)
        self.assertGreater((folded - unfolded).abs().max().item(), 1e-3)
        # what the fold computes: the per-unit affine map of the norm in eval mode
        norm = self.network[1]
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        self.assertGreater(scale.abs().max().item(), 0)

    def test_zero_width_site_has_nothing_to_split(self) -> None:
        network = nn.Sequential(nn.Linear(4, 0), nn.Tanh(), nn.Linear(0, 3))
        site = Site(network[0], network[2])
        with collect([site], extra_moments=[splitting_moment(torch.tanh)]) as moments:
            summed_squared_error(
                network(self.inputs.float()), self.targets.float()
            ).backward()
        self.assertEqual(tuple(moments[site]["splitting_matrices"].shape), (0, 5, 5))
        self.assertEqual(splitting(moments[site]).units.numel(), 0)

    def test_invalid_splits_are_refused(self) -> None:
        network = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).double()
        site = Site(network[0], network[2])
        with collect([site], extra_moments=[splitting_moment(torch.tanh)]) as moments:
            summed_squared_error(network(self.inputs), self.targets).backward()
        with self.assertRaises(ValueError):
            splitting(moments[site], maximum_splits=-1)
        proposal = splitting(moments[site], maximum_splits=1)
        repeated = type(proposal)(
            proposal.units.repeat(2),
            proposal.directions.repeat(2, 1),
            proposal.eigenvalues.repeat(2),
        )
        with self.assertRaises(ValueError):
            split_units(network[0], network[2], repeated, 0.0)
        reflect = nn.Conv2d(2, 3, 3, padding=1, padding_mode="reflect")
        with self.assertRaises(NotImplementedError):
            split_units(reflect, nn.Conv2d(3, 2, 3, padding=1), proposal, 0.0)
        wrong_norm = nn.Sequential(
            nn.Linear(4, 6), nn.BatchNorm1d(5), nn.Tanh(), nn.Linear(6, 3)
        ).double()
        with self.assertRaises(ValueError):
            split_copy(wrong_norm, "0", "3", proposal, 0.0, norm_names="1")
