"""Full addition loops of a residual block in gromo.lite, ported from ``test_growing_block``.

The scenarios of ``test_full_addition_loop_with_indicator_batch`` and
``test_full_addition_loop_with_features_identity_initialization`` on a plain
residual block ``x + fc2(fc1(x))`` (identity activation), with the loss
``||output||**2 / 2`` on an indicator batch, whose answers are known exactly:

- starting from 0 hidden units, TINY's new units cancel the identity shortcut,
  so the grown block outputs zero;
- starting from ``fc1 = I`` and ``fc2 = 0``, the consumer update is exactly the
  identity and no new unit can help (singular values about 0).

gromo.lite has no pending update to apply or delete: the proposal is installed by
``grown_copy``, which leaves the original block untouched. Each scenario is also
run on v1's ``LinearGrowingBlock`` with the same weights and data, and both
libraries must agree.
"""

import torch
from torch import nn

from gromo.containers.growing_block import LinearGrowingBlock
from gromo.lite import Site, collect, first_order_decrease, grown_copy, tiny
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import TorchTestCase, indicator_batch
except ImportError:
    from torch_unittest import TorchTestCase, indicator_batch


class ResidualBlock(nn.Module):
    """``x + fc2(fc1(x))``: a residual block with an identity activation."""

    def __init__(self, features: int, hidden: int, bias: bool, device: torch.device):
        super().__init__()
        self.fc1 = nn.Linear(features, hidden, bias=bias, device=device)
        self.fc2 = nn.Linear(hidden, features, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual output."""
        return x + self.fc2(self.fc1(x))


def half_squared_norm(output: torch.Tensor) -> torch.Tensor:
    """The loss of the v1 tests, ``||output||**2 / 2`` summed over the batch."""
    return (output**2).sum() / 2


class TestLiteFullAdditionLoop(TorchTestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = global_device()
        self.features = 3
        self.batch = indicator_batch((self.features,), device=self.device)

    def lite_moments(self, block: ResidualBlock):
        """Collect the moments of the site ``fc1 -> fc2`` on the indicator batch."""
        site = Site(block.fc1, block.fc2)
        with collect([site]) as moments:
            half_squared_norm(block(self.batch)).backward()
        return moments[site]

    def v1_block(self, hidden: int, bias: bool, identity: bool) -> LinearGrowingBlock:
        """v1's block in the same state, after its statistics pass and solve."""
        block = LinearGrowingBlock(
            in_features=self.features,
            out_features=self.features,
            hidden_features=hidden,
            activation=nn.Identity(),
            device=self.device,
            name="v1_block",
            kwargs_layer={"use_bias": bias},
        )
        if identity:
            with torch.no_grad():
                block.first_layer.weight.data = torch.eye(
                    self.features, device=self.device
                )
                block.second_layer.weight.data.zero_()
                for layer in (block.first_layer, block.second_layer):
                    if layer.use_bias:
                        layer.bias.data.zero_()
        block.init_computation()
        block.zero_grad()
        half_squared_norm(block(self.batch)).backward()
        block.update_computation()
        block.compute_optimal_updates(maximum_added_neurons=self.features)
        block.reset_computation()
        return block

    def test_full_addition_loop_with_indicator_batch(self) -> None:
        block = ResidualBlock(self.features, 0, bias=False, device=self.device)
        moments = self.lite_moments(block)
        proposal = tiny(moments, maximum_added_neurons=self.features)

        # no weight to update in an empty block
        self.assertAlmostEqual(proposal.parameter_update_decrease.item(), 0.0, places=5)
        # the new units learn the negative identity: the grown block outputs zero
        grown = grown_copy(block, "fc1", "fc2", proposal, 1.0)
        with torch.no_grad():
            output = grown(self.batch)
        self.assertShapeEqual(output, self.batch.shape)
        self.assertAllClose(output, torch.zeros_like(self.batch), atol=1e-4)
        decrease = first_order_decrease(moments, proposal)
        self.assertGreaterEqual(decrease.item(), 0)

        # keeping one unit decreases the loss by no more than keeping all of them
        kept = tiny(moments, maximum_added_neurons=1)
        self.assertEqual(kept.eigenvalues.shape[0], 1)
        self.assertLessEqual(first_order_decrease(moments, kept).item(), decrease.item())

        # installing it: the block had no unit, so its weights are the proposal's
        grown_one = grown_copy(block, "fc1", "fc2", kept, 1.0)
        self.assertEqual(grown_one.fc1.out_features, 1)
        self.assertEqual(grown_one.fc2.in_features, 1)
        self.assertAllClose(grown_one.fc1.weight, kept.alpha)
        self.assertAllClose(grown_one.fc2.weight, kept.omega)
        with torch.no_grad():
            grown_one(self.batch)
        # the original block is untouched: there is no pending update to delete
        self.assertEqual(block.fc1.out_features, 0)
        self.assertEqual(block.fc2.in_features, 0)

        # v1's block agrees on the same scenario
        v1 = self.v1_block(hidden=0, bias=False, identity=False)
        self.assertAllClose(proposal.eigenvalues, v1.eigenvalues_extension, atol=1e-5)
        self.assertAllClose(decrease, v1.first_order_improvement, atol=1e-5)
        v1.sub_select_optimal_added_parameters(1)
        self.assertAllClose(
            first_order_decrease(moments, kept), v1.first_order_improvement, atol=1e-5
        )

    def test_full_addition_loop_with_features_identity_initialization(self) -> None:
        for bias in (False, True):
            with self.subTest(bias=bias):
                block = ResidualBlock(
                    self.features, self.features, bias=bias, device=self.device
                )
                with torch.no_grad():
                    block.fc1.weight.copy_(torch.eye(self.features))
                    block.fc2.weight.zero_()
                    if bias:
                        block.fc1.bias.zero_()
                        block.fc2.bias.zero_()
                x_test = torch.randn(4, self.features, device=self.device)
                with torch.no_grad():
                    self.assertAllClose(block(x_test), x_test, atol=1e-6)

                moments = self.lite_moments(block)
                with torch.no_grad():
                    loss = half_squared_norm(block(self.batch))
                proposal = tiny(moments, maximum_added_neurons=self.features)
                assert proposal.optimal_delta is not None
                if not bias:
                    # the update removes the identity: W2 -= I
                    self.assertAllClose(
                        proposal.optimal_delta,
                        torch.eye(self.features, device=self.device),
                        atol=1e-5,
                    )
                # no new unit can improve an already optimal block
                self.assertTrue(
                    torch.all(proposal.eigenvalues.abs() < 1e-3),
                    f"eigenvalues should be about 0, got {proposal.eigenvalues}",
                )
                self.assertAlmostEqual(
                    proposal.parameter_update_decrease.item(),
                    2 * loss.item() / self.batch.shape[0],
                    places=3,
                )
                decrease = first_order_decrease(moments, proposal)
                self.assertAlmostEqual(decrease.item(), 1.0, places=5)
                # the grown block is optimal: it outputs zero
                with torch.no_grad():
                    output = grown_copy(block, "fc1", "fc2", proposal, 1.0)(self.batch)
                self.assertAllClose(output, torch.zeros_like(self.batch), atol=1e-5)

                # v1's block agrees on the same scenario
                v1 = self.v1_block(hidden=self.features, bias=bias, identity=True)
                v1_delta = v1.second_layer.optimal_delta_layer
                assert isinstance(v1_delta, nn.Linear)
                v1_rows = (
                    torch.cat([v1_delta.weight, v1_delta.bias[:, None]], dim=1)
                    if bias
                    else v1_delta.weight
                )
                self.assertAllClose(proposal.optimal_delta, v1_rows, atol=1e-5)
                self.assertAllClose(
                    proposal.parameter_update_decrease,
                    v1.parameter_update_decrease,
                    atol=1e-5,
                )
                self.assertAllClose(decrease, v1.first_order_improvement, atol=1e-5)
