"""Tests of the gromo.lite helpers shared by growth methods.

The extra-moment extension point of ``collect`` is checked against autograd.
"""

import torch
from torch import nn

from gromo.lite import Site, collect


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


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
