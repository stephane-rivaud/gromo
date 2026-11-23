r"""Natural-gradient invariance of gromo.lite, ported from ``test_natural_gradient_invariance``.

The setups are v1's, built by its own helpers: a purely linear chain of four
layers (``nn.Linear``) and a chain of four 3x3 convolutions (restricted scheme),
without biases. Rescaling ``W_1 <- c W_1`` and ``W_4 <- W_4 / c`` preserves the
network function. On the site layer 2 -> layer 3 the statistics then scale as

    tensor_s_growth ~ c**2   tensor_m_prev ~ 1   tensor_s ~ c**2   tensor_m ~ 1
    covariance_loss_gradient ~ 1 / c**2

so the consumer update scales as ``1 / c**2`` without the empirical Fisher and
is invariant with it, and the singular values of the new units are invariant
without covariance, scale as ``1 / c`` with covariance alone, and are invariant
with covariance and Fisher.

gromo.lite runs on plain ``torch.nn`` copies of v1's layers (same weights, same
data), and every check also compares gromo.lite with v1 on the same scenario,
so both libraries must satisfy the invariances and agree.

In the linear chain the gradient at layer 3 is ``W_4^T (y - t)``, so the
empirical Fisher has rank 2 of 4. Solving with it fails, and both libraries fall
back to the float64 pseudo-inverse with a warning ("The parameter update
decrease should be positive"): the linear Fisher checks exercise that fallback.
The convolutional chain has a full-rank Fisher.
"""

import torch
from torch import nn

from gromo.lite import Moments, Site, collect, natural_gradient, optimal_updates


try:
    from tests import test_natural_gradient_invariance as v1_scenarios
    from tests.torch_unittest import TorchTestCase
except ImportError:
    import test_natural_gradient_invariance as v1_scenarios
    from torch_unittest import TorchTestCase


MOMENT_NAMES = (
    "tensor_s_growth",
    "tensor_m_prev",
    "tensor_s",
    "tensor_m",
    "covariance_loss_gradient",
)
# power of c by which each statistic scales under the rescaling
MOMENT_SCALING = {
    "tensor_s_growth": 2,
    "tensor_m_prev": 0,
    "tensor_s": 2,
    "tensor_m": 0,
    "covariance_loss_gradient": -2,
}
# (use_covariance, use_fisher) -> power of c of the singular values of new units
SINGULAR_VALUE_SCALING = {(False, False): 0, (True, False): -1, (True, True): 0}


def plain_copy(layer: nn.Module) -> nn.Module:
    """A plain ``nn.Linear`` or ``nn.Conv2d`` with the weights of a v1 module.

    Parameters
    ----------
    layer : nn.Module
        ``nn.Linear``, ``nn.Conv2d`` or a v1 growing module wrapping one

    Returns
    -------
    nn.Module
        independent plain layer computing the same function
    """
    inner = getattr(layer, "layer", layer)
    copy = (
        nn.Linear(inner.in_features, inner.out_features, bias=inner.bias is not None)
        if isinstance(inner, nn.Linear)
        else nn.Conv2d(
            inner.in_channels,
            inner.out_channels,
            inner.kernel_size,
            stride=inner.stride,
            padding=inner.padding,
            bias=inner.bias is not None,
        )
    )
    copy = copy.to(inner.weight.device)
    with torch.no_grad():
        copy.weight.copy_(inner.weight)
        if inner.bias is not None:
            copy.bias.copy_(inner.bias)
    return copy


class InvarianceChecks:
    """Checks shared by the linear and the convolutional chain.

    A subclass sets ``v1_case``, an instance of v1's test case whose
    ``_build_net(c)`` returns ``(first, producer, consumer, last, x, target)``
    with the rescaling applied.
    """

    v1_case: TorchTestCase
    SCALE = 2.0

    def lite_moments(self, c: float) -> Moments:
        """Collect gromo.lite's moments on plain copies of v1's network.

        Parameters
        ----------
        c : float
            rescaling factor

        Returns
        -------
        Moments
            moments of the site producer -> consumer
        """
        first, producer, consumer, last, x, target = self.v1_case._build_net(c)
        network = nn.Sequential(
            *(plain_copy(layer) for layer in (first, producer, consumer, last))
        )
        site = Site(network[1], network[2])
        with collect([site]) as moments:
            (0.5 * ((network(x) - target) ** 2).sum()).backward()
        return moments[site]

    def v1_consumer(self, c: float) -> nn.Module:
        """Run v1's statistics pass and return its consumer module.

        Parameters
        ----------
        c : float
            rescaling factor

        Returns
        -------
        nn.Module
            v1 growing module holding the statistics of the site
        """
        layers = self.v1_case._build_net(c)
        self.v1_case._collect_stats(*layers)
        return layers[2]

    def test_statistics_scale_as_predicted_and_match_v1(self) -> None:
        references = self.lite_moments(1.0)
        for c in (1.0, self.SCALE):
            moments = self.lite_moments(c)
            consumer = self.v1_consumer(c)
            for name in MOMENT_NAMES:
                with self.subTest(c=c, moment=name):
                    self.assertAllClose(
                        moments[name], getattr(consumer, name)(), atol=1e-5, rtol=1e-4
                    )
                    self.assertAllClose(
                        moments[name],
                        references[name] * c ** MOMENT_SCALING[name],
                        atol=1e-5,
                        rtol=1e-4,
                    )

    def test_natural_gradient_is_invariant_with_the_fisher(self) -> None:
        c = self.SCALE
        reference, rescaled = self.lite_moments(1.0), self.lite_moments(c)
        plain_reference, _ = natural_gradient(reference)
        plain_rescaled, _ = natural_gradient(rescaled)
        fisher_reference, _ = natural_gradient(reference, use_fisher=True)
        fisher_rescaled, _ = natural_gradient(rescaled, use_fisher=True)
        self.assertGreater(plain_reference.norm().item(), 1e-3)
        self.assertGreater(fisher_reference.norm().item(), 1e-3)
        self.assertAllClose(plain_rescaled, plain_reference / c**2, atol=1e-5, rtol=1e-4)
        self.assertGreater((plain_rescaled - plain_reference).abs().max().item(), 1e-3)
        self.assertAllClose(fisher_rescaled, fisher_reference, atol=1e-5, rtol=1e-4)
        for scale in (1.0, c):
            consumer = self.v1_consumer(scale)
            moments = self.lite_moments(scale)
            for use_fisher in (False, True):
                with self.subTest(c=scale, use_fisher=use_fisher):
                    v1_delta, _, _ = consumer.compute_optimal_delta(
                        use_fisher=use_fisher, update=False
                    )
                    lite_delta, _ = natural_gradient(moments, use_fisher=use_fisher)
                    self.assertAllClose(
                        lite_delta.reshape(v1_delta.shape), v1_delta, atol=1e-5, rtol=1e-4
                    )

    def test_new_unit_singular_values_scale_as_predicted_and_match_v1(self) -> None:
        c = self.SCALE
        for (use_covariance, use_fisher), power in SINGULAR_VALUE_SCALING.items():
            with self.subTest(use_covariance=use_covariance, use_fisher=use_fisher):
                values = {}
                for scale in (1.0, c):
                    proposal = optimal_updates(
                        self.lite_moments(scale),
                        compute_delta=False,
                        use_projection=False,
                        use_covariance=use_covariance,
                        use_fisher=use_fisher,
                    )
                    consumer = self.v1_consumer(scale)
                    consumer.compute_optimal_updates(
                        compute_delta=False,
                        use_covariance=use_covariance,
                        use_fisher=use_fisher,
                        use_projection=False,
                    )
                    v1_values = consumer.eigenvalues_extension.sort(
                        descending=True
                    ).values
                    values[scale] = proposal.eigenvalues.sort(descending=True).values
                    self.assertAllClose(values[scale], v1_values, atol=1e-5, rtol=1e-4)
                self.assertGreater(values[1.0].max().item(), 1e-3)
                self.assertAllClose(
                    values[c], values[1.0] * c**power, atol=1e-5, rtol=1e-4
                )
                if power != 0:
                    self.assertGreater((values[c] - values[1.0]).abs().max().item(), 1e-3)


class TestLiteNaturalGradientInvarianceLinear(InvarianceChecks, TorchTestCase):
    """The purely linear chain ``Linear -> Linear -> Linear -> Linear``."""

    def setUp(self) -> None:
        self.v1_case = v1_scenarios.TestNaturalGradientInvariance()


class TestLiteNaturalGradientInvarianceRestrictedConv2d(InvarianceChecks, TorchTestCase):
    """The chain of four 3x3 convolutions, restricted scheme on the site."""

    def setUp(self) -> None:
        self.v1_case = v1_scenarios.TestNaturalGradientInvarianceRestrictedConv2d()


class TestFisherOptions(TorchTestCase):
    """Edge cases of the Fisher options."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        network = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        site = Site(network[0], network[2])
        with collect([site]) as moments:
            ((network(torch.randn(32, 3)) - torch.randn(32, 2)) ** 2).sum().backward()
        self.moments = moments[site]

    def test_full_shrinkage_whitens_by_a_multiple_of_the_identity(self) -> None:
        plain, _ = natural_gradient(self.moments)
        shrunk, _ = natural_gradient(self.moments, use_fisher=True, fisher_shrinkage=1.0)
        scale = self.moments["covariance_loss_gradient"].trace() / 2
        self.assertAllClose(shrunk, plain / scale, atol=1e-6, rtol=1e-5)

    def test_invalid_options_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            natural_gradient(self.moments, use_fisher=True, fisher_shrinkage=1.5)
        with self.assertRaises(ValueError):
            optimal_updates(self.moments, compute_delta=False, use_projection=True)

    def test_tiny_defaults_are_optimal_updates_defaults(self) -> None:
        from gromo.lite import tiny

        expected = optimal_updates(self.moments, use_fisher=True)
        proposal = tiny(self.moments, use_fisher=True)
        for name in ("optimal_delta", "alpha", "omega", "eigenvalues"):
            self.assertAllClose(getattr(proposal, name), getattr(expected, name))
