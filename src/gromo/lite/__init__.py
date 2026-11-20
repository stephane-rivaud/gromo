"""Growth on plain ``torch.nn`` models, without ``GrowingModule`` subclasses.

``gromo.lite`` separates four steps that ``GrowingModule`` combines:

1. capture: :func:`collect` attaches temporary hooks to a :class:`Site`; a
   method can add its own statistics through ``extra_moments``;
2. moments: :class:`Moments` holds running averages of outer products;
3. solve: :func:`tiny` and :func:`gradmax` are pure functions of the moments,
   :func:`gradmax_normalization` rescales a GradMax proposal and
   :func:`random_proposal` draws new units without statistics;
4. surgery: :func:`widen`, :func:`widen_norm` and :func:`grown_copy` return new
   modules.

:mod:`gromo.lite.splitting` adds splitting steepest descent with one module:
an extra statistic, a solver and a surgery that splits existing units.

Example::

    site = Site(model.fc1, model.fc2)
    with collect([site]) as moments:
        for inputs, targets in batches:
            loss_sum(model(inputs), targets).backward()
    proposal = tiny(moments[site], maximum_added_neurons=4)
    model = grown_copy(model, "fc1", "fc2", proposal, gamma=0.5)

Status: prototype. Sites are ``nn.Linear -> nn.Linear`` or
``nn.Conv2d -> nn.Conv2d``; convolution sites follow the restricted scheme of
``RestrictedConv2dGrowingModule`` (``1 x 1`` fan-out for new channels).
``tests/test_lite.py`` checks the results against v1.
"""

from gromo.lite.capture import (
    ExtraMoment,
    Moments,
    Site,
    SiteBatch,
    aligned_producer_patches,
    centre_tap,
    collect,
    input_patches,
    output_rows,
)
from gromo.lite.solve import (
    Proposal,
    first_order_decrease,
    fixed_proportional_budget,
    gradmax,
    gradmax_normalization,
    random_proposal,
    tensor_n,
    tiny,
)
from gromo.lite.splitting import (
    SplitProposal,
    activation_second_derivative,
    split_copy,
    split_units,
    splitting,
    splitting_moment,
)
from gromo.lite.surgery import (
    grown_copy,
    widen,
    widen_conv2d,
    widen_linear,
    widen_norm,
)


__all__ = [
    "ExtraMoment",
    "Moments",
    "Proposal",
    "Site",
    "SiteBatch",
    "SplitProposal",
    "activation_second_derivative",
    "aligned_producer_patches",
    "centre_tap",
    "collect",
    "first_order_decrease",
    "fixed_proportional_budget",
    "gradmax",
    "gradmax_normalization",
    "grown_copy",
    "input_patches",
    "output_rows",
    "random_proposal",
    "split_copy",
    "split_units",
    "splitting",
    "splitting_moment",
    "tensor_n",
    "tiny",
    "widen",
    "widen_conv2d",
    "widen_linear",
    "widen_norm",
]
