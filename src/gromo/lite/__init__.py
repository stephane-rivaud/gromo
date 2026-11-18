"""Growth on plain ``torch.nn`` models, without ``GrowingModule`` subclasses.

``gromo.lite`` separates four steps that ``GrowingModule`` combines:

1. capture: :func:`collect` attaches temporary hooks to a :class:`Site`;
2. moments: :class:`Moments` holds running averages of outer products;
3. solve: :func:`tiny` and :func:`gradmax` are pure functions of the moments;
4. surgery: :func:`widen` and :func:`grown_copy` return new modules.

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
    Moments,
    Site,
    aligned_producer_patches,
    centre_tap,
    collect,
    input_patches,
    output_rows,
)
from gromo.lite.solve import Proposal, gradmax, tensor_n, tiny
from gromo.lite.surgery import grown_copy, widen, widen_conv2d, widen_linear


__all__ = [
    "Moments",
    "Proposal",
    "Site",
    "aligned_producer_patches",
    "centre_tap",
    "collect",
    "gradmax",
    "grown_copy",
    "input_patches",
    "output_rows",
    "tensor_n",
    "tiny",
    "widen",
    "widen_conv2d",
    "widen_linear",
]
