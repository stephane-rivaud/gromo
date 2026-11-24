"""Splitting steepest descent: grow by splitting existing units in two.

Liu, Wu, Wang and Liu, *Splitting Steepest Descent for Growing Neural
Architectures*, NeurIPS 2019 (arXiv:1910.02366). For a hidden unit ``l`` of a
site with fan-in ``theta_l``, pre-activation ``h_l = theta_l . x`` and activation
``sigma``, the *splitting matrix* is

    S_l = E[ dL/dz_l * sigma''(h_l) * x x^T ],

where ``z_l = sigma(h_l)`` is the unit output and ``x`` carries a trailing one
when the producer has a bias. Replacing the unit by two copies with fan-ins
``theta_l + eps v`` and ``theta_l - eps v`` and half the fan-out each changes the
loss by ``eps**2 / 2 * v^T S_l v + O(eps**3)``; the best ``v`` is the eigenvector
of the smallest eigenvalue ``lambda_l`` of ``S_l`` (the splitting matrix
definition, the loss decomposition theorem and Algorithm 1 of the paper). Splits
of different units add up, so the units with the most negative ``lambda_l`` are
split.

The method has three parts, one per step of :mod:`gromo.lite`:

1. :func:`splitting_moment` is an extra statistic for
   :func:`~gromo.lite.capture.collect`;
2. :func:`splitting` solves for the units to split and their directions;
3. :func:`split_units` and :func:`split_copy` build the split modules.

At ``eps = 0`` the split network computes the same function. The activation
between producer and consumer must be elementwise. A batch normalisation in
between is supported: the moment folds its evaluation-mode affine map
``h = a z + b`` into the curvature, ``S_l = E[dL/dz_l * a_l**2 * sigma''(h_l) *
x x^T]``, and the split duplicates the unit's normalisation channel.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from gromo.lite.capture import _check_supported, input_patches, output_rows
from gromo.lite.surgery import _carry_state, _resized_conv2d, widen_norm


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from gromo.lite.capture import Moments, Site, SiteBatch


SPLITTING_MATRICES = "splitting_matrices"
_UNITS_PER_CHUNK = 32


def activation_second_derivative(
    activation: Callable[[torch.Tensor], torch.Tensor], pre_activation: torch.Tensor
) -> torch.Tensor:
    """Elementwise second derivative of ``activation`` at ``pre_activation``.

    Parameters
    ----------
    activation : Callable[[torch.Tensor], torch.Tensor]
        elementwise activation, for example ``torch.nn.functional.gelu``
    pre_activation : torch.Tensor
        points where the derivative is evaluated

    Returns
    -------
    torch.Tensor
        ``sigma''(pre_activation)``, of the same shape
    """
    point = pre_activation.detach().requires_grad_()
    with torch.enable_grad():
        (first,) = torch.autograd.grad(activation(point).sum(), point, create_graph=True)
        if not first.requires_grad:
            # piecewise-linear activations, such as ReLU, have no curvature
            return torch.zeros_like(point)
        (second,) = torch.autograd.grad(first.sum(), point)
    return second.detach()


def _evaluation_affine(
    norm: nn.modules.batchnorm._BatchNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature scale and shift of a batch normalisation in evaluation mode.

    Parameters
    ----------
    norm : nn.modules.batchnorm._BatchNorm
        batch normalisation with running statistics, in evaluation mode

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``a`` and ``b`` such that ``norm(z) = a * z + b`` feature-wise

    Raises
    ------
    ValueError
        if ``norm`` is in training mode or has no running statistics
    """
    if norm.training or norm.running_mean is None or norm.running_var is None:
        raise ValueError(
            "splitting_moment folds a batch normalisation only in evaluation mode "
            "with running statistics; call model.eval() before collecting."
        )
    scale = torch.rsqrt(norm.running_var + norm.eps)
    if norm.weight is not None:
        scale = scale * norm.weight
    shift = -norm.running_mean * scale
    if norm.bias is not None:
        shift = shift + norm.bias
    return scale.detach(), shift.detach()


def splitting_moment(
    activation: Callable[[torch.Tensor], torch.Tensor],
    norms: Mapping[Site, nn.modules.batchnorm._BatchNorm] | None = None,
) -> Callable[[SiteBatch, Moments], None]:
    """Extra statistic accumulating the splitting matrix of every producer unit.

    Pass it to :func:`~gromo.lite.capture.collect` as
    ``collect(sites, extra_moments=[splitting_moment(activation)])``. The moment
    ``"splitting_matrices"`` has shape ``(units, fan_in [+1], fan_in [+1])``.

    Parameters
    ----------
    activation : Callable[[torch.Tensor], torch.Tensor]
        elementwise activation between the producer and the consumer; its second
        derivative is the curvature of the splitting matrix
    norms : Mapping[Site, nn.modules.batchnorm._BatchNorm] | None
        batch normalisation between the producer and the activation, per site,
        matched by the site's producer and consumer layers; it must be in
        evaluation mode while the moments are collected

    Returns
    -------
    Callable[[SiteBatch, Moments], None]
        extra moment for :func:`~gromo.lite.capture.collect`
    """
    # match by the site's layers: an equal Site built separately must find its norm
    norm_of = {
        (site.producer, site.consumer): norm for site, norm in (norms or {}).items()
    }

    def accumulate(batch: SiteBatch, moments: Moments) -> None:
        producer = batch.site.producer
        pre_activation = output_rows(producer, batch.producer_output)
        unit_gradient = output_rows(producer, batch.consumer_input_gradient())
        norm = norm_of.get((batch.site.producer, batch.site.consumer))
        if norm is None:
            weights = unit_gradient * activation_second_derivative(
                activation, pre_activation
            )
        else:
            scale, shift = _evaluation_affine(norm)
            curvature = activation_second_derivative(
                activation, pre_activation * scale + shift
            )
            weights = unit_gradient * curvature * scale**2
        patches = input_patches(producer, batch.producer_input)
        chunks = [
            torch.einsum(
                "ru,ri,rj->uij",
                weights[:, start : start + _UNITS_PER_CHUNK],
                patches,
                patches,
            )
            for start in range(0, weights.shape[1], _UNITS_PER_CHUNK)
        ]
        if not chunks:  # a site with no unit yet has no unit to split
            fan_in = patches.shape[1]
            moments.add_sum(SPLITTING_MATRICES, patches.new_zeros((0, fan_in, fan_in)))
            return
        moments.add_sum(SPLITTING_MATRICES, torch.cat(chunks))

    return accumulate


@dataclass
class SplitProposal:
    """Units to split and the directions of their two copies.

    Attributes
    ----------
    units : torch.Tensor
        indices of the producer units to split, of shape ``(k,)``
    directions : torch.Tensor
        unit-norm splitting directions in the producer ``weight.flatten(1)``
        layout with the bias in the last column, of shape ``(k, fan_in [+1])``
    eigenvalues : torch.Tensor
        smallest eigenvalue of each unit's splitting matrix, of shape ``(k,)``;
        splitting with step ``eps`` changes the loss by about
        ``eps**2 / 2 * eigenvalues.sum()``
    """

    units: torch.Tensor
    directions: torch.Tensor
    eigenvalues: torch.Tensor


def splitting(
    moments: Moments, maximum_splits: int | None = None, threshold: float = 0.0
) -> SplitProposal:
    """Select the units with the most negative splitting index (Algorithm 1, step 3).

    Parameters
    ----------
    moments : Moments
        moments collected with :func:`splitting_moment`
    maximum_splits : int | None
        maximum number of units to split, all eligible units if None
    threshold : float
        only units whose smallest eigenvalue is below this value are split

    Returns
    -------
    SplitProposal
        units to split, sorted from the most negative eigenvalue

    Raises
    ------
    ValueError
        if ``maximum_splits`` is negative
    """
    if maximum_splits is not None and maximum_splits < 0:
        raise ValueError(f"maximum_splits must be non-negative, got {maximum_splits}.")
    matrices = moments[SPLITTING_MATRICES]
    if matrices.shape[0] == 0 or matrices.shape[-1] == 0:
        empty = matrices.new_zeros((0,))
        return SplitProposal(
            empty.long(), matrices.new_zeros((0, matrices.shape[-1])), empty
        )
    matrices = (matrices + matrices.transpose(1, 2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    smallest, directions = eigenvalues[:, 0], eigenvectors[:, :, 0]
    order = torch.argsort(smallest)
    order = order[smallest[order] < threshold]
    if maximum_splits is not None:
        order = order[:maximum_splits]
    return SplitProposal(order, directions[order], smallest[order])


def _split_bias(
    matrix: torch.Tensor, has_bias: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a ``weight.flatten(1)`` layout matrix whose last column may hold the bias.

    Parameters
    ----------
    matrix : torch.Tensor
        rows in the ``weight.flatten(1)`` layout, with the bias last if ``has_bias``
    has_bias : bool
        whether the last column is the bias

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        weight part and bias part (None without bias)
    """
    if has_bias:
        return matrix[:, :-1], matrix[:, -1]
    return matrix, None


def _unit_count(layer: nn.Module) -> int:
    """Number of output units of a ``nn.Linear`` or ``nn.Conv2d``.

    Parameters
    ----------
    layer : nn.Module
        producer of a site

    Returns
    -------
    int
        size of the first dimension of its weight
    """
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    return weight.shape[0]


@torch.no_grad()
def split_units(
    producer: nn.Module, consumer: nn.Module, proposal: SplitProposal, epsilon: float
) -> tuple[nn.Module, nn.Module]:
    """Return copies of ``producer`` and ``consumer`` with the proposed units split.

    Each selected unit keeps its index with fan-in ``theta + epsilon v`` and gets a
    new copy, appended after the existing units, with fan-in
    ``theta - epsilon v``. Both copies carry half the original fan-out.

    Parameters
    ----------
    producer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` producing the hidden units
    consumer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` (same type as ``producer``) reading them
    proposal : SplitProposal
        units and directions from :func:`splitting`
    epsilon : float
        splitting step

    Returns
    -------
    tuple[nn.Module, nn.Module]
        new producer and new consumer

    Raises
    ------
    TypeError
        if the layers are not two ``nn.Linear`` or two ``nn.Conv2d``
    ValueError
        if the proposed units repeat or are not units of ``producer``
    """
    _check_supported(producer)
    _check_supported(consumer)
    units = proposal.units.tolist()
    width = _unit_count(producer)
    if len(set(units)) != len(units) or any(not 0 <= unit < width for unit in units):
        raise ValueError(
            f"The units to split must be distinct units of the producer (0 to "
            f"{width - 1}), got {units}."
        )
    added = len(units)
    new_producer: nn.Linear | nn.Conv2d
    new_consumer: nn.Linear | nn.Conv2d
    if isinstance(producer, nn.Linear) and isinstance(consumer, nn.Linear):
        new_producer = nn.Linear(
            producer.in_features,
            producer.out_features + added,
            bias=producer.bias is not None,
            device=producer.weight.device,
            dtype=producer.weight.dtype,
        )
        new_consumer = nn.Linear(
            consumer.in_features + added,
            consumer.out_features,
            bias=consumer.bias is not None,
            device=consumer.weight.device,
            dtype=consumer.weight.dtype,
        )
    elif isinstance(producer, nn.Conv2d) and isinstance(consumer, nn.Conv2d):
        new_producer = _resized_conv2d(
            producer, producer.in_channels, producer.out_channels + added
        )
        new_consumer = _resized_conv2d(
            consumer, consumer.in_channels + added, consumer.out_channels
        )
    else:
        raise TypeError("split_units expects two nn.Linear or two nn.Conv2d layers.")
    _fill_split(producer, consumer, new_producer, new_consumer, proposal, epsilon)
    return _carry_state(new_producer, producer), _carry_state(new_consumer, consumer)


def _fill_split(
    producer: nn.Linear | nn.Conv2d,
    consumer: nn.Linear | nn.Conv2d,
    new_producer: nn.Linear | nn.Conv2d,
    new_consumer: nn.Linear | nn.Conv2d,
    proposal: SplitProposal,
    epsilon: float,
) -> None:
    """Write the split parameters into the resized modules.

    Parameters
    ----------
    producer : nn.Linear | nn.Conv2d
        producer before the split
    consumer : nn.Linear | nn.Conv2d
        consumer before the split
    new_producer : nn.Linear | nn.Conv2d
        producer with room for the new copies
    new_consumer : nn.Linear | nn.Conv2d
        consumer with room for the new copies
    proposal : SplitProposal
        units and directions from :func:`splitting`
    epsilon : float
        splitting step
    """
    units = proposal.units.to(producer.weight.device)
    step = epsilon * proposal.directions.to(producer.weight)
    step_weight, step_bias = _split_bias(step, producer.bias is not None)
    step_weight = step_weight.reshape(units.numel(), *producer.weight.shape[1:])

    producer_weight = producer.weight.clone()
    producer_weight[units] = producer.weight[units] + step_weight
    minus_weight = producer.weight[units] - step_weight
    new_producer.weight.copy_(torch.cat([producer_weight, minus_weight]))
    if (
        producer.bias is not None
        and new_producer.bias is not None
        and step_bias is not None
    ):
        producer_bias = producer.bias.clone()
        producer_bias[units] = producer.bias[units] + step_bias
        minus_bias = producer.bias[units] - step_bias
        new_producer.bias.copy_(torch.cat([producer_bias, minus_bias]))

    halved_fan_out = consumer.weight[:, units] / 2
    consumer_weight = consumer.weight.clone()
    consumer_weight[:, units] = halved_fan_out
    new_consumer.weight.copy_(torch.cat([consumer_weight, halved_fan_out], dim=1))
    if consumer.bias is not None and new_consumer.bias is not None:
        new_consumer.bias.copy_(consumer.bias)


@torch.no_grad()
def split_norm(
    norm: nn.modules.batchnorm._BatchNorm, units: torch.Tensor
) -> nn.modules.batchnorm._BatchNorm:
    """Return a copy of a batch normalisation with the features ``units`` duplicated.

    The copies are appended in the order of ``units``, as :func:`split_units`
    appends the new producer units, so a split at ``eps = 0`` computes the same
    function in training and evaluation mode.

    Parameters
    ----------
    norm : nn.modules.batchnorm._BatchNorm
        batch normalisation on the hidden units of the site
    units : torch.Tensor
        indices of the split units

    Returns
    -------
    nn.modules.batchnorm._BatchNorm
        normalisation with ``len(units)`` more features
    """
    new_norm = widen_norm(norm, int(units.numel()))
    count = norm.num_features
    for name in ("weight", "bias", "running_mean", "running_var"):
        old, new = getattr(norm, name), getattr(new_norm, name)
        if old is not None and new is not None:
            new[count:] = old[units.to(old.device)]
    return new_norm


def split_copy(
    model: nn.Module,
    producer_name: str,
    consumer_name: str,
    proposal: SplitProposal,
    epsilon: float,
    norm_names: Sequence[str] = (),
) -> nn.Module:
    """Return a deep copy of ``model`` with the proposed units of a site split.

    Parameters
    ----------
    model : nn.Module
        model to copy; it is not modified
    producer_name : str
        qualified name of the producer, as in ``model.get_submodule``
    consumer_name : str
        qualified name of the consumer, as in ``model.get_submodule``
    proposal : SplitProposal
        units and directions from :func:`splitting`
    epsilon : float
        splitting step
    norm_names : Sequence[str]
        qualified names of batch normalisations on the hidden units, whose
        features of the split units are duplicated

    Returns
    -------
    nn.Module
        split copy of ``model``

    Raises
    ------
    TypeError
        if a module named in ``norm_names`` is not a batch normalisation
    ValueError
        if such a normalisation does not normalise the producer's units
    """
    if isinstance(norm_names, str):
        norm_names = (norm_names,)
    width = _unit_count(model.get_submodule(producer_name))
    for norm_name in norm_names:
        norm = model.get_submodule(norm_name)
        if not isinstance(norm, nn.modules.batchnorm._BatchNorm):
            raise TypeError(f"{norm_name!r} is not a batch normalisation.")
        if norm.num_features != width:
            raise ValueError(
                f"{norm_name!r} normalises {norm.num_features} features, but "
                f"{producer_name!r} has {width} units."
            )
    split = copy.deepcopy(model)
    new_producer, new_consumer = split_units(
        split.get_submodule(producer_name),
        split.get_submodule(consumer_name),
        proposal,
        epsilon,
    )
    split.set_submodule(producer_name, new_producer)
    split.set_submodule(consumer_name, new_consumer)
    for norm_name in norm_names:
        norm = split.get_submodule(norm_name)
        assert isinstance(norm, nn.modules.batchnorm._BatchNorm)
        split.set_submodule(norm_name, split_norm(norm, proposal.units))
    return split
