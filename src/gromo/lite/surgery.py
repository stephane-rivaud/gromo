"""Apply a :class:`~gromo.lite.solve.Proposal` by building new modules.

Nothing is modified in place: the caller decides where the new modules go. A
line search over ``gamma`` is :func:`grown_copy` followed by an evaluation, so
no extended forward pass is needed.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

from gromo.lite.capture import _check_supported


ModuleType = TypeVar("ModuleType", bound=nn.Module)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from gromo.lite.solve import Proposal


def _split_bias(
    matrix: torch.Tensor, has_bias: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a matrix whose last column holds the bias.

    Parameters
    ----------
    matrix : torch.Tensor
        weight matrix, with the bias in the last column when ``has_bias``
    has_bias : bool
        whether the last column is a bias

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        weight and bias (None when ``has_bias`` is False)
    """
    if has_bias:
        return matrix[:, :-1], matrix[:, -1]
    return matrix, None


def _carry_state(new: ModuleType, old: nn.Module) -> ModuleType:
    """Give a rebuilt layer the training mode and gradient flags of the old one.

    Parameters
    ----------
    new : ModuleType
        layer built to replace ``old``
    old : nn.Module
        layer being replaced

    Returns
    -------
    ModuleType
        ``new``, in the mode of ``old`` and with the same ``requires_grad`` flags
    """
    new.train(old.training)
    for name, parameter in new.named_parameters(recurse=False):
        old_parameter = getattr(old, name, None)
        if isinstance(old_parameter, nn.Parameter):
            parameter.requires_grad_(old_parameter.requires_grad)
    return new


def _updated_consumer_parameters(
    consumer: nn.Linear | nn.Conv2d, proposal: Proposal, gamma: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the consumer weight and bias moved by ``-gamma**2 * optimal_delta``.

    Parameters
    ----------
    consumer : nn.Linear | nn.Conv2d
        layer whose input features grow
    proposal : Proposal
        proposal computed for this site
    gamma : float
        scale of the change

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        updated weight and bias (None when ``consumer`` has no bias)
    """
    weight = consumer.weight.clone()
    bias = None if consumer.bias is None else consumer.bias.clone()
    if proposal.optimal_delta is None:
        return weight, bias
    delta_weight, delta_bias = _split_bias(
        proposal.optimal_delta.to(weight), consumer.bias is not None
    )
    weight -= gamma**2 * delta_weight.reshape(weight.shape)
    if bias is not None and delta_bias is not None:
        bias -= gamma**2 * delta_bias
    return weight, bias


@torch.no_grad()
def widen_linear(
    producer: nn.Linear, consumer: nn.Linear, proposal: Proposal, gamma: float
) -> tuple[nn.Linear, nn.Linear]:
    """Return copies of ``producer`` and ``consumer`` with ``k`` more hidden units.

    The new units get fan-in ``gamma * alpha`` and fan-out ``gamma * omega``,
    and the existing consumer weights move by ``-gamma**2 * optimal_delta``.
    With ``gamma = 0`` the network computes the same function as before.

    Parameters
    ----------
    producer : nn.Linear
        layer whose output features grow
    consumer : nn.Linear
        layer whose input features grow
    proposal : Proposal
        proposal computed for this site
    gamma : float
        scale of the change

    Returns
    -------
    tuple[nn.Linear, nn.Linear]
        new producer and new consumer

    Raises
    ------
    TypeError
        if ``producer`` or ``consumer`` is not ``nn.Linear``
    """
    if not (isinstance(producer, nn.Linear) and isinstance(consumer, nn.Linear)):
        raise TypeError("widen_linear expects two nn.Linear layers.")
    added_units = proposal.alpha.shape[0]

    alpha_weight, alpha_bias = _split_bias(
        proposal.alpha.to(producer.weight), producer.bias is not None
    )
    new_producer = nn.Linear(
        producer.in_features,
        producer.out_features + added_units,
        bias=producer.bias is not None,
        device=producer.weight.device,
        dtype=producer.weight.dtype,
    )
    new_producer.weight.copy_(torch.cat([producer.weight, gamma * alpha_weight]))
    if producer.bias is not None and alpha_bias is not None:
        new_producer.bias.copy_(torch.cat([producer.bias, gamma * alpha_bias]))

    consumer_weight, consumer_bias = _updated_consumer_parameters(
        consumer, proposal, gamma
    )
    new_consumer = nn.Linear(
        consumer.in_features + added_units,
        consumer.out_features,
        bias=consumer.bias is not None,
        device=consumer.weight.device,
        dtype=consumer.weight.dtype,
    )
    new_consumer.weight.copy_(
        torch.cat([consumer_weight, gamma * proposal.omega.to(consumer_weight)], dim=1)
    )
    if consumer_bias is not None:
        new_consumer.bias.copy_(consumer_bias)
    return _carry_state(new_producer, producer), _carry_state(new_consumer, consumer)


def _resized_conv2d(layer: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    """Build a convolution with the geometry of ``layer`` and new channel counts.

    Parameters
    ----------
    layer : nn.Conv2d
        convolution whose kernel size, stride, padding, dilation, bias, device
        and dtype are copied
    in_channels : int
        number of input channels of the new convolution
    out_channels : int
        number of output channels of the new convolution

    Returns
    -------
    nn.Conv2d
        new convolution with uninitialised parameters
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=layer.kernel_size,  # type: ignore[arg-type]
        stride=layer.stride,  # type: ignore[arg-type]
        padding=layer.padding,  # type: ignore[arg-type]
        dilation=layer.dilation,  # type: ignore[arg-type]
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )


@torch.no_grad()
def widen_conv2d(
    producer: nn.Conv2d, consumer: nn.Conv2d, proposal: Proposal, gamma: float
) -> tuple[nn.Conv2d, nn.Conv2d]:
    """Return copies of ``producer`` and ``consumer`` with ``k`` more channels.

    Restricted scheme: a new channel gets the ``k x k`` fan-in ``gamma * alpha``
    and a ``1 x 1`` fan-out ``gamma * omega`` at the centre tap of the consumer
    kernel; the existing consumer weights move by ``-gamma**2 * optimal_delta``.
    With ``gamma = 0`` the network computes the same function as before.

    Parameters
    ----------
    producer : nn.Conv2d
        convolution whose output channels grow
    consumer : nn.Conv2d
        convolution whose input channels grow
    proposal : Proposal
        proposal computed for this site
    gamma : float
        scale of the change

    Returns
    -------
    tuple[nn.Conv2d, nn.Conv2d]
        new producer and new consumer

    Raises
    ------
    TypeError
        if ``producer`` or ``consumer`` is not ``nn.Conv2d``
    """
    if not (isinstance(producer, nn.Conv2d) and isinstance(consumer, nn.Conv2d)):
        raise TypeError("widen_conv2d expects two nn.Conv2d layers.")
    _check_supported(producer)
    _check_supported(consumer)
    added_units = proposal.alpha.shape[0]

    alpha_weight, alpha_bias = _split_bias(
        proposal.alpha.to(producer.weight), producer.bias is not None
    )
    new_producer = _resized_conv2d(
        producer, producer.in_channels, producer.out_channels + added_units
    )
    new_producer.weight.copy_(
        torch.cat(
            [
                producer.weight,
                gamma * alpha_weight.reshape(added_units, *producer.weight.shape[1:]),
            ]
        )
    )
    if producer.bias is not None and alpha_bias is not None:
        new_producer.bias.copy_(torch.cat([producer.bias, gamma * alpha_bias]))

    consumer_weight, consumer_bias = _updated_consumer_parameters(
        consumer, proposal, gamma
    )
    centre_height, centre_width = (size // 2 for size in consumer.kernel_size)
    fan_out = consumer_weight.new_zeros(
        consumer.out_channels, added_units, *consumer.kernel_size
    )
    fan_out[:, :, centre_height, centre_width] = gamma * proposal.omega.to(fan_out)
    new_consumer = _resized_conv2d(
        consumer, consumer.in_channels + added_units, consumer.out_channels
    )
    new_consumer.weight.copy_(torch.cat([consumer_weight, fan_out], dim=1))
    if consumer_bias is not None:
        new_consumer.bias.copy_(consumer_bias)
    return _carry_state(new_producer, producer), _carry_state(new_consumer, consumer)


_BATCH_NORM_TYPES: tuple[type[nn.modules.batchnorm._BatchNorm], ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


@torch.no_grad()
def widen_norm(
    norm: nn.modules.batchnorm._BatchNorm, added_units: int
) -> nn.modules.batchnorm._BatchNorm:
    """Return a copy of a batch normalisation with ``added_units`` more features.

    Matches ``GrowingBatchNorm.grow`` with its defaults: the new features get
    weight 1, bias 0, running mean 0 and running variance 1, and the existing
    features keep their parameters and statistics. ``num_batches_tracked`` is
    copied unchanged. A new unit with zero fan-in therefore leaves the
    normalisation as 0 in training and evaluation mode.

    Parameters
    ----------
    norm : nn.modules.batchnorm._BatchNorm
        ``nn.BatchNorm1d``, ``nn.BatchNorm2d`` or ``nn.BatchNorm3d`` on the hidden
        dimension of a site
    added_units : int
        number of new features

    Returns
    -------
    nn.modules.batchnorm._BatchNorm
        new normalisation of the same plain ``torch.nn`` class

    Raises
    ------
    TypeError
        if ``norm`` is not a batch normalisation
    ValueError
        if ``added_units`` is negative
    """
    norm_class = next((cls for cls in _BATCH_NORM_TYPES if isinstance(norm, cls)), None)
    if norm_class is None:
        raise TypeError(
            f"widen_norm expects a batch normalisation, got {type(norm).__name__}."
        )
    if added_units < 0:
        raise ValueError(f"added_units must be non-negative, got {added_units}.")
    reference = norm.weight if norm.weight is not None else norm.running_mean
    options: dict = (
        {"device": reference.device, "dtype": reference.dtype}
        if reference is not None
        else {}
    )
    if norm.affine and norm.bias is None:
        options["bias"] = False  # affine without bias exists from torch 2.13
    new_norm = norm_class(
        norm.num_features + added_units,
        eps=norm.eps,
        momentum=norm.momentum,  # type: ignore[arg-type]
        affine=norm.affine,
        track_running_stats=norm.track_running_stats,
        **options,
    )
    new_norm.train(norm.training)
    if norm.affine:
        new_norm.weight.copy_(torch.cat([norm.weight, norm.weight.new_ones(added_units)]))
    if norm.bias is not None:
        new_norm.bias.copy_(torch.cat([norm.bias, norm.bias.new_zeros(added_units)]))
    if norm.track_running_stats:
        assert norm.running_mean is not None and norm.running_var is not None
        new_norm.running_mean.copy_(  # type: ignore[union-attr]
            torch.cat([norm.running_mean, norm.running_mean.new_zeros(added_units)])
        )
        new_norm.running_var.copy_(  # type: ignore[union-attr]
            torch.cat([norm.running_var, norm.running_var.new_ones(added_units)])
        )
        new_norm.num_batches_tracked.copy_(norm.num_batches_tracked)  # type: ignore[union-attr]
    return _carry_state(new_norm, norm)


def widen(
    producer: nn.Module, consumer: nn.Module, proposal: Proposal, gamma: float
) -> tuple[nn.Module, nn.Module]:
    """Dispatch to :func:`widen_linear` or :func:`widen_conv2d`.

    Parameters
    ----------
    producer : nn.Module
        layer whose output features grow
    consumer : nn.Module
        layer whose input features grow
    proposal : Proposal
        proposal computed for this site
    gamma : float
        scale of the change

    Returns
    -------
    tuple[nn.Module, nn.Module]
        new producer and new consumer

    Raises
    ------
    TypeError
        if the layers are not two ``nn.Linear`` or two ``nn.Conv2d``
    """
    if isinstance(producer, nn.Linear) and isinstance(consumer, nn.Linear):
        return widen_linear(producer, consumer, proposal, gamma)
    if isinstance(producer, nn.Conv2d) and isinstance(consumer, nn.Conv2d):
        return widen_conv2d(producer, consumer, proposal, gamma)
    raise TypeError("widen expects two nn.Linear or two nn.Conv2d layers.")


def grown_copy(
    model: nn.Module,
    producer_name: str,
    consumer_name: str,
    proposal: Proposal,
    gamma: float,
    norm_names: Sequence[str] = (),
) -> nn.Module:
    """Return a deep copy of ``model`` with the proposal applied at scale ``gamma``.

    Parameters
    ----------
    model : nn.Module
        model to copy; it is not modified
    producer_name : str
        qualified name of the producer, as in ``model.get_submodule``
    consumer_name : str
        qualified name of the consumer, as in ``model.get_submodule``
    proposal : Proposal
        proposal computed for this site
    gamma : float
        scale of the change
    norm_names : Sequence[str]
        qualified names of the batch normalisations between producer and consumer,
        widened with :func:`widen_norm`

    Returns
    -------
    nn.Module
        grown copy of ``model``

    Raises
    ------
    ValueError
        if a normalisation does not have one feature per producer output
    """
    if isinstance(norm_names, str):
        norm_names = (norm_names,)
    grown = copy.deepcopy(model)
    producer = grown.get_submodule(producer_name)
    hidden_units = producer.weight.shape[0]
    added_units = proposal.alpha.shape[0]
    new_norms = {}
    for norm_name in norm_names:
        norm = grown.get_submodule(norm_name)
        if getattr(norm, "num_features", None) != hidden_units:
            raise ValueError(
                f"{norm_name!r} does not normalise the {hidden_units} outputs of "
                f"{producer_name!r}."
            )
        new_norms[norm_name] = widen_norm(norm, added_units)  # type: ignore[arg-type]
    new_producer, new_consumer = widen(
        producer, grown.get_submodule(consumer_name), proposal, gamma
    )
    grown.set_submodule(producer_name, new_producer)
    grown.set_submodule(consumer_name, new_consumer)
    for norm_name, new_norm in new_norms.items():
        grown.set_submodule(norm_name, new_norm)
    return grown
