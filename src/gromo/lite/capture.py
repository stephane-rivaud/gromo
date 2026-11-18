"""Capture growth statistics from plain ``torch.nn`` layers with temporary hooks.

A growth site is a pair of layers that share a hidden dimension::

    x --producer--> h --activation--> z --consumer--> a        g = dL/da

Inside :func:`collect`, forward hooks read ``x`` and ``z``, and a tensor hook
on ``a`` reads ``g`` during the backward pass. Each backward pass adds outer
products to a :class:`Moments` accumulator. The hooks are removed when the
context exits, so the model itself never stores statistics.

Moment names match the ``GrowingModule`` attributes that they reproduce:

====================  ===========  ====================================
name                  definition   v1 attribute (on the consumer)
====================  ===========  ====================================
``tensor_s``          E[z z^T]     ``tensor_s``
``tensor_m``          E[z g^T]     ``tensor_m``
``tensor_s_growth``   E[x x^T]     ``tensor_s_growth``
``tensor_m_prev``     E[x' g^T]    ``tensor_m_prev``
``cross_covariance``  E[x' z^T]    ``cross_covariance``
====================  ===========  ====================================

``x`` and ``z`` get a trailing column of ones when their layer has a bias, and
convolution inputs are unfolded into patches. As in v1, the sums run over all
positions and are divided by the number of batch elements.

``x'`` is ``x`` aligned with the consumer output positions, see
:func:`aligned_producer_patches`. For ``nn.Linear -> nn.Linear`` it is ``x``.
For ``nn.Conv2d -> nn.Conv2d`` it follows the restricted scheme of
``RestrictedConv2dGrowingModule``: a new unit has a ``k x k`` fan-in and a
``1 x 1`` fan-out at the centre tap of the consumer kernel. The full scheme of
``FullConv2dGrowingModule`` (``k x k`` fan-out, composed receptive field) is not
implemented.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.hooks import RemovableHandle


def _check_supported(layer: nn.Module) -> None:
    """Raise if ``layer`` is not a layer type that the patch views support.

    Parameters
    ----------
    layer : nn.Module
        layer to check

    Raises
    ------
    TypeError
        if ``layer`` is neither ``nn.Linear`` nor ``nn.Conv2d``
    NotImplementedError
        if ``layer`` is a grouped, string-padded or non-zero-padded ``nn.Conv2d``
    """
    if isinstance(layer, nn.Linear):
        return
    if not isinstance(layer, nn.Conv2d):
        raise TypeError(f"Unsupported layer type {type(layer).__name__}.")
    if (
        layer.groups != 1
        or isinstance(layer.padding, str)
        or layer.padding_mode != "zeros"
    ):
        raise NotImplementedError(
            "Only ungrouped Conv2d layers with numeric zero padding are supported."
        )


def input_patches(layer: nn.Module, layer_input: torch.Tensor) -> torch.Tensor:
    """Lay out a layer input as one row per application of the layer weight.

    Parameters
    ----------
    layer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer that receives ``layer_input``
    layer_input : torch.Tensor
        input of ``layer``

    Returns
    -------
    torch.Tensor
        matrix of shape ``(positions, fan_in [+1])``, with a trailing column of
        ones when ``layer`` has a bias
    """
    _check_supported(layer)
    if isinstance(layer, nn.Conv2d):
        unfolded = nn.functional.unfold(
            layer_input,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,  # type: ignore[arg-type]
            stride=layer.stride,
        )
        patches = unfolded.transpose(1, 2).flatten(0, 1)
    else:
        patches = layer_input.flatten(0, -2)
    if layer.bias is not None:
        patches = torch.cat([patches, torch.ones_like(patches[:, :1])], dim=1)
    return patches


def output_rows(layer: nn.Module, output_gradient: torch.Tensor) -> torch.Tensor:
    """Lay out the gradient of a layer output with the rows of :func:`input_patches`.

    Parameters
    ----------
    layer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer that produced the output
    output_gradient : torch.Tensor
        gradient of the loss with respect to the output of ``layer``

    Returns
    -------
    torch.Tensor
        matrix of shape ``(positions, fan_out)``
    """
    _check_supported(layer)
    if isinstance(layer, nn.Conv2d):
        return output_gradient.flatten(2).transpose(1, 2).flatten(0, 1)
    return output_gradient.flatten(0, -2)


def centre_tap(consumer: nn.Conv2d, feature_map: torch.Tensor) -> torch.Tensor:
    """Read ``feature_map`` where the centre tap of ``consumer`` reads its input.

    Output position ``p`` of ``consumer`` reads its centre tap at input position
    ``p * stride - padding + dilation * (kernel_size // 2)``. Positions that fall
    in the zero padding read zeros. This is the depthwise centre-tap convolution
    that v1 applies in ``apply_border_effect_on_unfolded``.

    Parameters
    ----------
    consumer : nn.Conv2d
        convolution whose geometry defines the read positions
    feature_map : torch.Tensor
        tensor of shape ``(batch, channels, height, width)`` with the spatial size
        of the consumer input

    Returns
    -------
    torch.Tensor
        tensor of shape ``(batch, channels, out_height, out_width)`` with the
        spatial size of the consumer output
    """
    pad_height, pad_width = consumer.padding  # type: ignore[misc]
    padded = nn.functional.pad(
        feature_map, (pad_width, pad_width, pad_height, pad_height)
    )
    slices = []
    for axis in range(2):
        input_size = feature_map.shape[2 + axis] + 2 * consumer.padding[axis]  # type: ignore[operator]
        span = consumer.dilation[axis] * (consumer.kernel_size[axis] - 1) + 1
        output_size = (input_size - span) // consumer.stride[axis] + 1
        start = consumer.dilation[axis] * (consumer.kernel_size[axis] // 2)
        stop = start + consumer.stride[axis] * (output_size - 1) + 1
        slices.append(slice(start, stop, consumer.stride[axis]))
    return padded[:, :, slices[0], slices[1]]


def aligned_producer_patches(
    site: Site, producer_patches: torch.Tensor, consumer_input: torch.Tensor
) -> torch.Tensor:
    """Align producer input patches with the consumer output positions.

    Row ``r`` holds the producer patch that a new hidden unit reads before its
    fan-out reaches consumer output row ``r`` of :func:`output_rows`.

    Parameters
    ----------
    site : Site
        ``nn.Linear -> nn.Linear`` or ``nn.Conv2d -> nn.Conv2d`` site
    producer_patches : torch.Tensor
        :func:`input_patches` of the producer input
    consumer_input : torch.Tensor
        input of the consumer

    Returns
    -------
    torch.Tensor
        matrix of shape ``(consumer_output_positions, producer_fan_in [+1])``

    Raises
    ------
    TypeError
        if the site mixes ``nn.Linear`` and ``nn.Conv2d`` layers
    """
    if isinstance(site.producer, nn.Linear) and isinstance(site.consumer, nn.Linear):
        return producer_patches
    if not (
        isinstance(site.producer, nn.Conv2d) and isinstance(site.consumer, nn.Conv2d)
    ):
        raise TypeError("Pair moments need two nn.Linear or two nn.Conv2d layers.")
    batch_size, _, height, width = consumer_input.shape
    patch_map = (
        producer_patches.reshape(batch_size, height * width, -1)
        .transpose(1, 2)
        .reshape(batch_size, -1, height, width)
    )
    centred = centre_tap(site.consumer, patch_map)
    return centred.flatten(2).transpose(1, 2).flatten(0, 1)


class Moments:
    """Running empirical averages of outer products, keyed by name."""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.samples = 0

    def add(self, name: str, left: torch.Tensor, right: torch.Tensor) -> None:
        """Add ``left^T @ right`` to the running sum called ``name``.

        Parameters
        ----------
        name : str
            name of the moment
        left : torch.Tensor
            matrix of shape ``(positions, p)``
        right : torch.Tensor
            matrix of shape ``(positions, q)``
        """
        update = left.T @ right
        if name in self.sums:
            self.sums[name] += update
        else:
            self.sums[name] = update

    def __contains__(self, name: object) -> bool:
        return name in self.sums

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the empirical average of the moment called ``name``.

        Parameters
        ----------
        name : str
            name of the moment

        Returns
        -------
        torch.Tensor
            accumulated sum divided by the number of samples

        Raises
        ------
        KeyError
            if the moment was not collected
        """
        if name not in self.sums:
            raise KeyError(
                f"Moment {name!r} was not collected; available: {sorted(self.sums)}."
            )
        return self.sums[name] / self.samples


@dataclass(frozen=True, eq=False)
class Site:
    """Hidden dimension shared by the output of ``producer`` and the input of ``consumer``.

    Sites compare and hash by identity, so they can key dictionaries. Layers
    between ``producer`` and ``consumer`` must preserve the tensor shape
    (activations, normalisation), as new units pass through them unchanged.

    Attributes
    ----------
    producer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer whose output features would grow
    consumer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer whose input features would grow
    conv_scheme : Literal["restricted", "full"]
        fan-out of new units for ``nn.Conv2d -> nn.Conv2d`` sites; only
        ``"restricted"`` (``1 x 1`` fan-out at the consumer centre tap) is
        implemented
    """

    producer: nn.Module
    consumer: nn.Module
    conv_scheme: Literal["restricted", "full"] = "restricted"

    def __post_init__(self) -> None:
        """Reject schemes that are not implemented.

        Raises
        ------
        NotImplementedError
            if ``conv_scheme`` is ``"full"``
        ValueError
            if ``conv_scheme`` is not a known scheme
        """
        if self.conv_scheme == "full":
            raise NotImplementedError(
                "The full conv scheme of FullConv2dGrowingModule (k x k fan-out, "
                "composed receptive field) is not implemented in gromo.lite; use "
                "conv_scheme='restricted' or the v1 module."
            )
        if self.conv_scheme != "restricted":
            raise ValueError(f"Unknown conv_scheme {self.conv_scheme!r}.")

    @property
    def has_pair_moments(self) -> bool:
        """Whether ``tensor_m_prev`` and ``cross_covariance`` can be collected.

        Returns
        -------
        bool
            True for ``nn.Linear -> nn.Linear`` and ``nn.Conv2d -> nn.Conv2d``
        """
        both_linear = isinstance(self.producer, nn.Linear) and isinstance(
            self.consumer, nn.Linear
        )
        both_conv = isinstance(self.producer, nn.Conv2d) and isinstance(
            self.consumer, nn.Conv2d
        )
        return both_linear or both_conv


@torch.no_grad()
def _accumulate(
    site: Site,
    moments: Moments,
    producer_input: torch.Tensor,
    consumer_input: torch.Tensor,
    output_gradient: torch.Tensor,
) -> None:
    """Add the contribution of one batch to the moments of ``site``.

    Parameters
    ----------
    site : Site
        growth site the tensors were read from
    moments : Moments
        accumulator of ``site``
    producer_input : torch.Tensor
        input of ``site.producer``
    consumer_input : torch.Tensor
        input of ``site.consumer``
    output_gradient : torch.Tensor
        gradient of the loss with respect to the output of ``site.consumer``
    """
    producer_patches = input_patches(site.producer, producer_input)
    consumer_patches = input_patches(site.consumer, consumer_input)
    gradient_rows = output_rows(site.consumer, output_gradient)
    moments.add("tensor_s", consumer_patches, consumer_patches)
    moments.add("tensor_m", consumer_patches, gradient_rows)
    moments.add("tensor_s_growth", producer_patches, producer_patches)
    if site.has_pair_moments:
        aligned_patches = aligned_producer_patches(site, producer_patches, consumer_input)
        moments.add("tensor_m_prev", aligned_patches, gradient_rows)
        moments.add("cross_covariance", aligned_patches, consumer_patches)
    moments.samples += output_gradient.shape[0]


def _attach(site: Site, moments: Moments) -> list[RemovableHandle]:
    """Register the hooks that feed the moments of ``site``.

    Parameters
    ----------
    site : Site
        growth site to observe
    moments : Moments
        accumulator of ``site``

    Returns
    -------
    list[RemovableHandle]
        handles of the registered forward hooks
    """
    cached_producer: dict[str, torch.Tensor] = {}

    def on_producer_forward(
        _module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        cached_producer["input"] = args[0].detach()
        cached_producer["output"] = output.detach()

    def on_consumer_forward(
        _module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        producer_input = cached_producer.pop("input", None)
        producer_output = cached_producer.pop("output", None)
        if not output.requires_grad:
            return
        if producer_input is None or producer_output is None:
            raise RuntimeError(
                "The consumer ran before the producer; check the order of the Site."
            )
        consumer_input = args[0].detach()
        if producer_output.shape != consumer_input.shape:
            raise ValueError(
                f"The producer output {tuple(producer_output.shape)} and the consumer "
                f"input {tuple(consumer_input.shape)} differ; layers between them "
                f"must preserve the shape."
            )

        def on_output_gradient(output_gradient: torch.Tensor) -> None:
            _accumulate(
                site, moments, producer_input, consumer_input, output_gradient.detach()
            )

        output.register_hook(on_output_gradient)

    return [
        site.producer.register_forward_hook(on_producer_forward),
        site.consumer.register_forward_hook(on_consumer_forward),
    ]


@contextmanager
def collect(sites: list[Site]) -> Iterator[dict[Site, Moments]]:
    """Accumulate the moments of each site over the passes run inside the context.

    Run forward and backward passes inside the ``with`` block as in training.
    Use a loss with ``reduction="sum"`` so that the moments are per-sample
    averages, as in v1. Forward passes that do not track gradients are
    ignored. Each layer of a site must run once per forward pass.

    Parameters
    ----------
    sites : list[Site]
        growth sites to observe

    Yields
    ------
    dict[Site, Moments]
        moments of each site, filled in as backward passes run
    """
    moments = {site: Moments() for site in sites}
    handles = [handle for site in sites for handle in _attach(site, moments[site])]
    try:
        yield moments
    finally:
        for handle in handles:
            handle.remove()
