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
``tensor_m_prev``     E[x g^T]     ``tensor_m_prev``
``cross_covariance``  E[x z^T]     ``cross_covariance``
====================  ===========  ====================================

``x`` and ``z`` get a trailing column of ones when their layer has a bias, and
convolution inputs are unfolded into patches. As in v1, the sums run over all
positions and are divided by the number of batch elements.

For convolution sites, ``tensor_s_growth`` is the covariance of the producer's
own input patches, as in ``RestrictedConv2dGrowingModule``.
``FullConv2dGrowingModule`` unfolds the producer input over the composed
receptive field instead, which is not implemented here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    Sites compare and hash by identity, so they can key dictionaries.

    Attributes
    ----------
    producer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer whose output features would grow
    consumer : nn.Module
        ``nn.Linear`` or ``nn.Conv2d`` layer whose input features would grow
    """

    producer: nn.Module
    consumer: nn.Module

    @property
    def has_pair_moments(self) -> bool:
        """Whether ``tensor_m_prev`` and ``cross_covariance`` can be collected.

        Both moments need producer patches aligned with consumer output
        positions. That holds for ``nn.Linear -> nn.Linear``; convolution pairs
        need the composed receptive field and are not supported yet.

        Returns
        -------
        bool
            True when both layers are ``nn.Linear``
        """
        return isinstance(self.producer, nn.Linear) and isinstance(
            self.consumer, nn.Linear
        )


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
        moments.add("tensor_m_prev", producer_patches, gradient_rows)
        moments.add("cross_covariance", producer_patches, consumer_patches)
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
    cached_inputs: dict[str, torch.Tensor] = {}

    def on_producer_forward(
        _module: nn.Module, args: tuple[torch.Tensor, ...], _output: torch.Tensor
    ) -> None:
        cached_inputs["producer"] = args[0].detach()

    def on_consumer_forward(
        _module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        producer_input = cached_inputs.pop("producer", None)
        if not output.requires_grad:
            return
        if producer_input is None:
            raise RuntimeError(
                "The consumer ran before the producer; check the order of the Site."
            )
        consumer_input = args[0].detach()

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
