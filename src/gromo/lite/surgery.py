"""Apply a :class:`~gromo.lite.solve.Proposal` by building new modules.

Nothing is modified in place: the caller decides where the new modules go. A
line search over ``gamma`` is :func:`grown_copy` followed by an evaluation, so
no extended forward pass is needed.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
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
    factory = {"device": producer.weight.device, "dtype": producer.weight.dtype}

    alpha_weight, alpha_bias = _split_bias(proposal.alpha, producer.bias is not None)
    new_producer = nn.Linear(
        producer.in_features,
        producer.out_features + added_units,
        bias=producer.bias is not None,
        **factory,
    )
    new_producer.weight.copy_(torch.cat([producer.weight, gamma * alpha_weight]))
    if producer.bias is not None and alpha_bias is not None:
        new_producer.bias.copy_(torch.cat([producer.bias, gamma * alpha_bias]))

    consumer_weight = consumer.weight.clone()
    consumer_bias = None if consumer.bias is None else consumer.bias.clone()
    if proposal.optimal_delta is not None:
        delta_weight, delta_bias = _split_bias(
            proposal.optimal_delta, consumer.bias is not None
        )
        consumer_weight -= gamma**2 * delta_weight
        if consumer_bias is not None and delta_bias is not None:
            consumer_bias -= gamma**2 * delta_bias
    new_consumer = nn.Linear(
        consumer.in_features + added_units,
        consumer.out_features,
        bias=consumer.bias is not None,
        **factory,
    )
    new_consumer.weight.copy_(torch.cat([consumer_weight, gamma * proposal.omega], dim=1))
    if consumer_bias is not None:
        new_consumer.bias.copy_(consumer_bias)
    return new_producer, new_consumer


def grown_copy(
    model: nn.Module,
    producer_name: str,
    consumer_name: str,
    proposal: Proposal,
    gamma: float,
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

    Returns
    -------
    nn.Module
        grown copy of ``model``
    """
    grown = copy.deepcopy(model)
    new_producer, new_consumer = widen_linear(
        grown.get_submodule(producer_name),  # type: ignore[arg-type]
        grown.get_submodule(consumer_name),  # type: ignore[arg-type]
        proposal,
        gamma,
    )
    grown.set_submodule(producer_name, new_producer)
    grown.set_submodule(consumer_name, new_consumer)
    return grown
