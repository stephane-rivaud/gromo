"""Growth methods as pure functions of :class:`~gromo.lite.capture.Moments`.

A method only decides which moments feed the shared solvers in
:mod:`gromo.utils.tools`, the same solvers v1 uses. The solvers run in the
dtype of the moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from gromo.utils.tools import compute_optimal_added_parameters, optimal_delta


if TYPE_CHECKING:
    from gromo.lite.capture import Moments


@dataclass
class Proposal:
    """Parameter changes proposed for one growth site.

    Attributes
    ----------
    optimal_delta : torch.Tensor | None
        update of the consumer ``weight.flatten(1)`` with the bias in the last
        column, of shape ``(consumer_out, consumer_fan_in [+1])``; applied as
        ``W -= gamma**2 * optimal_delta``
    alpha : torch.Tensor
        fan-in of the new units in the producer ``weight.flatten(1)`` layout with
        the bias in the last column, of shape ``(k, producer_fan_in [+1])``
    omega : torch.Tensor
        fan-out of the new units, of shape ``(consumer_out, k)``
    eigenvalues : torch.Tensor
        singular values of the new units, of shape ``(k,)``
    parameter_update_decrease : torch.Tensor
        first-order loss decrease of ``optimal_delta`` alone
    """

    optimal_delta: torch.Tensor | None
    alpha: torch.Tensor
    omega: torch.Tensor
    eigenvalues: torch.Tensor
    parameter_update_decrease: torch.Tensor


def tensor_n(moments: Moments, delta: torch.Tensor) -> torch.Tensor:
    """Negative gradient of the new units, projected off the consumer update.

    ``N = -E[x' (g - delta z)^T] = -tensor_m_prev + cross_covariance @ delta^T``,
    as in ``GrowingModule.tensor_n``.

    Parameters
    ----------
    moments : Moments
        moments of a site with pair moments
    delta : torch.Tensor
        consumer update of shape ``(consumer_out, consumer_fan_in [+1])``

    Returns
    -------
    torch.Tensor
        tensor of shape ``(producer_fan_in [+1], consumer_out)``
    """
    return -moments["tensor_m_prev"] + moments["cross_covariance"] @ delta.T


def tiny(
    moments: Moments,
    maximum_added_neurons: int | None = None,
    statistical_threshold: float = 1e-3,
    numerical_threshold: float | None = None,
) -> Proposal:
    """TINY: natural-gradient update plus new units that fit the projected residual.

    Matches ``GrowingModule.compute_optimal_updates`` with ``compute_delta=True``,
    ``use_covariance=True`` and ``use_projection=True``. For a convolution site,
    ``optimal_delta`` follows the ``weight.flatten(1)`` layout and ``alpha`` the
    producer ``weight.flatten(1)`` layout, each with the bias in the last column.

    Parameters
    ----------
    moments : Moments
        moments of a ``nn.Linear -> nn.Linear`` or ``nn.Conv2d -> nn.Conv2d`` site
    maximum_added_neurons : int | None
        maximum number of new units, all significant ones if None
    statistical_threshold : float
        singular values of ``S^{-1/2} N`` below this are discarded
    numerical_threshold : float | None
        eigenvalues of ``S`` below this are treated as zero, numerical floor if None

    Returns
    -------
    Proposal
        weight update and new units
    """
    tensor_s = moments["tensor_s"]
    delta, parameter_update_decrease = optimal_delta(
        tensor_s, moments["tensor_m"], dtype=tensor_s.dtype
    )
    alpha, omega, eigenvalues = compute_optimal_added_parameters(
        moments["tensor_s_growth"],
        tensor_n(moments, delta),
        numerical_threshold=numerical_threshold,
        statistical_threshold=statistical_threshold,
        maximum_added_neurons=maximum_added_neurons,
    )
    return Proposal(delta, alpha, omega, eigenvalues, parameter_update_decrease)


def gradmax(
    moments: Moments,
    maximum_added_neurons: int | None = None,
    statistical_threshold: float = 1e-3,
) -> Proposal:
    """GradMax: zero fan-in, fan-out from the singular vectors of the raw gradient.

    Matches ``GrowingModule.compute_optimal_updates`` with ``compute_delta=False``,
    ``use_covariance=False``, ``alpha_zero=True``, ``use_projection=False`` and
    ``ignore_singular_values=True``.

    Parameters
    ----------
    moments : Moments
        moments of a ``nn.Linear -> nn.Linear`` or ``nn.Conv2d -> nn.Conv2d`` site
    maximum_added_neurons : int | None
        maximum number of new units, all significant ones if None
    statistical_threshold : float
        singular values of ``N`` below this are discarded

    Returns
    -------
    Proposal
        new units, without a weight update
    """
    negative_gradient = -moments["tensor_m_prev"]
    alpha, omega, eigenvalues = compute_optimal_added_parameters(
        None,
        negative_gradient,
        statistical_threshold=statistical_threshold,
        maximum_added_neurons=maximum_added_neurons,
        alpha_zero=True,
        ignore_singular_values=True,
    )
    no_decrease = torch.zeros(
        (), dtype=negative_gradient.dtype, device=negative_gradient.device
    )
    return Proposal(None, alpha, omega, eigenvalues, no_decrease)
