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
        update of the consumer weight with the bias in the last column, of shape
        ``(consumer_out, consumer_in [+1])``; applied as
        ``W -= gamma**2 * optimal_delta``
    alpha : torch.Tensor
        fan-in of the new units with the bias in the last column, of shape
        ``(k, producer_in [+1])``
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


def tiny(
    moments: Moments,
    maximum_added_neurons: int | None = None,
    statistical_threshold: float = 1e-3,
    numerical_threshold: float | None = None,
) -> Proposal:
    """TINY: natural-gradient update plus new units that fit the projected residual.

    Matches ``GrowingModule.compute_optimal_updates`` with ``compute_delta=True``,
    ``use_covariance=True`` and ``use_projection=True``.

    Parameters
    ----------
    moments : Moments
        moments of a ``nn.Linear -> nn.Linear`` site
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
    tensor_n = -moments["tensor_m_prev"] + moments["cross_covariance"] @ delta.T
    alpha, omega, eigenvalues = compute_optimal_added_parameters(
        moments["tensor_s_growth"],
        tensor_n,
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
        moments of a ``nn.Linear -> nn.Linear`` site
    maximum_added_neurons : int | None
        maximum number of new units, all significant ones if None
    statistical_threshold : float
        singular values of ``N`` below this are discarded

    Returns
    -------
    Proposal
        new units, without a weight update
    """
    tensor_n = -moments["tensor_m_prev"]
    alpha, omega, eigenvalues = compute_optimal_added_parameters(
        None,
        tensor_n,
        statistical_threshold=statistical_threshold,
        maximum_added_neurons=maximum_added_neurons,
        alpha_zero=True,
        ignore_singular_values=True,
    )
    no_decrease = torch.zeros((), dtype=tensor_n.dtype, device=tensor_n.device)
    return Proposal(None, alpha, omega, eigenvalues, no_decrease)
