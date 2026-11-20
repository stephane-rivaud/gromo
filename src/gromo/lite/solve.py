"""Growth methods as pure functions of :class:`~gromo.lite.capture.Moments`.

A method only decides which moments feed the shared solvers in
:mod:`gromo.utils.tools`, the same solvers v1 uses. The solvers run in the
dtype of the moments.
"""

from __future__ import annotations

import dataclasses
import warnings
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
        eigenvalues of ``S`` below this are treated as zero; None keeps the
        default of ``compute_optimal_added_parameters``

    Returns
    -------
    Proposal
        weight update and new units
    """
    tensor_s = moments["tensor_s"]
    delta, parameter_update_decrease = optimal_delta(
        tensor_s, moments["tensor_m"], dtype=tensor_s.dtype
    )
    # Pass the threshold only when given, as gradmax does: older v1 versions
    # take a float with a default and reject None.
    options = (
        {}
        if numerical_threshold is None
        else {"numerical_threshold": numerical_threshold}
    )
    alpha, omega, eigenvalues = compute_optimal_added_parameters(
        moments["tensor_s_growth"],
        tensor_n(moments, delta),
        statistical_threshold=statistical_threshold,
        maximum_added_neurons=maximum_added_neurons,
        **options,
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


@torch.no_grad()
def gradmax_normalization(
    proposal: Proposal, consumer_weight: torch.Tensor, scale: float = 1.0
) -> Proposal:
    """Rescale each new fan-out column to the mean norm of the existing ones.

    Port of ``GrowingModule.normalize_optimal_updates`` with
    ``normalization_type="gradmax_normalization"``. Let
    ``c = scale * mean_i ||W[:, i]||``, where ``W`` is ``consumer_weight`` and the
    norm runs over every axis except the input axis 1 (for a convolution, over the
    output channels and the kernel taps). Each non-zero column ``omega[:, j]`` is
    rescaled to norm ``c``, and its eigenvalue is multiplied by ``sqrt(r_j)``,
    where ``r_j`` is the factor applied to the column. The fan-in ``alpha`` is
    left unchanged. In the restricted scheme a new channel only uses the centre
    tap, so the norm of its fan-out is the norm of ``omega[:, j]``.

    As in v1, the proposal is returned unchanged with a ``UserWarning`` when
    ``consumer_weight`` has no axis to reduce, no input unit, or zero norm.

    Parameters
    ----------
    proposal : Proposal
        proposal computed for the site whose consumer has ``consumer_weight``
    consumer_weight : torch.Tensor
        weight of the consumer before growth, with the input units on axis 1
    scale : float
        factor ``s`` of the target norm ``c = s * mean_i ||W[:, i]||``

    Returns
    -------
    Proposal
        proposal with rescaled ``omega`` and ``eigenvalues``

    Raises
    ------
    ValueError
        if ``scale`` is not positive
    """
    if scale <= 0:
        raise ValueError(f"gradmax_normalization needs a positive scale, got {scale}.")
    if consumer_weight.dim() < 2:
        warnings.warn(
            "gradmax_normalization: consumer weight has no reducible dimensions, "
            "skipping.",
            UserWarning,
            stacklevel=2,
        )
        return proposal
    reduced_dims = tuple(dim for dim in range(consumer_weight.dim()) if dim != 1)
    existing_norms = (consumer_weight**2).sum(dim=reduced_dims).sqrt()
    if existing_norms.numel() == 0:
        warnings.warn(
            "gradmax_normalization: no existing units to compute the reference norm, "
            "skipping.",
            UserWarning,
            stacklevel=2,
        )
        return proposal
    target_norm = existing_norms.mean() * float(scale)
    if target_norm <= 0:
        warnings.warn(
            "gradmax_normalization: target norm is zero (all existing weights are "
            "zero), skipping.",
            UserWarning,
            stacklevel=2,
        )
        return proposal

    column_norms = (proposal.omega**2).sum(dim=0).sqrt()
    non_zero = column_norms > 0
    factors = torch.ones_like(column_norms)
    factors[non_zero] = target_norm.to(column_norms.dtype) / column_norms[non_zero]
    return dataclasses.replace(
        proposal,
        omega=proposal.omega * factors[None, :],
        eigenvalues=proposal.eigenvalues
        * factors.sqrt().to(dtype=proposal.eigenvalues.dtype),
    )


def fixed_proportional_budget(
    initial_units: int, target_units: int, number_of_growth_steps: int
) -> int:
    """Number of units to add at each of ``number_of_growth_steps`` growth steps.

    Port of ``GrowingModule.number_of_neurons_to_add("fixed_proportional")``: the
    integer division ``(target_units - initial_units) // number_of_growth_steps``,
    so up to ``number_of_growth_steps - 1`` units may remain after the last step.

    Parameters
    ----------
    initial_units : int
        width of the hidden dimension before the first growth step
    target_units : int
        width the hidden dimension should reach
    number_of_growth_steps : int
        number of planned growth steps

    Returns
    -------
    int
        units to add per growth step

    Raises
    ------
    ValueError
        if ``number_of_growth_steps`` is not positive
    """
    if number_of_growth_steps <= 0:
        raise ValueError(
            f"number_of_growth_steps must be positive, got {number_of_growth_steps}."
        )
    return (target_units - initial_units) // number_of_growth_steps


def first_order_decrease(
    moments: Moments, proposal: Proposal, activation_slope: float = 1.0
) -> torch.Tensor:
    """Exact first-order loss decrease of a proposal applied with scale ``gamma``.

    Growing with :func:`~gromo.lite.surgery.grown_copy` at scale ``gamma``
    changes the consumer output by ``-gamma**2 delta z + gamma omega
    sigma(gamma alpha x)``. With ``sigma(0) = 0`` and ``sigma'(0) = s``, the loss
    then decreases by ``gamma**2 * D + o(gamma**2)`` with

        D = parameter_update_decrease - s * trace(omega alpha tensor_m_prev).

    This is the directional derivative the surgery actually follows. TINY's
    ``parameter_update_decrease + sum(eigenvalues**2)`` differs from it by the
    cross term ``E[<delta z, omega alpha x>]``.

    Parameters
    ----------
    moments : Moments
        moments of a site with pair moments
    proposal : Proposal
        proposal computed from ``moments``
    activation_slope : float
        ``sigma'(0)`` of the activation between producer and consumer, for
        example 1 for tanh and 0.5 for GELU

    Returns
    -------
    torch.Tensor
        scalar ``D``
    """
    new_units = torch.trace(proposal.omega @ proposal.alpha @ moments["tensor_m_prev"])
    return proposal.parameter_update_decrease - activation_slope * new_units


_RANDOM_INITIALIZATIONS = ("zeros", "kaiming_normal", "copy_normal")


def _random_block(
    shape: tuple[int, int],
    initialization: str,
    fan_in: int,
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw one block of new weights as v1's extension initializations do.

    Parameters
    ----------
    shape : tuple[int, int]
        shape of the block
    initialization : str
        ``"zeros"``, ``"kaiming_normal"`` (std ``sqrt(2 / fan_in)``) or
        ``"copy_normal"`` (std of ``reference``)
    fan_in : int
        fan-in used by ``"kaiming_normal"``
    reference : torch.Tensor
        existing weights whose standard deviation ``"copy_normal"`` copies
    generator : torch.Generator | None
        random generator

    Returns
    -------
    torch.Tensor
        block of shape ``shape`` in the dtype and device of ``reference``

    Raises
    ------
    ValueError
        if ``initialization`` is unknown
    """
    if initialization not in _RANDOM_INITIALIZATIONS:
        raise ValueError(
            f"Unknown initialization {initialization!r}; "
            f"expected one of {_RANDOM_INITIALIZATIONS}."
        )
    if initialization == "zeros":
        return torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    if initialization == "copy_normal" and reference.numel() > 1:
        std = float(reference.std())
    else:
        std = (2.0 / fan_in) ** 0.5
    # draw on the generator's device: a CPU generator cannot fill an MPS tensor
    device = generator.device if generator is not None else reference.device
    block = torch.empty(shape, dtype=reference.dtype, device=device)
    return block.normal_(0.0, std, generator=generator).to(reference.device)


@torch.no_grad()
def random_proposal(
    producer: torch.nn.Linear | torch.nn.Conv2d,
    consumer: torch.nn.Linear | torch.nn.Conv2d,
    added_units: int,
    fan_in: str = "kaiming_normal",
    fan_out: str = "zeros",
    generator: torch.Generator | None = None,
) -> Proposal:
    """New units with random fan-in and fan-out, without statistics.

    Port of v1's extension initializations (``GrowingModule.initialize_extensions``):
    ``"kaiming_normal"`` draws with std ``sqrt(2 / fan_in)``, using the producer
    fan-in for the new fan-in and the grown consumer fan-in for the new fan-out;
    ``"copy_normal"`` copies the std of the existing producer or consumer weights.
    v1's ``noise_ratio`` is not ported. The bias of the new units is zero. The
    paper modes of the init-mode sweep are B ``("kaiming_normal", "zeros")``,
    C ``("zeros", "kaiming_normal")``, D ``("kaiming_normal", "kaiming_normal")``
    and E ``("copy_normal", "copy_normal")``. For a convolution, the fan-out is a
    ``1 x 1`` weight at the consumer centre tap, as in the restricted scheme.

    Parameters
    ----------
    producer : torch.nn.Linear | torch.nn.Conv2d
        layer producing the hidden units
    consumer : torch.nn.Linear | torch.nn.Conv2d
        layer reading them
    added_units : int
        number of new units
    fan_in : str
        initialization of the new fan-in
    fan_out : str
        initialization of the new fan-out
    generator : torch.Generator | None
        random generator

    Returns
    -------
    Proposal
        new units without weight update; ``eigenvalues`` is all zeros
    """
    producer_fan_in = producer.weight[0].numel()
    consumer_fan_in = consumer.weight[0].numel()
    unit_fan_in = consumer.weight[0, 0].numel()
    alpha = _random_block(
        (added_units, producer_fan_in),
        fan_in,
        producer_fan_in,
        producer.weight,
        generator,
    )
    if producer.bias is not None:
        alpha = torch.cat([alpha, alpha.new_zeros(added_units, 1)], dim=1)
    omega = _random_block(
        (consumer.weight.shape[0], added_units),
        fan_out,
        consumer_fan_in + added_units * unit_fan_in,
        consumer.weight,
        generator,
    )
    zero = alpha.new_zeros(())
    return Proposal(None, alpha, omega, alpha.new_zeros(added_units), zero)
