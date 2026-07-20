"""Net2Wider helpers for channel-concat (multi-branch) fan-in remapping.

Provides graph-level and edge-level APIs that accept a shared replica map ``g``
plus concat channel offsets so remapping stays consistent across branches and
the post-concat consumer (Net2Net §2.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

from gromo.modules.growing_normalisation import (
    GrowingBatchNorm1d,
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)


if TYPE_CHECKING:
    from gromo.modules.growing_module import GrowingModule

WidenMode = Literal["net2wider", "random_pad"]


def _iter_post_layer_modules(module: nn.Module):
    if isinstance(module, nn.Sequential):
        for child in module:
            yield from _iter_post_layer_modules(child)
    else:
        yield module


def _reject_unsupported_post_layer(producer: GrowingModule) -> None:
    for module in _iter_post_layer_modules(producer.post_layer_function):
        if isinstance(module, (GrowingGroupNorm, GrowingLayerNorm)):
            raise ValueError(
                f"net2wider does not support {type(module).__name__} in "
                "post_layer_function (GrowingBatchNorm1d/2d only)."
            )
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ) and not isinstance(module, (GrowingBatchNorm1d, GrowingBatchNorm2d)):
            raise ValueError(
                "net2wider requires GrowingBatchNorm1d/2d in "
                f"post_layer_function, got plain {type(module).__name__}."
            )


def _prepare_producer_batch_norms(
    producer: GrowingModule, selected_indices: torch.Tensor
) -> None:
    _reject_unsupported_post_layer(producer)
    for module in _iter_post_layer_modules(producer.post_layer_function):
        if isinstance(module, (GrowingBatchNorm1d, GrowingBatchNorm2d)):
            module.prepare_net2wider_extension(selected_indices)


def _resolve_selected_indices(
    num_base: int,
    extension_size: int,
    selected_indices: torch.Tensor | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    if extension_size <= 0:
        raise ValueError(f"extension_size must be positive, got {extension_size}.")
    if selected_indices is None:
        selected = torch.randint(
            0,
            num_base,
            (extension_size,),
            generator=generator,
            dtype=torch.long,
        )
    else:
        selected = selected_indices.detach().to(dtype=torch.long).reshape(-1)
        if selected.numel() != extension_size:
            raise ValueError(
                f"selected_indices must have length {extension_size}, "
                f"got {selected.numel()}."
            )
        if bool((selected < 0).any() or (selected >= num_base).any()):
            raise ValueError(
                f"selected_indices must be in [0, {num_base}), "
                f"got min={int(selected.min())}, max={int(selected.max())}."
            )
    return selected.to(device=device)


@torch.no_grad()
def create_concat_net2wider_extensions(
    *,
    producer: GrowingModule,
    consumer: GrowingModule,
    extension_size: int,
    selected_indices: torch.Tensor | None = None,
    channel_offset: int = 0,
    mode: WidenMode = "net2wider",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Allocate + fill producer out-ext and consumer in-ext for concat fan-in.

    Parameters
    ----------
    producer
        Branch-ending layer whose *output* channels are widened.
    consumer
        Post-concat layer whose *input* channels consume the concat.
    extension_size
        Number of channels to add.
    selected_indices
        Shared replica map ``g`` into the producer's output axis. Required
        semantics for ``mode="net2wider"``; for ``random_pad`` still used only
        as a length/shape check when provided (pad values are random).
    channel_offset
        Concat channel offset where this producer sits in the consumer fan-in.
    mode
        ``"net2wider"`` (function-preserving) or ``"random_pad"`` (true random
        weight-tensor pad; does **not** scale existing fan-in).
    generator
        RNG for sampling ``g`` (net2wider) or random pad values.

    Returns
    -------
    torch.Tensor
        The replica index map ``g`` (for random_pad, the provided/sampled
        indices used only for bookkeeping / parity tests).
    """
    if mode not in ("net2wider", "random_pad"):
        raise ValueError(f"Unknown widen mode {mode!r}.")
    if channel_offset < 0:
        raise ValueError(f"channel_offset must be >= 0, got {channel_offset}.")

    groups = getattr(getattr(producer, "layer", None), "groups", 1)
    if groups != 1:
        raise ValueError(f"net2wider requires groups==1, got groups={groups}.")
    groups_c = getattr(getattr(consumer, "layer", None), "groups", 1)
    if groups_c != 1:
        raise ValueError(f"net2wider requires groups==1 on consumer, got {groups_c}.")

    producer.create_layer_out_extension(extension_size)
    consumer.create_layer_in_extension(extension_size)

    ext_out = producer.extended_output_layer
    ext_in = consumer.extended_input_layer
    assert ext_out is not None and ext_in is not None

    num_base = int(producer.weight.shape[0])
    if channel_offset + num_base > int(consumer.weight.shape[1]):
        raise ValueError(
            f"channel_offset={channel_offset} + producer out={num_base} exceeds "
            f"consumer in_channels={int(consumer.weight.shape[1])}."
        )

    selected = _resolve_selected_indices(
        num_base,
        extension_size,
        selected_indices,
        generator,
        device=producer.weight.device,
    )

    if mode == "net2wider":
        replica_counts = torch.ones(
            num_base, dtype=torch.float32, device=producer.weight.device
        )
        replica_counts.scatter_add_(
            0,
            selected,
            torch.ones(
                extension_size, dtype=torch.float32, device=producer.weight.device
            ),
        )

        ext_out.weight.copy_(producer.weight[selected])
        if ext_out.bias is not None:
            if producer.bias is None:
                raise ValueError("Producer has no bias but its output extension does.")
            ext_out.bias.copy_(producer.bias[selected])

        next_weight = consumer.weight
        for j in range(num_base):
            count = float(replica_counts[j].item())
            if count != 1.0:
                next_weight[:, channel_offset + j].div_(count)

        for i, src_idx in enumerate(selected.tolist()):
            ext_in.weight[:, i].copy_(next_weight[:, channel_offset + src_idx])

        if ext_in.bias is not None:
            nn.init.zeros_(ext_in.bias)

        _prepare_producer_batch_norms(producer, selected)
    else:
        # True random weight-tensor pad (paper §3.1): new channels are random;
        # existing consumer columns are left untouched (no 1/c scaling).
        if generator is None:
            nn.init.normal_(ext_out.weight, mean=0.0, std=0.05)
            if ext_out.bias is not None:
                nn.init.zeros_(ext_out.bias)
            nn.init.normal_(ext_in.weight, mean=0.0, std=0.05)
        else:
            # Manual sampling so tests can pin a generator.
            out_noise = torch.randn(
                ext_out.weight.shape,
                generator=generator,
                dtype=ext_out.weight.dtype,
                device=ext_out.weight.device,
            )
            ext_out.weight.copy_(out_noise * 0.05)
            if ext_out.bias is not None:
                ext_out.bias.zero_()
            in_noise = torch.randn(
                ext_in.weight.shape,
                generator=generator,
                dtype=ext_in.weight.dtype,
                device=ext_in.weight.device,
            )
            ext_in.weight.copy_(in_noise * 0.05)
        if ext_in.bias is not None:
            nn.init.zeros_(ext_in.bias)
        # Do not prepare BN replicas for random_pad (defaults on grow).

    producer.net2wider_selected_indices = selected
    consumer.net2wider_selected_indices = selected
    producer.output_extension_scaling = 1.0
    consumer.input_extension_scaling = 1.0
    consumer.scaling_factor = 1.0
    return selected


@torch.no_grad()
def apply_concat_net2wider_change(
    *,
    producer: GrowingModule,
    consumer: GrowingModule,
    extension_size: int,
    channel_offset: int,
) -> None:
    """Commit producer out-ext + consumer in-ext with mid-concat insertion.

    New consumer input channels are inserted immediately after the producer's
    base block at ``channel_offset + num_base``, matching channel-concat layout
    (replicas sit next to their source branch, not appended after later branches).
    """
    if consumer.extended_input_layer is None:
        raise ValueError("consumer.extended_input_layer is required before apply.")
    if producer.extended_output_layer is None and extension_size > 0:
        raise ValueError("producer.extended_output_layer is required before apply.")

    # Capture producer base width *before* output growth.
    num_base = int(producer.weight.shape[0])
    insert_at = channel_offset + num_base
    if insert_at > int(consumer.weight.shape[1]):
        raise ValueError(
            f"insert_at={insert_at} exceeds consumer in_channels="
            f"{int(consumer.weight.shape[1])}."
        )

    in_ext_scale = float(consumer.input_extension_scaling.item())
    ext_weight = in_ext_scale * consumer.extended_input_layer.weight
    if ext_weight.shape[1] != extension_size:
        raise ValueError(
            f"Input extension size {ext_weight.shape[1]} != extension_size="
            f"{extension_size}."
        )

    # Insert (do not append) so concat channel order stays:
    # [left branches | producer base | producer replicas | right branches].
    left = consumer.weight[:, :insert_at]
    right = consumer.weight[:, insert_at:]
    new_weight = torch.cat([left, ext_weight, right], dim=1)
    if not hasattr(consumer, "layer_of_tensor"):
        raise TypeError(
            f"Consumer {type(consumer).__name__} lacks layer_of_tensor; "
            "concat Net2Wider apply currently supports Conv consumers."
        )
    consumer.layer = consumer.layer_of_tensor(weight=new_weight, bias=consumer.bias)

    # Keep tensor-statistic shapes consistent with layer_in_extension.
    if hasattr(consumer, "_tensor_s") and hasattr(consumer, "tensor_m"):
        from gromo.utils.tensor_statistic import TensorStatistic

        ks0, ks1 = consumer.layer.kernel_size
        in_f = (consumer.in_channels + consumer.use_bias) * ks0 * ks1
        consumer._tensor_s = TensorStatistic(
            (in_f, in_f),
            update_function=consumer.compute_s_update,
            name=consumer.tensor_s.name,
        )
        consumer.tensor_m = TensorStatistic(
            (in_f, consumer.out_channels),
            update_function=consumer.compute_m_update,
            name=consumer.tensor_m.name,
        )

    consumer.extended_input_layer = None

    producer.output_extension_scaling = 1.0
    producer._apply_output_changes(scaling_factor=1.0, extension_size=extension_size)
    producer.extended_output_layer = None
    producer._clear_net2wider_pending_in_post_layer()
    producer.net2wider_selected_indices = None
    consumer.net2wider_selected_indices = None


@torch.no_grad()
def net2wider_widen_graph(
    *,
    producer: GrowingModule,
    consumers: list[tuple[GrowingModule, int]],
    extension_size: int,
    selected_indices: torch.Tensor | None = None,
    mode: WidenMode = "net2wider",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Graph-level widen: shared ``g`` + per-consumer concat offsets.

    Currently supports a single post-concat consumer (typical Inception
    projection). Multiple consumers must share the same extension fill.
    """
    if not consumers:
        raise ValueError("net2wider_widen_graph requires at least one consumer.")
    if len(consumers) != 1:
        raise ValueError(
            "net2wider_widen_graph currently supports exactly one consumer "
            f"(got {len(consumers)}). Multi-consumer fan-out is unsupported."
        )
    consumer, offset = consumers[0]
    return create_concat_net2wider_extensions(
        producer=producer,
        consumer=consumer,
        extension_size=extension_size,
        selected_indices=selected_indices,
        channel_offset=offset,
        mode=mode,
        generator=generator,
    )
