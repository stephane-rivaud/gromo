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
    """Allocate + fill producer out-ext and one consumer in-ext (see multi)."""
    return create_concat_net2wider_extensions_multi(
        producer=producer,
        consumers=[(consumer, channel_offset)],
        extension_size=extension_size,
        selected_indices=selected_indices,
        mode=mode,
        generator=generator,
    )


@torch.no_grad()
def create_concat_net2wider_extensions_multi(
    *,
    producer: GrowingModule,
    consumers: list[tuple[GrowingModule, int]],
    extension_size: int,
    selected_indices: torch.Tensor | None = None,
    mode: WidenMode = "net2wider",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shared ``g`` + concat offsets for one producer and many consumers.

    All consumers typically are the entry convolutions of the next Inception
    module (each sees the full concat; same offset for the widened branch).
    """
    if mode not in ("net2wider", "random_pad"):
        raise ValueError(f"Unknown widen mode {mode!r}.")
    if not consumers:
        raise ValueError("consumers must be non-empty.")

    groups = getattr(getattr(producer, "layer", None), "groups", 1)
    if groups != 1:
        raise ValueError(f"net2wider requires groups==1, got groups={groups}.")

    producer.create_layer_out_extension(extension_size)
    ext_out = producer.extended_output_layer
    assert ext_out is not None

    num_base = int(producer.weight.shape[0])
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
    else:
        if generator is None:
            nn.init.normal_(ext_out.weight, mean=0.0, std=0.05)
            if ext_out.bias is not None:
                nn.init.zeros_(ext_out.bias)
        else:
            out_noise = torch.randn(
                ext_out.weight.shape,
                generator=generator,
                dtype=ext_out.weight.dtype,
                device=ext_out.weight.device,
            )
            ext_out.weight.copy_(out_noise * 0.05)
            if ext_out.bias is not None:
                ext_out.bias.zero_()

    for consumer, channel_offset in consumers:
        if channel_offset < 0:
            raise ValueError(f"channel_offset must be >= 0, got {channel_offset}.")
        groups_c = getattr(getattr(consumer, "layer", None), "groups", 1)
        if groups_c != 1:
            raise ValueError(f"net2wider requires groups==1 on consumer, got {groups_c}.")
        if channel_offset + num_base > int(consumer.weight.shape[1]):
            raise ValueError(
                f"channel_offset={channel_offset} + producer out={num_base} exceeds "
                f"consumer in_channels={int(consumer.weight.shape[1])}."
            )
        consumer.create_layer_in_extension(extension_size)
        ext_in = consumer.extended_input_layer
        assert ext_in is not None

        if mode == "net2wider":
            next_weight = consumer.weight
            for j in range(num_base):
                count = float(replica_counts[j].item())
                if count != 1.0:
                    next_weight[:, channel_offset + j].div_(count)
            for i, src_idx in enumerate(selected.tolist()):
                ext_in.weight[:, i].copy_(next_weight[:, channel_offset + src_idx])
            if ext_in.bias is not None:
                nn.init.zeros_(ext_in.bias)
        else:
            if generator is None:
                nn.init.normal_(ext_in.weight, mean=0.0, std=0.05)
            else:
                in_noise = torch.randn(
                    ext_in.weight.shape,
                    generator=generator,
                    dtype=ext_in.weight.dtype,
                    device=ext_in.weight.device,
                )
                ext_in.weight.copy_(in_noise * 0.05)
            if ext_in.bias is not None:
                nn.init.zeros_(ext_in.bias)

        consumer.net2wider_selected_indices = selected
        consumer.input_extension_scaling = 1.0
        consumer.scaling_factor = 1.0

    if mode == "net2wider":
        _prepare_producer_batch_norms(producer, selected)

    producer.net2wider_selected_indices = selected
    producer.output_extension_scaling = 1.0
    return selected


def _insert_consumer_input_extension(
    consumer: GrowingModule,
    *,
    extension_size: int,
    channel_offset: int,
    num_base: int,
) -> None:
    if consumer.extended_input_layer is None:
        raise ValueError("consumer.extended_input_layer is required before apply.")
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
    left = consumer.weight[:, :insert_at]
    right = consumer.weight[:, insert_at:]
    new_weight = torch.cat([left, ext_weight, right], dim=1)
    if not hasattr(consumer, "layer_of_tensor"):
        raise TypeError(
            f"Consumer {type(consumer).__name__} lacks layer_of_tensor; "
            "concat Net2Wider apply currently supports Conv consumers."
        )
    consumer.layer = consumer.layer_of_tensor(weight=new_weight, bias=consumer.bias)
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
    consumer.net2wider_selected_indices = None


@torch.no_grad()
def apply_concat_net2wider_change(
    *,
    producer: GrowingModule,
    consumer: GrowingModule,
    extension_size: int,
    channel_offset: int,
) -> None:
    """Commit producer out-ext + one consumer in-ext with mid-concat insertion."""
    apply_concat_net2wider_change_multi(
        producer=producer,
        consumers=[(consumer, channel_offset)],
        extension_size=extension_size,
    )


@torch.no_grad()
def apply_concat_net2wider_change_multi(
    *,
    producer: GrowingModule,
    consumers: list[tuple[GrowingModule, int]],
    extension_size: int,
) -> None:
    """Commit producer out-ext + many consumer in-exts (shared ``g``)."""
    if producer.extended_output_layer is None and extension_size > 0:
        raise ValueError("producer.extended_output_layer is required before apply.")
    if not consumers:
        raise ValueError("consumers must be non-empty.")

    num_base = int(producer.weight.shape[0])
    for consumer, channel_offset in consumers:
        _insert_consumer_input_extension(
            consumer,
            extension_size=extension_size,
            channel_offset=channel_offset,
            num_base=num_base,
        )

    producer.output_extension_scaling = 1.0
    producer._apply_output_changes(scaling_factor=1.0, extension_size=extension_size)
    producer.extended_output_layer = None
    producer._clear_net2wider_pending_in_post_layer()
    producer.net2wider_selected_indices = None


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
    """Graph-level widen: shared ``g`` + per-consumer concat offsets."""
    return create_concat_net2wider_extensions_multi(
        producer=producer,
        consumers=consumers,
        extension_size=extension_size,
        selected_indices=selected_indices,
        mode=mode,
        generator=generator,
    )
