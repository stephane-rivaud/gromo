"""Mini Inception container with channel-concat fan-in for Net2Wider M2a.

Topology (no residual shortcuts; not GrowingDAG sum-merge)::

    input
      ├─ branch_a: Conv1x1 → GrowingBN → ReLU
      └─ branch_b: Conv1x1 → GrowingBN → ReLU → Conv3x3 → GrowingBN → ReLU
    ChannelConcat([branch_a, branch_b])
      └─ project: Conv1x1 → GrowingBN → ReLU
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.channel_concat import ChannelConcat
from gromo.modules.conv2d_growing_module import FullConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm2d
from gromo.utils.net2wider_concat import (
    WidenMode,
    apply_concat_net2wider_change,
    create_concat_net2wider_extensions,
)


EdgeName = Literal["branch_a", "branch_b", "branch_b_reduce", "residual"]


class MiniInceptionGrowingModule(GrowingContainer):
    """Two-branch channel-concat mini Inception for Net2Wider FP spike."""

    def __init__(
        self,
        *,
        in_channels: int,
        branch_a_channels: int,
        branch_b_reduce_channels: int,
        branch_b_channels: int,
        out_channels: int,
        device: torch.device | str | None = None,
        name: str = "mini_inception",
    ) -> None:
        super().__init__(
            in_features=in_channels,
            out_features=out_channels,
            device=device,
            name=name,
        )
        self.branch_a_channels = branch_a_channels
        self.branch_b_reduce_channels = branch_b_reduce_channels
        self.branch_b_channels = branch_b_channels

        bn_a = GrowingBatchNorm2d(
            num_features=branch_a_channels, device=self.device, name=f"{name}.bn_a"
        )
        self.branch_a = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=branch_a_channels,
            kernel_size=1,
            padding=0,
            use_bias=True,
            device=self.device,
            name=f"{name}.branch_a",
            post_layer_function=nn.Sequential(bn_a, nn.ReLU(inplace=True)),
        )

        bn_b_red = GrowingBatchNorm2d(
            num_features=branch_b_reduce_channels,
            device=self.device,
            name=f"{name}.bn_b_reduce",
        )
        self.branch_b_reduce = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=branch_b_reduce_channels,
            kernel_size=1,
            padding=0,
            use_bias=True,
            device=self.device,
            name=f"{name}.branch_b_reduce",
            post_layer_function=nn.Sequential(bn_b_red, nn.ReLU(inplace=True)),
        )
        bn_b = GrowingBatchNorm2d(
            num_features=branch_b_channels, device=self.device, name=f"{name}.bn_b"
        )
        self.branch_b_expand = FullConv2dGrowingModule(
            in_channels=branch_b_reduce_channels,
            out_channels=branch_b_channels,
            kernel_size=3,
            padding=1,
            use_bias=True,
            device=self.device,
            name=f"{name}.branch_b_expand",
            previous_module=self.branch_b_reduce,
            post_layer_function=nn.Sequential(bn_b, nn.ReLU(inplace=True)),
        )

        concat_channels = branch_a_channels + branch_b_channels
        bn_proj = GrowingBatchNorm2d(
            num_features=out_channels, device=self.device, name=f"{name}.bn_proj"
        )
        self.project = FullConv2dGrowingModule(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            use_bias=True,
            device=self.device,
            name=f"{name}.project",
            post_layer_function=nn.Sequential(bn_proj, nn.ReLU(inplace=True)),
        )
        self.concat = ChannelConcat()
        self._pending_edge: EdgeName | None = None
        self._pending_offset: int | None = None
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        """Register growable edges for container-level iteration."""
        self._growing_layers = [
            self.branch_a,
            self.branch_b_expand,
            self.project,
        ]

    @property
    def concat_offsets(self) -> dict[str, int]:
        """Channel offsets of each branch inside the concat tensor."""
        return {
            "branch_a": 0,
            "branch_b": int(self.branch_a.out_channels),
        }

    @property
    def concat_width(self) -> int:
        """Total channel width after concatenating both branches."""
        return int(self.branch_a.out_channels) + int(self.branch_b_expand.out_channels)

    def _refresh_concat_layout(self) -> None:
        """Keep tracked branch widths in sync after a widen apply."""
        self.branch_a_channels = int(self.branch_a.out_channels)
        self.branch_b_reduce_channels = int(self.branch_b_reduce.out_channels)
        self.branch_b_channels = int(self.branch_b_expand.out_channels)
        self.in_features = int(self.branch_a.in_channels)
        self.out_features = int(self.project.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: parallel branches, channel-concat, then 1x1 project."""
        a = self.branch_a(x)
        b = self.branch_b_expand(self.branch_b_reduce(x))
        return self.project(self.concat(a, b))

    def net2wider_widen(
        self,
        *,
        edge: EdgeName,
        extension_size: int,
        selected_indices: torch.Tensor | None = None,
        mode: WidenMode = "net2wider",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Module-level widen entry: shared ``g`` + concat offsets.

        Call :meth:`apply_net2wider_widen` immediately after for FP probes
        (create → immediate apply; no train between).
        """
        if edge == "residual":
            raise ValueError(
                "net2wider rejects residual/shortcut growth on MiniInception "
                "(unsupported; abort rather than silent wrong FP)."
            )
        if edge not in ("branch_a", "branch_b", "branch_b_reduce"):
            raise ValueError(f"Unknown mini-inception edge {edge!r}.")

        if edge == "branch_b_reduce":
            # Internal pairwise edge: standard joint Net2Wider on reduce→expand.
            self.branch_b_expand.create_layer_extensions(
                extension_size=extension_size,
                output_extension_init="net2wider" if mode == "net2wider" else "kaiming",
                input_extension_init="net2wider" if mode == "net2wider" else "kaiming",
                selected_indices=selected_indices if mode == "net2wider" else None,
                generator=generator,
            )
            if mode == "random_pad":
                # Overwrite extensions with true random pad (not kaiming slice proxy).
                assert self.branch_b_reduce.extended_output_layer is not None
                assert self.branch_b_expand.extended_input_layer is not None
                nn.init.normal_(
                    self.branch_b_reduce.extended_output_layer.weight, mean=0.0, std=0.05
                )
                if self.branch_b_reduce.extended_output_layer.bias is not None:
                    self.branch_b_reduce.extended_output_layer.bias.zero_()
                nn.init.normal_(
                    self.branch_b_expand.extended_input_layer.weight, mean=0.0, std=0.05
                )
                # Undo any net2wider-style scaling if create path scaled (kaiming path
                # does not scale existing fan-in).
            g = (
                self.branch_b_expand.net2wider_selected_indices
                if mode == "net2wider"
                else (
                    selected_indices
                    if selected_indices is not None
                    else torch.zeros(extension_size, dtype=torch.long)
                )
            )
            assert g is not None
            self._pending_edge = edge
            return g

        producer = self.branch_a if edge == "branch_a" else self.branch_b_expand
        offset = self.concat_offsets["branch_a" if edge == "branch_a" else "branch_b"]
        g = create_concat_net2wider_extensions(
            producer=producer,
            consumer=self.project,
            extension_size=extension_size,
            selected_indices=selected_indices,
            channel_offset=offset,
            mode=mode,
            generator=generator,
        )
        self._pending_edge = edge
        self._pending_offset = offset
        return g

    def apply_net2wider_widen(self, *, edge: EdgeName, extension_size: int) -> None:
        """Apply a pending :meth:`net2wider_widen` (immediate apply)."""
        if self._pending_edge is not None and self._pending_edge != edge:
            raise ValueError(
                f"Pending widen edge is {self._pending_edge!r}, got apply for {edge!r}."
            )
        if edge == "branch_b_reduce":
            self.branch_b_expand.apply_change(
                scaling_factor=1.0, extension_size=extension_size
            )
            self.branch_b_expand.delete_update(include_previous=True)
            self._pending_edge = None
            self._pending_offset = None
            self._refresh_concat_layout()
            return
        if edge == "branch_a":
            producer = self.branch_a
        elif edge == "branch_b":
            producer = self.branch_b_expand
        else:
            raise ValueError(f"Unknown mini-inception edge {edge!r}.")
        if self._pending_offset is None:
            raise ValueError("No pending concat offset; call net2wider_widen first.")
        apply_concat_net2wider_change(
            producer=producer,
            consumer=self.project,
            extension_size=extension_size,
            channel_offset=self._pending_offset,
        )
        self._pending_edge = None
        self._pending_offset = None
        self._refresh_concat_layout()
