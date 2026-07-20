"""Growing BN-Inception with channel-concat fan-in (Net2Net §3.2 skeleton).

Uses :class:`ChannelConcat` (not GrowingDAG / Merge sum-merge). Channel tables
and √0.3 freeze live in :mod:`gromo.containers.bn_inception_channels`.
Full-graph Net2Wider remapping is M2b; this module provides construction,
forward shapes, and the module-level widen hooks used by the mini spike.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

from gromo.containers.bn_inception_channels import (
    FULL_INCEPTION_MODULES,
    STEM_CONV1_OUT,
    STEM_CONV2_OUT,
    STEM_CONV3_OUT,
    InceptionModuleChannels,
    assert_teacher_param_ratio,
    teacher_inception_modules,
)
from gromo.containers.growing_container import GrowingContainer
from gromo.modules.channel_concat import ChannelConcat
from gromo.modules.conv2d_growing_module import FullConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm2d


WidthPreset = Literal["full", "teacher_sqrt_0_3"]


def _bn_relu(num_features: int, device: torch.device, name: str) -> nn.Sequential:
    return nn.Sequential(
        GrowingBatchNorm2d(num_features=num_features, device=device, name=name),
        nn.ReLU(inplace=True),
    )


class GrowingInceptionModule(GrowingContainer):
    """One Inception block: 1x1 / 3x3 / 5x5 / pool branches → channel-concat."""

    def __init__(
        self,
        in_channels: int,
        channels: InceptionModuleChannels,
        device: torch.device | str | None = None,
        name: str = "inception",
    ) -> None:
        super().__init__(
            in_features=in_channels,
            out_features=channels.out_channels,
            device=device,
            name=name,
        )
        self.channels = channels
        c = channels

        self.branch1 = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=c.n1x1,
            kernel_size=1,
            padding=0,
            use_bias=False,
            device=self.device,
            name=f"{name}.b1",
            post_layer_function=_bn_relu(c.n1x1, self.device, f"{name}.bn1"),
        )

        self.branch3_reduce = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=c.n3x3_reduce,
            kernel_size=1,
            padding=0,
            use_bias=False,
            device=self.device,
            name=f"{name}.b3r",
            post_layer_function=_bn_relu(c.n3x3_reduce, self.device, f"{name}.bn3r"),
        )
        self.branch3 = FullConv2dGrowingModule(
            in_channels=c.n3x3_reduce,
            out_channels=c.n3x3,
            kernel_size=3,
            padding=1,
            use_bias=False,
            device=self.device,
            name=f"{name}.b3",
            previous_module=self.branch3_reduce,
            post_layer_function=_bn_relu(c.n3x3, self.device, f"{name}.bn3"),
        )

        self.branch5_reduce = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=c.n5x5_reduce,
            kernel_size=1,
            padding=0,
            use_bias=False,
            device=self.device,
            name=f"{name}.b5r",
            post_layer_function=_bn_relu(c.n5x5_reduce, self.device, f"{name}.bn5r"),
        )
        self.branch5 = FullConv2dGrowingModule(
            in_channels=c.n5x5_reduce,
            out_channels=c.n5x5,
            kernel_size=5,
            padding=2,
            use_bias=False,
            device=self.device,
            name=f"{name}.b5",
            previous_module=self.branch5_reduce,
            post_layer_function=_bn_relu(c.n5x5, self.device, f"{name}.bn5"),
        )

        self.branch_pool_proj = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=c.pool_proj,
            kernel_size=1,
            padding=0,
            use_bias=False,
            device=self.device,
            name=f"{name}.bpool",
            post_layer_function=_bn_relu(c.pool_proj, self.device, f"{name}.bnpool"),
        )
        self.concat = ChannelConcat()
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        """Register growable convolutions inside this module."""
        self._growing_layers = [
            self.branch1,
            self.branch3,
            self.branch5,
            self.branch_pool_proj,
        ]

    @property
    def concat_offsets(self) -> dict[str, int]:
        """Channel offsets of each branch inside the concat tensor."""
        o1 = 0
        o3 = self.branch1.out_channels
        o5 = o3 + self.branch3.out_channels
        op = o5 + self.branch5.out_channels
        return {"branch1": o1, "branch3": o3, "branch5": o5, "branch_pool": op}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through four branches and channel-concat."""
        b1 = self.branch1(x)
        b3 = self.branch3(self.branch3_reduce(x))
        b5 = self.branch5(self.branch5_reduce(x))
        bp = self.branch_pool_proj(
            functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        )
        return self.concat(b1, b3, b5, bp)

    def inception_parameter_count(self) -> int:
        """Trainable parameter count inside this Inception module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GrowingBNInception(GrowingContainer):
    """BN-Inception (GoogLeNet Table 1 + BN) with channel-concat Inception modules.

    Parameters
    ----------
    width_preset
        ``full`` student or ``teacher_sqrt_0_3`` (sqrt(0.3) inside modules).
    assert_param_ratio
        When building the teacher, also build a full twin and assert ~30% params.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        out_features: int = 1000,
        width_preset: WidthPreset = "full",
        assert_param_ratio: bool = True,
        device: torch.device | str | None = None,
        name: str = "bn_inception",
        **_ignored: object,
    ) -> None:
        if isinstance(input_shape, int):
            raise TypeError("BN-Inception requires CHW input_shape tuple.")
        in_ch, _, _ = input_shape
        super().__init__(
            in_features=in_ch,
            out_features=out_features,
            device=device,
            name=name,
        )
        self.width_preset = width_preset
        if width_preset == "full":
            modules = FULL_INCEPTION_MODULES
        elif width_preset == "teacher_sqrt_0_3":
            modules = teacher_inception_modules()
        else:
            raise ValueError(f"Unknown width_preset {width_preset!r}.")

        # Stem (unchanged under √0.3).
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_ch, STEM_CONV1_OUT, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(STEM_CONV1_OUT),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(STEM_CONV1_OUT, STEM_CONV2_OUT, kernel_size=1, bias=False),
            nn.BatchNorm2d(STEM_CONV2_OUT),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                STEM_CONV2_OUT, STEM_CONV3_OUT, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(STEM_CONV3_OUT),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        ).to(self.device)

        self.inceptions = nn.ModuleList()
        prev = STEM_CONV3_OUT
        pool_after = {"3b", "4e"}
        for spec in modules:
            block = GrowingInceptionModule(
                in_channels=prev,
                channels=spec,
                device=self.device,
                name=f"{name}.inc_{spec.name}",
            )
            self.inceptions.append(block)
            prev = spec.out_channels
            if spec.name in pool_after:
                self.inceptions.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # Classifier uses actual last inception out (teacher may differ from 1024).
        last_out = modules[-1].out_channels
        self.fc = nn.Linear(last_out, out_features, device=self.device)
        self._last_inception_out = last_out
        self.set_growing_layers()

        if width_preset == "teacher_sqrt_0_3" and assert_param_ratio:
            full = GrowingBNInception(
                input_shape=input_shape,
                out_features=out_features,
                width_preset="full",
                assert_param_ratio=False,
                device=self.device,
                name=f"{name}_full_ref",
            )
            t_params = self.inception_parameter_count()
            s_params = full.inception_parameter_count()
            assert_teacher_param_ratio(t_params, s_params)
            del full

    def set_growing_layers(self) -> None:
        """Collect growable Inception modules."""
        self._growing_layers = [
            m for m in self.inceptions if isinstance(m, GrowingInceptionModule)
        ]

    def inception_parameter_count(self) -> int:
        """Sum of trainable params in Inception modules only (stem/fc excluded)."""
        total = 0
        for m in self.inceptions:
            if isinstance(m, GrowingInceptionModule):
                total += m.inception_parameter_count()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stem → Inception stack → global pool → classifier."""
        x = self.stem(x)
        for layer in self.inceptions:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def init_bn_inception(
    input_shape: tuple[int, int, int],
    out_features: int,
    device: torch.device | str | None = None,
    width_preset: WidthPreset = "full",
    **kwargs: object,
) -> GrowingBNInception:
    """Builder entry point compatible with experimental_grow ``tools.models``."""
    return GrowingBNInception(
        input_shape=input_shape,
        out_features=out_features,
        device=device,
        width_preset=width_preset,
        **kwargs,
    )
