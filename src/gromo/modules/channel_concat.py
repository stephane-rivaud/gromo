"""Channel-concat fan-in (not GrowingDAG / Merge sum-merge)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelConcat(nn.Module):
    """Concatenate branch tensors along the channel axis (dim=1).

    This is the Inception-style fan-in. Do **not** replace with
    ``MergeGrowingModule`` / ``GrowingDAG`` sum fan-in.
    """

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        """Concatenate ``branches`` along dim=1 (channels)."""
        if not branches:
            raise ValueError("ChannelConcat requires at least one branch tensor.")
        return torch.cat(branches, dim=1)
