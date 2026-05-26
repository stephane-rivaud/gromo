from typing import Any, Dict, Literal

import torch
import torch.nn as nn
from torch import Tensor

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_transformer import GrowingTransformerBlock
from gromo.utils.utils import compute_tensor_stats


def check_patch_grid(
    image_height: int,
    image_width: int,
    patch_size: int | tuple[int, int],
) -> tuple[tuple[int, int], int]:
    """Validate the patch size and return the patch grid."""
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size

    num_patches_height, remainder_height = divmod(image_height, patch_height)
    num_patches_width, remainder_width = divmod(image_width, patch_width)

    assert remainder_height == 0, "`image_height` must be divisible by `patch_size`"
    assert remainder_width == 0, "`image_width` must be divisible by `patch_size`"

    return (patch_height, patch_width), num_patches_height * num_patches_width


class GrowingTransformer(GrowingContainer):
    """Vision transformer with fixed attention and growable feed-forward blocks."""

    def __init__(
        self,
        in_features: tuple[int, int, int] = (3, 32, 32),
        out_features: int = 10,
        patch_size: int | tuple[int, int] = 4,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_blocks: int = 8,
        dropout: float = 0.0,
        pooling: Literal["cls", "mean"] = "cls",
        use_cls_token: bool | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if len(in_features) != 3:
            raise ValueError(
                "`in_features` must be a tuple of the form (channels, height, width)."
            )
        if pooling not in {"cls", "mean"}:
            raise ValueError("`pooling` must be either 'cls' or 'mean'.")

        in_channels, image_height, image_width = in_features
        patch_size, num_patches = check_patch_grid(
            image_height,
            image_width,
            patch_size,
        )
        if use_cls_token is None:
            use_cls_token = pooling == "cls"
        if pooling == "cls" and not use_cls_token:
            raise ValueError("`pooling='cls'` requires `use_cls_token=True`.")

        super().__init__(
            in_features=torch.tensor(in_features).prod().int().item(),
            out_features=out_features,
            device=device,
            name="GrowingTransformer",
        )
        self.input_shape = in_features
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        self.patcher = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            device=self.device,
        )
        sequence_length = self.num_patches + int(self.use_cls_token)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, sequence_length, d_model, device=self.device)
        )
        self.cls_token: nn.Parameter | None = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, device=self.device))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks: nn.ModuleList[GrowingTransformerBlock] = nn.ModuleList(
            [
                GrowingTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model, device=self.device)
        self.classifier = nn.Linear(d_model, out_features, device=self.device)
        self.set_growing_layers()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize positional parameters with a small random scale."""
        nn.init.normal_(self.position_embeddings, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def set_growing_layers(self) -> None:
        """Reference each transformer block as an independent growth candidate."""
        self._growing_layers = list(self.blocks)

    def _embed_tokens(self, x: Tensor) -> Tensor:
        """Convert the input image to a sequence of patch embeddings."""
        x = self.patcher(x)
        batch_size, _, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embeddings[:, : x.size(1)]
        return self.embedding_dropout(x)

    def _pool_tokens(self, x: Tensor) -> Tensor:
        """Aggregate the encoded sequence into a single feature vector."""
        if self.pooling == "cls":
            return x[:, 0]
        if self.cls_token is not None:
            return x[:, 1:].mean(dim=1)
        return x.mean(dim=1)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the transformer classifier."""
        x = self._embed_tokens(x)
        for block in self.blocks:
            x = block(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(x)
        x = self._pool_tokens(x)
        return self.classifier(x)

    def extended_forward(
        self,
        x: Tensor,
        mask: dict | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass including the currently proposed GroMo extensions."""
        x = self._embed_tokens(x)
        for block in self.blocks:
            x = block.extended_forward(
                x,
                mask=mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(x)
        x = self._pool_tokens(x)
        return self.classifier(x)

    def update_information(self) -> Dict[str, Any]:
        """Collect growth information for each transformer block."""
        return {
            "blocks": {
                i: block.update_information() for i, block in enumerate(self.blocks)
            },
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_blocks": self.num_blocks,
        }

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect statistics for embeddings, blocks, and classifier head."""
        statistics: Dict[str, Any] = {
            "patcher": {
                "weight": compute_tensor_stats(self.patcher.weight),
            },
            "position_embeddings": compute_tensor_stats(self.position_embeddings),
            "blocks": {
                i: block.weights_statistics() for i, block in enumerate(self.blocks)
            },
            "classifier": {
                "weight": compute_tensor_stats(self.classifier.weight),
            },
        }
        if self.patcher.bias is not None:
            statistics["patcher"]["bias"] = compute_tensor_stats(self.patcher.bias)
        if self.cls_token is not None:
            statistics["cls_token"] = compute_tensor_stats(self.cls_token)
        if self.classifier.bias is not None:
            statistics["classifier"]["bias"] = compute_tensor_stats(self.classifier.bias)
        return statistics


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    model = GrowingTransformer(
        in_features=input_shape,
        out_features=10,
        patch_size=4,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_blocks=2,
    )

    x = torch.rand((2, *input_shape))
    y = model(x)
    print(model)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
