from typing import Optional

import torch
import torch.nn as nn

from gromo.modules.attention.model import ModelConfig, SelfAttentionBaseline


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        bias: bool = False,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionBaseline(ModelConfig(d_e=d_e, d_k=d_k, d_v=d_v, bias=bias))
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return sum(head(x) for head in self.heads)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
        self.attn = MultiHeadAttention(d_e, d_k, d_v, num_heads, bias)
        self.ln2 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
        width_factor = 2
        self.mlp = nn.Sequential(
            nn.Linear(d_e, width_factor * d_e, bias=bias),
            nn.GELU(),
            nn.Linear(width_factor * d_e, d_e, bias=bias),
        )

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x


class GrowingTransformer(nn.Module):
    def __init__(
        self,
        in_features: tuple[int, int, int] = (3, 32, 32),
        out_features: int = 10,
        patch_size: int = 4,
        num_features: int = 128,
        n_heads: int = 1,
        dim_k: int = 64,
        dim_v: int = 64,
        num_blocks: int = 1,
        device: Optional[torch.device] = None,
    ):
        super(GrowingTransformer, self).__init__()
        self.device = device
        self.patcher = nn.Conv2d(
            in_features[0],
            num_features,
            kernel_size=patch_size,
            stride=patch_size,
            device=self.device,
        )
        num_patches = in_features[1] // patch_size * in_features[2] // patch_size
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches, num_features, device=self.device)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(num_features, dim_k, dim_v, n_heads, bias=True)
                for _ in range(num_blocks)
            ]
        )
        self.projection = nn.Linear(num_features, out_features, device=self.device)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(batch_size, -1, num_features)
        patches += self.pos_emb
        for block in self.blocks:
            patches = block(patches)
        return self.projection(patches.mean(dim=1))


if __name__ == "__main__":
    model = GrowingTransformer(
        in_features=(3, 32, 32),
        out_features=10,
        patch_size=2,
        num_features=128,
        n_heads=2,
        dim_k=32,
        dim_v=64,
        num_blocks=1,
        device=torch.device("cpu"),
    )
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
