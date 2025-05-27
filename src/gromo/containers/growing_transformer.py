from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_e = d_e
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.scale = d_k ** -0.5

        # Linear projections for Q, K, V
        self.query = nn.Linear(d_e, d_k * num_heads, bias=bias)
        self.key = nn.Linear(d_e, d_k * num_heads, bias=bias)
        self.value = nn.Linear(d_e, d_v * num_heads, bias=bias)
        
        # Output projection
        self.proj = nn.Linear(d_v * num_heads, d_e, bias=bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project and split into multiple heads
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        bias: bool = True,
        width_factor: int = 1,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
        self.attn = MultiHeadAttention(d_e, d_k, d_v, num_heads, bias)
        self.ln2 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
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
        dim_e: int = 128,
        dim_k: int = 64,
        dim_v: int = 64,
        n_heads: int = 1,
        num_blocks: int = 1,
        width_factor: int = 1,
        use_bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super(GrowingTransformer, self).__init__()
        self.device = device
        self.patcher = nn.Conv2d(
            in_features[0],
            dim_e,
            kernel_size=patch_size,
            stride=patch_size,
            device=self.device,
        )
        num_patches = in_features[1] // patch_size * in_features[2] // patch_size
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches, dim_e, device=self.device)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim_e, dim_k, dim_v, n_heads, width_factor=width_factor, bias=use_bias)
                for _ in range(num_blocks)
            ]
        )
        self.projection = nn.Linear(dim_e, out_features, device=self.device)
    
    def embedding(self, x):
        patches = self.patcher(x)
        batch_size, dim_e, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(batch_size, -1, dim_e)
        embedding = patches + self.pos_emb
        return embedding

    def forward(self, x):
        embedding = self.embedding(x)
        for block in self.blocks:
            embedding = block(embedding)
        return self.projection(embedding.mean(dim=1))


if __name__ == "__main__":
    model = GrowingTransformer(
        in_features=(3, 32, 32),
        out_features=10,
        patch_size=2,
        dim_e=128,
        n_heads=2,
        dim_k=32,
        dim_v=64,
        num_blocks=1,
        width_factor=2,
        use_bias=True,
        device=torch.device("cpu"),
    )
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
