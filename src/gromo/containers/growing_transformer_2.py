from typing import Optional

import torch
import torch.nn as nn

from gromo.utils.tensor_statistic import TensorStatistic


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super(SelfAttention, self).__init__()
        self.d_e = d_e
        self.d_k = d_k
        self.d_v = d_v
        self.scale = d_k**-0.5

        # Linear projections for Q, K, V
        self.query = nn.Linear(d_e, d_k, bias=bias)
        self.key = nn.Linear(d_e, d_k, bias=bias)
        self.value = nn.Linear(d_e, d_v, bias=bias)

        # Output projection
        self.proj = nn.Linear(d_v, d_e, bias=bias)
        # TODO: check how the initialization of the bias affects performance (compared to growing_transformer)

        # Growth attributes
        self.store_input = False
        self.input = None
        self.store_attn_scores = False
        self.attn_scores = None
        self.scaling_factor = 0.0

        self.tensor_sigma = TensorStatistic(
            shape=(d_e**2, d_e**2),
            update_function=self.compute_sigma_update,
            device=device,
        )

        self.tensor_b = TensorStatistic(
            shape=(d_e**2,),
            update_function=self.compute_b_update,
            device=device,
        )

        self.tensor_dz: torch.Tensor = torch.empty((d_e, d_e), device=device)

    def forward(self, x):
        self.tensor_sigma.updated = False
        self.tensor_b.updated = False
        if self.store_input:
            self.input = x

        batch_size, seq_len, _ = x.size()

        # Project and split into multiple heads
        q = self.query(x).view(batch_size, seq_len, self.d_k)
        k = self.key(x).view(batch_size, seq_len, self.d_k)
        v = self.value(x).view(batch_size, seq_len, self.d_v)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Store the scores
        if self.store_attn_scores:
            self.attn_scores = scores
            self.attn_scores.retain_grad()

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Project
        out = self.proj(out)

        return out

    def extended_forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project and split into multiple heads
        q = self.query(x).view(batch_size, seq_len, self.d_k)
        k = self.key(x).view(batch_size, seq_len, self.d_k)
        v = self.value(x).view(batch_size, seq_len, self.d_v)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add the optimal dZ contribution
        if self.tensor_dz is not None:
            ds = torch.einsum("nab,bd,ncd->nac", x, self.tensor_dz, x)
            scores += ds * self.scaling_factor

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Project
        out = self.proj(out)

        return out

    def init_computation(self):
        self.store_input = True
        self.store_attn_scores = True
        self.tensor_sigma.init()
        self.tensor_b.init()

    def update_computation(self):
        self.tensor_sigma.update()
        self.tensor_b.update()

    def reset_computation(self):
        self.store_input = False
        self.store_attn_scores = False
        self.tensor_sigma.reset()
        self.tensor_b.reset()

    def compute_sigma_update(self):
        """
        Compute the update of the tensor sigma.

        Returns
        -------
        torch.Tensor
            update of the tensor sigma
        int
            number of samples used to compute the update
        """
        assert (
            self.store_input
        ), f"The input must be stored to compute the update of sigma. (error in {self.name})"
        assert (
            self.input is not None
        ), f"The input must be stored to compute the update of sigma. (error in {self.name})"
        cross_covariance = torch.matmul(self.input.transpose(-2, -1), self.input)
        kronecker = torch.einsum(
            "iab,icd->iacbd", cross_covariance, cross_covariance
        ).sum(dim=0)
        kronecker = kronecker.view(self.d_e**2, self.d_e**2)
        return kronecker, self.input.shape[0]

    def compute_b_update(self):
        """
        Compute the update of the tensor b.

        Returns
        -------
        torch.Tensor
            update of the tensor b
        int
            number of samples used to compute the update
        """
        assert (
            self.store_input
        ), f"The input must be stored to compute the update of b. (error in {self.name})"
        assert (
            self.input is not None
        ), f"The input must be stored to compute the update of b. (error in {self.name})"
        input_x = self.input
        scores_grad = self.attn_scores.grad
        b = torch.einsum("nab,nap,npq->bq", input_x, scores_grad, input_x).ravel()
        return b, self.input.shape[0]

    def compute_optimal_update(self):
        """
        Compute the optimal update for the tensor Z.

        Returns
        -------
        torch.Tensor
            optimal update dZ for the tensor Z
        """
        sigma = self.tensor_sigma()
        b = self.tensor_b()
        self.tensor_dz = torch.linalg.lstsq(sigma, b).solution.view(self.d_e, self.d_e)
        # sigma_pinv = torch.linalg.pinv(sigma)
        # self.tensor_dz = torch.matmul(sigma_pinv, b).view(self.d_e, self.d_e)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_e, d_k, d_v, bias, device=device) for _ in range(num_heads)]
        )

        self._growing_layers = list(self.heads)

    def forward(self, x):
        out = sum(attn(x) for attn in self.heads)
        return out

    def extended_forward(self, x):
        out = sum(attn.extended_forward(x) for attn in self.heads)
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
        device: torch.device | str | None = None,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
        self.attn = MultiHeadAttention(d_e, d_k, d_v, num_heads, bias, device=device)
        self.ln2 = nn.LayerNorm(d_e, eps=1e-5, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(d_e, width_factor * d_e, bias=bias),
            nn.GELU(),
            nn.Linear(width_factor * d_e, d_e, bias=bias),
        )

        self._growing_layers = self.attn._growing_layers

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def extended_forward(self, x):
        x = x + self.attn.extended_forward(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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
                TransformerBlock(
                    dim_e,
                    dim_k,
                    dim_v,
                    n_heads,
                    width_factor=width_factor,
                    bias=use_bias,
                    device=self.device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.projection = nn.Linear(dim_e, out_features, device=self.device)

        self._growing_layers = []
        for block in self.blocks:
            self._growing_layers.extend(block._growing_layers)

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

    def extended_forward(self, x):
        embedding = self.embedding(x)
        for block in self.blocks:
            embedding = block.extended_forward(embedding)
        return self.projection(embedding.mean(dim=1))

    def init_computation(self):
        for layer in self._growing_layers:
            layer.init_computation()

    def update_computation(self):
        for layer in self._growing_layers:
            layer.update_computation()

    def reset_computation(self):
        for layer in self._growing_layers:
            layer.reset_computation()

    def compute_optimal_update(self):
        for layer in self._growing_layers:
            layer.compute_optimal_update()


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)

    # Create model
    model = GrowingTransformer(
        in_features=(3, 32, 32),
        out_features=10,
        patch_size=4,
        dim_e=128,
        n_heads=1,
        dim_k=64,
        dim_v=64,
        num_blocks=1,
        width_factor=1,
        use_bias=True,
    )

    # Test the forward
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)

    # Test the growth
    batch_size = 8
    model.blocks[0].attn.heads[0].init_computation()
    for i in range(10):
        print(f"Step {i}")
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        model.blocks[0].attn.heads[0].update_computation()
    model.blocks[0].attn.heads[0].compute_optimal_update()

    # Test the extended forward
    x = torch.randn(1, 3, 32, 32)
    model.blocks[0].attn.heads[0].scaling_factor = 1.0
    y = model.extended_forward(x)
    print(y.shape)
