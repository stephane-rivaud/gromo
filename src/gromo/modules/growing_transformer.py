import torch
import torch.nn as nn


from gromo.modules.linear_growing_module import LinearGrowingModule



class  ResidualBlock(nn.Module):
    """Class encapsulating Residal blocks with internal subalayer"""""
    def __init__(self, sublayer, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x, *args, **kwargs):
        return x + self.sublayer(self.norm(x), *args, **kwargs)
    

class SelfAttentionLayer(nn.Module):
    # x: [batch, seq_len, d_model]
    """Multihead attention Module for self attention layer of the Transformer Block"""   
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        y, _ = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return y


class GrowingLinearBlock(nn.Module):
    """Class defining linear block for the """
    def __init__(self, d_model, d_ff):
        super().__init__()
        "Growing net by growing linear blocks"
        self.mlp_1 = LinearGrowingModule(
            in_features=d_model,
            out_features=d_ff,
            use_bias=True,
            post_layer_function=torch.nn.GELU(),
            name="mlp_1",
        )

        self.mlp_2 = LinearGrowingModule(
            in_features=d_ff,
            out_features=d_model,
            use_bias=True,
            post_layer_function=torch.nn.Identity(),
            previous_module=self.mlp_1,
            name="mlp_2",
        )
        self.net = nn.Sequential(
            self.mlp_1,
            self.mlp_2,
        )
    def forward(self, x):
        return self.net(x)
 


class GrowingTransformerBlock(nn.Module):
    """Transformer block with standard attention residual and growing MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attnlayer = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mlp = GrowingLinearBlock(d_model=d_model, d_ff=d_ff)

        self.attn_block = ResidualBlock(sublayer=self.attnlayer, d_model=d_model)
        self.mlp_block = ResidualBlock(sublayer=self.mlp, d_model=d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.attn_block(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.mlp_block(x)
        return x
    
    