from typing import Any, Dict
from warnings import warn

import torch
import torch.nn as nn

from gromo.containers.growing_block import LinearGrowingBlock
from gromo.containers.growing_container import GrowingContainer
from gromo.utils.utils import compute_tensor_stats


class ResidualBlock(nn.Module):
    """Pre-norm residual wrapper for transformer sublayers."""

    def __init__(
        self,
        sublayer: nn.Module,
        d_model: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)
        self.sublayer = sublayer

    def forward(self, x, *args, **kwargs):
        """Apply layer normalization, the wrapped sublayer, and the residual add."""
        return x + self.sublayer(self.norm(x), *args, **kwargs)


class SelfAttentionLayer(nn.Module):
    """Multi-head self-attention sublayer for transformer blocks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Compute self-attention over a sequence of hidden states."""
        y, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return y


class GrowingTransformerBlock(GrowingContainer):
    """Transformer block with fixed attention and growable feed-forward branch."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            in_features=d_model,
            out_features=d_model,
            device=device,
            name="GrowingTransformerBlock",
        )
        self.attn_block = ResidualBlock(
            SelfAttentionLayer(
                d_model,
                num_heads,
                dropout,
                device=self.device,
            ),
            d_model,
            device=self.device,
        )
        self.mlp = LinearGrowingBlock(
            in_features=d_model,
            out_features=d_model,
            hidden_features=d_ff,
            pre_activation=nn.LayerNorm(d_model, device=self.device),
            mid_activation=nn.GELU(),
            pre_addition_function=nn.Identity(),
            name="mlp",
            kwargs_layer={"device": self.device},
            device=self.device,
        )
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        """Register the submodules that participate in the growth procedure."""
        self._growing_layers = [self.mlp]

    @property
    def optimal_delta_layer(self) -> torch.nn.Module | None:
        """Expose the optimal delta layer of the growable MLP branch."""
        return self.mlp.optimal_delta_layer

    @optimal_delta_layer.setter
    def optimal_delta_layer(self, value: torch.nn.Module | None) -> None:
        """Set the optimal delta layer on the growable MLP branch."""
        self.mlp.optimal_delta_layer = value

    @property
    def hidden_features(self) -> int:
        """Backward-compatible alias for the number of hidden neurons."""
        warn(
            "hidden_features is deprecated, use hidden_neurons instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.hidden_neurons

    @property
    def hidden_neurons(self) -> int:
        """Return the number of hidden neurons in the feed-forward branch."""
        return self.mlp.hidden_neurons

    @property
    def eigenvalues_extension(self) -> torch.Tensor | None:
        """Expose the current eigenvalues associated with the pending extension."""
        return self.mlp.eigenvalues_extension

    @property
    def parameter_update_decrease(self) -> torch.Tensor | None:
        """Expose the first-order decrease from the parameter-only update."""
        return self.mlp.parameter_update_decrease

    @parameter_update_decrease.setter
    def parameter_update_decrease(self, value: torch.Tensor | float) -> None:
        """Set the cached first-order decrease on the growable branch."""
        self.mlp.parameter_update_decrease = value

    @property
    def scaling_factor(self) -> torch.Tensor:
        """Expose the scaling factor used by the growable branch."""
        return self.mlp.scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, value: float) -> None:
        """Set the scaling factor used by the growable branch."""
        self.mlp.scaling_factor = value

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Return the total first-order improvement of the current proposal."""
        return self.mlp.first_order_improvement

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Run the attention branch followed by the growable feed-forward branch."""
        x = self.attn_block(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.mlp(x)
        return x

    def extended_forward(self, x, mask=None, attn_mask=None, key_padding_mask=None):
        """Run the block while applying any pending GroMo extensions in the MLP."""
        x = self.attn_block(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.mlp.extended_forward(x, mask=mask)
        return x

    def delete_update(self, **kwargs: Any) -> None:
        """Discard the currently cached growth proposal."""
        self.mlp.delete_update(**kwargs)

    def set_scaling_factor(self, factor: float) -> None:
        """Assign the same scaling factor to the growable branch."""
        self.mlp.set_scaling_factor(factor)

    def apply_change(
        self,
        extension_size: int | None = None,
        scaling_factor: float | torch.Tensor | None = None,
        apply_delta: bool = True,
        apply_extension: bool = True,
    ) -> None:
        """Apply the selected growth change to the MLP branch."""
        self.mlp.apply_change(
            extension_size=extension_size,
            scaling_factor=scaling_factor,
            apply_delta=apply_delta,
            apply_extension=apply_extension,
        )

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int | None = None,
        threshold: float | None = None,
        sub_select_previous: bool = True,
        zeros_if_not_enough: bool = False,
        zeros_fan_in: bool = True,
        zeros_fan_out: bool = False,
    ) -> None:
        """Keep only a subset of the proposed added neurons."""
        self.mlp.sub_select_optimal_added_parameters(
            keep_neurons=keep_neurons,
            threshold=threshold,
            sub_select_previous=sub_select_previous,
            zeros_if_not_enough=zeros_if_not_enough,
            zeros_fan_in=zeros_fan_in,
            zeros_fan_out=zeros_fan_out,
        )

    def update_information(self) -> Dict[str, Any]:
        """Summarize the current growth proposal for reporting and comparison."""
        eigenvalues = self.eigenvalues_extension
        return {
            "update_value": self.first_order_improvement,
            "parameter_improvement": self.parameter_update_decrease,
            "eigenvalues_extension": eigenvalues,
            "scaling_factor": self.scaling_factor,
            "added_neurons": 0 if eigenvalues is None else eigenvalues.shape[0],
            "d_model": self.in_features,
            "d_ff": self.hidden_neurons,
        }

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect summary statistics for attention and MLP parameters."""
        attn = self.attn_block.sublayer.attn
        statistics: Dict[str, Any] = {
            "attention": {
                "in_proj_weight": compute_tensor_stats(attn.in_proj_weight),
                "out_proj_weight": compute_tensor_stats(attn.out_proj.weight),
            },
            "mlp": self.mlp.weights_statistics(),
            "d_model": self.in_features,
            "d_ff": self.hidden_neurons,
        }
        if attn.in_proj_bias is not None:
            statistics["attention"]["in_proj_bias"] = compute_tensor_stats(
                attn.in_proj_bias
            )
        if attn.out_proj.bias is not None:
            statistics["attention"]["out_proj_bias"] = compute_tensor_stats(
                attn.out_proj.bias
            )
        return statistics
