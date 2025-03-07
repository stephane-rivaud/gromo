import torch
from torch import nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


class GrowingModel(torch.nn.Module):
    def __init__(
        self,
    ):
        super(GrowingModel, self).__init__()

        self.growing_layers = nn.ModuleList()
        self.currently_updated_layer = None
        self.currenlty_updated_layer_index = None

    def set_growing_layers(self):
        """
        Reference all growable layers of the model in the growing_layers attribute. This method should be implemented
        in the child class and called in the __init__ method.
        """
        raise NotImplementedError

    def forward(self, x):
        """Forward pass through the network"""
        raise NotImplementedError

    def extended_forward(self, x):
        """Extended forward pass through the network"""
        raise NotImplementedError

    def init_computation(self):
        """Initialize statistics computations for growth procedure"""
        for layer in self.growing_layers:
            layer.init_computation()

    def reset_computation(self):
        """Reset statistics computations for growth procedure"""
        for layer in self.growing_layers:
            layer.reset_computation()

    def update_computation(self):
        """Update statistics computations for growth procedure"""
        for layer in self.growing_layers:
            layer.update_computation()

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-10,
        statistical_threshold: float = 1e-5,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Compute optimal update for growth procedure"""
        for layer in self.growing_layers:
            if part == "parameter":
                layer.compute_optimal_delta(dtype=dtype)
            elif part == "neuron":
                layer.compute_optimal_updates(
                    zero_delta=True,
                    numerical_threshold=numerical_threshold,
                    statistical_threshold=statistical_threshold,
                    maximum_added_neurons=maximum_added_neurons,
                    dtype=dtype,
                )
                layer.optimal_delta_layer = None  # makes extended_forward more efficient
                layer.parameter_update_decrease = 0
            elif part == "all":
                layer.compute_optimal_updates(
                    numerical_threshold=numerical_threshold,
                    statistical_threshold=statistical_threshold,
                    maximum_added_neurons=maximum_added_neurons,
                    dtype=dtype,
                )

    def select_best_update(self, verbose: bool = False) -> int:
        """Select the best update for growth procedure"""
        update_values = [layer.first_order_improvement for layer in self.growing_layers]
        max_update = max(update_values)
        for i, layer in enumerate(self.growing_layers):
            if verbose:
                print(f"Layer {i} update: {update_values[i]}")
                print(
                    f"Layer {i} parameter improvement: {layer.parameter_update_decrease}"
                )
                print(f"Layer {i} eigenvalues extension: {layer.eigenvalues_extension}")
            if self.updates_values[i] < max_update:
                if verbose:
                    print(f"Deleting layer {i}")
                layer.delete_update()
            else:
                self.currently_updated_layer = layer
                self.currently_updated_layer_index = i
        return self.currently_updated_layer_index

    def apply_update(self):
        self.currently_updated_layer.apply_change()
        self.currently_updated_layer.delete_update()
        self.currently_updated_layer = None
        self.currently_updated_layer_index = None

    @property
    def scaling_factor(self):
        assert self.currently_updated_layer is not None, "No layer is currently updated"
        return self.currently_updated_layer.scaling_factor

    def __setattr__(self, key, value):
        if key == "scaling_factor":
            for layer in self.layers:
                layer.scaling_factor = value
        else:
            nn.Module.__setattr__(self, key, value)
