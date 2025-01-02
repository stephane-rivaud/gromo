"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

import torch

from gromo.growing_module import AdditionGrowingModule, GrowingModule
from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.tensor_statistic import TensorStatistic


all_layer_types = {
    "linear": {"layer": LinearGrowingModule, "addition": LinearAdditionGrowingModule},
}


class GrowingResidualBlock(torch.nn.Module):
    """
    Represents a block of a growing network.

    Sequence of layers:
    - Activation pre
    - Layer first
    - Activation mid
    - Layer second
    """

    def __init__(
        self,
        in_out_features: int,
        hidden_features: int = 0,
        layer_type: str = "linear",
        activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        in_out_features: int
            number of input and output features, in cas of convolutional layer, the number of channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer, if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers, if None use the activation function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        kwargs_first_layer: dict | None
            dictionary of arguments for the first layer, if None use kwargs_layer
        kwargs_second_layer: dict | None
            dictionary of arguments for the second layer, if None use kwargs_layer
        """
        assert layer_type in all_layer_types, f"Layer type {layer_type} not supported."
        if kwargs_layer is None:
            kwargs_layer = {}

        super(GrowingResidualBlock, self).__init__()
        self.name = name
        self.hidden_features = hidden_features

        self.activation: torch.nn.Module = activation
        self.first_layer = all_layer_types[layer_type]["layer"](
            in_out_features,
            hidden_features,
            post_layer_function=activation,
            name=f"{name}_first_layer",
            **kwargs_layer,
        )
        self.second_layer = all_layer_types[layer_type]["layer"](
            hidden_features,
            in_out_features,
            post_layer_function=torch.nn.Identity(),
            previous_module=self.first_layer,
            name=f"{name}_second_layer",
            **kwargs_layer,
        )

        self.enable_extended_forward = False
        self.eigenvalues = None
        self.parameter_update_decrease = None

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

    @staticmethod
    def set_default_values(
        activation: torch.nn.Module | None = None,
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
    ) -> tuple[torch.nn.Module | None, torch.nn.Module | None, dict | None, dict | None]:
        """
        Set default values for the block.
        """
        if pre_activation is None:
            pre_activation = activation
        if mid_activation is None:
            mid_activation = activation
        if kwargs_first_layer is None:
            kwargs_first_layer = kwargs_layer
        if kwargs_second_layer is None:
            kwargs_second_layer = kwargs_layer
        return pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        x = self.activation(x)
        if self.hidden_features > 0:
            y, y_ext = self.first_layer.extended_forward(x)
            y, _ = self.second_layer.extended_forward(y, y_ext)
            assert (
                _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."
            del y_ext
            x = y + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        x = self.activation(x)
        if self.hidden_features > 0:
            y = self.first_layer(x)
            y = self.second_layer(y)
            x = y + x
        return x

    @property
    def in_activity(self) -> torch.Tensor:
        """
        Get the input activity of the block.

        Returns
        -------
        torch.Tensor
            input activity
        """
        return self.first_layer.input

    def set_store_in_activity(self, value: bool):
        """
        Set the store_in_activity parameter of the block.
        If True, the block will store the activity after the first activation
        function.

        Parameters
        ----------
        value: bool
            value to set
        """
        self.first_layer.store_input = True

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        self.first_layer.store_input = True
        self.first_layer.store_pre_activity = True
        self.second_layer.store_input = True
        self.second_layer.store_pre_activity = True
        self.second_layer.tensor_s.init()
        self.second_layer.tensor_m.init()
        self.second_layer.tensor_m_prev.init()
        self.second_layer.cross_covariance.init()

    def update_computation(self, desired_activation: torch.Tensor | None = None):
        """
        Update the computation of the block.

        Parameters
        ----------
        desired_activation: torch.Tensor
            desired direction of the output variation of the block
        """
        # self.second_layer.tensor_m.update(desired_activation=desired_activation)
        # self.second_layer.tensor_s.update()
        # self.second_layer.tensor_m_prev.update(desired_activation=desired_activation)
        # self.second_layer.cross_covariance.update()
        # self.first_layer.update_computation()
        self.second_layer.update_computation()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        self.first_layer.store_input = False
        self.second_layer.store_input = False
        self.second_layer.tensor_s.reset()
        self.second_layer.tensor_m.reset()
        self.second_layer.tensor_m_prev.reset()
        self.second_layer.cross_covariance.reset()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        self.second_layer.optimal_delta_layer = None
        self.second_layer.extended_input_layer = None
        self.first_layer.extended_input_layer = None

    def compute_optimal_updates(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        """
        _, _, self.parameter_update_decrease = self.second_layer.compute_optimal_delta()
        alpha, alpha_bias, _, self.eigenvalues = (
            self.second_layer.compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
            )
        )
        self.first_layer.extended_output_layer = self.first_layer.layer_of_tensor(
            alpha, alpha_bias
        )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.first_layer.apply_change()
        self.second_layer.apply_change()
        self.hidden_features += self.eigenvalues.shape[0]

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        self.eigenvalues = self.eigenvalues[:keep_neurons]
        self.first_layer.sub_select_optimal_added_parameters(keep_neurons)
        self.second_layer.sub_select_optimal_added_parameters(keep_neurons)

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        return (
            self.parameter_update_decrease
            + self.activation_derivative * (self.eigenvalues**2).sum()
        )


if __name__ == "__main__":
    # Test the GrowingResidualBlock
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Define the dataset
    batch_size = 2
    num_batch = 4
    in_features = 10
    dataset = [(torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)) for _ in range(num_batch)]

    # Define the block
    hidden_features = 8
    block = GrowingResidualBlock(in_features, hidden_features, activation=nn.ReLU(), name="block")

    # Define the optimizer
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    # Regular training
    for x, y in dataset:
        optimizer.zero_grad()
        y_hat = block(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss}')

    # Gathering growing statistics
    block.init_computation()
    for x, y in dataset:
        y_hat = block(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        loss.backward()
        block.update_computation()

    # Compute the optimal update
    block.compute_optimal_updates()

    # Apply the change
    block.apply_change()

    # Sub select the optimal added parameters
    block.sub_select_optimal_added_parameters(3)

    # Get the first order improvement
    print(block.first_order_improvement)
