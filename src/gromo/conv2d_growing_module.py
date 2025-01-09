from warnings import warn

import torch

from gromo.growing_module import AdditionGrowingModule, GrowingModule
from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.tensor_statistic import TensorStatistic
from gromo.tools import compute_mask_tensor_t
from gromo.utils.utils import global_device


class Conv2dAdditionGrowingModule(AdditionGrowingModule):
    pass


class Conv2dGrowingModule(GrowingModule):
    """
    Conv2dGrowingModule is a GrowingModule for a Conv2d layer.

    Parameters
    ----------
    For the parameters in_channels, out_channels, kernel_size, stride, padding, dilation,
     use_bias they are the same as in torch.nn.Conv2d.

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int]
    padding: int | tuple[int, int]
    dilation: int | tuple[int, int]
    post_layer_function: torch.nn.Module
        function applied after the layer (e.g. activation function)
    previous_module: GrowingModule | AdditionGrowingModule | None
        previous module in the network (None if the first module),
        needed to extend the layer
    next_module: GrowingModule | AdditionGrowingModule | None
        next module in the network (None if the last module)
    allow_growing: bool
        whether the layer can grow in input size
    device: torch.device | None
        device for the layer
    name: str | None
        name of the layer used for debugging purpose
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        input_size: tuple[int, int] = (-1, -1),
        # groups: int = 1,
        use_bias: bool = True,
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        previous_module: GrowingModule | AdditionGrowingModule | None = None,
        next_module: GrowingModule | AdditionGrowingModule | None = None,
        allow_growing: bool = False,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        device = device if device is not None else global_device()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        super(Conv2dGrowingModule, self).__init__(
            layer=torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
                device=device,
            ),
            post_layer_function=post_layer_function,
            previous_module=previous_module,
            next_module=next_module,
            allow_growing=allow_growing,
            tensor_s_shape=(
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
            ),
            tensor_m_shape=(
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
                out_channels,
            ),
            device=device,
            name=name,
        )
        self.kernel_size = self.layer.kernel_size

        # TODO: add S_growth
        self.input_size: tuple[int, int] = input_size
        self._mask_tensor_t: torch.Tensor | None = None
        self.use_bias = use_bias

    @property
    def mask_tensor_t(self) -> torch.Tensor:
        """
        Compute the tensor T for the layer.

        Returns
        -------
        torch.Tensor
            mask tensor T
        """
        assert self.input_size != (-1, -1), (
            f"The input size should be set before computing the mask tensor T "
            f"for {self.name}."
        )
        self.layer: torch.nn.Conv2d  # CHECK: why do we need to specify the type here?
        if self._mask_tensor_t is None:
            self._mask_tensor_t = compute_mask_tensor_t(self.input_size, self.layer)
        return self._mask_tensor_t

    # Information functions
    # TODO: implement activation_gradient ?
    # this function is used to estimate the F.O. improvement of the loss after the extension of the network
    # however this won't work if we do not have only the activation function as the post_layer_function

    @property
    def unfolded_extended_input(self) -> torch.Tensor:
        """
        Return the unfolded input extended with a channel of ones if the bias is used.

        Returns
        -------
        torch.Tensor
            unfolded input extended
        """
        # TODO: maybe we could compute it only once if we use it multiple times (e.g. in S, S_prev, M, M_prev, P...)
        unfolded_input = torch.nn.functional.unfold(
            self.input,
            self.layer.kernel_size,
            padding=self.layer.padding,
            stride=self.layer.stride,
            dilation=self.layer.dilation,
        )
        if self.use_bias:
            return torch.cat(
                (
                    unfolded_input,
                    torch.ones(
                        unfolded_input.shape[0],
                        1,
                        unfolded_input.shape[2],
                        device=self.device,
                    ),
                ),
                dim=1,
            )
        else:
            return unfolded_input

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the layer.

        Returns
        -------
        int
            number of parameters
        """
        return (
            self.layer.in_channels
            * self.layer.out_channels
            * self.layer.kernel_size[0]
            * self.layer.kernel_size[1]
            + self.layer.out_channels * self.use_bias
        )

    def __str__(self, verbose: int = 0) -> str:
        if verbose == 0:
            return (
                f"Conv2dGrowingModule({self.name if self.name else ' '})"
                f"(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"use_bias={self.use_bias})"
            )
        else:
            return super(Conv2dGrowingModule, self).__str__(verbose=verbose)

    # Statistics computation
    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S.
        With the input tensor B, the update is
        S := (B^c_F)^T B^c_F \in (C d[+1]d[+1], C d[+1]d[+1]).

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        assert (
            self.store_input
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        assert (
            self.input is not None
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        return (
            torch.einsum(
                "iam, ibm -> ab",
                self.unfolded_extended_input,
                self.unfolded_extended_input,
            ),
            self.input.shape[0],
        )

    def compute_m_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M.
        With the input tensor X and dLoss/dA the gradient of the loss
        with respect to the pre-activity:
        M = B[-1]^T dA

        Parameters
        ----------
        desired_activation: torch.Tensor | None
            desired variation direction of the output  of the layer

        Returns
        -------
        torch.Tensor
            update of the tensor M
        int
            number of samples used to compute the update
        """
        if desired_activation is None:
            desired_activation = self.pre_activity.grad
        desired_activation = desired_activation.flatten(start_dim=-2)

        return (
            torch.einsum(
                "iam, icm -> ac", self.unfolded_extended_input, desired_activation
            ),
            self.input.shape[0],
        )

    # Layer edition
    def layer_of_tensor(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.nn.Conv2d:
        """
        Create a layer with the same characteristics (excepted the shape)
         with weight as parameter and bias as bias.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the layer
        bias: torch.Tensor | None
            bias of the layer

        Returns
        -------
        torch.nn.Linear
            layer with the same characteristics
        """
        assert self.use_bias is (bias is not None), (
            f"The new layer should have a bias ({bias is not None=}) if and only if "
            f"the main layer bias ({self.use_bias =}) is not None."
        )
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"
        self.layer.kernel_size: tuple[int, int]
        self.layer.stride: tuple[int, int]
        self.layer.dilation: tuple[int, int]

        new_layer = torch.nn.Conv2d(
            weight.shape[1],
            weight.shape[0],
            bias=self.use_bias,
            device=self.device,
            kernel_size=self.layer.kernel_size,
            stride=self.layer.stride,
            padding=self.layer.padding,
            dilation=self.layer.dilation,
        )
        new_layer.weight = torch.nn.Parameter(weight)
        if self.use_bias:
            new_layer.bias = torch.nn.Parameter(bias)
        return new_layer

    # FIXME: should we implement .add_parameters

    def layer_in_extension(self, weight: torch.Tensor) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the input of the layer is extended but not the output.

        Parameters
        ----------
        weight: torch.Tensor (out_channels, K, kernel_size[0], kernel_size[1])
            weight of the extension
        """
        assert (
            weight.shape[0] == self.out_channels
        ), f"{weight.shape[0]=} should be equal to {self.out_features=}"
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"

        # TODO: check this is working
        self.layer = self.layer_of_tensor(
            weight=torch.cat((self.weight, weight), dim=1), bias=self.bias
        )

        self.in_channels += weight.shape[1]
        self._tensor_s = TensorStatistic(
            (
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
            ),
            update_function=self.compute_s_update,
            name=self.tensor_s.name,
        )
        self.tensor_m = TensorStatistic(
            (
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
                self.out_channels,
            ),
            update_function=self.compute_m_update,
            name=self.tensor_m.name,
        )

    def layer_out_extension(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the output of the layer is extended but not the input.

        Parameters
        ----------
        weight: torch.Tensor (K, in_features)
            weight of the extension
        bias: torch.Tensor (K) | None
            bias of the extension if needed
        """
        assert (
            weight.shape[1] == self.in_channels
        ), f"{weight.shape[1]=} should be equal to {self.in_channels=}"
        assert (
            bias is None or bias.shape[0] == weight.shape[0]
        ), f"{bias.shape[0]=} should be equal to {weight.shape[0]=}"

        if self.use_bias:
            assert (
                bias is not None
            ), f"The bias of the extension should be provided because the layer has a bias"
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0),
                bias=torch.cat((self.layer.bias, bias), dim=0),
            )
        else:
            if bias is not None:
                warn(
                    f"The bias of the extension should not be provided because the layer has no bias.",
                    UserWarning,
                )
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0), bias=None
            )

        self.out_channels += weight.shape[0]
        self.tensor_m = TensorStatistic(
            (self.in_channels + self.use_bias, self.out_channels),
            update_function=self.compute_m_update,
            name=self.tensor_m.name,
        )

    # Optimal update computation
    def compute_optimal_delta(
        self,
        update: bool = True,
        dtype: torch.dtype = torch.float32,
        force_pseudo_inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]:
        """
        Compute the optimal delta for the layer using current S and M tensors.

        dW* = M S[-1]^-1 (if needed we use the pseudo-inverse)

        Compute dW* (and dBias* if needed) and update the optimal_delta_layer attribute.
        L(A + gamma * B * dW) = L(A) - gamma * d + o(gamma)
        where d is the first order decrease and gamma the scaling factor.

        Parameters
        ----------
        update: bool
            if True update the optimal delta layer attribute
        dtype: torch.dtype
            dtype for S and M during the computation
        force_pseudo_inverse: bool
            whether to use the pseudo-inverse in any case

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]
            optimal delta for the weights, the biases if needed and the first order decrease
        """
        tensor_s = self.tensor_s()
        tensor_m = self.tensor_m()

        if tensor_s.dtype != dtype:
            tensor_s = tensor_s.to(dtype=dtype)
        if tensor_m.dtype != dtype:
            tensor_m = tensor_m.to(dtype=dtype)

        if not force_pseudo_inverse:
            try:
                self.delta_raw = torch.linalg.solve(tensor_s, tensor_m).t()
            except torch.linalg.LinAlgError:
                force_pseudo_inverse = True
                # self.delta_raw = torch.linalg.lstsq(tensor_s, tensor_m).solution.t()
                # do not use lstsq because it does not work with the GPU
                warn(
                    f"Using the pseudo-inverse for the computation of the optimal delta "
                    f"for {self.name}."
                )
        if force_pseudo_inverse:
            self.delta_raw = (torch.linalg.pinv(tensor_s) @ tensor_m).t()

        assert self.delta_raw is not None, "self.delta_raw should be computed by now."
        assert (
            self.delta_raw.isnan().sum() == 0
        ), f"The optimal delta should not contain NaN values for {self.name}."
        self.parameter_update_decrease = torch.trace(tensor_m @ self.delta_raw)
        if self.parameter_update_decrease < 0:
            warn(
                f"The parameter update decrease should be positive, "
                f"but got {self.parameter_update_decrease=} for layer {self.name}."
            )
            if not force_pseudo_inverse:
                warn(
                    f"Trying to use the pseudo-inverse for {self.name} with torch.float64."
                )
                return self.compute_optimal_delta(
                    update=update, dtype=torch.float64, force_pseudo_inverse=True
                )
            else:
                warn(
                    f"Failed to compute the optimal delta for {self.name}, set"
                    f"delta to zero."
                )
                self.delta_raw = torch.zeros_like(self.delta_raw)
        self.delta_raw = self.delta_raw.to(dtype=torch.float32)

        assert self.delta_raw.shape[0] == self.out_channels, (
            f"delta_raw should have shape ({self.out_features=},...)"
            f"but got {self.delta_raw.shape=}"
        )
        if self.use_bias:
            assert self.delta_raw.shape[1] == (
                self.in_channels * self.kernel_size[0] * self.kernel_size[1] + 1
            ), (
                f"delta_raw should have shape (..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1] + 1=}) "
                f"but got (..., {self.delta_raw.shape[1]})"
            )
            delta_weight = self.delta_raw[:, :-1]
            delta_bias = self.delta_raw[:, -1]
        else:
            assert self.delta_raw.shape[1] == (
                self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            ), (
                f"delta_raw should have shape (..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1]=})"
                f"but got {self.delta_raw.shape=}"
            )
            delta_weight = self.delta_raw
            delta_bias = None

        delta_weight = delta_weight.reshape(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )

        if update:
            self.optimal_delta_layer = self.layer_of_tensor(delta_weight, delta_bias)
        return delta_weight, delta_bias, self.parameter_update_decrease

    # TODO: implement compute_optimal_added_parameters
