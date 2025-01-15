import warnings
from typing import Iterator

import numpy as np
import torch

from gromo.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device


class AdditionGrowingModule(torch.nn.Module):
    """
    Module to connect multiple modules with an addition operation.
    This module does not perform the addition operation, it is done by the user.
    """

    def __init__(
        self,
        post_addition_function: torch.nn.Module = torch.nn.Identity(),
        previous_modules: list["AdditionGrowingModule | GrowingModule"] = None,
        next_modules: list["AdditionGrowingModule | GrowingModule"] = None,
        allow_growing: bool = False,
        tensor_s_shape: tuple[int, int] = None,
        device: torch.device | None = None,
        name: str = None,
    ) -> None:

        super(AdditionGrowingModule, self).__init__()
        self._name = name
        self.name = (
            self.__class__.__name__
            if name is None
            else f"{self.__class__.__name__}({name})"
        )

        self.device = (
            device if device else global_device()
        )  # FIXME: this could be removed

        self.post_addition_function: torch.nn.Module = post_addition_function
        if self.post_addition_function:
            self.post_addition_function = self.post_addition_function.to(self.device)
        self._allow_growing = allow_growing

        self.store_input = 0
        self.input = None

        self.store_activity = 0
        self.activity = None

        self.tensor_s = TensorStatistic(
            tensor_s_shape,
            update_function=self.compute_s_update,
            device=self.device,
            name=f"S({name})",
        )

        self.previous_tensor_s: TensorStatistic | None = None
        self.previous_tensor_m: TensorStatistic | None = None

        self.previous_modules: list[AdditionGrowingModule | GrowingModule] = []
        self.set_previous_modules(previous_modules)
        self.next_modules: list[AdditionGrowingModule | GrowingModule] = []
        self.set_next_modules(next_modules)

    @property
    def number_of_successors(self):
        return len(self.next_modules)

    @property
    def number_of_predecessors(self):
        return len(self.previous_modules)

    def grow(self):
        """
        Function to call after growing previous or next modules.
        """
        # mainly used to reset the shape of the tensor S, M, prev S and prev M
        self.set_next_modules(self.next_modules)
        self.set_previous_modules(self.previous_modules)

    def add_next_module(self, module: "AdditionGrowingModule | GrowingModule") -> None:
        """
        Add a module to the next modules of the current module.

        Parameters
        ----------
        module
            next module to add
        """
        self.next_modules.append(module)
        self.set_next_modules(
            self.next_modules
        )  # TODO: maybe it is possible to avoid this

    def add_previous_module(
        self, module: "AdditionGrowingModule | GrowingModule"
    ) -> None:
        """
        Add a module to the previous modules of the current module.

        Parameters
        ----------
        module
            previous module to add
        """
        self.previous_modules.append(module)
        self.set_previous_modules(self.previous_modules)

    def set_next_modules(
        self, next_modules: list["AdditionGrowingModule | GrowingModule"]
    ) -> None:
        """
        Set the next modules of the current module.

        Parameters
        ----------
        next_modules
            list of next modules
        """
        raise NotImplementedError

    def set_previous_modules(
        self, previous_modules: list["AdditionGrowingModule | GrowingModule"]
    ) -> None:
        """
        Set the previous modules of the current module.

        Parameters
        ----------
        previous_modules
            list of previous modules
        """
        raise NotImplementedError

    def __str__(self, verbose=1):
        if verbose == 0:
            return f"{self.__class__.__name__} module."
        elif verbose == 1:
            previous_modules = (
                len(self.previous_modules) if self.previous_modules else "no"
            )
            next_modules = len(self.next_modules) if self.next_modules else "no"
            return (
                f"{self.__class__.__name__} module with {previous_modules} "
                f"previous modules and {next_modules} next modules."
            )
        elif verbose >= 2:
            txt = [
                f"{self.__class__.__name__} module.",
                f"\tPrevious modules : {self.previous_modules}",
                f"\tNext modules : {self.next_modules}",
                f"\tPost addition function : {self.post_addition_function}",
                f"\tAllow growing : {self._allow_growing}",
                f"\tStore input : {self.store_input}",
                f"\tStore activity : {self.store_activity}",
                f"\tTensor S : {self.tensor_s}",
                f"\tPrevious tensor S : {self.previous_tensor_s}",
                f"\tPrevious tensor M : {self.previous_tensor_m}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for t in (self.tensor_s, self.previous_tensor_s, self.previous_tensor_m):
            if t:
                t.updated = False

        if self.store_input > 0:
            self.input = x
            self.input.retain_grad()

        if (self.post_addition_function) and (x is not None):
            y = self.post_addition_function(x)
        else:
            y = x

        if self.store_activity > 0:
            self.activity = y
            self.tensor_s.updated = False  # reset the update flag

        return y

    @property
    def pre_activity(self):
        return self.input

    def projected_v_goal(self) -> torch.Tensor:
        """
        Compute the projected gradient of the goal with respect to the activity of the layer.

        dLoss/dA_proj := dLoss/dA - dW B[-1] where A is the pre-activation vector of the
        layer, and dW the optimal delta for all the previous layers

        Returns
        -------
        torch.Tensor
            projected gradient of the goal with respect to the activity of the next layer
            dLoss/dA - dW B[-1]
        """
        V_proj = self.pre_activity.grad.clone().detach()
        for module in self.previous_modules:
            V_proj -= module.optimal_delta_layer(module.input)

        return V_proj

    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    def compute_previous_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S for the input of all previous modules.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    def compute_previous_m_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M for the input of all previous modules.

        Returns
        -------
        torch.Tensor
            update of the tensor M
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    def init_computation(self) -> None:
        """
        Initialize the computation of the optimal added parameters.
        """
        self.store_input = True
        self.store_pre_activity = True
        for module in self.previous_modules:
            module.store_input = True
            module.store_pre_activity = True
        self.previous_tensor_s.init()
        self.previous_tensor_m.init()

    def reset_computation(self) -> None:
        """
        Reset the computation of the optimal added parameters.
        """
        self.store_input = False
        self.store_pre_activity = False
        self.store_activity = False
        for module in self.previous_modules:
            module.store_input = False
            module.store_pre_activity = False
        self.previous_tensor_s.reset()
        self.previous_tensor_m.reset()

    def delete_update(self, include_previous: bool = False) -> None:
        """
        Delete the update of the optimal added parameters.
        """
        self.optimal_delta_layer = None
        self.extended_input_layer = None
        self.scaling_factor = 0.0
        self.parameter_update_decrease = None
        self.eigenvalues_extension = None
        self.activity = None
        self.input = None
        # TODO: include_previous

    def update_size(self) -> None:
        """
        Update the input and output size of the module
        """
        prev_total_in_features = self.total_in_features
        prev_in_features = self.in_features
        if len(self.previous_modules) > 0:
            new_size = self.previous_modules[0].out_features
            self.in_features = new_size
            self.out_features = new_size
        self.total_in_features = self.sum_in_features(with_bias=True)

        if self.total_in_features > 0:
            if self.previous_tensor_s._shape != (
                self.total_in_features,
                self.total_in_features,
            ):
                self.previous_tensor_s = TensorStatistic(
                    (
                        self.total_in_features,
                        self.total_in_features,
                    ),
                    device=self.device,
                    name=f"S[-1]({self.name})",
                    update_function=self.compute_previous_s_update,
                )
            if self.previous_tensor_m._shape != (
                self.total_in_features,
                self.in_features,
            ):
                self.previous_tensor_m = TensorStatistic(
                    (self.total_in_features, self.in_features),
                    device=self.device,
                    name=f"M[-1]({self.name})",
                    update_function=self.compute_previous_m_update,
                )
        else:
            self.previous_tensor_s = None
            self.previous_tensor_m = None

    @property
    def number_of_parameters(self):
        return 0

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return iter([])

    def sum_in_features(self, with_bias: bool = False) -> int:
        """Count total in_features of previous modules

        Returns
        -------
        int
            sum of previous in_features
        """
        if with_bias:
            return np.sum(
                [module.in_features + module.use_bias for module in self.previous_modules]
            )
        return np.sum([module.in_features for module in self.previous_modules])

    def sum_out_features(self) -> int:
        """Count total out_features of next modules

        Returns
        -------
        int
            sum of next out_features
        """
        return np.sum([module.out_features for module in self.next_modules])


class GrowingModule(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        tensor_s_shape: tuple[int, int],
        tensor_m_shape: tuple[int, int],
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        allow_growing: bool = True,
        previous_module: torch.nn.Module | None = None,
        next_module: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        assert len(tensor_s_shape) == 2, "The shape of the tensor S must be 2D."
        assert tensor_s_shape[0] == tensor_s_shape[1], "The tensor S must be square."
        assert tensor_s_shape[0] == tensor_m_shape[0], (
            f"The input matrices S and M must have compatible shapes."
            f"(got {tensor_s_shape=} and {tensor_m_shape=})"
        )

        super(GrowingModule, self).__init__()
        self._name = name
        self.name = (
            self.__class__.__name__
            if name is None
            else f"{self.__class__.__name__}({name})"
        )

        self.device = device if device else global_device()

        self.layer: torch.nn.Module = layer.to(self.device)
        self.post_layer_function: torch.nn.Module = post_layer_function.to(self.device)
        self._allow_growing = allow_growing
        assert not self._allow_growing or isinstance(
            previous_module, (GrowingModule, AdditionGrowingModule)
        ), (
            f"to grow previous_module must be an instance of GrowingModule"
            f"or AdditionGrowingModule, but got {type(previous_module)}"
        )

        self.next_module: torch.nn.Module | None = next_module
        self.previous_module: torch.nn.Module | None = previous_module

        self.__dict__["store_input"] = False
        self.__dict__["store_pre_activity"] = False
        # self.store_activity = False

        self._internal_store_input = False
        self._internal_store_pre_activity = False
        # self._internal_store_activity = False

        self._input: torch.Tensor | None = None
        self._pre_activity: torch.Tensor | None = None
        # self._activity = None

        self._tensor_s = TensorStatistic(
            tensor_s_shape,
            update_function=self.compute_s_update,
            device=self.device,
            name=f"S({name})",
        )
        self.tensor_m = TensorStatistic(
            tensor_m_shape,
            update_function=self.compute_m_update,
            device=self.device,
            name=f"M({name})",
        )
        # self.tensor_n = TensorStatistic(output_shape, update_function=self.compute_n_update)

        # the optimal update used to compute v_projected
        self.optimal_delta_layer: torch.nn.Module | None = None
        self.scaling_factor: torch.Tensor = torch.zeros(1, device=self.device)
        self.scaling_factor.requires_grad = True
        self._scaling_factor_next_module: torch.Tensor = torch.zeros(1, device=self.device)

        self.extended_input_layer: torch.nn.Module | None = None
        self.extended_output_layer: torch.nn.Module | None = None

        # when updating a layer with t * optimal_delta_layer having a change of activity of dA
        # we have L(A + dA) = L(A) - t * parameter_update_decrease + o(t)
        self.parameter_update_decrease: torch.Tensor | None = None

        # when increasing this layer with sqrt(t) * extended_input_layer and
        # the previous with sqrt(t) * extended_output_layer having a change of activity of dA
        # we have L(A + dA) = L(A) - t * sigma'(0) * (eigenvalues_extension ** 2).sum() + o(t)
        self.eigenvalues_extension: torch.Tensor | None = None

        self.delta_raw: torch.Tensor | None = None

        # if self._allow_growing: # FIXME: should we add this condition?
        self.tensor_m_prev = TensorStatistic(
            None,
            update_function=self.compute_m_prev_update,
            device=self.device,
            name=f"M_prev({name})",
        )
        self.cross_covariance = TensorStatistic(
            None,
            update_function=self.compute_cross_covariance_update,
            device=self.device,
            name=f"C({name})",
        )

    # Information functions
    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    @property
    def activation_gradient(self) -> torch.Tensor:
        """
        Return the derivative of the activation function before this layer at 0+.

        Returns
        -------
        torch.Tensor
            derivative of the activation function before this layer at 0+
        """
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Return the parameters of the layer.

        Parameters
        ----------
        recurse: bool
            if True, return the parameters of the submodules

        Returns
        -------
        Iterator[Parameter]
            iterator over the parameters of the layer
        """
        return self.layer.parameters(recurse=recurse)

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the layer.

        Returns
        -------
        int
            number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def __str__(self, verbose=0):
        if verbose == 0:
            return f"{self.name} module with {self.number_of_parameters()} parameters."
        elif verbose == 1:
            return (
                f"{self.name} module with {self.number_of_parameters()} parameters "
                f"({self._allow_growing=}, {self.store_input=}, "
                f"{self.store_pre_activity=})."
            )
        elif verbose >= 2:
            txt = [
                f"{self.name} module with {self.number_of_parameters()} parameters.",
                f"\tLayer : {self.layer}",
                f"\tPost layer function : {self.post_layer_function}",
                f"\tAllow growing : {self._allow_growing}",
                f"\tStore input : {self.store_input}",
                f"\t{self._internal_store_input=}",
                f"\tStore pre-activity : {self.store_pre_activity}",
                f"\t{self._internal_store_pre_activity=}",
                f"\tTensor S (internal) : {self._tensor_s}",
                f"\tTensor S : {self.tensor_s}",
                f"\tTensor M : {self.tensor_m}",
                f"\tOptimal delta layer : {self.optimal_delta_layer}",
                f"\tExtended input layer : {self.extended_input_layer}",
                f"\tExtended output layer : {self.extended_output_layer}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    def __setattr__(self, key, value):
        if key == "store_input" and value is not self.store_input:
            self.__dict__["store_input"] = value
            if isinstance(self.previous_module, AdditionGrowingModule):
                # As a AdditionGrowingModule may have multiple next modules
                # we need to keep track of the number of modules that require the activity
                # to be stored. Hence we store it as long as one of the module requires it.
                self.previous_module.store_activity += 1 if value else -1
            else:
                self._internal_store_input = value
        elif key == "store_pre_activity" and value is not self.store_pre_activity:
            self.__dict__["store_pre_activity"] = value
            if isinstance(self.next_module, AdditionGrowingModule):
                self.next_module.store_input += 1 if value else -1
            else:
                self._internal_store_pre_activity = value
        elif key == "previous_module" or key == "next_module":
            self.__dict__[key] = value
        elif key == "scaling_factor":
            if isinstance(value, torch.Tensor):
                assert value.shape == (1,), "The scaling factor must be a scalar."
                torch.nn.Module.__setattr__(self, key, value)
                if isinstance(self.previous_module, GrowingModule):
                    self.previous_module._scaling_factor_next_module = value
            else:
                assert isinstance(
                    value, (int, float)
                ), "The scaling factor must be a scalar."
                self.__dict__[key].data[0] = value
                # FIXME: should we not recreate the tensor? (problem with the gradient)
        elif key == "weight":
            self.layer.weight = value
        elif key == "bias":
            self.layer.bias = value
        else:
            # Warning: if you use __dict__ to set an attribute instead of
            # Module.__setattr__, the attribute will not be registered as a
            # parameter of the module ie .parameters() will not return it.
            torch.nn.Module.__setattr__(self, key, value)

    # Forward and storing
    def forward(self, x):
        """
        Forward pass of the module.
        If needed, store the activity and pre-activity tensors.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        self._tensor_s.updated = False
        self.tensor_m.updated = False
        self.tensor_m_prev.updated = False
        self.cross_covariance.updated = False

        if self._internal_store_input:
            self._input = x.detach()

        pre_activity: torch.Tensor = self.layer(x)

        if self._internal_store_pre_activity:
            self._pre_activity = pre_activity
            self._pre_activity.retain_grad()

        return self.post_layer_function(pre_activity)

    def extended_forward(
        self, x: torch.Tensor, x_ext: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the module with layer extension and layer update.
        WARNING: does not store the input and pre-activity tensors.
        WARNING: the scaling factor is squared for the optimal delta and
        linear for the extension. (Instead of linear for the optimal delta and
        squared for the extension as in the theory).

        Parameters
        ----------
        x: torch.Tensor
            input tensor
        x_ext: torch.Tensor | None
            extension tensor

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            output tensor and extension tensor
        """
        pre_activity = self.layer(x)

        # FIXME: should the scaling factor be squared with torch.sign?
        linear_factor = self.scaling_factor**2 * torch.sign(self.scaling_factor)
        sqrt_factor = self.scaling_factor

        if self.optimal_delta_layer is not None:
            pre_activity -= linear_factor * self.optimal_delta_layer(x)

        if self.extended_input_layer:
            if x_ext is None:
                raise ValueError(
                    f"x_ext must be provided got None for {self.name}."
                    f"As the input is extended, an extension is needed."
                )
            pre_activity += sqrt_factor * self.extended_input_layer(x_ext)
        else:
            if x_ext is not None:
                warnings.warn(
                    f"x_ext must be None got {x_ext} for {self.name}. As the input is not extended, no extension is needed.",
                    UserWarning,
                )

        if self.extended_output_layer:
            supplementary_pre_activity = self._scaling_factor_next_module * self.extended_output_layer(x)
            supplementary_activity = self.post_layer_function(supplementary_pre_activity)
        else:
            supplementary_activity = None

        activity = self.post_layer_function(pre_activity)

        return activity, supplementary_activity

    @property
    def input(self) -> torch.Tensor:
        if self.store_input:
            if self._internal_store_input:
                assert self._input is not None, (
                    "The input is not stored." "Apparently it was not computed yet."
                )
                return self._input
            else:
                assert self.previous_module, (
                    "A previous module is needed to store the input."
                    "Otherwise self._internal_store_input must be set to True."
                )
                return self.previous_module.activity
        else:
            raise ValueError("The input is not stored.")

    @property
    def input_extended(self) -> torch.Tensor:
        """
        Return the input extended ones if the bias is used.

        Returns
        -------
        torch.Tensor
            input extended
        """
        if self.use_bias:
            raise NotImplementedError
        else:
            return self.input

    @property
    def pre_activity(self) -> torch.Tensor:
        if self.store_pre_activity:
            if self._internal_store_pre_activity:
                assert self._pre_activity is not None, (
                    "The pre-activity is not stored."
                    "Apparently it was not computed yet."
                )
                return self._pre_activity
            else:
                assert self.next_module, (
                    "A next module is needed to store the input."
                    "Otherwise self._internal_store_pre_activity must be set to True."
                )
                return self.next_module.input
        else:
            raise ValueError("The pre-activity is not stored.")

    # Statistics computation
    def projected_v_goal(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute the projected gradient of the goal with respect to the activity of the layer.

        dLoss/dA_proj := dLoss/dA - dW B[-1] where A is the pre-activation vector of the
        layer, and dW the optimal delta for the layer

        Parameters
        ----------
        input_vector: torch.Tensor of shape (n_samples, in_features)
            input vector B[-1]

        Returns
        -------
        torch.Tensor
            projected gradient of the goal with respect to the activity of the next layer
            dLoss/dA - dW B[-1]
        """
        assert self.optimal_delta_layer, (
            "The optimal delta layer is not computed."
            "Therefore the projected gradient cannot be computed."
        )
        return self.pre_activity.grad - self.optimal_delta_layer(input_vector)

    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    @property
    def tensor_s(self) -> TensorStatistic:
        """
        Return the tensor S of the layer.
        Either the tensor S computed locally or the tensor S of the previous addition layer.

        Returns
        -------
        TensorStatistic
            tensor S
        """
        if isinstance(self.previous_module, AdditionGrowingModule):
            return self.previous_module.tensor_s
        else:
            return self._tensor_s

    def compute_m_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M. Should be added to the type of layer.

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
        raise NotImplementedError

    def compute_m_prev_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M_{-2} := dA B[-2]^T.

        Parameters
        ----------
        desired_activation: torch.Tensor | None
            desired variation direction of the output  of the layer

        Returns
        -------
        torch.Tensor
            update of the tensor M_{-2}
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    def compute_cross_covariance_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor C := B[-1] B[-2]^T.

        Returns
        -------
        torch.Tensor
            update of the tensor C
        int
            number of samples used to compute the update
        """
        raise NotImplementedError

    def compute_n_update(self):
        """
        Compute the update of the tensor N. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor N
        """
        raise NotImplementedError

    @property
    def tensor_n(self) -> torch.Tensor:
        """
        Compute the tensor N for the layer with the current M_-2, C and optimal delta.

        Returns
        -------
        torch.Tensor
            N
        """
        raise NotImplementedError

    # Layer edition
    def layer_of_tensor(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.nn.Linear:
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
        raise NotImplementedError

    def add_parameters(self, **kwargs) -> None:
        """
        Grow the module by adding new parameters to the layer.

        Parameters
        ----------
        kwargs: dict
            typically include the values of the new parameters to add to the layer
        """
        raise NotImplementedError

    def layer_in_extension(self, weight: torch.Tensor) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the input of the layer is extended but not the output.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the extension
        """
        raise NotImplementedError

    def layer_out_extension(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the output of the layer is extended but not the input.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the extension
        bias: torch.Tensor | None
            bias of the extension if needed
        """
        raise NotImplementedError

    def parameter_step(
        self, delta_weights: torch.Tensor, delta_biases: torch.Tensor | None = None
    ) -> None:
        """
        Update the parameters of the layer with the given deltas.

        Parameters
        ----------
        delta_weights: torch.Tensor
            delta values for the weights
        delta_biases: torch.Tensor | None
            delta values for the biases, if None, the biases are not updated
        """
        self.layer.weight.data += delta_weights
        if delta_biases is not None:
            self.layer.bias.data += delta_biases

    def _sub_select_added_output_dimension(self, keep_neurons: int) -> None:
        """
        Select the first `keep_neurons` neurons of the optimal added output dimension.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        raise NotImplementedError

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
        sub_select_previous: bool = True,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters
        linked to this layer.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        sub_select_previous: bool
            if True, sub-select the previous layer added parameters as well
        """
        raise NotImplementedError

    def apply_change(
        self,
        scaling_factor: float | torch.Tensor | None = None,
        apply_previous: bool = True,
    ) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.

        Parameters
        ----------
        scaling_factor: float | torch.Tensor | None
            scaling factor to apply to the optimal delta,
             if None use the current scaling factor
        apply_previous: bool
            if True apply the change to the previous layer
        """
        # print(f"==================== Applying change to {self.name} ====================")
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor  # type: ignore
            # this type problem is due to the use of the setter to change the scaling factor
        linear_factor = self.scaling_factor**2 * torch.sign(self.scaling_factor)
        sqrt_factor = self.scaling_factor
        if self.optimal_delta_layer is not None:
            self.parameter_step(
                delta_weights=-linear_factor * self.optimal_delta_layer.weight.data,
                delta_biases=(
                    -linear_factor * self.optimal_delta_layer.bias.data
                    if self.optimal_delta_layer.bias is not None
                    else None
                ),
            )
        if self.extended_input_layer:
            # if abs(sqrt_factor * self.extended_input_layer.weight).max() < 1e-15:
            #     print(f"Warning: the input extension of {self.name} is null.")
            #     print(f"{self.extended_input_layer.weight=}")
            #     print(f"{sqrt_factor=}")
            assert self.extended_input_layer.bias is None or torch.allclose(
                self.extended_input_layer.bias,
                torch.zeros_like(self.extended_input_layer.bias),
            ), "The bias of the input extension must be null."
            self.layer_in_extension(weight=sqrt_factor * self.extended_input_layer.weight)
        if self.extended_output_layer:
            if abs(sqrt_factor * self.extended_output_layer.weight).max() < 1e-15:
                print(f"Warning: the output extension of {self.name} is null.")
                print(f"{self.extended_output_layer.weight=}")
                print(f"{self.extended_output_layer.bias=}")
                print(f"{sqrt_factor=}")
            self.layer_out_extension(
                weight=sqrt_factor * self.extended_output_layer.weight,
                bias=(
                    sqrt_factor * self.extended_output_layer.bias
                    if self.extended_output_layer.bias is not None
                    else None
                ),
            )
        if apply_previous and self.previous_module is not None:
            if isinstance(self.previous_module, GrowingModule):
                self.previous_module.apply_change(
                    apply_previous=False, scaling_factor=self.scaling_factor
                )
            elif isinstance(self.previous_module, AdditionGrowingModule):
                raise NotImplementedError  # TODO
            else:
                raise NotImplementedError
        # print("====================================")
        # Update the size of the previous and next modules
        if isinstance(self.previous_module, AdditionGrowingModule):
            self.previous_module.update_size()
        if isinstance(self.next_module, AdditionGrowingModule):
            self.next_module.update_size()

    # Optimal update computation
    def compute_optimal_delta(
        self,
        update: bool = True,
        dtype: torch.dtype = torch.float32,
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
            if True update the optimal delta layer attribute and the first order decrease
        dtype: torch.dtype
            dtype for S and M during the computation

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]
            optimal delta for the weights, the biases if needed and the first order decrease
        """
        raise NotImplementedError

    def compute_optimal_added_parameters(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Compute the optimal added parameters to extend the input layer.
        Update the extended_input_layer and the eigenvalues_extension.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        update_previous: bool
            whether to change the previous layer extended_output_layer
        dtype: torch.dtype
            dtype for S and N during the computation

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
            optimal added weights alpha weights, alpha bias, omega and eigenvalues lambda
        """
        raise NotImplementedError

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        assert self.parameter_update_decrease is not None, (
            "The first order improvement is not computed. "
            "Use compute_optimal_delta before."
        )
        if self.eigenvalues_extension is not None:
            return (
                self.parameter_update_decrease
                + self.activation_gradient * (self.eigenvalues_extension**2).sum()
            )
        else:
            return self.parameter_update_decrease

    def compute_optimal_updates(
        self,
        numerical_threshold: float = 1e-10,
        statistical_threshold: float = 1e-5,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        zero_delta: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute the optimal update  and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        update_previous: bool
            whether to change the previous layer extended_output_layer
        zero_delta: bool
            if True, set the optimal delta to zero
        dtype: torch.dtype
            dtype for the computation of the optimal delta and added parameters

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            optimal extension for the previous layer (weights and biases)
        """
        self.compute_optimal_delta(dtype=dtype)
        assert self.parameter_update_decrease is not None, (
            "The first order improvement is not computed. "
            "Use compute_optimal_delta before."
        )
        if zero_delta:
            if self.optimal_delta_layer is not None:
                self.optimal_delta_layer.weight.data.zero_()
                if self.optimal_delta_layer.bias is not None:
                    self.optimal_delta_layer.bias.data.zero_()

        if self.previous_module is None:
            return  # FIXME: change the definition of the function
        elif isinstance(self.previous_module, GrowingModule):
            alpha_weight, alpha_bias, _, _ = self.compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=update_previous,
                dtype=dtype,
            )
            return alpha_weight, alpha_bias
        elif isinstance(self.previous_module, AdditionGrowingModule):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

    def init_computation(self) -> None:
        """
        Initialize the computation of the optimal added parameters.
        """
        self.store_input = True
        self.store_pre_activity = True
        self.tensor_s.init()
        self.tensor_m.init()
        if self.previous_module is None:
            return
        elif isinstance(self.previous_module, GrowingModule):
            self.tensor_m_prev.init()
            self.cross_covariance.init()
            self.previous_module.tensor_s.init()
            self.previous_module.store_input = True
        elif isinstance(self.previous_module, AdditionGrowingModule):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

    def reset_computation(self) -> None:
        """
        Reset the computation of the optimal added parameters.
        """
        self.store_input = False
        self.store_pre_activity = False
        self.tensor_s.reset()
        self.tensor_m.reset()
        self.tensor_m_prev.reset()
        self.cross_covariance.reset()

    def delete_update(
        self,
        include_previous: bool = True,
        include_output: bool = False,
    ) -> None:
        """
        Delete the update of the optimal added parameters.

        By default, we do not delete the extended_output_layer of this layer because it
        could be required by the next layer.

        Parameters
        ----------
        include_previous: bool
            if True delete the extended_output_layer of the previous layer
        include_output: bool
            if True delete the extended_output_layer of this layer,
            warning: this does not delete the extended_input_layer of the next layer
        """
        self.optimal_delta_layer = None
        self.scaling_factor = 0.0  # type: ignore
        # this type problem is due to the use of the setter to change the scaling factor
        self.parameter_update_decrease = None
        self.eigenvalues_extension = None
        self._pre_activity = None
        self._input = None

        # delete extended_output_layer
        if include_output:
            self.extended_output_layer = None

        # delete previous module extended_output_layer
        if self.extended_input_layer is not None:
            # delete extended_input_layer
            self.extended_input_layer = None
            if self.previous_module is not None:
                # normal behavior
                if include_previous:
                    if isinstance(self.previous_module, GrowingModule):
                        self.previous_module.extended_output_layer = None
                    elif isinstance(self.previous_module, AdditionGrowingModule):
                        raise NotImplementedError  # TODO
                    else:
                        raise TypeError(
                            f"Unexpected type for previous_module of {self.name}"
                            f"got {type(self.previous_module)} instead of GrowingModule or AdditionGrowingModule."
                        )
                # risky behavior
                else:  # include_previous is False
                    if isinstance(self.previous_module, GrowingModule):
                        if self.previous_module.extended_output_layer is not None:
                            warnings.warn(
                                f"The extended_input_layer of {self.name} has been deleted."
                                f"However, the extended_output_layer associated stored in the previous module"
                                f"named {self.previous_module.name} has not been deleted."
                                "This may lead to errors when using extended_forward.",
                                UserWarning,
                            )
                        # otherwise it is ok as user already deleted the extended_output_layer
                    elif isinstance(self.previous_module, AdditionGrowingModule):
                        raise NotImplementedError
                    else:
                        raise TypeError(
                            f"Unexpected type for previous_module of {self.name}"
                            f"got {type(self.previous_module)} instead of GrowingModule or AdditionGrowingModule."
                        )
            # incorrect behavior
            else:  # self.previous_module is None
                warnings.warn(
                    f"The extended_input_layer of {self.name} has been deleted."
                    "However, no previous module is associated with this layer."
                    "Therefore, no extended_output_layer has been deleted."
                    "This is not supposed to happen as to grow a layer a previous module is needed.",
                    UserWarning,
                )

    def update_computation(self) -> None:
        """
        Update the computation of the optimal added parameters.
        """
        self.tensor_s.update()
        self.tensor_m.update()
        if self.previous_module is None:
            return
        elif isinstance(self.previous_module, GrowingModule):
            self.previous_module.tensor_s.update()
            self.tensor_m_prev.update()
            self.cross_covariance.update()
        elif isinstance(self.previous_module, AdditionGrowingModule):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError


if __name__ == "__main__":
    help(GrowingModule)