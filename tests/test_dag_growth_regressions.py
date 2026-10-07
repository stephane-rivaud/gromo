"""Regression tests for the DAG growth bugs reproduced in the April 2026 audit.

Each test fails on ``repro/darts-baseline`` and passes once the corresponding
fix is applied. They check loss, least squares, or LayerNorm survival, not
tensor shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from gromo.containers.growing_dag import Expansion, ExpansionType
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.utils.utils import set_device


def test_new_edge_small_step_decreases_loss() -> None:
    """A small step on a new edge must be installed along the fitted direction."""
    torch.manual_seed(0)
    set_device("cpu")
    in_f, out_f, n = 4, 3, 64
    net = GrowingGraphNetwork(
        in_features=in_f,
        out_features=out_f,
        loss_fn=nn.MSELoss(),
        neurons=2,
        neuron_epochs=40,
        neuron_lrate=5e-2,
        neuron_batch_size=32,
        use_bias=True,
        use_layer_norm=False,
        layer_type="linear",
        device="cpu",
    )
    edge = net.dag.get_edge_module(net.dag.root, net.dag.end)
    with torch.no_grad():
        edge.weight.zero_()
        edge.bias.zero_()

    x = torch.randn(n, in_f)
    y = torch.randn(n, out_f)
    pre = torch.zeros(n, out_f, requires_grad=True)
    bottleneck = torch.autograd.grad(functional.mse_loss(pre, y), pre)[0].detach().clone()
    end = net.dag.get_node_module(net.dag.end)
    start = net.dag.get_node_module(net.dag.root)
    expansion = Expansion(
        net.dag,
        ExpansionType.NEW_EDGE,
        previous_node=net.dag.root,
        next_node=net.dag.end,
    )
    net.update_edge_weights(
        expansion,
        bottlenecks={end._name: bottleneck},
        activities={start._name: x},
        verbose=False,
    )

    gamma = 0.25
    with torch.no_grad():
        edge.weight.zero_()
        edge.bias.zero_()
    edge.apply_change(scaling_factor=gamma, apply_previous=False, apply_extension=False)
    installed_w = edge.weight.detach().clone()
    installed_b = edge.bias.detach().clone()
    stored_w = edge.optimal_delta_layer.weight.detach()
    stored_b = edge.optimal_delta_layer.bias.detach()
    scale = gamma**2
    assert torch.allclose(installed_w, -scale * stored_w, atol=1e-6)
    assert torch.allclose(installed_b, -scale * stored_b, atol=1e-6)

    def mse_of(weight: torch.Tensor, bias: torch.Tensor) -> float:
        return float(functional.mse_loss(functional.linear(x, weight, bias), y))

    loss_zero = float(functional.mse_loss(torch.zeros(n, out_f), y))
    loss_installed = mse_of(installed_w, installed_b)
    loss_opposite = mse_of(-installed_w, -installed_b)
    assert loss_installed < loss_zero
    assert loss_opposite > loss_installed
