"""Regression tests for the DAG growth bugs reproduced in the April 2026 audit.

Each test fails on ``repro/darts-baseline`` and passes once the corresponding
fix is applied. They check loss, least squares, or LayerNorm survival, not
tensor shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from gromo.containers.growing_dag import Expansion, ExpansionType, GrowingDAG
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.modules.growing_normalisation import GrowingLayerNorm
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import set_device


def _batch_mean_sse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error averaged over the batch and summed over outputs.

    Mean cross-entropy divides by the batch size only. Elementwise
    ``mse_loss(reduction="mean")`` also divides by the output width, which
    the DAG rescale does not undo.
    """
    return ((pred - target) ** 2).sum() / pred.shape[0]


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


def test_merge_delta_matches_least_squares_and_step_decreases_loss() -> None:
    """Mean-reduced DAG statistics must match a sum-reduced least-squares step."""
    torch.manual_seed(0)
    set_device("cpu")
    in_f, out_f, batch, n_batches = 5, 3, 16, 2
    xs = [torch.randn(batch, in_f) for _ in range(n_batches)]
    ys = [torch.randn(batch, out_f) for _ in range(n_batches)]
    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    w_ls = torch.linalg.lstsq(x_all, y_all).solution.T

    edge = LinearGrowingModule(
        in_features=in_f,
        out_features=out_f,
        use_bias=False,
        name="edge",
        device="cpu",
    )
    merge = LinearMergeGrowingModule(
        in_features=out_f,
        post_merge_function=nn.Identity(),
        name="merge",
        device="cpu",
    )
    merge.set_previous_modules([edge])
    for x, y in zip(xs, ys, strict=True):
        pre = torch.zeros(x.shape[0], out_f, requires_grad=True)
        _batch_mean_sse(pre, y).backward()
        edge.store_input = True
        edge._internal_store_input = True
        edge._input = x
        merge.input = pre
        merge.previous_tensor_s.updated = False
        merge.previous_tensor_m.updated = False
        merge.previous_tensor_s.update()
        merge.previous_tensor_m.update()
    merge.compute_optimal_delta()
    delta = edge.optimal_delta_layer.weight.detach().clone()
    # d(sum of squares)/d(pred) = 2 (pred - y). At a zero prediction that is
    # -2 y, so the subtracted Newton step is 2 W_ls.
    assert torch.allclose(delta, -2 * w_ls, rtol=1e-4, atol=1e-4)

    gamma = 0.25
    step = -(gamma**2) * delta
    loss_zero = float(((y_all) ** 2).mean())
    loss_after = float(((functional.linear(x_all, step) - y_all) ** 2).mean())
    assert loss_after < loss_zero

    # Sequential modules already see a sum-reduced gradient. Do not scale it.
    seq = LinearGrowingModule(
        in_features=in_f,
        out_features=out_f,
        use_bias=False,
        name="seq",
        device="cpu",
    )
    pre = torch.zeros(x_all.shape[0], out_f, requires_grad=True)
    ((pre - y_all) ** 2).sum().backward()
    seq.store_input = True
    seq._internal_store_input = True
    seq._input = x_all
    seq.store_pre_activity = True
    seq._internal_store_pre_activity = True
    seq._pre_activity = pre
    update, n_samples = seq.compute_m_update()
    assert n_samples == x_all.shape[0]
    assert torch.allclose(update, x_all.T @ (-2 * y_all), rtol=1e-4, atol=1e-4)


def test_expand_node_keeps_layernorm() -> None:
    """Scoring a node expansion must not replace the live LayerNorm."""
    torch.manual_seed(0)
    set_device("cpu")
    hidden = "1@t"
    dag_parameters = {
        "edges": [("start@t", hidden), (hidden, "end@t")],
        "node_attributes": {
            "start@t": {"type": "linear", "size": 4, "use_layer_norm": False},
            hidden: {
                "type": "linear",
                "size": 4,
                "use_layer_norm": True,
                "activation": "selu",
            },
            "end@t": {"type": "linear", "size": 3, "use_layer_norm": False},
        },
        "edge_attributes": {"type": "linear", "use_bias": True},
    }
    net = GrowingGraphNetwork(
        in_features=4,
        out_features=3,
        loss_fn=nn.MSELoss(),
        neurons=2,
        neuron_epochs=1,
        neuron_lrate=1e-3,
        neuron_batch_size=8,
        use_bias=True,
        use_layer_norm=True,
        layer_type="linear",
        name="t",
        device="cpu",
    )
    net.dag = GrowingDAG(
        in_features=4,
        out_features=3,
        neurons=2,
        use_bias=True,
        use_layer_norm=True,
        name="t",
        device="cpu",
        DAG_parameters=dag_parameters,
    )
    node = net.dag.get_node_module(hidden)
    assert isinstance(node.post_merge_function[0], GrowingLayerNorm)
    expansion = Expansion(
        net.dag,
        ExpansionType.EXPANDED_NODE,
        expanding_node=hidden,
    )
    start = net.dag.get_node_module(net.dag.root)
    end = net.dag.get_node_module(net.dag.end)
    net.expand_node(
        expansion,
        bottlenecks={end._name: torch.randn(8, 3), node._name: torch.randn(8, 4)},
        activities={start._name: torch.randn(8, 4), node._name: torch.randn(8, 4)},
        verbose=False,
    )
    assert isinstance(node.post_merge_function[0], GrowingLayerNorm)
