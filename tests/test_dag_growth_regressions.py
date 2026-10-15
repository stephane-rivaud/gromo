"""Regression tests for the DAG growth bugs on gromo main c7df6e0.

Each test fails on that main commit and passes once its fix is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from gromo.utils.utils import set_device


def test_new_edge_small_step_decreases_loss() -> None:
    """A small step on a new edge must be installed along the fitted direction."""
    from gromo.containers.growing_dag import Expansion, ExpansionType
    from gromo.containers.growing_graph_network import GrowingGraphNetwork

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

    def mse_of(weight: torch.Tensor, bias: torch.Tensor) -> float:
        return float(functional.mse_loss(functional.linear(x, weight, bias), y))

    loss_zero = float(functional.mse_loss(torch.zeros(n, out_f), y))
    loss_installed = mse_of(installed_w, installed_b)
    assert loss_installed < loss_zero


def _batch_mean_sse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error averaged over the batch and summed over outputs.

    Mean cross-entropy divides by the batch size only. Elementwise
    ``mse_loss(reduction="mean")`` also divides by the output width, which
    the DAG rescale does not undo.
    """
    return ((pred - target) ** 2).sum() / pred.shape[0]


def test_merge_delta_matches_least_squares_and_step_decreases_loss() -> None:
    """Mean-reduced DAG statistics must match a sum-reduced least-squares step."""
    from gromo.modules.linear_growing_module import (
        LinearGrowingModule,
        LinearMergeGrowingModule,
    )

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
    from gromo.containers.growing_dag import Expansion, ExpansionType, GrowingDAG
    from gromo.containers.growing_graph_network import GrowingGraphNetwork
    from gromo.modules.growing_normalisation import GrowingLayerNorm

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
    assert expansion.metrics["active_neurons"] >= 1
    y, _ = net.dag.extended_forward(torch.randn(8, 4), mask=expansion.create_mask())
    assert tuple(y.shape) == (8, 3)


def test_shared_norm_grows_once_for_a_node_with_several_incoming_edges() -> None:
    """Widening a node must grow its shared norm once, not once per incoming edge."""
    from gromo.containers.growing_dag import Expansion, ExpansionType
    from gromo.containers.growing_graph_network import GrowingGraphNetwork
    from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingLayerNorm

    torch.manual_seed(0)
    set_device("cpu")
    out_features = 3
    added = 4
    net = GrowingGraphNetwork(
        in_features=4,
        out_features=out_features,
        loss_fn=nn.MSELoss(),
        neurons=2,
        neuron_epochs=1,
        neuron_lrate=1e-2,
        neuron_batch_size=8,
        use_bias=True,
        use_layer_norm=True,
        layer_type="linear",
        device="cpu",
    )
    dag = net.dag
    dag.add_node_with_two_edges(
        dag.root,
        "hidden",
        dag.end,
        node_attributes={"type": "linear", "size": out_features, "activation": "selu"},
        zero_weights=True,
    )
    dag.toggle_node_candidate("hidden", candidate=False)
    incoming = [
        src for src in dag.predecessors(dag.end) if not dag.is_node_candidate(src)
    ]
    assert len(incoming) >= 2

    end = dag.get_node_module(dag.end)
    layer_norm = GrowingLayerNorm(out_features, elementwise_affine=True)
    batch_norm = GrowingBatchNorm1d(out_features, affine=True)
    with torch.no_grad():
        layer_norm.weight.copy_(
            torch.arange(out_features, dtype=layer_norm.weight.dtype) + 1
        )
        layer_norm.bias.copy_(
            -(torch.arange(out_features, dtype=layer_norm.bias.dtype) + 1)
        )
        batch_norm.weight.fill_(2)
        batch_norm.bias.fill_(-3)
        batch_norm.running_mean.fill_(0.5)
        batch_norm.running_var.fill_(1.5)
    previous_weight = layer_norm.weight.detach().clone()
    previous_bias = layer_norm.bias.detach().clone()
    previous_bn_weight = batch_norm.weight.detach().clone()
    previous_bn_bias = batch_norm.bias.detach().clone()
    previous_mean = batch_norm.running_mean.detach().clone()
    previous_var = batch_norm.running_var.detach().clone()
    pool = nn.Identity()
    end.post_merge_function = nn.Sequential(layer_norm, batch_norm, nn.SELU(), pool)

    for src in incoming:
        edge = dag.get_edge_module(src, dag.end)
        edge.create_layer_out_extension(added)
        with torch.no_grad():
            edge.extended_output_layer.weight.zero_()
            if edge.extended_output_layer.bias is not None:
                edge.extended_output_layer.bias.zero_()

    expansion = Expansion(dag, ExpansionType.EXPANDED_NODE, expanding_node=dag.end)
    expansion.metrics["scaling_factor"] = 1.0
    expansion.metrics["active_neurons"] = added
    net.chosen_action = expansion
    net.apply_change()

    assert tuple(layer_norm.normalized_shape) == (out_features + added,)
    assert batch_norm.num_features == out_features + added
    assert torch.equal(layer_norm.weight[:out_features], previous_weight)
    assert torch.equal(layer_norm.bias[:out_features], previous_bias)
    assert torch.equal(batch_norm.weight[:out_features], previous_bn_weight)
    assert torch.equal(batch_norm.bias[:out_features], previous_bn_bias)
    assert torch.equal(batch_norm.running_mean[:out_features], previous_mean)
    assert torch.equal(batch_norm.running_var[:out_features], previous_var)
    assert torch.equal(batch_norm.running_mean[out_features:], torch.zeros(added))
    assert torch.equal(batch_norm.running_var[out_features:], torch.ones(added))
    assert pool is end.post_merge_function[3]

    activation = net(torch.randn(8, 4))
    assert activation.shape[-1] == out_features + added


def test_new_node_between_nodes_without_a_direct_edge_can_be_selected() -> None:
    """Choosing a one-hop node must not toggle a direct edge that was never added."""
    from gromo.containers.growing_dag import Expansion, ExpansionType
    from gromo.containers.growing_graph_network import GrowingGraphNetwork

    torch.manual_seed(0)
    set_device("cpu")
    net = GrowingGraphNetwork(
        in_features=4,
        out_features=3,
        loss_fn=nn.MSELoss(),
        neurons=2,
        neuron_epochs=1,
        neuron_lrate=1e-2,
        neuron_batch_size=8,
        use_bias=True,
        use_layer_norm=True,
        layer_type="linear",
        device="cpu",
    )
    dag = net.dag
    node_attributes = {"type": "linear", "size": 3, "activation": "selu"}
    dag.add_node_with_two_edges(
        dag.root, "left", dag.end, node_attributes=node_attributes, zero_weights=True
    )
    dag.toggle_node_candidate("left", candidate=False)
    dag.add_node_with_two_edges(
        dag.root, "right", dag.end, node_attributes=node_attributes, zero_weights=True
    )
    dag.toggle_node_candidate("right", candidate=False)
    assert ("left", "right") not in dag.edges

    bridge = Expansion(
        dag,
        ExpansionType.NEW_NODE,
        expanding_node="bridge",
        previous_node="left",
        next_node="right",
        node_attributes=node_attributes,
    )
    bridge.expand()
    bridge.metrics["loss_val"] = 0.5
    net.choose_growth_best_action([bridge], use_bic=False)

    assert "bridge" in dag.nodes
    assert not dag.is_node_candidate("bridge")
    assert ("left", "bridge") in dag.edges
    assert ("bridge", "right") in dag.edges
    assert ("left", "right") not in dag.edges
    activation = net(torch.randn(8, 4))
    assert activation.shape == (8, 3)
