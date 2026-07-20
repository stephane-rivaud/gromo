"""Tests for joint Net2Wider function-preserving growth."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from gromo.modules.conv2d_growing_module import FullConv2dGrowingModule
from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingBatchNorm2d
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


FP_ATOL = 1e-4


def _oracle_widen_linear(
    wa: torch.Tensor,
    ba: torch.Tensor | None,
    wb: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Reference Net2Wider on a Linear→Linear pair (shared replica map)."""
    num_base = wa.shape[0]
    n_added = selected_indices.numel()
    replica_counts = torch.ones(num_base, dtype=torch.float32, device=wa.device)
    replica_counts.scatter_add_(
        0,
        selected_indices.to(device=wa.device),
        torch.ones(n_added, dtype=torch.float32, device=wa.device),
    )

    wa_new = torch.cat([wa, wa[selected_indices]], dim=0)
    ba_new = None
    if ba is not None:
        ba_new = torch.cat([ba, ba[selected_indices]], dim=0)

    wb_new = torch.cat([wb.clone(), wb[:, selected_indices].clone()], dim=1)
    for j in range(num_base):
        count = float(replica_counts[j].item())
        indices = [j] + [
            num_base + i for i, src in enumerate(selected_indices.tolist()) if src == j
        ]
        for idx in indices:
            wb_new[:, idx] = wb_new[:, idx] / count
    return wa_new, ba_new, wb_new


class TestNet2WiderJointLinear(TorchTestCase):
    """Linear→Linear joint Net2Wider mechanism tests."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = global_device()

    def _make_pair(
        self,
        in_features: int = 5,
        hidden: int = 4,
        out_features: int = 3,
        bias: bool = True,
        activation: torch.nn.Module | None = None,
    ) -> tuple[LinearGrowingModule, LinearGrowingModule]:
        if activation is None:
            activation = torch.nn.Identity()
        first = LinearGrowingModule(
            in_features,
            hidden,
            use_bias=bias,
            name="first",
            device=self.device,
            post_layer_function=activation,
        )
        second = LinearGrowingModule(
            hidden,
            out_features,
            use_bias=bias,
            name="second",
            previous_module=first,
            device=self.device,
        )
        with torch.no_grad():
            first.weight.normal_(0.0, 0.5)
            second.weight.normal_(0.0, 0.5)
            if first.bias is not None:
                first.bias.uniform_(-0.3, 0.3)
            if second.bias is not None:
                second.bias.uniform_(-0.3, 0.3)
        return first, second

    def _forward_pair(
        self,
        first: LinearGrowingModule,
        second: LinearGrowingModule,
        x: torch.Tensor,
        *,
        use_extensions: bool,
    ) -> torch.Tensor:
        if use_extensions:
            first.scaling_factor = 1.0  # type: ignore[assignment]
            second.scaling_factor = 1.0  # type: ignore[assignment]
            h, h_ext = first.extended_forward(x)
            y, y_ext = second.extended_forward(h, h_ext)
            assert y_ext is None
            return y
        return second(first(x))

    def test_function_preservation_pre_and_post_apply_change(self) -> None:
        first, second = self._make_pair()
        x = torch.randn(16, first.in_features, device=self.device)
        selected = torch.tensor([0, 2, 2, 1], dtype=torch.long)

        with torch.no_grad():
            y_before = self._forward_pair(first, second, x, use_extensions=False).clone()

        second.create_layer_extensions(
            extension_size=4,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=selected,
        )

        with torch.no_grad():
            y_pre = self._forward_pair(first, second, x, use_extensions=True)
        self.assertAllClose(y_before, y_pre, atol=FP_ATOL)

        second.apply_change(scaling_factor=1.0, extension_size=4)
        second.delete_update(include_previous=True)

        with torch.no_grad():
            y_post = self._forward_pair(first, second, x, use_extensions=False)
        self.assertAllClose(y_before, y_post, atol=FP_ATOL)
        self.assertEqual(first.out_features, 8)
        self.assertEqual(second.in_features, 8)
        self.assertTrue(torch.equal(second.net2wider_selected_indices.cpu(), selected))

    def test_two_step_cumulative_growth(self) -> None:
        first, second = self._make_pair(hidden=3)
        x = torch.randn(8, first.in_features, device=self.device)
        with torch.no_grad():
            y0 = self._forward_pair(first, second, x, use_extensions=False).clone()

        second.create_layer_extensions(
            extension_size=2,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=torch.tensor([0, 1]),
        )
        second.apply_change(scaling_factor=1.0, extension_size=2)
        second.delete_update(include_previous=True)

        with torch.no_grad():
            y1 = self._forward_pair(first, second, x, use_extensions=False)
        self.assertAllClose(y0, y1, atol=FP_ATOL)

        second.create_layer_extensions(
            extension_size=3,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=torch.tensor([0, 0, 4]),
        )
        second.apply_change(scaling_factor=1.0, extension_size=3)
        second.delete_update(include_previous=True)

        with torch.no_grad():
            y2 = self._forward_pair(first, second, x, use_extensions=False)
        self.assertAllClose(y0, y2, atol=FP_ATOL)
        self.assertEqual(first.out_features, 8)
        self.assertEqual(second.in_features, 8)

    def test_rejects_output_only_with_kaiming_input(self) -> None:
        _, second = self._make_pair()
        with self.assertRaisesRegex(ValueError, "both input_extension_init"):
            second.create_layer_extensions(
                extension_size=2,
                output_extension_init="net2wider",
                input_extension_init="kaiming",
            )

    def test_rejects_neuron_pairing(self) -> None:
        _, second = self._make_pair()
        with self.assertRaisesRegex(ValueError, "neuron_pairing"):
            second.create_layer_extensions(
                extension_size=2,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
                neuron_pairing="vv_z_negz",
            )

    def test_rejects_rescaling(self) -> None:
        _, second = self._make_pair()
        with self.assertRaisesRegex(ValueError, "rescaling"):
            second.create_layer_extensions(
                extension_size=2,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
                rescaling="default_vt",
            )

    def test_rejects_unequal_extension_sizes(self) -> None:
        _, second = self._make_pair()
        with self.assertRaisesRegex(ValueError, "equal input and output"):
            second.create_layer_extensions(
                extension_size=2,
                output_extension_size=2,
                input_extension_size=3,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
            )

    def test_probe_oracle_cross_check_injected_indices(self) -> None:
        first, second = self._make_pair(bias=True)
        x = torch.randn(12, first.in_features, device=self.device)
        selected = torch.tensor([1, 0, 1], dtype=torch.long)

        wa0 = first.weight.detach().clone()
        ba0 = first.bias.detach().clone() if first.bias is not None else None
        wb0 = second.weight.detach().clone()

        with torch.no_grad():
            y_before = second(first(x)).clone()

        wa_ref, ba_ref, wb_ref = _oracle_widen_linear(wa0, ba0, wb0, selected)
        bb0 = second.bias.detach().clone() if second.bias is not None else None
        with torch.no_grad():
            h_ref = torch.nn.functional.linear(x, wa_ref, ba_ref)
            y_ref = torch.nn.functional.linear(h_ref, wb_ref, bb0)

        second.create_layer_extensions(
            extension_size=3,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=selected,
        )
        second.apply_change(scaling_factor=1.0, extension_size=3)
        second.delete_update(include_previous=True)

        with torch.no_grad():
            y_gromo = second(first(x))

        self.assertAllClose(y_before, y_gromo, atol=FP_ATOL)
        self.assertAllClose(y_ref, y_gromo, atol=FP_ATOL)
        self.assertAllClose(first.weight, wa_ref, atol=1e-6)
        self.assertAllClose(second.weight, wb_ref, atol=1e-6)

    def test_complete_growth_entry_point(self) -> None:
        first, second = self._make_pair(hidden=4)
        second.target_in_neurons = 7
        x = torch.randn(8, first.in_features, device=self.device)
        with torch.no_grad():
            y_before = second(first(x)).clone()

        selected = torch.tensor([0, 3, 1], dtype=torch.long)
        second.complete_growth(
            {
                "output_extension_init": "net2wider",
                "input_extension_init": "net2wider",
                "selected_indices": selected,
            }
        )
        with torch.no_grad():
            y_after = second(first(x))
        self.assertAllClose(y_before, y_after, atol=FP_ATOL)
        self.assertEqual(second.in_features, 7)


class TestNet2WiderConvSmoke(TorchTestCase):
    """Conv→Conv joint path smoke (groups==1)."""

    def setUp(self) -> None:
        torch.manual_seed(1)
        self.device = global_device()

    def _make_pair(self) -> tuple[FullConv2dGrowingModule, FullConv2dGrowingModule]:
        first = FullConv2dGrowingModule(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
            use_bias=True,
            device=self.device,
            name="conv1",
        )
        second = FullConv2dGrowingModule(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            padding=1,
            use_bias=True,
            device=self.device,
            name="conv2",
            previous_module=first,
        )
        return first, second

    def test_function_preservation_eval_mode(self) -> None:
        first, second = self._make_pair()
        x = torch.randn(2, 2, 8, 8, device=self.device)
        selected = torch.tensor([0, 2], dtype=torch.long)

        first.eval()
        second.eval()
        with torch.no_grad():
            y_before = second(first(x)).clone()

        second.create_layer_extensions(
            extension_size=2,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=selected,
        )
        first.scaling_factor = 1.0  # type: ignore[assignment]
        second.scaling_factor = 1.0  # type: ignore[assignment]
        with torch.no_grad():
            h, h_ext = first.extended_forward(x)
            y_pre, y_ext = second.extended_forward(h, h_ext)
        assert y_ext is None
        self.assertAllClose(y_before, y_pre, atol=FP_ATOL)

        second.apply_change(scaling_factor=1.0, extension_size=2)
        second.delete_update(include_previous=True)
        with torch.no_grad():
            y_post = second(first(x))
        self.assertAllClose(y_before, y_post, atol=FP_ATOL)

    def test_groups_assert(self) -> None:
        first, second = self._make_pair()
        # Simulate grouped conv without relying on constructor support.
        object.__setattr__(first.layer, "groups", 2)
        with self.assertRaisesRegex(ValueError, "groups==1"):
            second.create_layer_extensions(
                extension_size=1,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
                selected_indices=torch.tensor([0]),
            )


class TestNet2WiderGrowingBatchNorm(TorchTestCase):
    """Linear/Conv → GrowingBatchNorm → next, joint Net2Wider FP."""

    def setUp(self) -> None:
        torch.manual_seed(2)
        # CPU float32: layer extensions are allocated in the module default dtype
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def _make_linear_bn_pair(
        self,
        *,
        affine: bool = True,
        track_running_stats: bool = True,
        hidden: int = 4,
    ) -> tuple[LinearGrowingModule, LinearGrowingModule, GrowingBatchNorm1d]:
        bn = GrowingBatchNorm1d(
            num_features=hidden,
            affine=affine,
            track_running_stats=track_running_stats,
            device=self.device,
            dtype=self.dtype,
            name="bn1d",
        )
        first = LinearGrowingModule(
            5,
            hidden,
            use_bias=True,
            name="first",
            device=self.device,
            post_layer_function=bn,
        )
        second = LinearGrowingModule(
            hidden,
            3,
            use_bias=True,
            name="second",
            previous_module=first,
            device=self.device,
        )
        with torch.no_grad():
            first.weight.normal_(0.0, 0.5)
            second.weight.normal_(0.0, 0.5)
            first.bias.uniform_(-0.3, 0.3)
            second.bias.uniform_(-0.3, 0.3)
            if affine:
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.uniform_(-0.2, 0.2)
        first.to(dtype=self.dtype)
        second.to(dtype=self.dtype)
        return first, second, bn

    def _forward_pair(
        self,
        first: LinearGrowingModule,
        second: LinearGrowingModule,
        x: torch.Tensor,
        *,
        use_extensions: bool,
    ) -> torch.Tensor:
        if use_extensions:
            first.scaling_factor = 1.0  # type: ignore[assignment]
            second.scaling_factor = 1.0  # type: ignore[assignment]
            h, h_ext = first.extended_forward(x)
            y, y_ext = second.extended_forward(h, h_ext)
            assert y_ext is None
            return y
        return second(first(x))

    def _warmup_bn(
        self,
        first: LinearGrowingModule,
        second: LinearGrowingModule,
        in_features: int,
        batches: int = 4,
    ) -> None:
        first.train()
        second.train()
        for _ in range(batches):
            x = torch.randn(16, in_features, device=self.device, dtype=self.dtype)
            second(first(x))

    def _assert_bn_replicas(
        self,
        bn: GrowingBatchNorm1d | GrowingBatchNorm2d,
        selected: torch.Tensor,
        num_base: int,
        *,
        mean_at_prepare: torch.Tensor | None = None,
        var_at_prepare: torch.Tensor | None = None,
    ) -> None:
        idx = selected.tolist()
        if bn.affine:
            self.assertAllClose(bn.weight[num_base:], bn.weight[idx], atol=1e-6)
            self.assertAllClose(bn.bias[num_base:], bn.bias[idx], atol=1e-6)
        if bn.track_running_stats:
            assert bn.running_mean is not None and bn.running_var is not None
            # Compare to prepare-time snapshots: train-mode forwards after
            # prepare update main running_* while pending stays frozen.
            mean_ref = (
                mean_at_prepare[idx]
                if mean_at_prepare is not None
                else bn.running_mean[idx]
            )
            var_ref = (
                var_at_prepare[idx] if var_at_prepare is not None else bn.running_var[idx]
            )
            self.assertAllClose(bn.running_mean[num_base:], mean_ref, atol=1e-6)
            self.assertAllClose(bn.running_var[num_base:], var_ref, atol=1e-6)

    def _run_linear_bn_fp(
        self,
        *,
        affine: bool = True,
        track_running_stats: bool = True,
        train_mode: bool,
    ) -> None:
        first, second, bn = self._make_linear_bn_pair(
            affine=affine, track_running_stats=track_running_stats
        )
        selected = torch.tensor([0, 2, 2, 1], dtype=torch.long)
        self._warmup_bn(first, second, first.in_features)

        if train_mode:
            first.train()
            second.train()
        else:
            first.eval()
            second.eval()

        x = torch.randn(16, first.in_features, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y_before = self._forward_pair(first, second, x, use_extensions=False).clone()

        gamma0 = bn.weight.detach().clone() if bn.affine else None
        beta0 = bn.bias.detach().clone() if bn.affine else None
        mean0 = (
            bn.running_mean.detach().clone()
            if bn.track_running_stats and bn.running_mean is not None
            else None
        )
        var0 = (
            bn.running_var.detach().clone()
            if bn.track_running_stats and bn.running_var is not None
            else None
        )

        second.create_layer_extensions(
            extension_size=4,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=selected,
        )
        self.assertIsNotNone(bn._net2wider_pending)
        pending = bn._net2wider_pending
        assert pending is not None
        pending_mean = (
            pending.running_mean.detach().clone()
            if pending.running_mean is not None
            else None
        )
        pending_var = (
            pending.running_var.detach().clone()
            if pending.running_var is not None
            else None
        )

        with torch.no_grad():
            y_pre = self._forward_pair(first, second, x, use_extensions=True)
        self.assertAllClose(y_before, y_pre, atol=FP_ATOL)

        # Ext path must not mutate pending running buffers.
        if pending_mean is not None and pending_var is not None:
            assert pending.running_mean is not None and pending.running_var is not None
            self.assertAllClose(pending.running_mean, pending_mean, atol=1e-12)
            self.assertAllClose(pending.running_var, pending_var, atol=1e-12)
        # In eval, main running_* should also stay frozen across the ext forward.
        if not train_mode and mean0 is not None and var0 is not None:
            self.assertAllClose(bn.running_mean, mean0, atol=1e-12)
            self.assertAllClose(bn.running_var, var0, atol=1e-12)

        second.apply_change(scaling_factor=1.0, extension_size=4)
        second.delete_update(include_previous=True)
        self.assertIsNone(bn._net2wider_pending)
        self.assertEqual(bn.num_features, 8)
        self._assert_bn_replicas(
            bn,
            selected,
            num_base=4,
            mean_at_prepare=mean0,
            var_at_prepare=var0,
        )

        with torch.no_grad():
            y_post = self._forward_pair(first, second, x, use_extensions=False)
        self.assertAllClose(y_before, y_post, atol=FP_ATOL)

        if gamma0 is not None and beta0 is not None:
            self.assertAllClose(bn.weight[:4], gamma0, atol=1e-6)
            self.assertAllClose(bn.bias[:4], beta0, atol=1e-6)

    def test_linear_bn1d_fp_eval(self) -> None:
        self._run_linear_bn_fp(train_mode=False)

    def test_linear_bn1d_fp_train(self) -> None:
        self._run_linear_bn_fp(train_mode=True)

    def test_linear_bn1d_affine_false_fp(self) -> None:
        self._run_linear_bn_fp(affine=False, train_mode=False)
        self._run_linear_bn_fp(affine=False, train_mode=True)

    def test_linear_bn1d_track_running_stats_false_fp(self) -> None:
        self._run_linear_bn_fp(track_running_stats=False, train_mode=False)
        self._run_linear_bn_fp(track_running_stats=False, train_mode=True)

    def test_conv_bn2d_fp_eval_and_train(self) -> None:
        selected = torch.tensor([0, 2], dtype=torch.long)

        def _make_conv_pair(
            name_suffix: str,
        ) -> tuple[FullConv2dGrowingModule, FullConv2dGrowingModule, GrowingBatchNorm2d]:
            bn = GrowingBatchNorm2d(
                num_features=3,
                device=self.device,
                dtype=self.dtype,
                name=f"bn2d{name_suffix}",
            )
            first = FullConv2dGrowingModule(
                in_channels=2,
                out_channels=3,
                kernel_size=3,
                padding=1,
                use_bias=True,
                device=self.device,
                name=f"conv1{name_suffix}",
                post_layer_function=bn,
            )
            second = FullConv2dGrowingModule(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                padding=1,
                use_bias=True,
                device=self.device,
                name=f"conv2{name_suffix}",
                previous_module=first,
            )
            first.to(dtype=self.dtype)
            second.to(dtype=self.dtype)
            with torch.no_grad():
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.uniform_(-0.2, 0.2)
            return first, second, bn

        for train_mode in (False, True):
            first, second, bn = _make_conv_pair("_eval" if not train_mode else "_train")
            first.train()
            second.train()
            for _ in range(4):
                second(
                    first(torch.randn(2, 2, 8, 8, device=self.device, dtype=self.dtype))
                )
            if train_mode:
                first.train()
                second.train()
            else:
                first.eval()
                second.eval()

            x = torch.randn(2, 2, 8, 8, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                y_before = second(first(x)).clone()
            mean0 = (
                bn.running_mean.detach().clone() if bn.running_mean is not None else None
            )
            var0 = bn.running_var.detach().clone() if bn.running_var is not None else None
            second.create_layer_extensions(
                extension_size=2,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
                selected_indices=selected,
            )
            first.scaling_factor = 1.0  # type: ignore[assignment]
            second.scaling_factor = 1.0  # type: ignore[assignment]
            with torch.no_grad():
                h, h_ext = first.extended_forward(x)
                y_pre, y_ext = second.extended_forward(h, h_ext)
            assert y_ext is None
            self.assertAllClose(y_before, y_pre, atol=FP_ATOL)
            second.apply_change(scaling_factor=1.0, extension_size=2)
            second.delete_update(include_previous=True)
            self._assert_bn_replicas(
                bn,
                selected,
                num_base=3,
                mean_at_prepare=mean0,
                var_at_prepare=var0,
            )
            with torch.no_grad():
                y_post = second(first(x))
            self.assertAllClose(y_before, y_post, atol=FP_ATOL)

    def test_rejects_plain_batchnorm(self) -> None:
        first = LinearGrowingModule(
            4, 3, name="f", device=self.device, post_layer_function=nn.BatchNorm1d(3)
        )
        second = LinearGrowingModule(
            3, 2, name="s", previous_module=first, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "GrowingBatchNorm"):
            second.create_layer_extensions(
                extension_size=1,
                output_extension_init="net2wider",
                input_extension_init="net2wider",
                selected_indices=torch.tensor([0]),
            )

    def test_delete_update_clears_pending_without_consume(self) -> None:
        _, second, bn = self._make_linear_bn_pair()
        selected = torch.tensor([1, 0], dtype=torch.long)
        with torch.no_grad():
            bn.weight.fill_(2.0)
            bn.bias.fill_(-1.0)

        second.create_layer_extensions(
            extension_size=2,
            output_extension_init="net2wider",
            input_extension_init="net2wider",
            selected_indices=selected,
        )
        self.assertIsNotNone(bn._net2wider_pending)
        pending_w = bn._net2wider_pending.weight.detach().clone()  # type: ignore[union-attr]
        self.assertAllClose(pending_w, torch.tensor([2.0, 2.0], dtype=self.dtype))

        second.delete_update(include_previous=True)
        self.assertIsNone(bn._net2wider_pending)
        self.assertEqual(bn.num_features, 4)

        # Non-net2wider grow must not pick up stale replicas (defaults: weight=1, bias=0).
        bn.grow(2, consume_net2wider=False)
        self.assertAllClose(
            bn.weight[-2:], torch.ones(2, dtype=self.dtype, device=self.device)
        )
        self.assertAllClose(
            bn.bias[-2:], torch.zeros(2, dtype=self.dtype, device=self.device)
        )
        self.assertFalse(torch.allclose(bn.weight[-2:], pending_w))


if __name__ == "__main__":
    unittest.main()
