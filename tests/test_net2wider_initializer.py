"""Tests for joint Net2Wider function-preserving growth."""

from __future__ import annotations

import unittest

import torch

from gromo.modules.conv2d_growing_module import FullConv2dGrowingModule
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
    """Conv→Conv joint path smoke (groups==1; BN/residual out of scope)."""

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


if __name__ == "__main__":
    unittest.main()
