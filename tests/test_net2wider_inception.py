"""M2a: Mini Inception channel-concat Net2Wider FP + random-pad parity."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from gromo.containers.growing_mini_inception import MiniInceptionGrowingModule
from gromo.modules.growing_normalisation import GrowingGroupNorm
from gromo.utils.net2wider_concat import (
    apply_concat_net2wider_change,
    create_concat_net2wider_extensions,
    net2wider_widen_graph,
)
from tests.torch_unittest import TorchTestCase


FP_ATOL = 1e-4


class TestMiniInceptionNet2Wider(TorchTestCase):
    """Channel-concat mini Inception Net2Wider (create → immediate apply)."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def _make_mini(
        self,
        *,
        in_channels: int = 4,
        c_a: int = 3,
        c_b_reduce: int = 2,
        c_b: int = 3,
        out_channels: int = 5,
    ) -> MiniInceptionGrowingModule:
        return MiniInceptionGrowingModule(
            in_channels=in_channels,
            branch_a_channels=c_a,
            branch_b_reduce_channels=c_b_reduce,
            branch_b_channels=c_b,
            out_channels=out_channels,
            device=self.device,
            name="mini_inc",
        ).to(dtype=self.dtype)

    def _warmup(self, model: MiniInceptionGrowingModule, batches: int = 4) -> None:
        model.train()
        for _ in range(batches):
            x = torch.randn(
                2, model.in_features, 8, 8, device=self.device, dtype=self.dtype
            )
            model(x)

    def test_forward_concat_shapes(self) -> None:
        model = self._make_mini()
        x = torch.randn(2, 4, 8, 8, device=self.device, dtype=self.dtype)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 5, 8, 8))
        self.assertEqual(model.concat_offsets["branch_a"], 0)
        self.assertEqual(model.concat_offsets["branch_b"], 3)
        self.assertEqual(model.concat_width, 6)

    def test_module_widen_branch_a_fp_eval_and_train(self) -> None:
        selected = torch.tensor([0, 2], dtype=torch.long)
        for train_mode in (False, True):
            model = self._make_mini()
            self._warmup(model)
            if train_mode:
                model.train()
            else:
                model.eval()
            x = torch.randn(2, 4, 8, 8, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                y_before = model(x).clone()

            g = model.net2wider_widen(
                edge="branch_a",
                extension_size=2,
                selected_indices=selected,
                mode="net2wider",
            )
            self.assertTrue(torch.equal(g, selected))
            # create → immediate apply (no train between)
            model.apply_net2wider_widen(edge="branch_a", extension_size=2)

            with torch.no_grad():
                y_after = model(x)
            delta = (y_before - y_after).abs().max().item()
            self.assertLessEqual(
                delta,
                FP_ATOL,
                msg=f"branch_a FP failed train_mode={train_mode}: delta={delta}",
            )
            self.assertEqual(model.branch_a.out_channels, 5)
            self.assertEqual(model.project.in_channels, 8)
            self.assertEqual(model.concat_offsets["branch_b"], 5)

    def test_module_widen_branch_b_fp(self) -> None:
        model = self._make_mini()
        self._warmup(model)
        model.eval()
        selected = torch.tensor([1, 0], dtype=torch.long)
        x = torch.randn(2, 4, 8, 8, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y_before = model(x).clone()

        model.net2wider_widen(
            edge="branch_b",
            extension_size=2,
            selected_indices=selected,
            mode="net2wider",
        )
        model.apply_net2wider_widen(edge="branch_b", extension_size=2)

        with torch.no_grad():
            y_after = model(x)
        self.assertAllClose(y_before, y_after, atol=FP_ATOL)
        self.assertEqual(model.branch_b_expand.out_channels, 5)
        self.assertEqual(model.concat_offsets["branch_a"], 0)
        self.assertEqual(model.concat_offsets["branch_b"], 3)

    def test_internal_branch_b_reduce_pairwise_fp(self) -> None:
        """Internal reduce→expand uses shared g via module API."""
        model = self._make_mini()
        self._warmup(model)
        model.eval()
        selected = torch.tensor([0, 1], dtype=torch.long)
        x = torch.randn(2, 4, 8, 8, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y_before = model(x).clone()

        model.net2wider_widen(
            edge="branch_b_reduce",
            extension_size=2,
            selected_indices=selected,
            mode="net2wider",
        )
        model.apply_net2wider_widen(edge="branch_b_reduce", extension_size=2)

        with torch.no_grad():
            y_after = model(x)
        self.assertAllClose(y_before, y_after, atol=FP_ATOL)
        self.assertEqual(model.branch_b_reduce.out_channels, 4)
        self.assertEqual(model.branch_b_expand.in_channels, 4)

    def test_graph_level_widen_api_shared_g_and_offsets(self) -> None:
        model = self._make_mini()
        self._warmup(model)
        model.eval()
        selected = torch.tensor([2, 1], dtype=torch.long)
        x = torch.randn(2, 4, 8, 8, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            y_before = model(x).clone()

        g = net2wider_widen_graph(
            producer=model.branch_a,
            consumers=[(model.project, model.concat_offsets["branch_a"])],
            extension_size=2,
            selected_indices=selected,
            mode="net2wider",
        )
        self.assertTrue(torch.equal(g, selected))
        apply_concat_net2wider_change(
            producer=model.branch_a,
            consumer=model.project,
            extension_size=2,
            channel_offset=model.concat_offsets["branch_a"],
        )
        model._refresh_concat_layout()

        with torch.no_grad():
            y_after = model(x)
        self.assertAllClose(y_before, y_after, atol=FP_ATOL)

    def test_rejects_groupnorm(self) -> None:
        model = self._make_mini()
        # Inject unsupported GroupNorm into branch_a post-layer
        gn = GrowingGroupNorm(1, 3, device=self.device)
        model.branch_a.post_layer_function = nn.Sequential(gn, nn.ReLU())
        with self.assertRaisesRegex(ValueError, "GroupNorm|GrowingGroupNorm|net2wider"):
            model.net2wider_widen(
                edge="branch_a",
                extension_size=1,
                selected_indices=torch.tensor([0]),
                mode="net2wider",
            )

    def test_rejects_residual_shortcut(self) -> None:
        model = self._make_mini()
        with self.assertRaisesRegex(ValueError, "residual|shortcut"):
            model.net2wider_widen(
                edge="residual",
                extension_size=1,
                selected_indices=torch.tensor([0]),
                mode="net2wider",
            )


class TestRandomPadParity(TorchTestCase):
    """Random-pad must be true weight-tensor pad, not slice-copy proxy."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def test_random_pad_is_not_slice_copy_proxy(self) -> None:
        model = MiniInceptionGrowingModule(
            in_channels=4,
            branch_a_channels=4,
            branch_b_reduce_channels=2,
            branch_b_channels=3,
            out_channels=5,
            device=self.device,
            name="mini_pad",
        ).to(dtype=self.dtype)
        model.eval()
        selected = torch.tensor([0, 1, 2], dtype=torch.long)

        w_prod_before = model.branch_a.weight.detach().clone()
        w_cons_before = model.project.weight.detach().clone()
        # Snapshot columns that would be scaled under Net2Wider
        col0_before = w_cons_before[:, 0].clone()

        create_concat_net2wider_extensions(
            producer=model.branch_a,
            consumer=model.project,
            extension_size=3,
            selected_indices=selected,
            channel_offset=0,
            mode="random_pad",
            generator=torch.Generator().manual_seed(123),
        )
        apply_concat_net2wider_change(
            producer=model.branch_a,
            consumer=model.project,
            extension_size=3,
            channel_offset=0,
        )

        # Existing producer channels unchanged
        self.assertAllClose(model.branch_a.weight[:4], w_prod_before, atol=1e-7)
        # New producer channels are NOT replicas of selected (true random pad)
        new_out = model.branch_a.weight[4:]
        for i, src in enumerate(selected.tolist()):
            self.assertFalse(
                torch.allclose(new_out[i], w_prod_before[src], atol=1e-5),
                msg=f"random_pad producer channel {i} equals replica of {src}",
            )
        # Consumer existing columns NOT scaled by 1/c (Net2Wider would change col0)
        self.assertAllClose(model.project.weight[:, :4], w_cons_before[:, :4], atol=1e-7)
        self.assertAllClose(model.project.weight[:, 0], col0_before, atol=1e-7)
        # New consumer in-channels are random, not copies of selected columns
        new_in = model.project.weight[:, 4:]
        for i, src in enumerate(selected.tolist()):
            self.assertFalse(
                torch.allclose(new_in[:, i], w_cons_before[:, src], atol=1e-5),
                msg=f"random_pad consumer col {i} equals copy of src {src}",
            )


if __name__ == "__main__":
    unittest.main()
