import unittest

import torch
import torch.nn as nn

from gromo.containers.growing_vision_transformer import GrowingTransformer
from tests.test_growing_container import create_synthetic_data, gather_statistics
from tests.torch_unittest import TorchTestCase


class TestGrowingTransformer(TorchTestCase):
    def setUp(self):
        self.in_features = (3, 16, 16)
        self.out_features = 5
        self.num_samples = 12
        self.batch_size = 3
        self.dataloader = create_synthetic_data(
            self.num_samples, self.in_features, (self.out_features,), self.batch_size
        )

        self.patch_size = 4
        self.d_model = 16
        self.num_heads = 4
        self.d_ff = 8
        self.num_blocks = 2

        self.model = GrowingTransformer(
            in_features=self.in_features,
            out_features=self.out_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_blocks=self.num_blocks,
            device=torch.device("cpu"),
        )

        self.loss = nn.MSELoss()

        gather_statistics(self.dataloader, self.model, self.loss)
        with self.assertMaybeWarns(
            UserWarning,
            "Using the pseudo-inverse for the computation of the optimal delta",
        ):
            self.model.compute_optimal_updates()

    def test_init(self):
        model = GrowingTransformer(
            in_features=self.in_features,
            out_features=self.out_features,
            patch_size=self.patch_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_blocks=self.num_blocks,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(model, GrowingTransformer)
        self.assertIsInstance(model, torch.nn.Module)

    def test_forward(self):
        x = torch.randn(1, *self.in_features)
        y = self.model(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_extended_forward(self):
        x = torch.randn(1, *self.in_features)
        y = self.model.extended_forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_set_growing_layers(self):
        self.model.set_growing_layers()
        self.assertEqual(len(self.model._growing_layers), self.num_blocks)

    def test_weights_statistics(self):
        stats = self.model.weights_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("blocks", stats)
        self.assertEqual(len(stats["blocks"]), self.num_blocks)

    def test_update_information(self):
        info = self.model.update_information()
        self.assertIsInstance(info, dict)
        self.assertIn("blocks", info)
        self.assertEqual(len(info["blocks"]), self.num_blocks)

    def test_select_update(self):
        layer_index = 0
        selected_index = self.model.select_update(layer_index=layer_index)
        self.assertEqual(selected_index, layer_index)


if __name__ == "__main__":
    unittest.main()
