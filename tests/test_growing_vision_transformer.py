import unittest

import torch
import torch.nn as nn

from gromo.containers.growing_transformer import GrowingTransformerBlock, MaskedAttention
from gromo.containers.growing_vision_transformer import (
    GrowingCCT,
    GrowingCVT,
    GrowingTextViTLite,
    GrowingTransformer,
    GrowingTransformerClassifier,
    GrowingViTLite,
)
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
        with self.assertMaybeWarns(UserWarning):
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

    def test_cct_encoder_layer_signature(self):
        block = GrowingTransformerBlock(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, 7, self.d_model)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_cct_style_mask_matches_key_padding_mask(self):
        attn = MaskedAttention(
            dim=self.d_model,
            num_heads=self.num_heads,
            attention_dropout=0.0,
            projection_dropout=0.0,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 5, self.d_model)
        valid_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )

        y_from_mask = attn(x, mask=valid_mask)
        y_from_padding = attn(x, key_padding_mask=~valid_mask)

        self.assertAllClose(y_from_mask, y_from_padding)

    def test_growing_transformer_classifier_forward(self):
        classifier = GrowingTransformerClassifier(
            sequence_length=7,
            embedding_dim=self.d_model,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, 7, self.d_model)
        mask = torch.tensor([[True, True, True, True, False, False, False], [True] * 7])
        y = classifier(x)
        y_masked = classifier(x, mask=mask)
        y_ext = classifier.extended_forward(x)

        self.assertEqual(y.shape, (2, self.out_features))
        self.assertEqual(y_masked.shape, (2, self.out_features))
        self.assertEqual(y_ext.shape, (2, self.out_features))

    def test_growing_transformer_generic_forward(self):
        model = GrowingTransformer(
            img_size=16,
            embedding_dim=self.d_model,
            n_input_channels=self.in_features[0],
            n_conv_layers=1,
            kernel_size=3,
            stride=1,
            padding=1,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, *self.in_features)

        self.assertEqual(model(x).shape, (2, self.out_features))
        self.assertEqual(model.extended_forward(x).shape, (2, self.out_features))

    def test_growing_cct_forward(self):
        model = GrowingCCT(
            img_size=16,
            embedding_dim=self.d_model,
            n_input_channels=self.in_features[0],
            n_conv_layers=1,
            kernel_size=3,
            stride=1,
            padding=1,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, *self.in_features)
        y = model(x)
        y_ext = model.extended_forward(x)

        self.assertIsInstance(model, GrowingTransformer)
        self.assertEqual(y.shape, (2, self.out_features))
        self.assertEqual(y_ext.shape, (2, self.out_features))

    def test_growing_vit_lite_and_cvt_forward(self):
        vit = GrowingViTLite(
            img_size=16,
            embedding_dim=self.d_model,
            n_input_channels=self.in_features[0],
            kernel_size=4,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )
        cvt = GrowingCVT(
            img_size=16,
            embedding_dim=self.d_model,
            n_input_channels=self.in_features[0],
            kernel_size=4,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, *self.in_features)

        self.assertIsInstance(vit, GrowingTransformer)
        self.assertIsInstance(cvt, GrowingTransformer)
        self.assertEqual(vit(x).shape, (2, self.out_features))
        self.assertEqual(vit.extended_forward(x).shape, (2, self.out_features))
        self.assertEqual(cvt(x).shape, (2, self.out_features))
        self.assertEqual(cvt.extended_forward(x).shape, (2, self.out_features))

    def test_growing_text_vit_lite_forward(self):
        model = GrowingTextViTLite(
            seq_len=8,
            word_embedding_dim=self.d_model,
            embedding_dim=self.d_model,
            patch_size=2,
            vocab_size=20,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=torch.device("cpu"),
        )

        x = torch.randint(0, 20, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)

        self.assertEqual(model(x, mask=mask).shape, (2, self.out_features))
        self.assertEqual(
            model.extended_forward(x, attention_mask=mask).shape,
            (2, self.out_features),
        )


if __name__ == "__main__":
    unittest.main()
