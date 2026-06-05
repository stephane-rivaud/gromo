import runpy
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from gromo.containers.growing_transformer import (
    Attention,
    DropPath,
    ExtendedDropout,
    GrowingTransformerBlock,
    MaskedAttention,
    drop_path,
)
from gromo.containers.growing_vision_transformer import (
    Embedder,
    GrowingCCT,
    GrowingCVT,
    GrowingTextViTLite,
    GrowingTransformer,
    GrowingTransformerClassifier,
    GrowingViTLite,
    TextTokenizer,
    Tokenizer,
    check_patch_grid,
)
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.growing_module import GrowingModule
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
        self.assertIsInstance(model, SequentialGrowingModel)

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

    def test_set_growing_layers_with_index(self):
        self.model.set_growing_layers(index=1)
        self.assertEqual(self.model.layer_to_grow_index, 1)
        self.assertEqual(len(self.model._growing_layers), 1)
        self.assertIs(self.model._growing_layers[0], self.model.blocks[1].growth_module)

    def test_set_growing_layers_sequential(self):
        self.model.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(self.model.layer_to_grow_index, 0)
        self.assertEqual(len(self.model._growing_layers), 1)
        self.assertIs(self.model._growing_layers[0], self.model.blocks[0].growth_module)

        self.model.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(self.model.layer_to_grow_index, 1)
        self.assertEqual(len(self.model._growing_layers), 1)
        self.assertIs(self.model._growing_layers[0], self.model.blocks[1].growth_module)

    def test_compute_optimal_updates_on_selected_block_only(self):
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
        gather_statistics(self.dataloader, model, self.loss)
        model.set_growing_layers(index=1)

        with self.assertMaybeWarns(UserWarning):
            model.compute_optimal_updates()

        self.assertIsNone(model.blocks[0].optimal_delta_layer)
        self.assertIsNotNone(model.blocks[1].optimal_delta_layer)

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
        self.assertIs(
            self.model.currently_updated_layer, self.model.blocks[0].growth_module
        )
        self.assertIsInstance(self.model.currently_updated_layer, GrowingModule)

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

        self.assertIsInstance(classifier, SequentialGrowingModel)
        x = torch.randn(2, 7, self.d_model)
        mask = torch.tensor([[True, True, True, True, False, False, False], [True] * 7])
        y = classifier(x)
        y_masked = classifier(x, mask=mask)
        y_ext = classifier.extended_forward(x)

        self.assertEqual(y.shape, (2, self.out_features))
        self.assertEqual(y_masked.shape, (2, self.out_features))
        self.assertEqual(y_ext.shape, (2, self.out_features))

        classifier.set_growing_layers(index=1)
        self.assertEqual(classifier.layer_to_grow_index, 1)
        self.assertEqual(len(classifier._growing_layers), 1)
        self.assertIs(classifier._growing_layers[0], classifier.blocks[1].growth_module)

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
        model.set_growing_layers(index=1)
        self.assertIs(model._growing_layers[0], model.blocks[1].growth_module)

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

        self.assertIsInstance(model, SequentialGrowingModel)
        x = torch.randint(0, 20, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)

        self.assertEqual(model(x, mask=mask).shape, (2, self.out_features))
        self.assertEqual(
            model.extended_forward(x, attention_mask=mask).shape,
            (2, self.out_features),
        )

        model.set_growing_layers(index=1)
        self.assertEqual(model.layer_to_grow_index, 1)
        self.assertEqual(len(model._growing_layers), 1)
        self.assertIs(model._growing_layers[0], model.classifier.blocks[1].growth_module)


class TestGrowingTransformerCoveragePaths(TorchTestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.in_features = (3, 16, 16)
        self.out_features = 5
        self.d_model = 16
        self.num_heads = 4
        self.d_ff = 8

    def test_drop_path_and_extended_dropout_helpers(self):
        x = torch.ones(2, 3, 4)
        x_ext = torch.randn(2, 3, 4)

        self.assertAllClose(drop_path(x, drop_prob=0.0, training=True), x)
        self.assertAllClose(drop_path(x, drop_prob=0.5, training=False), x)

        with patch(
            "gromo.containers.growing_transformer.torch.rand",
            return_value=torch.zeros((2, 1, 1)),
        ):
            dropped = drop_path(x, drop_prob=0.5, training=True)
            self.assertAllClose(dropped, torch.zeros_like(x))

        drop_module = DropPath(0.5)
        drop_module.train()
        with patch(
            "gromo.containers.growing_transformer.torch.rand",
            return_value=torch.zeros((2, 1, 1)),
        ):
            dropped, forwarded_ext = drop_module.extended_forward(x, x_ext)
            self.assertAllClose(dropped, torch.zeros_like(x))
            self.assertIs(forwarded_ext, x_ext)

        dropped_none, forwarded_ext = drop_module.extended_forward(None, x_ext)
        self.assertIsNone(dropped_none)
        self.assertIs(forwarded_ext, x_ext)

        dropout = ExtendedDropout(0.0)
        kept, forwarded_ext = dropout.extended_forward(x, x_ext)
        self.assertAllClose(kept, x)
        self.assertIs(forwarded_ext, x_ext)

        kept_none, forwarded_ext = dropout.extended_forward(None, x_ext)
        self.assertIsNone(kept_none)
        self.assertIs(forwarded_ext, x_ext)

    def test_attention_mask_resolution_and_validation(self):
        with self.assertRaises(ValueError):
            Attention(dim=10, num_heads=4, device=self.device)

        plain_attention = Attention(
            dim=self.d_model,
            num_heads=self.num_heads,
            attention_dropout=0.0,
            projection_dropout=0.0,
            device=self.device,
        )
        plain_output = plain_attention(torch.randn(2, 5, self.d_model))
        self.assertEqual(plain_output.shape, (2, 5, self.d_model))

        attn = MaskedAttention(
            dim=self.d_model,
            num_heads=self.num_heads,
            attention_dropout=0.0,
            projection_dropout=0.0,
            device=self.device,
        )
        x = torch.randn(2, 5, self.d_model)
        valid_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )

        y_from_mask = attn(x, mask=valid_mask)
        y_from_attn_mask = attn(x, attn_mask=valid_mask)
        y_from_padding = attn(x, key_padding_mask=~valid_mask)

        self.assertAllClose(y_from_mask, y_from_attn_mask)
        self.assertAllClose(y_from_mask, y_from_padding)

        resolved_mask = attn._resolve_mask(
            batch_size=2,
            sequence_length=5,
            key_padding_mask=~valid_mask,
        )
        assert resolved_mask is not None
        self.assertTrue(torch.equal(resolved_mask, valid_mask))

        with self.assertRaises(ValueError):
            attn(x, attn_mask=torch.ones(2, 5, 5, dtype=torch.bool))

        with self.assertRaises(ValueError):
            attn._resolve_mask(
                batch_size=2,
                sequence_length=5,
                mask=torch.ones(2, 4, dtype=torch.bool),
            )

    def test_transformer_block_validation_delegation_and_statistics(self):
        with self.assertRaises(TypeError):
            GrowingTransformerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                0.0,
                self.device,
                None,
                None,
                None,
                0.0,
                None,
                "unexpected",
            )
        with self.assertRaises(TypeError):
            GrowingTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                device=self.device,
                unsupported=True,
            )
        with self.assertRaises(ValueError):
            GrowingTransformerBlock(d_model=self.d_model, d_ff=self.d_ff)
        with self.assertRaises(ValueError):
            GrowingTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=None,
                dim_feedforward=None,
            )

        block = GrowingTransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            drop_path_rate=0.0,
            device=self.device,
        )

        with self.assertWarns(DeprecationWarning):
            hidden_features = block.hidden_features
        self.assertEqual(hidden_features, block.hidden_neurons)

        dummy_delta = nn.Identity()
        GrowingTransformerBlock.optimal_delta_layer.fset(block, dummy_delta)
        self.assertIs(block.optimal_delta_layer, dummy_delta)

        block.parameter_update_decrease = torch.tensor(1.25)
        self.assertEqual(block.parameter_update_decrease.item(), 1.25)
        block.scaling_factor = 0.5
        self.assertAlmostEqual(block.scaling_factor.item(), 0.5)

        with patch.object(block.mlp, "delete_update") as delete_update:
            block.delete_update(reset=True)
            delete_update.assert_called_once_with(reset=True)
        with patch.object(block.mlp, "set_scaling_factor") as set_scaling_factor:
            block.set_scaling_factor(0.75)
            set_scaling_factor.assert_called_once_with(0.75)
        with patch.object(block.mlp, "apply_change") as apply_change:
            block.apply_change(
                extension_size=2,
                scaling_factor=0.5,
                apply_delta=False,
                apply_extension=True,
            )
            apply_change.assert_called_once_with(
                extension_size=2,
                scaling_factor=0.5,
                apply_delta=False,
                apply_extension=True,
            )
        with patch.object(block.mlp, "missing_neurons", return_value=3) as missing:
            self.assertEqual(block.missing_neurons(), 3)
            missing.assert_called_once_with()
        with patch.object(
            block.mlp, "number_of_neurons_to_add", return_value=2
        ) as number_to_add:
            self.assertEqual(block.number_of_neurons_to_add(method="fixed"), 2)
            number_to_add.assert_called_once_with(method="fixed")
        with patch.object(block.mlp, "complete_growth") as complete_growth:
            block.complete_growth(extension_kwargs={"extension_size": 1})
            complete_growth.assert_called_once_with(
                extension_kwargs={"extension_size": 1}
            )
        with patch.object(block.mlp, "create_layer_extensions") as create_extensions:
            block.create_layer_extensions(extension_size=2)
            create_extensions.assert_called_once_with(extension_size=2)
        with patch.object(block.mlp, "sub_select_optimal_added_parameters") as sub_select:
            block.sub_select_optimal_added_parameters(
                keep_neurons=1,
                threshold=0.2,
                sub_select_previous=False,
                zeros_if_not_enough=True,
                zeros_fan_in=False,
                zeros_fan_out=True,
            )
            sub_select.assert_called_once_with(
                keep_neurons=1,
                threshold=0.2,
                sub_select_previous=False,
                zeros_if_not_enough=True,
                zeros_fan_in=False,
                zeros_fan_out=True,
            )
        with patch.object(block.mlp, "apply_rescaling") as apply_rescaling:
            block.apply_rescaling(strategy="unit-test")
            apply_rescaling.assert_called_once_with(strategy="unit-test")
        with patch.object(block.mlp, "apply_neuron_pairing") as apply_neuron_pairing:
            block.apply_neuron_pairing(neuron_pairing="unit-test")
            apply_neuron_pairing.assert_called_once_with(neuron_pairing="unit-test")
        with patch.object(block.mlp, "normalize_optimal_updates") as normalize_updates:
            block.normalize_optimal_updates(mode="unit-test")
            normalize_updates.assert_called_once_with(mode="unit-test")

        block.mlp.second_layer.eigenvalues_extension = None
        info = block.update_information()
        self.assertEqual(info["added_neurons"], 0)
        self.assertEqual(info["d_model"], self.d_model)
        self.assertEqual(info["d_ff"], self.d_ff)

        block.self_attn.qkv = nn.Linear(
            self.d_model,
            self.d_model * 3,
            bias=True,
            device=self.device,
        )
        stats = block.weights_statistics()
        self.assertIn("qkv_bias", stats["attention"])
        self.assertIn("proj_bias", stats["attention"])

        stochastic_block = GrowingTransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.1,
            attention_dropout=0.0,
            drop_path_rate=0.2,
            device=self.device,
        )
        stochastic_output = stochastic_block(torch.randn(2, 4, self.d_model))
        self.assertEqual(stochastic_output.shape, (2, 4, self.d_model))
        stochastic_block.parameter_update_decrease = torch.tensor(0.5)
        stochastic_block.mlp.second_layer.eigenvalues_extension = torch.tensor([0.3, 0.2])
        stochastic_info = stochastic_block.update_information()
        self.assertEqual(stochastic_info["added_neurons"], 2)

    def test_patch_grid_tokenizer_text_tokenizer_and_embedder_helpers(self):
        patch_shape, num_patches = check_patch_grid(16, 8, (4, 2))
        self.assertEqual(patch_shape, (4, 2))
        self.assertEqual(num_patches, 16)

        square_patch_shape, num_square_patches = check_patch_grid(16, 16, 4)
        self.assertEqual(square_patch_shape, (4, 4))
        self.assertEqual(num_square_patches, 16)

        with self.assertRaises(AssertionError):
            check_patch_grid(15, 16, 4)

        tokenizer = Tokenizer(
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_kernel_size=2,
            pooling_stride=2,
            pooling_padding=0,
            n_conv_layers=2,
            n_input_channels=3,
            n_output_channels=6,
            in_planes=4,
            activation=nn.ReLU,
            max_pool=False,
            conv_bias=True,
        )
        image_batch = torch.randn(2, 3, 8, 8)
        token_sequence = tokenizer(image_batch)
        self.assertEqual(token_sequence.shape[0], 2)
        self.assertEqual(
            tokenizer.sequence_length(n_channels=3, height=8, width=8),
            token_sequence.shape[1],
        )

        text_tokenizer = TextTokenizer(
            kernel_size=2,
            stride=2,
            padding=0,
            pooling_kernel_size=2,
            pooling_stride=2,
            pooling_padding=0,
            embedding_dim=4,
            n_output_channels=3,
            activation=nn.ReLU,
            max_pool=True,
        )
        text_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        token_mask = text_tokenizer.forward_mask(text_mask)
        text_embeddings = torch.randn(2, 4, 4)
        tokenized_text, propagated_mask = text_tokenizer(text_embeddings, mask=text_mask)
        tokenized_text_no_mask, propagated_none = text_tokenizer(text_embeddings)

        self.assertEqual(tokenized_text.shape[-1], 3)
        self.assertEqual(token_mask.dtype, torch.bool)
        self.assertIsNotNone(propagated_mask)
        self.assertIsNone(propagated_none)
        self.assertEqual(
            text_tokenizer.seq_len(seq_len=4, embed_dim=4),
            tokenized_text_no_mask.shape[1],
        )

        pretrained_weight = torch.randn(7, 4)
        embedder = Embedder(
            word_embedding_dim=4,
            vocab_size=7,
            pretrained_weight=pretrained_weight,
            embed_freeze=True,
        )
        self.assertFalse(embedder.embeddings.weight.requires_grad)
        embedded_tokens, returned_mask = embedder(
            torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            mask=text_mask,
        )
        self.assertEqual(embedded_tokens.shape, (2, 4, 4))
        self.assertTrue(torch.equal(returned_mask, text_mask))

        linear = nn.Linear(4, 2)
        embedding = nn.Embedding(7, 4)
        Embedder.init_weight(linear)
        Embedder.init_weight(embedding)

    def test_classifier_padding_pooling_and_positional_embedding_branches(self):
        with self.assertRaises(ValueError):
            GrowingTransformerClassifier(
                sequence_length=None,
                embedding_dim=self.d_model,
                num_layers=1,
                num_heads=self.num_heads,
                mlp_ratio=0.5,
                num_classes=self.out_features,
                positional_embedding="learnable",
                device=self.device,
            )

        classifier = GrowingTransformerClassifier(
            sequence_length=4,
            seq_pool=False,
            embedding_dim=self.d_model,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            positional_embedding="unsupported",
            device=self.device,
        )
        self.assertEqual(classifier.positional_embedding, "sine")

        token_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
        prepared_tokens, prepared_mask = classifier._prepare_tokens(
            torch.randn(2, 4, self.d_model),
            token_mask,
        )
        self.assertEqual(prepared_tokens.shape, (2, 5, self.d_model))
        self.assertEqual(prepared_mask.shape, (2, 5))
        self.assertEqual(
            GrowingTransformerClassifier.sinusoidal_embedding(4, self.d_model).shape,
            (1, 4, self.d_model),
        )

        classifier_stats = classifier.weights_statistics()
        self.assertIn("class_emb", classifier_stats)
        self.assertIn("positional_emb", classifier_stats)
        self.assertNotIn("attention_pool", classifier_stats)

        with self.assertRaises(ValueError):
            classifier._prepare_tokens(torch.randn(2, 5, self.d_model))
        with self.assertRaises(ValueError):
            classifier.set_growing_layers(0, index=1)

        no_pos_classifier = GrowingTransformerClassifier(
            sequence_length=6,
            seq_pool=True,
            embedding_dim=self.d_model,
            num_layers=1,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            positional_embedding="none",
            device=self.device,
        )
        short_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        padded_tokens, padded_mask = no_pos_classifier._prepare_tokens(
            torch.randn(2, 4, self.d_model),
            short_mask,
        )
        self.assertEqual(padded_tokens.shape, (2, 6, self.d_model))
        self.assertEqual(padded_mask.shape, (2, 6))
        self.assertTrue(torch.equal(padded_mask[:, 4:], torch.zeros(2, 2).bool()))
        no_pos_stats = no_pos_classifier.weights_statistics()
        self.assertIn("attention_pool", no_pos_stats)
        self.assertIn("bias", no_pos_stats["attention_pool"])

        no_pos_classifier.set_growing_layers(0)
        self.assertEqual(no_pos_classifier.layer_to_grow_index, 0)

        variable_length_classifier = GrowingTransformerClassifier(
            sequence_length=None,
            seq_pool=True,
            embedding_dim=self.d_model,
            num_layers=1,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            positional_embedding="none",
            device=self.device,
        )
        variable_tokens = torch.randn(2, 5, self.d_model)
        passthrough_tokens, passthrough_mask = variable_length_classifier._prepare_tokens(
            variable_tokens
        )
        self.assertEqual(passthrough_tokens.shape, variable_tokens.shape)
        self.assertIsNone(passthrough_mask)
        self.assertEqual(
            variable_length_classifier(variable_tokens).shape,
            (2, self.out_features),
        )

        already_long_tokens, already_long_mask = (
            no_pos_classifier._pad_to_sequence_length(
                torch.randn(2, 7, self.d_model),
                torch.ones(2, 7, dtype=torch.bool),
            )
        )
        self.assertEqual(already_long_tokens.shape, (2, 7, self.d_model))
        self.assertEqual(already_long_mask.shape, (2, 7))

    def test_classifier_and_image_model_first_order_improvement_paths(self):
        classifier = GrowingTransformerClassifier(
            sequence_length=4,
            seq_pool=True,
            embedding_dim=self.d_model,
            num_layers=2,
            num_heads=self.num_heads,
            mlp_ratio=0.5,
            num_classes=self.out_features,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=self.device,
        )
        for idx, block in enumerate(classifier.blocks):
            block.parameter_update_decrease = torch.tensor(float(idx + 1))
            block.mlp.second_layer.eigenvalues_extension = None
        self.assertEqual(classifier.first_order_improvement.item(), 2.0)
        classifier.select_update(layer_index=0)
        self.assertEqual(classifier.first_order_improvement.item(), 1.0)

        image_model = GrowingTransformer(
            in_features=self.in_features,
            out_features=self.out_features,
            patch_size=4,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_blocks=2,
            device=self.device,
        )
        for idx, block in enumerate(image_model.classifier.blocks):
            block.parameter_update_decrease = torch.tensor(float(idx + 1))
            block.mlp.second_layer.eigenvalues_extension = None
        self.assertEqual(image_model.first_order_improvement.item(), 2.0)
        image_model.select_update(layer_index=1)
        self.assertEqual(image_model.first_order_improvement.item(), 2.0)

    def test_transformer_and_text_models_legacy_error_and_statistics_paths(self):
        legacy_model = GrowingTransformer(
            in_features=self.in_features,
            out_features=self.out_features,
            patch_size=4,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_blocks=2,
            device=self.device,
        )
        self.assertTrue(legacy_model.legacy_api)
        legacy_stats = legacy_model.weights_statistics()
        self.assertIn("patcher", legacy_stats)
        self.assertIn("classifier_head", legacy_stats)
        self.assertIn("position_embeddings", legacy_stats)
        self.assertIn("cls_token", legacy_stats)

        with self.assertRaises(ValueError):
            GrowingTransformer(
                out_features=self.out_features,
                patch_size=4,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_blocks=2,
                device=self.device,
            )
        with self.assertRaises(ValueError):
            GrowingTransformer(
                in_features=(3, 16),
                out_features=self.out_features,
                patch_size=4,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_blocks=2,
                device=self.device,
            )
        with self.assertRaises(ValueError):
            GrowingTransformer(
                in_features=self.in_features,
                out_features=self.out_features,
                patch_size=4,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_blocks=2,
                pooling="mean",
                device=self.device,
            )
        with self.assertRaises(ValueError):
            GrowingTransformer(
                in_features=self.in_features,
                out_features=self.out_features,
                patch_size=4,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_blocks=2,
                use_cls_token=False,
                device=self.device,
            )
        with self.assertRaises(ValueError):
            GrowingTransformer(
                in_features=self.in_features,
                out_features=self.out_features,
                patch_size=(4, 8),
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_blocks=2,
                device=self.device,
            )

        with self.assertRaises(ValueError):
            legacy_model(
                torch.randn(1, *self.in_features),
                mask=torch.ones(1, 4, dtype=torch.bool),
                attention_mask=torch.ones(1, 4, dtype=torch.bool),
            )

        with self.assertRaises(ValueError):
            GrowingTextViTLite(
                seq_len=7,
                patch_size=2,
                embedding_dim=self.d_model,
                word_embedding_dim=self.d_model,
                num_layers=2,
                num_heads=self.num_heads,
                mlp_ratio=0.5,
                num_classes=self.out_features,
                vocab_size=20,
                device=self.device,
            )

        text_model = GrowingTextViTLite(
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
            device=self.device,
        )
        text_stats = text_model.weights_statistics()
        self.assertIn("embedder", text_stats)
        self.assertIn("tokenizer", text_stats)
        self.assertIn("classifier", text_stats)
        text_model.set_growing_layers(0)
        self.assertEqual(text_model.layer_to_grow_index, 0)

        text_model.tokenizer.conv_layers[0] = nn.Conv2d(
            1,
            self.d_model,
            kernel_size=(2, self.d_model),
            stride=(2, 1),
            padding=(0, 0),
            bias=True,
        )
        text_stats_with_bias = text_model.weights_statistics()
        self.assertIn("bias", text_stats_with_bias["tokenizer"])
        text_model.set_growing_layers(scheduling_method="all")

        for idx, block in enumerate(text_model.classifier.blocks):
            block.parameter_update_decrease = torch.tensor(float(idx + 1))
            block.mlp.second_layer.eigenvalues_extension = None
        text_info = text_model.update_information()
        self.assertIn("classifier", text_info)
        self.assertEqual(text_model.first_order_improvement.item(), 2.0)
        text_model.select_update(layer_index=1)
        self.assertEqual(text_model.first_order_improvement.item(), 2.0)

        with self.assertRaises(ValueError):
            text_model.set_growing_layers(1, index=0)

    def test_non_legacy_model_aliases_and_module_main_entrypoint(self):
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
            positional_embedding="none",
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.0,
            device=self.device,
        )
        self.assertFalse(model.legacy_api)
        self.assertIs(model.blocks, model.classifier.blocks)

        model.set_growing_layers(1)
        self.assertEqual(model.layer_to_grow_index, 1)

        seq_len = model.classifier.sequence_length
        assert seq_len is not None
        attention_mask = torch.ones(1, seq_len, dtype=torch.bool)
        self.assertEqual(
            model(
                torch.randn(1, *self.in_features),
                attention_mask=attention_mask,
            ).shape,
            (1, self.out_features),
        )

        non_legacy_stats = model.weights_statistics()
        self.assertIn("tokenizer", non_legacy_stats)
        self.assertIn("classifier", non_legacy_stats)
        self.assertNotIn("patcher", non_legacy_stats)
        self.assertNotIn("classifier_head", non_legacy_stats)

        with patch("builtins.print"):
            main_globals = runpy.run_module(
                "gromo.containers.growing_vision_transformer",
                run_name="__main__",
            )
        self.assertIn("model", main_globals)
        self.assertIn("x", main_globals)
        self.assertIn("y", main_globals)


if __name__ == "__main__":
    unittest.main()
