import torch

from gromo.containers.growing_block import Conv2dGrowingBlock
from gromo.containers.resnet import init_full_resnet_structure


def test_insert_block_before_and_after_increases_growable_layers() -> None:
    model = init_full_resnet_structure(
        input_shape=(3, 32, 32),
        out_features=10,
        nb_stages=3,
        number_of_blocks_per_stage=1,
        inplanes=16,
        hidden_channels=(0, 0, 0),
        use_preactivation=True,
    )
    initial_blocks = len(model._growable_layers)
    target = model._growable_layers[1]

    model.insert_block(stage_index=1, block_index=1, hidden_channels=0)
    model.insert_block(stage_index=1, block_index=1, hidden_channels=0)

    assert len(model._growable_layers) == initial_blocks + 2
    stage_blocks = [m for m in model.stages[1] if isinstance(m, Conv2dGrowingBlock)]
    assert len(stage_blocks) == 3
    assert all(block.hidden_neurons == 0 for block in stage_blocks)
    assert model.find_block_location(target)[0] == 1


def test_insert_block_forward_shape_unchanged() -> None:
    model = init_full_resnet_structure(
        input_shape=(3, 32, 32),
        out_features=10,
        nb_stages=3,
        number_of_blocks_per_stage=1,
        inplanes=16,
        hidden_channels=(0, 0, 0),
        use_preactivation=True,
    )
    x = torch.randn(2, 3, 32, 32)
    before = model(x)
    model.insert_block(stage_index=0, block_index=0, hidden_channels=0)
    after = model(x)
    assert before.shape == after.shape == (2, 10)
