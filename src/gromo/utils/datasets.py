from math import ceil, floor
from warnings import warn

import torch
from torch.utils import data
from torchvision import datasets, transforms

import misc.auxilliary_functions

known_datasets = {
    "sin": misc.auxilliary_functions.SinDataset,
    "mnist": datasets.MNIST,
    # "fashion-mnist": datasets.FashionMNIST, # not used, but can be added with the right augmentation
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    # "svhn": datasets.SVHN, # not used, but can be added with the right augmentation
}


def get_dataset(
    dataset_name: str,
    dataset_path: str,
    nb_class: int | None = None,
    split_train_val: float = 0.1,
    data_augmentation: list[str] | None = None,
) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
    """
    Get the dataset

    Parameters
    ----------
    dataset_name: str
        The name of the dataset
    dataset_path: str
        The path to the dataset or where to download it
    nb_class: int | None
        The number of classes to keep in the dataset, for example, if nb_class=5, only the first 5 classes will be kept
    split_train_val: float
        The proportion of the training set to use as validation set
    data_augmentation: list[str] | None
        The data augmentation to apply to the training set
    seed: int | None
        The seed to use for the random number generator
    kwargs: dict
        Additional arguments

    Returns
    -------
    tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]
        The training, validation, and test datasets
    """
    # check if the dataset is known
    if dataset_name not in known_datasets:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # sin dataset
    if dataset_name == "sin":
        train_data = known_datasets[dataset_name]()
        val_data = known_datasets[dataset_name]()
        test_data = known_datasets[dataset_name]()
        return train_data, val_data, test_data

    # other datasets
    datasets_transforms = {
        "mnist": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,), std=(0.3081,)
            ),
        ],
        "cifar10": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ],
        "cifar100": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ],

    }
    augmentation_transforms = []
    if data_augmentation:
        for aug in data_augmentation:
            if aug == "horizontal_flip":
                augmentation_transforms.append(transforms.RandomHorizontalFlip())
            elif aug == "rotation":
                augmentation_transforms.append(transforms.RandomRotation(10))
            elif aug == "crop":
                augmentation_transforms.append(
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect")
                )
            else:
                warn(f"Unknown augmentation {aug}")
            # Add more augmentations as needed

    dataset = known_datasets[dataset_name]
    datasets_transform = datasets_transforms[dataset_name]

    train_val_data = dataset(
        root=dataset_path,
        train=True,
        download=True,
        transform=None,
    )

    test_data = dataset(
        root=dataset_path,
        train=False,
        download=True,
        transform=transforms.Compose(datasets_transform),
    )

    # Keep only the first nb_class classes
    if nb_class is not None and nb_class <= 1:
        warn(f"{nb_class=} should be greater than 1")
    if nb_class is not None:
        targets = train_val_data.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        train_idx = targets <= (nb_class - 1)
        train_val_data.data = train_val_data.data[train_idx]
        train_val_data.targets = targets[train_idx]

        targets = test_data.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        test_idx = targets <= (nb_class - 1)
        test_data.data = test_data.data[test_idx]
        test_data.targets = targets[test_idx]

    # Split the training set into training and validation sets
    if split_train_val < 0 or split_train_val > 1:
        raise ValueError(f"{split_train_val=} should be in [0, 1]")
    if split_train_val > 0.5:
        warn(
            f"{split_train_val=} is greater than 0.5 this means that "
            f"the validation set is greater than the training set"
        )

    train_size = ceil(len(train_val_data) * (1 - split_train_val))
    val_size = floor(len(train_val_data) * split_train_val)
    assert train_size + val_size == len(train_val_data)

    train_data, val_data = torch.utils.data.random_split(
        train_val_data, [train_size, val_size]
    )
    train_data.dataset.transform = transforms.Compose(
        datasets_transform + augmentation_transforms
    )
    val_data.dataset.transform = transforms.Compose(datasets_transform)

    return train_data, val_data, test_data
