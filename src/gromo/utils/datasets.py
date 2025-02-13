from math import ceil, floor
from warnings import warn

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

from gromo.utils.utils import global_device


class SinDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.nb_sample = 1_000

    def __len__(self):
        return self.nb_sample

    def __getitem__(self, _):
        data = torch.rand(1) * 2 * torch.pi
        target = torch.sin(data)
        return data, target


known_datasets = {
    "sin": SinDataset,
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
}


def get_dataloaders(
    dataset_name: str = "cifar10",
    dataset_path: str = "dataset",
    nb_class: int | None = None,
    split_train_val: float = 0.0,
    data_augmentation: list[str] | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = global_device(),
    shuffle: bool = True,
):
    # load the dataset and create the dataloaders
    train_dataset, val_dataset, test_dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        nb_class=nb_class,
        split_train_val=split_train_val,
        data_augmentation=data_augmentation,
    )
    # print(f"Input shape: {train_dataset[0][0].shape}")
    data_shape = train_dataset[0][0].shape

    pin_memory = device != torch.device("cpu")
    num_workers = num_workers if pin_memory else 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    del train_dataset, val_dataset, test_dataset

    return train_dataloader, val_dataloader, test_loader, data_shape


def get_dataset(
    dataset_name: str,
    dataset_path: str,
    nb_class: int | None = None,
    split_train_val: float = 0.0,
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

    # sin dataset (special case)
    if dataset_name == "sin":
        train_data = known_datasets[dataset_name]()
        val_data = known_datasets[dataset_name]()
        test_data = known_datasets[dataset_name]()
        return train_data, val_data, test_data

    # get the dataset
    dataset = known_datasets[dataset_name]
    datasets_transforms, augmentation_transforms = get_transforms(
        dataset_name, data_augmentation
    )

    # load the train and test datasets
    train_test_args = {
        "root": dataset_path,
        "download": True,
        "transform": transforms.Compose(datasets_transforms),
    }
    train_split_args = {"train": True} if dataset_name != "svhn" else {"split": "train"}
    test_split_args = {"train": False} if dataset_name != "svhn" else {"split": "test"}

    train_val_data = dataset(**train_test_args, **train_split_args)
    test_data = dataset(**train_test_args, **test_split_args)

    # filter the classes
    train_val_data = filter_classes(
        train_val_data, nb_class, "labels" if dataset_name == "svhn" else "targets"
    )
    test_data = filter_classes(
        test_data, nb_class, "labels" if dataset_name == "svhn" else "targets"
    )

    # split the training set
    train_data, val_data = split_train_val_data(train_val_data, split_train_val)
    train_data.dataset.transform = transforms.Compose(
        augmentation_transforms + datasets_transforms
    )
    val_data.dataset.transform = transforms.Compose(datasets_transforms)

    return train_data, val_data, test_data


def filter_classes(dataset, nb_class, field_name):
    if nb_class is not None and nb_class <= 1:
        warn(f"{nb_class=} should be greater than 1")
    if nb_class is not None:
        targets = getattr(dataset, field_name)
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        elif isinstance(targets, np.ndarray):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        idx = targets <= (nb_class - 1)
        dataset.data = dataset.data[idx]
        setattr(dataset, field_name, targets[idx])
    return dataset


def split_train_val_data(train_val_data, split_train_val):
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
    return train_data, val_data


def get_transforms(
    dataset_name: str, data_augmentation: list[str] | None = None
) -> tuple[list, list]:
    datasets_transforms = {
        "mnist": [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ],
        "fashion-mnist": [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
        ],
        "cifar10": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ],
        "cifar100": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ],
        "svhn": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)
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
            elif aug == "autoaugment":
                if dataset_name in ["cifar10", "cifar100"]:
                    policy = transforms.AutoAugmentPolicy.CIFAR10
                elif dataset_name == "svhn":
                    policy = transforms.AutoAugmentPolicy.SVHN
                else:
                    raise ValueError(f"AutoAugment not available for {dataset_name}")
                augmentation_transforms.append(transforms.AutoAugment(policy=policy))
            elif aug == "randaugment":
                augmentation_transforms.append(transforms.RandAugment())
            else:
                warn(f"Unknown augmentation {aug}")
            # Add more augmentations as needed
    return datasets_transforms[dataset_name], augmentation_transforms
