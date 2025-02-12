from math import ceil, floor
from warnings import warn

import torch
from torch.utils import data
from torchvision import datasets, transforms

from gromo.utils.utils import global_device

known_datasets = {
    # "sin": misc.auxilliary_functions.SinDataset,
    "mnist": datasets.MNIST,
    # "fashion-mnist": datasets.FashionMNIST, # not used, but can be added with the right augmentation
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    # "svhn": datasets.SVHN, # not used, but can be added with the right augmentation
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
    print(f"Input shape: {train_dataset[0][0].shape}")
    in_channels, image_size, _ = train_dataset[0][0].shape

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
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    del train_dataset, val_dataset, test_dataset

    return train_dataloader, val_dataloader, in_channels, image_size


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
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
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
            elif aug == "autoaugment":
                if dataset_name == "cifar10":
                    policy = transforms.AutoAugmentPolicy.CIFAR10
                elif dataset_name == "cifar100":
                    policy = transforms.AutoAugmentPolicy.CIFAR100
                else:
                    raise ValueError(f"AutoAugment not available for {dataset_name}")
                augmentation_transforms.append(
                    transforms.AutoAugment(policy=policy)
                )
            elif aug == "randaugment":
                augmentation_transforms.append(transforms.RandAugment())
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
        augmentation_transforms + datasets_transform
    )
    # val_data.dataset.transform = transforms.Compose(datasets_transform)

    return train_data, val_data, test_data


class SinDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        self.nb_sample = 1_000

    def __len__(self):
        return self.nb_sample

    def __getitem__(self, _):
        data = torch.rand(1, 1) * 2 * torch.pi
        target = torch.sin(data)
        return data, target