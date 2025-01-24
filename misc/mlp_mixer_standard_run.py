import argparse
from functools import partial
import torch
import torch.nn as nn
from time import time

from auxilliary_functions import evaluate_model, topk_accuracy, train, LabelSmoothingLoss
from schedulers import get_scheduler
from gromo.growing_mlp_mixer import GrowingMLPMixer
from gromo.utils.datasets import get_dataloaders, known_datasets
from gromo.utils.utils import global_device, set_device

known_optimizers = {
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
}

selection_methods = [
    "none",
    "fo",
]


def create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the command line arguments
    Returns:
        parser: argparse.ArgumentParser
            parser for the command line arguments
    """
    parser = argparse.ArgumentParser(description="MLP training")

    # general arguments
    general_group = parser.add_argument_group("general")
    general_group.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    general_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="number of workers for the dataloader (default: 4)",
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset to use (default: mnist)",
        choices=known_datasets.keys(),
    )
    dataset_group.add_argument(
        "--nb-class", type=int, default=None, help="number of classes (default: None)"
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        default="dataset",
        help="path to the dataset (default: dataset)",
    )
    dataset_group.add_argument(
        "--data-augmentation",
        nargs="+",
        default=None,
        help="data augmentation to use (default: None)",
    )

    # model arguments
    architecture_group = parser.add_argument_group("architecture")
    architecture_group.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="patch size (default: 4)",
    )
    architecture_group.add_argument(
        "--num-blocks",
        type=int,
        default=8,
        help="number of hidden layers (default: 1)",
    )
    architecture_group.add_argument(
        "--num-features",
        type=int,
        default=128,
    )
    architecture_group.add_argument(
        "--hidden-dim-token",
        type=int,
        default=64,
        help="hidden dimension for the token mixer (default: 256)",
    )
    architecture_group.add_argument(
        "--hidden-dim-channel",
        type=int,
        default=512,
        help="hidden dimension for the channel mixer (default: 512)",
    )

    # classical training arguments
    training_group = parser.add_argument_group("training")
    training_group.add_argument(
        "--seed", type=int, default=None, help="random seed (default: 0)"
    )
    training_group.add_argument(
        "--nb-step", type=int, default=10, help="number of cycles (default: 10)"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    training_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=known_optimizers.keys(),
        help="optimizer to use (default: sgd)",
    )
    training_group.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay (default: 0)",
    )
    training_group.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate (default: 0.0)"
    )

    # scheduler arguments
    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="none",
        help="scheduler to use (default: step)",
        choices=['cosine'],
    )
    scheduler_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="number of warmup iterations (default: 0)",
    )

    return parser


def preprocess_and_check_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Preprocess and check the arguments
    Args:
        args: argparse.Namespace
            command line arguments

    Returns:
        args: argparse.Namespace
            processed arguments

    """
    # Check if the seed is None and generate a random seed if necessary
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32 - 1)

    # Dataset arguments
    if args.dataset == "sin":
        raise ValueError("The sin dataset is not supported by MLP Mixer.")

    if args.nb_class is None:
        if args.dataset == "mnist":
            args.nb_class = 10
        elif args.dataset == "cifar10":
            args.nb_class = 10
        elif args.dataset == "cifar100":
            args.nb_class = 100
        else:
            raise ValueError(f"Number of classes not specified for dataset {args.dataset}")

    # Model arguments
    if args.num_blocks < 1:
        raise ValueError("The number of hidden layers must be greater than 0.")

    return args


def display_args(args: argparse.Namespace) -> None:
    """
    Display the arguments
    Args:
        args: argparse.Namespace
            command line arguments
    """
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    print()


def main(args: argparse.Namespace):
    """
    Main function
    Args:
        args: argparse.Namespace
            command line arguments
    """
    start_time = time()

    # set the device
    if args.no_cuda:
        set_device(torch.device("cpu"))
    device: torch.device = global_device()

    # create the dataloaders
    train_dataloader, val_dataloader, in_channels, image_size = get_dataloaders(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        nb_class=args.nb_class,
        split_train_val=0.0,
        data_augmentation=args.data_augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Training dataset size: {len(train_dataloader.dataset)}")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")

    # create the model
    model = GrowingMLPMixer(
        image_size=image_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        num_features=args.num_features,
        hidden_dim_token=args.hidden_dim_token,
        hidden_dim_channel=args.hidden_dim_channel,
        num_layers=args.num_blocks,
        num_classes=args.nb_class,
        dropout=args.dropout,
    )
    print(f"Model before training:\n{model}")
    print(f"Number of parameters: {model.number_of_parameters(): ,}")

    # loss function
    if args.dataset == "sin":
        raise ValueError("The sin dataset is not supported by MLP Mixer.")
    else:
        train_loss_fn = LabelSmoothingLoss(smoothing=0.1, reduction="mean")
        test_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        top_1_accuracy = partial(topk_accuracy, k=1)

    # optimizer
    if args.optimizer == "sgd":
        optim_kwargs = {"lr": args.lr, "momentum": 0.9, "weight_decay": args.weight_decay}
    elif args.optimizer == "adamw":
        optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.99), "weight_decay": args.weight_decay}
    elif args.optimizer == "adam":
        optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.99), "weight_decay": args.weight_decay}
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    optimizer = known_optimizers[args.optimizer](
        model.parameters(), **optim_kwargs
    )
    print(f"Optimizer: {optimizer}")

    # scheduler
    scheduler = get_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        num_epochs=args.nb_step,
        num_batches_per_epoch=len(train_dataloader),
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
    )
    print(f"Scheduler: {scheduler}")

    # Training
    for step in range(args.nb_step):
        step_start_time = time()
        # reset peak memory stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss, train_accuracy = train(
            model=model,
            train_dataloader=train_dataloader,
            loss_function=train_loss_fn,
            aux_loss_function=top_1_accuracy,
            optimizer=optimizer,
            device=device,
            cutmix_beta=1.0,
            cutmix_prob=0.5,
            scheduler=scheduler,
        )

        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=test_loss_fn,
            aux_loss_function=top_1_accuracy,
            dataloader=val_dataloader,
            device=device,
        )

        print(f"Epoch [{step + 1}/{args.nb_step}] -- Loss: {val_loss:.3f} ({train_loss:.3f}) -- Accuracy: {val_accuracy*100:.3f}% ({train_accuracy*100:.3f}%) [lr: {optimizer.param_groups[0]['lr']:.6f}]")
        # display epoch type, maximum memory allocated and maximum memory reserved
        # if device.type == "cuda":
        #     print(
        #         f"Peak memory allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 3): .2f} GB"
        #         f" -- Peak memory reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3): .2f} GB"
        #     )

    print(f"Total duration: {time() - start_time: .2f} seconds")
    print(f"Model after training:\n{model}")
    print(f"Number of parameters: {model.number_of_parameters(): ,}")


if __name__ == "__main__":
    import os
    import random

    # check cuda
    if torch.cuda.is_available():
        # Check if the CUDA_VISIBLE_DEVICES environment variable is set
        print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available")

    parser = create_parser()
    args = parser.parse_args()
    args = preprocess_and_check_args(args)
    display_args(args)
    main(args)
