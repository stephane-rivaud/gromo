import argparse
import sys
import logging
from functools import partial
import mlflow
import torch
import torch.nn as nn
from time import time
from warnings import warn

from auxilliary_functions import evaluate_model, compute_statistics, line_search, topk_accuracy, train, LabelSmoothingLoss
from schedulers import get_scheduler, known_schedulers
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
        "--split-train-val",
        type=float,
        default=0.0,
        help="proportion of the training set to use as validation set (default: 0.3)",
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
        default=1,
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
    architecture_group.add_argument(
        "--no-bias", action="store_true", default=False, help="disables bias"
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
    training_group.add_argument(
        "--training-threshold",
        type=float,
        default=None,
        help="training is stopped when the loss is below this threshold (default: None)",
    )

    # scheduler arguments
    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="none",
        help="scheduler to use (default: step)",
        choices=known_schedulers.keys(),
    )
    scheduler_group.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="number of warmup iterations (default: 0)",
    )

    # growing training arguments
    growing_group = parser.add_argument_group("growing")
    growing_group.add_argument(
        "--epochs-per-growth",
        type=int,
        default=-1,
        help="number of epochs before growing the model (default: -1)",
    )
    growing_group.add_argument(
        "--selection-method",
        type=str,
        default="none",
        help="selection method to use (default: none)",
        choices=selection_methods,
    )
    growing_group.add_argument(
        "--growing-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )
    growing_group.add_argument(
        "--growing-part",
        type=str,
        default="all",
        help="part of the model to grow (default: all)",
        choices=["all", "parameter", "neuron"],
    )
    growing_group.add_argument(
        "--growing-numerical-threshold",
        type=float,
        default=1e-5,
        help="numerical threshold for growing (default: 1e-5)",
    )
    growing_group.add_argument(
        "--growing-statistical-threshold",
        type=float,
        default=1e-3,
        help="statistical threshold for growing (default: 1e-3)",
    )
    growing_group.add_argument(
        "--growing-maximum-added-neurons",
        type=int,
        default=10,
        help="maximum number of neurons to add (default: None)",
    )
    growing_group.add_argument(
        "--growing-computation-dtype",
        type=str,
        default="float32",
        help="dtype to use for the computation (default: float32)",
        choices=["float32", "float64"],
    )
    growing_group.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help="normalize the weights after growing (default: False)",
    )

    # line search arguments
    line_search_group = parser.add_argument_group("line search")
    line_search_group.add_argument(
        "--line-search-alpha",
        type=float,
        default=0.1,
        help="line search alpha (default: 0.1)",
    )
    line_search_group.add_argument(
        "--line-search-beta",
        type=float,
        default=0.5,
        help="line search beta (default: 0.5)",
    )
    line_search_group.add_argument(
        "--line-search-max-iter",
        type=int,
        default=20,
        help="line search max iteration (default: 100)",
    )
    line_search_group.add_argument(
        "--line-search-epsilon",
        type=float,
        default=1e-7,
        help="line search epsilon (default: 1e-7)",
    )
    line_search_group.add_argument(
        "--line-search-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )

    # logging arguments
    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument(
        "--log-dir",
        type=str,
        default="misc/logs",
        help="directory to save logs (default: logs)",
    )
    logging_group.add_argument(
        "--log-file-name",
        type=str,
        default=None,
        help="name of the log file (default: None)",
    )
    logging_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="name of the experiment (default: None)",
    )
    logging_group.add_argument(
        "--tags",
        type=str,
        default=None,
        help="tags to add to the experiment (default: None)",
    )
    logging_group.add_argument(
        "--log-system-metrics",
        action="store_true",
        default=False,
        help="log system metrics (default: False)",
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

    # Logging arguments
    if args.log_file_name is None:
        args.log_file_name = f"mlp_mixer_{args.dataset}_{args.num_blocks}x{args.num_features}x{args.hidden_dim_token}x{args.hidden_dim_channel}"

    if args.experiment_name is None:
        args.experiment_name = f"{args.dataset}"

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

    # Growing arguments
    if args.normalize_weights and args.activation.lower().strip() != "relu":
        warn("Normalizing the weights is only an invariant for ReLU activation functions.")

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


def log_layers_metrics(layer_metrics: dict, step: int, prefix: str | None = None) -> None:
    """
    Log the metrics of the layers

    Parameters
    ----------
    layer_metrics : dict
        metrics of the layers
    step : int
        step number
    prefix : str, optional
        prefix to add to the keys (default: "")
    """
    for key, value in layer_metrics.items():
        prefix_key = f"{prefix}_{key}" if prefix is not None else key
        if isinstance(value, dict):
            log_layers_metrics(value, step, prefix=prefix_key)
        else:
            try:
                mlflow.log_metric(prefix_key, value, step=step)
            except mlflow.exceptions.MlflowException as e:
                print(f"Cannot log {prefix_key} with value {value} ({e})")


def log_metrics(metrics: dict, step: int) -> None:
    """
    Log the metrics
    Args:
        metrics: dict
            metrics to log
        step: int
            step number
    """
    for key, value in metrics.items():
        if isinstance(value, dict):
            log_layers_metrics(value, step, prefix=key)
        elif key == "epoch_type":
            continue
        elif key == "device" or key == "device_model":
            mlflow.log_param(key, str(value))
        else:
            try:
                mlflow.log_metric(key, value, step=step)
            except mlflow.exceptions.MlflowException as e:
                print(f"Cannot log {key} with value {value} ({e})")


def get_logger(log_dir: str, log_file_name: str) -> logging.Logger:
    """
    Get the logger
    Args:
        log_dir: str
            directory to save the logs
        log_file_name: str
            name of the log file

    Returns:
        logger: logging.Logger
            logger
    """
    # configure the logger
    logging.basicConfig(
        filename=f"{log_dir}/{log_file_name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger


def setup_mlflow(log_dir: str, experiment_name: str, tags: str | None, logger: logging.Logger) -> None:
    """
    Setup MLflow
    Args:
        log_dir: str
            directory to save the logs
        experiment_name: str
            name of the experiment
        tags: str, optional
            tags to add to the experiment
        logger: logging.Logger
            logger
    """
    mlflow.set_tracking_uri(f"{log_dir}/mlruns")
    logger.info(f"MLflow tracking uri: {mlflow.get_tracking_uri()}")
    try:
        mlflow.create_experiment(
            name=f"{experiment_name}",
            tags={"tags": tags} if tags is not None else None,
        )
    except mlflow.exceptions.MlflowException:
        logger.warning(f"Experiment {experiment_name} already exists.")

    mlflow.set_experiment(f"{experiment_name}")


def main(args: argparse.Namespace):
    """
    Main function
    Args:
        args: argparse.Namespace
            command line arguments
    """
    start_time = time()

    # get the logger
    logger = get_logger(args.log_dir, args.log_file_name)

    # configure MLflow experiment
    setup_mlflow(args.log_dir, args.experiment_name, args.tags, logger)

    # start the MLflow run
    with mlflow.start_run(run_name=args.log_file_name, log_system_metrics=args.log_system_metrics):
        # log the arguments
        if args.tags is not None:
            mlflow.set_tags({"tags": args.tags})
        logger.info(str(args))
        mlflow.log_params(vars(args))

        # set the device
        if args.no_cuda:
            set_device(torch.device("cpu"))
        device: torch.device = global_device()
        logger.info(f'Using device: {device}')

        # create the dataloaders
        train_dataloader, val_dataloader, in_channels, image_size = get_dataloaders(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            nb_class=args.nb_class,
            split_train_val=args.split_train_val,
            data_augmentation=args.data_augmentation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

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
        logger.info(f"Model before training:\n{model}")
        logger.info(f"Number of parameters: {model.number_of_parameters(): ,}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad): ,}")


        # loss function
        if args.dataset == "sin":
            loss_function = nn.MSELoss(reduction="sum")
            loss_function_mean = nn.MSELoss(reduction="mean")
            top_1_accuracy = None
        else:
            loss_function = nn.CrossEntropyLoss(reduction="sum")
            loss_function_mean = LabelSmoothingLoss(smoothing=0.1, reduction="mean")
            top_1_accuracy = partial(topk_accuracy, k=1)

        # optimizer
        if args.optimizer == "sgd":
            optim_kwargs = {"lr": args.lr, "momentum": 0.9, "weight_decay": args.weight_decay}
        elif args.optimizer == "adamw":
            optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.99), "weight_decay": args.weight_decay}
        elif args.optimizer == "adam":
            optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.99), "weight_decay": args.weight_decay}
        optimizer = known_optimizers[args.optimizer](
            model.parameters(), **optim_kwargs
        )

        # scheduler
        scheduler = get_scheduler(
            scheduler=args.scheduler,
            nb_step=args.nb_step,
            lr=args.lr,
            warmup_iters=args.warmup_iters,
        )

        # set the dtype for growing computations
        growing_dtype = torch.float32
        if args.growing_computation_dtype == "float64":
            growing_dtype = torch.float64
        elif args.growing_computation_dtype != "float32":
            raise ValueError(f"Unknown growing dtype: {args.growing_computation_dtype}")

        # Initial train and validation scores
        train_loss, train_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function_mean,
            aux_loss_function=top_1_accuracy,
            dataloader=train_dataloader,
            device=device,
        )

        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function_mean,
            aux_loss_function=top_1_accuracy,
            dataloader=val_dataloader,
            device=device,
        )

        logger.info(
            f"Initialization: loss {val_loss: .4f} ({train_loss: .4f})"
            f" -- accuracy {val_accuracy*100: 2.2f}% ({train_accuracy*100: 2.2f}%)"
        )

        logs = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "number_of_parameters": model.number_of_parameters(),
            "layers_statistics": model.weights_statistics(),
            "device": device,
        }
        if device.type == "cuda":
            logs["device_model"] = torch.cuda.get_device_name(device)
        log_metrics(logs, step=0)

        # Training
        cycle_index = 0
        for step in range(1, args.nb_step + 1):
            step_start_time = time()
            # reset peak memory stats
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            logs = dict()
            if cycle_index == args.epochs_per_growth:
                # grow the model
                cycle_index = -1

                # compute the growth statistics
                train_loss, train_accuracy = compute_statistics(
                    growing_model=model,
                    dataloader=train_dataloader,
                    loss_function=loss_function,
                    aux_loss_function=top_1_accuracy,
                    batch_limit=args.growing_batch_limit,
                    device=device,
                )

                # compute the optimal updates
                model.compute_optimal_update(
                    part=args.growing_part,
                    numerical_threshold=args.growing_numerical_threshold,
                    statistical_threshold=args.growing_statistical_threshold,
                    maximum_added_neurons=args.growing_maximum_added_neurons,
                    dtype=growing_dtype,
                )

                # select the update to be applied
                if args.selection_method == "none":
                    raise NotImplementedError(f"Selection method '{args.selection_method}' not implemented.")
                elif args.selection_method == "fo":
                    model.select_best_update()
                else:
                    raise NotImplementedError(f"Unknown selection method: {args.selection_method}")

                # line search to find the optimal amplitude factor
                gamma, estimated_loss, gamma_history, loss_history = line_search(
                    model=model,
                    dataloader=train_dataloader,
                    loss_function=loss_function,
                    batch_limit=args.line_search_batch_limit,
                    initial_loss=train_loss,
                    first_order_improvement=model.currently_updated_block.first_order_improvement,
                    alpha=args.line_search_alpha,
                    beta=args.line_search_beta,
                    max_iter=args.line_search_max_iter,
                    epsilon=args.line_search_epsilon,
                    device=device,
                )

                logs["epoch_type"] = "growth"
                logs["train_loss"] = train_loss
                logs["train_accuracy"] = train_accuracy
                logs["updates_information"] = model.update_information()
                if model.currently_updated_block.eigenvalues_extension is not None:
                    logs["added_neurons"] = model.currently_updated_block.eigenvalues_extension.size(0)
                else:
                    logs["added_neurons"] = 0
                logs["gamma"] = gamma
                logs["gamma_history"] = gamma_history
                logs["loss_history"] = loss_history
                logs["number_of_line_search_iterations"] = len(gamma_history) - 1

                model.currently_updated_block.scaling_factor = gamma ** 0.5
                model.apply_change()

                # reset the optimizer after growing
                optimizer = known_optimizers[args.optimizer](model.parameters(), **optim_kwargs)

                if args.normalize_weights:
                    model.normalise()
                    logs["layers_statistics_pre_normalization"] = model.weights_statistics()

            else:
                # set the learning rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = scheduler(step - 1)

                train_loss, train_accuracy = train(
                    model=model,
                    train_dataloader=train_dataloader,
                    loss_function=loss_function_mean,
                    aux_loss_function=top_1_accuracy,
                    optimizer=optimizer,
                    device=device,
                    cutmix_beta=1.0,
                    cutmix_prob=0.5,
                )

                logs["selected_update"] = -1
                logs["epoch_type"] = "training"
                logs["train_loss"] = train_loss
                logs["train_accuracy"] = train_accuracy

            val_loss, val_accuracy = evaluate_model(
                model=model,
                loss_function=loss_function_mean,
                aux_loss_function=top_1_accuracy,
                dataloader=val_dataloader,
                device=device,
            )

            logs["val_loss"] = val_loss
            logs["val_accuracy"] = val_accuracy
            logs["step_duration"] = time() - step_start_time
            logs["layers_statistics"] = model.weights_statistics()
            logs["number_of_parameters"] = model.number_of_parameters()
            if device.type == "cuda":
                logs["GPU utilization"] = torch.cuda.utilization(device)

            logger.info(
                f"Epoch [{step}/{args.nb_step}]: loss {val_loss: .4f} ({train_loss: .4f}) -- accuracy {val_accuracy*100: 2.2f}% ({train_accuracy*100: 2.2f}%)"
            )
            # display epoch type, maximum memory allocated and maximum memory reserved
            if device.type == "cuda":
                logger.info(
                    f"Epoch type: {logs['epoch_type']} -- Maximum memory allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 3): .2f} GB -- Maximum memory reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3): .2f} GB"
                )

            log_metrics(logs, step=step)
            cycle_index += 1

            if (
                    args.training_threshold is not None
                    and train_loss < args.training_threshold
            ):
                logger.info(f"Training threshold reached at step {step}")
                break

    logger.info(f"Total duration: {time() - start_time: .2f} seconds")
    logger.info(f"Model after training:\n{model}")
    logger.info(f"Number of parameters: {model.number_of_parameters(): ,}")


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
