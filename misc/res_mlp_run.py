import argparse
import mlflow
from auxilliary_functions import *
from gromo.growing_residual_mlp import GrowingResidualMLP
from gromo.utils.datasets import get_dataset
from gromo.utils.utils import global_device, set_device

activation_functions = {
    "relu": torch.nn.ReLU(),
    "selu": torch.nn.SELU(),
    "elu": torch.nn.ELU(),
    "gelu": torch.nn.GELU(),
}

known_optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

selection_methods = ["none", "fo", "scaled_fo", "one_step_fo"]

def create_parser():
    parser = argparse.ArgumentParser(description="MLP training")
    general_group = parser.add_argument_group("general")
    general_group.add_argument("--log-dir", type=str, default="misc/logs", help="directory to save logs")
    general_group.add_argument("--log-dir-suffix", type=str, default=None, help="suffix to add to the log directory")
    general_group.add_argument("--log-file-name", type=str, default=None, help="name of the log file")
    general_group.add_argument("--log-file-prefix", type=str, default=None, help="prefix to add to the log file name")
    general_group.add_argument("--experiment-name", type=str, default=None, help="name of the experiment")
    general_group.add_argument("--tags", type=str, default=None, help="tags to add to the experiment")
    general_group.add_argument("--nb-step", type=int, default=10, help="number of cycles")
    general_group.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    general_group.add_argument("--training-threshold", type=float, default=None, help="training is stopped when the loss is below this threshold")
    general_group.add_argument("--log-system-metrics", action="store_true", default=False, help="log system metrics")
    general_group.add_argument("--num-workers", type=int, default=0, help="number of workers for the dataloader")

    architecture_group = parser.add_argument_group("architecture")
    architecture_group.add_argument("--num-blocks", type=int, default=1, help="number of hidden layers")
    architecture_group.add_argument("--hidden-size", type=int, default=10, help="hidden size")
    architecture_group.add_argument("--activation", type=str, default="selu", help="activation function", choices=activation_functions.keys())
    architecture_group.add_argument("--no-bias", action="store_true", default=False, help="disables bias")

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--dataset", type=str, default="mnist", help="dataset to use", choices=["sin", "mnist", "cifar10"])
    dataset_group.add_argument("--nb-class", type=int, default=10, help="number of classes")
    dataset_group.add_argument("--split-train-val", type=float, default=0.3, help="proportion of the training set to use as validation set")
    dataset_group.add_argument("--dataset-path", type=str, default="dataset", help="path to the dataset")
    dataset_group.add_argument("--data-augmentation", nargs="+", default=None, help="data augmentation to use")

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--seed", type=int, default=None, help="random seed")
    training_group.add_argument("--batch-size", type=int, default=64, help="input batch size for training")
    training_group.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use")
    training_group.add_argument("--lr", type=float, default=0.01, help="learning rate")
    training_group.add_argument("--weight-decay", type=float, default=0, help="weight decay")

    growing_group = parser.add_argument_group("growing")
    growing_group.add_argument("--epochs-per-growth", type=int, default=-1, help="number of epochs before growing the model")
    growing_group.add_argument("--selection-method", type=str, default="none", help="selection method to use", choices=selection_methods)
    growing_group.add_argument("--growing-batch-limit", type=int, default=-1, help="maximum number of batches to use")
    growing_group.add_argument("--growing-part", type=str, default="all", help="part of the model to grow", choices=["all", "parameter", "neuron"])
    growing_group.add_argument("--growing-numerical-threshold", type=float, default=1e-5, help="numerical threshold for growing")
    growing_group.add_argument("--growing-statistical-threshold", type=float, default=1e-3, help="statistical threshold for growing")
    growing_group.add_argument("--growing-maximum-added-neurons", type=int, default=10, help="maximum number of neurons to add")
    growing_group.add_argument("--growing-computation-dtype", type=str, default="float32", help="dtype to use for the computation", choices=["float32", "float64"])
    growing_group.add_argument("--normalize-weights", action="store_true", default=False, help="normalize the weights after growing")
    growing_group.add_argument("--init-new-neurons-with-random-in-and-zero-out", action="store_true", default=False, help="initialize the new neurons with random fan-in weights and zero fan-out weights")

    line_search_group = parser.add_argument_group("line search")
    line_search_group.add_argument("--line-search-alpha", type=float, default=0.1, help="line search alpha")
    line_search_group.add_argument("--line-search-beta", type=float, default=0.5, help="line search beta")
    line_search_group.add_argument("--line-search-max-iter", type=int, default=20, help="line search max iteration")
    line_search_group.add_argument("--line-search-epsilon", type=float, default=1e-7, help="line search epsilon")
    line_search_group.add_argument("--line-search-batch-limit", type=int, default=-1, help="maximum number of batches to use")
    return parser

def preprocess_and_check_args(args):
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32 - 1)
    if args.log_file_name is None:
        args.log_file_name = f"residual_mlp_{args.dataset}_{args.activation}_model_{args.hidden_size}x{args.num_blocks}"
    if args.log_dir_suffix is not None:
        args.log_dir = f"{args.log_dir}/{args.log_dir_suffix}"
    if args.log_file_prefix is not None:
        args.log_file_name = f"{args.log_file_prefix}_{args.log_file_name}"
    if args.experiment_name is None:
        args.experiment_name = f"{args.dataset}"
    if args.normalize_weights and args.activation.lower().strip() != "relu":
        warn("Normalizing the weights is only an invariant for ReLU activation functions.")
    if args.init_new_neurons_with_random_in_and_zero_out:
        args.selection_method = "none"
    return args

def display_args(args):
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"    {key}: {value}")
    print()

def log_layers_metrics(layer_metrics, step, prefix=None):
    for key, value in layer_metrics.items():
        prefix_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            log_layers_metrics(value, step, prefix=prefix_key)
        else:
            try:
                mlflow.log_metric(prefix_key, value, step=step)
            except mlflow.exceptions.MlflowException:
                pass

def main(args):
    start_time = time()
    file_path = f"{args.log_dir}/{args.log_file_name}_{start_time:.0f}.txt"
    print(f"Log file: {file_path}")

    mlflow.set_tracking_uri(f"{args.log_dir}/mlruns")
    print(f"MLflow tracking uri: {mlflow.get_tracking_uri()}")
    try:
        mlflow.create_experiment(name=f"{args.experiment_name}", tags={"tags": args.tags} if args.tags else None)
    except mlflow.exceptions.MlflowException:
        warn(f"Experiment {args.dataset} already exists.")

    mlflow.set_experiment(f"{args.experiment_name}")
    with mlflow.start_run(run_name=args.log_file_name, log_system_metrics=args.log_system_metrics):
        if args.tags:
            mlflow.set_tags({"tags": args.tags})
        with open(file_path, "w") as f:
            f.write(str(args) + "\n")
        mlflow.log_params(vars(args))

        if args.no_cuda:
            set_device(torch.device("cpu"))
        device = global_device()
        print(f'Using device: {device}')

        if args.dataset == "sin":
            input_shape = 1
            train_dataloader = SinDataloader(nb_sample=1_000, batch_size=args.batch_size, seed=args.seed, device=device)
            val_dataloader = train_dataloader
            loss_function = AxisMSELoss(reduction="sum")
            loss_function_mean = AxisMSELoss(reduction="mean")
            top_1_accuracy = None
            args.nb_class = 1
        else:
            train_dataset, _, val_dataset = get_dataset(
                dataset_name=args.dataset,
                dataset_path=args.dataset_path,
                nb_class=args.nb_class,
                split_train_val=0.0,
                data_augmentation=args.data_augmentation,
                seed=args.seed,
            )
            input_shape = train_dataset[0][0].shape[0]
            pin_memory = device != torch.device("cpu")
            num_workers = args.num_workers if pin_memory else 0
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=num_workers > 0
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=num_workers > 0
            )
            del train_dataset, val_dataset
            loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
            loss_function_mean = torch.nn.CrossEntropyLoss(reduction="mean")
            top_1_accuracy = Accuracy(k=1)

        model = GrowingResidualMLP(
            in_features=input_shape,
            hidden_features=args.hidden_size,
            out_features=args.nb_class,
            num_blocks=args.num_blocks,
            activation=activation_functions[args.activation],
        )

        print(f"Model before training:\n{model}")
        print(f"Number of parameters: {model.number_of_parameters(): ,}")

        growing_dtype = torch.float32 if args.growing_computation_dtype == "float32" else torch.float64

        train_loss, train_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function,
            aux_loss_function=top_1_accuracy,
            dataloader=train_dataloader,
            device=device,
        )

        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function,
            aux_loss_function=top_1_accuracy,
            dataloader=val_dataloader,
            device=device,
        )

        print(f"Initialization: loss {val_loss: .4f} ({train_loss: .4f}) -- accuracy {val_accuracy: .4f} ({train_accuracy: .4f})")

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

        with open(file_path, "a") as f:
            f.write(str(logs) + "\n")

        for key, value in logs.items():
            if key in ["device", "device_model"]:
                mlflow.log_param(key, str(value))
            elif isinstance(value, dict):
                log_layers_metrics(value, step=0, prefix=key)
            else:
                try:
                    mlflow.log_metric(key, value, step=0)
                except TypeError:
                    warn(f"Cannot log {key} with value {value}")

        logs = {}
        last_updated_block = -1
        cycle_index = 0
        for step in range(1, args.nb_step + 1):
            step_start_time = time()
            if cycle_index == args.epochs_per_growth:
                logs["epoch_type"] = "growth"
                cycle_index = -1

                initial_train_loss, initial_train_accuracy = compute_statistics(
                    growing_model=model,
                    dataloader=train_dataloader,
                    loss_function=loss_function,
                    aux_loss_function=top_1_accuracy,
                    batch_limit=args.growing_batch_limit,
                    device=device,
                )
                logs["train_loss"] = initial_train_loss
                logs["train_accuracy"] = initial_train_accuracy

                model.compute_optimal_update(
                    part=args.growing_part,
                    numerical_threshold=args.growing_numerical_threshold,
                    statistical_threshold=args.growing_statistical_threshold,
                    maximum_added_neurons=args.growing_maximum_added_neurons,
                    dtype=growing_dtype,
                )
                logs["updates_information"] = model.update_information()

                if args.selection_method == "none":
                    last_updated_block = (last_updated_block + 1) % len(model.blocks)
                    model.select_update(block_index=last_updated_block)
                elif args.selection_method == "fo":
                    last_updated_block = model.select_best_update()
                else:
                    raise NotImplementedError("Growing the model is not implemented yet")

                logs["selected_update"] = last_updated_block
                logs["added_neurons"] = model.currently_updated_block.eigenvalues_extension.size(0) if model.currently_updated_block.eigenvalues_extension else 0

                if not args.init_new_neurons_with_random_in_and_zero_out:
                    gamma, estimated_loss, gamma_history, loss_history = line_search(
                        model=model,
                        dataloader=train_dataloader,
                        loss_function=loss_function,
                        batch_limit=args.line_search_batch_limit,
                        initial_loss=initial_train_loss,
                        first_order_improvement=model.blocks[model.currently_updated_block_index].first_order_improvement,
                        alpha=args.line_search_alpha,
                        beta=args.line_search_beta,
                        max_iter=args.line_search_max_iter,
                        epsilon=args.line_search_epsilon,
                        device=device,
                    )
                else:
                    gamma = 1
                    gamma_history = [1]
                    loss_history = [-1]

                if args.init_new_neurons_with_random_in_and_zero_out and model.currently_updated_block_index - 1 >= 0:
                    model[model.currently_updated_block_index - 1].extended_output_layer.reset_parameters()
                    model[model.currently_updated_block_index].extended_input_layer.weight.data.zero_()

                if args.init_new_neurons_with_random_in_and_zero_out:
                    model[model.currently_updated_block_index].optimal_delta_layer = None

                logs["gamma"] = gamma
                logs["gamma_history"] = gamma_history
                logs["loss_history"] = loss_history
                logs["number_of_line_search_iterations"] = len(gamma_history) - 1

                model.amplitude_factor = gamma**0.5
                model.apply_change()

                train_loss = initial_train_loss
                train_accuracy = initial_train_accuracy

                if args.normalize_weights:
                    model.normalise()
                    logs["layers_statistics_pre_normalization"] = model.weights_statistics()
            else:
                logs["selected_update"] = -1
                logs["epoch_type"] = "training"
                optimizer = known_optimizers[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                train_loss, train_accuracy, _, _ = train(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=None,
                    loss_function=loss_function_mean,
                    aux_loss_function=top_1_accuracy,
                    optimizer=optimizer,
                    nb_epoch=1,
                )
                logs["train_loss"] = train_loss[-1]
                logs["train_accuracy"] = train_accuracy[-1]
                train_loss = train_loss[-1]
                train_accuracy = train_accuracy[-1]

            val_loss, val_accuracy = evaluate_model(
                model=model,
                loss_function=loss_function,
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

            print(f"Epoch [{step}/{args.nb_step}]: loss {val_loss: .4f} ({train_loss: .4f}) -- accuracy {val_accuracy: .4f} ({train_accuracy: .4f})  [{logs['epoch_type']}]")

            with open(file_path, "a") as f:
                f.write(str(logs) + "\n")

            for key, value in logs.items():
                if key == "epoch_type":
                    continue
                elif isinstance(value, dict):
                    log_layers_metrics(value, step=step, prefix=key)
                else:
                    try:
                        mlflow.log_metric(key, value, step=step)
                    except TypeError:
                        pass

            logs = {}
            cycle_index += 1

            if args.training_threshold is not None and train_loss < args.training_threshold:
                print(f"Training threshold reached at step {step}")
                break

    print(f"Total duration: {time() - start_time: .2f} seconds")
    print(f"Model after training:\n{model}")
    print(f"Number of parameters: {model.number_of_parameters(): ,}")

if __name__ == "__main__":
    import os
    import random

    if torch.cuda.is_available():
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