import math
import torch
import matplotlib.pyplot as plt
from misc.auxilliary_functions import compute_statistics


def gather_statistics_and_update(model, dataloader, loss_fn, batch_limit, maximum_added_neurons, part):
    # Gathering growing statistics
    stat_loss, stat_acc = compute_statistics(model, dataloader, loss_fn, batch_limit=batch_limit)
    print(f'Training Loss after gathering statistics: {stat_loss: .6f}')

    # Compute the optimal update
    growing_dtype = torch.get_default_dtype()
    model.compute_optimal_update(maximum_added_neurons=maximum_added_neurons, dtype=growing_dtype, part=part)

    # Select the best update
    model.select_best_update()

    return stat_loss, stat_acc


def get_functional_gradient(model):
    assert model.currently_updated_block._pre_activity_original is None, "Model has not been reset from an extended forward."
    return model.currently_updated_block.pre_activity.grad.clone()


def get_full_activity(model):
    return model.currently_updated_block.pre_activity.clone()


def get_original_activity(model):
    assert model.currently_updated_block._pre_activity_original is not None, "Model has not completed extended forward."
    return model.currently_updated_block._pre_activity_original.clone()


def get_param_step_activity(model):
    if hasattr(model.currently_updated_block, '_param_step_pre_activity') and getattr(model.currently_updated_block,
                                                                                      '_param_step_pre_activity') is not None:
        return model.currently_updated_block._param_step_pre_activity.clone()
    else:
        return torch.zeros_like(model.currently_updated_block.pre_activity)


def get_neuron_activity(model):
    if hasattr(model.currently_updated_block, '_neuron_step_pre_activity') and getattr(model.currently_updated_block,
                                                                                       '_neuron_step_pre_activity') is not None:
        return model.currently_updated_block._neuron_step_pre_activity.clone()
    else:
        return torch.zeros_like(model.currently_updated_block.pre_activity)


def get_desired_update(model, functional_gradient=None):
    param_step_activity = get_param_step_activity(model)
    return -functional_gradient - param_step_activity


def compare_activation_patches(a1, a2):
    rmse = (a1 - a2).norm(dim=-1).mean()
    results = {
        'norm ratio': a1.norm(dim=-1).mean() / a2.norm(dim=-1).mean(),
        'cosine similarity': torch.nn.functional.cosine_similarity(a1, a2, dim=-1).mean(),
        'rmse': rmse,
        'relative rmse': rmse / a2.norm(dim=-1).mean()
    }
    return results


def compare_activities(function_gradient, desired_update, param_step_activity, neuron_step_activity):
    full_activity = param_step_activity + neuron_step_activity
    results = {
        'norm': {
            'param': param_step_activity.norm(dim=-1).mean(),
            'neuron': neuron_step_activity.norm(dim=-1).mean(),
            'fg': function_gradient.norm(dim=-1).mean(),
            'du': desired_update.norm(dim=-1).mean(),
            'full': full_activity.norm(dim=-1).mean()
        },
        'param_fg': compare_activation_patches(param_step_activity, -function_gradient),
        'neuron_du': compare_activation_patches(neuron_step_activity, desired_update),
        'neuron_fg': compare_activation_patches(neuron_step_activity, -function_gradient),
        'param_du': compare_activation_patches(param_step_activity, desired_update),
        'full_fg': compare_activation_patches(full_activity, -function_gradient),
        'param_neuron': compare_activation_patches(param_step_activity, neuron_step_activity),
        'fg_du': compare_activation_patches(-function_gradient, desired_update)
    }
    return results


def create_running_results():
    results = {
        'norm': {key: [] for key in ['param', 'neuron', 'fg', 'du', 'full']},
        **{key: {metric: [] for metric in ['norm ratio', 'cosine similarity', 'rmse', 'relative rmse']}
           for key in ['param_fg', 'neuron_du', 'neuron_fg', 'param_du', 'full_fg', 'param_neuron', 'fg_du']}
    }
    return results


def update_results(running_results, results):
    for key in running_results.keys():
        if key == 'norm':
            for sub_key in running_results[key].keys():
                running_results[key][sub_key].append(results[key][sub_key])
        else:
            for metric in running_results[key].keys():
                running_results[key][metric].append(results[key][metric])
    return running_results


def init_model(model):
    model.currently_updated_block.store_input = True
    model.currently_updated_block.store_pre_activity = True
    model.currently_updated_block._input = None
    model.currently_updated_block._pre_activity_original = None
    model.currently_updated_block._pre_activity_with_param_step = None
    model.currently_updated_block._pre_activity = None


def reset_model(model):
    model.currently_updated_block.store_input = False
    model.currently_updated_block.store_pre_activity = False
    model.currently_updated_block._input = None
    model.currently_updated_block._pre_activity_original = None
    model.currently_updated_block._pre_activity_with_param_step = None
    model.currently_updated_block._pre_activity = None


def summarize_activity_comparison(model, x, y, loss_fn):
    # Set up the model
    init_model(model)

    # Without extension
    model.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()

    # Get the activities and the functional gradient
    original_activity_ref = get_full_activity(model)
    functional_gradient = get_functional_gradient(model)

    # Reset the model
    reset_model(model)
    init_model(model)

    # With extension
    model.zero_grad()
    y_ext_pred = model.extended_forward(x)
    ext_loss = loss_fn(y_ext_pred, y)
    ext_loss.backward()

    # Check the pre-activity
    original_activity = get_original_activity(model)
    assert torch.allclose(original_activity_ref, original_activity), "Pre-activity mismatch"

    # Get the activities
    param_step_activity = get_param_step_activity(model)
    neuron_activity = get_neuron_activity(model)
    desired_update = get_desired_update(model, functional_gradient)

    # Compare the activities
    results = compare_activities(functional_gradient, desired_update, param_step_activity, neuron_activity)

    # Reset the model
    reset_model(model)

    return results


def compare_activities_and_gradients(model, dataloader, loss_fn, batch_limit):
    running_results = create_running_results()
    for i, (x, y) in enumerate(dataloader):
        if i == batch_limit:
            break

        results = summarize_activity_comparison(model, x, y, loss_fn)
        running_results = update_results(running_results, results)
    return running_results


def display_results(running_results):
    # convert running results to tensors
    for key in running_results.keys():
        if key == 'norm':
            for sub_key in running_results[key].keys():
                running_results[key][sub_key] = torch.tensor(running_results[key][sub_key])
        else:
            for metric in running_results[key].keys():
                running_results[key][metric] = torch.tensor(running_results[key][metric])

    # Display the results
    for key in running_results.keys():
        if key == 'norm':
            for sub_key in running_results[key].keys():
                print(f"Norm -- {sub_key}: {running_results[key][sub_key].mean(): .6f}")
        else:
            for metric in running_results[key].keys():
                print(f"{key} -- {metric}: {running_results[key][metric].mean(): .6f}")
    print()

    # Plot histograms for each of the metrics
    for key in running_results.keys():
        sub_keys = list(running_results[key].keys())
        num_sub_keys = len(sub_keys)
        num_cols = min(5, num_sub_keys)
        num_rows = math.ceil(num_sub_keys / num_cols)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 10))
        fig.suptitle(f'{key} metrics', fontsize=16)
        for i, sub_key in enumerate(sub_keys):
            ax = axs[i // num_cols, i % num_cols] if num_rows > 1 else axs[i]
            ax.hist(running_results[key][sub_key].numpy(), bins=30)
            ax.set_title(f'{sub_key}')
            ax.grid(True)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()


if __name__ == "__main__":
    import os
    import time
    from torch import nn

    from gromo.utils.datasets import get_dataloaders
    from gromo.growing_mlp_mixer import GrowingMLPMixer
    from misc.schedulers import get_scheduler
    from misc.auxilliary_functions import train, evaluate_model, topk_accuracy
    from gromo.utils.utils import set_device, global_device

    # Define the device
    # set_device('mps')
    device = global_device()

    # Fix the random seed
    torch.manual_seed(0)

    # Set the default data type
    if device.type != 'mps':
        torch.set_default_dtype(torch.float64)

    # Define the dataset
    dataset_name = 'mnist'
    dataset_path = 'dataset'
    num_classes = 100 if dataset_name == 'cifar100' else 10
    split_train_val = 0.0
    data_augmentation = None
    batch_size = 128
    num_workers = 2
    train_loader, val_loader, in_channels, image_size = get_dataloaders(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        nb_class=num_classes,
        split_train_val=split_train_val,
        data_augmentation=data_augmentation,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
    )
    print(f"Number of batches: {len(train_loader)}")

    # Define the model
    patch_size = 4
    num_layers = 1
    num_features = 32 if dataset_name == 'mnist' else 64
    hidden_dim_token = 1
    hidden_dim_channel = 1
    dropout = 0.0
    model = GrowingMLPMixer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_features=num_features,
        hidden_dim_token=hidden_dim_token,
        hidden_dim_channel=hidden_dim_channel,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.to(device)
    print(model)

    # Define the losses
    loss_fn_train = nn.CrossEntropyLoss(reduction='mean')
    loss_fn_growth = nn.CrossEntropyLoss(reduction='sum')

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=5e-5)

    # define the scheduler
    num_epochs = 40
    scheduler = get_scheduler(
        scheduler_name='cosine',
        optimizer=optimizer,
        base_lr=0.001,
        warmup_epochs=5,
        num_epochs=300,
        num_batches_per_epoch=len(train_loader),
    )

    # Define the path to save the model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)


    def save_training_state(dataset_name, model, optimizer, epoch, training_loss):
        checkpoint_path = os.path.join(model_dir, f'mlp_mixer-{dataset_name}-epoch_{epoch}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'training_loss': training_loss
        }, checkpoint_path)


    def retrieve_checkpoint(dataset, epoch):
        checkpoint_path = None
        for epoch_test in range(epoch, -1, -1):
            checkpoint_test = os.path.join(model_dir, f'mlp_mixer-{dataset}-epoch_{epoch_test}.pth')
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test
                break
        if checkpoint_path is None:
            checkpoint_test = os.path.join(model_dir, f'mlp_mixer-{dataset_name}.pth')
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test
        if checkpoint_path is None:
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint['epoch'] > epoch:
            return None
        else:
            return checkpoint


    # Check if the checkpoint exists
    checkpoint = retrieve_checkpoint(dataset_name, num_epochs - 1)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler.current_epoch = start_epoch
        train_loss_init = checkpoint['training_loss']
        print(f"Model loaded from checkpoint, resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print(f"Model not found, starting from scratch")

    # Regular training
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_fn_train,
            aux_loss_function=topk_accuracy,
            scheduler=scheduler,
            device=device,
        )
        end_time = time.time()
        epoch_time = end_time - start_time
        save_training_state(dataset_name, model, optimizer, epoch, train_loss)
        print(
            f'Epoch {epoch}, Training Loss: {train_loss: .6f}, Training Accuracy: {train_acc * 100: 2.1f}%, Time: {epoch_time: .2f}')

    # Training loss after training
    train_loss_after, train_acc_after = evaluate_model(
        model, train_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    print(f'Training Loss after training: {train_loss_after: .6f}')

    # early stopping
    if device.type == 'mps':
        print(f"Early stopping, the rest of the code is not supported on MPS.")
        exit()

    # Compute the update
    print(f"Computing the parameter update")
    keep_neurons = 10
    part = 'all'
    stat_loss, stat_acc = gather_statistics_and_update(
        model,
        train_loader,
        loss_fn_growth,
        batch_limit=-1,
        maximum_added_neurons=keep_neurons,
        part=part
    )
    model.reset_computation()
    if model.currently_updated_block.eigenvalues_extension is not None:
        print(f"Number of added neurons: {model.currently_updated_block.eigenvalues_extension.size(0)}")
    else:
        print("No neurons added")

    # Compare activities and gradients
    print("Comparing the activities and gradients")
    model.currently_updated_block.scaling_factor = 1.0
    running_results = compare_activities_and_gradients(model, train_loader, loss_fn_growth, batch_limit=-1)
    display_results(running_results)
