import torch

from misc.auxilliary_functions import AverageMeter


def gather_statistics_and_update(model, dataloader, loss_fn, batch_limit, maximum_added_neurons, part):
    # Gathering growing statistics
    stat_loss, stat_acc = compute_statistics(model, dataloader, loss_fn, batch_limit=batch_limit, device=device)
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
    rmse = torch.nn.functional.mse_loss(a1.flatten(end_dim=-2), a2.flatten(end_dim=-2), reduction='mean').sqrt()
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
    # check_weight_update(model, model.currently_updated_block.input_extended, functional_gradient)

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
    model.currently_updated_block.store_input = False
    model.currently_updated_block.store_pre_activity = False
    model.currently_updated_block._input = None
    model.currently_updated_block._pre_activity_original = None
    model.currently_updated_block._pre_activity_with_param_step = None
    model.currently_updated_block._pre_activity = None

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
        if key == 'norm':
            for sub_key in running_results[key].keys():
                plt.figure()
                plt.hist(running_results[key][sub_key].numpy(), bins=30)
                plt.title(f'{key} -- {sub_key}')
                plt.grid(True)
                plt.show()
        else:
            for metric in running_results[key].keys():
                plt.figure()
                plt.hist(running_results[key][metric].numpy(), bins=30)
                plt.title(f'{key} -- {metric}')
                plt.grid(True)
                plt.show()


def train_optimal_delta_layer(model, dataloader, loss_fn, optimizer, batch_limit):
    model.currently_updated_block.optimal_delta_layer.train()
    loss_meter = AverageMeter()
    model.zero_grad()
    for i, (x, y) in enumerate(dataloader):
        if i == batch_limit:
            break

        # Get the functional gradient
        init_model(model)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        input_stored = model.currently_updated_block.input.clone()
        functional_gradient = get_functional_gradient(model)
        reset_model(model)

        # Get the optimal delta layer
        init_model(model)
        model.zero_grad()
        param_step_activity = model.currently_updated_block.optimal_delta_layer(input_stored)
        loss = torch.nn.functional.mse_loss(
            param_step_activity.flatten(end_dim=-2),
            -functional_gradient.flatten(end_dim=-2),
            reduction='mean'
        )
        loss.backward()
        loss_meter.update(loss.item(), x.size(0) * x.size(1))
        reset_model(model)

    # normalize gradients
    for param in model.currently_updated_block.optimal_delta_layer.parameters():
        param.grad /= len(dataloader)
        print(f"Parameter grad norm: {param.grad.norm()} (shape: {param.grad.size()})")
    optimizer.step()

    return loss_meter.avg


if __name__ == "__main__":
    import os
    import time
    import math
    from torch import nn
    import matplotlib.pyplot as plt

    from gromo.utils.datasets import get_dataloaders
    from gromo.growing_mlp_mixer import GrowingMLPMixer
    from misc.schedulers import get_scheduler
    from misc.auxilliary_functions import train, evaluate_model, compute_statistics, extended_evaluate_model, topk_accuracy
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
    dataset_name = 'cifar10'
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
    num_features = 8
    hidden_dim_token = 4
    hidden_dim_channel = 32
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

    model_copy = GrowingMLPMixer(
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
    model_copy.to(device)

    # Define the losses
    loss_fn_mean = nn.CrossEntropyLoss(reduction='mean')
    loss_fn_sum = nn.CrossEntropyLoss(reduction='sum')

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=5e-5)

    # define the scheduler
    num_epochs = 10
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
            checkpoint_test = os.path.join(model_dir, f'mlp_mixer-{dataset}-epoch_{epoch}.pth')
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test
                break
        if checkpoint_path is None:
            checkpoint_test = os.path.join(model_dir, f'mlp_mixer-{dataset_name}.pth')
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test

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
            loss_function=loss_fn_mean,
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
        model, train_loader, loss_fn_mean, topk_accuracy, batch_limit=-1, device=device
    )
    print(f'Training Loss after training: {train_loss_after: .6f}')

    # early stopping
    if device.type == 'mps':
        print(f"Early stopping, the rest of the code is not supported on MPS.")
        exit()

    # Synchronize the models
    model_copy.load_state_dict(model.state_dict())

    # Compute the parameter update
    print(f"Computing the parameter update")
    keep_neurons = 10
    part = 'all'
    stat_loss, stat_acc = gather_statistics_and_update(
        model,
        train_loader,
        loss_fn_sum,
        batch_limit=-1,
        maximum_added_neurons=keep_neurons,
        part=part
    )
    if model.currently_updated_block.eigenvalues_extension is not None:
        print(f"Number of added neurons: {model.currently_updated_block.eigenvalues_extension.size(0)}")
    else:
        print("No neurons added")

    # # Compare the weights
    # print("Comparing the weights")
    # compare_weight_updates(model, model_copy, part=part)
    # print()

    # Compare activities and gradients
    print("Comparing the activities and gradients")
    running_results = compare_activities_and_gradients(model, train_loader, loss_fn_sum, batch_limit=-1)
    display_results(running_results)
    model.reset_computation()

    # Performs batch gradient descent on the optimal delta layer
    optimizer_delta = torch.optim.SGD(model.currently_updated_block.optimal_delta_layer.parameters(), lr=0.01,
                                      momentum=0.0)
    # optimizer_delta = torch.optim.AdamW(model.currently_updated_block.optimal_delta_layer.parameters(), lr=0.001, betas=(0.9, 0.99))
    delta_losses = []
    num_steps = 30
    for i in range(num_steps):
        delta_train_loss = train_optimal_delta_layer(model, train_loader, loss_fn_sum, optimizer_delta, batch_limit=-1)
        delta_losses.append(delta_train_loss)
        print(f"Iteration {i}, Delta Training Loss: {delta_train_loss: .6f}")

    # Plot the delta training losses
    plt.figure()
    plt.plot(range(1, num_steps + 1), delta_losses, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Delta Training Loss')
    plt.title('Delta Training Loss vs Iteration')
    plt.grid(True)
    plt.show()

    # Compare activities and gradients
    print("Comparing the activities and gradients")
    compare_activities_and_gradients(model, train_loader, loss_fn_sum, batch_limit=-1)
    print()

    # # Compute the update with a limited number of batches
    # ratio = 0.3
    # batch_limit = int(len(train_loader) * ratio)
    # stat_loss_limited, stat_acc_limited = gather_statistics_and_update(
    #     model_copy,
    #     train_loader,
    #     loss_fn_sum,
    #     batch_limit=batch_limit,
    #     maximum_added_neurons=keep_neurons,
    #     part=part
    # )

    # # Compute the full update
    # print(f"Computing the full update")
    # keep_neurons = 10
    # part = 'all'
    # stat_loss, stat_acc = gather_statistics_and_update(
    #     model,
    #     train_loader,
    #     loss_fn_sum,
    #     batch_limit=-1,
    #     maximum_added_neurons=keep_neurons,
    #     part=part
    # )
    # if model.currently_updated_block.eigenvalues_extension is not None:
    #     print(f"Number of added neurons: {model.currently_updated_block.eigenvalues_extension.size(0)}")
    # else:
    #     print("No neurons added")

    # # Compare activities and gradients
    # print("Comparing the activities and gradients")
    # compare_activities_and_gradients(model, train_loader, loss_fn_sum, batch_limit=-1)
    # model.reset_computation()

    # Compare the activity of different estimates
    # print("Comparing the activity updates")
    # compare_activity_updates(model, model_copy, train_loader)
    # print()

    # Compare the parameter step and the functional gradient
    # if part in ['all', 'parameter']:
    #     print("Comparing the parameter step and the functional gradient")
    #     compare_param_step_and_functional_gradient(
    #         model,
    #         train_loader,
    #         loss_fn_mean,
    #         batch_limit=-1
    #     )
    #     print()
    #
    # # Compare the neuron contribution and the desired update
    # if part in ['all', 'neuron']:
    #     print("Comparing the neuron contribution and the desired update")
    #     compare_neuron_contrib_and_desired_update(
    #         model,
    #         train_loader,
    #         loss_fn_mean,
    #         batch_limit=-1
    #     )
    #     print()
    #
    # # Compare the neuron contribution and the functional gradient
    # if part in ['all', 'neuron']:
    #     print("Comparing the neuron contribution and the functional gradient")
    #     compare_neuron_contrib_and_functional_gradient(
    #         model,
    #         train_loader,
    #         loss_fn_mean,
    #         batch_limit=-1
    #     )
    #     print()
    #
    # # Compare the full contribution and the functional gradient
    # print("Comparing the full contribution and the functional gradient")
    # compare_full_contrib_and_functional_gradient(
    #     model,
    #     train_loader,
    #     loss_fn_mean,
    #     batch_limit=-1
    # )
    # print()

    # # Line search
    # gammas = torch.linspace(-8.0, 8.0, 8*2*3 + 1)
    # losses = torch.zeros(len(gammas))
    # first_order_approx = torch.zeros(len(gammas))
    # for i, gamma in enumerate(gammas):
    #     model.currently_updated_block.scaling_factor = gamma.item()
    #     losses[i] = extended_evaluate_model(model, train_loader, loss_fn_sum, batch_limit=batch_limit, device=device)
    #     first_order_approx[i] = train_loss_after - model.first_order_improvement * gamma
    #
    #     # Compare the natural gradient and the new contribution
    #     cosine_sim = compare_natural_gradient_and_new_contribution(
    #         model, train_loader, loss_fn_mean, batch_limit=-1
    #     )
    #     print(f'Gamma: {gamma: .2f}'
    #           f', Loss with extension: {losses[i]: .6f}'
    #           f', First order approximation: {first_order_approx[i]: .6f}'
    #           f', Cosine similarity: {cosine_sim: .6f}')
    #
    # # Plot the losses against gamma values
    # plt.figure()
    # plt.plot(gammas.numpy(), losses.numpy(), marker='o')
    # plt.plot(gammas.numpy(), first_order_approx.numpy(), marker='x')
    # plt.title(f'Loss with extension vs Gamma [batch limit: {batch_limit}]')
    # plt.xlabel('Gamma')
    # plt.grid(True)
    # plt.show()
    #
    # # Check the update
    # model.currently_updated_block.scaling_factor = 6.0
    # # print(f"Selected update: {model.currently_updated_block}")
    #
    # alpha = model.currently_updated_block.previous_module.extended_output_layer.weight.clone()
    # alpha_bias = model.currently_updated_block.previous_module.extended_output_layer.bias.clone()
    # omega = model.currently_updated_block.extended_input_layer.weight.clone()
    # eigenvalues = model.currently_updated_block.eigenvalues_extension.clone()
    #
    # # Compare the natural gradient and the new contribution
    # cosine_sim = compare_natural_gradient_and_new_contribution(
    #     model, train_loader, loss_fn_mean, batch_limit=-1
    # )
    # print(f'Cosine similarity natural gradient and new contribution: {cosine_sim: .6f}')

    # for factor in torch.linspace(0.0, 1.0, 11)[1:-1]:
    # Gathering growing statistics
    # model.reset_computation()
    # model.delete_update()
    # batch_limit = int(390 * factor)
    # stat_loss, stat_acc = compute_statistics(model, train_loader, loss_fn_sum, batch_limit=batch_limit, device=device)
    #
    # # Compute the optimal update
    # # print(f'Factor: {factor: .2f}, Loss before change: {stat_loss: .6f}')
    # model.compute_optimal_update(maximum_added_neurons=keep_neurons, dtype=growing_dtype, part=part)
    #
    # # Select the best update
    # model.select_best_update()
    # print(f"Selected update: {model.currently_updated_block}")

    # new_alpha = model.currently_updated_block.previous_module.extended_output_layer.weight
    # new_alpha_bias = model.currently_updated_block.previous_module.extended_output_layer.bias
    # new_omega = model.currently_updated_block.extended_input_layer.weight
    # new_eigenvalues = model.currently_updated_block.eigenvalues_extension
    #
    # # compare the weights
    # norm_ratio_alpha, cosine_sim_alpha, rel_mse_alpha = compare_weights(alpha, new_alpha)
    # norm_ratio_alpha_bias, cosine_sim_alpha_bias, rel_mse_alpha_bias = compare_weights(alpha_bias, new_alpha_bias)
    # norm_ratio_omega, cosine_sim_omega, rel_mse_omega = compare_weights(omega, new_omega)
    # norm_ratio_eigenvalues, cosine_sim_eigenvalues, rel_mse_eigenvalues = compare_weights(eigenvalues, new_eigenvalues)
    #
    # print(f'Factor: {factor: .2f}, Norm ratio alpha: {norm_ratio_alpha: .6f}, Cosine similarity alpha: {cosine_sim_alpha: .6f}, Relative MSE alpha: {rel_mse_alpha: .6f}')
    # print(f'Factor: {factor: .2f}, Norm ratio alpha bias: {norm_ratio_alpha_bias: .6f}, Cosine similarity alpha bias: {cosine_sim_alpha_bias: .6f}, Relative MSE alpha bias: {rel_mse_alpha_bias: .6f}')
    # print(f'Factor: {factor: .2f}, Norm ratio omega: {norm_ratio_omega: .6f}, Cosine similarity omega: {cosine_sim_omega: .6f}, Relative MSE omega: {rel_mse_omega: .6f}')
    # print(f'Factor: {factor: .2f}, Norm ratio eigenvalues: {norm_ratio_eigenvalues: .6f}, Cosine similarity eigenvalues: {cosine_sim_eigenvalues: .6f}, Relative MSE eigenvalues: {rel_mse_eigenvalues: .6f}')

    # Compare the natural gradient and the new contribution
    # cosine_sim = compare_natural_gradient_and_new_contribution(
    #     model, train_loader, loss_fn_mean, batch_limit=batch_limit
    # )
    # print(f'Factor: {factor: .2f}, Cosine similarity natural gradient and new contribution: {cosine_sim: .6f}'
    #       f' [Batch limit: {batch_limit}]')

    # Compare the natural gradient and the new contribution
    # cosine_sim = compare_natural_gradient_and_new_contribution(
    #     model, train_loader, loss_fn_mean, batch_limit=-1
    # )
    # print(f'Factor: {factor: .2f}, Cosine similarity natural gradient and new contribution: {cosine_sim: .6f}'
    #       f' [Batch limit: {-1}]')

    # # Training loss with the change
    # gammas = torch.linspace(-8.0, 8.0, 30)
    # losses = torch.zeros(len(gammas))
    # first_order_approx = torch.zeros(len(gammas))
    # for i, gamma in enumerate(gammas):
    #     model.currently_updated_block.scaling_factor = gamma.item()
    #     losses[i] = extended_evaluate_model(model, train_loader, loss_fn_sum, batch_limit=batch_limit, device=device)
    #     first_order_approx[i] = stat_loss - model.first_order_improvement * gamma
    #     print(f'Gamma: {gamma: .2f}, Loss with extension: {losses[i]: .6f}, First order approximation: {first_order_approx[i]: .6f}')
    #
    # # Plot the losses against gamma values
    # plt.figure()
    # plt.plot(gammas.numpy(), losses.numpy(), marker='o')
    # plt.plot(gammas.numpy(), first_order_approx.numpy(), marker='x')
    # plt.xlabel('Gamma')
    # plt.ylabel('Loss with extension')
    # plt.title('Loss with extension vs Gamma')
    # plt.grid(True)
    # plt.show()
    #
    # gamma_index = torch.argmin(losses)
    # scaling_factor = gammas[gamma_index].item()
    # loss_with_extension = losses[gamma_index].item()
    # print(f'Training Loss with the change: {loss_with_extension: .6f}')
    # print(f'Scaling factor: {scaling_factor: .6f}')
    # print(f'First order improvement: {model.first_order_improvement: 1.3e}')
    # print(f'Scaled first order improvement: {model.first_order_improvement * scaling_factor: 1.3e}')
    # print(f'Zero-th order improvement: {train_loss_after - loss_with_extension: 1.3e}')
    #
    # # Apply the change
    # model.currently_updated_block.scaling_factor = scaling_factor
    # model.apply_change()
    # model.delete_update()
    # model.reset_computation()
    #
    # # Assert the two values are "close enough" within the tolerance
    # train_loss_after_change, _ = evaluate_model(
    #     model, train_loader, loss_fn_mean, batch_limit=batch_limit, device=device
    # )
    # tolerance = 1e-6  # Adjust this value based on the required precision
    # assert torch.isclose(
    #     torch.tensor(loss_with_extension),
    #     torch.tensor(train_loss_after_change),
    #     atol=tolerance
    # ), (
    #     f"Loss with extension ({loss_with_extension}) "
    #     f"and training loss after change ({train_loss_after_change}) "
    #     f"are not close enough. "
    #     f"(Absolute difference: {torch.abs(torch.tensor(loss_with_extension - train_loss_after_change))})"
    # )
    #
    # # Training loss after the change
    # train_loss_after_change, _ = evaluate_model(
    #     model, train_loader, loss_fn_mean, batch_limit=-1, device=device
    # )
    # print(f'Training Loss after the change: {train_loss_after_change: .6f}')
    # print(f'Zero-th order improvement: {train_loss_after - train_loss_after_change: 1.3e}')
    # print(model)
