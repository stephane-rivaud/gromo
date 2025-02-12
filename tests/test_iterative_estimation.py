import torch
from tests.test_activities_and_gradients import get_full_activity


def init_model(model):
    model.currently_updated_block.previous_module.store_input = True
    model.currently_updated_block.store_pre_activity = True


def get_stored_input(model):
    return model.currently_updated_block.previous_module.input


def get_functional_gradient(model):
    return model.currently_updated_block._pre_activity.grad.clone()


def iterative_natural_gradient_step(model, dataloader, loss_fn, num_epochs, batch_limit=-1):
    """

    Args:
        model:
        dataloader:
        loss_fn:
        batch_limit:

    Returns:

    """

    # collects input and functional gradients
    init_model(model)
    previous_layer_inputs = []
    original_activity = []
    functional_gradients = []
    for i, (inputs, targets) in enumerate(dataloader):
        if i == batch_limit:
            break
        # inputs, targets = inputs.to(model.device), targets.to(model.device)

        # get the functional gradient
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        functional_gradients.append(get_functional_gradient(model).detach())
        previous_layer_inputs.append(get_stored_input(model).detach())
        original_activity.append(get_full_activity(model).detach())

    # natural gradient descent
    param_list = [
        p for p in model.currently_updated_block.parameters() if p.requires_grad
    ]
    param_list += [
        p for p in model.currently_updated_block.previous_module.parameters() if p.requires_grad
    ]
    local_optimizer = torch.optim.SGD(param_list, lr=0.001, momentum=0.7)

    def functional_loss(contribution, gradient):
        return torch.nn.functional.mse_loss(
            contribution.flatten(end_dim=-2),
            -gradient.flatten(end_dim=-2),
            reduction='sum'
        )

    losses = torch.zeros(num_epochs)
    num_samples = 0
    for epoch in range(num_epochs):
        local_optimizer.zero_grad()
        for i, (input, activity, gradient) in enumerate(zip(previous_layer_inputs, original_activity, functional_gradients)):
            optimizer.zero_grad()
            new_activity = model.currently_updated_block.previous_module(input)
            new_activity = model.currently_updated_block(new_activity)
            contribution = new_activity - activity
            loss = functional_loss(contribution, gradient)
            loss.backward()
            losses[epoch] += loss.item()
            num_samples += activity.size(0) * activity.size(1)

        for param in param_list:
            param.grad /= num_samples
        local_optimizer.step()

    # average the losses
    losses /= num_samples
    return losses


if __name__ == "__main__":
    import os
    import time
    import math
    from torch import nn
    import matplotlib.pyplot as plt

    from gromo.utils.datasets import get_dataloaders
    from gromo.growing_mlp_mixer import GrowingMLPMixer
    from misc.schedulers import get_scheduler
    from misc.auxilliary_functions import train, evaluate_model, compute_statistics, extended_evaluate_model, \
        topk_accuracy
    from gromo.utils.utils import set_device, global_device
    from tests.test_activities_and_gradients import gather_statistics_and_update, compare_activities_and_gradients, \
        display_results

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
    print(f"Computing the update")
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

    # Compute the natural gradient step
    losses = iterative_natural_gradient_step(model, train_loader, loss_fn_growth, num_epochs=15, batch_limit=-1)

    # Plot the losses
    plt.figure()
    plt.plot(losses.numpy())
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses over Epochs')
    plt.show()

    # Compute the update
    print(f"Computing the parameter update")
    keep_neurons = 10
    part = 'neuron'
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
