import torch


def train(model, dataset, loss_fn, optimizer):
    total_loss = 0
    for x, y in dataset:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss
        loss.backward()
        optimizer.step()
    return total_loss / len(dataset)


def evaluate(model, dataset, loss_fn):
    total_loss = 0
    with torch.no_grad():
        for x, y in dataset:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss
    return total_loss / len(dataset)


def compute_statistics(model, dataset, loss_fn):
    model.init_computation()
    total_loss = 0
    for x, y in dataset:
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss
        loss.backward()
        model.update_computation()
    return total_loss / len(dataset)


def evaluate_with_extension(model, dataset, loss_fn):
    total_loss = 0
    with torch.no_grad():
        for x, y in dataset:
            y_hat = model.extended_forward(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss
    return total_loss / len(dataset)


if __name__ == "__main__":
    from torch import nn
    from gromo.growing_residual_mlp import GrowingResidualMLP

    # Fix the random seed
    torch.manual_seed(0)

    # Define the dataset
    batch_size = 10
    num_batch = 10
    in_features = 10
    dataset = [(torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)) for _ in range(num_batch)]

    # Define the model
    hidden_features = 5
    model = GrowingResidualMLP(
        in_features,
        hidden_features,
        in_features,
        num_blocks=2,
        activation=nn.ReLU()
    )
    print(model)

    # Define the losses
    loss_fn_mean = nn.MSELoss(reduction='mean')
    loss_fn_sum = nn.MSELoss(reduction='sum')

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    # Regular training
    for epoch in range(10):
        training_loss = train(model, dataset, loss_fn_mean, optimizer)
        print(f'Epoch {epoch}, Training Loss: {training_loss}')

    # Training loss after training
    training_loss_after = evaluate(model, dataset, loss_fn_mean)
    print(f'Training Loss after training: {training_loss_after}')

    # Gathering growing statistics
    statistics_loss = compute_statistics(model, dataset, loss_fn_sum) / (batch_size * in_features)
    print(f'Training Loss after gathering statistics: {statistics_loss}')

    # Compute the optimal update
    keep_neurons = 1
    model.compute_optimal_update(maximum_added_neurons=keep_neurons)

    # Select the best update
    model.select_best_update()

    # Training loss with the change
    scaling_factor = 0.01
    model.blocks[model.currently_updated_block_index].scaling_factor = scaling_factor
    loss_with_extension = evaluate_with_extension(model, dataset, loss_fn_mean)
    print(f'Training Loss with the change: {loss_with_extension}')
    print(f'First order improvement: {model.first_order_improvement}')
    print(f'Zero-th order improvement: {training_loss_after - loss_with_extension}')

    # Apply the change
    print('Apply the change')
    model.apply_change()

    # Delete the update
    print('Delete the update')
    model.delete_update()
    model.reset_computation()

    # Training loss after the change
    training_loss_after_change = evaluate(model, dataset, loss_fn_mean)
    print(f'Training Loss after the change: {training_loss_after_change}')
    print(f'Zero-th order improvement: {training_loss_after - training_loss_after_change}')
    print(model)

    # Assert the two values are "close enough" within the tolerance
    tolerance = 1e-6  # Adjust this value based on the required precision
    assert torch.isclose(
        loss_with_extension,
        training_loss_after_change,
        atol=tolerance
    ), (
        f"Loss with extension ({loss_with_extension}) "
        f"and training loss after change ({training_loss_after_change}) "
        f"are not close enough. (Absolute difference: {torch.abs(loss_with_extension - training_loss_after_change)})"
    )
