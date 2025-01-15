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
    from gromo.growing_residual_mlp import GrowingResidualBlock

    # Fix the random seed
    torch.manual_seed(0)

    # Define the dataset
    batch_size = 100
    num_batch = 10
    in_features = 10
    dataset = [(torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)) for _ in range(num_batch)]

    # Define the model
    hidden_features = 5
    block = GrowingResidualBlock(in_features, hidden_features, activation=nn.ReLU(), name="block")
    print(block)

    # Define the losses
    loss_fn_mean = nn.MSELoss(reduction='mean')
    loss_fn_sum = nn.MSELoss(reduction='sum')

    # Define the optimizer
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    # Regular training
    for epoch in range(30):
        training_loss = train(block, dataset, loss_fn_mean, optimizer)
        print(f'Epoch {epoch}, Training Loss: {training_loss}')

    # Training loss after training
    training_loss_after = evaluate(block, dataset, loss_fn_mean)
    print(f'Training Loss after training: {training_loss_after}')

    # Gathering growing statistics
    statistics_loss = compute_statistics(block, dataset, loss_fn_sum) / (batch_size * in_features)
    print(f'Training Loss after gathering statistics: {statistics_loss}')

    # Compute the optimal update
    keep_neurons = 1
    block.compute_optimal_update(maximum_added_neurons=keep_neurons)

    # Training loss with the change
    scaling_factor = 0.5
    block.second_layer.scaling_factor = scaling_factor
    loss_with_extension = evaluate_with_extension(block, dataset, loss_fn_mean)
    print(f'Training Loss with the change: {loss_with_extension}')

    # debug: print all members of the block
    for members in dir(block):
        print(members)
        assert hasattr(block, members), f"block does not have member {members}"

    print(f'First order improvement: {block.first_order_improvement}')
    print(f'Zero-th order improvement: {training_loss_after - loss_with_extension}')

    # Store updates before applying changes
    weight, bias = tuple(param.clone() for param in block.second_layer.parameters())
    delta, delta_bias = tuple(param.clone() for param in block.second_layer.optimal_delta_layer.parameters())
    alpha, alpha_bias = tuple(param.clone() for param in block.first_layer.extended_output_layer.parameters())
    omega = block.second_layer.extended_input_layer.weight.clone()

    # Apply the change
    print('Apply the change')
    block.apply_change()

    # Assert the two values are "close enough" within the tolerance
    tolerance = 1e-6  # Adjust this value based on the required precision
    linear_factor = scaling_factor**2 * torch.sign(torch.tensor(scaling_factor))
    assert torch.allclose(
        weight - linear_factor * delta,
        block.second_layer.weight[:, :weight.shape[1]],
        atol=tolerance,
        ), (
        f"Weight ({weight}) and updated weight ({block.second_layer.weight}) "
        f"are not close enough. "
        f"(Absolute difference: {torch.abs(weight - block.second_layer.weight)})"
    )
    assert torch.allclose(
        bias - linear_factor * delta_bias,
        block.second_layer.bias,
        atol=tolerance,
        ), (
        f"Bias ({bias}) and updated bias ({block.second_layer.bias}) "
        f"are not close enough. "
        f"(Absolute difference: {torch.abs(bias - block.second_layer.bias)}) "
    )
    assert torch.allclose(
        alpha * scaling_factor,
        block.first_layer.weight[-alpha.shape[0]:, :],
        atol=tolerance,
        ), (
        f"Alpha ({alpha}) and updated alpha ({block.first_layer.weight[-alpha.shape[0]:, :]}) "
        f"are not close enough. "
        f"(Absolute difference: {torch.abs(alpha - block.first_layer.weight[-alpha.shape[0]:, :])}) "
    )
    assert torch.allclose(
        alpha_bias * scaling_factor,
        block.first_layer.bias[-alpha_bias.shape[0]:],
        atol=tolerance,
        ), (
        f"Alpha bias ({alpha_bias}) and updated alpha bias ({block.first_layer.bias[-alpha_bias.shape[0]:]}) "
        f"are not close enough. "
        f"(Absolute difference: {torch.abs(alpha_bias - block.first_layer.bias[-alpha_bias.shape[0]:])}) "
    )
    assert torch.allclose(
        omega * scaling_factor,
        block.second_layer.weight[:, -omega.shape[1]:],
        atol=tolerance,
        ), (
        f"Omega ({omega}) and updated omega ({block.second_layer.weight[:, -omega.shape[1]:]}) "
        f"are not close enough. "
        f"(Absolute difference: {torch.abs(omega - block.second_layer.weight[:, -omega.shape[1]:])}) "
    )

    # Delete the update
    print('Delete the update')
    block.delete_update()
    block.reset_computation()

    # Training loss after the change
    training_loss_after_change = evaluate(block, dataset, loss_fn_mean)
    print(f'Training Loss after the change: {training_loss_after_change}')
    print(f'Zero-th order improvement: {training_loss_after - training_loss_after_change}')
    print(block)

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
