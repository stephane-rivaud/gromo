from typing import Callable, Tuple
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

from gromo.growing_mlp import GrowingMLP
from gromo.utils.utils import global_device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def topk_accuracy(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """
    Compute the top-k accuracy of the model on the given data
    Args:
        y_pred: The predictions of the model
        y: The true values
        k: The number of top values to consider

    Returns:
        The top-k accuracy
    """
    result = y_pred.topk(k, dim=1).indices == y.unsqueeze(1)
    return result.sum() / y.size(0)


# CutMix function
def cutmix_data(x, y, beta=1.0, cutmix_prob=0.5):
    if np.random.rand() > cutmix_prob:
        return x, y, y, 1.0  # No CutMix applied

    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    return x, y, shuffled_y, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module = nn.CrossEntropyLoss(reduction="mean"),
    val_dataloader: torch.utils.data.DataLoader | None = None,
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    scheduler=None,
    nb_epoch: int = 10,
    cutmix_beta: float = 1.0,
    cutmix_prob: float = 0.0,
    show: bool = False,
    device: torch.device = global_device(),
):
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    epoch_loss_train = []
    epoch_accuracy_train = []
    epoch_loss_val = []
    epoch_accuracy_val = []

    iterator = range(nb_epoch)
    if show:
        iterator = tqdm(iterator)

    for epoch in iterator:
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        nb_examples = 0
        for x, y in train_dataloader:
            x, y = x.to(global_device()), y.to(global_device())
            x, y, y_shuffled, lam = cutmix_data(x, y, beta=cutmix_beta, cutmix_prob=cutmix_prob)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = lam * loss_function(y_pred, y) + (1 - lam) * loss_function(y_pred, y_shuffled)
            assert (
                loss.isnan().sum() == 0
            ), f"During training of {model}, loss is NaN: {loss}"

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            if aux_loss_function:
                accuracy_meter.update(aux_loss_function(y_pred, y).item(), x.size(0))
            nb_examples += y.shape[0]

        epoch_loss_train.append(loss_meter.avg)
        epoch_accuracy_train.append(accuracy_meter.avg)

        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                loss_function=loss_function,
                aux_loss_function=aux_loss_function,
                device=device,
            )
            epoch_loss_val.append(val_loss)
            epoch_accuracy_val.append(val_accuracy)
            model.train()

        if scheduler is not None:
            scheduler.epoch_step()

        if show and epoch % max(1, (nb_epoch // 10)) == 0:
            print(
                f"Epoch {epoch}:\t",
                f"Train: loss={epoch_loss_train[-1]:.3e}, accuracy={epoch_accuracy_train[-1]:.2f}",
                (
                    f"Val: loss={epoch_loss_val[-1]:.3e}, accuracy={epoch_accuracy_val[-1]:.2f}"
                    if val_dataloader is not None
                    else ""
                ),
            )
    return epoch_loss_train, epoch_accuracy_train, epoch_loss_val, epoch_accuracy_val


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.CrossEntropyLoss(reduction="mean"),
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = -1,
    device: torch.device = global_device(),
) -> tuple[float, float]:
    """
    Evaluate the model on the dataloader with a limit of batch_limit batches.

    Parameters
    ----------
    model: nn.Module
        The model to evaluate
    dataloader: DataLoader
        The dataloader to use
    loss_function: nn.Module
        The loss function to use.
        /!/ The loss function should not be averaged over the batch
    aux_loss_function: nn.Module | None
        The auxiliary loss function to use.
    batch_limit: int
        The maximum number of batches to use
    device: torch.device
        The device to use

    Returns
    -------
    tuple[float, float]
        The average loss and the average auxiliary loss
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should not be averaged over the batch"

    model.eval()
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss.item(), x.size(0))
        if aux_loss_function is not None:
            aux_loss = aux_loss_function(y_pred, y)
            aux_loss_meter.update(aux_loss.item(), x.size(0))

    return loss_meter.avg, aux_loss_meter.avg


def extended_evaluate_model(
    growing_model: GrowingMLP,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
    batch_limit: int = -1,
    device: torch.device = global_device(),
) -> float:
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    growing_model.eval()
    loss_meter = AverageMeter()
    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model.extended_forward(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss.item() / x.size(0), x.size(0))
    return loss_meter.avg


def compute_statistics(
    growing_model: GrowingMLP,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = 1_000_000,
    device: torch.device = global_device(),
    show: bool = False,
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    growing_model: GrowingMLP
        The model to evaluate
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function to use.
        /!/ The loss function should not be averaged over the batch
    aux_loss_function: nn.Module | None
        The auxiliary loss function to use.
    dataloader: DataLoader
        The dataloader to use
    batch_limit: int
        The maximum number of batches to use
    device: torch.device
        The device to use
    show: bool
        If True, display a progress bar
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    growing_model.init_computation()
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    if show:
        dataloader = tqdm(dataloader)

    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        growing_model.update_computation()
        loss_meter.update(loss.item() / x.size(0), x.size(0))
        if aux_loss_function is not None:
            aux_loss = aux_loss_function(y_pred, y)
            aux_loss_meter.update(aux_loss.item(), x.size(0))

    return loss_meter.avg, aux_loss_meter.avg


def line_search(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
    batch_limit: int = -1,
    initial_loss: float | None = None,
    first_order_improvement: float = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    extended_search: bool = True,
    max_iter: int = 100,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device = global_device(),
) -> tuple[float, float, list[float], list[float]]:
    gammas = []
    losses = []
    beta = np.sqrt(beta)
    epsilon = np.sqrt(epsilon)
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma):
        model.amplitude_factor = sqrt_gamma
        loss = extended_evaluate_model(
            growing_model=model,
            loss_function=loss_function,
            dataloader=dataloader,
            batch_limit=batch_limit,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        if verbose:
            print(f"gamma n° {len(gammas)}: {sqrt_gamma ** 2:.3e} -> Loss: {loss:.3e}")
        return loss

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    if initial_loss is None:
        warn("Initial loss is not provided, computing it")
        initial_loss = test_gamma(0.0)
        print(f"Initial loss: {initial_loss:.3e}")
    else:
        gammas.append(0.0)
        losses.append(initial_loss)

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    l0 = test_gamma(t)
    l1 = l0
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = True
            while go:
                l0 = l1
                t /= beta
                l1 = test_gamma(t)
                go = l1 < l0 and i < max_iter
                i += 1
            t *= beta
        model.amplitude_factor = t
        return t**2, l0, gammas, losses
    else:
        go = True
        while go:
            l0 = l1
            t *= beta
            l1 = test_gamma(t)
            go = (
                ((not under_bound(t, l1)) or (l1 < l0 and extended_search))
                and i < max_iter
                and t > epsilon
            )
            i += 1
        t /= beta
        model.amplitude_factor = t
        return t**2, l0, gammas, losses


def full_search(
    model: nn.Module,
    loss: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    batch_limit: int = 1_000_000,
    initial_loss: float = None,
    first_order_improvement: float = 1,
    min_value: float = -100,
    max_value: float = 100,
    nb_points: int = 100,
):
    xs = np.linspace(min_value, max_value, nb_points)
    values = []
    for v in tqdm(xs):
        model.amplitude_factor = np.sign(v) * np.sqrt(abs(v))
        values.append(
            extended_evaluate_model(
                growing_model=model,
                loss_function=loss,
                dataloader=dataloader,
                batch_limit=batch_limit,
            )
        )

    return xs, values
