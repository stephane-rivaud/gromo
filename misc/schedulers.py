import torch
import numpy as np


def get_scheduler(
        scheduler_name: str,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_batches_per_epoch: int,
        base_lr: float,
        warmup_epochs: int,
):
    if scheduler_name == "step":
        return StepScheduler(optimizer, step_size=num_epochs // 3, gamma=0.1, lr_init=base_lr, num_batches_per_epoch=num_batches_per_epoch, warmup_epochs=warmup_epochs)
    elif scheduler_name == "multistep":
        return MultistepScheduler(optimizer, milestones=[num_epochs // 2, 3 * num_epochs // 4], gamma=0.1, lr_init=base_lr, num_batches_per_epoch=num_batches_per_epoch, warmup_epochs=warmup_epochs)
    elif scheduler_name == "cosine":
        return CosineScheduler(optimizer, lr_init=base_lr, warmup_epochs=warmup_epochs, total_epochs=num_epochs, num_batches_per_epoch=num_batches_per_epoch, min_lr=1e-6)
    elif scheduler_name == "none":
        return ConstantScheduler(optimizer, base_lr)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")



def warm_up_lr(iter, total_iters, lr_final, lr_init=0.0):
    return lr_init + (lr_final - lr_init) * iter / total_iters


class StepScheduler:
    def __init__(self, optimizer, step_size, gamma, lr_init, num_batches_per_epoch, warmup_epochs):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lr = lr_init
        self.current_epoch = 0
        self.current_step = 0
        self.num_batches_per_epoch = num_batches_per_epoch
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            current_step = self.current_epoch * self.num_batches_per_epoch + self.current_step
            lr = warm_up_lr(current_step, self.warmup_epochs * self.num_batches_per_epoch, self.lr)
        else:
            lr = self.lr * (self.gamma ** (self.current_epoch // self.step_size))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


class MultistepScheduler:
    def __init__(self, optimizer, milestones, gamma, lr_init, num_batches_per_epoch, warmup_epochs):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.lr = lr_init
        self.current_epoch = 0
        self.current_step = 0
        self.num_batches_per_epoch = num_batches_per_epoch
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            current_step = self.current_epoch * self.num_batches_per_epoch + self.current_step
            lr = warm_up_lr(current_step, self.warmup_epochs * self.num_batches_per_epoch, self.lr)
        else:
            lr = self.lr
            for milestone in self.milestones:
                if self.current_epoch >= milestone:
                    lr *= self.gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


class ConstantScheduler:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def epoch_step(self):
        pass


# Learning rate scheduler
class CosineScheduler:
    def __init__(self, optimizer, lr_init, warmup_epochs, total_epochs, num_batches_per_epoch, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.min_lr = min_lr
        self.base_lr = lr_init
        self.current_epoch = 0
        self.current_step = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"CosineScheduler(base_lr={self.base_lr}, min_lr={self.min_lr}, warmup_epochs={self.warmup_epochs}, total_epochs={self.total_epochs})"

    def step(self):
        self.current_step += 1
        minibatch_step = self.current_step + self.current_epoch * self.num_batches_per_epoch
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (minibatch_step / (self.warmup_epochs * self.num_batches_per_epoch))
        else:
            progress = ((minibatch_step - self.warmup_epochs * self.num_batches_per_epoch) /
                        ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


known_schedulers = {
    "step": StepScheduler,
    "multistep": MultistepScheduler,
    "cosine": CosineScheduler,
    "none": ConstantScheduler,
}