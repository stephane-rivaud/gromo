import math
from functools import partial


def get_scheduler(scheduler, nb_step, lr, warmup_iters):
    if scheduler == "step":
        scheduler_kwargs = {
            "step_size": nb_step // 3,
            "gamma": 0.1,
            "lr_init": lr,
            "warmup_iters": warmup_iters,
        }
    elif scheduler == "multistep":
        scheduler_kwargs = {
            "milestones": [nb_step // 2, 3 * (nb_step // 4)],
            "gamma": 0.1,
            "lr_init": lr,
            "warmup_iters": warmup_iters,
        }
    elif scheduler == "cosine":
        scheduler_kwargs = {
            "total_iters": nb_step,
            "lr_init": lr,
            "lr_min": 1e-6,
            "warmup_iters": warmup_iters,
        }
    elif scheduler == "none":
        scheduler_kwargs = {"lr_init": lr}
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    scheduler = partial(known_schedulers[scheduler], **scheduler_kwargs)
    return scheduler


def warm_up_lr(iter, total_iters, lr_final):
    gamma = (iter + 1) / total_iters
    return gamma * lr_final


def step_lr(iter, step_size, gamma, lr_init, warmup_iters=0):
    if iter < warmup_iters:
        return warm_up_lr(iter, warmup_iters, lr_init)
    else:
        return lr_init * (gamma ** (iter // step_size))


def multistep_lr(iter, milestones, gamma, lr_init, warmup_iters=0):
    if iter < warmup_iters:
        return warm_up_lr(iter, warmup_iters, lr_init)
    else:
        return lr_init * (gamma ** sum(iter >= m for m in milestones))


def cosine_lr(iter, total_iters, lr_init, lr_min=0., warmup_iters=0):
    if iter < warmup_iters:
        return warm_up_lr(iter, warmup_iters, lr_init)
    else:
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters)))


def constant_lr(iter, lr_init):
    return lr_init


# Learning rate scheduler
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_batches_per_epoch, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_step / (self.warmup_epochs * self.num_batches_per_epoch))
        else:
            progress = ((self.current_step - self.warmup_epochs * self.num_batches_per_epoch) /
                        ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


known_schedulers = {
    "step": step_lr,
    "multistep": multistep_lr,
    "cosine": cosine_lr,
    "constant": constant_lr,
}