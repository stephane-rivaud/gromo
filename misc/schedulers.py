import math


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


known_schedulers = {
    "step": step_lr,
    "multistep": multistep_lr,
    "cosine": cosine_lr,
    "constant": constant_lr,
}