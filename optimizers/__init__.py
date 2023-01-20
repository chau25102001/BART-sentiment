from torch.optim import *
from .adan import *
from .sam import *
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_lr_scheduler(optimizer, config, train_loader):
    name = config['lr_scheduler']
    if name is None:
        lr_scheduler = None
    elif name == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['epoch'] * len(train_loader), eta_min=1e-6)
    elif name == "linear_warm_up":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                       num_training_steps=config['epoch'] * len(train_loader))
    elif name == "constant_warm_up":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=len(train_loader))
    else:
        print(f"not supported {name} lr scheduler, use default: None")
        lr_scheduler = None
    return lr_scheduler
