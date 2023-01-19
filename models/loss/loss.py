import torch
import torch.nn as nn

def cross_entropy(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)