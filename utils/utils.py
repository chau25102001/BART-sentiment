# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import time
import json
from pathlib import Path
import torch
from transformers import BartTokenizer


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count += weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)

        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def read_json(fname):
    ''' read json file '''
    fname = Path(fname)
    with fname.open('r') as handle:
        return json.load(handle)


def text_collate(batch, tokenizer: BartTokenizer, device, max_seq_length=256):
    targets = []
    texts = []
    for _, sample in enumerate(batch):
        texts.append(sample[0])
        targets.append(sample[1])
    output_text = tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=max_seq_length)

    return output_text.to(device), torch.tensor(targets, dtype=torch.long).to(device)
