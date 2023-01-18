import torch
from tqdm import tqdm
import os
from torch.nn.utils import clip_grad_norm_
from utils.utils import AverageMeter
from termcolor import colored
import wandb
import numpy as np


def accuracy(predict, label, num_classes=2):
    predict = predict.view(-1)
    label = label.view(-1)
    assert len(torch.unique(predict)) <= num_classes and len(
        torch.unique(label)) <= num_classes, 'invalid prediction, check it!'
    acc = []
    for c in range(num_classes):
        p = torch.where(predict == c, 1, 0)
        l = torch.where(label == c, 1, 0)
        TP = (p * l).sum()
        TN = ((1 - p) * (1 - l)).sum()
        FP = (p * (1 - l)).sum()
        FN = ((1 - p) * l).sum()
        acc.append(((TP + TN) / (TP + TN + FP + FN)).item())
    return np.mean(acc)


class Trainer:
    def __init__(self, model, criterion, num_epoch, optimizer, device, train_loader, test_loader, save_dir,
                 lr_scheduler=None, logger=False, freeze_backbone=False):
        self.model = model
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.current_epoch = 0
        self.best_acc = 0
        if freeze_backbone:
            self.prepare_finetune()
        count_grad = 0
        count_freeze = 0
        for p in self.model.parameters():
            if p.requires_grad:
                count_grad += 1
            else:
                count_freeze += 1
        print(colored(f"freezing: {count_freeze} weights, update: {count_grad} weights", "red"))

    def prepare_finetune(self):
        if self.model.module:  # DataParallel
            for p in self.model.module.bart.parameters():
                p.requires_grad = False
        else:
            for p in self.model.bart.parameters():
                p.requires_grad = False

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), desc=f'Train epoch {epoch + 1}/{self.num_epoch}: ',
                    total=len(self.train_loader))
        result = None
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for i, (inputs, labels) in pbar:
            input_ids = inputs['input_ids'].to(self.device)
            attn_mask = inputs['attention_mask'].to(self.device)
            inputs = {'input_ids': input_ids, 'attention_mask': attn_mask}
            labels = labels.to(self.device)
            logits = self.model(inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler:
                self.lr_scheduler.step()
            pred = torch.argmax(logits, dim=1)
            acc = accuracy(pred, labels)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            result = {"loss": loss_meter.average(),
                      "acc": acc_meter.average(),
                      "lr": lr}
            wandb.log({'train/lr': lr})
            pbar.set_postfix(result)
        return result

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(enumerate(self.test_loader), desc=f'Eval epoch {epoch + 1}/{self.num_epoch}: ',
                    total=len(self.test_loader))
        result = None
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for i, (inputs, labels) in pbar:
            input_ids = inputs['input_ids'].to(self.device)
            attn_mask = inputs['attention_mask'].to(self.device)
            inputs = {'input_ids': input_ids, 'attention_mask': attn_mask}
            labels = labels.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                pred = torch.argmax(logits, dim=1)
                acc = accuracy(pred, labels)
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                result = {"loss": loss_meter.average(),
                          "acc": acc_meter.average()}
                pbar.set_postfix(result)
        return result

    def _load_model(self):
        checkpoint = torch.load(os.path.join(self.save_dir, 'checkpoint_last.pt'), map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.param_groups[0]['capturable'] = True
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("------> load checkpoint")

    def train(self, resume=False):
        if resume:
            self._load_model()
        for epoch in range(self.current_epoch, self.num_epoch):
            train_result = self._train_epoch(epoch)
            train_loss = train_result['loss']
            train_acc = train_result['acc']

            test_result = self._eval_epoch(epoch)
            test_loss = test_result['loss']
            test_acc = test_result['acc']
            if self.logger:
                wandb.log({'train/loss': train_loss})
                wandb.log({'train/acc': train_acc})
                wandb.log({'test/loss': test_loss})
                wandb.log({'test/acc': test_acc})
            test_result['state_dict'] = self.model.module.state_dict() if self.model.module else self.model.state_dict()
            test_result['optimizer'] = self.optimizer.state_dict()
            test_result['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
            test_result['epoch'] = epoch
            if test_acc > self.best_acc:
                torch.save(test_result, os.path.join(self.save_dir, 'checkpoint_best.pt'))
            torch.save(test_result, os.path.join(self.save_dir, 'checkpoint_last.pt'))
