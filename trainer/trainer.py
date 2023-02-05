import torch
from tqdm import tqdm
import os
from torch.nn.utils import clip_grad_norm_
from utils.utils import AverageMeter
from termcolor import colored
import wandb
import numpy as np
from optimizers import SAM
from models.lstm_sentiment import LSTMSentimentAnalysis


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
    def __init__(self, model, config, criterion, optimizer, device, train_loader, test_loader,
                 lr_scheduler=None, logger=False):
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.current_epoch = 0
        self.best_acc = 0
        if config['freeze_backbone']:
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
        pbar = tqdm(enumerate(self.train_loader), desc=f'Train epoch {epoch + 1}/{self.config["epoch"]}: ',
                    total=len(self.train_loader))
        result = None
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for _, (inputs, labels) in pbar:
            if isinstance(inputs, list):
                words, chars = inputs
                words, chars = words.to(self.device), chars.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(words, chars)
            else:
                # to('cuda') is handled in text_collate function of dataloader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(logits, labels)
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), 5)
            if isinstance(self.optimizer, SAM):
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(inputs), labels).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
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
        pbar = tqdm(enumerate(self.test_loader), desc=f'Eval epoch {epoch + 1}/{self.config["epoch"]}: ',
                    total=len(self.test_loader))
        result = None
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for i, (inputs, labels) in pbar:
            if isinstance(inputs, list):
                words, chars = inputs
                words, chars = words.to(self.device), chars.to(self.device)
                labels = labels.to(self.device)
                inputs = (words, chars)
            else:
                # to('cuda') is handled in text_collate function of dataloader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                if isinstance(self.model, LSTMSentimentAnalysis):
                    logits = self.model(*inputs)
                else:
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
        checkpoint = torch.load(os.path.join(self.config['save_dir'], 'checkpoint_last.pt'), map_location=self.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.param_groups[0]['capturable'] = True
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("------> load checkpoint")

    def train(self, resume=False):
        with torch.autograd.set_detect_anomaly(True):
            if resume:
                self._load_model()
            for epoch in range(self.current_epoch, self.config['epoch']):
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
                    torch.save(test_result, os.path.join(self.config['save_dir'], 'checkpoint_best.pt'))
                torch.save(test_result, os.path.join(self.config['save_dir'], 'checkpoint_last.pt'))
