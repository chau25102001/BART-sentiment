import argparse

import termcolor
import torch
from datasets.imbd_dataset import IMDBDataset, text_collate
from models.bart_sentiment import BartSentimentAnalysis
from transformers import BartConfig
from trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import json

parser = argparse.ArgumentParser(description="BART-sentiment")
parser.add_argument("--resume", default=False, action="store_true", help="resume training?")
parser.add_argument("--pretrain", default=False, action="store_true", help="use pretrained backbone?")
parser.add_argument("--no_lr_scheduler", default=True, action="store_false", help="add this to remove lr scheduler")
parser.add_argument("--config", default="./config/bart.json", type=str, help="path to train config")
parser.add_argument("--freeze_backbone", default=False, action='store_true', help="freeze BART pretrained backbone?")
args = parser.parse_args()


def train(args):
    train_config = json.load(open(args.config))
    config = BartConfig(num_labels=2) if not args.pretrain else BartConfig.from_pretrained('facebook/bart-base')
    config.num_labels = 2
    model = BartSentimentAnalysis(config=config, pretrained=args.pretrain)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model: BART-sentiment, num params: {sum(p.numel() for p in model.parameters())}")
    tokenizer = model.tokenizer
    model = DataParallel(model).to(device)
    trainset = IMDBDataset(csv_path=train_config['train']['path'])
    testset = IMDBDataset(csv_path=train_config['test']['path'])
    print(f"train: {len(trainset)} samples, test: {len(testset)} samples")
    train_loader = DataLoader(trainset, batch_size=train_config['train']['batch_size'],
                              shuffle=train_config['train']['shuffle'],
                              collate_fn=lambda batch: text_collate(batch, tokenizer))
    test_loader = DataLoader(testset, batch_size=train_config['test']['batch_size'],
                             shuffle=train_config['test']['shuffle'],
                             collate_fn=lambda batch: text_collate(batch, tokenizer))
    optimizer = AdamW(model.parameters())
    lr_scheduler = None
    if args.no_lr_scheduler:
        if train_config['scheduler'] == 'cosine':  # cosine decrease lr
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epoch"] * len(train_loader),
                                             eta_min=1e-6)
        elif train_config['scheduler'] == 'linear_warm_up':  # warm up from 0 to init lr for 1 epoch, linear decrease for the rest
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                           num_training_steps=len(train_loader) * train_config['epoch'])
        else:
            print(termcolor.colored("Warning: not supported lr scheduler, using default: None", 'red'))
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                      criterion=criterion,
                      num_epoch=train_config['epoch'],
                      optimizer=optimizer,
                      device=device,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      save_dir=train_config['save_dir'],
                      lr_scheduler=lr_scheduler,
                      logger=True, freeze_backbone=args.freeze_backbone)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    import warnings

    wandb.init(project="BART-sentiment-analysis")
    warnings.filterwarnings('ignore')
    train(args)
