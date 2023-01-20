import argparse
import warnings

import termcolor
import torch

import models as model_module
import datasets as dataset_module
import models.loss as loss_module
import optimizers as optim_module
from trainer.trainer import Trainer
from torch.nn import DataParallel
import wandb
from config_parser import ConfigParser
from utils.utils import text_collate

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=False, action="store_true", help="resume training?")
parser.add_argument("--config", default="./config/bart.json", type=str, help="path to config file")
args = parser.parse_args()


def train(args):
    config = ConfigParser(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = config.init_obj(model_module, "arch")
    tokenizer = model.tokenizer
    model = DataParallel(model).to(device)

    print(f"Model: {config['name']}, num params: {sum(p.numel() for p in model.parameters())}")

    train_set = config.init_obj(dataset_module, "train_set")
    test_set = config.init_obj(dataset_module, "test_set")

    collate_fn = lambda batch: text_collate(batch, tokenizer, device, max_seq_length=config['max_seq_length'])
    train_loader = config.init_obj(torch.utils.data, "train_loader", dataset=train_set, collate_fn=collate_fn)
    test_loader = config.init_obj(torch.utils.data, "train_loader", dataset=test_set, collate_fn=collate_fn)

    optimizer = config.init_obj(optim_module, "optimizer", model.parameters())
    if isinstance(optimizer, optim_module.SAM):
        lr_scheduler = optim_module.get_lr_scheduler(optimizer.base_optimizer, config, train_loader)
    else:
        lr_scheduler = optim_module.get_lr_scheduler(optimizer, config, train_loader)
    criterion = config.init_obj(loss_module, "criterion")

    trainer = Trainer(model=model,
                      config=config,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      logger=True)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    wandb.init(project="BART-sentiment-analysis")
    warnings.filterwarnings('ignore')
    train(args)
