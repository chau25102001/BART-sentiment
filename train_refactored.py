import argparse

import termcolor
import torch

import models as model_module
import datasets as dataset_module
import models.loss as loss_module

from trainer.trainer import Trainer
from torch.nn import DataParallel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
# import wandb
from config_parser import ConfigParser
from utils.utils import text_collate

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=False, action="store-true", help="resume training?")
parser.add_argument("--config", default="./config/demo_config.json", type=str, help="path to config file")
args = parser.parse_args()

def train(args):
    config = ConfigParser(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = config.init_obj(model_module, "arch")
    model = DataParallel(model).to(device)

    print(f"Model: {config['name']}, num params: {sum(p.numel() for p in model.parameters())}")


    train_set = config.init_obj(dataset_module, "train_set")
    test_set = config.init_obj(dataset_module, "test_set")
    
    tokenizer = model.tokenizer
    collate_fn = lambda batch: text_collate(batch, tokenizer, device)
    train_loader = config.init_obj(torch.utils.data, "train_loader")(dataset=train_set, collate_fn=collate_fn)
    test_loader = config.init_obj(torch.utils.data, "train_loader")(dataset=test_set, collate_fn=collate_fn)
    
    optimizer = config.init_obj(torch.optim, "optimizer")(model.parameters())

    #TODO: lr scheduler
    if config['lr_scheduler'] is not None:
        if config['lr_scheduler'] == 'cosine':  # cosine decrease lr
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=config["epoch"] * len(train_loader),
                                                eta_min=1e-6)
        elif config['lr_scheduler'] == 'linear_warm_up':  # warm up from 0 to init lr for 1 epoch, linear decrease for the rest
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                            num_training_steps=len(train_loader) * config['epoch'])
        else:
            print(termcolor.colored("Warning: not supported lr scheduler, using default: None", 'red'))
    
    criterion = config.init_obj(loss_module, "criterion")

    trainer = Trainer(model=model,
                      config = config,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      logger=False)
    # trainer.train(resume=args.resume)

if __name__ == "__main__":
    import warnings

    # wandb.init(project="BART-sentiment-analysis")
    # warnings.filterwarnings('ignore')
    train(args)
