"""Training entry point for supervised experiments.

This script loads a configuration file describing a model, dataset, and
optimization setup, selects an initialization close to a target class-frequency
bias, and then trains the model with Lightning while logging classwise metrics.
"""

import torch
import torch.nn as nn
from lightning import Trainer
from torchvision.datasets import CIFAR10, CIFAR100,  FashionMNIST
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import numpy as np
import argparse
from src.train_utils import BinarizedDataset
import os
hf_cache_dir = "data/"  # or any scratch directory
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["HF_HOME"] = hf_cache_dir  # optional, covers models too
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
# Make sure the directory exists
os.makedirs(hf_cache_dir, exist_ok=True)

from src.models import MLP, CNN, LargeVIT
from src.vit import VIT
from pytorch_lightning.loggers import CSVLogger
from src.datasets import TinyImageNetDataset
from datasets import load_dataset



def select_init(model, batch, target, eps=1e-1):
    """Reinitialize a model until its initial class-frequency bias is close to a target.

    The script uses the maximum predicted class frequency on a validation-style
    pass as a proxy for initialization bias. Reinitialization continues until
    the distance from the requested target is below ``eps``.
    """
    target_dist = np.inf
    while target_dist > eps:
        model._init()
        loss = model.training_step(batch, 0)
        loss.backward()
        model.validation_step(batch, 0)
        model.on_validation_epoch_end() 
        # compute max classification frequency (proxy for IGB)
        max_freq = max(model.train_freqs.numpy())
        target_dist = abs(max_freq - target)
        print(f"Max class freq: {max_freq}, target: {target}, dist: {target_dist}")

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--spec_file', type=str, help="File containing the specification of the model.")
    
    cfg = vars(parser.parse_args())
    return cfg


def train(args):
    """Build data and model objects from a config file and run training."""
    # get args 
    spec_file = args["spec_file"]
    seed = args["seed"]

    with open(spec_file, 'r') as f:
        cfg = eval(f.read())
    
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    start_dim = -1
    # define flatten
    if cfg["model"] in ["MLP", "RESMLP"]:
        start_dim = 0 # flattent the whole tensor

    
    ### define dataset and dataloaders
    if cfg["data"] == "random":
        n_data_samples = 1000
        n_classes = 2
        data = torch.randn(n_data_samples, 3072, requires_grad=True)
        out_label = torch.randint(0, n_classes, (n_data_samples,), dtype=torch.long)

    elif cfg["data"] == "bfmnist":
        mean = 0.2860
        std = 0.3530
        train_dataset = FashionMNIST(root="data/", train=True, download=True, transform=Compose([ToTensor(), Normalize((mean,), (std,)), torch.nn.Flatten(start_dim)]))
        test_dataset = FashionMNIST(root="data/", train=False, download=True, transform=Compose([ToTensor(), Normalize((mean,), (std,)), torch.nn.Flatten(start_dim)]))
        train_dataset = BinarizedDataset(train_dataset)
        test_dataset = BinarizedDataset(test_dataset)
        in_channels = 1
        fig_size = 28
        input_size = in_channels * fig_size**2
        n_classes = 2

    elif cfg["data"] == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2023, 0.1994, 0.2010]
        train_dataset = CIFAR10(root="data/", train=True, download=True, transform=Compose([ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        test_dataset = CIFAR10(root="data/", train=False, download=True, transform=Compose([ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        fig_size = 32
        input_size = in_channels * fig_size**2
        n_classes = 10

    elif cfg["data"] == "bcifar":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2023, 0.1994, 0.2010]
        train_dataset = CIFAR10(root="data/", train=True, download=True, transform=Compose([ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        test_dataset = CIFAR10(root="data/", train=False, download=True, transform=Compose([ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        train_dataset = BinarizedDataset(train_dataset)
        test_dataset = BinarizedDataset(test_dataset)
        in_channels = 3
        fig_size = 32
        input_size = in_channels * fig_size**2
        n_classes = 2
    
    elif cfg["data"] == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std =  [0.2675, 0.2565, 0.2761]
        train_dataset = CIFAR100(root="data/", train=True, download=True, transform=Compose([Resize(384), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        test_dataset = CIFAR100(root="data/", train=False, download=True, transform=Compose([Resize(384), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        fig_size = 32
        input_size = in_channels * fig_size**2
        n_classes = 100

    elif cfg["data"] == "tinyimagenet":
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        hf_dataset = load_dataset("zh-plus/tiny-imagenet") # from https://huggingface.co/datasets/zh-plus/tiny-imagenet 
        train_dataset = TinyImageNetDataset(hf_dataset["train"], transform=Compose([Resize(384), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        test_dataset = TinyImageNetDataset(hf_dataset["valid"], transform=Compose([Resize(384), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        fig_size = 384
        input_size = in_channels * fig_size**2
        n_classes = 200
    
    # train and test dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size_train"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size_test"], shuffle=False,num_workers=2)
    
    # data, _ = next(iter(train_loader))
    # print(data.shape)
    # exit()
    # define trainer
    logger = CSVLogger("tb_logs", name=spec_file)
    trainer = Trainer(max_steps=cfg["max_steps"],
                      accelerator=cfg["accelerator"],
                      enable_progress_bar=True,
                      val_check_interval=1,
                      limit_val_batches=1.0,
                      log_every_n_steps=1,
                      logger=logger,
                      enable_checkpointing=False,
                      precision="bf16-mixed")
   
    ### single node activation function
    if cfg["act_func"] is not None:
        act = getattr(nn, cfg["act_func"])()
    else:
        act = None

    # select model
    if cfg["model"] == "MLP":
        model = MLP(input_size=input_size,
                    hidd_layer_size=cfg["width"],
                    hidd_layers=cfg["depth"],
                    act=act,
                    act2=None,
                    sigma_w=np.sqrt(cfg["Vw"]),
                    sigma_b=np.sqrt(cfg["Vb"]), 
                    lr=cfg["lr"],
                    n_classes=n_classes,
                    residual=False,
                    init=cfg["init"])
    elif cfg["model"] == "RESMLP":
        model = MLP(input_size=input_size,
                    hidd_layer_size=cfg["width"],
                    hidd_layers=cfg["depth"],
                    act=act,
                    act2=None,
                    sigma_w=np.sqrt(cfg["Vw"]),
                    sigma_b=np.sqrt(cfg["Vb"]), 
                    lr=cfg["lr"],
                    n_classes=n_classes,
                    residual=True,
                    init=cfg["init"])
    elif cfg["model"] == "CNN":
        model = CNN(fig_size=fig_size,
                    in_channels=in_channels,
                    hidd_channels=cfg["width"],
                    hidd_layers=cfg["depth"],
                    act=act,
                    sigma_w=np.sqrt(cfg["Vw"]),
                    sigma_b=np.sqrt(cfg["Vb"]), 
                    lr=cfg["lr"],
                    n_classes=n_classes,
                    init=cfg["init"])
    elif cfg["model"] == "VIT":
        model = VIT(fig_size, cfg["patch_size"], in_channels, cfg["hidden_size"], cfg["num_hidden_layers"],
                 cfg["num_attention_heads"], cfg["qkv_bias"], cfg["intermediate_size"], n_classes, sigma_w=np.sqrt(cfg["Vw"]),
                 sigma_b=np.sqrt(cfg["Vb"]), lr=cfg["lr"], use_faster_attention=cfg["use_faster_attention"], act=act, init=cfg["init"], output_attentions=cfg["output_attentions"], residual=cfg["residual"])
    elif cfg["model"] == "LargeVIT":
        model = LargeVIT(sigma_w=np.sqrt(cfg["Vw"]), sigma_b=np.sqrt(cfg["Vb"]), n_classes=n_classes, save_hooks=False, weight_decay=0.05, init=None, train=True)
    else:
        raise ValueError(f"Model {cfg['model']} not implemented.")
   
    
    print("Device", model.device)
    # fit model
    batch_init = next(iter(train_loader))
    select_init(model, batch_init, target=cfg["target_max_freq"], eps=0.05)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    
if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)


