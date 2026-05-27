"""Empirical initialization scan for gradient and class-bias statistics.

The script evaluates randomly initialized models on real datasets, performs a
single backward pass over batches, aggregates per-layer gradient statistics, and
stores the results as HDF5 tensors for later plotting and comparison with
theoretical predictions.
"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from src.datasets import TinyImageNetDataset
import os
os.environ["HF_DATASETS_CACHE"] = "./data/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "./data/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "./data/huggingface_cache"
from datasets import load_dataset

import numpy as np
import random
import argparse
import src.utils as ut
import multiprocessing
import time
from mpi4py import MPI
from src.models import MLP, CNN, LargeVIT, RESNET, MLPMIXER
from src.vit import VIT
from src.train_utils import BinarizedDataset
from src.utils import str2bool, none_or_type
dtype = np.float32
torch.set_default_dtype(torch.float32)
import tqdm



def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="MLP", choices=["MLP", "RESMLP", "CNN", "VIT", "LargeVIT", "RESNET", "MLPMIXER"], help="Model to use.")
    parser.add_argument('--max_depth', type=none_or_type(int), default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--width', type=none_or_type(int), default=1000, help="Number MLP neurons per layer, i.e. width.")
    parser.add_argument('--act_func', type=none_or_type(str), default="ReLU",  help="Activation function.")
    parser.add_argument('--act_func2', type=none_or_type(str), default="Linear",  help="Activation function.")
    parser.add_argument('--Vw_max', type=float, default=2.0)
    parser.add_argument('--Vw_min', type=float, default=0.1)
    parser.add_argument('--Vb_max', type=float, default=0.2)
    parser.add_argument('--Vb_min', type=float, default=0.0)
    parser.add_argument('--n_w', type=int, default=3)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--n_data_samples', type=int, default=100, help="Number of samples of the dataset.")
    parser.add_argument('--batch_size', type=int, default=100, help="Number of samples of the dataset.")
    parser.add_argument('--data', type=str, default="bfmnist", choices=["random", "bfmnist", "cifar10", "bcifar", "cifar100", "tinyimagenet"])
    parser.add_argument('--use_faster_attention', type=str2bool, default=True, help="Use faster attention for VIT model.")
    parser.add_argument('--debug_mode',type=str2bool,  default=False, help="Run in debug mode, i.e. with less samples and less processes.")
    parser.add_argument('--init', type=none_or_type(str), default="Gaussian")
    parser.add_argument('--seed', type=int, default=42)

    cfg = vars(parser.parse_args())
    return cfg


def process(args):
    """Run one initialization measurement for a specific ``(Vb, Vw)`` pair.

    Returns a tensor containing global and per-class summaries of the samplewise
    gradient norms recorded across registered layers.
    """
    # get args 
    Vb, Vw, cfg = args
    width = cfg["width"]
    max_depth = cfg["max_depth"]
    act_func = cfg["act_func"]
    act_func2 = cfg["act_func2"]
   
    start_dim = -1
    # define flatten
    if cfg["model"] in ["MLP", "RESMLP"]:
        start_dim = 0 # flatten the whole tensor

    if cfg["data"] == "random":
        n_classes = 2
        input_size = 3072
        data = torch.randn(cfg["n_data_samples"], input_size, requires_grad=True)
        out_label = torch.randint(0, n_classes, (cfg["n_data_samples"],), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data, out_label)
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    
    elif cfg["data"] == "bfmnist":
        mean = [0.2860]
        std = [0.3530]
        fig_size = 28
        dataset = FashionMNIST(root="data/", train=True, download=True, transform=Compose([Resize(fig_size),ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        dataset = BinarizedDataset(dataset)
        in_channels = 1
        input_size = in_channels * fig_size**2
        n_classes = 2
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
       
    elif cfg["data"] == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2023, 0.1994, 0.2010]
        fig_size = 384
        dataset = CIFAR10(root="data/", train=True, download=True, transform=Compose([Resize(fig_size),ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        input_size = in_channels * fig_size**2
        n_classes = 10
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
        
    elif cfg["data"] == "bcifar":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2023, 0.1994, 0.2010]
        fig_size = 32
        dataset = CIFAR10(root="data/", train=True, download=True, transform=Compose([Resize(fig_size),ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        dataset = BinarizedDataset(dataset)
        in_channels = 3
        input_size = in_channels * fig_size**2
        n_classes = 2
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
       

    elif cfg["data"] == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std =  [0.2675, 0.2565, 0.2761]
        fig_size = 384
        dataset = CIFAR100(root="data/", train=True, download=True, transform=Compose([Resize(fig_size), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        input_size = in_channels * fig_size**2
        n_classes = 100
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
        # Normalize(mean, std)

    elif cfg["data"] == "tinyimagenet":
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        fig_size = 384
        hf_dataset = load_dataset("zh-plus/tiny-imagenet", keep_in_memory=True) # from https://huggingface.co/datasets/zh-plus/tiny-imagenet 
        dataset = TinyImageNetDataset(hf_dataset["train"], transform=Compose([Resize(fig_size), ToTensor(), Normalize(mean, std), torch.nn.Flatten(start_dim)]))
        in_channels = 3
        input_size = in_channels * fig_size**2
        n_classes = 200
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
      
    # first activation function
    if act_func is not None:
        act = getattr(nn, act_func)()
    else:
        act = None
    
    ### second activation function
    if act_func2 is not None and not "Linear":
        act2 = getattr(nn, act_func2)(kernel_size=2, stride=2)
    else:
        act2 = None
    
    # create model
    if cfg["model"] == "MLP":
        model = MLP(input_size=input_size,
                hidd_layer_size=width,
                hidd_layers=max_depth,
                act=act,
                act2=act2,
                sigma_w=np.sqrt(Vw),
                sigma_b=np.sqrt(Vb),
                n_classes=n_classes,
                residual=False,
                init=cfg["init"])
    elif cfg["model"] == "RESMLP":
        model = MLP(input_size=input_size,
                hidd_layer_size=width,
                hidd_layers=max_depth,
                act=act,
                act2=act2,
                sigma_w=np.sqrt(Vw),
                sigma_b=np.sqrt(Vb),
                n_classes=n_classes,
                residual=True,
                init=cfg["init"])
    elif cfg["model"]== "CNN":      
        model = CNN(fig_size=fig_size,
                in_channels=in_channels,
                hidd_channels=width,
                hidd_layers=max_depth,
                act=act,
                sigma_w=np.sqrt(Vw),
                sigma_b=np.sqrt(Vb),
                n_classes=n_classes,
                init=cfg["init"])
    elif cfg["model"] == "VIT":
        model = VIT(fig_size, 4, in_channels, width, max_depth,
                num_attention_heads=1, qkv_bias=True, intermediate_size=width, n_classes=n_classes, sigma_w=np.sqrt(Vw),
                sigma_b=np.sqrt(Vb), use_faster_attention=cfg["use_faster_attention"], act=act, init=cfg["init"])
    elif cfg["model"] == "LargeVIT":
        model = LargeVIT(sigma_w=np.sqrt(Vw), sigma_b=np.sqrt(Vb), n_classes=n_classes, init=cfg["init"], save_hooks=True, train=False)
    elif cfg["model"] == "RESNET":
        model = RESNET(sigma_w=np.sqrt(Vw), sigma_b=np.sqrt(Vb), n_classes=n_classes, init=cfg["init"], save_hooks=True, train=False)
    elif cfg["model"] == "MLPMIXER":
        model = MLPMIXER(sigma_w=np.sqrt(Vw), sigma_b=np.sqrt(Vb), n_classes=n_classes, init=cfg["init"], save_hooks=True, train=False)
    else:
        raise ValueError(f"Model {cfg['model']} not implemented.")

    # debug
    if cfg["debug_mode"]:
        for nl in range(model.registered_layers):
            print(nl+1, model.registered_names[nl])
        exit()
        # print(model)
        # print("Device", model.device)
        # #run training step
        # num_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters: {num_params}")


    n_data_processed = 0
    grads = []
    #igb_metric = []
    #c_metric = []
    out_label = []
    assert cfg["n_data_samples"] >= cfg["batch_size"] and cfg["n_data_samples"] % cfg["batch_size"]==0 # sanity check
    for nb, batch in tqdm.tqdm(enumerate(dataloader)):
        x, y = batch
        n_data_processed += x.shape[0] # add batch size
        if n_data_processed > cfg["n_data_samples"]:
            break
        else:
            print(f"Processing batch {nb}")
            out_label.append(y)
            loss = model.training_step(batch, nb)
            loss.backward()
            #model.validation_step(batch, nb)
    
            # retrieve grads, correlation and IGB measures per batch
            grads_batch = np.zeros([len(model.batch_grad_storage.items()), cfg["batch_size"]], dtype=dtype)
            #c_batch = np.zeros([len(model.batch_preact_storage), cfg["batch_size"] * (cfg["batch_size"]-1)//2, 1], dtype=dtype)
            #igb_batch = np.zeros([len(model.batch_preact_storage), 2, 1], dtype=dtype)
            # retrieve grads and pre-activations
            for nl, (layer_name, grads_l) in enumerate(reversed(model.batch_grad_storage.items())):
                #pa = model.batch_preact_storage[layer_name].cpu().numpy() # shape (batch_size, width)
                # correlation coefficient
                #c = np.corrcoef(pa)
                #c = np.triu(c, k=1).flatten()
                #c = c[c!=0]
                #c_batch[nl, :, 0] = c
                # gradients
                grads_batch[nl, :] = grads_l.cpu().numpy() * x.shape[0]**2
                # IGB measures
                # compute V and Vtilde
                #mean_var_y = np.mean(np.var(pa, axis=0)) # variance of pre-activation w.r.t dataset
                #mu_y = np.mean(pa, axis=0)
                #var_mu_y = np.var(mu_y) 
                #igb_batch[nl, 0, 0] = mean_var_y
                #igb_batch[nl, 1, 0] = var_mu_y
                
               
            # appendd
            grads.append(grads_batch)
            #c_metric.append(c_batch)
            #igb_metric.append(igb_batch)
    
 
    torch.cuda.empty_cache()
    model.on_validation_epoch_end() 
    # compute max classification frequency (proxy for IGB)
    print("Max frequency: ", max(model.train_freqs.numpy()))
    sort_classes = np.argsort(model.train_freqs.numpy())[::-1]

    # stack
    grads = np.concatenate(grads, axis=1)
    out_label = np.concatenate(out_label)
    #igb_metric = np.concatenate(igb_metric, axis=2)  # shape (n_layers, 2, n_batches)
    #c_metric = np.concatenate(c_metric, axis=1)  # shape (n_layers, n_pairs, n_batches)
    #mean_igb_batch = np.mean(igb_metric, axis=2)  
    #mean_c_batch = np.mean(c_metric, axis=2)
   
    # remove last layer
    grads = grads[:-1,:] 
    #mean_igb_batch =  mean_igb_batch[:-1, :] 
    #mean_c_batch =  mean_c_batch[:-1,:]
    
    buf = np.zeros((14, n_classes+1, len(model.batch_grad_storage.items())-1), dtype=dtype) 
    buf[:] = np.nan

    ### Gradients 
    buf[0,0,:] = np.quantile(grads, 0.05, axis=1, keepdims=False) # 5 % quantile
    buf[1,0,:] = np.quantile(grads, 0.25, axis=1, keepdims=False) # 25 % quantile
    buf[2,0,:] = np.quantile(grads, 0.5, axis=1, keepdims=False) # median
    buf[3,0,:] = np.quantile(grads, 0.75, axis=1, keepdims=False) # 75% quantile
    buf[4,0,:] =  np.quantile(grads, 0.95, axis=1, keepdims=False) # 95 % quantile
    buf[5,0,:] =  np.mean(grads, axis=1, keepdims=False) # mean

    # IGB metrics
    #buf[6,0,:] = mean_igb_batch[:,0]  # mean_var_y 
    #buf[7,0,:] = mean_igb_batch[:,1]  # var_mu_y

    # corr coefficients statistics
        ### Gradients 
    #buf[8,0,:] = np.quantile(mean_c_batch, 0.05, axis=1, keepdims=False) # 5 % quantile
    #buf[9,0,:] = np.quantile(mean_c_batch, 0.25, axis=1, keepdims=False) # 25 % quantile
    #buf[10,0,:] = np.quantile(mean_c_batch, 0.5, axis=1, keepdims=False) # median
    #buf[11,0,:] = np.quantile(mean_c_batch, 0.75, axis=1, keepdims=False) # 75% quantile
    #buf[12,0,:] =  np.quantile(mean_c_batch, 0.95, axis=1, keepdims=False) # 95 % quantile
    #buf[13,0,:] =  np.mean(mean_c_batch, axis=1, keepdims=False) # mean

    # per-class
    for ind, c in enumerate(sort_classes):
        mask = out_label==c
        if mask.any() != False:
            buf[0,ind+1,:] = np.quantile(grads[:,mask], 0.05, axis=1, keepdims=False) # median per class
            buf[1,ind+1,:] = np.quantile(grads[:,mask], 0.25, axis=1, keepdims=False) # median per class
            buf[2,ind+1,:] = np.quantile(grads[:,mask], 0.5, axis=1, keepdims=False) # median per class
            buf[3,ind+1,:] = np.quantile(grads[:,mask], 0.75, axis=1, keepdims=False) # median per class
            buf[4,ind+1,:] = np.quantile(grads[:,mask], 0.95, axis=1, keepdims=False) # median per class
            buf[5,ind+1,:] = np.mean(grads[:,mask], axis=1, keepdims=False) # mean per class
            buf[6,ind+1,:] = np.std(grads[:,mask], axis=1, keepdims=False) # std per class
            
    return buf

# def common world
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

# program specifics
cfg = get_args()
max_depth = cfg["max_depth"]
width = cfg["width"]
act_func = cfg["act_func"]
act_func2 = cfg["act_func2"]

if cfg["data"] == "random" or cfg["data"] == "bfmnist" or cfg["data"] == "bfcifar":
    n_classes = 2
elif cfg["data"] == "cifar10":
    n_classes = 10
elif cfg["data"] == "tinyimagenet":
    n_classes = 200
elif cfg["data"] == "cifar100":
    n_classes = 100

# model hyper-parameters
n_w, n_b = cfg["n_w"], cfg["n_b"]
Vw_min, Vw_max = cfg["Vw_min"], cfg["Vw_max"]
Vb_min, Vb_max = cfg["Vb_min"], cfg["Vb_max"]
Vw_vec = np.linspace(Vw_min,Vw_max, n_w)
Vb_vec = np.linspace(Vb_min,Vb_max,n_b)
# select effective depth of the transfoormer
effective_depth = max_depth
if cfg["model"] == "VIT":
    if cfg["use_faster_attention"]:
        effective_depth = 4 * max_depth + 1 if cfg["model"] == "VIT" else max_depth
    else:
        effective_depth = 6 * max_depth + 1 if cfg["model"] == "VIT" else max_depth


# seed
seed = cfg["seed"] + 1000*my_rank
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
#seed = int(np.random.uniform(low=0, high=1e2)) * my_rank +1
random.seed(seed)
np.random.seed(seed)

### SEQUENTIAL MODE HANDLING ###
if my_size == 1:
    print("Running in SEQUENTIAL mode (no mpirun detected).")
    all_results = np.zeros((14, n_classes+1, 1, n_b, n_w), dtype=dtype)

    for id_vb, Vb in enumerate(Vb_vec):
        for id_vw, Vw in enumerate(Vw_vec):
            print(f"Hello! I am computing Vb={Vb}, Vw={Vw} and width = {width}")
            buf = process((Vb, Vw, cfg))  # run all samples on same CPU
            effective_depth = buf.shape[2]
            if id_vb==0 and id_vw==0:
                all_results =  np.repeat(all_results, effective_depth, axis=2) 
            all_results[:, :, :, id_vb, id_vw] = buf

    if act_func2 == "Linear":
        ut.save_data(all_results, f"data/init/{cfg['model']}_{cfg['data']}_{act_func}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)
    else:
        ut.save_data(all_results, f"data/init/{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{cfg['max_depth']}_W{cfg['width']}.h5",cfg)
    exit(0)  # End program after sequential run


# ### MPI CODE HANDLING ###

# # check number of cpus
# if my_size < n_b * n_w :
#     raise ValueError(f"Number of MPI processes ({multiprocessing.cpu_count()}) is too low. At least {n_b * n_w } necessary")


# # get ids corresponding to the rank
# id_vb = my_colour%n_b
# id_vw = (my_colour - id_vb)//n_b
# Vb = Vb_vec[id_vb]
# Vw = Vw_vec[id_vw]

# print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}! I am computing Vb={Vb}, Vw={Vw} and width = {width}")

# # run computation
# buf = process((Vb, Vw, cfg)) # buf.shape = (8,  n_classes, max_depth)
# effective_depth = buf.shape[2]

# if my_colour == 0:
#     # compute local sums
#     local_sum = np.sum(buf, axis=(-1), keepdims=False)
#     global_sum = np.zeros((8,  n_classes+1, effective_depth), dtype=dtype)
#     sub_comm.Reduce(local_sum, (global_sum, 8*(n_classes+1)*effective_depth, MPI.FLOAT), op=MPI.SUM, root=0)
    
#     # aggregation of the result for each sub communicator
#     if my_sub_rank==0: # colour 0 process 0 (i.e. global 0)
#         # save 0th buffer
#         save_buf = np.zeros((8,  n_classes+1, effective_depth, n_b,n_w), dtype=dtype)
#         save_buf[:,:,:,0, 0] =  global_sum / (n_net_samples) 
#         # save receive other buffers
#         for sub_colour in range(1,n_b * n_w):
#             id_vb = sub_colour%n_b
#             id_vw = (sub_colour - id_vb)//n_b
#             rcv_buf = np.zeros((8,  n_classes+1, effective_depth), dtype=dtype)
#             comm.Recv([rcv_buf, 8*(n_classes+1)*effective_depth,MPI.FLOAT], source=sub_colour, tag=sub_colour) 
#             save_buf[:, :, :, id_vb, id_vw] = rcv_buf

#         # save data
#         if act_func2=="Linear":
#             ut.save_data(save_buf, f"data/init/{cfg['model']}_{cfg['data']}_{act_func}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)
#         else:
#             ut.save_data(save_buf, f"data/init/{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)

# elif my_colour >0 and my_colour < n_b*n_w:
#     # compute local sums
#     local_sum = np.sum(buf, axis=(-1), keepdims=False)
#     # aggregate
#     global_sum = np.zeros((8,  n_classes+1, effective_depth), dtype=dtype)
#     sub_comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
#     global_mean = global_sum / (n_net_samples) 

#     # aggregation of the result for each sub communicator
#     if my_sub_rank==0:
#         comm.Send([global_mean,8*(n_classes+1)*effective_depth,MPI.FLOAT], dest=0, tag=my_colour)

# else:
#     print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I should not even exist.")
