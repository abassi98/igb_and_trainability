"""MPI forward-statistics simulation for random MLP ensembles.

For each point in a grid of weight and bias variances, the script samples random
MLPs in NumPy, propagates Gaussian inputs through depth, and saves empirical
statistics such as layerwise variance, mean-induced variance, correlation,
variance, and covariance distributions.
"""

import numpy as np
import random
import argparse
import src.utils as ut
import multiprocessing
import time
from mpi4py import MPI
from src.utils import MLP_Numpy as MLP

dtype = np.float64

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--width', type=int, default=1000, help="Number MLP neurons per layer, i.e. width.")
    parser.add_argument('--act_func', type=str, default="ReLU", choices=["Linear", "ReLU", "Tanh"], help="Single point activation function.")
    parser.add_argument('--act_func2', type=str, default="Linear", choices=["Linear", "MaxPool","AveragePool"], help="Two point activation function.")
    parser.add_argument('--Vw_max', type=float, default=3.0)
    parser.add_argument('--Vw_min', type=float, default=1.0)
    parser.add_argument('--Vb_max', type=float, default=0.2)
    parser.add_argument('--Vb_min', type=float, default=0.0)
    parser.add_argument('--n_w', type=int, default=1)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--n_net_samples', type=int, default=1, help="Number of samples of the net ensemble.")
    parser.add_argument('--n_net_processes', type=int, default=1, help="Number processes over which net samples computation is splitted.")
    parser.add_argument('--n_data_samples', type=int, default=100, help="Number of samples of the dataset.")
    parser.add_argument('--seed', type=int, default=42)

    cfg = vars(parser.parse_args())
    return cfg


def process(args):
    """Compute forward statistics for one parameter pair on one MPI worker group."""
    # get args 
    Vb, Vw, n_net_samples_per_process, cfg = args
    width = cfg["width"]
    max_depth = cfg["max_depth"]
    act_func = cfg["act_func"]
    act_func2 = cfg["act_func2"]
    n_data_samples = cfg["n_data_samples"]
    buf = np.zeros((5, max_depth, n_data_samples, n_net_samples_per_process), dtype=dtype) # V, V_tilde the first dimension
    data = np.random.randn(width, n_data_samples)
    
    for kk in range(n_net_samples_per_process):
        model = MLP(hidd_layer_size=width,
                    num_layers=max_depth,
                    act=getattr(ut, act_func)(),
                    act2=getattr(ut, act_func2)(),
                    sigma_w=np.sqrt(Vw),
                    sigma_b=np.sqrt(Vb))

        # compute output
        _ = model(data)
        buf[0,:,0,kk] = model.V
        buf[1,:,0,kk] = model.Vtilde
        buf[2,:,:,kk] = model.c_dist
        buf[3,:,:,kk] = model.q_dist
        buf[4,:,:,kk] = model.cov_dist
       
        
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
n_net_samples = cfg["n_net_samples"]
n_data_samples = cfg["n_data_samples"]


n_w, n_b, n_net_processes = cfg["n_w"], cfg["n_b"], cfg["n_net_processes"]
Vw_min, Vw_max = cfg["Vw_min"], cfg["Vw_max"]
Vb_min, Vb_max = cfg["Vb_min"], cfg["Vb_max"]
Vw_vec = np.linspace(Vw_min,Vw_max, n_w)
Vb_vec = np.linspace(Vb_min,Vb_max,n_b)

# define mpi variables
my_colour = my_rank // n_net_processes # every contigous number of n_net_processes (according to my_rank), i.e. colour, processes same input
sub_comm = comm.Split(my_colour, 0)
my_sub_size = sub_comm.Get_size()
my_sub_rank = sub_comm.Get_rank()


# seed
seed = cfg["seed"]
np.random.seed(seed)
seed = int(np.random.uniform(low=0, high=1e2)) * my_rank +1
random.seed(seed)
np.random.seed(seed)

# check number of cpus
if my_size < n_b * n_w * n_net_processes:
    raise ValueError(f"Number of MPI processes ({multiprocessing.cpu_count()}) is too low. At least {n_b * n_w * n_net_processes} necessary")


# check the number of net processes fed to each process
if n_net_samples%n_net_processes != 0:
    raise ValueError("The number of net sample processes must be divisor of the total number of net samples.")

# define number of net samples per process
n_net_samples_per_process = n_net_samples // n_net_processes

# get ids corresponding to the rank
id_vb = my_colour%n_b
id_vw = (my_colour - id_vb)//n_b
Vb = Vb_vec[id_vb]
Vw = Vw_vec[id_vw]

# print id message
print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I am computing Vb={Vb}, Vw={Vw} and width = {width}")

# run computation
buf = process((Vb, Vw, n_net_samples_per_process, cfg)) # buf.shape = (2, max_depth, width, n_net_samples_per_process)

if my_colour == 0:
   
    # compute local sums
    local_sum = np.sum(buf, axis=(3), keepdims=False)
    global_sum = np.zeros((5, max_depth, n_data_samples), dtype=dtype)
    sub_comm.Reduce(local_sum, (global_sum, 5*max_depth*n_data_samples, MPI.DOUBLE), op=MPI.SUM, root=0)
    
    # aggregation of the result for each sub communicator
    if my_sub_rank==0: # colour 0 process 0 (i.e. global 0)
        # save 0th buffer
        save_buf = np.zeros((5, max_depth,n_data_samples, n_b,n_w), dtype=dtype)
        save_buf[:,:,:,0, 0] =  global_sum / (n_net_samples) 
        # save receive other buffers
        for sub_colour in range(1,n_b * n_w):
            id_vb = sub_colour%n_b
            id_vw = (sub_colour - id_vb)//n_b
            rcv_buf = np.zeros((5, max_depth, n_data_samples), dtype=dtype)
            comm.Recv([rcv_buf, 5*max_depth*n_data_samples, MPI.DOUBLE], source=sub_colour*n_net_processes, tag=sub_colour) 
            save_buf[:,:,:,id_vb, id_vw] = rcv_buf

        # save data
        if act_func2 == "Linear":
            ut.save_data(save_buf, f"data/{act_func}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)
        else:
            ut.save_data(save_buf, f"data/{act_func}+{act_func2}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)

elif my_colour >0 and my_colour < n_b*n_w:
    # compute local sums
    local_sum = np.sum(buf, axis=(3), keepdims=False)
    # aggregate
    global_sum = np.zeros((5, max_depth, n_data_samples), dtype=dtype)
    sub_comm.Reduce(local_sum, (global_sum, 5*max_depth*n_data_samples, MPI.DOUBLE), op=MPI.SUM, root=0)
    global_mean = global_sum / (n_net_samples) 

    # aggregation of the result for each sub communicator
    if my_sub_rank==0:
        comm.Send([global_mean,5*max_depth*n_data_samples, MPI.DOUBLE], dest=0, tag=my_colour)

else:
    print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I should not even exist.")
