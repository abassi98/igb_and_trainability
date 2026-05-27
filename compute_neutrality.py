import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as np
from scipy.integrate import quad, dblquad, nquad
from autograd import grad
import src.utils as ut
from scipy.special import erf
import pandas as pd
import argparse
from mpi4py import MPI

dtype = np.float32


def update_variance(q, Vw, Vb, act_func):
    fnum = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*act_func(np.sqrt(q)*z)**2
    I, err = quad(fnum, -np.inf, np.inf)
    return Vb + Vw*I

def update_covariance(c, q, Vw, Vb, act_func):
    fnum = lambda z1, z2: 1/(2*np.pi)*np.exp(-z1**2/2-z2**2/2)*act_func(np.sqrt(q)*z1)*act_func(np.sqrt(q)*(c*z1+np.sqrt(1.0-c**2)*z2))
    I, err = dblquad(fnum, -np.inf, np.inf,-np.inf, np.inf)
    return Vb + Vw*I


def update_variance_2nodes(q, Vw, Vb, act_func, act_func2):
    f = lambda z1, z2  : 1/(2*np.pi)*np.exp(-z1**2/2-z2**2/2)*act_func2(act_func(np.sqrt(q)*z1), act_func(np.sqrt(q)*z2))**2
    I, err = dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
    return Vw*I + Vb

# def update_covariance_2nodes(c, q, Vw, Vb, act_func, act_func2):
#     f = lambda z1, z2, z1p, z2p : 1/(2*np.pi)**2*np.exp(-z1**2/2-z2**2/2-z1p**2/2-z2p**2/2)*act_func2(act_func(np.sqrt(q)*z1), act_func(np.sqrt(q)*z2))*act_func2(act_func(np.sqrt(q)*(c*z1+np.sqrt(1-c**2)*z1p)), act_func(np.sqrt(q)*(c*z2+np.sqrt(1-c**2)*z2p)))
#     ranges = [[-np.inf, np.inf]]*4
#     I, err = nquad(f, ranges)
#     return Vw*I + Vb

def update_covariance_2nodes(c, q, Vw, Vb, act_func, act_func2, N=100000):
    # Sample from standard normal
    z1 = np.random.randn(N)
    z2 = np.random.randn(N)
    z1p = np.random.randn(N)
    z2p = np.random.randn(N)

    # Precompute constants
    sq_q = np.sqrt(q)
    sq_1mc2 = np.sqrt(1 - c**2)

    # Compute arguments
    x1 = act_func(sq_q * z1)
    x2 = act_func(sq_q * z2)
    x1p = act_func(sq_q * (c * z1 + sq_1mc2 * z1p))
    x2p = act_func(sq_q * (c * z2 + sq_1mc2 * z2p))

    # Evaluate integrand
    values = act_func2(x1, x2) * act_func2(x1p, x2p)

    # Monte Carlo estimate (no PDF needed since we're sampling from it)
    I = np.mean(values)
    return Vw * I + Vb

class Linear:
    def __init__(self):
        pass
    def __call__(self, x):
        return x

class MaxPool:
    def __init__(self):
        pass
    def __call__(self, x, y):
        return np.maximum(x,y)

class AveragePool:
    def __init__(self):
        pass
    def __call__(self, x, y):
        return (x+y)/2



def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_func', type=str, default="Tanh", choices=["Tanh"], help="Activation function.")
    parser.add_argument('--act_func2', type=str, default="Linear", choices=["Linear", "MaxPool", "AveragePool"], help="Two point activation function.")
    parser.add_argument('--Vw_max', type=float, default=4.0)
    parser.add_argument('--dVw', type=float, default=0.1)
    parser.add_argument('--n_proc', type=int, default=1, help="Number of processes")
    parser.add_argument('--n_iter', type=int, default=100, help="Max number of recursive iteration")

    cfg = vars(parser.parse_args())
    return cfg

def process(Vb_vec, eoc_vec, Vw_max, dVw, n_iter, act_func_, act_func2_):
    assert len(Vb_vec)==len(eoc_vec)
    assert Vw_max >= np.max(eoc_vec)
    buf = np.zeros_like(Vb_vec, dtype=dtype)
    act_func = getattr(ut, act_func_)()
    act_func2 = globals()[act_func2_]()  

    ii = 0
    neutral = eoc_vec[0] # for sure neutral is above eoc, assuming eoc increasing
    for Vb in Vb_vec:
        q=1.0
        cov = 0.01
        c = cov/q
        c_new = c
        q_new = q
        while neutral <= Vw_max:
            for kk in range(n_iter):
                if act_func2_=="Linear":
                    cov_new = update_covariance(c, q, neutral, Vb, act_func)
                    q_new = update_variance(q, neutral, Vb, act_func)
                else:
                    cov_new = update_covariance_2nodes(c, q, neutral, Vb, act_func, act_func2)
                    q_new = update_variance_2nodes(q, neutral, Vb, act_func, act_func2)
                
                c_new = np.clip(cov_new/q_new, -1,1)
                if np.absolute(q-q_new)<=1e-8 and np.absolute(c-c_new)<=1e-8: # until convergence
                    q = q_new
                    c = c_new
                    break
                else:
                    q = q_new
                    c = c_new
            # stop if c becomes lower than 0.5 (at eoc, c=1 by definition)
            if c<=0.5:
                break
            
            neutral += dVw # increase neutral
        
        buf[ii] = neutral
        ii += 1
    
    return buf

# def common world
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

# program specifics
cfg = get_args()
act_func = cfg["act_func"]
act_func2 = cfg["act_func2"]
Vw_max, dVw = cfg["Vw_max"], cfg["dVw"]
n_proc, n_iter = cfg["n_proc"],  cfg["n_iter"]

if act_func2=="Linear":
    eoc_vec = np.array(pd.read_csv(f"data/eoc_{act_func}.csv", index_col=0).loc[:,"eoc"])
    Vb_vec = np.array(pd.read_csv(f"data/eoc_{act_func}.csv", index_col=0).loc[:,"Vb"])
else:
    eoc_vec = np.array(pd.read_csv(f"data/eoc_{act_func}+{act_func2}.csv", index_col=0).loc[:,"eoc"])
    Vb_vec = np.array(pd.read_csv(f"data/eoc_{act_func}+{act_func2}.csv", index_col=0).loc[:,"Vb"])

Vb_points = len(Vb_vec)

assert Vb_points % n_proc == 0 # equal number of points for each process
Vb_points_proc = Vb_points // n_proc

# compute for every rank
Vb_vec_proc = Vb_vec[my_rank*Vb_points_proc:(my_rank+1)*Vb_points_proc]
Vb_min_proc, Vb_max_proc = np.min(Vb_vec_proc), np.max(Vb_vec_proc)
print(f"Hello from rank {my_rank} on {MPI.Get_processor_name()}. I am computing chunk from Vb={Vb_min_proc} to Vb={Vb_max_proc}")
eoc_vec_proc = eoc_vec[my_rank*Vb_points_proc:(my_rank+1)*Vb_points_proc]
buf = process(Vb_vec_proc, eoc_vec_proc, Vw_max, dVw, n_iter, act_func, act_func2)

if my_rank==0: # rank 0 save
    save_buf = np.zeros_like(Vb_vec, dtype=dtype)
    save_buf[0:Vb_points_proc] = buf # save zeroth buffer data

    # save receive other buffers
    for sub_rank in range(1,n_proc):
        rcv_buf = np.zeros_like(Vb_vec_proc, dtype=dtype)
        comm.Recv([rcv_buf, Vb_points_proc, MPI.FLOAT], source=sub_rank, tag=sub_rank) 
        save_buf[sub_rank*Vb_points_proc:(sub_rank+1)*Vb_points_proc] = rcv_buf

    # create pandas dataframe
    df = pd.DataFrame(data={"Vb":Vb_vec, "neutrality":save_buf})
    
    # save data
    if act_func2=="Linear":
        df.to_csv(f"data/neutrality_{act_func}.csv")
    else:
        df.to_csv(f"data/neutrality_{act_func}+{act_func2}.csv")

elif my_rank > 0:
    # send results to 0
    comm.Send([buf,Vb_points_proc, MPI.FLOAT], dest=0, tag=my_rank)