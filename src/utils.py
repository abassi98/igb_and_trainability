
"""General-purpose utilities for theory code, serialization, and plotting."""

import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pandas as pd 
dtype = np.float64
import argparse

def generate_blue_to_red_shades(n, alpha):
    """Generate a simple blue-to-red RGBA gradient with fixed transparency."""
    return [(i / (n-1), 0, 1 - i / (n-1), alpha) for i in range(n)]

def generate_colour_shades(length, colour):
    """Generate alpha-varied shades from a base Matplotlib-compatible color."""
    # Ensure the length is valid
    if length <= 0:
        return []

    shades = []
    step = 255 // max(1, length )  # Step for variation in shades
    rgb = to_rgb(colour)

    for i in range(length):
        value = (i+1) * step / 256.0  # Incrementally adjust the blue intensity
        shades.append(rgb+(value,))

    return shades
    

### single input activation functions
class Linear:
    """Identity activation used by the NumPy theory code."""
    def __init__(self):
        pass
    def __call__(self, x):
        return x

class ReLU:
    """ReLU activation used by the NumPy theory code."""
    def __init__(self):
        pass
    def __call__(self, x):
        return np.maximum(0.0,x)

class Tanh:
    """Tanh activation used by the NumPy theory code."""
    def __init__(self):
        pass
    def __call__(self, x):
        return np.tanh(x)

class TanhLike:
    """Smooth odd saturating activation used in some forward scans."""
    def __init__(self):
        pass
    def __call__(self, x):
        return np.sign(x)*(1-np.exp(-np.abs(x)))
    
### two inputs acitvation functions
class MaxPool:
    """Pairwise max operator acting on neighboring rows of a NumPy array."""
    def __init__(self):
        pass
    def __call__(self, x):
        # default axis 0
        assert x.shape[0]%2 == 0
        out = np.zeros_like(x)
        for ii in range(x.shape[0]//2):
            out[2*ii, :] = np.maximum(x[2*ii,:], x[2*ii+1,:])
            out[2*ii+1, :] = np.maximum(x[2*ii,:], x[2*ii+1,:])
        return out

class MinPool:
    """Pairwise min operator acting on neighboring rows of a NumPy array."""
    def __init__(self):
        pass
    def __call__(self, x):
        # default axis 0
        assert x.shape[0]%2 == 0
        out = np.zeros_like(x)
        for ii in range(x.shape[0]//2):
            out[2*ii, :] = np.minimum(x[2*ii,:], x[2*ii+1,:])
            out[2*ii+1, :] = np.minimum(x[2*ii,:], x[2*ii+1,:])

        return out

class AveragePool:
    """Pairwise average operator acting on neighboring rows of a NumPy array."""
    def __init__(self):
        pass
    def __call__(self, x):
        # default axis 0
        assert x.shape[0]%2 == 0
        out = np.zeros_like(x)
        for ii in range(x.shape[0]//2):
            out[2*ii, :] = (x[2*ii,:] + x[2*ii+1,:])/2
            out[2*ii+1,:] = (x[2*ii,:] + x[2*ii+1,:])/2
            
        return out


class MLP_Numpy:
    """
    NumPy implementation of a fixed-width random MLP for forward-statistics scans.

    The model is re-sampled layer by layer and records ensemble-style empirical
    statistics across a batch of synthetic inputs.
    """
    def __init__(self,
                 hidd_layer_size,
                 num_layers,
                 act,
                 act2,
                 sigma_w,
                 sigma_b):
        self.hidd_layer_size = hidd_layer_size
        self.num_layers = num_layers
        self.act = act
        self.act2 = act2
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        ### network architecture
        self.weights = np.empty([self.hidd_layer_size,self.hidd_layer_size], dtype=dtype)
        self.biases = np.empty([self.hidd_layer_size, 1], dtype=dtype)
        
    def __call__(self, x):
        """
        x.shape = (width, n_data_samples)
        """
        n_data_samples = x.shape[1]
        self.V = np.empty([self.num_layers], dtype=dtype)
        self.Vtilde = np.empty([self.num_layers], dtype=dtype)
        self.c_dist = np.empty([self.num_layers, n_data_samples], dtype=dtype) #  track total correlation (not averaged over dataset, only over network ensemble)
        self.q_dist = np.empty([self.num_layers, n_data_samples], dtype=dtype) #  track total correlation (not averaged over dataset, only over network ensemble)
        self.cov_dist = np.empty([self.num_layers, n_data_samples], dtype=dtype) #  track total correlation (not averaged over dataset, only over network ensemble)
       
        # compute attention scores with a MLP 
        for ii in range(self.num_layers):
            # x.shape = (width, num_data_samples)
            self.weights[:] = np.random.normal(loc=0.0, scale=self.sigma_w / np.sqrt(self.hidd_layer_size), size=(self.hidd_layer_size,self.hidd_layer_size)) # initialize weights
            self.biases[:] = np.random.normal(loc=0.0, scale=self.sigma_b, size=(self.hidd_layer_size,1)) # initialize biases

            x = np.matmul(self.weights, x) + self.biases # propagate the signal and compute pre-activation
            var_data = np.var(x, axis=1, keepdims=False) # shape (width)
            mean_data = np.mean(x, axis=1,keepdims=False) # shape (width)
            #print(np.var(var_data)/np.mean(var_data)**2) # self-averaging of V
            self.V[ii] = np.mean(var_data, axis=0, keepdims=False)# compute V, initial value = 1.0
            self.Vtilde[ii] = np.var(mean_data, axis=0, keepdims=False) # compute Vtilde, initial value = 0.0

            # correlations
            corr = np.triu(np.corrcoef(x, rowvar=False), k=1) # take upper triangula correlations, excluding the main diagional, putting everything else to zero
            corr = corr[corr != 0] # filter out zeros
            self.c_dist[ii,:] = np.random.choice(corr, size=n_data_samples, replace=False) # save only some samples to save space
            
            # variance
            var = np.var(x, axis=0)
            self.q_dist[ii,:] = var

            # covariance
            cov = np.triu(np.cov(x, rowvar=False), k=1) # take upper triangula correlations, excluding the main diagional, putting everything else to zero
            cov = cov[cov != 0] # filter out zeros
            self.cov_dist[ii,:] = np.random.choice(cov, size=n_data_samples, replace=False)
           

            x = self.act(x) # compute post activation values
            x = self.act2(x)

        return None

def str2bool(v):
    """Convert common textual boolean forms into a Python bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def none_or_type(type_func):
    """Build an argparse converter that accepts either ``None`` or a target type."""
    def converter(value):
        if value.lower() == "none":
            return None
        return type_func(value)
    return converter

def calculate_gain(act_func, act_func2="Linear", Vb=0.1):
    """
    Calculate gain for the activation functions considered.
    Returns the gain for the given activation function and sigma_b.
    If the activation function is not supported, raises a ValueError.
    """
    act_func = act_func.lower()
    act_func2 = act_func2.lower()
    if act_func=="linear":
        return 1.0
    elif act_func=="relu":
        return 2.0
    elif act_func=="tanh":  
        if  act_func2=="maxpool1d":
            df = pd.read_csv("data/eoc_Tanh+MaxPool.csv", index_col=0)  
        elif act_func2=="avgpool1d":
            df = pd.read_csv("data/eoc_Tanh+Averagepool.csv", index_col=0)
        elif act_func2=="linear":
            df = pd.read_csv("data/eoc_Tanh.csv", index_col=0)  
        else:
            raise ValueError(f"Unknown activation function: {act_func2}")

        Vb_vec = df.loc[:,"Vb"]
        Vb_min = Vb_vec.to_numpy()[0]
        Vb_max = Vb_vec.to_numpy()[-1]
        eoc = df.loc[:,"eoc"]
        # check if sigma_b is in the range of eoc
        if Vb*2 < Vb_min or Vb**2 > Vb_max:
            raise ValueError(f"Vb {Vb} is out of range of eoc Tanh: {Vb_min} - {Vb_max}")
        
        return np.interp(Vb, Vb_vec, eoc)
    
    else:
        raise ValueError(f"Unknown activation function: {act_func}")

def save_data(buf_: np.ndarray,
                    file_path: str,
                    cfg: dict,
                    ):
    """Persist a result tensor and its run configuration into an HDF5 file."""

    with h5py.File(file_path, 'w') as out_f:
        buf = out_f.create_dataset(
            'buf',
            shape=buf_.shape,
            chunks=True,
            dtype=np.float64,
            compression='gzip')

        # assign values
        buf[:] = buf_
      
        # buf[0] = V, buf[1] = Vtilde
        # save cfg
        for key, value in cfg.items():
            if value is None:
                out_f.attrs[key] = "None"   # store as string
            else:
                out_f.attrs[key] = value

def read_data(file_path):
    """Load a saved HDF5 tensor together with the configuration attributes."""
    with h5py.File(file_path, 'r') as f:
        buf = np.array(f["buf"])

        cfg = {}
        for key, value in f.attrs.items():  # Reading attributes
            cfg[key] = value
        
    return cfg, buf

dict_phase_color = {
    "ordered_deep_prejudice" : "#9bbcff",
    "edge_of_chaos" : "#a14646",
    "chaotic_deep_prejudice" : "#4861c1",
    "prejudice" : "#6346a1",
    "neutrality" : "#a08bdd",
}
