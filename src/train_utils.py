
"""Training utilities for dataset binarization, metrics, and initialization."""

import torch
from torch.utils.data import Dataset
import numpy as np
import math
import torch.nn.init as init

### two inputs acitvation functions
class MaxPool:
    """NumPy max-pooling helper operating on paired rows."""
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
    """NumPy min-pooling helper operating on paired rows."""
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
    """NumPy average-pooling helper operating on paired rows."""
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
    


class BinarizedDataset(Dataset):
    """Wrap a multiclass dataset and map labels to even/odd parity classes."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        binary_label = label % 2  # 0 for even, 1 for odd
        return image, binary_label
    
    def __len__(self):
        return len(self.dataset)
    
class ElementsPerClass():
    """Track the number of observed samples for each class."""
    def __init__(self, n_classes, device='cpu'):
        self.n_classes = n_classes
        self.device = device
        self.fc = torch.zeros((n_classes,), device=self.device)
        self.n_samples = 0

    def update(self, y):
        """
        Update the g0 vector with the maximum value for each class.
        x_hat: tensor of shape (batch_size, n_classes)
        """
        self.n_samples += y.shape[0]
        # update g0 with the maximum value for each class
        self.fc += torch.bincount(y, minlength=self.n_classes).to(self.device)
    
    def compute(self):
        """
        Compute the g0 vector as the fraction of points classified to each class.
        Returns:
            g0: tensor of shape (n_classes)
        """
        
        return self.fc 
    
    def reset(self):
        self.fc = torch.zeros((self.n_classes,), device=self.device)
        self.n_samples = 0  

class MultiClassFreq():
    """Track predicted class frequencies from model logits."""
    def __init__(self, n_classes, device='cpu'):
        self.n_classes = n_classes
        self.device = device
        self.fc = torch.zeros((n_classes,), device=self.device)
        self.n_samples = 0

    def update(self, x_hat):
        """
        Update the g0 vector with the maximum value for each class.
        x_hat: tensor of shape (batch_size, n_classes)
        """
        self.n_samples += x_hat.shape[0]
        
        # update g0 with the maximum value for each class
        self.fc += torch.bincount(torch.argmax(x_hat, dim=-1), minlength=self.n_classes).to(self.device)
    
    def compute(self):
        """
        Compute the g0 vector as the fraction of points classified to each class.
        Returns:
            g0: tensor of shape (n_classes)
        """
        return self.fc / self.n_samples if self.n_samples > 0 else torch.zeros(self.n_classes, device=self.device)
    
    def reset(self):
        self.fc = torch.zeros((self.n_classes,), device=self.device)
        self.n_samples = 0  

class MultiClassLoss():
    """Accumulate unreduced losses and report classwise averages."""
    def __init__(self, n_classes, device='cpu'):
        self.n_classes = n_classes
        self.device = device
        self.losses = torch.zeros((n_classes,), device=self.device)
        self.n_samples = 0

    def update(self, losses, y):
        """
        Update the losses for each class.
        loss: tensor of shape (batch_size)
        y: tensor of shape (batch_size)
        """
        self.n_samples += y.shape[0]
        for c in range(self.n_classes):
            class_mask = (y == c)
            class_loss = losses[class_mask].sum() if class_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            self.losses[c] += class_loss.to(self.device)

    def compute(self):
        """
        Compute the losses for each class.
        Returns:
            losses: tensor of shape (n_classes)
        """
        return self.losses / self.n_samples if self.n_samples > 0 else torch.zeros(self.n_classes, device=self.device)
    
    def reset(self):
        self.losses = torch.zeros((self.n_classes,), device=self.device)
        self.n_samples = 0  


def delta_orthogonal_(weight, gain=1.0):
    """Apply delta-orthogonal initialization to a convolution kernel.

    Parameters
    ----------
    weight:
        Tensor of shape ``(out_channels, in_channels, kH, kW)``.
    gain:
        Multiplicative scaling factor, typically chosen from the target
        activation's variance-preserving initialization rule.
    """
    o, i, kH, kW = weight.shape
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError("Delta-orthogonal init requires odd kernel size.")

    with torch.no_grad():
        weight.zero_()
        # Create orthogonal matrix for channels
        central = torch.empty(o, i)
        init.orthogonal_(central)
        # Scale variance like in He/Xavier initialization
        fan_in = i  # only 1 spatial position active
        scale = gain / math.sqrt(fan_in)
        central *= scale
        # Place orthogonal matrix at the center
        weight[:, :, kH // 2, kW // 2] = central

    
     
    
