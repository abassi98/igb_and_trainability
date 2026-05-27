import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SimpleAttention(nn.Module):
    """
    Simple Torch attention mechanism
    """
    def __init__(self, d, 
                N, 
                sigma_w=1.0):
        
        super(SimpleAttention, self).__init__()

        self.d = d # token dimension
        self.N = N # number of tokens

        self.Q = nn.parameter.Parameter(torch.randn(d, d)) # Query matrix
        self.K =  nn.parameter.Parameter(torch.randn(d, d)) # Key matrix
        self.V =  nn.parameter.Parameter(torch.randn(d, d)) # Value natrix

        self.std = sigma_w / np.sqrt(d) * np.log(N)**(1/4)
        self.initialize()


    def initialize(self):
        nn.init.normal_(self.Q, mean=0, std=self.std)
        nn.init.normal_(self.K, mean=0, std=self.std)
        nn.init.normal_(self.V, mean=0, std=self.std)

    def forward(self, X):
        Att = torch.nn.functional.softmax(torch.matmul(torch.t(torch.matmul(self.Q, X)), torch.matmul(self.K, X)) / np.sqrt(self.d), dim=1) # attention coefficients
        V = torch.matmul(self.V, X)
        self_att = torch.matmul(Att, torch.t(V))

        return self_att

def generate_correlated_gaussian(N, d, c, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Create covariance matrix (equicorrelation structure)
    cov = (1 - c) * torch.eye(d) + c * torch.ones((d, d))

    # Mean vector (zero-mean)
    mean = torch.zeros(d)

    # Create distribution
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

    # Sample N vectors
    samples = mvn.sample((N,))  # shape: (N, d)
    return torch.t(samples)

if __name__ == "__main__":
    beta = 0.0
    fig, ax = plt.subplots(1,1)
    for alpha in [0.1, 1.0,  2.0, 10,20]:
        d = 200
        N = int(d * alpha)
        ws = np.linspace(0.0, 10.0, 200)
        gammas = []
        for sigma_w in ws:
            X = generate_correlated_gaussian(N, d, c=0.01, seed=None)
            model = SimpleAttention(d, N, sigma_w)
            self_att = model(X)
            mean_dim = torch.mean(self_att, dim=1)
            mean_data = torch.mean(self_att, dim=0)
            var_data = torch.var(self_att, dim=0)
            var_var_data = torch.var(var_data)
            sigma_y_squared = torch.mean(var_data)
            sigma_mu_squared = torch.mean(mean_data**2)
            gamma = sigma_mu_squared / sigma_y_squared
            gammas.append(gamma.detach().numpy())



        gammas = np.array(gammas)
        ax.set_ylim(1e-3, 1e8)
        ax.loglog(ws**2, gammas * alpha**(beta), label=f"alpha = {alpha}")
        ax.set_xlabel("sigma_w squared")
        ax.set_ylabel("gamma")

    fig.legend()
    fig.savefig("figures/Attention.png", dpi=300)
  