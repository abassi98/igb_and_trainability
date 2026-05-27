import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as np
from scipy.integrate import quad, dblquad, nquad
from autograd import grad
import src.utils as ut
from scipy.special import erf
import pandas as pd

Vb_min = 0.0
Vb_max = 0.2
Vw_min = 0.0
Vw_max = 4.0
n_points=100

x = np.linspace(Vb_min, Vb_max, n_points)
y = np.linspace(Vw_min, Vw_max, n_points)

def der_tanh(x):
    return 1.0 - np.tanh(x)**2


def update_variance_tanh(q, Vw, Vb):
    fnum = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*np.tanh(np.sqrt(q)*z)**2
    I, err = quad(fnum, -np.inf, np.inf)
    return Vb + Vw*I

def update_covariance_tanh(c, q, Vw, Vb):
    fnum = lambda z1, z2: 1/(2*np.pi)*np.exp(-z1**2/2-z2**2/2)*np.tanh(np.sqrt(q)*z1)*np.tanh(np.sqrt(q)*(c*z1+np.sqrt(1.0-c**2)*z2))
    I, err = dblquad(fnum, -np.inf, np.inf,-np.inf, np.inf)
    return Vb + Vw*I


def eoc_variance_tanh(q, Vb):
    fnum = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*np.tanh(np.sqrt(q)*z)**2
    num, err = quad(fnum, -np.inf, np.inf)
    fden = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*der_tanh(np.sqrt(q)*z)**2
    den, err = quad(fden, -np.inf, np.inf)
    return num/den + Vb


def eoc_der_tanh(q):
    fden = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*der_tanh(np.sqrt(q)*z)**2
    den, err = quad(fden, -np.inf, np.inf)
    return 1/den


def V1_tanh(q):
    f = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*np.tanh(np.sqrt(q)*z)
    I, err = quad(f, -np.inf, np.inf)
    return I

def V2_tanh(q):
    f = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*np.tanh(np.sqrt(q)*z)**2
    I, err = quad(f, -np.inf, np.inf)
    return I

def Vprime2_tanh(q):
    f = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*der_tanh(np.sqrt(q)*z)**2
    I, err = quad(f, -np.inf, np.inf)
    return I

def update_variance_tanhaveragepool(q, Vw, Vb):
    return Vw*0.5*(V2_tanh(q) + V1_tanh(q)**2)+Vb


def eoc_variance_tanhaveragepool(q, Vb):
    num = V2_tanh(q) + V1_tanh(q)**2
    den = Vprime2_tanh(q)

    return num/den + Vb

def eoc_der_tanhaveragepool(q):
    den = 0.5*Vprime2_tanh(q)
    return 1/den


def update_variance(q, Vw, Vb, act_func, act_func2):
    f = lambda z1, z2  : 1/(2*np.pi)*np.exp(-z1**2/2-z2**2/2)*act_func2(act_func(np.sqrt(q)*z1), act_func(np.sqrt(q)*z2))**2
    I, err = dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
    return Vw*I + Vb

def update_covariance(c, q, Vw, Vb, act_func, act_func2):
    f = lambda z1, z2, z1p, z2p : 1/(2*np.pi)**2*np.exp(-z1**2/2-z2**2/2-z1p**2/2-z2p**2/2)*act_func2(act_func(np.sqrt(q)*z1), act_func(np.sqrt(q)*z2))*act_func2(act_func(np.sqrt(q)*(c*z1+np.sqrt(1-c**2)*z1p)), act_func(np.sqrt(q)*(c*z2+np.sqrt(1-c**2)*z2p)))
    ranges = [[-np.inf, np.inf]]*4
    I, err = nquad(f, ranges)
    return Vw*I + Vb

def update_covariance_mc(c, q, Vw, Vb, act_func, act_func2, N=100000):
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

Vw  = 0.1
Vb = 0.1
act_func = np.tanh
def maxpool(x, y):
    return np.maximum(x,y)

def averagepool(x, y):
    return (x+y)/2


### Tanh and Tanh+MaxPool
#draw edge of chaos
# eoc = []
# for sigma_b_squared in x:
#     q=1.0
#     n_iter = 1000
#     for ii in range(n_iter):
#         q = eoc_variance_tanh(q, sigma_b_squared)
#     eoc.append(eoc_der_tanh(q))

# df = pd.DataFrame(data={"Vb":x, "eoc":eoc})
# df.to_csv("data/eoc_tanh.txt")

### Tanh+AveragePool

#draw edge of chaos
eoc = []
for sigma_b_squared in x:
    q=1.0
    n_iter = 1000
    for ii in range(n_iter):
        q = eoc_variance_tanhaveragepool(q, sigma_b_squared)
    eoc.append(eoc_der_tanhaveragepool(q))

df = pd.DataFrame(data={"Vb":x, "eoc":eoc})
df.to_csv("data/eoc_tanh_averagepool.csv")