import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as np
from scipy.integrate import quad, dblquad
from autograd import grad
import src.utils as ut
from scipy.special import erf
import pandas as pd

Vb_min = 0.0
Vb_max = 0.2
Vw_min = 0.0
Vw_max = 4.0
n_points=100

import matplotlib
#matplotlib.font_manager.findfont("Symbol")
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[:10]
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
dict_phase_color = ut.dict_phase_color

x = np.linspace(Vb_min, Vb_max, n_points)
y = np.linspace(Vw_min, Vw_max, n_points)

#################################
# draw phase diagrams
# for single input act functions
#################################

fig, ax = plt.subplots(1,2,figsize=(10,3), sharey=True, sharex=True)
fig.supxlabel("$\\sigma_{b}^2$",fontsize=20, rotation=0)
fig.supylabel("$\\sigma_{w}^2$", fontsize=20, rotation=0)
for ii in range(2):
    ax[ii].set_xlim(Vb_min, Vb_max)
    ax[ii].set_ylim(Vw_min, Vw_max)


### draw Linear phase diagram
# ax[0].set_title("Linear", fontsize=20)
# # divergent chaos
# ax[0].fill_between(x, 1, Vw_max, color=dict_phase_color["divergent_chaos"])
# ax[0].annotate("Divergent Chaos", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.1, 2.),
#             color="white",  fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))

# # finite order
# ax[0].fill_between(x, 0, 1.0, color=dict_phase_color["finite_order"])
# ax[0].annotate("Finite order", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.3, 0.65),
#             color="white", fontsize=15)     # Text position
# # edge of chaos
# ax[0].scatter(0.0, 1.0, color=dict_phase_color["edge_of_chaos"], marker="o", s=100, clip_on=False, zorder=5)

# # order/chaos phase transition
# ax[0].axhline(y=1.0, xmin=0, xmax=1, lw=3, ls="--", color=dict_phase_color["edge_of_chaos"])

# # vanishig chaos
# ax[0].axvline(x=0.0, ymin=0.0, ymax=1.0/Vw_max, color=dict_phase_color["vanishing_chaos"], lw=3,zorder=3, clip_on=False)
# ax[0].annotate("Vanishing chaos", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.4),
#             color=dict_phase_color["vanishing_chaos"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))



### draw ReLU phase diagram
ax[0].set_title("ReLU", fontsize=20)
ax[0].set_xticks([0, 0.05, 0.1, 0.15, 0.2])
# divergent correlated chaos
ax[0].fill_between(x, 2, Vw_max, color=dict_phase_color["chaotic_deep_prejudice"])
ax[0].annotate("Chaotic-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 2.6),
            color="white",  fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite order
ax[0].fill_between(x, 0, 2.0, color=dict_phase_color["ordered_deep_prejudice"])
ax[0].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 1.15),
            color="white", fontsize=18)     # Text position

# edge of chaos
ax[0].scatter(0.0, 2.0, color=dict_phase_color["edge_of_chaos"], marker="o", s=100, clip_on=False, zorder=5)

# order/chaos phase transition
ax[0].axhline(y=2.0, xmin=0, xmax=1, lw=3, ls="--", color=dict_phase_color["edge_of_chaos"])

# # vanishig order
# ax[0].axvline(x=0.0, ymin=0.0, ymax=2.0/Vw_max, color=dict_phase_color["vanishing_order"], lw=5,zorder=3, clip_on=False)
# ax[0].annotate("Vanishing order", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.8),
#             color=dict_phase_color["vanishing_order"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))
    
### draw Tanh phase diagram
ax[1].set_title("Tanh", fontsize=20)

#draw edge prejudice/neutrality transition
eoc = pd.read_csv("data/eoc_tanh.csv", index_col=0).loc[:,"eoc"]
neutrality = pd.read_csv("data/neutrality_Tanh.csv", index_col=0).loc[:,"neutrality"]

# neutrality
ax[1].plot(x,neutrality,lw=1, color="black", zorder=3,clip_on=False)
ax[1].fill_between(x, neutrality, Vw_max, color=dict_phase_color["neutrality"])
ax[1].annotate("Neutrality", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.02, 3.),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))


# edge of chaos
ax[1].plot(x,eoc,lw=3, color=dict_phase_color["edge_of_chaos"], zorder=3,clip_on=False)

# finite chaos
ax[1].fill_between(x, eoc, neutrality, color=dict_phase_color["prejudice"])
ax[1].annotate("Prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.15, 2.8),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite chaos
ax[1].fill_between(x, 0.0, eoc, color=dict_phase_color["ordered_deep_prejudice"])
ax[1].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 1.15),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# # vanishig chaos
# ax[1].axvline(x=0.0, ymin=0.0, ymax=1.0/Vw_max, color=dict_phase_color["vanishing_chaos"], lw=3,zorder=3, clip_on=False)
# ax[1].annotate("Vanishing chaos", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.4),
#             color=dict_phase_color["vanishing_chaos"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))

fig.tight_layout()
fig.savefig("figures/phase_diagrams.png", dpi=300)


#################################
# draw phase diagrams
# for two input act functions
#################################

Vb_min = 0.0
Vb_max = 0.2
Vw_min = 0.0
Vw_max = 4.0
n_points=100

fig, ax = plt.subplots(2,2,figsize=(10,6), sharex=True)
fig.supxlabel("$\\sigma_{b}^2$",fontsize=20, rotation=0)
fig.supylabel("$\\sigma_{w}^2$", fontsize=20, rotation=0)
for ii in range(2):
    ax[0,ii].set_xlim(Vb_min, Vb_max)
    ax[0,ii].set_ylim(Vw_min, Vw_max)

### ReLU +Maxpool
Vw_crit= 4*np.pi/(3*np.pi+2)
ax[0,0].set_title("ReLU+MaxPool", fontsize=20)
# divergent correlated chaos
ax[0,0].fill_between(x, Vw_crit, Vw_max, color=dict_phase_color["chaotic_deep_prejudice"])
ax[0,0].annotate("Chaotic-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 2.5),
            color="white",  fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite order
ax[0,0].fill_between(x, 0, Vw_crit, color=dict_phase_color["ordered_deep_prejudice"])
ax[0,0].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 0.55),
            color="white", fontsize=18)     # Text position

# edge of chaos
ax[0,0].scatter(0.0, Vw_crit, color=dict_phase_color["edge_of_chaos"], marker="o", s=100, clip_on=False, zorder=5)
ax[0,0].annotate("$\\frac{4\pi}{3\pi+2}$", 
            xy=(0.0, Vw_crit),                 # Point to annotate
            xytext=(0.05, 1.7),
            color="white",  fontsize=15,   # Text position
            arrowprops=dict(color="white", arrowstyle='->'))


# order/chaos phase transition
ax[0,0].axhline(y=Vw_crit, xmin=0, xmax=1, lw=3, ls="--", color=dict_phase_color["edge_of_chaos"])

# # vanishig order
# ax[0,0].axvline(x=0.0, ymin=0.0, ymax=Vw_crit/Vw_max, color=dict_phase_color["vanishing_order"], lw=5,zorder=3, clip_on=False)
# ax[0,0].annotate("Vanishing order", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.7),
#             color=dict_phase_color["vanishing_order"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))
    

ax[0,1].set_title("Tanh+MaxPool", fontsize=20)
eoc = pd.read_csv("data/eoc_Tanh+MaxPool.csv", index_col=0).loc[:,"eoc"]
# edge of chaos
ax[0,1].plot(x,eoc,lw=3, color=dict_phase_color["edge_of_chaos"], zorder=3,clip_on=False)

# finite chaos
ax[0,1].fill_between(x, eoc, Vw_max, color=dict_phase_color["prejudice"])
ax[0,1].annotate("Prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 3.),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite chaos
ax[0,1].fill_between(x, 0.0, eoc, color=dict_phase_color["ordered_deep_prejudice"])
ax[0,1].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 1.1),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# # vanishig chaos
# ax[1,0].axvline(x=0.0, ymin=0.0, ymax=eoc[0]**2/Vw_max, color=dict_phase_color["vanishing_chaos"], lw=3,zorder=3, clip_on=False)
# ax[1,0].annotate("Vanishing chaos", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.4),
#             color=dict_phase_color["vanishing_chaos"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))

#fig.tight_layout()
#fig.savefig("figures/phase_diagrams_maxpool.png", dpi=300)



Vb_min = 0.0
Vb_max = 0.2
Vw_min = 0.0
Vw_max = 6.0
n_points=100
for ii in range(2):
    ax[1,ii].set_xlim(Vb_min, Vb_max)
    ax[1,ii].set_ylim(Vw_min, Vw_max)
ax[0,0].set_xticks([0, 0.05, 0.1, 0.15, 0.2])
### Tanh + Maxpool
tanh = np.tanh
def act_func(z1, z2):
    return np.maximum(tanh(z1), tanh(z2))

grad_act_func = grad(act_func)


# #draw edge of chaos
# eoc = []
# for sigma_b_squared in x:
#     q=1.0
#     n_iter = 1000
#     for ii in range(n_iter):
#         q = update_variance_tanhmaxpool(q, sigma_b_squared)

#     eoc.append(compute_der_tanhmaxpool(q))


# np.savetxt("eoc_tanh_maxpool.txt", np.array(eoc))
### ReLU +AveragePool
Vw_crit= 4*np.pi/(np.pi+1)

# order/chaos phase transition
ax[1,0].axhline(y=Vw_crit, xmin=0, xmax=1, lw=3, ls="--", color=dict_phase_color["edge_of_chaos"])

ax[1,0].set_title("ReLU+AveragePool", fontsize=20)
# divergent correlated chaos
ax[1,0].fill_between(x, Vw_crit, Vw_max, color=dict_phase_color["chaotic_deep_prejudice"])
ax[1,0].annotate("Chaotic-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 4.3),
            color="white",  fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite order
ax[1,0].fill_between(x, 0, Vw_crit, color=dict_phase_color["ordered_deep_prejudice"])
ax[1,0].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 1.2),
            color="white", fontsize=18)     # Text position

# edge of chaos
ax[1,0].scatter(0.0, Vw_crit, color=dict_phase_color["edge_of_chaos"], marker="o", s=100, clip_on=False, zorder=5)
ax[1,0].annotate("$\\frac{4\pi}{\pi+1}$", 
            xy=(0.0, Vw_crit),                 # Point to annotate
            xytext=(0.05, 2.3),
            color="white",  fontsize=15,     # Text position
            arrowprops=dict(color="white", arrowstyle='->'))

# # vanishig order
# ax[0,1].axvline(x=0.0, ymin=0.0, ymax=Vw_crit/Vw_max, color=dict_phase_color["vanishing_order"], lw=5,zorder=3, clip_on=False)
# ax[0,1].annotate("Vanishing order", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.8),
#             color=dict_phase_color["vanishing_order"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))
    




### Tanh + Averagepool
tanh = np.tanh
def act_func(z1, z2):
    return np.maximum(tanh(z1), tanh(z2))

grad_act_func = grad(act_func)


ax[1,1].set_title("Tanh+AveragePool", fontsize=20)
eoc = pd.read_csv("data/eoc_Tanh+AveragePool.csv", index_col=0).loc[:,"eoc"]
#draw edge prejudice/neutrality transition
neutrality = pd.read_csv("data/neutrality_Tanh+AveragePool.csv", index_col=0).loc[:,"neutrality"]

# neutrality
ax[1,1].plot(x,neutrality,lw=1, color="black", zorder=3,clip_on=False)
ax[1,1].fill_between(x, neutrality, Vw_max, color=dict_phase_color["neutrality"])
ax[1,1].annotate("Neutrality", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.005, 5.),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))



# edge of chaos
ax[1,1].plot(x,eoc,lw=3, color=dict_phase_color["edge_of_chaos"], zorder=3,clip_on=False)

# finite chaos
ax[1,1].fill_between(x, eoc, neutrality, color=dict_phase_color["prejudice"])
ax[1,1].annotate("Prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.1, 5.0),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# finite chaos
ax[1,1].fill_between(x, 0.0, eoc, color=dict_phase_color["ordered_deep_prejudice"])
ax[1,1].annotate("Ordered-deep prejudice", 
            xy=(0.0, 2.5),                 # Point to annotate
            xytext=(0.05, 1.2),
            color="white", fontsize=18)     # Text position
            #arrowprops=dict(facecolor='white', arrowstyle='->'))

# # vanishig chaos
# ax[1,1].axvline(x=0.0, ymin=0.0, ymax=eoc[0]**2/Vw_max, color=dict_phase_color["vanishing_chaos"], lw=3,zorder=3, clip_on=False)
# ax[1,1].annotate("Vanishing chaos", 
#             xy=(0.0, 2.5),                 # Point to annotate
#             xytext=(0.02, 0.4),
#             color=dict_phase_color["vanishing_chaos"], fontsize=15)     # Text position
#             #arrowprops=dict(facecolor='white', arrowstyle='->'))

fig.tight_layout()
fig.savefig("figures/phase_diagrams_pool.png", dpi=300)
