import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.special import erf
np.random.seed(42)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#matplotlib.font_manager.findfont("Symbol")
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[:10]
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


V = [0.6]*3
Vtilde = [0.1, 3.0, 6.0]
assert len(V)==len(Vtilde)
n_mus = 4
colors = ["royalblue", "indianred"]
fig, ax = plt.subplots(n_mus, len(V), figsize=(6, 6), sharex=True, sharey=True)

for jj in range(len(V)):
    mu_vec = np.random.randn(n_mus)*np.sqrt(Vtilde[jj])
    mu_vec -= np.mean(mu_vec)
    mu_vec = mu_vec / np.std(mu_vec) * np.sqrt(Vtilde[jj])
    gamma = Vtilde[jj]/V[jj]
    a = mu_vec + np.random.randn(n_mus)*np.sqrt(V[jj])
    b = mu_vec + np.random.randn(n_mus)*np.sqrt(V[jj])
    
    qab = np.cov(a,b)[0,1]
    for ii in range(n_mus):
        ax[ii,jj].spines['top'].set_visible(False)
        ax[ii,jj].spines['right'].set_visible(False)
        ax[ii,jj].spines['left'].set_visible(False)
        ax[ii,jj].spines['bottom'].set_visible(False)
        ax[ii,jj].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[ii,jj].set_ylim(-0.1, 0.6)
        ax[ii,00].set_ylabel(f"$y_{ii+1}^{{(l)}}$", rotation=0, fontsize=20)
        mu_i = mu_vec[ii]
        ax[ii,jj].axvline(color="black")
        x = np.linspace(mu_i - 4 * np.sqrt(V[jj]), mu_i + 4 * np.sqrt(V[jj]), 1000)
        y = norm.pdf(x, mu_i, np.sqrt(V[jj]))
        ax[ii,jj].plot(x, y, color="black")
        
        ax[ii,jj].scatter(a[ii], 0.0, color=colors[0], marker="^")
        ax[ii,jj].scatter(b[ii], 0.0, color=colors[1], marker="^")

    ax[00,jj].set_title(f"{np.round(gamma,2)} - {np.round(qab,2)}")



rect = plt.Rectangle((0,0),1,1, transform=fig.transFigure, edgecolor='black', facecolor='none', lw=2)
fig.add_artist(rect)
fig.suptitle("$\\gamma^{(l)}$ - $q_{ab}^{(l)}$", fontsize=25)
fig.tight_layout()
fig.savefig("plot.png", dpi=300)


V = 1.0
Vtilde = [0.05, 1.0, 10.0]
n_mus = 2

fig, ax = plt.subplots(1, len(Vtilde),  figsize=(20,6), sharex=True, sharey=True)

width, height = 2.0, 0.2

fig.supylabel("$p_{y^{(L)}}$", rotation=0, fontsize=35, x=-0.00005)
for jj in range(len(Vtilde)):
    mu_vec = np.random.randn(n_mus)*np.sqrt(Vtilde[jj])
    mu_vec -= np.mean(mu_vec)
    mu_vec = mu_vec / np.std(mu_vec) * np.sqrt(Vtilde[jj])
    gamma = Vtilde[jj]/V
    
    #ax[jj].set_title(f"{np.round(gamma,2)} - {np.round(qab,2)}")
    
    ax[jj].set_title(f"$\\gamma^{{(L)}} = {np.round(gamma,2)}$", fontsize=30)
    ax[jj].set_ylim(0.0, 0.8)

    for ii in range(n_mus):
        mu_i = mu_vec[ii]
        x = np.linspace(mu_i - 4 * np.sqrt(V), mu_i + 4 * np.sqrt(V), 1000)
        y = norm.pdf(x, mu_i, np.sqrt(V))
        ax[jj].plot(x, y, color=colors[ii], label=f"$p_{{y_{ii+1}^{{(L)}}}}$")
        
        # insect box of the difference

    # draw delta distributions
    inset1 = inset_axes(ax[jj], width="100%", height="100%", bbox_to_anchor=(0.3 , 0.68, 0.4, 0.3), bbox_transform=ax[jj].transAxes)
    mu = mu_vec[1] - mu_vec[0]
    Var = V + Vtilde[jj]

    # # compute f0 numerically
    # n_net_samples = 10000
    # n_points = 1000
    # f0 = []
    # for ii in range(n_net_samples):
    #     mu_vec = np.random.randn(n_mus)*np.sqrt(Vtilde[jj])
    #     y1, y2 = np.random.normal(mu_vec[0], np.sqrt(V), size=(n_points)),  np.random.normal(mu_vec[1], np.sqrt(Var), size=(n_points))
    #     diff = y1 - y2
    #     f0.append(len(diff[diff>0])/n_points)
       
    
    # f0 = np.array(f0)
    
    # inset1.set_xlim(0,1)
    # sns.kdeplot(f0, bw_adjust=0.5, label='KDE', fill=False, ax=inset1, color="red")
    # inset1.set_ylabel("$p_{f_0}(y)$", fontsize=15)
    # inset1.set_xlabel("y")

    # compute f0 analytically
    n_net_samples = 100000
    delta = np.random.randn(n_net_samples)
    f0 =0.5*(1+erf(delta*np.sqrt(gamma/2)))
    inset1.set_xlim(0,1)
    sns.kdeplot(f0, bw_adjust=0.3, label='KDE', fill=True, ax=inset1, color="black")
    inset1.set_ylabel("$p_{g_0}$", fontsize=25, rotation=0, labelpad=30)
    inset1.set_xlabel("$g_0$", fontsize=30)
    # # draw center distributions
    # x = np.linspace( - 4 * np.sqrt(Vtilde[jj]), 4 * np.sqrt(Vtilde[jj]), 1000)
    # y = norm.pdf(x, 0, np.sqrt(Vtilde[jj]))
    # inset2 = inset_axes(ax[jj], width="100%", height="100%", bbox_to_anchor=(0.69, 0.65, 0.3, 0.2), bbox_transform=ax[jj].transAxes)
    # inset2.plot(x, y, color="black")
    # inset2.set_ylabel("$p_{\\mu}(z)$")
    # inset2.set_xlabel("z")
    #ax[jj].add_patch(outer_box)


#ax[00].set_xlim(-6, 6)
handles, labels = ax[00].get_legend_handles_labels()
#ax[0].text(-1.3, -0.1, "(a)", fontsize=12)
#ax[1].text(1.1, -0.1, "(b)", fontsize=12)
fig.supxlabel("$y^{{(L)}}$", fontsize=35)
fig.legend(handles, labels,loc='center', handletextpad=0.5, bbox_to_anchor=(0.0, 0.0, 0.81, 1.0), ncol=1,frameon=True, fontsize=30)
fig.tight_layout()
fig.savefig("plot2.png", dpi=300)


