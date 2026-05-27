import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
import src.utils as ut
import numpy as np
from matplotlib.cm import ScalarMappable
from cmcrameri import cm
from src.utils import calculate_gain, str2bool, none_or_type
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

plt.subplots_adjust(right=0.8)
seed = 42
np.seterr(under="warn")

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
#plt.rcParams['font.weight'] = 'bold'


def average(data, window=4):
    """Average data with a sliding window."""
    n_points = data.shape[0]
    assert n_points % window == 0, "Data length must be divisible by window size."
    n_views = n_points // window
    data_view = data.reshape(n_views, window)
    data_avg = data_view.mean(axis=1)
    data_avg = data_avg.reshape(n_views, 1)
    data_avg = np.repeat(data_avg, window, axis=1).reshape(n_points)
    return data_avg

def parse_list(arg):
    # Strip brackets and split by commas
    return [int(x) for x in arg.strip('[]').split(',')]


def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="MLP", choices=["MLP", "RESMLP", "CNN", "VIT", "LargeVIT", "RESNET", "MLPMIXER"], help="Model to use, 0 for MLP.")
    parser.add_argument('--max_depth', type=none_or_type(int), default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--widths', type=none_or_type(parse_list), default=1000, help="Widths to plot")
    parser.add_argument('--act_func', type=none_or_type(str), default="ReLU", help="Activation function.")
    parser.add_argument('--act_func2', type=none_or_type(str), default="Linear", help="Activation function.")
    parser.add_argument('--data', type=str, default="bfmnist", choices=["random", "bfmnist", "bcifar","cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument('--use_faster_attention', type=str2bool, default=True, help="Use faster attention for VIT.")
    parser.add_argument('--plot_mode', type=str, default="semilogy", help="Plot type from pyplot, e.g. plot, semilog etc")
    cfg = vars(parser.parse_args())
    return cfg

if __name__=="__main__":
    cfg = get_args()
    model = cfg["model"]
    max_depth = cfg["max_depth"]
    widths = cfg["widths"] if isinstance(cfg["widths"], list)  else [cfg["widths"]]
    n_widths = len(widths) 
    act_func = cfg["act_func"]
    act_func2 = cfg["act_func2"]
    plot_mode = cfg["plot_mode"]
    random.seed(seed)
    np.random.seed(seed)
    
    # load experimental values from differente wiodths
    if act_func2=="Linear":
        run_cfg, buf_prime = ut.read_data(f"data/init/{cfg['model']}_{cfg['data']}_{act_func}_D{max_depth}_W{widths[0]}.h5")
    else:
        run_cfg, buf_prime = ut.read_data(f"data/init/{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{max_depth}_W{widths[0]}.h5")
 
    n_classes = buf_prime.shape[1] - 1
    n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
    effective_depth = buf_prime.shape[2]
    cmap = cm.roma_r
    # select effective depth of the transfoormer
    buf = np.zeros((14, n_classes+1, effective_depth, n_widths, n_b,n_w))
    buf[:,:,:, 0, :, :] =  buf_prime
    for ww, width in enumerate(widths[1:]):
        if act_func2 == "Linear":
            _, buf_prime  = ut.read_data(f"data/init/{cfg['model']}_{cfg['data']}_{act_func}_D{max_depth}_W{width}.h5")
        else:
            _, buf_prime  = ut.read_data(f"data/init/{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{max_depth}_W{width}.h5")
        buf[:,:,:,ww+1,:,:] = buf_prime


    cmap = cm.roma_r
    gain = 1.0
    if cfg["model"] in ["MLP", "RESMLP", "CNN", "RESNET"]:
        if act_func is not None and act_func2 is not None:
            gain = calculate_gain(act_func, act_func2, Vb=0.1)
        else:
            gain = 1.0
        Vw_range = gain*np.array(Vw_range)

    # define hyper-params range
    Vw_vec = np.linspace(Vw_range[0],Vw_range[1],n_w)
    Vb_vec = np.linspace(Vb_range[0],Vb_range[1],n_b)
    Vw_mean = (Vw_range[0] + Vw_range[1])/2
 
    norm = TwoSlopeNorm(vmin=Vw_range[0], vcenter=Vw_mean, vmax=Vw_range[1])
    # set up figures
    fig_grad, ax_grad = plt.subplots(1,3,figsize=(7*3,7), sharey=True, sharex=True)
    fig_corr, ax_corr = plt.subplots(1,figsize=(8, 5), sharey=True, sharex=True)
    n_widths = 10
   
    plot_depth = effective_depth-1
    depths = np.arange(1,plot_depth+1)
    pairs = [[0,ii] for ii in range(n_w)] # sigma_b^2 = 0.1

    fig_grad.supxlabel("$\mathrm{layer}\; l$", fontsize=20)
    
    # set titles
    if act_func is not None:
        if act_func2=="Linear":
            fig_grad.suptitle(f"{act_func} {model}", fontsize=25, fontweight='bold')
        else:
            if act_func2.startswith("Avg"):
                t2 = "AveragePool"
            elif act_func2.startswith("Max"):
                t2 = "MaxPool"
            fig_grad.suptitle(f"{act_func}+{t2} {model}", fontsize=25, fontweight='bold')


    # retrieve IGB and MF quantities
    mean_var_y ,var_mu_y = buf[6,0, ...], buf[7,0, ...]
    gamma = var_mu_y / mean_var_y
    c_igb = gamma/(gamma+1)

    for ind_c, pp in enumerate(pairs):
        Vb_ind, Vw_ind = pp
        Vw, Vb = Vw_vec[Vw_ind], Vb_vec[Vb_ind]
        color = cmap(norm(Vw))
    
        # Plot experimental
        for oo, width in enumerate(widths):
            lw = 1.0 +  5* (oo +1)/ (len(widths)+1)
            # plot only last width
            if oo == len(widths)-1:
                ### PLOT DIAG
                # trace sgd
                    ax_grad[0].set_title("Global", fontsize=20)
                    ax_grad[1].set_title(f"Favoured class", fontsize=20)
                    ax_grad[2].set_title(f"Unfavoured class", fontsize=20)
                    ax_grad[0].grid()
                    ax_grad[1].grid()
                    ax_grad[2].grid()
                    # means
                    #print(buf[2,0,:plot_depth,oo, Vb_ind,Vw_ind].shape)
                    window = 4
                    data = buf[2,0,:plot_depth,oo, Vb_ind,Vw_ind]
                    smoothed = average(data,window=window)
                    getattr(ax_grad[0], plot_mode)(depths, smoothed, ls="-", c=color, lw=2) 
                    data =buf[2,1,:plot_depth,oo, Vb_ind,Vw_ind]
                    smoothed =  average(data,window=window)
                    getattr(ax_grad[1], plot_mode)(smoothed, ls="-", c=color, lw=2) 
                    data = buf[2,-1,:plot_depth,oo, Vb_ind,Vw_ind]
                    smoothed =  average(data,window=window)
                    getattr(ax_grad[2], plot_mode)(depths, smoothed, ls="-", c=color, lw=2) 
                    # quantiles
                    #ax_grad[0].fill_between(depths, buf[1,0,:plot_depth,oo, Vb_ind,Vw_ind], buf[3,0,:plot_depth,oo, Vb_ind,Vw_ind], color=color, alpha=0.3) # 5, 95 % range
                    #ax_grad[1].fill_between(depths, buf[1,1,:plot_depth,oo, Vb_ind,Vw_ind], buf[3,1,:plot_depth,oo, Vb_ind,Vw_ind], color=color, alpha=0.3) # 5, 95 % range
                    #ax_grad[2].fill_between(depths, buf[1,-1,:plot_depth,oo, Vb_ind,Vw_ind], buf[3,-1,:plot_depth,oo, Vb_ind,Vw_ind], color=color, alpha=0.3) # 5, 95 % range
                    # plot c vs cigb
                    # ax_corr.plot(depths, c_igb[:plot_depth,oo, Vb_ind,Vw_ind], ls="-", c=color, lw=2)
                    # ax_corr.fill_between(depths, buf[8,0,:plot_depth,oo, Vb_ind,Vw_ind], buf[12,0,:plot_depth,oo, Vb_ind,Vw_ind], color=color, alpha=0.3) # 5, 95 % range
                    # ax_corr.grid()
                #print(Vw, Vb, buf[0,:plot_depth,oo, Vb_ind,Vw_ind])

    # grid and colrobar
    divider = make_axes_locatable(ax_grad[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig_grad.colorbar(sm, cax=cax)
    cb.ax.set_title("$\\sigma_w^2$", fontsize=15)
    #cb.ax.set_yticklabels([f'{t:.2f}' for t in ticks])  # optionally format tick labels
    # save diag figure
    #handles, labels = ax_grad.get_legend_handles_labels()
    #fig_grad.legend(handles, labels, loc='center right', handletextpad=0.5, bbox_to_anchor=(0.0, 0.0, 0.85, 0.7), ncol=1,frameon=True, fontsize=15)
    fig_grad.supylabel("$\\tilde{q}_{ab}^{(l)}$", fontsize=20, rotation=0, x=0)
    fig_grad.tight_layout()
    
    if act_func2 == "Linear":
        fig_grad.savefig(f"figures/init/grads_{cfg['model']}_{cfg['data']}_{act_func}_D{max_depth}_W{max(widths)}.png", dpi=300)
    else: 
        fig_grad.savefig(f"figures/init/grads_{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{max_depth}_W{max(widths)}.png", dpi=300)

    # grid and colrobar
    divider = make_axes_locatable(ax_corr)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig_corr.colorbar(sm, cax=cax)
    cb.ax.set_title("$\\sigma_w^2$", fontsize=15)
    #cb.ax.set_yticklabels([f'{t:.2f}' for t in ticks])  # optionally format tick labels
    # save diag figure
    #handles, labels = ax_corr.get_legend_handles_labels()
    #fig_corr.legend(handles, labels, loc='center right', handletextpad=0.5, bbox_to_anchor=(0.0, 0.0, 0.85, 0.7), ncol=1,frameon=True, fontsize=15)
    ax_corr.set_title(model, fontsize=20, fontweight='bold')
    fig_corr.supxlabel("$\\mathrm{layer}\;l$", fontsize=20)
    fig_corr.supylabel("$c_{ab}^{(l)}$", fontsize=20, rotation=0, x=0)
    fig_corr.tight_layout()
    
    if act_func2 == "Linear":
        fig_corr.savefig(f"figures/init/corr_{cfg['model']}_{cfg['data']}_{act_func}_D{max_depth}_W{max(widths)}.png", dpi=300)
    else: 
        fig_corr.savefig(f"figures/init/corr_{cfg['model']}_{cfg['data']}_{act_func}+{act_func2}_D{max_depth}_W{max(widths)}.png", dpi=300)

    
