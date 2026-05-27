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
    parser.add_argument('--plot_mode', type=str, default="semilogy", help="Plot type from pyplot, e.g. plot, semilog etc")
    cfg = vars(parser.parse_args())
    return cfg

if __name__=="__main__":
    cfg = get_args()
    model = "LargeVIT"
    dataset = "cifar100"
    plot_mode = cfg["plot_mode"]
    random.seed(seed)
    np.random.seed(seed)
    
    # load experimental values
    run_cfg, buf = ut.read_data(f"data/init/{model}_{dataset}_None+None_DNone_WNone.h5")
   
    # load experimental values from paper fig
    run_cfg, buf_p = ut.read_data(f"data/init/{model}_{dataset}_None+None_DNone_WNone_paper.h5")
    
    n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
    cmap = cm.roma_r
    # define hyper-params range
    Vw_vec = np.linspace(Vw_range[0],Vw_range[1],n_w)
    Vb_vec = np.linspace(Vb_range[0],Vb_range[1],n_b)
    Vw_mean = (Vw_range[0] + Vw_range[1])/2
 
    norm = TwoSlopeNorm(vmin=Vw_range[0], vcenter=Vw_mean, vmax=Vw_range[1])

    # set up figures
    window = 1 # for smooting
    fig_grad, ax_grad = plt.subplots(1,2,figsize=(4*3,4), sharey=True, sharex=True)
    plot_depth = buf.shape[2]-1
    depths = np.arange(1,plot_depth+1)
    pairs = [[0,ii] for ii in range(n_w)] # sigma_b^2 = 0.1
    fig_grad.supxlabel("$\mathrm{layer}\; l$", fontsize=20)

    for ind_c, pp in enumerate(pairs):
        Vb_ind, Vw_ind = pp
        Vw, Vb = Vw_vec[Vw_ind], Vb_vec[Vb_ind]
        color = cmap(norm(Vw))
    
        ax_grad[0].set_title("Pretrained", fontsize=20)
        ax_grad[1].set_title(f"Untrained", fontsize=20)
        ax_grad[0].grid()
        ax_grad[1].grid()
      
    
        # plot init grads
        data = buf_p[2,0,:plot_depth,Vb_ind,Vw_ind]
        smoothed = average(data,window=window) 
        smoothed /= smoothed[-1]
        getattr(ax_grad[0], plot_mode)(depths, smoothed, ls="-", c=color, lw=2) 

        # plot pretrained grads (paper)
        data = buf[2,0,:plot_depth,Vb_ind,Vw_ind]
        smoothed = average(data,window=window) 
        smoothed /= smoothed[-1]
        getattr(ax_grad[1], plot_mode)(depths, smoothed, ls="-", c=color, lw=2) 
                   
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
    fig_grad.savefig(f"figures/init/grads_{model}_{dataset}_initvspretrained.png", dpi=300)

