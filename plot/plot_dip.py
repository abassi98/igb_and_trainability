import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, ListedColormap, Normalize, TwoSlopeNorm
import math
from src.utils import generate_colour_shades, generate_blue_to_red_shades
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from cmcrameri import cm
import pandas as pd

import src.utils as ut
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
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--act_func2', type=str, default="Linear", help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--width', type=int, default=10000, help="Width for experimental values")
    
    cfg = vars(parser.parse_args())
    return cfg

if __name__=="__main__":
    cfg = get_args()
    max_depth = cfg["max_depth"]
    act_func2 = cfg["act_func2"]
    width = cfg["width"]
    random.seed(seed)
    np.random.seed(seed)

    act_funcs = ["ReLU", "Tanh"]
    fig, axs = plt.subplots(1,2,figsize=(14,4.5))
    for ii, act_func in enumerate(act_funcs):
        ax = axs[ii]
        inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.45 , 0.1, 0.35, 0.35), bbox_transform=ax.transAxes)
        inset.grid()
        inset.set_ylabel("$|c - c_{ab}^{(l)}|$", fontsize=15, rotation=90)
        
        # load experimental values (IGB) for comparison
        if act_func2 == "Linear":
            run_cfg, buf = ut.read_data(f"data/{act_func}_D{max_depth}_W{width}.h5")
        else:
            run_cfg, buf = ut.read_data(f"data/{act_func}+{act_func2}_D{max_depth}_W{width}.h5")
        n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
  
        Vexp = buf[0,:, 0, :, :]
        Vtildeexp= buf[1,:, 0, :, :]
        c_5pc = np.quantile(buf[2,...], 0.05, axis=1)
        c_95pc = np.quantile(buf[2,...], 0.95, axis=1)
        Gexp = Vtildeexp / Vexp # experimental gamma from IGB
        Cigb = Gexp/(Gexp+1) # c from IGB

        # load theoretical values (DIP)
        if act_func2=="Linear":
            run_cfg2, buf2 = ut.read_data(f"data/{act_func}_D200_DIP.h5")
            qaa = buf2[0,...]
            qbb = buf2[1,...]
            qab = buf2[2,...]
            c_star = np.mean(qab/np.sqrt(qaa*qbb), axis=1)[-1,...]
        else:
            run_cfg2, buf2 = ut.read_data(f"data/{act_func}+{act_func2}_D200_DIP.h5")
            qaa = buf2[0,...]
            qbb = buf2[1,...]
            qab = buf2[2,...]
            c_star = np.mean(qab/np.sqrt(qaa*qbb), axis=1)[-1,...]
       
        n_w, n_b = run_cfg["n_w"], run_cfg["n_b"]
        Vw_max, Vw_min = run_cfg["Vw_max"], run_cfg["Vw_min"]
        Vb_max, Vb_min = run_cfg["Vb_max"], run_cfg["Vb_min"]
        Vw_vec = np.linspace(Vw_min,Vw_max,n_w)
        Vb_vec = np.linspace(Vb_min,Vb_max,n_b)
        
        # colors 
        cmap = cm.roma_r
        if act_func=="Tanh":
            if act_func2=="Linear":
                df = pd.read_csv("data/eoc_Tanh.csv", index_col=0)
                custom_mid = df.iloc[51,1]
            elif act_func2=="MaxPool":
                df = pd.read_csv("data/eoc_Tanh+Maxpool.csv", index_col=0)
                custom_mid = df.iloc[51,1]
            elif act_func2=="AveragePool":
                df = pd.read_csv("data/eoc_Tanh+Averagepool.csv", index_col=0)
                custom_mid = df.iloc[51,1]
        elif act_func=="ReLU":
            if act_func2=="MaxPool":
                custom_mid = 4*np.pi/(3*np.pi+2)
            elif act_func2=="AveragePool":
                custom_mid = 4*np.pi/(np.pi+1)
            elif act_func2=="Linear":
                custom_mid = 2.0

        print(f"{act_func} + {act_func2} custom_mid", custom_mid)
        norm = TwoSlopeNorm(vmin=Vw_min, vcenter=custom_mid, vmax=Vw_max)
        #norm = Normalize(vmin=Vw_min, vmax=Vw_max)
        
        ### plot exp dip vs exp IGB
        # set up figures
        if act_func=="ReLU":
            init_depth = 10
        else:
            init_depth = 0
        
        if act_func2=="Linear":
            final_depth = cfg["max_depth"]
        else:
            final_depth = 100
        depths = np.arange(init_depth+1,final_depth+1)
        # plot only certain among the Vw and Vb values
        # plot experimental IGB
        pairs = [[1,jj] for jj in range(n_w)] 
        
        fig.supxlabel("$\mathrm{layer}\; l$", fontsize=20)
        if ii==0:   
            ax.set_ylabel("$c_{ab}^{(l)}$", fontsize=20, rotation=0, labelpad=30)
        
        # set titles
        if act_func2=="Linear":
            ax.set_title(act_func, fontsize=20)
        else:
            ax.set_title(f"{act_func}+{act_func2}", fontsize=20, fontweight="heavy")

        if act_func=="Tanh":
            pass#ax.set_ylim(1e-6, 1e-2)
        for ind_c, pp in enumerate(pairs):
            Vb_ind, Vw_ind = pp
            Vw, Vb = Vw_vec[Vw_ind], Vb_vec[Vb_ind]
            color = cmap(norm(Vw))
        
            if act_func=="ReLU":
                c_star[:] = 1
        
            #print(Vw_vec[Vw_ind], Vb_vec[Vb_ind])
            #label=f"{act_func} with ($\\sigma_{{w}}^2$,  $\\sigma_{{b}}^2$) = ({Vw_vec[Vw_ind]}, {Vb_vec[Vb_ind]})",
            ax.plot(depths,Cigb[init_depth:final_depth, Vb_ind,Vw_ind],  ls="-", color=color,  lw=2)
            # plot exp DIP
            #if act_func2=="Linear":
            im = ax.scatter(final_depth, c_star[Vb_ind,Vw_ind],  s=100, color=color)
            #getattr(ax, plot_mode)(depths, cmeanexp[:plot_depth,Vb_ind,Vw_ind], marker="+", c=colors[ind_c])
            ax.fill_between(depths,np.abs(c_5pc[init_depth:final_depth,Vb_ind,Vw_ind]), (c_95pc[init_depth:final_depth,Vb_ind,Vw_ind]), color=color, alpha=0.3)
            ### inset loglog
            inset.loglog(np.arange(11,final_depth+1), np.abs(c_star[Vb_ind,Vw_ind]-Cigb[10:, Vb_ind,Vw_ind]), color=color)

        ax.grid()
        ### colorbar 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # Draw the colorbar on that axes
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Not used, but required for colorbar
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.set_title("$\sigma_w^2$", fontsize=12)
        #cb.set_label(f"{cmap_igb} scale")        # customize the label
        #cb.set_ticks(np.linspace(vlim[0], vlim[1], 5))  # explicit tick locations

        # save main figure
        handles, labels = ax.get_legend_handles_labels()
        
    #fig.legend(handles, labels, loc='center right', handletextpad=0.5, bbox_to_anchor=(0.0, 0.0, 0.85, 0.7), ncol=1,frameon=True, fontsize=15)
    #fig.tight_layout()
    if act_func2 == "Linear":
        fig.savefig(f"figures/DIPExp_vs_IGBExp_D{max_depth}_W{width}.png", dpi=300)
    else:
        fig.savefig(f"figures/DIPExp_vs_IGBExp_{act_func2}_D{max_depth}_W{width}.png", dpi=300)

