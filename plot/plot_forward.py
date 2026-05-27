import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, ListedColormap
import math
from src.utils import generate_colour_shades

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


def g_relu(x):
    if x <=np.inf:
        a = 2.0
        val =  a/np.pi *x* np.arctan(np.sqrt(a*x+1.0))   + np.sqrt(a*x+1.0)/np.pi #- (1)/np.pi# + np.exp(x/10000.0) #- 1.0/(3.0*np.pi*np.sqrt(2*x))
        return val
    else:
        return x

def f_relu(x):
    if x.any() >=0:
        val = np.ones_like(x)  +x -g_relu(x)
        return  val
    
    else:
        raise ValueError("gamma cannot be negative")

def g_frac_f(x):
    return g_relu(x)/f_relu(x) 

def I(x):
    den = np.sqrt(np.pi*x)
    a = x * np.arctan(np.sqrt(2*x+1))
    b = x*x / ((x+1)*np.sqrt(2*x+1))
    return den * (2/np.pi*(a+b)-x/2)
#gradient = grad(g_frac_f)

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--widths', type=parse_list, default=[10000], help="Widths to plot")
    parser.add_argument('--act_func', type=str, default="ReLU", choices=["Linear", "ReLU", "TanhLike", "Tanh"], help="Single point activation function.")
    parser.add_argument('--act_func2', type=str, default="Linear", choices=["Linear", "MaxPool", "MinPool", "AveragePool"], help="Two point activation function.")
    parser.add_argument('--plot_mode', type=str, default="semilogy", help="Plot type from pyplot, e.g. plot, semilog etc")
    cfg = vars(parser.parse_args())
    return cfg

def func_gamma_ther(X, Y, l):
    out = 0.0
    for k in range(l+1):
        out += 1. / X**(k+1)
    return out*Y

if __name__=="__main__":
    cfg = get_args()
    max_depth = cfg["max_depth"]
    widths = cfg["widths"]
    print(widths)
    n_widths = len(widths)
    act_func = cfg["act_func"]
    act_func2 = cfg["act_func2"]
    plot_mode = cfg["plot_mode"]
    random.seed(seed)
    np.random.seed(seed)


    # load experimental values
    # buf = Vexp_prime, Vtildeexp_prime, cmeanprime, cvarprime
    if act_func2=="Linear":
        run_cfg, buf_prime = ut.read_data(f"data/{act_func}_D{max_depth}_W{widths[0]}.h5")
    else:
        run_cfg, buf_prime = ut.read_data(f"data/{act_func}+{act_func2}_D{max_depth}_W{widths[0]}.h5")

    n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
    n_data_samples = run_cfg["n_data_samples"]
    buf = np.zeros((5, max_depth, n_widths,n_data_samples, n_b, n_w))
    buf[:,:,0,:,:,:] = buf_prime
    
    for ww, width in enumerate(widths[1:]):
        if act_func2=="Linear":
            _, buf_prime = ut.read_data(f"data/{act_func}_D{max_depth}_W{width}.h5")
        else:
            _, buf_prime = ut.read_data(f"data/{act_func}+{act_func2}_D{max_depth}_W{width}.h5")
        buf[:,:,ww+1,:,:,:] = buf_prime
    

    # update here
    Vexp = buf[0,:,:,0,:,:]
    Vtildeexp= buf[1,:,:,0,:,:]
    c_dist = buf[2,...]
    q_dist = buf[3,...]
    cov_dist = buf[4,...]
    c_5pc = np.quantile(c_dist, 0.05, axis=2)
    c_95pc = np.quantile(c_dist, 0.95, axis=2)
    q_inter = 100*(np.quantile(q_dist, 0.95, axis=2) - np.quantile(q_dist, 0.05, axis=2))/np.quantile(q_dist, 0.5, axis=2)
    cov_inter = 100*(np.quantile(cov_dist, 0.95, axis=2) - np.quantile(cov_dist, 0.05, axis=2))/np.quantile(q_dist, 0.5, axis=2)
    
    Gexp = Vtildeexp / Vexp # experimental gamma
    Cigb = Gexp/(Gexp+1) # c from IGB
    
    Vw_vec = np.linspace(Vw_range[0],Vw_range[1],n_w)
    Vb_vec = np.linspace(Vb_range[0],Vb_range[1],n_b)
   
    # set up figures
    fig, ax = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True)
    fig_corr, ax_corr = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True, sharey=True)
    fig_var, ax_var = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True, sharey=True)
    fig_cov, ax_cov = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True, sharey=True)
    fig_g, ax_g = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True,sharey=True)
    fig_res, ax_res = plt.subplots(n_w, n_b,figsize=(15,15), sharex=True)

    # compute theoretical values
    Vtildeth = np.zeros((max_depth, n_b, n_w))
    Vth = np.zeros((max_depth, n_b, n_w))
    Gth = np.zeros((max_depth, n_b, n_w))
    Qth = np.zeros((max_depth, n_b, n_w))
    blue_shades = generate_colour_shades(n_widths, colour="blue")
    green_shades = generate_colour_shades(n_widths, colour="green")
    red_shades = generate_colour_shades(n_widths, colour="red")
    orange_shades = generate_colour_shades(n_widths, colour="orange")
    plot_depth = max_depth
    depths = np.arange(100)
    for ii, Vb in enumerate(Vb_vec):
        for jj, Vw in enumerate(Vw_vec):
            # initialize first values
            Vth[0,ii,jj] = Vw
            Vtildeth[0,ii,jj] = Vb + Vw/n_data_samples
            Gth[0,ii,jj] = Vtildeth[0,ii,jj]/Vth[0,ii,jj]
            for kk in range(max_depth-1): # iterate corresponding to number of layers
                if act_func2=="Linear":
                    if act_func == "ReLU":
                        Vth[kk+1,ii,jj] = Vw*Vth[kk,ii,jj]/2.0 * f_relu(Gth[kk,ii,jj])
                        Vtildeth[kk+1, ii,jj] = Vw*Vth[kk,ii,jj]/2.0 * g_relu(Gth[kk,ii,jj]) + Vb 
                    elif act_func == "Linear":
                        Vtildeth[kk+1, ii,jj] = Vw*Vtildeth[kk,ii,jj]+Vb
                        Vth[kk+1,ii,jj] = Vw*Vth[kk,ii,jj]
                    else:
                        pass
                    Gth[kk+1,ii,jj] = Vtildeth[kk+1,ii,jj] / (Vth[kk+1,ii,jj])
                else:
                    pass  

            Qth = Vtildeth +  Vth
            # Plot theoretical
            if ii==0:
                ax[jj,ii].set_ylabel(f"{np.round(Vw,2)}",labelpad=20, fontsize=20, rotation=0)
                ax_corr[jj,ii].set_ylabel(f"{np.round(Vw,2)}", labelpad=20,fontsize=20, rotation=0)
                ax_var[jj,ii].set_ylabel(f"{np.round(Vw,2)}", labelpad=20,fontsize=20, rotation=0)
                ax_cov[jj,ii].set_ylabel(f"{np.round(Vw,2)}",labelpad=20, fontsize=20, rotation=0)
                ax_g[jj,ii].set_ylabel(f"{np.round(Vw,2)}",labelpad=20, fontsize=20, rotation=0)
                ax_res[jj,ii].set_ylabel(f"{np.round(Vw,2)}",labelpad=20, fontsize=20, rotation=0)
            if jj==0:
                ax[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}",  fontsize=20, rotation=0)
                ax_corr[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}",  fontsize=20, rotation=0)
                ax_var[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}",fontsize=20, rotation=0)
                ax_cov[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}", fontsize=20, rotation=0)
                ax_g[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}", fontsize=20, rotation=0)
                ax_res[jj,ii].set_title(f"$\\sigma_{{b}}^2$ = {np.round(Vb,2)}", fontsize=20, rotation=0)
            
            ax[jj,ii].grid()
            ax_corr[jj,ii].grid()
            ax_g[jj,ii].grid()
            ax_var[jj,ii].grid()
            ax_cov[jj,ii].grid()
            
            if act_func=="ReLU" or act_func=="Linear":
                getattr(ax[jj,ii], plot_mode)(Vth[:plot_depth,ii,jj], ls=(0,(5,10)), c="green")
                getattr(ax[jj,ii], plot_mode)(Vtildeth[:plot_depth,ii,jj],  ls=(0,(5,10)),c="red")
                getattr(ax[jj,ii], plot_mode)(Gth[:plot_depth,ii,jj],  ls=(0,(5,10)), c="blue")
            
            # Plot experimental
            for oo, width in enumerate(widths):
                lw = 1.0 +  5* (oo +1)/ (len(widths)+1)
                # plot error
                #getattr(ax_var[ii,jj], plot_mode)(100*np.abs((Vth[:,ii,jj]-Vexp[:plot_depth,oo, ii,jj])/Vth[:,ii,jj]), lw=lw,label="V th.", c=green_shades[oo])
                #getattr(ax_cov[ii,jj], plot_mode)(100*np.abs((Vtildeth[:,ii,jj]-Vtildeexp[:plot_depth,oo, ii,jj])/Vtildeth[:,ii,jj]), lw=lw,label="$\\tilde{V}$ th.", c=red_shades[oo])
                getattr(ax_g[jj,ii], plot_mode)(100*np.abs((Gth[:,ii,jj]-Gexp[:plot_depth,oo, ii,jj])/Gth[:,ii,jj]), label=widths[oo], lw=lw, c=blue_shades[oo])
                getattr(ax_var[jj,ii], plot_mode)(q_inter[:plot_depth,oo, ii,jj],color=blue_shades[oo], label=widths[oo], lw=lw)
                getattr(ax_cov[jj,ii], plot_mode)(cov_inter[:plot_depth,oo, ii,jj],color=blue_shades[oo], label=widths[oo], lw=lw)
                

                # plot only last width
                if oo == len(widths)-1:
                    getattr(ax[jj,ii], plot_mode)(Vexp[:plot_depth,oo, ii,jj],ls="dotted",label="$\\sigma_{y^{(l)}}^2$",  c=green_shades[oo])
                    getattr(ax[jj,ii], plot_mode)(Vtildeexp[:plot_depth,oo, ii,jj], ls="dotted",label="$\\sigma_{\\mu^{(l)}}^2$", c=red_shades[oo])
                    getattr(ax[jj,ii], plot_mode)(Gexp[:plot_depth,oo, ii,jj], ls="dotted", label="$\\gamma^{(l)}$", c=blue_shades[oo])
                    ax_corr[jj,ii].plot(Cigb[:plot_depth,oo, ii,jj], ls="dotted", c=blue_shades[oo],  lw=1)
                    ax_corr[jj,ii].fill_between(depths, c_5pc[:plot_depth,oo, ii,jj], c_95pc[:plot_depth,oo, ii,jj], color=red_shades[oo], alpha=0.5)
                    

            # set xticks labels
            ax[jj,ii].tick_params(axis='both', labelsize=20)  
            ax_g[jj,ii].tick_params(axis='both', labelsize=20)  
            ax_corr[jj,ii].tick_params(axis='both', labelsize=20)  
            ax_var[jj,ii].tick_params(axis='both', labelsize=20)  
            ax_cov[jj,ii].tick_params(axis='both', labelsize=20)  

    # save main figure
    handles, labels = ax[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    fig.legend(handles, labels, loc='lower left', handletextpad=0.5, bbox_to_anchor=(0.1, 0.0, 1.0, 1.0), ncol=3,frameon=True, fontsize=25, title_fontsize=25)
    fig.supxlabel("$\\mathrm{layer}\;l$",x=0.7, fontsize=35)
    fig.supylabel("$\\sigma_{w}^2$", fontsize=35, rotation=0, x=-0.01)
    #fig.suptitle("$\\sigma_{b}^2$",fontsize=35, rotation=0)
    fig.tight_layout(rect=[0, 0, 0.8, 1]) 
    if act_func2=="Linear":
        fig.savefig(f"figures/{plot_mode}_{act_func}_EXPTH_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')
    else:
        fig.savefig(f"figures/{plot_mode}_{act_func}+{act_func2}_EXPTH_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')
    
    # save Corr figure
    handles, labels = ax_corr[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    #fig_corr.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.1, 0.0, 1.0, 1.0), ncol=3,frameon=True, fontsize=25)
    fig_corr.supxlabel("$\\mathrm{layer}\;l$", x=0.5, fontsize=35)
    fig_corr.supylabel("$\\sigma_{w}^2$", fontsize=35, rotation=0,  x=-0.01)
    #fig_corr.suptitle("$\\sigma_{b}^2$",fontsize=35, rotation=0, )
    fig_corr.tight_layout(rect=[0, 0, 1, 0.95]) 
    if act_func2=="Linear":
        fig_corr.savefig(f"figures/plot_{act_func}_Corr_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')
    else:
        fig_corr.savefig(f"figures/plot_{act_func}+{act_func2}_Corr_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

    # save distribution of VARIANCE
    handles, labels = ax_var[00,00].get_legend_handles_labels()
    fig_var.supxlabel("$\\mathrm{layer}\;l$", x=0.5, fontsize=35)
    fig_var.supylabel("$\\sigma_{w}^2$", fontsize=35, rotation=0)
    #fig_var.suptitle("$\\sigma_{b}^2$",fontsize=35, rotation=0)
    fig_var.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig_var.legend(handles, labels, loc='upper left', handletextpad=0.5, bbox_to_anchor=(0.4, 1.00), ncol=4,frameon=True, fontsize=15, title="Width $N$", title_fontsize=20)
    fig_var.savefig(f"figures/{plot_mode}_{act_func}_VAR_D{max_depth}_W{max(widths)}.png", dpi=300)
    
    # save distribution of COV
    handles, labels = ax_cov[00,00].get_legend_handles_labels()
    fig_cov.supxlabel("$\\mathrm{layer}\;l$", x=0.5, fontsize=35)
    fig_cov.supylabel("$\\sigma_{w}^2$", fontsize=35, rotation=0)
    #fig_cov.suptitle("$\\sigma_{b}^2$",fontsize=35, rotation=0)
    fig_cov.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig_cov.legend(handles, labels, loc='upper left', handletextpad=0.5, bbox_to_anchor=(0.4, 1.0), ncol=4,frameon=True, fontsize=15, title="Width $N$", title_fontsize=20)
    fig_cov.savefig(f"figures/{plot_mode}_{act_func}_COV_D{max_depth}_W{max(widths)}.png", dpi=300)

    # error on CORR
    handles, labels = ax_g[00,00].get_legend_handles_labels()
    fig_g.supxlabel("$\\mathrm{layer}\;l$", x=0.5,fontsize=35)
    fig_g.supylabel("$\\sigma_{w}^2$", fontsize=35, rotation=0)
    #fig_g.suptitle("$\\sigma_{b}^2$",fontsize=35, rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    leg = fig_g.legend(handles, labels, loc='upper left', handletextpad=0.5, bbox_to_anchor=(0.3, 0.98), ncol=4,frameon=True, fontsize=15, title="Width $N$", title_fontsize=20)
    fig_g.savefig(f"figures/{plot_mode}_{act_func}_GERR_D{max_depth}_MaxW{max(widths)}.png", dpi=300)
    

    # # save comparison figure for residuals
    # handles, labels = ax_res[00,00].get_legend_handles_labels()
    # #fig.legend(handles, labels, loc='upper center')
    # fig_res.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    # fig_res.supxlabel("# of layers", fontsize=20)
    # fig_res.supylabel("$V_B$", fontsize=20)
    # fig_res.suptitle("$V_W$",fontsize=20)
    # fig_res.tight_layout(rect=[0, 0, 0.85, 1]) 
    # fig_res.savefig(f"figures/{plot_mode}_{act_func}_Cresidual_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

    # # plot in function of the width
    # fig_var, ax_var = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # fig_G, ax_G = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # fig_cov, ax_cov = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # cmap = plt.cm.Blues
    # blue_shades = generate_blue_shades(n_widths)
    # widths = np.logspace(np.log10(10), np.log10(max_width), num=n_widths).astype(np.int16)
    # for ii, Vb in enumerate(Vb_vec):
    #     for jj, Vw in enumerate(Vw_vec):
    #         for oo, width in enumerate(widths): 
    #             # Plot experimental
    #             getattr(ax_var[ii,jj], plot_mode)(Vexp[:,oo, ii,jj], label=f"{width}", c=blue_shades[oo],  ls="dotted")
    #             getattr(ax_cov[ii,jj], plot_mode)(Vtildeexp[:,oo, ii,jj], c=blue_shades[oo], label=f"{width}", ls="dotted")
    #             getattr(ax_G[ii,jj], plot_mode)(Gexp[:, oo, ii,jj], c=blue_shades[oo], label=f"{width}", ls="dotted")
            
    # handles, labels = ax_var[00,00].get_legend_handles_labels()
    
    # # savefigs
    # fig_var.supxlabel("Width")
    # fig_var.supylabel("$V_B$")
    # fig_var.suptitle("$V_W$")
    # fig_var.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_var.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_var.savefig(f"figures/V_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)

    # fig_cov.supxlabel("Width")
    # fig_cov.supylabel("$V_B$")
    # fig_cov.suptitle("$V_W$")
    # fig_cov.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_cov.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_cov.savefig(f"figures/Vtilde_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)

    # fig_G.supxlabel("Width")
    # fig_G.supylabel("$V_B$")
    # fig_cov.suptitle("$V_W$")
    # fig_G.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_G.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_G.savefig(f"figures/Gamma_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)
    