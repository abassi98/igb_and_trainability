import numpy as np
import matplotlib.pyplot as plt 
import argparse
import os
import pandas as pd
import re
from collections import Counter
from src.utils import calculate_gain, dict_phase_color, str2bool
import cmcrameri.cm as cm
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

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


fontsize=12

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folder', type=str, help="File containing the specification of the model.")
    parser.add_argument('--plot_df', type=str2bool, default=True, help="Whether plot delta frequency")
    cfg = vars(parser.parse_args())
    return cfg


def align_versions(metrics):
    versions = metrics.index.get_level_values("version").unique()
    new_metrics = {}
    
    for version in versions:
        v = version.split("v")[1]
        df_v = metrics.loc[f"v{v}"].copy()
        n_classes = sum(bool(re.fullmatch(r"test_acc_class_\d+", col)) for col in df_v.columns)
        
        ### order test
        # Select columns matching the pattern
        cols_test_acc = [col for col in df_v.columns if re.fullmatch(r"test_acc_class_\d+", col)]
        cols_test_freq = [col for col in df_v.columns if re.fullmatch(r"test_freq_class_\d+", col)]
        df_v["max_test_freq"] = df_v.loc[:,cols_test_freq].max(axis=1)
        df_v["max_test_acc"] = df_v.loc[:,cols_test_acc].max(axis=1)
        df_v["min_test_acc"] = df_v.loc[:,cols_test_acc].min(axis=1)
        # Sort these columns by their value at index 0, descending
        #order_test = sorted(cols_test_acc, key=lambda c: df_v.loc[0, c], reverse=True)
        #order_test = [int(col.split('_')[-1])for col in order_test]
        
        
        ### order train
        # Select columns matching the pattern
        cols_train_acc = [col for col in df_v.columns if re.fullmatch(r"train_acc_class_\d+", col)]
        cols_train_freq = [col for col in df_v.columns if re.fullmatch(r"train_freq_class_\d+", col)]
        df_v["max_train_freq"] = df_v.loc[:,cols_train_freq].max(axis=1)
        df_v["max_train_acc"] = df_v.loc[:,cols_train_acc].max(axis=1)
        df_v["min_train_acc"] = df_v.loc[:,cols_train_acc].min(axis=1)
        # Sort these columns by their value at index 0, descending
        #order_train = sorted(cols_train_acc, key=lambda c: df_v.loc[0, c], reverse=True)
        #order_train = [int(col.split('_')[-1])for col in order_train]
        
    
        #print(order, values)
        #cols_class = [col for col in df_v.columns if "_class_" in col]

        # order each time step
        #df_v[cols_train_acc] = df_v[cols_train_acc].apply(lambda row: sorted(row, reverse=True), axis=1, result_type="expand")
        #df_v[cols_test_acc] = df_v[cols_test_acc].apply(lambda row: sorted(row, reverse=True), axis=1, result_type="expand")

        # rename the columns to have a suffix with the order
        #rename_dict = {}
        # for col in cols_class:
        #     # Extract the old class number
        #     old_class = int(col.split('_class_')[-1])
        #     # Map to new class according to order
            
        #     if "train" in col:
        #         new_class = order_train.index(old_class)
        #     else:
        #         new_class = order_test.index(old_class)

        #     # Build new column name
        #     new_col = col.replace(f'_class_{old_class}', f'_class_{new_class}')
        #     rename_dict[col] = new_col

        # Step 4: Rename columns in the dataframe
        
        #df_v = df_v.rename(columns=rename_dict)
        
        #print("new_version")
        #print([df_v.loc[0,f"test_freq_class_{i}"] for i in range(n_classes)])
        #print([df_v.loc[0,f"test_acc_class_{i}"] for i in range(n_classes)])
        
        new_metrics[f"v{v}"] = df_v
        #print(rename_dict)
        #print(df_v.loc[0, "test_acc_class_0"], df_v.loc[0, "test_acc_class_1"])

    return pd.concat(new_metrics, names = ['version'])

def get_phase(act_func, run):
    if act_func == "linear" or act_func == "relu":
        if run==1:
            label="ordered_deep_prejudice"
        elif run==2:
            label="edge_of_chaos"
        elif run==3:
            label="chaotic_deep_prejudice"
        else:
            raise ValueError(f"Unknown run: {run} for activation function {act_func}")
        
    elif act_func == "tanh":
        if run==1:
            label="ordered_deep_prejudice"
        elif run==2:
            label="edge_of_chaos"
        elif run==3:
            label="prejudice"
        elif run==4:
            label="neutrality"
        else:
            raise ValueError(f"Unknown run: {run} for activation function {act_func}")

    return label

def get_nc(data):
    if data=="bcifar" or data=="bfmnist":
        return 2
    elif data=="cifar10" or data=="fmnist":
        return 10
    elif data=="tinyimagenet":
        return 200
    elif data=="cifar100":
        return 100
    else:
        raise ValueError("Dataset not recognized.")

if __name__ == "__main__":
    args = get_args()
    seed = args["seed"]
    folder = args["folder"]
    plot_df = args["plot_df"]
    nrows = 4 if plot_df else 3
    data = {}

    
    metrics_dict = {}
    specifics = []
    Vw_dict = {}
    Vb_dict = {}
    for foldername in os.listdir(folder):
        # split by run
        model = foldername.split('_')[0]
        dataset = foldername.split('_')[2]
        spec_file = f"train_spec_{model}_{dataset}/{foldername}"
        with open(spec_file, 'r') as f:
            cfg = eval(f.read())
        Vw_dict[foldername] = cfg["Vw"]
        Vb_dict[foldername] = cfg["Vb"]

        specific = foldername.split('_run')[0].split('/')[-1]    
        specifics.append(specific)
        run = int(foldername.split("run")[-1].split('.')[0])
        metrics = {}
        #print(foldername)
        for v, version_folder in enumerate(os.listdir(os.path.join(folder, foldername))):
            metrics_version = pd.read_csv(os.path.join(folder, foldername, version_folder, 'metrics.csv'))
            metrics_version.drop(columns=['epoch'], inplace=True, errors='ignore')
            metrics[f"v{v}"] = metrics_version

        # concatenate all metrics into a single DataFrame and retrieve means, std, and quantiles
        metrics = pd.concat(metrics, names = ['version'])
        metrics = align_versions(metrics)
        metrics_dict[foldername] = metrics
        
    specific_counter = Counter(specifics)
    act_funcs = [specific.split('_')[1] for specific in specific_counter.keys()]
    # check that all act functions have the same number of runs
    #print(act_funcs )
    # set up figures
    fig_loss, ax_loss = plt.subplots(3, len(act_funcs), figsize=(10,10), sharey="row", sharex=True)
    fig_acc, ax_acc = plt.subplots(nrows,len(act_funcs), figsize=(10, 10/8*nrows), sharex=True, sharey="row")
    # plot
    for specific in specific_counter.keys():
        nruns = specific_counter[specific]
        model = specific.split('_')[0]
        act_func = specific.split('_')[1]
        ind_act_func = act_funcs.index(act_func)
        dataset = specific.split('_')[2]
        n_classes = get_nc(dataset)
       
    

        # select cmap
        cmap = cm.roma_r
        
        Vw_specific = {k: v for k, v in Vw_dict.items() if specific in k}
        Vb_specific = {k: v for k, v in Vb_dict.items() if specific in k}
        Vw_max = np.max(np.array(list(Vw_specific.values())))
        Vw_min = np.min(np.array(list(Vw_specific.values())))
        gain = calculate_gain(act_func, "Linear", 0.1)
        norm = TwoSlopeNorm(vmin=Vw_min*gain-0.1, vcenter=gain, vmax=Vw_max*gain+0.1)

        for run in range(1,nruns+1):
            name = f"{specific}_run{run}.cfg"
            Vw = Vw_specific[name]*calculate_gain(act_func, "Linear", 0.1)
            #color = cmap(norm(Vw))
            phase = get_phase(act_func, run)
            label = phase.replace("_", " ")
            #label = f"{name.title()} ({np.round(Vw,2)})"
            color = dict_phase_color[phase]
            metrics = metrics_dict[f"{specific}_run{run}.cfg"]
            #print(metrics.columns)
            #print(metrics.xs(0, level=1)["test_acc_class_0"])
            #print(metrics)
            #print(name, metrics.xs(0, level=1)["test_acc_class_0"])
            Vw = Vw_dict[f"{specific}_run{run}.cfg"]
            # retrieve stats and take stats over the version
            mean = metrics.groupby(level=1).mean()
            std = metrics.groupby(level=1).std()
            q05 = metrics.groupby(level=1).quantile(0.25)    
            median = metrics.groupby(level=1).quantile(0.5)
            #print(name, median.iloc[:2]["test_acc_class_0"])
            q95 = metrics.groupby(level=1).quantile(0.75)
            max = metrics.groupby(level=1).max()
            min = metrics.groupby(level=1).min()

            # test
            ### losses
            # test loss
            ax_loss[0, ind_act_func].set_title(act_func.title())
            ax_loss[0, ind_act_func].loglog(median.index +1, median['test_loss'], c=color, label=label)
            ax_loss[0, ind_act_func].fill_between(median.index +1, q05['test_loss'], q95['test_loss'], alpha=0.5, color=color)
            ax_loss[0, 0].set_ylabel('Global', fontsize=fontsize)
            ax_loss[0, ind_act_func].grid(True)
            # test loss Favoured
            ax_loss[1, ind_act_func].loglog(median.index +1, median[f'test_loss_class_{0}'], c=color)
            ax_loss[1,ind_act_func].fill_between(median.index +1, q05[f'test_loss_class_{0}'], q95[f'test_loss_class_{0}'], alpha=0.5, color=color)
            ax_loss[1, 0].set_ylabel(f'Favoured', fontsize=fontsize)
            ax_loss[1,ind_act_func].grid(True)
            # test loss Unfavoured
            ax_loss[2,ind_act_func].loglog(median.index +1, median[f'test_loss_class_{n_classes-1}'], c=color)
            ax_loss[2,ind_act_func].fill_between(median.index +1, q05[f'test_loss_class_{n_classes-1}'], q95[f'test_loss_class_{n_classes-1}'], alpha=0.5, color=color)
            ax_loss[2,0].set_ylabel(f'Unfavoured', fontsize=fontsize)
            ax_loss[2,ind_act_func].grid(True)
            
            ### accuracy and frequency
            ax_acc[0,ind_act_func].set_title(act_func.title())
            # test accuracy
            ax_acc[0,ind_act_func].semilogx(median.index +1, median['test_acc'], c=color, label=label)
            ax_acc[0,ind_act_func].fill_between(median.index +1, q05['test_acc'], q95['test_acc'], alpha=0.5, color=color)
            ax_acc[0, 0].set_ylabel('Global', fontsize=fontsize)
            ax_acc[0,ind_act_func].grid(True)
            # test accuracy Favoured
            ax_acc[1,ind_act_func].semilogx(median.index +1, median[f'max_test_acc'], c=color)
            ax_acc[1,ind_act_func].fill_between(median.index +1, q05[f'max_test_acc'], q95[f'max_test_acc'], alpha=0.5, color=color)
            ax_acc[1, 0].set_ylabel(f'Favoured', fontsize=fontsize)
            ax_acc[1,ind_act_func].grid(True)
            # test accuracy Unfavoured
            ax_acc[2,ind_act_func].semilogx(median.index +1, median[f'min_test_acc'], c=color)
            ax_acc[2,ind_act_func].fill_between(median.index +1, q05[f'min_test_acc'], q95[f'min_test_acc'], alpha=0.5, color=color)
            ax_acc[2,0].set_ylabel(f'Unfavoured', fontsize=fontsize)
            ax_acc[2,ind_act_func].grid(True)
            if plot_df:
                # delta fr
                if n_classes <10:
                    ax_acc[3,ind_act_func].semilogx(median.index +1, median[f'max_test_freq'], c=color)
                else:
                    ax_acc[3,ind_act_func].loglog(median.index +1, median[f'max_test_freq'], c=color)
                ax_acc[3,ind_act_func].fill_between(median.index +1, q05[f'max_test_freq'], q95[f'max_test_freq'], alpha=0.5, color=color)
                #ax_acc[3,ind_act_func].set_ylabel(f'Max Test Frequency')
                ax_acc[3,ind_act_func].axhline(1/n_classes)
                ax_acc[3,ind_act_func].grid(True)

    
            

    ## savings
    # divider = make_axes_locatable(ax_loss[-1])
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # sm = ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cb = fig_acc.colorbar(sm, cax=cax)
    # cb.ax.set_title("$\\sigma_w^2$", fontsize=12)
    # labels, ha = ax_loss[0].get_legend_handles_labels()
    
    ncol = 5
    handles1, labels1 = ax_loss[0,0].get_legend_handles_labels()
    handles2, labels2 = ax_loss[0,1].get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    by_label = dict(zip(labels, handles))
    fig_loss.supxlabel('Number of training steps', fontsize=fontsize)
    fig_loss.tight_layout(rect=[0, 0, 1, 0.95])
    legend=fig_loss.legend(by_label.values(), by_label.keys(),handletextpad=0.5, loc="upper center",bbox_to_anchor=(0.5, 0.98), ncol=ncol, frameon=True,fontsize=10)
    for legline in legend.get_lines():
        legline.set_linewidth(4)  # make legend lines thicker
    fig_loss.savefig(os.path.join("figures", "train", f'{model}_{dataset}_loss_test.png'), dpi=300)


    handles1, labels1 = ax_acc[0,0].get_legend_handles_labels()
    handles2, labels2 = ax_acc[0,1].get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    by_label = dict(zip(labels, handles))
    fig_acc.supxlabel('Number of training of steps', fontsize=fontsize)
    fig_acc.tight_layout(rect=[0, 0, 1, 0.95])
    legend=fig_acc.legend(by_label.values(), by_label.keys(), handletextpad=0.5, loc="upper center",bbox_to_anchor=(0.5, 1.0), ncol=ncol,frameon=True, fontsize=10)
    for legline in legend.get_lines():
        legline.set_linewidth(4)  # make legend lines thicker
    fig_acc.savefig(os.path.join("figures", "train", f'{model}_{dataset}_accuracy_test.png'), dpi=300)

       