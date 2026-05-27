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
    folders = ["tb_logs/train_spec_mlp_bfmnist", "tb_logs/train_spec_largevit_cifar100"]

    metrics_dict = {}
    specifics = []
    Vw_dict = {}
    Vb_dict = {}

    for folder in folders:
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
            print(foldername)
            for v, version_folder in enumerate(os.listdir(os.path.join(folder, foldername))):
                metrics_version = pd.read_csv(os.path.join(folder, foldername, version_folder, 'metrics.csv'))
                metrics_version.drop(columns=['epoch'], inplace=True, errors='ignore')
                metrics[f"v{v}"] = metrics_version

            # concatenate all metrics into a single DataFrame and retrieve means, std, and quantiles
            metrics = pd.concat(metrics, names = ['version'])
            metrics = align_versions(metrics)
            metrics_dict[foldername] = metrics
            
    specific_counter = Counter(specifics)
    specific_counter = {k: v for k, v in specific_counter.items() if "relu" not in k}
    titles = ["Tanh MLP trained on binarized fashion MNIST", "Vision Transformer fine-tuned on CIFAR100"]
    # plot
    for n_spec, specific in enumerate(specific_counter.keys()):
        # set up figures
        # set up figures
        fig_acc, ax_acc = plt.subplots(2, 2, figsize=(10, 2.5), sharey="row", sharex=True)
        nruns = specific_counter[specific]
        model = specific.split('_')[0]
        act_func = specific.split('_')[1]
        dataset = specific.split('_')[2]
        n_classes = get_nc(dataset)
       
        # select cmap
        cmap = cm.roma_r
        
        Vw_specific = {k: v for k, v in Vw_dict.items() if specific in k}
        Vb_specific = {k: v for k, v in Vb_dict.items() if specific in k}
        Vw_max = np.max(np.array(list(Vw_specific.values())))
        Vw_min = np.min(np.array(list(Vw_specific.values())))
        gain = calculate_gain(act_func, "Linear", 0.1)
        norm = TwoSlopeNorm(vmin=Vw_min*gain, vcenter=gain, vmax=Vw_max*gain)

        for run in range(1,nruns+1):
            name = f"{specific}_run{run}.cfg"
            Vw = Vw_specific[name]*calculate_gain(act_func, "Linear", 0.1)
            #color = cmap(norm(Vw))
            phase = get_phase(act_func, run)
            
            name = phase.replace("_", " ")
            label = f"{name.title()} ($\\sigma_w^2$ = {np.round(Vw,2)})"
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

            ### train
            ### accuracy 
            if model=="mlp":
                ax_acc[0,0].set_title("Training", fontsize=fontsize)
            # train accuracy
            ax_acc[0,0].semilogx(median.index +1, median['train_acc'], c=color, label=label)
            ax_acc[0,0].fill_between(median.index +1, q05['train_acc'], q95['train_acc'], alpha=0.5, color=color)
            ax_acc[0,0].set_ylabel('Accuracy', fontsize=fontsize)
            ax_acc[0,0].grid(True)
          
            # delta fr
            if n_classes < 10:
                ax_acc[1,0].semilogx(median.index +1, median[f'max_train_freq'], c=color)
                ax_acc[1,0].set_ylim(1/n_classes*0.9,1)
            else:
                ax_acc[1,0].loglog(median.index +1, median[f'max_train_freq'], c=color)
                ax_acc[1,0].set_ylim(1/n_classes*0.5,1)
            ax_acc[1,0].fill_between(median.index +1, q05[f'max_train_freq'], q95[f'max_train_freq'], alpha=0.5, color=color)
            ax_acc[1,0].set_ylabel(f'Max Freq.', fontsize=fontsize)
            ax_acc[1,0].axhline(1/n_classes, color="green")
            ax_acc[1,0].grid(True)
            

            ### test
            # accuracy 
            if model=="mlp":
                ax_acc[0,1].set_title("Test", fontsize=fontsize)
            # test accuracy
            ax_acc[0,1].semilogx(median.index +1, median['test_acc'], c=color, label=label)
            ax_acc[0,1].fill_between(median.index +1, q05['test_acc'], q95['test_acc'], alpha=0.5, color=color)
            ax_acc[0,1].grid(True)
        
            # delta fr
            if n_classes <10:
                ax_acc[1,1].semilogx(median.index +1, median[f'max_test_freq'], c=color)
                ax_acc[1,1].set_ylim(1/n_classes*0.9,1)
            else:
                ax_acc[1,1].loglog(median.index +1, median[f'max_test_freq'], c=color)
                ax_acc[1,1].set_ylim(1/n_classes*0.5,1)
            ax_acc[1,1].fill_between(median.index +1, q05[f'max_test_freq'], q95[f'max_test_freq'], alpha=0.5, color=color)
            #ax_acc[1,1].set_ylabel(f'Max Test Frequency')
            ax_acc[1,1].axhline(1/n_classes, color="green")
            ax_acc[1,1].grid(True)


        handles1, labels1 = ax_acc[0,0].get_legend_handles_labels()
        handles2, labels2 = ax_acc[1,0].get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        by_label = dict(zip(labels, handles))
        if model=="largevit":
            fig_acc.supxlabel('Number of training of steps', fontsize=fontsize)

        fig_acc.suptitle(titles[n_spec], fontsize=fontsize*1.2)
        legend = fig_acc.legend(by_label.values(), by_label.keys(), handletextpad=0.5, loc="upper center",bbox_to_anchor=(0.5, 0.93), ncol=4,frameon=True, fontsize=10)
        fig_acc.tight_layout(rect=[0, 0, 1, 0.95])
        for legline in legend.get_lines():
            legline.set_linewidth(4)  # make legend lines thicker
        fig_acc.savefig(os.path.join("figures", "train", f'{model}_train_main.png'), dpi=300)

       