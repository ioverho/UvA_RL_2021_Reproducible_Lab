"""
make_plots.py
Loads the training results of main.py, and saves several plots in './plots'
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

OUTPUT_DIR = "./figures/"
json_folder = "./continuous_trpo/logs_json/"

def get_stats_filename(json_folder, target_measure="Gradient_Norm"):
    relevant_filenames = []
    pathlist = Path(json_folder).rglob('*.json')
    for pathname in pathlist:
        pathname = str(pathname)
        if target_measure in pathname:
            print("[get_stats_filename] found relevant file ", pathname)
            relevant_filenames.append(pathname)

    return relevant_filenames

def combine_json_files(relevant_filenames):
    combined_json = {}
    for filename in relevant_filenames:
        filename_list = filename.replace('-','_').replace('/','_').split("_")
        run = int(filename_list[7])
        print("[combine_json_files] run {} added".format(run))
        with open(filename) as json_file:
            data = json.load(json_file)
            combined_json[run] = np.array(data)
    return combined_json

def plot_line(x_list, y_list, label, normalize=False, n_smooth=10):
    
    if normalize:
        y = np.array(y_list)
        y_list = (y - np.min(y)) / (np.max(y) - np.min(y))

    y_mean = np.array(y_list).mean(0)
    y_std = np.array(y_list).std(0)
    x = x_list[0]

    x = smooth(x, n_smooth)
    y_mean = smooth(y_mean, 10)
    y_std = smooth(y_std, 10)

    sns.lineplot(
        x=x,
        y=y_mean,
        label=label,
    )
    lower_bound = y_mean - y_std
    upper_bound = y_mean + y_std
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.3)


def plot_lines(target_measure, combined_json_allenvs, envs, 
            axis_lbl=["iteration", "gradient"], title="Continuous", 
            normalize=False, hide_axis=True):
    plt.clf()
    fig1 = plt.figure(figsize=(8, 5))
    plt.tight_layout()

    for env in envs:
        print("[plot_lines] env ", env)
        combined_json = combined_json_allenvs[env]
        x_list = []
        y_list = []
        for run, data in combined_json.items():
            x_list.append(combined_json[run][:,1])
            y_list.append(combined_json[run][:,2])
        plot_line(x_list, y_list, label=env, normalize=normalize)

    if hide_axis:
        plt.xticks([])

    if title: 
        plt.title(title.title(), size=14)

    if axis_lbl[0]:
        plt.xlabel(axis_lbl[0],  size=14)
        plt.legend()
    if axis_lbl[1]:
        plt.ylabel(axis_lbl[1],  size=14)
        plt.legend()

    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.2f}'))
    plt.savefig(OUTPUT_DIR + "cont_trpo_" + target_measure + ".png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR + "cont_trpo_" + target_measure + ".pdf", bbox_inches='tight')
    print('[saved]: ', target_measure)

def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_target_measure(json_folder, envs, target_measure="Gradient_Norm", title="Continuous", axis_lbl=["iteration", "gradient"], normalize=False, hide_axis=True):
    combined_json_allenvs = {}
    for env in envs:
        relevant_filenames = get_stats_filename(json_folder + env, target_measure=target_measure)
        combined_json = combine_json_files(relevant_filenames)
        combined_json_allenvs[env] = combined_json
    plot_lines(target_measure, combined_json_allenvs, envs=envs, axis_lbl=axis_lbl, title=title,  normalize=normalize, hide_axis=hide_axis)

if __name__=="__main__":
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    envs = ["Swimmer-v2", "Walker2d-v2", "Hopper-v2"]

    plot_target_measure(json_folder, envs, target_measure="Mean_Reward", title="Continuous", axis_lbl=["Iteration",None], normalize=True)
    plot_target_measure(json_folder, envs, target_measure="TRPO_Learning_Rate_Max", title=None, axis_lbl=[None,None], normalize=True)
    plot_target_measure(json_folder, envs, target_measure="TRPO_Learning_Rate_Line_Search_Scalar", title=None, axis_lbl=["Iteration",None])
    plot_target_measure(json_folder, envs, target_measure="TRPO_Learning_Rate_Effective_Learning_rate", title=None, axis_lbl=[None,None], normalize=True)
    plot_target_measure(json_folder, envs, target_measure="Update_Loss_Improvement", title=None, axis_lbl=[None,None])
    plot_target_measure(json_folder, envs, target_measure="Update_KL_Divergence", title=None, axis_lbl=["Iteration",None])

    plot_target_measure(json_folder, envs, target_measure="Norms_grad", title="Continuous", axis_lbl=[None,None])
    plot_target_measure(json_folder, envs, target_measure="Norms_update", title=None, axis_lbl=["Iteration",None], normalize=False, hide_axis=False)
    plot_target_measure(json_folder, envs, target_measure="Norms_search_dir", title=None, axis_lbl=[None,None])
