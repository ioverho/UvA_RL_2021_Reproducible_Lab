# %%
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(
        dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            #assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def pull_results(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    #for index, tag in enumerate(tags):
    #    df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
    #    df.to_csv(get_file_path(dpath, tag))

    return dirs, tags, np_values


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


#if __name__ == '__main__':


# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def smoothed_time_plot(x, N_smoothing: int = 0, min_max_rescale: bool = False, label=""):

    if min_max_rescale:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    mean = np.median(x, axis=-1)
    se = np.std(x, axis=-1) / np.sqrt(x.shape[-1])

    if N_smoothing > 0:
        mean = np.convolve(mean, np.ones(N_smoothing)/N_smoothing, mode='valid')
        se = np.convolve(se, np.ones(N_smoothing)/N_smoothing, mode='valid')

    plt.plot(mean, label=label)
    plt.fill_between(np.arange(0, mean.shape[0]),
                    mean+se,
                    mean-se,
                    alpha=0.25)

    return mean+se, mean-se


def discrete_plot(img_paths: dict, i: int, min_max_rescale: bool, N_smoothing: int, bottom: bool = True, fontsize:int =12):
    lims = []
    for env_name, env_path in img_paths.items():
        dirs, tags, vals = pull_results(env_path)

        ul, ll = smoothed_time_plot(
            vals[i],
            min_max_rescale=min_max_rescale, N_smoothing=N_smoothing,
            label=env_name
            )

        lims.append([np.min(ll), np.max(ul)])

    lims = np.stack(lims)
    lims = np.min(lims, axis=0)[0], np.max(lims, axis=0)[1]

    plt.ylim(lims)

    y_lab = tags[i].rsplit("/")[-1]

    plt.ylabel(y_lab + (" (scaled)" if min_max_rescale else ""), fontsize=fontsize)

    if bottom:
        plt.xlabel("Iteration", fontsize=fontsize)
    else:
        plt.xticks([])

    plt.legend(fontsize=fontsize)

    return y_lab

img_paths = {
    "CartPole-v1": "./discrete_trpo/checkpoints/CartPole-v1_report_v2",
    "Acrobot-v1": "./discrete_trpo/checkpoints/Acrobot-v1_report_v3"
}

fontsize = 16

for i in range(11):
    if i in [0, 1, 3, 6]:
        scale = True
    else:
        scale = False

    if i in [3,8,9]:
        bottom=False
    else:
        bottom=True

    plt.figure(figsize=(8, 5))

    y_lab = discrete_plot(img_paths, i=i, min_max_rescale=scale, N_smoothing=10, bottom=bottom, fontsize=fontsize)

    if i in [8, 9, 10]:
        plt.ylabel("Norm " + y_lab, fontsize=fontsize)

    if i == 7:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2e}'))
    else:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    if i not in [3, 9, 10]:
        plt.title("Discrete", fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(f"./figures/discrete_trpo_{y_lab}.pdf", bbox_inches='tight')
    plt.show()

print()

# %%
path = img_paths["CartPole-v1"]

dirs, tags, vals = pull_results(path)

fig, axes = plt.subplots(vals.shape[0], vals.shape[0], figsize=(20, 20))

for i in range(vals.shape[0]):
    for ii in range(vals.shape[0]):
        axes[i, ii].scatter(vals[i, :, :], vals[ii, :, :], alpha=0.05)
        axes[i, ii].set_xticks([])
        axes[i, ii].set_yticks([])

        xlim = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]
        ylim = np.sort(np.ravel(vals[ii, :, :]))[[10, -10]]

        axes[i, ii].set_xlim(*xlim)
        axes[i, ii].set_ylim(*ylim)

        xlab = tags[ii].rsplit("/")[-1]
        ylab = tags[i].rsplit("/")[-1]

        if i == 0 and ii % 2 == 0:
            axes[i, ii].set_xlabel(xlab, fontsize=14)
            axes[i, ii].xaxis.set_label_position('top')
        elif i == vals.shape[0] - 1 and ii % 2 == 1:
            axes[i, ii].set_xlabel(xlab, fontsize=14)

        if ii == 0 and i % 2 == 0:
            axes[i, ii].set_ylabel(ylab, fontsize=14)
        elif ii == vals.shape[0] - 1 and i % 2 == 1:
            axes[i, ii].set_ylabel(ylab, fontsize=14)
            axes[i, ii].yaxis.set_label_position('right')

# %%
path = img_paths["Acrobot-v1"]

dirs, tags, vals = pull_results(path)

fig, axes = plt.subplots(vals.shape[0], vals.shape[0], figsize=(20, 20))

for i in range(vals.shape[0]):
    for ii in range(vals.shape[0]):
        axes[i, ii].scatter(vals[i, :, :], vals[ii, :, :], alpha=0.05)
        axes[i, ii].set_xticks([])
        axes[i, ii].set_yticks([])

        xlim = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]
        ylim = np.sort(np.ravel(vals[ii, :, :]))[[10, -10]]

        axes[i, ii].set_xlim(*xlim)
        axes[i, ii].set_ylim(*ylim)

        xlab = tags[ii].rsplit("/")[-1]
        ylab = tags[i].rsplit("/")[-1]

        if i == 0 and ii % 2 == 0:
            axes[i, ii].set_xlabel(xlab, fontsize=14)
            axes[i, ii].xaxis.set_label_position('top')
        elif i == vals.shape[0] - 1 and ii % 2 == 1:
            axes[i, ii].set_xlabel(xlab, fontsize=14)

        if ii == 0 and i % 2 == 0:
            axes[i, ii].set_ylabel(ylab, fontsize=14)
        elif ii == vals.shape[0] - 1 and i % 2 == 1:
            axes[i, ii].set_ylabel(ylab, fontsize=14)
            axes[i, ii].yaxis.set_label_position('right')

# %%
import seaborn as sns

def plot_pairplots(path, xmetrics, ymetrics, fontsize: int):

    dirs, tags, vals = pull_results(path)

    fig, axes = plt.subplots(len(xmetrics), len(ymetrics),
                             figsize=(5.95114, 3.08059))

    for i, xmetric in enumerate(xmetrics):
        for ii, ymetric in enumerate(ymetrics):
            xvals = vals[xmetric, :, :]
            yvals = vals[ymetric, :, :]

            #axes[i, ii].scatter(xvals, yvals, alpha=0.1)
            sns.scatterplot(x=np.ravel(xvals), y=np.ravel(yvals),
                            ax=axes[i, ii],
                            s=5, color=".15")

            sns.kdeplot(data=np.ravel(xvals), data2=np.ravel(yvals),
                        ax=axes[i, ii],
                        kind="kde", space=0,
                        cmap="viridis", shade=False, shade_lowest=False)
            axes[i, ii].set_xticks([])
            axes[i, ii].set_yticks([])

            xlim = np.sort(np.ravel(xvals))[[10, -10]]
            ylim = np.sort(np.ravel(yvals))[[10, -10]]

            axes[i, ii].set_xlim(*xlim)
            axes[i, ii].set_ylim(*ylim)

            ylab = tags[xmetric].rsplit("/")[-1]
            xlab = "Norm " + tags[ymetric].rsplit("/")[-1]

            if i == 0:
                axes[i, ii].set_xlabel(xlab, fontsize=fontsize-2)
                axes[i, ii].xaxis.set_label_position('top')

            if ii == 0 and i % 2 == 0:
                axes[i, ii].set_ylabel(ylab, fontsize=fontsize-2)
            elif ii == len(xmetrics) and i % 2 == 1:
                axes[i, ii].yaxis.set_label_position('right')
                axes[i, ii].set_ylabel(ylab, fontsize=fontsize-2)

    plt.suptitle(f"{task}", fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(f"./figures/{task}_pair_plot.png", bbox_inches='tight')
    plt.show()

# %%

task = "Acrobot-v1"

xmetrics = [0,3]
ymetrics = [8,9,10]
path = img_paths[task]

plot_pairplots(path, xmetrics, ymetrics, fontsize=11)

# %%

task = "CartPole-v1"

xmetrics = [0, 3]
ymetrics = [8, 9, 10]
path = img_paths[task]

plot_pairplots(path, xmetrics, ymetrics, fontsize=11)

# %%
