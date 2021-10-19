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


def to_csv(dpath):
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

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"

dirs, tags, vals = to_csv(path)

fig, axes = plt.subplots(vals.shape[0], vals.shape[0], figsize=(20,20))

for i in range(vals.shape[0]):
    for ii in range(vals.shape[0]):
        axes[i,ii].scatter(vals[i, :, :], vals[ii, :, :], alpha=0.05)
        axes[i,ii].set_xticks([])
        axes[i,ii].set_yticks([])

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
import matplotlib.pyplot as plt

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
i = 0

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

smoothed_time_plot(vals[i], min_max_rescale=True, N_smoothing=10, label="PoleCart-v1")

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v2"
dirs, tags, vals = to_csv(path)

smoothed_time_plot(vals[i], min_max_rescale=True,  N_smoothing=10, label="Acrobot-v1")

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("Reward (scaled)")

plt.legend()
plt.show()

# %%

i = 2

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

print(tags)

a = smoothed_time_plot(vals[i], N_smoothing=10, label="PoleCart-v1")

ylim_a = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v2"
dirs, tags, vals = to_csv(path)

b = smoothed_time_plot(vals[i], N_smoothing=10, label="Acrobot-v1")

ylim_b = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]

lims = np.stack([ylim_a, ylim_b])
lims = np.min(lims, axis=0)[0], np.max(lims, axis=0)[1]

plt.ylim(*lims)

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("Line search coefficient")

plt.legend()
plt.show()

# %%

i = 3

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

print(tags)

a = smoothed_time_plot(vals[i], N_smoothing=10, min_max_rescale=True, label="PoleCart-v1")

ylim_a = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v2"
dirs, tags, vals = to_csv(path)

b = smoothed_time_plot(vals[i], N_smoothing=10, min_max_rescale=True, label="Acrobot-v1")

ylim_b = np.sort(np.ravel(vals[i, :, :]))[[10, -10]]

lims = np.stack([ylim_a, ylim_b])
lims = np.min(lims, axis=0)[0], 0.5

plt.ylim(*lims)

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("Learning rate (scaled)")

plt.legend()
plt.show()

# %%
i = -2

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

a = smoothed_time_plot(vals[i], N_smoothing=5, min_max_rescale=True, label="PoleCart-v1")

ylim_a = np.sort(np.ravel(vals[i, :, :]))[[20, -20]]

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v1"
dirs, tags, vals = to_csv(path)

b = smoothed_time_plot(vals[i], N_smoothing=10, min_max_rescale=True, label="Acrobot-v1")

ylim_b = np.sort(np.ravel(vals[i, :, :]))[[20, -50]]

lims = np.stack([ylim_a, ylim_b])
lims = np.min(lims, axis=0)[0], np.max(lims, axis=0)[1]
lims = (0, 0.5)

plt.ylim(*lims)

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("Norm")

plt.legend()
plt.show()

# %%

i = -1

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

a = smoothed_time_plot(vals[i], N_smoothing=10, label="PoleCart-v1")

ylim_a = np.sort(np.ravel(vals[i, :, :]))[[20, -20]]

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v2"
dirs, tags, vals = to_csv(path)

b = smoothed_time_plot(vals[i], N_smoothing=10, label="Acrobot-v1")

ylim_b = np.sort(np.ravel(vals[i, :, :]))[[20, -50]]

lims = np.stack([ylim_a, ylim_b])
lims = np.min(lims, axis=0)[0], 2.5

plt.ylim(*lims)

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("Norm")

plt.legend()
plt.show()

# %%

i = -3

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/CartPole-v1_report_v1"
dirs, tags, vals = to_csv(path)

a = smoothed_time_plot(vals[i], N_smoothing=10,  label="PoleCart-v1")

ylim_a = np.sort(np.ravel(vals[i, :, :]))[[20, -20]]

path = "C:/Users/ivoon/Documents/GitHub/UvA_RL_2021_Reproducibility_Lab/discrete_trpo/checkpoints/Acrobot-v1_report_v2"
dirs, tags, vals = to_csv(path)

b = smoothed_time_plot(vals[i], N_smoothing=10, label="Acrobot-v1")

#ylim_b = np.sort(np.ravel(vals[i, :, :]))[[20, -20]]

#lims = np.stack([ylim_a, ylim_b])
#lims = np.min(lims, axis=0)[0], np.max(lims, axis=0)[1]

#plt.ylim(*lims)

plt.title(tags[i])

plt.xlabel("Iteration")
plt.ylabel("KL divergence")

plt.legend()
plt.show()

# %%
