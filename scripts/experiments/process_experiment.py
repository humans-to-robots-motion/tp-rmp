import sys
from os.path import join, dirname, abspath
import matplotlib
import logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

ROOT_DIR = join(dirname(abspath(__file__)), '..', '..')
sys.path.append(ROOT_DIR)

from tprmp.utils.loading import load  # noqa

task = 'pick'
name = 'adaptation_diff_15.p'
DATA_DIR = join(ROOT_DIR, 'data', 'tasks', task, 'experiments')
file_name = join(DATA_DIR, name)
data = load(file_name)
# baseline = load(join(DATA_DIR, 'tracking_baseline.p'))
for trial in data:
    for f in trial:
        trial[f] = trial[f][15]

fig, ax = plt.subplots()

# for 2d plots
# green_diamond = dict(markerfacecolor='g', marker='D')
# ax.boxplot(list(data.values()), flierprops=green_diamond)
# labels = [round(v, 1) for v in data.keys()]
# ax.set_xticklabels(labels)
# ax.set_xlabel('Number of components K')
# ax.set_ylabel('Tracking MSE')
# ax.axhline(y=np.mean(baseline), color='g', linestyle='--')

# for 6d plots
green_diamond = dict(markerfacecolor='g', marker='D')
plot_data = np.array([list(trial.values()) for trial in data])
ax.boxplot(np.squeeze(plot_data), flierprops=green_diamond)
labels = [round(v, 1) for v in data[0].keys()]
ax.set_xticklabels(labels)
ax.set_xlabel('Moving goal radius')
ax.set_ylabel('Goal Errors')
ax.set_ylim(0., 1.)
# ax.axhline(y=np.mean(baseline), color='g', linestyle='--')

plt.show()
