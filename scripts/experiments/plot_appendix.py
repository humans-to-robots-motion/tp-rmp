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
matplotlib.rcParams['font.size'] = 8

ROOT_DIR = join(dirname(abspath(__file__)), '..', '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.visualization.dynamics import plot_potential_field, plot_dissipation_field  # noqa
mode = 1
task = 'test'
demo_path = join(ROOT_DIR, 'data', 'tasks', task, 'demos')
num_comp = 9
N1, N2 = 3, 5
limits = [0., 4.5]
demo_names = ['C', 'C1', 'G', 'hat', 'hat1', 'I', 'I1', 'J', 'L', 'L1', 'P', 'S', 'S1', 'S2', 'U']
dt = 0.01
res = 0.05

X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
fig, axs = plt.subplots(N1, N2)
for n, name in enumerate(demo_names):
    k, m = int(n / N2), n % N2
    data_file = join(demo_path, name + '.p')
    model_file = name + '_' + str(num_comp) + '.p'
    demos = load_demos_2d(data_file, dt=dt)
    model = TPRMP.load(task, model_name=model_file)
    sample = demos[0]
    frames = sample.get_task_parameters()
    ax = axs[k, m]
    plt.sca(ax)
    ax.set_xlim([limits[0], limits[1]])
    ax.set_ylim([limits[0], limits[1]])
    if mode == 1:
        plot_potential_field(model, frames, limits=limits, new_ax=False, res=res)
    else:
        plot_dissipation_field(model, frames, limits=limits, new_ax=False, res=res)

for ax in fig.axes:
    try:
        ax.label_outer()
    except:  # noqa
        pass
plt.show()
