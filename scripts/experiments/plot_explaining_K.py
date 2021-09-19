import sys
from os.path import join, dirname, abspath
import matplotlib
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

ROOT_DIR = join(dirname(abspath(__file__)), '..', '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.visualization.demonstration import _plot_traj_global  # noqa
from tprmp.visualization.models import _plot_gmm_global  # noqa

dt = 0.01
res = 0.05
limits = [0., 4.5]
task = 'test'
demo_path = join(ROOT_DIR, 'data', 'tasks', task, 'demos')
demo_name = 'S'
data_file = join(demo_path, demo_name + '.p')
num_comp = [5, 9, 15]

fig, axs = plt.subplots(1, 3)
X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
for k, n in enumerate(num_comp):
    demos = load_demos_2d(data_file, dt=dt)
    model_file = demo_name + '_' + str(n) + '.p'
    model = TPRMP.load(task, model_name=model_file)
    frames = demos[0].get_task_parameters()
    ax = axs[k]
    ax.set_aspect('equal')
    plt.sca(ax)
    _plot_traj_global(demos, title=f'{n} components', limits=limits, legend=False, three_d=False, new_ax=False)
    _plot_gmm_global(model.model, frames, three_d=False, new_ax=False)
    Z = np.zeros_like(X)
    model.generate_global_gmm(frames)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = model.compute_potential_field(np.array([X[i, j], Y[i, j]]))
    c = ax.pcolormesh(X, Y, Z, cmap='RdBu', shading='auto', vmin=0., vmax=50, alpha=0.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(c, cax=cax)
plt.show()
