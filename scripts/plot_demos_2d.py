import sys
import matplotlib.pyplot as plt
import matplotlib
from os.path import join, dirname, abspath
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 8

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos, load_demos_2d  # noqa
from tprmp.visualization.demonstration import _plot_traj_global  # noqa

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', 'test', 'demos')
dt = 0.01
N1, N2 = 3, 5
limits = [0., 4.5]
demo_names = ['C', 'C1', 'G', 'hat', 'hat1', 'I', 'I1', 'J', 'L', 'L1', 'P', 'S', 'S1', 'S2', 'U']
fig, axs = plt.subplots(N1, N2)

for n, name in enumerate(demo_names):
    data_file = join(DATA_DIR, name + '.p')
    demos = load_demos_2d(data_file, dt=dt)
    i, j = int(n / N2), n % N2
    plt.sca(axs[i, j])
    axs[i, j].set_aspect('equal')
    _plot_traj_global(demos, limits=limits, legend=False, three_d=False, new_ax=False)
    last = demos[0].traj[:, -1]
    plt.scatter(last[0], last[1], marker='*', color='k', s=30)
# fig.tight_layout()
plt.show()
