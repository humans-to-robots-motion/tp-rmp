import sys
import argparse
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python plot_model.py sample.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The model file', type=str, default='sample2.p')
args = parser.parse_args()


DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
demos = load_demos(data_file)
model = TPRMP.load(args.task, args.data)
res = 0.02
eps = 1.6
step = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = demos[0].traj[0, :], demos[0].traj[1, :], demos[0].traj[2, :]
x, y, z = np.meshgrid(np.arange(X.min() - res, X.max() + res, res),
                      np.arange(Y.min() - res, Y.max() + res, res),
                      np.arange(Z.min() - res, Z.max() + res, res))
xf, yf, zf = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
f = np.zeros_like(x)
frames = demos[0].get_task_parameters()
global_mvns = model.model.generate_global_gmm(frames)

for t in range(0, demos[0].length, step):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                p = np.array([x[i, j, k], y[i, j, k], z[i, j, k], 1., 0., 0., 0.])
                pf = -model.compute_potential_force(p, t, global_mvns)
                xf[i, j, k], yf[i, j, k], zf[i, j, k] = pf[:3]
                f[i, j, k] = np.linalg.norm(pf[:3])
    min_id = np.unravel_index(np.argmin(f, axis=None), f.shape)
    xa, ya, za = x[min_id], y[min_id], z[min_id]
    ax.quiver(x, y, z, xf, yf, zf, length=0.003)
    ax.scatter(xa, ya, za, marker='^', color='r')
    plot_demo(demos[0], title=f'Timestep: {t}', plot_quat=False, new_fig=False, show=False)
    plt.draw()
    plt.pause(0.0001)
    plt.cla()
