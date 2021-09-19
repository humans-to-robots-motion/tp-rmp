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
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.visualization.demonstration import plot_frame_2d  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa

mode = 1
task = 'test'
demo_path = join(ROOT_DIR, 'data', 'tasks', task, 'demos')
num_comp = 9
N1, N2 = 3, 5
limits = [0., 4.5]
demo_names = ['C', 'C1', 'G', 'hat', 'hat1', 'I', 'I1', 'J', 'L', 'L1', 'P', 'S', 'S1', 'S2', 'U']
dt = 0.01
v_eps = 5e-2
goal_eps = 0.2
wait = 10
res = 0.05
colormap = 'RdBu' if mode == 1 else 'YlOrBr'
manifold = Manifold.get_euclidean_manifold(2)


def execute(model, frames, x0, dx0):
    x, dx = x0, dx0
    traj = [x]
    model.generate_global_gmm(frames)
    count = 0
    while True:
        ddx = model.retrieve(x, dx)
        dx = ddx * dt + dx
        x = manifold.exp_map(dx * dt, base=x)
        traj.append(x)
        if np.linalg.norm(dx) < v_eps:
            count += 1
            if count >= wait:
                break
        else:
            count = 0
    return np.array(traj).T


X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
fig, axs = plt.subplots(N1, N2)
for n, name in enumerate(demo_names):
    k, m = int(n / N2), n % N2
    data_file = join(demo_path, name + '.p')
    if k == 2 and m == 0:
        model_file = name + '_' + str(7) + '.p'
    else:
        model_file = name + '_' + str(num_comp) + '.p'
    demos = load_demos_2d(data_file, dt=dt)
    model = TPRMP.load(task, model_name=model_file)
    sample = demos[0]
    frames = sample.get_task_parameters()
    x0, dx0 = sample.traj[:, 0], sample._d_traj[:, 0]
    traj = execute(model, frames, x0, dx0)
    ax = axs[k, m]
    plt.sca(ax)
    ax.set_aspect('equal')
    ax.set_xlim([limits[0], limits[1]])
    ax.set_ylim([limits[0], limits[1]])
    ax.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.7)
    ax.plot(traj[0], traj[1], color="b")
    plot_frame_2d(frames.values())
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mode == 1:
                Z[i, j] = model.compute_potential_field(np.array([X[i, j], Y[i, j]]))
            else:
                Z[i, j] = model.compute_dissipation_field(np.array([X[i, j], Y[i, j]]))
    c = ax.pcolormesh(X, Y, Z, cmap=colormap, shading='auto', vmin=0., vmax=Z.max(), alpha=0.5)
    fig.colorbar(c, ax=ax)
    last = demos[0].traj[:, -1]
    plt.scatter(last[0], last[1], marker='*', color='r', s=40)

for ax in fig.axes:
    try:
        ax.label_outer()
    except:  # noqa
        pass
plt.show()
