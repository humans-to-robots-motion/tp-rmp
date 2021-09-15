import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath
import logging
import matplotlib.pyplot as plt
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa
from tprmp.models.pd import PDController  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_pd_controller.py test.p')
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--demo', help='The data file', type=str, default='S.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
# parameters
T = 2000
dt = 0.01
dt_update = 0.01
ratio = int(dt_update / dt)
limits = [0., 4.5]
Kp = 2 * np.eye(2)
Kd = np.sqrt(2) * np.eye(2)
# load data
demos = load_demos_2d(data_file, dt=dt)
sample = demos[0]
# plot_demo(demos, only_global=False, new_fig=True, new_ax=True, three_d=False, limits=limits, show=True)
model = PDController(sample.manifold, Kp=Kp, Kd=Kd)
x, dx = sample.traj[:, 0], sample._d_traj[:, 0]
traj = [x]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([limits[0], limits[1]])
ax.set_ylim([limits[0], limits[1]])
ax.set_aspect('equal')
ax.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.6)
line1, = ax.plot(x[0], x[1], marker="o", color="b", markersize=1, linestyle='None', alpha=1.)
line2, = ax.plot(x[0], x[1], marker="o", color="b", markersize=1, linestyle='None', alpha=0.3)
model.update_targets(x, np.zeros_like(dx))
for t in range(1, T):
    if t % ratio == 0:
        i = int(t / ratio)
        if i < sample.traj.shape[1]:
            model.update_targets(sample.traj[:, i], np.zeros_like(dx))
    ddx = model.retrieve(x, dx)
    dx = ddx * dt + dx
    x = dx * dt + x
    traj.append(x)
    line1.set_xdata(x[0])
    line1.set_ydata(x[1])
    line2.set_xdata(np.array(traj)[:, 0])
    line2.set_ydata(np.array(traj)[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
